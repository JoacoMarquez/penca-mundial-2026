"""Corrida extra cerca del cierre — el T-3h del Mundial, versión Clausura.

La planilla del día se genera a las 12:00 UTC (09:00 UY). Entre esa corrida y los
cierres (17:45-22:45 UTC) los inputs se mueven: odds de Supermatch, snapshot del
pool (rivales que cargaron a la tarde), resultados de partidos ya jugados de la
fecha. Este módulo re-corre el pipeline una vez por TANDA de cierres (~2h antes de
cada grupo de cierres; un sábado con partidos a las 13:45 y a las 21:45 recibe dos
pasadas), y notifica por Telegram SOLO si algún pick de un partido todavía abierto
cambió respecto de la planilla vigente.

El disparador del aviso es el VALOR, no el diff. Un diff de picks no significa nada:
el óptimo es plano y dos corridas con insumos idénticos reasignan ~43 de 96 picks
(medido el 2026-08-08, cuando este módulo pidió recargar 56 picks hacia una planilla
PEOR: $238k → $221k de E[premio]). Así que después de re-optimizar se liquidan las
dos planillas —la vigente y la nueva— en el MISMO simulador con sorteos comunes, y
solo se notifica si el Δ E[premio] supera su propio error de Monte Carlo y un piso
absoluto que justifique volver a cargar doce planillas a mano. El aviso lleva el
número, para que la decisión de recargar sea informada.

La corrida versiona una planilla nueva igual (v+1, regla de trabajo #2), avise o no.

Uso:
    python -m src.clausura.rerun_cierre                    # corre si toca (timer)
    python -m src.clausura.rerun_cierre --force            # ignora ventana y estado
    python -m src.clausura.rerun_cierre --dry-run          # sin Telegram ni estado
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.clausura.api import TZ_UY

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "state" / "rerun_cierre.json"

# Disparo: el primer cierre del día está a menos de este umbral.
TRIGGER_H = 2.5

# Cuánto tiene que mejorar la planilla nueva para molestarte con una recarga manual.
# El disparador NO puede ser "los picks cambiaron": el óptimo es plano y dos corridas
# con insumos idénticos reasignan ~43 de 96 picks sin que nada haya pasado (medido el
# 2026-08-08; ese día el rerun pidió 56 cambios hacia una planilla PEOR, $238k → $221k).
# Entonces se avisa por VALOR: Δ E[premio] evaluado con sorteos comunes, y solo si
# supera su propio error de Monte Carlo y un piso absoluto — recargar 12 planillas a
# mano tiene un costo real (tiempo y riesgo de tipeo) que un Δ de $400 no paga.
UMBRAL_SE = 2.0
UMBRAL_ABS = 2_000.0
EVAL_SEEDS = 5


# -------------------- lógica de ventana (pura, testeable) --------------------

def cierres_en_ventana(eventos: list[dict], now: datetime,
                       horizon_h: float = TRIGGER_H) -> list[datetime]:
    """Cierres abiertos dentro del horizonte, SIN filtrar por fecha calendario.

    La versión anterior exigía `cierre.date() == now.date()` en UTC, y eso dejaba
    ciego el caso real de un partido 21:00+ UY: su cierre cae pasada la medianoche
    UTC, "mañana" para el tick del viernes de noche y ya pasado para el primer
    tick del sábado — nunca había rerun. El horizonte rodante no depende del
    calendario.
    """
    out = set()
    for ev in eventos:
        cierre = datetime.fromisoformat(ev["cierre_pronostico_utc"])
        if now < cierre <= now + timedelta(hours=horizon_h):
            out.add(cierre)
    return sorted(out)


def debe_correr(eventos: list[dict], now: datetime, ya_corridos: set[str]) -> bool:
    """¿Hay algún cierre en ventana que esta tanda todavía no cubrió?

    El estado guarda los CIERRES ya cubiertos (ISO), no el día: así un sábado con
    tanda de 13:45 y tanda de 21:45 recibe un rerun por tanda (antes era uno solo
    por día, y los partidos de la noche quedaban con insumos de T-9h), y un fallo
    del tick no quema la única oportunidad del día — el tick siguiente reintenta
    mientras el cierre siga en ventana.
    """
    return any(c.isoformat() not in ya_corridos
               for c in cierres_en_ventana(eventos, now))


# -------------------- diff de planillas (puro, testeable) --------------------

def diff_planillas(
    prev: dict,
    nuevo: dict,
    now: datetime,
) -> list[tuple[dict, list[tuple[int, tuple[int, int], tuple[int, int]]]]]:
    """Cambios en partidos aún abiertos: [(fila_evento_nuevo, [(col, viejo, nuevo)])].

    Solo compara eventos presentes en ambas planillas y con cierre futuro — lo
    cerrado ya no es accionable y lo congelado no puede cambiar.
    """
    prev_by_eid = {int(r["evento_id"]): r for r in prev.get("picks", [])}
    out = []
    for row in nuevo.get("picks", []):
        eid = int(row["evento_id"])
        prow = prev_by_eid.get(eid)
        if prow is None:
            continue
        if datetime.fromisoformat(row["cierre_pronostico_utc"]) <= now:
            continue
        cambios = []
        for k, (old, new) in enumerate(zip(prow.get("scores", []), row.get("scores", []))):
            if list(old) != list(new):
                cambios.append((k, (int(old[0]), int(old[1])), (int(new[0]), int(new[1]))))
        if cambios:
            out.append((row, cambios))
    return out


def vale_avisar(comp, umbral_se: float = UMBRAL_SE,
                umbral_abs: float = UMBRAL_ABS) -> bool:
    """¿La planilla nueva mejora lo suficiente para pedir una recarga manual?

    Dos condiciones, las dos necesarias:
      * el Δ tiene que ser POSITIVO y distinguible del ruido (> umbral_se × su se),
        porque el óptimo es plano y el churn produce Δ de cualquier signo;
      * y tiene que valer una plata que justifique volver a cargar a mano.
    """
    return comp.delta > umbral_abs and comp.delta > umbral_se * comp.se


def picks_previos(prev: dict, contexto: dict, n_participaciones: int):
    """Matriz de picks del portfolio nuevo con la fecha objetivo PISADA por la vieja.

    Así las dos matrices difieren solo en lo accionable (los partidos de la fecha que
    todavía se pueden cambiar) y comparten todo lo demás — pasado congelado y fechas
    futuras—, que es la comparación que interesa.
    """
    import numpy as np
    from src.clausura.economics import score_index

    picks_nuevos = contexto["portfolio"].picks
    idx_of = contexto["idx_of"]
    matriz = np.array(picks_nuevos, dtype=np.int64).copy()
    for row in prev.get("picks", []):
        col = idx_of.get(int(row["evento_id"]))
        if col is None:
            continue
        for k, (gl, gv) in enumerate(row.get("scores", [])[:n_participaciones]):
            matriz[k, col] = score_index(int(gl), int(gv))
    return matriz


def valor_del_cambio(prev: dict, contexto: dict, n_participaciones: int):
    """Δ E[premio] de la planilla nueva vs la vieja, con sorteos comunes.

    None si el contexto no alcanza (por ejemplo si picks.run no lo llenó).
    """
    ev = contexto.get("evaluador")
    if ev is None or contexto.get("portfolio") is None:
        log.warning("sin evaluador en el contexto — no puedo medir el valor del cambio")
        return None
    try:
        # El evaluador construye un SeasonSimulator por semilla: medido en el droplet,
        # 351 MB de pico a 2400 sims × 684 rivales (65 s). Corre DESPUÉS de la
        # optimización, así que conviene devolver primero las matrices que el
        # optimizador ya no usa — en 1 GB de RAM el margen no sobra.
        import gc
        gc.collect()
        viejos = picks_previos(prev, contexto, n_participaciones)
        comp = ev.comparar(viejos, contexto["portfolio"].picks, n_seeds=EVAL_SEEDS)
        log.info("valor del cambio: %s", comp)
        return comp
    except Exception as e:                                    # noqa: BLE001
        # ERROR + traceback, no warning: este except tapó durante días un
        # AttributeError determinístico en EvaluadorPortfolio (arreglado el
        # 2026-08-08) y el gate por valor nunca corrió. Seguir avisando ante un
        # fallo es lo correcto —mejor un aviso de más que perderse una mejora
        # real— pero tiene que hacer ruido, porque "no pude medir" en el 100% de
        # las corridas se lee igual que en el 1%.
        log.error("no pude medir el valor del cambio (%s) — aviso igual, PERO esto "
                  "no debería pasar: el gate por valor está inhabilitado mientras "
                  "falle", e, exc_info=True)
        return None


def _guardar_veredicto(path: Path, comp, avisar: bool, n_picks: int) -> None:
    """Anota en la planilla si sus cambios valen la pena, para que el panel lo diga.

    `comp is None` significa que el gate NO pudo medir (evaluador roto), no que el
    cambio sea nulo: ahí se avisa por las dudas y el panel tiene que decir eso mismo,
    porque "no pude medir" y "medí y da cero" piden acciones distintas.
    """
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        d["veredicto_cambio"] = {
            "avisar": bool(avisar),
            "medido": comp is not None,
            "n_picks": int(n_picks),
            **({} if comp is None else {
                "delta": float(comp.delta), "se": float(comp.se),
                "valor_a": float(comp.valor_a), "valor_b": float(comp.valor_b),
                "n_seeds": int(comp.n_seeds), "significativa": bool(comp.significativa),
            }),
        }
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:                                    # noqa: BLE001
        # No es fatal: sin veredicto el panel cae al aviso genérico de siempre.
        log.warning("no pude anotar el veredicto en %s (%s)", path.name, e)


def formatear_diff(
    cambios: list[tuple[dict, list[tuple[int, tuple[int, int], tuple[int, int]]]]],
    mis_numeros: list[int],
    now: datetime,
    comp=None,
) -> str:
    """Aviso compacto: solo lo que cambió, con hora de cierre en UY."""
    n_picks = sum(len(c) for _, c in cambios)
    lines = ["<b>🔄 Corrida T-2h — cambios vs la planilla de la mañana</b>"]
    if comp is not None:
        lines.append(f"Vale <b>{comp.delta:+,.0f}</b> ± {comp.se:,.0f} de E[premio] "
                     f"(${comp.valor_a:,.0f} → ${comp.valor_b:,.0f}).")
    lines.append(f"{n_picks} pick(s) cambiaron en {len(cambios)} partido(s) aún abiertos:")
    for row, cs in cambios:
        cierre = datetime.fromisoformat(row["cierre_pronostico_utc"])
        cierre_uy = cierre.astimezone(TZ_UY)
        hoy = " HOY" if cierre.date() == now.date() else cierre_uy.strftime(" %a").lower()
        pref = " ⭐x2" if row.get("preferencial") else ""
        lines.append(f"\n<b>{row['partido']}</b>{pref} · cierra {cierre_uy:%H:%M}{hoy}")
        for k, old, new in cs:
            numero = mis_numeros[k] if k < len(mis_numeros) else f"col{k + 1}"
            lines.append(f"  {numero}: {old[0]}-{old[1]} → <b>{new[0]}-{new[1]}</b>")
    lines.append("\nSi ya cargaste la planilla de la mañana, actualizá SOLO estos picks "
                 "en la web. La planilla nueva quedó versionada.")
    if comp is not None and not comp.significativa:
        lines.append("\n⚠️ El Δ no supera su propio ruido de simulación: el aviso sale "
                     "por el piso absoluto, no porque la mejora esté confirmada.")
    return "\n".join(lines)


# -------------------- estado --------------------

def load_state(now: datetime | None = None) -> set[str]:
    """Cierres ya cubiertos. Poda entradas viejas (>2 días) — incluye las claves
    por-día del formato anterior, que quedan obsoletas solas."""
    if not STATE_PATH.exists():
        return set()
    now = now or datetime.now(timezone.utc)
    vivos = set()
    for k in json.loads(STATE_PATH.read_text(encoding="utf-8")):
        try:
            ts = datetime.fromisoformat(k)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if ts > now - timedelta(days=2):
            vivos.add(k)
    return vivos


def save_state(corridos: set[str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(sorted(corridos)), encoding="utf-8")


# -------------------- main --------------------

# Solo si la planilla previa no dice con cuántas corrió. Tiene que seguir a
# deploy/clausura-picks.service: era 2400 y quedó peligrosamente viejo cuando
# producción pasó a 19.200 el 2026-08-10.
DEFAULT_SIMS = 19_200

# Debajo de esto el ascenso por coordenadas fitea ruido, y desde el 2026-08-11 eso
# no es solo "menos preciso": con K_EV=5 el menú de 5 mide −$4.192 a 2.400 sorteos
# (a 19.200 mide +$4.568). O sea que correr con pocos sorteos pone al optimizador en
# el régimen donde NUESTRA PROPIA configuración de menú es dañina. No se clampea
# —quien pide --sims a mano sabe lo que hace, y los tests corren con valores chicos
# a propósito— pero tiene que hacer ruido en el log.
SIMS_MIN_SEGURO = 9_600



def sims_de(prev: dict, pedido: int | None = None) -> int:
    """Cuántos sorteos usar: los MISMOS que la planilla contra la que se diffea.

    Sin esto el diff no significa nada. Medido el 2026-08-08: la planilla de la
    mañana corrió con 2400 y el rerun con 1600, y con insumos idénticos —mismo
    snapshot del pool, misma temperatura, mismo ancla de EV en 7 de 8 partidos—
    reasignó 56 de 96 picks y bajó el E[premio] out-of-sample de $238k a $221k.
    La semilla es fija, pero cambiar la cantidad de sorteos cambia la superficie
    que el ascenso por coordenadas escala: cae en otro óptimo local, no mejor.
    """
    previo = prev.get("n_sims")
    if pedido:
        sims = int(pedido)
    elif not previo:
        log.warning("la planilla previa no registra n_sims (es anterior a este "
                    "cambio) — uso %d; el diff puede traer churn del optimizador",
                    DEFAULT_SIMS)
        sims = DEFAULT_SIMS
    else:
        sims = int(previo)
    if sims < SIMS_MIN_SEGURO:
        log.warning(
            "corriendo con %d sorteos, por debajo del piso seguro de %d. El ascenso "
            "fitea ruido ahí, y con K_EV=5 el menú de producción mide NEGATIVO en ese "
            "régimen (−$4.192 a 2.400 vs +$4.568 a 19.200). La planilla que salga de "
            "acá no es comparable con las de producción.", sims, SIMS_MIN_SEGURO)
    return sims


def run(
    n_participaciones: int = 12,
    n_sims: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    now: datetime | None = None,
) -> list | None:
    import src.clausura.picks as picks
    from src.clausura.rivals import mis_numeros_env
    from src.utils.versions import latest_version

    now = now or datetime.now(timezone.utc)
    cfg = picks.load_config()
    eventos = picks.flat_eventos(cfg)
    estado = load_state(now)

    if not force and not debe_correr(eventos, now, estado):
        log.info("fuera de ventana (primer cierre de hoy a >%sh, sin cierres hoy, "
                 "o ya corrido) — no toca", TRIGGER_H)
        return None

    # Lo que esta corrida cubre: los cierres en ventana AHORA. Se marca al final
    # (éxito) — si el pipeline muere, el próximo tick reintenta estos mismos cierres.
    cubiertos = {c.isoformat() for c in cierres_en_ventana(eventos, now)}

    target_fecha = picks.resolve_fecha("auto")
    d = picks.fecha_dir(target_fecha)
    prev_path = latest_version(d.glob("v*_*.json")) if d.exists() else None
    if prev_path is None:
        log.warning("no hay planilla previa de la fecha %d — corré primero el pipeline "
                    "de la mañana; no hay contra qué diffear", target_fecha)
        return None
    prev = json.loads(prev_path.read_text(encoding="utf-8"))

    contexto: dict = {}
    sims = sims_de(prev, n_sims)
    log.info("re-corriendo pipeline de la fecha %d con %d sims (planilla previa: %s)",
             target_fecha, sims, prev_path.name)
    new_path = picks.run(target_fecha, n_participaciones=n_participaciones,
                         telegram=False, n_sims=sims, contexto=contexto)
    nuevo = json.loads(new_path.read_text(encoding="utf-8"))

    cambios = diff_planillas(prev, nuevo, now)
    mis_numeros = sorted(mis_numeros_env())

    comp = None
    if cambios:
        comp = valor_del_cambio(prev, contexto, n_participaciones)
        avisar = comp is None or vale_avisar(comp)
        # El veredicto va DENTRO de la planilla nueva. Si solo queda en el log, el
        # dashboard muestra los picks distintos sin decir que fueron descartados, y
        # eso invita a recargar a mano justo lo que el gate resolvió no tocar (pasó
        # el 2026-08-10: 7 picks en amarillo con Δ -456 ± 171 detrás).
        _guardar_veredicto(new_path, comp, avisar,
                           n_picks=sum(len(cs) for _, cs in cambios))
        if not avisar:
            log.info("la planilla nueva NO mejora lo suficiente (%s) — no aviso; "
                     "%d picks distintos quedan versionados en %s",
                     comp, sum(len(cs) for _, cs in cambios), new_path.name)
            if not dry_run:
                save_state(estado | cubiertos)
            return []

    # El estado se marca ANTES de intentar el Telegram: con el orden inverso, una
    # tarde con la Bot API caída re-corría el pipeline COMPLETO cada tick (~30 min
    # de CPU por hora, pila de versiones churn sobre churn) justo en la ventana de
    # cierres. El costo del orden nuevo es acotado: si el send falla, el aviso de
    # ESTA tanda se pierde (la planilla igual quedó versionada y visible en el
    # dashboard) y el fallo suena una vez vía OnFailure.
    if not dry_run:
        save_state(estado | cubiertos)

    if cambios:
        msg = formatear_diff(cambios, mis_numeros, now, comp)
        if prev.get("n_sims") and sims != prev["n_sims"]:
            msg = (f"⚠️ <b>Este diff no es confiable</b>: la planilla previa se optimizó "
                   f"con {prev['n_sims']} sorteos y ésta con {sims}. Con distinto "
                   f"n_sims el optimizador cae en otro óptimo local y reasigna picks "
                   f"sin que se haya movido ningún insumo.\n\n" + msg)
        print(msg.replace("<b>", "").replace("</b>", ""))
        if not dry_run:
            from src.notifier.telegram import TelegramConfig, TelegramNotifier
            TelegramNotifier(TelegramConfig.from_env()).send(msg)
    else:
        log.info("sin cambios en partidos abiertos — no se notifica (planilla %s igual "
                 "quedó versionada)", new_path.name)
    return cambios


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--participaciones", type=int, default=12)
    ap.add_argument("--sims", type=int, default=0,
                    help="0 (default) = heredar de la planilla previa, que es lo "
                         "único que hace comparable el diff")
    ap.add_argument("--force", action="store_true",
                    help="correr aunque no toque (ignora ventana y estado)")
    ap.add_argument("--dry-run", action="store_true",
                    help="sin Telegram ni marca de estado")
    args = ap.parse_args()
    run(n_participaciones=args.participaciones, n_sims=args.sims,
        force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

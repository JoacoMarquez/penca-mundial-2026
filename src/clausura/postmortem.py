"""Postmortem automático por fecha (regla de trabajo #4 del proyecto).

Cuando una fecha tiene suficientes partidos jugados (MIN_JUGADOS, no todos: un
suspendido no puede bloquear el análisis de los demás), genera y manda por Telegram
un análisis de qué pasó, y persiste el detalle en JSON:

  - Resultados reales + puntos de nuestras 12 participaciones por partido
    (kernel real de Supermatch, estrella x2), esperado vs real — el E[pts] se toma
    de la planilla generada ANTES del cierre de cada partido, porque en las
    posteriores la grilla es una delta y el "esperado" sería el puntaje real.
  - Distribución de puntos del POOL en la fecha (desde el snapshot de picks
    públicos): mediana, máximo, cuántos rivales nos ganaron, y si alguna nuestra
    ganó el premio por fecha.
  - Qué pegó y qué no: exactos nuestros vs exactos del pool por partido, y el
    rendimiento en el partido estrella (x2).
  - Insumos de recalibración: tasa de exactos del pool observada en la fecha
    (la calibración en sí ya es online: el pipeline de picks la lee del ranking, y
    la Q empírica del pool sale del snapshot — acá se REPORTA, no se duplica).

Idempotente: un archivo por fecha en data/postmortems/clausura/fecha_NN.json.
El timer corre de madrugada; si la última fecha completa ya tiene postmortem,
sale en silencio.

Uso:
    python -m src.clausura.postmortem                # detecta la fecha, genera y manda
    python -m src.clausura.postmortem --fecha 3      # fuerza una fecha puntual
    python -m src.clausura.postmortem --dry-run      # imprime sin Telegram ni estado
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.clausura.api import PencaApiClient
from src.clausura.rivals import mis_numeros_env
from src.clausura.scoring import supermatch_points

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PM_DIR = ROOT / "data" / "postmortems" / "clausura"


@dataclass
class FechaStats:
    fecha_n: int
    resultados: dict[int, tuple[int, int]]              # evento_id → real
    eventos: list[dict]                                  # los de la fecha, con metadata
    # nuestras participaciones
    puntos: dict[int, int] = field(default_factory=dict)          # numero → pts fecha
    exactos: dict[int, int] = field(default_factory=dict)         # numero → exactos fecha
    esperados: dict[int, float] = field(default_factory=dict)     # numero → ΣE[pts] (si hay)
    detalle: dict[int, dict[int, dict]] = field(default_factory=dict)  # eid → numero → {...}
    # pool
    pool_puntos: list[int] = field(default_factory=list)
    pool_exactos_por_evento: dict[int, float] = field(default_factory=dict)  # eid → frac


# -------------------- datos --------------------

def resultados_de_fecha(
    cfg: dict, fecha_n: int, min_jugados: int = 1,
) -> tuple[dict[int, tuple[int, int]], int] | None:
    """(evento_id → (gl, gv) de los JUGADOS, cuántos faltan). None si no alcanza.

    Antes exigía la fecha COMPLETA y devolvía None si faltaba un solo resultado. Con
    eso el postmortem —el único circuito que compara lo pronosticado contra lo que
    pasó— nunca corrió: la Fecha 1 tiene Torque–Peñarol SUSPENDIDO, y un suspendido
    no se resuelve en días. Un partido que se reprograma para dentro de un mes
    bloqueaba el aprendizaje de los otros siete.

    Ahora analiza lo jugado y REPORTA lo que falta. El postmortem es idempotente por
    fecha, así que si después se juega el pendiente se puede regenerar y el nuevo
    archivo incluye todo.
    """
    nombre = f"Fecha {fecha_n}"
    f = cfg["fechas"].get(nombre)
    if f is None:
        raise ValueError(f"no existe {nombre} en el config")
    out: dict[int, tuple[int, int]] = {}
    faltan = 0
    with PencaApiClient() as api:
        data = api._get(f"/front/campeonatos/fechas/{f['fecha_id']}/eventos")
    for e in data:
        res = e.get("resultado") or {}
        gl, gv = res.get("golesEquipoLocal"), res.get("golesEquipoVisitante")
        if gl is None or gv is None:
            faltan += 1
            continue
        out[int(e["id"])] = (int(gl), int(gv))
    if len(out) < min_jugados:
        return None
    return out, faltan


def picks_de_planilla(fecha_n: int, mis_numeros: list[int]) -> tuple[dict, dict]:
    """(numero → {evento_id → pick}, evento_id → {numero → e_pts PRE-PARTIDO}).

    Los PICKS salen de la última planilla (la que refleja lo cargado). El `e_pts`,
    en cambio, sale de la última versión generada ANTES del cierre de cada partido.

    La distinción no es cosmética, es lo que hace que "esperado vs real" signifique
    algo. Para un partido ya jugado `build_season_grids` usa una grilla DELTA —toda
    la masa en el resultado— así que el `e_pts` que guarda una planilla posterior ES
    el puntaje obtenido. Comparar eso contra lo real daba identidad perfecta en las
    cinco filas y no medía nada (detectado el 2026-08-10).
    """
    from src.clausura.picks import fecha_dir
    from src.utils.versions import latest_version, version_num

    d = fecha_dir(fecha_n)
    latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
    if latest is None:
        raise FileNotFoundError(f"no hay planilla guardada para la fecha {fecha_n}")
    data = json.loads(latest.read_text(encoding="utf-8"))
    picks: dict[int, dict[int, tuple[int, int]]] = {n: {} for n in mis_numeros}
    for row in data.get("picks", []):
        eid = int(row["evento_id"])
        for k, score in enumerate(row.get("scores", [])):
            if k < len(mis_numeros):
                picks[mis_numeros[k]][eid] = (int(score[0]), int(score[1]))

    # e_pts pre-partido: recorriendo las versiones de la MÁS VIEJA a la más nueva y
    # quedándose con la última cuyo `generado_utc` sea anterior al cierre del evento.
    e_pts: dict[int, dict[int, float]] = {}
    for path in sorted(d.glob("v*_*.json"), key=version_num):
        payload = json.loads(path.read_text(encoding="utf-8"))
        generado = payload.get("generado_utc")
        if not generado:
            continue
        gen = datetime.fromisoformat(generado)
        for row in payload.get("picks", []):
            if "e_pts" not in row or not row.get("cierre_pronostico_utc"):
                continue
            if gen >= datetime.fromisoformat(row["cierre_pronostico_utc"]):
                continue                    # generada con el partido ya cerrado
            e_pts[int(row["evento_id"])] = {
                mis_numeros[k]: float(v)
                for k, v in enumerate(row["e_pts"]) if k < len(mis_numeros)
            }
    return picks, e_pts


def latest_snapshot_participaciones() -> list[dict]:
    """Participaciones del último snapshot FRESCO (≤48h, la misma cota que picks).

    Sin la cota: si el escaneo del ExecStartPre aborta (429), el postmortem
    comparaba contra un snapshot sin los picks de los partidos recién cerrados →
    puntos del pool subestimados → "ganaste la fecha" en el reporte sin haberla
    ganado. Mejor un postmortem sin distribución del pool que uno que miente.
    """
    from src.clausura.pool_snapshot import load_latest_snapshot
    snap = load_latest_snapshot(max_age_hours=48)
    if snap is None:
        log.warning("sin snapshot fresco (≤48h) — el postmortem sale sin "
                    "distribución del pool en vez de usar una vieja que miente")
        return []
    return snap.get("participaciones", [])


# -------------------- cómputo (puro, testeable) --------------------

def compute_stats(
    fecha_n: int,
    eventos_fecha: list[dict],
    resultados: dict[int, tuple[int, int]],
    picks: dict[int, dict[int, tuple[int, int]]],
    e_pts: dict[int, dict[int, float]],
    pool_participaciones: list[dict],
    mis_numeros: set[int],
) -> FechaStats:
    st = FechaStats(fecha_n=fecha_n, resultados=resultados, eventos=eventos_fecha)
    pref_de = {ev["evento_id"]: bool(ev.get("preferencial")) for ev in eventos_fecha}

    for numero, mios in picks.items():
        total = exactos = 0
        esperado = 0.0
        tiene_esperado = False
        for eid, real in resultados.items():
            pick = mios.get(eid)
            if pick is None:
                continue
            pts = supermatch_points(pick, real, pref_de.get(eid, False))
            total += pts
            if pick == real:
                exactos += 1
            st.detalle.setdefault(eid, {})[numero] = {"pick": pick, "pts": pts}
            if eid in e_pts and numero in e_pts[eid]:
                esperado += e_pts[eid][numero]
                tiene_esperado = True
        st.puntos[numero] = total
        st.exactos[numero] = exactos
        if tiene_esperado:
            st.esperados[numero] = round(esperado, 1)

    # pool: puntos de la fecha por rival (desde el snapshot de picks públicos)
    exact_hits: dict[int, int] = {eid: 0 for eid in resultados}
    n_con_picks = 0
    for r in pool_participaciones:
        if int(r.get("numero", -1)) in mis_numeros:
            continue
        rp = {int(k): (int(v[0]), int(v[1])) for k, v in r.get("picks", {}).items()}
        relevantes = {eid: rp[eid] for eid in resultados if eid in rp}
        if not relevantes:
            continue
        n_con_picks += 1
        pts = 0
        for eid, pick in relevantes.items():
            pts += supermatch_points(pick, resultados[eid], pref_de.get(eid, False))
            if pick == resultados[eid]:
                exact_hits[eid] += 1
        st.pool_puntos.append(pts)
    if n_con_picks:
        st.pool_exactos_por_evento = {eid: h / n_con_picks for eid, h in exact_hits.items()}
    return st


def _percentil(valor: int, pool: list[int]) -> float:
    """Fracción del pool que quedó ESTRICTAMENTE por debajo."""
    if not pool:
        return 0.0
    return sum(1 for p in pool if p < valor) / len(pool)


# -------------------- reporte --------------------

def formatear_postmortem(st: FechaStats, premio_fecha: float | None = None) -> str:
    ev_by_id = {ev["evento_id"]: ev for ev in st.eventos}
    lines = [f"<b>📊 Postmortem — Fecha {st.fecha_n}</b>"]

    # partidos: resultado + cómo nos fue + exactos del pool
    lines.append("")
    for eid, real in st.resultados.items():
        ev = ev_by_id.get(eid, {})
        pref = " ⭐" if ev.get("preferencial") else ""
        det = st.detalle.get(eid, {})
        nuestros = [d["pts"] for d in det.values()]
        exactos_nuestros = sum(1 for d in det.values() if d["pick"] == real)
        pool_ex = st.pool_exactos_por_evento.get(eid)
        pool_txt = f" · pool exacto {pool_ex:.0%}" if pool_ex is not None else ""
        mejor = max(nuestros) if nuestros else 0
        lines.append(
            f"{ev.get('local', '?')} <b>{real[0]}-{real[1]}</b> {ev.get('visitante', '?')}{pref}"
            f" — nuestras: mejor {mejor} pts, {exactos_nuestros}/{max(len(det), 1)} "
            f"exacto{pool_txt}")

    # participaciones: real (vs esperado), percentil en el pool
    lines.append("")
    lines.append("<b>Por participación</b> (pts fecha · vs pool)")
    for numero in sorted(st.puntos, key=lambda n: -st.puntos[n]):
        pts = st.puntos[numero]
        pct = _percentil(pts, st.pool_puntos)
        esp = st.esperados.get(numero)
        esp_txt = f" (E {esp:.0f})" if esp is not None else ""
        ex = st.exactos[numero]
        ex_txt = f" · {ex} exacto{'s' if ex != 1 else ''}" if ex else ""
        lines.append(f"  {numero % 1000:03d}: <b>{pts}</b>{esp_txt} · top {1 - pct:.0%}{ex_txt}")

    # pool
    if st.pool_puntos:
        arr = np.array(st.pool_puntos)
        mejor_nuestra = max(st.puntos.values(), default=0)
        nos_ganaron = int((arr > mejor_nuestra).sum())
        lines.append("")
        lines.append(f"<b>Pool</b>: {len(arr)} rivales · mediana {int(np.median(arr))} · "
                     f"máx {int(arr.max())} · {nos_ganaron} arriba de nuestra mejor")
        if premio_fecha and nos_ganaron == 0 and mejor_nuestra >= int(arr.max()):
            lines.append(f"🏆 Nuestra mejor participación empató o ganó la fecha "
                         f"(premio ${premio_fecha:,.0f})")
        pool_exact_rate = float(np.mean(list(st.pool_exactos_por_evento.values()))) \
            if st.pool_exactos_por_evento else None
        if pool_exact_rate is not None:
            lines.append(f"Tasa de exactos del pool en la fecha: {pool_exact_rate:.1%} "
                         f"(insumo de calibración — el pipeline la absorbe solo)")
    else:
        lines.append("")
        lines.append("<i>Sin snapshot del pool para esta fecha — comparación vs rivales omitida.</i>")

    return "\n".join(lines)


# -------------------- persistencia / orquestación --------------------

def pm_path(fecha_n: int) -> Path:
    return PM_DIR / f"fecha_{fecha_n:02d}.json"


# Cuántos partidos de la fecha tienen que estar jugados para que valga un postmortem.
# 6 de 8 deja afuera la fecha en curso (viernes/sábado) pero no se traba con uno o dos
# suspendidos, que es lo que impedía que este módulo corriera alguna vez.
MIN_JUGADOS = 6


def _resultados_guardados(n: int) -> int:
    """Cuántos resultados tiene el postmortem ya escrito de esa fecha. 0 si no hay."""
    p = pm_path(n)
    if not p.exists():
        return 0
    try:
        return len(json.loads(p.read_text(encoding="utf-8")).get("resultados") or {})
    except Exception:                                          # noqa: BLE001
        return 0


def fecha_a_analizar(cfg: dict) -> int | None:
    """La fecha más vieja cuyo postmortem falta o quedó INCOMPLETO.

    "Suficientes" y no "todos": un partido suspendido —Torque–Peñarol en la Fecha 1—
    se reprograma para dentro de semanas, y exigir la fecha completa dejaba el
    análisis de los otros siete sin correr nunca.

    Pero disparar con el mínimo y no volver nunca congela datos parciales: el
    postmortem de la Fecha 1 se escribió el 2026-08-10 con **6 de 8** partidos —le
    faltaban Torque–Peñarol y Maldonado–Racing, que se jugaron esa misma noche— y no
    se iba a rehacer jamás. La herramienta que existe para aprender de lo que pasó
    quedaba permanentemente equivocada sobre la primera fecha.

    Por eso la condición no es "existe el archivo" sino "el archivo tiene TODOS los
    resultados que hoy se pueden ver". Se rehace solo cuando aparecen más.
    """
    nums = sorted(int(n.split()[-1]) for n in cfg["fechas"])
    for n in nums:
        res = resultados_de_fecha(cfg, n, min_jugados=MIN_JUGADOS)
        if res is None:
            break   # las fechas van en orden: si esta no llegó, las siguientes tampoco
        disponibles, _ = res
        guardados = _resultados_guardados(n)
        if guardados >= len(disponibles):
            continue          # al día: o está completo, o no aparecieron resultados nuevos
        if guardados:
            log.info("rehaciendo el postmortem de la fecha %d: tenía %d resultados y "
                     "ahora hay %d", n, guardados, len(disponibles))
        return n    # una por corrida: la más vieja pendiente
    return None


def run(fecha: int | None = None, dry_run: bool = False) -> str | None:
    from src.clausura.picks import flat_eventos, load_config

    cfg = load_config()
    if fecha is None:
        fecha = fecha_a_analizar(cfg)
        if fecha is None:
            log.info("sin fechas completas pendientes de postmortem")
            return None

    datos = resultados_de_fecha(cfg, fecha, min_jugados=1)
    if datos is None:
        log.error("la fecha %d no tiene ningún partido jugado todavía", fecha)
        return None
    resultados, faltan = datos
    if faltan:
        log.warning("fecha %d: %d partido(s) sin resultado (suspendido o por jugar) — "
                    "se analiza lo jugado; regenerá el postmortem cuando se resuelvan",
                    fecha, faltan)

    mis_numeros = sorted(mis_numeros_env())
    eventos_fecha = [ev for ev in flat_eventos(cfg) if ev["fecha_n"] == fecha]

    # Antes de leer la planilla, re-adoptar lo que la web dice de los partidos
    # cerrados. La adopción del drift de las 23:50 puede quedar SOMBREADA por un
    # rerun en vuelo (arranca 23:35, lee la latest pre-adopción y escribe 30-60
    # min después): sin esta pasada, el postmortem atribuía puntos con picks que
    # nunca se jugaron — y no se rehace, porque su condición de regeneración mira
    # resultados nuevos, no cambios de planilla.
    if not dry_run:
        try:
            from src.clausura.drift_audit import adoptar_picks_cerrados, fetch_cargados
            cargados = fetch_cargados(cfg["pencas"]["paga"]["id"], set(mis_numeros))
            adoptar_picks_cerrados(cargados, mis_numeros, eventos_fecha,
                                   datetime.now(timezone.utc))
        except Exception as e:                                 # noqa: BLE001
            log.warning("no pude re-adoptar los picks reales antes del postmortem "
                        "(%s) — sigo con la planilla guardada", e)

    picks, e_pts = picks_de_planilla(fecha, mis_numeros)
    pool = latest_snapshot_participaciones()

    st = compute_stats(fecha, eventos_fecha, resultados, picks, e_pts,
                       pool, set(mis_numeros))
    premios = {p["tipo"]: p["monto"] for p in cfg.get("premios", [])}
    reporte = formatear_postmortem(st, premio_fecha=premios.get("FECHA"))

    print(reporte.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    if not dry_run:
        PM_DIR.mkdir(parents=True, exist_ok=True)
        pm_path(fecha).write_text(json.dumps({
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "fecha": fecha,
            "resultados": {str(k): list(v) for k, v in st.resultados.items()},
            "puntos": st.puntos,
            "exactos": st.exactos,
            "esperados": st.esperados,
            "pool_puntos": st.pool_puntos,
            "pool_exactos_por_evento": {str(k): v for k, v in st.pool_exactos_por_evento.items()},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        from src.notifier.telegram import TelegramConfig, TelegramNotifier
        TelegramNotifier(TelegramConfig.from_env()).send(reporte)
    return reporte


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", type=int, default=None,
                    help="forzar una fecha puntual (default: detectar la pendiente)")
    ap.add_argument("--dry-run", action="store_true",
                    help="imprime el reporte sin Telegram ni archivo de estado")
    args = ap.parse_args()
    run(fecha=args.fecha, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

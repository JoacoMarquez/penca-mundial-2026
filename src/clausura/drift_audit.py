"""Auditoría anti-drift: lo cargado en la web vs la planilla guardada.

La carga en Supermatch es manual, y el optimizador CONGELA los picks de fechas
anteriores leyendo la planilla versionada (load_frozen). Si lo que quedó cargado en
la web difiere de la planilla — un dedo en el celular, una participación salteada —
el modelo optimiza sobre un estado falso. Este módulo cierra ese riesgo:

  1. Baja los picks reales de NUESTRAS participaciones (públicos post-inicio,
     mismo endpoint que pool_snapshot; el gate es el inicio del campeonato).
  2. Los compara contra la última planilla guardada de cada fecha en
     data/predictions/clausura/fecha_NN/ (convención: columna i ↔ i-ésimo número
     de CLAUSURA_MIS_PARTICIPACIONES ordenado ascendente — la misma que muestran
     las tarjetas del modo carga del dashboard).
  3. Avisa por Telegram cada discrepancia, UNA vez por (evento, número, valores)
     — estado en disco, igual que carga_alert.

Tipos de discrepancia:
  - distinto:            cargado ≠ planilla (lo más grave; si el cierre no pasó, corrégelo)
  - sin_cargar_cerrado:  el cierre pasó y esa participación quedó sin pick (0 seguro)
  - sin_planilla:        hay un pick cargado para un evento sin planilla guardada
  (faltantes ANTES del cierre no son drift — de eso se ocupa carga_alert)

También audita los especiales (campeón/goleador) contra la última planilla que los
tenga, con una asimetría deliberada (2026-08-05: el usuario cargó especiales por la
web mientras los endpoints públicos de opciones daban 500 — el front autenticado usa
otro canal):

  - **Campeón**: la planilla lo asignó → mismatch = drift, se reporta.
  - **Goleador**: la planilla nunca pudo asignarlo (sin menú vía API) → lo cargado
    en la web ES la verdad. Post-inicio, el audit lo ADOPTA: versiona una planilla
    nueva con los goleadores reales para que el freeze del optimizador y las
    auditorías siguientes queden alineados. Se notifica una vez.

Uso:
    python -m src.clausura.drift_audit               # audita y avisa (error si no inició)
    python -m src.clausura.drift_audit --if-started  # sale 0 en silencio pre-inicio (timer)
    python -m src.clausura.drift_audit --dry-run     # imprime sin Telegram ni estado
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.clausura.api import BASE, HEADERS, PencaApiClient
from src.clausura.rivals import mis_numeros_env

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "state" / "drift_audit.json"
GATE_MSG = "campeonato ya inicio"


class CampeonatoNoIniciado(RuntimeError):
    pass


@dataclass(frozen=True)
class Cargado:
    """Estado real de una participación nuestra en la web."""
    numero: int
    picks: dict[int, tuple[int, int]]      # evento_id → (gl, gv)
    campeon: str | None = None
    goleador: str | None = None
    # pre-inicio el endpoint de especiales devuelve 400 (gate): no se puede saber
    # si están cargados — False evita el falso positivo "sin cargar"
    especiales_visibles: bool = True


@dataclass(frozen=True)
class Discrepancia:
    tipo: str                              # distinto | sin_cargar_cerrado | sin_planilla | especial
    numero: int
    detalle: str                           # texto ya formateado para el aviso
    clave: str                             # identidad para no re-avisar


# -------------------- planilla esperada (disco) --------------------

def planilla_esperada(
    eventos: list[dict],
    mis_numeros: list[int],
) -> dict[int, dict[int, tuple[int, int]]]:
    """evento_id → {numero: (gl, gv)} desde la última planilla guardada de cada fecha.

    `mis_numeros` viene ORDENADO ascendente: columna k de la planilla ↔ mis_numeros[k]
    (convención del modo carga del dashboard).
    """
    from src.clausura.picks import fecha_dir
    from src.utils.versions import latest_version

    fechas = sorted({ev["fecha_n"] for ev in eventos})
    esperado: dict[int, dict[int, tuple[int, int]]] = {}
    for f in fechas:
        d = fecha_dir(f)
        latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
        if latest is None:
            continue
        data = json.loads(latest.read_text(encoding="utf-8"))
        for row in data.get("picks", []):
            por_numero = {}
            for k, score in enumerate(row.get("scores", [])):
                if k < len(mis_numeros):
                    por_numero[mis_numeros[k]] = (int(score[0]), int(score[1]))
            esperado[int(row["evento_id"])] = por_numero
    return esperado


def especiales_esperados(mis_numeros: list[int]) -> dict[int, tuple[str | None, str | None]]:
    """numero → (campeón, goleador) desde la última planilla que tenga especiales."""
    from src.clausura.picks import fecha_dir, load_config
    from src.utils.versions import latest_version

    cfg = load_config()
    n_fechas = len(cfg.get("fechas", []))
    for f in range(n_fechas, 0, -1):
        d = fecha_dir(f)
        latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
        if latest is None:
            continue
        rows = (json.loads(latest.read_text(encoding="utf-8"))
                .get("especiales", {}).get("por_participacion", []))
        if rows:
            return {mis_numeros[i]: (r.get("campeon"), r.get("goleador"))
                    for i, r in enumerate(rows) if i < len(mis_numeros)}
    return {}


# -------------------- estado real (API) --------------------

def fetch_cargados(penca_id: int, mis_numeros: set[int]) -> list[Cargado]:
    """Picks + especiales reales de nuestras participaciones. Lanza
    CampeonatoNoIniciado si el gate del API sigue cerrado."""
    with PencaApiClient() as api:
        ranking = api.ranking(penca_id)
    mios = [r for r in ranking if r.numero_participacion in mis_numeros]
    faltan = mis_numeros - {r.numero_participacion for r in mios}
    if faltan:
        log.warning("números no encontrados en el ranking: %s", sorted(faltan))

    out: list[Cargado] = []
    with httpx.Client(base_url=BASE, timeout=20.0, headers=HEADERS) as c:
        for r in mios:
            resp = c.get(f"/front/pencas/{r.participacion_id}/pronosticosEventos")
            if resp.status_code == 400 and GATE_MSG in resp.text:
                raise CampeonatoNoIniciado(resp.text[:120])
            picks: dict[int, tuple[int, int]] = {}
            if resp.status_code == 200:
                for p in resp.json().get("data", []):
                    gl, gv = p.get("golesEquipoLocal"), p.get("golesEquipoVisitante")
                    eid = p.get("encuentroId")
                    if gl is not None and gv is not None and eid is not None:
                        picks[int(eid)] = (int(gl), int(gv))
            else:
                log.warning("pronosticosEventos %d → %d", r.participacion_id, resp.status_code)

            campeon = goleador = None
            visibles = True
            resp = c.get(f"/front/pencas/{r.participacion_id}/pronosticoCampeonGoleador")
            if resp.status_code == 200:
                d = resp.json()
                campeon = (d.get("equipoCampeon") or {}).get("nombre")
                goleador = (d.get("opcionGoleador") or {}).get("goleador")
            elif resp.status_code == 400 and GATE_MSG in resp.text:
                visibles = False
            else:
                log.warning("pronosticoCampeonGoleador %d → %d",
                            r.participacion_id, resp.status_code)
                visibles = False

            out.append(Cargado(numero=r.numero_participacion, picks=picks,
                               campeon=campeon, goleador=goleador,
                               especiales_visibles=visibles))
    return out


# -------------------- diff (puro, testeable) --------------------

def diff_picks(
    eventos: list[dict],
    esperado: dict[int, dict[int, tuple[int, int]]],
    cargados: list[Cargado],
    now: datetime,
) -> list[Discrepancia]:
    ev_by_id = {ev["evento_id"]: ev for ev in eventos}
    out: list[Discrepancia] = []

    for c in cargados:
        for eid, esp_por_numero in esperado.items():
            ev = ev_by_id.get(eid)
            if ev is None:
                continue
            esp = esp_por_numero.get(c.numero)
            if esp is None:
                continue
            cierre = datetime.fromisoformat(ev["cierre_pronostico_utc"])
            car = c.picks.get(eid)
            partido = f"{ev['local']} vs {ev['visitante']}"

            if car is None:
                if now >= cierre:
                    out.append(Discrepancia(
                        "sin_cargar_cerrado", c.numero,
                        f"{partido}: cerró sin pick (planilla decía {esp[0]}-{esp[1]})",
                        f"faltante:{eid}:{c.numero}"))
            elif car != esp:
                estado = "cerrado — puntos comprometidos" if now >= cierre else "AÚN CORREGIBLE"
                out.append(Discrepancia(
                    "distinto", c.numero,
                    f"{partido}: cargado {car[0]}-{car[1]} pero la planilla dice "
                    f"{esp[0]}-{esp[1]} ({estado})",
                    f"distinto:{eid}:{c.numero}:{esp[0]}-{esp[1]}:{car[0]}-{car[1]}"))

        for eid, car in c.picks.items():
            if eid not in esperado and eid in ev_by_id:
                ev = ev_by_id[eid]
                out.append(Discrepancia(
                    "sin_planilla", c.numero,
                    f"{ev['local']} vs {ev['visitante']}: hay {car[0]}-{car[1]} cargado "
                    f"pero ninguna planilla guardada lo cubre",
                    f"sin_planilla:{eid}:{c.numero}:{car[0]}-{car[1]}"))
    return out


def diff_especiales(
    esperados: dict[int, tuple[str | None, str | None]],
    cargados: list[Cargado],
) -> list[Discrepancia]:
    out: list[Discrepancia] = []
    for c in cargados:
        if not c.especiales_visibles:
            continue        # gate cerrado (pre-inicio): no verificable, no es drift
        esp = esperados.get(c.numero)
        if esp is None:
            continue
        for etiqueta, esperado, cargado in (("campeón", esp[0], c.campeon),
                                            ("goleador", esp[1], c.goleador)):
            if esperado is None:
                continue        # la planilla no lo definió (p.ej. menú de goleador aún caído)
            if cargado != esperado:
                car_txt = cargado or "sin cargar"
                out.append(Discrepancia(
                    "especial", c.numero,
                    f"{etiqueta}: cargado «{car_txt}» pero la planilla dice «{esperado}»",
                    f"especial:{etiqueta}:{c.numero}:{esperado}:{car_txt}"))
    return out


def formatear_reporte(discrepancias: list[Discrepancia]) -> str:
    """Mensaje de Telegram agrupado por participación."""
    lines = ["<b>⚠️ Auditoría de carga — Penca Clausura</b>",
             f"{len(discrepancias)} discrepancia(s) entre la web y la planilla:"]
    por_numero: dict[int, list[Discrepancia]] = {}
    for d in discrepancias:
        por_numero.setdefault(d.numero, []).append(d)
    for numero in sorted(por_numero):
        lines.append(f"\n<b>Participación {numero}</b>")
        for d in por_numero[numero]:
            icono = {"distinto": "❌", "sin_cargar_cerrado": "🕳️",
                     "sin_planilla": "❓", "especial": "⭐"}[d.tipo]
            lines.append(f"  {icono} {d.detalle}")
    lines.append("\nEl optimizador congela lo que dice la PLANILLA — si la web quedó "
                 "distinta, corregí la web o regenerá la planilla.")
    return "\n".join(lines)


# -------------------- adopción de especiales reales --------------------

def payload_con_goleadores_reales(
    payload: dict,
    cargados: list[Cargado],
    mis_numeros: list[int],
) -> tuple[dict, list[tuple[int, str]]] | None:
    """(payload nuevo, [(numero, goleador)]) adoptando los goleadores cargados en la
    web donde la planilla no tiene (None). None si no hay nada que adoptar.

    Nota: goleador_idx queda en -1 (sin menú vía API no hay índice); si el endpoint
    de opciones alguna vez responde, el nombre adoptado es la referencia.
    """
    gol_por_numero = {c.numero: c.goleador for c in cargados
                      if c.especiales_visibles and c.goleador}
    if not gol_por_numero:
        return None
    esp = payload.setdefault("especiales", {})
    rows = esp.get("por_participacion") or []
    while len(rows) < len(mis_numeros):
        rows.append({"campeon_idx": -1, "campeon": None,
                     "goleador_idx": -1, "goleador": None})
    adoptados: list[tuple[int, str]] = []
    for k, numero in enumerate(mis_numeros):
        real = gol_por_numero.get(numero)
        if real and not rows[k].get("goleador"):
            rows[k]["goleador"] = real
            rows[k]["goleador_idx"] = -1
            adoptados.append((numero, real))
    if not adoptados:
        return None
    esp["por_participacion"] = rows
    payload["especiales_adoptados_utc"] = datetime.now(timezone.utc).isoformat()
    return payload, adoptados


def adoptar_goleadores(cargados: list[Cargado], mis_numeros: list[int]) -> str | None:
    """Versiona una planilla nueva con los goleadores reales. Devuelve el texto del
    aviso, o None si no había nada que adoptar."""
    import src.clausura.picks as picks
    from src.utils.versions import latest_version

    target = picks.resolve_fecha("auto")
    d = picks.fecha_dir(target)
    latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
    if latest is None:
        log.warning("sin planilla de la fecha %d — no puedo adoptar goleadores", target)
        return None
    payload = json.loads(latest.read_text(encoding="utf-8"))
    resultado = payload_con_goleadores_reales(payload, cargados, mis_numeros)
    if resultado is None:
        return None
    payload, adoptados = resultado
    path = picks.save_version(target, payload)
    log.info("goleadores reales adoptados → %s", path.name)
    lines = ["<b>📌 Goleadores cargados en la web, adoptados a la planilla</b>",
             "(la planilla no tenía asignación — sin menú vía API):"]
    lines += [f"  {numero}: <b>{gol}</b>" for numero, gol in adoptados]
    return "\n".join(lines)


# -------------------- estado --------------------

def load_state() -> set[str]:
    if STATE_PATH.exists():
        return set(json.loads(STATE_PATH.read_text(encoding="utf-8")))
    return set()


def save_state(avisadas: set[str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(sorted(avisadas)), encoding="utf-8")


# -------------------- main --------------------

def run(dry_run: bool = False, now: datetime | None = None) -> list[Discrepancia]:
    from src.clausura.picks import flat_eventos, load_config

    now = now or datetime.now(timezone.utc)
    cfg = load_config()
    eventos = flat_eventos(cfg)
    mis_numeros = sorted(mis_numeros_env())
    if not mis_numeros:
        log.error("CLAUSURA_MIS_PARTICIPACIONES vacío — no hay qué auditar")
        return []

    esperado = planilla_esperada(eventos, mis_numeros)
    if not esperado:
        log.info("sin planillas guardadas todavía — nada que auditar")
        return []

    cargados = fetch_cargados(cfg["pencas"]["paga"]["id"], set(mis_numeros))

    # goleadores cargados en la web sin asignación en la planilla → adoptarlos
    # (una vez; después la planilla ya los tiene y esto devuelve None)
    aviso_adopcion = None
    if not dry_run:
        try:
            aviso_adopcion = adoptar_goleadores(cargados, mis_numeros)
        except Exception as e:
            log.warning("adopción de goleadores falló (%s)", e)

    discrepancias = (diff_picks(eventos, esperado, cargados, now)
                     + diff_especiales(especiales_esperados(mis_numeros), cargados))

    avisadas = load_state()
    nuevas = [d for d in discrepancias if d.clave not in avisadas]
    n_eventos = len(esperado)
    if not discrepancias:
        log.info("auditoría OK: %d participaciones vs %d eventos con planilla, sin drift",
                 len(cargados), n_eventos)
    elif not nuevas:
        log.info("%d discrepancia(s) persisten pero ya fueron avisadas", len(discrepancias))

    partes = []
    if aviso_adopcion:
        partes.append(aviso_adopcion)
    if nuevas:
        partes.append(formatear_reporte(nuevas))
    if partes:
        reporte = "\n\n".join(partes)
        print(reporte.replace("<b>", "").replace("</b>", ""))
        if not dry_run:
            from src.notifier.telegram import TelegramConfig, TelegramNotifier
            TelegramNotifier(TelegramConfig.from_env()).send(reporte)
            if nuevas:
                save_state(avisadas | {d.clave for d in nuevas})
    return discrepancias


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-started", action="store_true",
                    help="si el campeonato no inició, salir 0 en silencio (para timers)")
    ap.add_argument("--dry-run", action="store_true",
                    help="imprime el reporte sin mandar Telegram ni marcar estado")
    args = ap.parse_args()
    try:
        run(dry_run=args.dry_run)
    except CampeonatoNoIniciado:
        if args.if_started:
            print("campeonato no iniciado — auditoría omitida")
            sys.exit(0)
        print("ERROR: el campeonato no inició; los picks aún no son públicos", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

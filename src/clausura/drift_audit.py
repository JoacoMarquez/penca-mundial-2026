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

Además de avisar, el audit ADOPTA: para partidos CERRADOS la web es la verdad
inmutable, así que si la planilla difiere (típicamente porque el rerun T-2h
versionó picks que el gate por valor descartó y nadie recargó), se versiona una
planilla realineada. Sin eso, load_frozen congela picks que nunca jugamos y el
optimizador diversifica contra una posición propia falsa toda la temporada.

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
from datetime import datetime, timedelta, timezone
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
    # un fallo HTTP en pronosticosEventos NO significa "no hay pick": False saca a
    # la participación del diff de esa corrida (mismo patrón que especiales_visibles)
    picks_visibles: bool = True


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


# -------------------- penca GRATUITA (1 participación, columna 1) --------------------

def gratuita_esperada(eventos: list[dict]) -> dict[int, tuple[int, int]]:
    """evento_id → pick de la COLUMNA 1 (ancla EV) de la última planilla de cada fecha.

    La gratuita se juega con EV puro: premio indivisible con desempate por exactos,
    así que empatar no diluye y diferenciarse no paga (análisis 2026-08-05).
    """
    from src.clausura.picks import fecha_dir
    from src.utils.versions import latest_version

    out: dict[int, tuple[int, int]] = {}
    for f in sorted({ev["fecha_n"] for ev in eventos}):
        d = fecha_dir(f)
        latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
        if latest is None:
            continue
        data = json.loads(latest.read_text(encoding="utf-8"))
        for row in data.get("picks", []):
            scores = row.get("scores") or []
            if scores:
                out[int(row["evento_id"])] = (int(scores[0][0]), int(scores[0][1]))
    return out


def buscar_por_picks(
    penca_id: int,
    esperado: dict[int, tuple[int, int]],
    min_coincidencias: int = 4,
) -> list[tuple[int, int, int]]:
    """Escanea el ranking buscando qué participación tiene los picks esperados.

    Devuelve [(numero, coincidencias, comparables)] ordenado por coincidencias desc,
    solo las que llegan a `min_coincidencias`. Es la detección que pedimos para el
    caso "el número configurado no matchea": con 8 marcadores la combinación es
    prácticamente única, así que si otra participación calza, ESA es la nuestra.

    Caro (1 request por participación), así que solo se llama cuando hace falta.
    """
    import time
    with PencaApiClient() as api:
        ranking = api.ranking(penca_id)
    out: list[tuple[int, int, int]] = []
    with httpx.Client(base_url=BASE, timeout=20.0, headers=HEADERS) as c:
        for r in ranking:
            resp = c.get(f"/front/pencas/{r.participacion_id}/pronosticosEventos")
            if resp.status_code == 400 and GATE_MSG in resp.text:
                raise CampeonatoNoIniciado(resp.text[:120])
            if resp.status_code != 200:
                continue
            hits = comparables = 0
            for p in resp.json().get("data", []):
                eid = p.get("encuentroId")
                gl, gv = p.get("golesEquipoLocal"), p.get("golesEquipoVisitante")
                if eid is None or gl is None or gv is None:
                    continue
                esp = esperado.get(int(eid))
                if esp is not None:
                    comparables += 1
                    hits += (int(gl), int(gv)) == esp
            if hits >= min_coincidencias:
                out.append((r.numero_participacion, hits, comparables))
            time.sleep(0.12)
    return sorted(out, key=lambda t: -t[1])


def auditar_gratuita(eventos: list[dict], cfg: dict, now: datetime) -> list[Discrepancia]:
    """Audita la participación de la penca gratuita contra la columna 1.

    Si el número configurado no matchea, escanea el ranking por picks para
    detectar cuál es realmente nuestra (o avisar que no hay ninguna).
    """
    import os

    penca = (cfg.get("pencas") or {}).get("gratuita") or {}
    penca_id = penca.get("id")
    raw = os.environ.get("CLAUSURA_MI_PARTICIPACION_GRATUITA", "").strip()
    if penca_id is None or not raw.isdigit():
        return []
    numero = int(raw)

    esperado = gratuita_esperada(eventos)
    if not esperado:
        return []

    cargados = fetch_cargados(penca_id, {numero})
    if not cargados:
        return [Discrepancia("gratuita", numero,
                             f"la participación {numero} no aparece en el ranking de "
                             f"la penca gratuita (id {penca_id})",
                             f"gratuita:ausente:{numero}")]

    c = cargados[0]
    ev_by_id = {ev["evento_id"]: ev for ev in eventos}
    out: list[Discrepancia] = []
    hits = comparables = 0
    for eid, esp in esperado.items():
        ev = ev_by_id.get(eid)
        if ev is None:
            continue
        real = c.picks.get(eid)
        cerrado = datetime.fromisoformat(ev["cierre_pronostico_utc"]) <= now
        if real is None:
            if cerrado:
                out.append(Discrepancia(
                    "gratuita", numero,
                    f"gratuita · {ev['local']} vs {ev['visitante']}: sin cargar y "
                    f"el cierre ya pasó (0 puntos)",
                    f"gratuita:sincargar:{eid}:{numero}"))
            continue
        comparables += 1
        if real == esp:
            hits += 1
        else:
            estado = "cerrado, puntos comprometidos" if cerrado else "AÚN CORREGIBLE"
            out.append(Discrepancia(
                "gratuita", numero,
                f"gratuita · {ev['local']} vs {ev['visitante']}: cargado "
                f"{real[0]}-{real[1]} pero la planilla dice {esp[0]}-{esp[1]} ({estado})",
                f"gratuita:distinto:{eid}:{numero}:{real[0]}-{real[1]}"))

    # ninguna coincidencia sobre varios partidos ⇒ probablemente el número está mal:
    # buscamos por picks cuál es la nuestra de verdad
    if comparables >= 4 and hits == 0:
        log.warning("gratuita: 0/%d picks coinciden — escaneando el ranking por picks",
                    comparables)
        try:
            candidatas = buscar_por_picks(penca_id, esperado)
        except Exception as e:
            log.warning("escaneo por picks falló (%s)", e)
            candidatas = []
        if candidatas:
            n, h, tot = candidatas[0]
            out.append(Discrepancia(
                "gratuita", numero,
                f"gratuita · NINGÚN pick de {numero} coincide, pero la participación "
                f"<b>{n}</b> tiene {h}/{tot} iguales a la columna 1 → revisá "
                f"CLAUSURA_MI_PARTICIPACION_GRATUITA (¿es {n}?)",
                f"gratuita:numero_sospechoso:{numero}:{n}"))
        else:
            out.append(Discrepancia(
                "gratuita", numero,
                f"gratuita · ningún pick de {numero} coincide con la columna 1 y "
                f"ninguna otra participación calza — ¿se cargó la gratuita?",
                f"gratuita:sin_match:{numero}"))
    elif comparables and not out:
        log.info("gratuita OK: %d/%d picks coinciden con la columna 1", hits, comparables)
    return out


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

    # Pacing y backoff como el escaneo del pool (mismo endpoint, mismas ~24
    # requests): la corrida de las 23:50 llega con el presupuesto de rate-limit
    # más gastado del día, y un 429 pelado dejaba picks={} — que diff_picks leía
    # como "cerró sin pick", una alarma falsa POR EVENTO en el canal de señales
    # reales, con la clave de estado quemada para siempre.
    from src.clausura.pool_snapshot import REQUEST_PAUSE_S, _get_pacing

    out: list[Cargado] = []
    with httpx.Client(base_url=BASE, timeout=20.0, headers=HEADERS) as c:
        for r in mios:
            resp = _get_pacing(c, f"/front/pencas/{r.participacion_id}/pronosticosEventos",
                               REQUEST_PAUSE_S)
            if resp.status_code == 400 and GATE_MSG in resp.text:
                raise CampeonatoNoIniciado(resp.text[:120])
            picks: dict[int, tuple[int, int]] = {}
            picks_visibles = True
            if resp.status_code == 200:
                for p in resp.json().get("data", []):
                    gl, gv = p.get("golesEquipoLocal"), p.get("golesEquipoVisitante")
                    eid = p.get("encuentroId")
                    if gl is not None and gv is not None and eid is not None:
                        picks[int(eid)] = (int(gl), int(gv))
            else:
                log.warning("pronosticosEventos %d → %d — participación NO verificable "
                            "en esta corrida", r.participacion_id, resp.status_code)
                picks_visibles = False

            campeon = goleador = None
            visibles = True
            resp = _get_pacing(c, f"/front/pencas/{r.participacion_id}/pronosticoCampeonGoleador",
                               REQUEST_PAUSE_S)
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
                               especiales_visibles=visibles,
                               picks_visibles=picks_visibles))
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
        if not c.picks_visibles:
            continue        # fallo HTTP: no verificable esta corrida, no es drift
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
                     "sin_planilla": "❓", "especial": "⭐",
                     "gratuita": "🎁"}.get(d.tipo, "•")
            lines.append(f"  {icono} {d.detalle}")
    lines.append("\nEl optimizador congela lo que dice la PLANILLA — si la web quedó "
                 "distinta, corregí la web o regenerá la planilla.")
    return "\n".join(lines)


# -------------------- adopción de picks reales (partidos cerrados) --------------------

def payload_con_picks_reales(
    payload: dict,
    cargados: list[Cargado],
    mis_numeros: list[int],
    ev_cerrados: set[int],
) -> tuple[dict, list[tuple[int, int, tuple[int, int], tuple[int, int]]]] | None:
    """(payload nuevo, [(evento_id, numero, planilla, web)]) reescribiendo los picks
    de partidos CERRADOS con lo que quedó cargado en la web. None si coinciden.

    Post-cierre la web es la verdad inmutable — el pick de la planilla ya no es una
    intención, es un registro, y si difiere el registro está MAL. La fuente típica
    del desvío no es un dedo: es el rerun T-2h, que versiona la planilla nueva
    aunque el gate por valor descarte el cambio (regla de trabajo #2). Sin esta
    adopción, load_frozen congela picks que NUNCA jugamos y el optimizador
    diversifica contra una posición propia falsa el resto de la temporada; el
    postmortem atribuye puntos con los mismos picks fantasma.

    Los `sin_cargar_cerrado` (la web no tiene pick) NO se adoptan: la planilla no
    puede representar "sin pick" y ese caso ya tiene alarma propia.
    """
    car_por_numero = {c.numero: c.picks for c in cargados}
    cambios: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
    for row in payload.get("picks", []):
        eid = int(row["evento_id"])
        if eid not in ev_cerrados:
            continue
        scores = row.get("scores") or []
        for k, numero in enumerate(mis_numeros):
            if k >= len(scores):
                break
            real = car_por_numero.get(numero, {}).get(eid)
            if real is None:
                continue
            plan = (int(scores[k][0]), int(scores[k][1]))
            if plan != real:
                cambios.append((eid, numero, plan, real))
                scores[k] = [real[0], real[1]]
    if not cambios:
        return None
    payload["picks_adoptados_utc"] = datetime.now(timezone.utc).isoformat()
    return payload, cambios


def adoptar_picks_cerrados(
    cargados: list[Cargado],
    mis_numeros: list[int],
    eventos: list[dict],
    now: datetime,
) -> str | None:
    """Versiona, por cada fecha con desvíos en partidos cerrados, una planilla
    realineada con la web. Devuelve el texto del aviso, o None si no hubo nada."""
    import src.clausura.picks as picks
    from src.utils.versions import latest_version

    por_fecha: dict[int, set[int]] = {}
    for ev in eventos:
        if datetime.fromisoformat(ev["cierre_pronostico_utc"]) <= now:
            por_fecha.setdefault(int(ev["fecha_n"]), set()).add(int(ev["evento_id"]))

    lineas: list[str] = []
    for f in sorted(por_fecha):
        d = picks.fecha_dir(f)
        latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
        if latest is None:
            continue
        payload = json.loads(latest.read_text(encoding="utf-8"))
        resultado = payload_con_picks_reales(payload, cargados, mis_numeros,
                                             por_fecha[f])
        if resultado is None:
            continue
        payload, cambios = resultado
        gate = (payload.get("veredicto_cambio") or {})
        path = picks.save_version(f, payload)
        log.info("picks reales adoptados en fecha %d → %s (%d picks)",
                 f, path.name, len(cambios))
        detalle = ", ".join(f"{n}: {p[0]}-{p[1]}→{r[0]}-{r[1]}"
                            for _, n, p, r in cambios[:6])
        extra = " y más" if len(cambios) > 6 else ""
        causa = (" — la planilla traía picks que el gate del rerun descartó "
                 "(esperado, no es error de carga)"
                 if gate.get("avisar") is False else "")
        lineas.append(f"  Fecha {f}: {len(cambios)} pick(s) realineados "
                      f"({detalle}{extra}){causa}")
    if not lineas:
        return None
    return ("<b>🧾 Planilla realineada con la web (partidos cerrados)</b>\n"
            "Post-cierre la web es la verdad: el freeze del optimizador y el "
            "postmortem ahora usan lo realmente cargado.\n" + "\n".join(lineas))


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

# Cuánto vive una clave avisada. Generoso a propósito: la clave incluye los VALORES
# (`distinto:{eid}:{numero}:{esperado}:{cargado}`), así que un cambio real genera
# clave nueva y se avisa igual — podar solo acota el archivo entre temporadas. Con
# 90 días ninguna clave expira dentro de un torneo, o sea que la poda nunca puede
# resucitar un aviso ya dado.
STATE_TTL_DIAS = 90


def load_state(now: datetime | None = None) -> set[str]:
    """Claves ya avisadas, podando las de más de STATE_TTL_DIAS."""
    if not STATE_PATH.exists():
        return set()
    now = now or datetime.now(timezone.utc)
    data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return set(data)     # formato viejo sin fechas: se migra al guardar
    corte = now - timedelta(days=STATE_TTL_DIAS)
    vivas = set()
    for clave, ts in data.items():
        try:
            if datetime.fromisoformat(ts) > corte:
                vivas.add(clave)
        except (TypeError, ValueError):
            vivas.add(clave)     # sin fecha legible: se conserva, no se pierde un aviso
    return vivas


def save_state(avisadas: set[str], now: datetime | None = None) -> None:
    """Guarda {clave: primera_vez_avisada}, conservando la fecha de las ya conocidas."""
    now = now or datetime.now(timezone.utc)
    previo: dict[str, str] = {}
    if STATE_PATH.exists():
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            previo = data
    ahora = now.isoformat()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({c: previo.get(c, ahora) for c in sorted(avisadas)}, indent=0),
        encoding="utf-8")


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

    # picks de partidos CERRADOS: la web es la verdad — realinear la planilla para
    # que load_frozen y el postmortem no arrastren picks que no jugamos. El diff de
    # abajo usa `esperado` (leído ANTES de adoptar), así que la discrepancia se
    # avisa igual esta corrida; las siguientes quedan en silencio.
    aviso_picks = None
    if not dry_run:
        try:
            aviso_picks = adoptar_picks_cerrados(cargados, mis_numeros, eventos, now)
        except Exception as e:
            log.warning("adopción de picks cerrados falló (%s)", e)

    discrepancias = (diff_picks(eventos, esperado, cargados, now)
                     + diff_especiales(especiales_esperados(mis_numeros), cargados))

    # penca gratuita: 1 participación contra la columna 1 (no rompe el audit de la paga)
    try:
        discrepancias += auditar_gratuita(eventos, cfg, now)
    except CampeonatoNoIniciado:
        raise
    except Exception as e:
        log.warning("auditoría de la gratuita falló (%s)", e)

    avisadas = load_state(now)
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
    if aviso_picks:
        partes.append(aviso_picks)
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

"""Pipeline de picks para el flujo MANUAL de la Penca Supermatch.

Decisión operativa 2026-08-04: la carga en la web la hace el usuario a mano (menos
superficie de riesgo con la cuenta; son 8 partidos por semana). Este módulo hace
todo lo demás:

    1. Refresca fixture/resultados desde el penca-api público.
    2. Ajusta ratings con TODO el histórico disponible (temporadas previas + partidos
       ya jugados del Clausura).
    3. Trae odds de Supermatch ES para los partidos que ya tienen mercado y ajusta λ
       (blend 70% mercado / 30% ratings, mismo criterio que la Capa 2 del Mundial).
    4. Calibra el pool con la tasa de exactos del ranking público (si ya hay fechas
       jugadas).
    5. Optimiza el portfolio de la TEMPORADA completa con lo ya cargado congelado,
       y emite la planilla de la fecha objetivo.
    6. Versiona en data/predictions/clausura/ y (opcional) manda la planilla por
       Telegram lista para copiar en la web.

Uso:
    python -m src.clausura.picks --fecha 1                  # planilla de la Fecha 1
    python -m src.clausura.picks --fecha 1 --telegram       # y la manda al bot
    python -m src.clausura.picks --fecha 1 --participaciones 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.clausura.api import TZ_UY, PencaApiClient
from src.clausura.economics import MAX_GOALS, SimConfig, index_score, score_index
from src.clausura.historical import DATA_DIR, load_dataset
from src.clausura.market_grid import refine_grid
from src.clausura.odds import EventOdds, _norm, fetch_primera_odds
from src.clausura.pool import PoolConfig, calibrate_from_exact_rate, observed_exact_rate_from_ranking
from src.clausura.especiales import (
    fetch_opciones,
    goleador_prior_from_ratings,
    p_campeon_from_grids,
    pool_campeon_distribution,
    pool_goleador_distribution,
)
from src.clausura.ratings import TeamRatings, fit_ratings
from src.clausura.scoring import expected_points_grid
from src.clausura.strategy import EspecialesInput, PortfolioClausura, build_portfolio
from src.model.market_probs import devig
from src.model.poisson import MarketConstraints, fit_params, score_grid
from src.utils.versions import latest_version

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "clausura2026.yaml"
PRED_DIR = ROOT / "data" / "predictions" / "clausura"

MARKET_WEIGHT = 0.7   # blend λ: 70% mercado / 30% ratings (criterio Capa 1+2 del Mundial)

# Cuánto se encoge hacia uniforme el consenso del pool para estimar P(goleador), cuando
# el menú se reconstruye del snapshot. 0 = creerle al pool tal cual, 1 = uniforme.
# NO está calibrado — no hay datos a nivel jugador. Está acá arriba y expuesto porque
# lo que corresponde es medir cuánto dependen las decisiones de él, no afinarlo a ojo.
# Medido el 2026-08-10: ver el comentario de `goleador_prior_desde_pool`.
GOLEADOR_SHRINK = float(os.environ.get("CLAUSURA_GOLEADOR_SHRINK", "0.5"))

# Reconstruir el menú de goleador del snapshot y meterlo al Monte Carlo. APAGADO por
# medición, no por falta de ganas: el 2026-08-10 se midió que prenderlo EMPEORA las
# decisiones, con los tres priors probados y por lo tanto no por culpa del shrink:
#     shrink 0.0 → −$609 ± 308 · shrink 0.5 → −$716 ± 324 · shrink 1.0 → −$199 ± 442
#
# Dos razones. (1) Nuestros 12 goleadores están CONGELADOS desde el lock y los 12 caen
# en el top-3, junto con el 84,6% del pool: los 25 puntos son un desplazamiento casi
# común y no mueven posiciones relativas. (2) Es un término grumoso —25 pts con 20-34%
# de probabilidad— que sube la varianza de cada estimación de E[premio] y degrada al
# ascenso por coordenadas; el mismo mecanismo que hace ganar al menú chico.
#
# El "+$48k" del sweep del 7/8 era sobre ELEGIR goleadores y comprar participaciones
# ANTES del lock. Eso ya no está disponible; son dos cosas distintas.
#
# Vale la pena prenderlo si algún día aparece un prior REAL (cuotas de mercado de
# goleador), o el año que viene antes del lock, cuando la elección sí esté abierta.
GOLEADOR_EN_MC = os.environ.get("CLAUSURA_GOLEADOR_MC", "0") == "1"


# -------------------- carga de insumos --------------------

def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def flat_eventos(cfg: dict) -> list[dict]:
    """Todos los eventos de la temporada, ordenados por inicio, con nº de fecha."""
    out = []
    for nombre, f in cfg["fechas"].items():
        n = int(nombre.split()[-1])
        for ev in f["eventos"]:
            out.append({**ev, "fecha_n": n, "fecha_id": f["fecha_id"]})
    out.sort(key=lambda e: e["inicio_utc"])
    return out


def ensure_ratings() -> TeamRatings:
    """Ratings con todo lo jugado: 5 temporadas previas + Intermedio 2026 + Clausura.

    El Intermedio no está en el penca-api; se ingiere aparte desde Wikipedia
    (src.clausura.intermedio) y load_dataset_completo lo suma si existe.
    """
    path = DATA_DIR / "primera_uy_historico.json"
    if not path.exists():
        log.info("histórico ausente — descargando del penca-api…")
        from src.clausura.historical import main as hist_main
        import sys
        argv, sys.argv = sys.argv, ["historical"]
        try:
            hist_main()
        finally:
            sys.argv = argv
    from src.clausura.intermedio import load_dataset_completo
    return fit_ratings(load_dataset_completo())


_STOP_TOKENS = {"de", "del", "la", "las", "los", "el", "club", "fc", "uru"}


def _team_tokens(name: str) -> frozenset[str]:
    """Tokens significativos del nombre: sin puntuación ("M.C."), paréntesis
    ("Liverpool (URU)"), iniciales sueltas ni conectores ("Juventud de Las Piedras")."""
    clean = re.sub(r"[^a-z0-9 ]", " ", _norm(name))
    return frozenset(t for t in clean.split() if len(t) > 1 and t not in _STOP_TOKENS)


def match_odds(eventos: list[dict], odds: list[EventOdds]) -> dict[int, EventOdds]:
    """evento_id → EventOdds, matcheando por nombres normalizados de ambos equipos.

    Los nombres del penca-api y del Elasticsearch difieren en detalles
    ("Montevideo City Torque" vs "M.C. Torque", "Liverpool" vs "Liverpool (URU)",
    "Juventud" vs "Juventud de Las Piedras"): además del substring, matcheamos por
    tokens significativos, aceptando abreviaturas por prefijo ("Sp." → "Sporting").
    El guardia real contra cruces (Cerro vs Cerro Largo) es exigir que matcheen
    los DOS equipos del partido.
    """
    def covers(small: frozenset[str], big: frozenset[str]) -> bool:
        return all(any(bt.startswith(st) for bt in big) for st in small)

    def similar(a: str, b: str) -> bool:
        na, nb = _norm(a), _norm(b)
        if na == nb or na in nb or nb in na:
            return True
        ta, tb = _team_tokens(a), _team_tokens(b)
        return bool(ta) and bool(tb) and (covers(ta, tb) or covers(tb, ta))

    out = {}
    for ev in eventos:
        for o in odds:
            if similar(ev["local"], o.home) and similar(ev["visitante"], o.away):
                out[ev["evento_id"]] = o
                break
    sin_match = [o for o in odds if o.event_id not in {m.event_id for m in out.values()}]
    if sin_match:
        log.info("odds sin evento matcheado (posible drift de nombres): %s",
                 ["%s vs %s" % (o.home, o.away) for o in sin_match])
    return out


# -------------------- construcción de grillas --------------------

def delta_grid(gl: int, gv: int) -> np.ndarray:
    """Partido ya jugado: toda la masa en el resultado real."""
    g = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    g[min(gl, MAX_GOALS), min(gv, MAX_GOALS)] = 1.0
    return g


def market_lambdas(o: EventOdds) -> tuple[float, float] | None:
    """λ del mercado: fit de Poisson contra 1X2 (+ over 2.5 si está)."""
    if not o.x1x2:
        return None
    p = devig(o.x1x2, "proportional")
    o25 = devig(o.totals["2.5"], "proportional").get("over") if "2.5" in o.totals else None
    c = MarketConstraints(p_home_win=p["home"], p_draw=p["draw"], p_away_win=p["away"],
                          p_over_2_5=o25)
    lam_l, lam_v, _ = fit_params(c)
    return lam_l, lam_v


def mercados_ricos_activos() -> bool:
    """CLAUSURA_MERCADOS_RICOS=1 activa el refinado con marcador exacto/goles.

    Default APAGADO: el backtest de recuperación (scripts/backtest_market_grid.py,
    2026-08-05) mostró que si la liga es Poisson dado el λ del partido — y ε≈0-0.1
    dice que lo es — el refinado solo agrega ruido de vig. El valor real, si existe,
    es información del mercado que el 1X2 no revela, y eso se mide en modo sombra:
    ambas grillas se versionan en el payload y el postmortem las compara contra los
    resultados reales. Prender el flag recién si la sombra gana.
    """
    return os.environ.get("CLAUSURA_MERCADOS_RICOS", "0") == "1"


def build_season_grids(
    eventos: list[dict],
    ratings: TeamRatings,
    odds_by_evento: dict[int, EventOdds],
    resultados: dict[int, tuple[int, int]],
) -> tuple[list[np.ndarray], list[str], list[np.ndarray], list[dict | None]]:
    """(grids, fuentes, pred_grids, shadow) por evento.

    `grids` es la grilla de LIQUIDACIÓN: delta en los partidos ya jugados (toda la
    masa en el resultado real), predictiva en los futuros. `pred_grids` es la grilla
    PREDICTIVA pre-partido para todos los eventos: es la única válida para calibrar
    el pool, construir Q y ajustar γ — sobre una delta, cualquier pencista "acierta"
    el exacto con probabilidad 1 y la calibración degenera.

    `shadow[i]` guarda ambas grillas (base Poisson y refinada con mercados ricos)
    cuando hay mercados ricos, para la comparación del postmortem contra resultados
    reales. Cuál de las dos va a `pred_grids` lo decide mercados_ricos_activos().
    """
    usar_ricos = mercados_ricos_activos()
    grids, fuentes, pred_grids, shadow = [], [], [], []
    for ev in eventos:
        eid = ev["evento_id"]
        lam_rt = ratings.lambdas(ev["local"], ev["visitante"])
        o = odds_by_evento.get(eid)
        lam_mkt = market_lambdas(o) if o else None
        if lam_mkt:
            lam_l = MARKET_WEIGHT * lam_mkt[0] + (1 - MARKET_WEIGHT) * lam_rt[0]
            lam_v = MARKET_WEIGHT * lam_mkt[1] + (1 - MARKET_WEIGHT) * lam_rt[1]
            fuente = "mercado+ratings"
        else:
            lam_l, lam_v = lam_rt
            fuente = "ratings"
        base = score_grid(lam_l, lam_v, 0.0, max_goals=MAX_GOALS)
        rica, ricos = refine_grid(base, o)

        if ricos:
            shadow.append({
                "mercados": ricos,
                "base": [round(float(x), 6) for x in base.ravel()],
                "rica": [round(float(x), 6) for x in rica.ravel()],
            })
        else:
            shadow.append(None)

        if usar_ricos and ricos:
            pred = rica
            fuente += "+" + "+".join(ricos)
        else:
            pred = base
        pred_grids.append(pred)

        if eid in resultados:
            gl, gv = resultados[eid]
            grids.append(delta_grid(gl, gv))
            fuentes.append(f"final {gl}-{gv}")
        else:
            grids.append(pred)
            fuentes.append(fuente)
    return grids, fuentes, pred_grids, shadow


# -------------------- estado congelado (picks ya cargados) --------------------

def fecha_dir(n: int) -> Path:
    return PRED_DIR / f"fecha_{n:02d}"


def load_frozen(
    eventos: list[dict],
    target_fecha: int,
    n_participaciones: int,
    cerrados: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(frozen_picks, frozen_mask) desde los archivos versionados.

    Un partido queda congelado si pertenece a una fecha anterior a la objetivo Y hay
    picks guardados para él. Si falta el archivo de una fecha pasada avisamos: eso
    significa que lo cargado en la web no está registrado acá.

    `cerrados`: evento_ids de la PROPIA fecha objetivo cuyo pick ya no se puede
    cambiar (cierre pasado o partido jugado). Sin esto, una regeneración intra-fecha
    (p.ej. la corrida del sábado con el partido del viernes ya jugado) pisaría esos
    picks con 0-0 y rompería drift audit, postmortem y el estado del simulador.
    """
    cerrados = cerrados or set()
    n = len(eventos)
    frozen = np.zeros((n_participaciones, n), dtype=np.int64)
    mask = np.zeros(n, dtype=bool)
    idx_by_evento = {ev["evento_id"]: i for i, ev in enumerate(eventos)}

    for f in range(1, target_fecha + 1):
        d = fecha_dir(f)
        latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
        if latest is None:
            if f < target_fecha:
                log.warning("fecha %d sin picks guardados — sus partidos quedan libres "
                            "(si ya los cargaste en la web, hay drift)", f)
            continue
        data = json.loads(latest.read_text(encoding="utf-8"))
        for row in data["picks"]:
            i = idx_by_evento.get(row["evento_id"])
            if i is None:
                continue
            if f == target_fecha and row["evento_id"] not in cerrados:
                continue   # la fecha objetivo solo congela sus partidos ya cerrados
            if len(row["scores"]) < n_participaciones:
                log.warning("fecha %d: el archivo tiene %d participaciones y se piden "
                            "%d — las filas faltantes quedan congeladas en 0-0",
                            f, len(row["scores"]), n_participaciones)
            for k, (gl, gv) in enumerate(row["scores"][:n_participaciones]):
                frozen[k, i] = score_index(gl, gv)
            mask[i] = True
    return frozen, mask


def load_frozen_especiales(
    target_fecha: int,
    n_participaciones: int,
    opciones_goleador: list | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Campeón/Goleador ya guardados en cualquier corrida anterior (-1 = libre).

    Los especiales se eligen una vez (idealmente antes de la Fecha 1); si un archivo
    previo los tiene, se asumen cargados en la web y quedan fijos.

    Campeón y goleador se buscan POR SEPARADO, cada uno en la última planilla que
    lo tenga asignado: una corrida sin menú de goleadores (API 500) escribe
    goleador_idx=-1, y si eso pisara el freeze, la próxima corrida CON menú
    reasignaría goleadores que ya están cargados en la web (pasó con la v8 del
    5/8, que tiró los de la v7).

    Una planilla puede traer el goleador solo por NOMBRE con goleador_idx=-1
    (asignado a mano o arrastrado mientras el menú estaba en 500, caso v14 del
    7/8): con `opciones_goleador` disponible, ese nombre se resuelve contra el
    menú y congela igual — el nombre cargado en la web es la verdad, no el idx.
    """
    idx_por_nombre = ({o.nombre: i for i, o in enumerate(opciones_goleador)}
                      if opciones_goleador else {})
    idx_por_nombre_norm = {_norm(n): i for n, i in idx_por_nombre.items()}

    def _valor_goleador(row: dict) -> int:
        nombre = row.get("goleador")
        if nombre and idx_por_nombre:
            i = idx_por_nombre.get(nombre)
            if i is None:
                i = idx_por_nombre_norm.get(_norm(nombre))
            if i is not None:
                return i
            log.warning("goleador %r ya cargado en la web no matchea el menú de "
                        "Supermatch — queda LIBRE y el optimizador puede cambiarlo: "
                        "verificar a mano contra la web", nombre)
        return int(row.get("goleador_idx", -1))

    def _extraer(campo: str, valor=None) -> np.ndarray | None:
        valor = valor or (lambda row: int(row.get(campo, -1)))
        for rows in _especiales_hacia_atras(target_fecha):
            vals = [valor(row) for row in rows]
            if any(v >= 0 for v in vals):
                out = np.full(n_participaciones, -1, dtype=np.int64)
                for i, v in enumerate(vals[:n_participaciones]):
                    out[i] = v
                return out
        return None

    return _extraer("campeon_idx"), _extraer("goleador_idx", _valor_goleador)


def _payloads_hacia_atras(target_fecha: int):
    """Planillas guardadas, de la más nueva a la más vieja — versiones incluidas."""
    from src.utils.versions import version_num
    for f in range(target_fecha, 0, -1):
        d = fecha_dir(f)
        if not d.exists():
            continue
        for path in sorted(d.glob("v*_*.json"), key=version_num, reverse=True):
            yield json.loads(path.read_text(encoding="utf-8"))


def _especiales_hacia_atras(target_fecha: int):
    """Filas de especiales de TODAS las planillas guardadas, de la más nueva a la
    más vieja — versiones dentro de cada fecha incluidas (el caso real es v8 sin
    goleadores pisando a v7 que sí los tenía, misma fecha)."""
    for data in _payloads_hacia_atras(target_fecha):
        rows = (data.get("especiales", {}) or {}).get("por_participacion", [])
        if rows:
            yield rows


def load_warm_start(
    eventos: list[dict],
    target_fecha: int,
    n_participaciones: int,
) -> np.ndarray | None:
    """Matriz (n_part, n_partidos) desde la planilla anterior; -1 = sin dato.

    Es el punto de partida del ascenso por coordenadas (ver `build_portfolio`):
    arrancar de la planilla de ayer en vez del ancla de EV vale +$3.622 ± 699 de
    E[premio] y —lo que más importa en el día a día— baja el churn de picks de 49%
    a 22%, o sea corta a menos de la mitad las recargas manuales que hoy se piden
    por el salto entre óptimos locales equivalentes y no por información nueva.

    Se prefiere `picks_temporada` (la matriz de los 120 partidos, que se guarda
    desde el 2026-08-08). Las planillas viejas solo tienen los 8 partidos de su
    fecha: se componen hacia atrás, y lo que falte queda en -1 y cae al ancla.

    Devuelve None si no hay ninguna planilla previa (arranque en frío).
    """
    idx_of = {ev["evento_id"]: i for i, ev in enumerate(eventos)}
    warm = np.full((n_participaciones, len(eventos)), -1, dtype=np.int64)
    visto: set[int] = set()

    def _cargar(filas) -> None:
        for row in filas:
            eid = int(row["evento_id"])
            col = idx_of.get(eid)
            if col is None or eid in visto:
                continue          # la planilla más nueva manda
            scores = row.get("scores") or []
            if len(scores) < n_participaciones:
                continue          # planilla de otro tamaño: no se puede heredar
            visto.add(eid)
            for k, (gl, gv) in enumerate(scores[:n_participaciones]):
                warm[k, col] = score_index(int(gl), int(gv))

    for data in _payloads_hacia_atras(target_fecha):
        _cargar(data.get("picks_temporada") or [])
        _cargar(data.get("picks") or [])

    if not visto:
        log.info("sin planilla previa utilizable — el ascenso arranca en el ancla EV")
        return None
    return warm


def goleadores_previos(target_fecha: int, n_participaciones: int) -> list[dict] | None:
    """[{goleador_idx, goleador}] de la última planilla que los tenga, o None.

    Es el arrastre para cuando el menú de la API sigue en 500: los goleadores ya
    cargados en la web son la verdad y tienen que sobrevivir en cada planilla
    nueva (dashboard, drift audit y freeze dependen de eso).
    """
    for rows in _especiales_hacia_atras(target_fecha):
        if any(row.get("goleador") for row in rows):
            return [{"goleador_idx": row.get("goleador_idx", -1),
                     "goleador": row.get("goleador")}
                    for row in rows[:n_participaciones]]
    return None


def format_especiales(port: PortfolioClausura, equipo_nombres: list[str],
                      opciones_goleador, gol_previos: list[dict] | None = None) -> str:
    """Sección de la planilla con Campeón/Goleador por participación.

    `gol_previos` es el arrastre cuando el menú de la API sigue en 500 pero los
    goleadores ya fueron elegidos y cargados en la web (planillas anteriores).
    """
    if port.campeon is None:
        return ""
    lines = ["<b>⭐ Especiales (25 pts c/u — pestaña Especial)</b>"]
    if port.p_campeon is not None:
        top = np.argsort(-port.p_campeon)[:4]
        probs = " · ".join(f"{equipo_nombres[t]} {port.p_campeon[t]:.0%}" for t in top)
        lines.append(f"<i>P(campeón): {probs}</i>")
    for i in range(len(port.campeon)):
        campeon = equipo_nombres[int(port.campeon[i])]
        gol = ""
        if port.goleador is not None and opciones_goleador:
            gol = f" · goleador: {opciones_goleador[int(port.goleador[i])].nombre}"
        elif gol_previos and i < len(gol_previos) and gol_previos[i].get("goleador"):
            gol = f" · goleador: {gol_previos[i]['goleador']}"
        lines.append(f"  {i + 1}: campeón <b>{campeon}</b>{gol}")
    if port.goleador is None and not gol_previos:
        lines.append("  <i>goleador: menú aún no publicado por Supermatch — "
                     "recomiendo cargarlo apenas aparezca</i>")
    return "\n".join(lines)


def format_gratuita(
    eventos_fecha: list[dict],
    port: PortfolioClausura,
    idx_of: dict[int, int],
    equipo_nombres: list[str],
    opciones_goleador,
    gol_previos: list[dict] | None = None,
) -> str:
    """Sección lista para copiar en la penca GRATUITA (1 sola participación).

    El premio de la gratuita es una PlayStation 5 — INDIVISIBLE, con desempate por
    exactos y después sorteo. A diferencia de la paga (plata repartida entre
    empatados), empatar no diluye: da derecho al desempate. Sin la penalidad por
    converger, y con una sola planilla, el óptimo es EV puro — que es exactamente
    la participación 1 (el ancla que el optimizador nunca perturba).

    Verificado por simulación 2026-08-05 (245 rivales, evaluación out-of-sample):
    EV duplica a chalk en P(ganar), y optimizar P(ganar) directamente EMPEORA
    fuera de muestra (winner's curse).

    Salvedad: los especiales SÍ se diversifican en todas las participaciones,
    incluida la 1 — así que acá va el argmax de P(campeón), no el de la fila 1.
    """
    lines = ["", "<b>🎁 Penca GRATUITA (1 participación — PlayStation 5)</b>",
             "<i>Óptimo = EV puro: premio indivisible con desempate por exactos, "
             "diferenciarse no paga.</i>"]
    for ev in eventos_fecha:
        i = idx_of[ev["evento_id"]]
        gl, gv = index_score(int(port.picks[0, i]))
        estrella = " ⭐x2" if ev["preferencial"] else ""
        lines.append(f"  {ev['local']} vs {ev['visitante']}{estrella}: <b>{gl}-{gv}</b>")
    if port.p_campeon is not None:
        mejor = int(np.argmax(port.p_campeon))
        lines.append(f"  campeón: <b>{equipo_nombres[mejor]}</b> "
                     f"({port.p_campeon[mejor]:.0%} — el más probable, "
                     f"NO el de la participación 1)")
    if port.goleador is not None and opciones_goleador:
        lines.append(f"  goleador: <b>{opciones_goleador[int(port.goleador[0])].nombre}</b>")
    elif gol_previos and gol_previos[0].get("goleador"):
        lines.append(f"  goleador: <b>{gol_previos[0]['goleador']}</b>")
    return "\n".join(lines)


def save_version(target_fecha: int, payload: dict) -> Path:
    """Escribe v<N>_<ts>.json sin sobreescribir (regla de trabajo #2)."""
    d = fecha_dir(target_fecha)
    d.mkdir(parents=True, exist_ok=True)
    prev = latest_version(d.glob("v*_*.json"))
    n = 1
    if prev is not None:
        from src.utils.versions import version_num
        n = version_num(prev) + 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = d / f"v{n}_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# -------------------- planilla --------------------

def format_planilla(
    eventos_fecha: list[dict],
    portfolio: PortfolioClausura,
    idx_of: dict[int, int],
    fuentes: list[str],
    n_rivales: int,
) -> str:
    """Planilla en texto plano/HTML-Telegram, ordenada como la web (por kickoff)."""
    r = portfolio.resultado
    lines = []
    lines.append(f"<b>🏆 Penca Clausura — Fecha {eventos_fecha[0]['fecha_n']}</b>")
    lines.append(f"E[premio] portfolio: ${r.e_premio_total:,.0f} · "
                 f"P(premio grande): {r.p_gana_penca:.0%} · pool: {n_rivales}")
    lines.append("")
    for ev in eventos_fecha:
        i = idx_of[ev["evento_id"]]
        estrella = " ⭐x2" if ev["preferencial"] else ""
        cierre = datetime.fromisoformat(ev["cierre_pronostico_utc"]).astimezone(TZ_UY)
        cierre_uy = cierre.strftime("%d/%m %H:%M")
        scores = [index_score(int(portfolio.picks[k, i]))
                  for k in range(portfolio.picks.shape[0])]
        marcadores = " | ".join(f"{a}-{b}" for a, b in scores)
        lines.append(f"<b>{ev['local']} vs {ev['visitante']}</b>{estrella}")
        lines.append(f"  {marcadores}")
        lines.append(f"  <i>cierra {cierre_uy} UY · {fuentes[i]}</i>")
    lines.append("")
    lines.append("Orden de participaciones: 1=EV puro, 2..N=perturbaciones óptimas.")
    lines.append("Cargar en supermatch.com.uy → Pencas → Torneo Clausura 2026.")
    return "\n".join(lines)


# -------------------- main --------------------

def run(
    target_fecha: int,
    n_participaciones: int = 5,
    telegram: bool = False,
    n_sims: int = 800,
    liberar_especiales: bool = False,
    contexto: dict | None = None,
) -> Path:
    """Corre el pipeline de una fecha y versiona la planilla.

    `contexto` es un dict de salida opcional: si se pasa, se le cargan el portfolio,
    el evaluador (para re-liquidar otras matrices de picks con los mismos sorteos) y
    el mapa evento→columna. Lo usa src.clausura.rerun_cierre para decidir si el
    cambio de planilla vale la pena en PLATA antes de pedir una recarga manual.
    """
    cfg = load_config()
    eventos = flat_eventos(cfg)
    idx_of = {ev["evento_id"]: i for i, ev in enumerate(eventos)}

    # resultados ya conocidos + tamaño real del pool + tasa de exactos.
    # El ranking incluye NUESTRAS participaciones: para el tamaño del pool rival y
    # los observables de calibración hay que excluirlas (si no, simulamos contra
    # nosotros mismos y nos dividimos el premio).
    from src.clausura.rivals import mis_numeros_env
    mis_numeros = mis_numeros_env()
    resultados: dict[int, tuple[int, int]] = {}
    n_rivales, exact_rate = 151, None
    # numero de participación → puntos del ranking AHORA. El modelo de rivales los
    # prefiere a los del snapshot, que es una foto y puede tener horas (ver
    # rivals.build_rival_model).
    puntos_vivos: dict[int, int] = {}
    penca_id = cfg["pencas"]["paga"]["id"]
    try:
        with PencaApiClient() as api:
            # resultados de partidos finalizados
            for nombre, f in cfg["fechas"].items():
                data = api._get(f"/front/campeonatos/fechas/{f['fecha_id']}/eventos")
                for e in data:
                    res = e.get("resultado") or {}
                    gl, gv = res.get("golesEquipoLocal"), res.get("golesEquipoVisitante")
                    if gl is not None and gv is not None:
                        resultados[e["id"]] = (int(gl), int(gv))
            ranking = api.ranking(penca_id)
            propias = sum(r.numero_participacion in mis_numeros for r in ranking)
            n_rivales = max(len(ranking) - propias, 1)
            puntos_vivos = {r.numero_participacion: r.puntos_totales for r in ranking}
            jugados = len(resultados)
            exact_rate = observed_exact_rate_from_ranking(ranking, jugados, mis_numeros)
    except Exception as e:  # red caída: seguimos con defaults
        log.warning("penca-api no disponible (%s) — sigo con defaults", e)

    ratings = ensure_ratings()

    try:
        odds = fetch_primera_odds()
    except Exception as e:
        log.warning("odds de Supermatch no disponibles (%s)", e)
        odds = []
    odds_by_evento = match_odds(eventos, odds)

    grids, fuentes, pred_grids, shadow = build_season_grids(
        eventos, ratings, odds_by_evento, resultados)
    log.info("mercados ricos: %s (%d eventos con sombra)",
             "ACTIVOS" if mercados_ricos_activos() else "sombra",
             sum(s is not None for s in shadow))

    # calibración del pool: SIEMPRE con las grillas predictivas de los jugados —
    # sobre la delta el exacto esperado es ≈1 para cualquier temperatura y el
    # calibrador elige un extremo arbitrario del grid.
    pool_cfg = PoolConfig()
    if exact_rate is not None:
        jugables = [g for g, ev in zip(pred_grids, eventos)
                    if ev["evento_id"] in resultados]
        if jugables:
            previo = pool_cfg
            pool_cfg = calibrate_from_exact_rate(jugables, exact_rate, pool_cfg)
            # La concentración efectiva del pool es chalk/temperature: el calibrador
            # mueve SOLO la temperatura, así que subirla deshace el chalk_strength que
            # se midió contra 4.791 picks reales (+$8.531 ± 10 a 19.200). Y los dos
            # canales no valen lo mismo: éste ajusta un escalar (la tasa de exactos del
            # ranking, que además ya nos mintió una vez dando T=3.0 con el pool real en
            # ~0.5), contra la distribución completa de picks del otro. Si el escalar
            # nos lleva a menos de la mitad de la concentración medida, hacemos ruido.
            efectiva = pool_cfg.chalk_strength / max(pool_cfg.temperature, 1e-6)
            objetivo = previo.chalk_strength / max(previo.temperature, 1e-6)
            log.info("pool calibrado: temperatura=%.2f (exact_rate=%.3f) — "
                     "concentración efectiva chalk/T = %.2f",
                     pool_cfg.temperature, exact_rate, efectiva)
            if efectiva < 0.5 * objetivo:
                log.warning(
                    "la calibración por tasa de exactos bajó la concentración efectiva "
                    "de %.2f a %.2f (temperatura %.2f). Eso deshace la calibración "
                    "contra picks reales, que vale +$8.531 de E[premio]. Revisar si el "
                    "exact_rate observado (%.3f) es confiable antes de creerle: se "
                    "llena recién al CIERRE de la fecha.",
                    objetivo, efectiva, pool_cfg.temperature, exact_rate)

    # partidos de la fecha objetivo que ya no se pueden cambiar (jugados o cerrados):
    # se congelan con SUS picks guardados, no se regeneran
    now = datetime.now(timezone.utc)
    cerrados = {
        ev["evento_id"] for ev in eventos
        if ev["fecha_n"] == target_fecha
        and (ev["evento_id"] in resultados
             or datetime.fromisoformat(ev["cierre_pronostico_utc"]) <= now)
    }
    frozen, mask = load_frozen(eventos, target_fecha, n_participaciones, cerrados)
    # los partidos ya jugados también quedan fijos (su pick ya no se puede cambiar)
    for i, ev in enumerate(eventos):
        if ev["evento_id"] in resultados and not mask[i]:
            mask[i] = True
            log.warning("partido jugado sin picks guardados (%s vs %s) — queda como "
                        "0-0 en el simulador; los puntos propios de esa fecha van a "
                        "estar mal hasta registrar lo cargado",
                        ev["local"], ev["visitante"])

    # ---------- pool: prior + Q empírica del snapshot (si el campeonato ya inició) ----------
    # Q y γ también van sobre las grillas predictivas: la Q de un partido jugado es
    # "qué jugó el pool ANTES del partido", no una delta con el resultado.
    from src.clausura.pool import pool_distribution
    from src.clausura.pool_snapshot import (
        blended_q, empirical_campeon_counts, empirical_counts,
        empirical_goleador_counts, load_latest_snapshot,
    )
    pool_qs = [pool_distribution(g, pool_cfg) for g in pred_grids]
    snapshot = load_latest_snapshot(max_age_hours=48)
    campeon_counts = None
    if snapshot:
        counts = empirical_counts(snapshot, mis_numeros)
        observados = 0
        for i, ev in enumerate(eventos):
            c = counts.get(ev["evento_id"])
            if c is not None and c.sum() > 0:
                pool_qs[i] = blended_q(pool_qs[i], c)
                observados += 1
        log.info("pool empírico: snapshot de %s participaciones, %d/%d eventos observados",
                 snapshot.get("n_participaciones"), observados, len(eventos))
        campeon_counts = snapshot  # se resuelve más abajo, con el índice de equipos armado

    # ---------- especiales: Campeón (siempre modelable) + Goleador (si hay menú) ----------
    equipos_cfg: dict[int, str] = cfg["equipos"]
    equipo_nombres = [equipos_cfg[k] for k in sorted(equipos_cfg)]
    equipo_idx = {nombre: i for i, nombre in enumerate(equipo_nombres)}
    local_de = np.array([equipo_idx[ev["local"]] for ev in eventos])
    visita_de = np.array([equipo_idx[ev["visitante"]] for ev in eventos])

    p_champ_prior = p_campeon_from_grids(grids, local_de, visita_de, len(equipo_nombres))
    # sesgo asimétrico del pool hacia equipos puntuales (señal cualitativa pre-lock):
    #   CLAUSURA_POOL_CAMPEON_LEAN="Nacional:1.6,Peñarol:0.9"
    lean = None
    lean_raw = os.environ.get("CLAUSURA_POOL_CAMPEON_LEAN", "")
    if lean_raw:
        try:
            lean = {k.strip(): float(v) for k, v in
                    (par.split(":") for par in lean_raw.split(","))}
            log.info("pool campeón con lean: %s", lean)
        except ValueError:
            log.warning("CLAUSURA_POOL_CAMPEON_LEAN inválido (%r) — ignorado", lean_raw)
    pool_q_campeon = pool_campeon_distribution(p_champ_prior, equipo_nombres, lean=lean)
    if campeon_counts is not None:
        c = empirical_campeon_counts(campeon_counts, equipo_idx, len(equipo_nombres),
                                     mis_numeros)
        pool_q_campeon = blended_q(pool_q_campeon, c)

    # El menú se pide ANTES del modelo de rivales: sin él no se puede indexar el
    # goleador observado de cada rival (el snapshot guarda el id de la opción).
    opciones_campeon, opciones_goleador = None, None
    try:
        opciones_campeon, opciones_goleador = fetch_opciones(penca_id)
    except Exception as e:
        log.warning("opciones de especiales no disponibles (%s)", e)

    # Sin menú del API (500 crónico en esta penca) el goleador quedaba FUERA del Monte
    # Carlo: 25 de los 50 puntos de especiales no existían, ni los nuestros ni los de
    # los 474 rivales que sí lo cargaron, y `sin_goleador` tampoco se marcaba. El menú
    # se reconstruye del snapshot, que trae goleador_id + nombre de cada participación.
    desde_snapshot = False
    if GOLEADOR_EN_MC and not opciones_goleador and snapshot:
        from src.clausura.especiales import (
            goleador_prior_desde_pool, opciones_goleador_desde_snapshot,
        )
        opciones_goleador = opciones_goleador_desde_snapshot(snapshot)
        desde_snapshot = bool(opciones_goleador)
        if desde_snapshot:
            log.info("menú de goleador reconstruido del snapshot: %d opciones "
                     "(el API sigue en 500)", len(opciones_goleador))

    p_gol, pool_q_gol = None, None
    if opciones_goleador:
        if desde_snapshot:
            # El snapshot no trae equipo_id, así que el prior de ratings no aplica.
            # Se usa el consenso del pool encogido hacia uniforme — ver
            # `goleador_prior_desde_pool` para por qué no se usa Q tal cual.
            from src.clausura.especiales import goleador_prior_desde_pool
            g_counts = empirical_goleador_counts(snapshot, opciones_goleador, mis_numeros)
            p_gol = goleador_prior_desde_pool(g_counts, shrink=GOLEADOR_SHRINK)
        else:
            equipo_id_por_nombre = {v: k for k, v in equipos_cfg.items()}
            p_gol = goleador_prior_from_ratings(
                opciones_goleador, equipo_nombres, equipo_id_por_nombre, ratings.ataque
            )
        pool_q_gol = pool_goleador_distribution(p_gol)
        # los especiales de los rivales son públicos incluso ANTES del inicio del
        # campeonato (verificado 2026-08-07): si hay snapshot, la Q modelada del
        # goleador se corrige con lo realmente cargado, igual que la del campeón
        if snapshot:
            g_counts = empirical_goleador_counts(snapshot, opciones_goleador, mis_numeros)
            if g_counts.sum() > 0:
                pool_q_gol = blended_q(pool_q_gol, g_counts)
                log.info("pool goleador empírico: %d picks observados", int(g_counts.sum()))
    else:
        log.info("goleador: Supermatch aún no publicó el menú de opciones (500) — "
                 "solo se recomienda Campeón por ahora")

    # ---------- rivales empíricos por participación (standings reales + estilo) ----------
    rival_model = None
    if snapshot:
        from src.clausura.rivals import build_rival_model
        goleador_op_idx = ({o.id: i for i, o in enumerate(opciones_goleador)}
                           if opciones_goleador else None)
        rival_model = build_rival_model(
            snapshot, eventos, pool_qs, resultados, equipo_idx=equipo_idx,
            goleador_op_idx=goleador_op_idx,
            puntos_vivos=puntos_vivos or None,
        )
        if rival_model is not None:
            n_rivales = rival_model.n_rivales

    if liberar_especiales:
        # Solo tiene sentido MIENTRAS los especiales se pueden editar en la web
        # (hasta 15 min antes del primer partido del campeonato, Art. 4). Después
        # del lock, correr con esto desalinearía la planilla de lo cargado.
        frozen_campeon, frozen_goleador = None, None
        log.warning("especiales LIBERADOS por flag: campeón/goleador se re-optimizan "
                    "desde cero — cargar a mano en la web lo que cambie, ANTES del lock")
    else:
        frozen_campeon, frozen_goleador = load_frozen_especiales(
            target_fecha, n_participaciones, opciones_goleador)
    gol_previos = (goleadores_previos(target_fecha, n_participaciones)
                   if not opciones_goleador else None)

    especiales = EspecialesInput(
        local_de=local_de,
        visita_de=visita_de,
        n_teams=len(equipo_nombres),
        pool_q_campeon=pool_q_campeon,
        p_goleador=p_gol,
        pool_q_goleador=pool_q_gol,
        frozen_campeon=frozen_campeon,
        frozen_goleador=frozen_goleador,
    )

    port = build_portfolio(
        grids=grids,
        fecha_de_partido=[ev["fecha_id"] for ev in eventos],
        preferencial=[ev["preferencial"] for ev in eventos],
        n_participaciones=n_participaciones,
        pool_cfg=pool_cfg,
        sim=SimConfig(n_sims=n_sims, n_rivales=n_rivales),
        frozen_picks=frozen,
        frozen_mask=mask,
        especiales=especiales,
        pool_qs=pool_qs,
        rivals=rival_model,
        warm_start=load_warm_start(eventos, target_fecha, n_participaciones),
    )

    eventos_fecha = [ev for ev in eventos if ev["fecha_n"] == target_fecha]
    planilla = format_planilla(eventos_fecha, port, idx_of, fuentes, n_rivales)
    planilla += "\n" + format_especiales(port, equipo_nombres, opciones_goleador,
                                         gol_previos)
    planilla += "\n" + format_gratuita(eventos_fecha, port, idx_of, equipo_nombres,
                                       opciones_goleador, gol_previos)

    payload = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "fecha": target_fecha,
        "n_participaciones": n_participaciones,
        # Queda registrado porque el diff entre dos planillas SOLO significa "se
        # movieron los insumos" si ambas se optimizaron con el mismo número de
        # sorteos: el ascenso por coordenadas cae en otro óptimo local si cambia
        # n_sims, aun con la misma semilla. src.clausura.rerun_cierre lo hereda.
        "n_sims": n_sims,
        "pool": {"n_rivales": n_rivales, "temperatura": pool_cfg.temperature,
                 "exact_rate_observada": exact_rate,
                 "rival_model": rival_model.resumen() if rival_model else None},
        "resultado_sim": {
            "e_premio_total": port.resultado.e_premio_total,
            "e_premio_penca": port.resultado.e_premio_penca,
            "e_premio_fechas": port.resultado.e_premio_fechas,
            "p_gana_penca": port.resultado.p_gana_penca,
        },
        "especiales": {
            "p_campeon": {equipo_nombres[i]: round(float(p), 4)
                          for i, p in enumerate(port.p_campeon)
                          if p > 0.001} if port.p_campeon is not None else None,
            "por_participacion": [
                {
                    "campeon_idx": int(port.campeon[i]),
                    "campeon": equipo_nombres[int(port.campeon[i])],
                    # sin menú (API 500) el optimizador no asigna goleador: se
                    # ARRASTRAN los de la última planilla que los tenga — los ya
                    # cargados en la web son la verdad y no pueden perderse
                    **({"goleador_idx": int(port.goleador[i]),
                        "goleador": (opciones_goleador[int(port.goleador[i])].nombre
                                     if opciones_goleador else None)}
                       if port.goleador is not None
                       else (gol_previos[i] if gol_previos and i < len(gol_previos)
                             else {"goleador_idx": -1, "goleador": None})),
                }
                for i in range(n_participaciones)
            ] if port.campeon is not None else [],
        },
        "picks": [
            {
                "evento_id": ev["evento_id"],
                "partido": f"{ev['local']} vs {ev['visitante']}",
                "preferencial": ev["preferencial"],
                "cierre_pronostico_utc": ev["cierre_pronostico_utc"],
                "fuente_modelo": fuentes[idx_of[ev["evento_id"]]],
                "scores": [list(index_score(int(port.picks[k, idx_of[ev["evento_id"]]])))
                           for k in range(n_participaciones)],
                # E[pts] de cada pick bajo la grilla del modelo al momento de generar
                # (insumo del postmortem: esperado vs real)
                "e_pts": [round(expected_points_grid(
                              index_score(int(port.picks[k, idx_of[ev["evento_id"]]])),
                              grids[idx_of[ev["evento_id"]]], ev["preferencial"]), 2)
                          for k in range(n_participaciones)],
                # sombra de mercados ricos: base Poisson vs refinada, para que el
                # postmortem compare log-loss contra el resultado real y decidamos
                # con datos si prender CLAUSURA_MERCADOS_RICOS
                "grilla_shadow": shadow[idx_of[ev["evento_id"]]],
            }
            for ev in eventos_fecha
        ],
        # Matriz de la TEMPORADA completa (no solo la fecha objetivo). No se carga en
        # la web —solo se cargan los 8 partidos de la fecha— pero es el warm start de
        # la próxima corrida: sin esto el ascenso arranca del ancla EV y cae en otro
        # óptimo local, reasignando la mitad de los picks sin que nada haya cambiado.
        "picks_temporada": [
            {
                "evento_id": ev["evento_id"],
                "scores": [list(index_score(int(port.picks[k, i])))
                           for k in range(n_participaciones)],
            }
            for i, ev in enumerate(eventos)
        ],
    }
    if contexto is not None:
        contexto.update(portfolio=port, evaluador=port.evaluador,
                        idx_of=idx_of, eventos=eventos)

    path = save_version(target_fecha, payload)

    print(planilla.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    print(f"\nguardado: {path}")

    if telegram:
        from src.notifier.telegram import TelegramConfig, TelegramNotifier
        TelegramNotifier(TelegramConfig.from_env()).send(planilla)
        print("enviado por Telegram ✓")

    return path


def resolve_fecha(arg: str) -> int:
    """'auto' → primera fecha con partidos pendientes (para el timer del VPS)."""
    if arg.lower() == "auto":
        from src.clausura.dashboard_loader import fecha_actual
        return fecha_actual(load_config())
    return int(arg)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", default="auto",
                    help="número de fecha (1-15) o 'auto' = próxima con partidos pendientes")
    ap.add_argument("--participaciones", type=int, default=None,
                    help="default: cantidad de CLAUSURA_MIS_PARTICIPACIONES (5 sin env)")
    ap.add_argument("--telegram", action="store_true", help="mandar la planilla al bot")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--liberar-especiales", action="store_true",
                    help="ignorar el freeze de campeón/goleador y re-optimizarlos "
                         "(solo mientras los especiales siguen editables en la web)")
    args = ap.parse_args()
    n_part = args.participaciones
    if n_part is None:
        from src.clausura.rivals import mis_numeros_env
        n_part = len(mis_numeros_env()) or 5
    run(resolve_fecha(args.fecha), n_part, args.telegram, args.sims,
        liberar_especiales=args.liberar_especiales)


if __name__ == "__main__":
    main()

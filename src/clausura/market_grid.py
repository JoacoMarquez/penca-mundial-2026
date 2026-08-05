"""Capa 1 rica: refinar la grilla de marcadores con los mercados finos de Supermatch.

El camino base (market_lambdas de picks.py) usa solo 1X2 + over 2.5 → Poisson. Pero
Supermatch cotiza para la Primera uruguaya exactamente lo que el scoring de la penca
paga (src/clausura/odds.py los parsea desde 2026-08-04):

    'Goles exactos' por equipo (0/1/2/3+)  → marginales de goles por lado, que son
                                             los payoffs INCONDICIONALES del kernel
    'Marcador exacto' (celdas + 'otro')    → la distribución conjunta directa,
                                             incluido el 0-0/1-1 que un Poisson con
                                             λ12=0 no puede replicar
    'Margen de victoria'                   → redundante con lo anterior; se usa solo
                                             como DIAGNÓSTICO (agregarlo como
                                             restricción dura junto a las marginales
                                             puede ser infactible: tres particiones
                                             solapadas de-vigueadas por separado no
                                             tienen por qué ser mutuamente consistentes)

Orden de refinamiento sobre la grilla base:

  1. **Raking (IPF) a las marginales de goles.** Los buckets {0,1,2,3+} de cada lado
     se escalan alternadamente hasta matchear el objetivo. El objetivo NO es el
     mercado puro sino el blend W_MERCADO·mercado + (1−W_MERCADO)·marginal de la
     grilla base — mismo criterio 70/30 que MARKET_WEIGHT en picks.py. IPF sobre
     filas/columnas converge siempre (grilla y objetivos estrictamente positivos)
     y preserva la estructura de dependencia de la base.

  2. **Overlay del marcador exacto.** De-vig Shin sobre todas las opciones (celdas +
     'otro'); las celdas cotizadas arman una grilla de mercado M donde la masa de
     'otro' se reparte proporcionalmente a la grilla actual sobre las celdas no
     cotizadas. Resultado: W_MERCADO·M + (1−W_MERCADO)·grilla. Va al final porque
     es la información más fina y debe dominar.

De-vig: Shin para el marcador exacto (mercado de colas con favoritos fuertes),
proportional para goles/margen — criterio del docstring de odds.py y books.yaml.

VEREDICTO DEL BACKTEST (scripts/backtest_market_grid.py, 2026-08-05): en el
experimento de recuperación, dado el λ correcto del partido la liga es ~Poisson
independiente (ε de mezcla empírica ≈ 0-0.1), y en ese mundo este refinado solo
agrega ruido de vig (Δlog-loss +0.004, picks cambian <6%). Su valor real, si
existe, es información match-específica del mercado que el 1X2 no revela — eso no
es simulable y se mide EN SOMBRA: picks.py versiona ambas grillas en el payload
(`grilla_shadow`) y el postmortem compara log-loss contra los resultados reales.
Por eso el flag CLAUSURA_MERCADOS_RICOS arranca APAGADO (regla de trabajo #1:
nada entra activo sin evidencia).
"""

from __future__ import annotations

import logging
import re

import numpy as np

from src.clausura.economics import MAX_GOALS
from src.clausura.odds import EventOdds
from src.model.market_probs import devig

log = logging.getLogger(__name__)

# Peso del mercado vs modelo en cada refinamiento (mismo criterio que MARKET_WEIGHT).
W_MERCADO = 0.7

# Buckets del mercado 'Goles exactos': 0 / 1 / 2 / 3+ (3+ agrega las filas 3..MAX_GOALS)
GOAL_BUCKETS = ("0", "1", "2", "3+")
MIN_SCORE_CELLS = 4          # celdas cotizadas mínimas para confiar en el overlay
IPF_MAX_ITERS = 200
IPF_TOL = 1e-10

_SCORE_KEY_RE = re.compile(r"^(\d+):(\d+)$")


# -------------------- marginales de goles (raking IPF) --------------------

def goals_marginal_target(odds: dict[str, float]) -> np.ndarray | None:
    """Mercado 'Goles exactos' de un equipo → probabilidades sobre {0,1,2,3+}.

    Devuelve None si el mercado no está completo (los 4 buckets exactos).
    """
    if not odds or set(odds) != set(GOAL_BUCKETS):
        return None
    p = devig(odds, "proportional")
    return np.array([p[b] for b in GOAL_BUCKETS])


def _bucket_marginal(grid: np.ndarray, axis: int) -> np.ndarray:
    """Marginal de la grilla agregada a los buckets {0,1,2,3+}."""
    marg = grid.sum(axis=1 - axis)
    return np.array([marg[0], marg[1], marg[2], marg[3:].sum()])


def rake_to_goal_marginals(
    grid: np.ndarray,
    target_home: np.ndarray,
    target_away: np.ndarray,
    w_mercado: float = W_MERCADO,
) -> np.ndarray:
    """IPF sobre buckets de filas/columnas hacia el blend mercado/base.

    Los objetivos se blendean UNA vez contra las marginales de la grilla de entrada
    (no se re-blendean por iteración): IPF clásico contra objetivos fijos.
    """
    tgt_h = w_mercado * target_home + (1 - w_mercado) * _bucket_marginal(grid, 0)
    tgt_a = w_mercado * target_away + (1 - w_mercado) * _bucket_marginal(grid, 1)
    tgt_h, tgt_a = tgt_h / tgt_h.sum(), tgt_a / tgt_a.sum()

    g = grid.copy()
    rows = [np.s_[0:1], np.s_[1:2], np.s_[2:3], np.s_[3:]]   # buckets 0/1/2/3+
    for _ in range(IPF_MAX_ITERS):
        cur_h = _bucket_marginal(g, 0)
        for sl, t, c in zip(rows, tgt_h, cur_h):
            g[sl, :] *= t / max(c, 1e-15)
        cur_a = _bucket_marginal(g, 1)
        for sl, t, c in zip(rows, tgt_a, cur_a):
            g[:, sl] *= t / max(c, 1e-15)
        if (np.abs(_bucket_marginal(g, 0) - tgt_h).max() < IPF_TOL
                and np.abs(_bucket_marginal(g, 1) - tgt_a).max() < IPF_TOL):
            break
    return g / g.sum()


# -------------------- overlay del marcador exacto --------------------

def exact_score_overlay(
    grid: np.ndarray,
    correct_score: dict[str, float],
    w_mercado: float = W_MERCADO,
) -> np.ndarray | None:
    """Blend celda a celda con el mercado 'Marcador exacto' de-vigueado (Shin).

    La masa de 'otro' (+ celdas cotizadas fuera de la grilla de trabajo) se reparte
    proporcionalmente a la grilla actual sobre las celdas NO cotizadas — el mercado
    solo opina sobre lo que cotiza. Devuelve None si hay pocas celdas cotizadas.
    """
    p = devig(correct_score, "shin")
    quoted: dict[tuple[int, int], float] = {}
    resto = 0.0
    for k, prob in p.items():
        m = _SCORE_KEY_RE.match(k)
        if m and int(m.group(1)) <= MAX_GOALS and int(m.group(2)) <= MAX_GOALS:
            quoted[(int(m.group(1)), int(m.group(2)))] = prob
        else:
            resto += prob   # 'otro' y goleadas fuera de la grilla
    if len(quoted) < MIN_SCORE_CELLS:
        return None

    market = np.zeros_like(grid)
    for (gl, gv), prob in quoted.items():
        market[gl, gv] = prob
    libre = market == 0
    masa_libre = float(grid[libre].sum())
    if masa_libre > 0:
        market[libre] = resto * grid[libre] / masa_libre

    out = w_mercado * market + (1 - w_mercado) * grid
    return out / out.sum()


# -------------------- margen de victoria (solo diagnóstico) --------------------

_MARGIN_KEY_RE = re.compile(r"^([HA])\+(\d+)$")


def margin_discrepancy(grid: np.ndarray, margin: dict[str, float]) -> float | None:
    """Máx |P_grilla − P_mercado| sobre los buckets de margen. None si no hay mercado.

    El bucket numérico más alto de cada lado se trata como 'N o más' (el parser de
    odds.py normaliza 'por 3+' → '+3').
    """
    if not margin or "D" not in margin:
        return None
    p = devig(margin, "proportional")
    diff = np.arange(MAX_GOALS + 1)[:, None] - np.arange(MAX_GOALS + 1)[None, :]

    tope = {"H": 0, "A": 0}
    for k in p:
        if (m := _MARGIN_KEY_RE.match(k)):
            tope[m.group(1)] = max(tope[m.group(1)], int(m.group(2)))

    worst = abs(float(grid[diff == 0].sum()) - p["D"])
    for k, prob in p.items():
        if not (m := _MARGIN_KEY_RE.match(k)):
            continue
        side, n = m.group(1), int(m.group(2))
        signed = n if side == "H" else -n
        if n == tope[side]:   # bucket abierto: N o más
            g = grid[diff >= signed].sum() if side == "H" else grid[diff <= signed].sum()
        else:
            g = grid[diff == signed].sum()
        worst = max(worst, abs(float(g) - prob))
    return worst


# -------------------- entrada única --------------------

def refine_grid(base: np.ndarray, o: EventOdds | None) -> tuple[np.ndarray, list[str]]:
    """Refina la grilla predictiva con los mercados ricos disponibles.

    Devuelve (grilla, mercados_usados). Sin mercados ricos, la grilla vuelve intacta
    (mismo objeto): el camino 1X2+over sigue siendo el piso.
    """
    if o is None:
        return base, []
    grid, usados = base, []

    tgt_h = goals_marginal_target(o.home_goals)
    tgt_a = goals_marginal_target(o.away_goals)
    if tgt_h is not None and tgt_a is not None:
        grid = rake_to_goal_marginals(grid, tgt_h, tgt_a)
        usados.append("goles")

    if o.correct_score:
        refined = exact_score_overlay(grid, o.correct_score)
        if refined is not None:
            grid = refined
            usados.append("exacto")

    if usados:
        disc = margin_discrepancy(grid, o.margin)
        if disc is not None and disc > 0.05:
            log.warning("%s vs %s: margen de victoria difiere de la grilla refinada "
                        "en %.3f — revisar consistencia de mercados", o.home, o.away, disc)
    return grid, usados

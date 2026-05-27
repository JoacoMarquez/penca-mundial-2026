"""Bivariate Poisson para distribución de marcadores.

Sea (X, Y) = (goles_local, goles_visit). Modelamos la pareja como bivariate Poisson:
    X = X1 + X12
    Y = X2 + X12
donde X1, X2, X12 son Poisson independientes con medias λ1, λ2, λ12.

Entonces:
    E[X]    = λ1 + λ12
    E[Y]    = λ2 + λ12
    Cov(X,Y)= λ12

Parametrizamos en términos de (λ_L, λ_V, λ12), donde λ_L = E[goles_local] y λ_V = E[goles_visit].
Internamente: λ1 = λ_L - λ12, λ2 = λ_V - λ12 (ambos deben ser ≥ 0).

λ12 captura la "co-ocurrencia" de goles — útil para modelar empates más realistas que Poisson independiente.
Si λ12 = 0, recuperás Poisson independiente.

Calibración: dados constraints del mercado (P(home win), P(over 2.5), P(btts), etc.) hacemos fit por mínimos cuadrados.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, exp, factorial
from typing import Mapping

import numpy as np
from scipy.optimize import minimize


# -------------------- PMF de bivariate Poisson --------------------

def bivariate_poisson_pmf(x: int, y: int, lam_L: float, lam_V: float, lam12: float) -> float:
    """P(X=x, Y=y) bajo bivariate Poisson con marginales (λ_L, λ_V) y covarianza λ12."""
    if lam12 < 0 or lam12 > min(lam_L, lam_V):
        raise ValueError(f"λ12={lam12} fuera de [0, min(λ_L, λ_V)={min(lam_L, lam_V)}]")
    if x < 0 or y < 0:
        return 0.0

    lam1 = lam_L - lam12
    lam2 = lam_V - lam12

    # Suma sobre k = 0..min(x,y)
    s = 0.0
    for k in range(min(x, y) + 1):
        s += (
            comb(x, k) * comb(y, k) * factorial(k)
            * (lam12 / (lam1 * lam2)) ** k
            if lam1 > 0 and lam2 > 0 else
            (1.0 if k == 0 else 0.0)
        )

    pmf = exp(-(lam1 + lam2 + lam12)) * (lam1 ** x / factorial(x)) * (lam2 ** y / factorial(y)) * s
    return pmf


def score_grid(
    lam_L: float,
    lam_V: float,
    lam12: float = 0.0,
    max_goals: int = 7,
) -> np.ndarray:
    """Grid 2D P[gL, gV] de marcadores, normalizada para sumar 1 (la cola > max_goals se redistribuye)."""
    grid = np.zeros((max_goals + 1, max_goals + 1))
    for gL in range(max_goals + 1):
        for gV in range(max_goals + 1):
            grid[gL, gV] = bivariate_poisson_pmf(gL, gV, lam_L, lam_V, lam12)
    # Truncamos cola y normalizamos
    total = grid.sum()
    if total > 0:
        grid = grid / total
    return grid


# -------------------- marginales de la grilla --------------------

@dataclass(frozen=True)
class GridMarginals:
    p_home_win: float
    p_draw: float
    p_away_win: float
    p_over_0_5: float
    p_over_1_5: float
    p_over_2_5: float
    p_over_3_5: float
    p_btts: float
    expected_goals_L: float
    expected_goals_V: float


def marginals(grid: np.ndarray) -> GridMarginals:
    """Extrae las probabilidades 'mercado-comparables' desde la grilla."""
    n = grid.shape[0]
    p_home = sum(grid[gL, gV] for gL in range(n) for gV in range(n) if gL > gV)
    p_draw = sum(grid[g, g] for g in range(n))
    p_away = sum(grid[gL, gV] for gL in range(n) for gV in range(n) if gL < gV)

    def p_over(threshold: float) -> float:
        return sum(
            grid[gL, gV]
            for gL in range(n) for gV in range(n)
            if (gL + gV) > threshold
        )

    p_btts = sum(grid[gL, gV] for gL in range(1, n) for gV in range(1, n))

    eg_L = sum(gL * grid[gL, gV] for gL in range(n) for gV in range(n))
    eg_V = sum(gV * grid[gL, gV] for gL in range(n) for gV in range(n))

    return GridMarginals(
        p_home_win=p_home, p_draw=p_draw, p_away_win=p_away,
        p_over_0_5=p_over(0.5), p_over_1_5=p_over(1.5),
        p_over_2_5=p_over(2.5), p_over_3_5=p_over(3.5),
        p_btts=p_btts,
        expected_goals_L=eg_L, expected_goals_V=eg_V,
    )


# -------------------- calibración contra constraints del mercado --------------------

@dataclass(frozen=True)
class MarketConstraints:
    """Probabilidades del mercado (después de de-vig + agregación)."""
    p_home_win: float
    p_draw: float
    p_away_win: float
    p_over_2_5: float | None = None
    p_btts: float | None = None


def fit_params(
    constraints: MarketConstraints,
    max_goals: int = 7,
    init_lam_L: float = 1.3,
    init_lam_V: float = 1.1,
    init_lam12: float = 0.1,
) -> tuple[float, float, float]:
    """Encuentra (λ_L, λ_V, λ12) que mejor matcheen las constraints del mercado.

    Minimiza error cuadrático sobre P(home), P(draw), P(away), [P(over 2.5)], [P(btts)].
    """

    targets: list[tuple[str, float]] = [
        ("home", constraints.p_home_win),
        ("draw", constraints.p_draw),
        ("away", constraints.p_away_win),
    ]
    if constraints.p_over_2_5 is not None:
        targets.append(("o25", constraints.p_over_2_5))
    if constraints.p_btts is not None:
        targets.append(("btts", constraints.p_btts))

    def loss(params: np.ndarray) -> float:
        lam_L, lam_V, lam12 = params
        # Penalización suave en lugar de hard constraint
        if lam_L <= 0 or lam_V <= 0 or lam12 < 0 or lam12 > min(lam_L, lam_V):
            return 1e6
        grid = score_grid(lam_L, lam_V, lam12, max_goals=max_goals)
        m = marginals(grid)
        d = {
            "home": m.p_home_win, "draw": m.p_draw, "away": m.p_away_win,
            "o25": m.p_over_2_5, "btts": m.p_btts,
        }
        return sum((d[k] - v) ** 2 for k, v in targets)

    res = minimize(
        loss,
        x0=np.array([init_lam_L, init_lam_V, init_lam12]),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 500},
    )
    return tuple(res.x.tolist())  # type: ignore[return-value]


# -------------------- expected points helpers --------------------

def expected_points(
    pick: tuple[int, int],
    grid: np.ndarray,
    points_rule: callable | None = None,
) -> float:
    """E[points | pick] bajo la grilla de marcadores `grid`.

    points_rule(pick, actual) → int. Default: regla JMLM 5/4/3/1/0.
    """
    if points_rule is None:
        points_rule = jmlm_points

    n = grid.shape[0]
    e = 0.0
    for gL in range(n):
        for gV in range(n):
            e += grid[gL, gV] * points_rule(pick, (gL, gV))
    return e


def jmlm_points(pick: tuple[int, int], actual: tuple[int, int]) -> int:
    """Regla de puntuación de la penca JMLM.

    5: marcador exacto
    4: ganador correcto + goles de UN equipo
    3: solo ganador (o empate)
    1: goles de UN equipo sin acertar ganador
    0: nada
    """
    pgL, pgV = pick
    agL, agV = actual

    if (pgL, pgV) == (agL, agV):
        return 5

    pick_winner = "H" if pgL > pgV else ("A" if pgL < pgV else "D")
    actual_winner = "H" if agL > agV else ("A" if agL < agV else "D")
    winner_match = pick_winner == actual_winner

    one_team_correct = (pgL == agL) or (pgV == agV)

    if winner_match and one_team_correct:
        return 4
    if winner_match:
        return 3
    if one_team_correct:
        return 1
    return 0


if __name__ == "__main__":
    # Smoke: España vs Cabo Verde, supuesto fuerte favoritismo
    lam_L, lam_V, lam12 = 2.5, 0.5, 0.05
    g = score_grid(lam_L, lam_V, lam12)
    m = marginals(g)
    print(f"P(home)={m.p_home_win:.3f}  P(draw)={m.p_draw:.3f}  P(away)={m.p_away_win:.3f}")
    print(f"P(over 2.5)={m.p_over_2_5:.3f}  P(btts)={m.p_btts:.3f}")
    print(f"E[gL]={m.expected_goals_L:.2f}  E[gV]={m.expected_goals_V:.2f}")

    # Pick 2-0 vs pick 1-0: comparar EV
    print(f"E[pts | 2-0] = {expected_points((2, 0), g):.3f}")
    print(f"E[pts | 1-0] = {expected_points((1, 0), g):.3f}")
    print(f"E[pts | 3-0] = {expected_points((3, 0), g):.3f}")

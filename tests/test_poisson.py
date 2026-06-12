"""Tests de Poisson bivariada + scoring."""

import math

import numpy as np
import pytest

from src.model.poisson import (
    bivariate_poisson_pmf,
    expected_points,
    fit_params,
    jmlm_points,
    marginals,
    score_grid,
    MarketConstraints,
)


def test_pmf_sums_to_one_with_independent_poisson():
    """Si λ12=0, recuperamos Poisson independiente. Suma sobre grid grande ≈ 1."""
    lam_L, lam_V = 1.5, 1.0
    s = 0.0
    for x in range(20):
        for y in range(20):
            s += bivariate_poisson_pmf(x, y, lam_L, lam_V, lam12=0.0)
    assert abs(s - 1.0) < 1e-6, f"sumó {s}"


def test_score_grid_normalized():
    g = score_grid(1.5, 1.0, lam12=0.1)
    assert abs(g.sum() - 1.0) < 1e-9


def test_marginals_recover_expected_goals():
    g = score_grid(2.0, 1.0, lam12=0.0, max_goals=10)
    m = marginals(g)
    assert abs(m.expected_goals_L - 2.0) < 0.05
    assert abs(m.expected_goals_V - 1.0) < 0.05


def test_marginals_1x2_complete():
    g = score_grid(1.5, 1.0, lam12=0.05)
    m = marginals(g)
    assert abs(m.p_home_win + m.p_draw + m.p_away_win - 1.0) < 1e-9


# ---------- regla JMLM 6/4/3/0 (vigente desde 2026-06-12) ----------

@pytest.mark.parametrize("pick,actual,expected", [
    ((2, 1), (2, 1), 6),   # marcador exacto
    ((1, 1), (1, 1), 6),   # empate exacto
    ((1, 0), (2, 1), 4),   # ganador + diferencia de gol (1)
    ((3, 1), (2, 0), 4),   # ganador + diferencia de gol (2)
    ((2, 0), (2, 1), 3),   # ganador correcto, diferencia errada (2 vs 1)
    ((1, 0), (2, 0), 3),   # ganador correcto, diferencia errada (1 vs 2)
    ((1, 1), (2, 2), 3),   # empate acertado no exacto → 3 (NO 4: dif 0 es trivial)
    ((0, 0), (2, 2), 3),   # idem
    ((2, 2), (1, 2), 0),   # ganador errado → 0 (ya no hay tier de 1pt)
    ((2, 1), (1, 2), 0),   # ganador errado
    ((0, 0), (1, 2), 0),   # nada
])
def test_jmlm_points(pick, actual, expected):
    assert jmlm_points(pick, actual) == expected


def test_expected_points_modal_is_optimal_for_chalk():
    """Para grid muy concentrado en un solo marcador, ese marcador debe maximizar E[pts]."""
    g = np.zeros((6, 6))
    g[2, 0] = 1.0   # 100% prob de 2-0
    e_correct = expected_points((2, 0), g)
    e_other = expected_points((1, 0), g)
    assert e_correct > e_other
    assert e_correct == 6.0


def test_fit_params_recovers_marginals():
    """Si le damos al fitter las marginales de un grid conocido, debería volver a (λ_L, λ_V, λ12) parecidos."""
    true_L, true_V, true_12 = 1.6, 1.1, 0.08
    g = score_grid(true_L, true_V, true_12)
    m = marginals(g)
    c = MarketConstraints(
        p_home_win=m.p_home_win,
        p_draw=m.p_draw,
        p_away_win=m.p_away_win,
        p_over_2_5=m.p_over_2_5,
        p_btts=m.p_btts,
    )
    fit_L, fit_V, fit_12 = fit_params(c)
    # Tolerancia generosa porque hay equifinalidad
    assert abs(fit_L - true_L) < 0.2
    assert abs(fit_V - true_V) < 0.2

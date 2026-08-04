"""Tests del kernel de puntos de la Penca Supermatch (Clausura 2026).

Los primeros seis casos son LITERALES los ejemplos del Art. 6 del reglamento
(capturado de la web el 2026-08-04) — si alguno falla, o el kernel está mal o
Supermatch cambió las reglas.
"""

import numpy as np
import pytest

from src.clausura.scoring import (
    best_pick,
    expected_points_grid,
    rank_picks,
    supermatch_points,
)
from src.model.poisson import expected_points, score_grid


# -------------------- ejemplos literales del reglamento (pick 2-0) --------------------

@pytest.mark.parametrize("actual,esperado", [
    ((2, 3), 1),  # acertó goles del local
    ((4, 1), 2),  # acertó solo el ganador
    ((3, 1), 3),  # ganador + diferencia
    ((1, 0), 3),  # ganador + goles del visitante
    ((2, 0), 8),  # exacto: 2+1+1+1+3
    ((0, 2), 0),  # nada
])
def test_ejemplos_reglamento(actual, esperado):
    assert supermatch_points((2, 0), actual) == esperado


# -------------------- empates --------------------

def test_empate_acertado_no_exacto():
    # ganador (empate) 2 + diferencia 1 (0=0 por definición) + goles 0 = 3
    assert supermatch_points((1, 1), (2, 2)) == 3


def test_empate_exacto():
    assert supermatch_points((1, 1), (1, 1)) == 8


def test_empate_errado_con_un_lado_acertado():
    # pick 1-1, resultado 1-0: goles local acertados, nada más
    assert supermatch_points((1, 1), (1, 0)) == 1


def test_diferencia_no_paga_sin_ganador():
    # pick 2-1 (dif +1), resultado 0-1 (dif -1): nada. Y pick 1-2 vs 2-1:
    # misma "distancia" pero ganador opuesto → la dif condicionada no paga.
    assert supermatch_points((1, 2), (2, 1)) == 0


def test_goles_visitante_paga_con_ganador_errado():
    # pick 0-2, resultado 3-2: goles visitante acertados aunque ganó el local
    assert supermatch_points((0, 2), (3, 2)) == 1


# -------------------- preferencial (estrella x2) --------------------

def test_preferencial_duplica():
    assert supermatch_points((2, 0), (2, 0), preferencial=True) == 16
    assert supermatch_points((2, 0), (3, 1), preferencial=True) == 6
    assert supermatch_points((2, 0), (0, 2), preferencial=True) == 0


# -------------------- propiedades --------------------

def test_bounds_exhaustivo():
    for pgL in range(5):
        for pgV in range(5):
            for agL in range(5):
                for agV in range(5):
                    pts = supermatch_points((pgL, pgV), (agL, agV))
                    assert 0 <= pts <= 8
                    # 8 si y solo si exacto
                    assert (pts == 8) == ((pgL, pgV) == (agL, agV))


def test_exacto_siempre_8():
    for gL in range(6):
        for gV in range(6):
            assert supermatch_points((gL, gV), (gL, gV)) == 8


# -------------------- E[pts] sobre la grilla --------------------

@pytest.fixture(scope="module")
def grid():
    # favorito local moderado, típico de la Primera uruguaya
    return score_grid(lam_L=1.6, lam_V=0.9, lam12=0.1)


def test_ev_cerrado_coincide_con_fuerza_bruta(grid):
    """expected_points_grid (forma cerrada) == suma celda a celda con el kernel."""
    for pick in [(1, 0), (2, 1), (0, 0), (1, 1), (2, 0), (0, 2)]:
        brute = expected_points(pick, grid, points_rule=supermatch_points)
        closed = expected_points_grid(pick, grid)
        assert closed == pytest.approx(brute, abs=1e-9)


def test_ev_preferencial_es_doble(grid):
    for pick in [(1, 0), (1, 1)]:
        assert expected_points_grid(pick, grid, preferencial=True) == pytest.approx(
            2 * expected_points_grid(pick, grid), abs=1e-12
        )


def test_best_pick_es_max_de_rank(grid):
    ranked = rank_picks(grid)
    assert best_pick(grid) == ranked[0]
    evs = [ev for _, ev in ranked]
    assert evs == sorted(evs, reverse=True)


def test_argmax_difiere_de_jmlm_en_general():
    """Sanity del insight estratégico: con λ bajos el óptimo Supermatch puede no ser
    la celda modal (el componente por-lado empuja a las marginales modales)."""
    g = score_grid(lam_L=1.05, lam_V=1.0, lam12=0.15)
    (pick_sm, _) = best_pick(g)
    modal = np.unravel_index(np.argmax(g), g.shape)
    # No exigimos que difieran siempre (depende de λ) pero el EV del pick óptimo
    # tiene que ser >= al de la celda modal — estricto o igual, nunca menor.
    assert expected_points_grid(pick_sm, g) >= expected_points_grid(tuple(modal), g)

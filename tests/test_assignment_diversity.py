"""Tests del fix A: la asignación voraz no debe amontonar las pencas sobrantes en un
solo marcador (chalk).

Regresión de m108 (EEUU 4-1 Paraguay): con 15 pencas y ~5 candidatos, el voraz ponía
13/15 en el 1-0 porque las pencas dominadas empataban en el objetivo de la unión y el
desempate caía siempre al candidato 0. El fix reparte hacia el candidato menos usado.
"""

from collections import Counter

import numpy as np

from src.model.poisson import score_grid
from src.strategy.assignment import greedy_assignment


def _menu():
    """Menú de 5 candidatos distintos (como generate_candidates), ev primero."""
    return [
        {"objective": "ev", "score": [1, 0]},
        {"objective": "differentiated", "score": [2, 0]},
        {"objective": "tail", "score": [3, 0]},
        {"objective": "upset", "score": [0, 0]},
        {"objective": "variance", "score": [1, 1]},
    ]


def _exposure(assignment):
    return Counter(f'{p["score"][0]}-{p["score"][1]}' for _, p, _ in assignment)


def test_emax_reparte_15_pencas_sin_flood():
    grid = score_grid(1.4, 0.74, 0.13)
    menu = _menu()
    penca_ids = list(range(1, 16))  # 15 pencas
    assignment, meta = greedy_assignment(menu, penca_ids, grid, pencas_standings={})
    exp = _exposure(assignment)
    # Las 5 picks del menú se usan, y ninguna acapara (antes: 13/15 en 1-0)
    assert len(exp) == len(menu), f"esperaba {len(menu)} marcadores distintos, hubo {dict(exp)}"
    assert max(exp.values()) <= 4, f"flood: {dict(exp)}"
    # El líder (procesado primero) ancla la EV
    lead_pick = next(p for pid, p, _ in assignment if pid == 1)
    assert tuple(lead_pick["score"]) == (1, 0)


def test_fallback_ptopk_cero_tambien_reparte():
    # Régimen exacto de m108: cutoff inalcanzable en un partido → P(top-K)=0 → fallback e_max.
    grid = score_grid(1.4, 0.74, 0.13)
    menu = _menu()
    penca_ids = list(range(1, 16))
    assignment, meta = greedy_assignment(
        menu, penca_ids, grid,
        pencas_standings={pid: {"points_total": 0} for pid in penca_ids},
        pool_top_k_threshold=999,  # imposible en un partido
        pool_q=grid,               # cualquier q
    )
    assert "P(top-K)=0" in meta["objective"]
    exp = _exposure(assignment)
    assert max(exp.values()) <= 4, f"flood en fallback: {dict(exp)}"
    assert len(exp) == len(menu)


def test_standings_heterogeneos_no_amontonan():
    # Pencas con puntajes dispares (lo que rompía la diversificación a fecha avanzada).
    grid = score_grid(1.4, 0.74, 0.13)
    menu = _menu()
    penca_ids = list(range(1, 16))
    standings = {pid: {"points_total": pid} for pid in penca_ids}  # 1..15, todas distintas
    assignment, _ = greedy_assignment(menu, penca_ids, grid, pencas_standings=standings)
    exp = _exposure(assignment)
    assert max(exp.values()) <= 4, f"flood con standings dispares: {dict(exp)}"


def test_menos_pencas_que_candidatos_sin_repetir():
    grid = score_grid(1.4, 0.74, 0.13)
    menu = _menu()
    assignment, _ = greedy_assignment(menu, [1, 2, 3], grid, pencas_standings={})
    exp = _exposure(assignment)
    assert all(v == 1 for v in exp.values()), f"con 3 pencas no debería repetir: {dict(exp)}"
    assert len(exp) == 3

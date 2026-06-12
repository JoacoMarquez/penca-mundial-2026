"""Tests del cutoff con horizonte (β·√(partidos restantes)) en la asignación."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.agent.pipeline import _matches_remaining
from src.model.poisson import jmlm_points, score_grid
from src.strategy.assignment import greedy_assignment
from src.strategy.portfolio import generate_candidates, picks_to_dicts
from src.meta.pool import PoolModelConfig, pool_pick_distribution


def _setup(lam_L=1.8, lam_V=0.7):
    grid = score_grid(lam_L, lam_V, 0.1, max_goals=7)
    cands = picks_to_dicts(generate_candidates(
        grid, market_p_home=0.6, market_p_away=0.15,
        pool_config=PoolModelConfig(), max_candidates=10,
    ))
    q = pool_pick_distribution(grid, PoolModelConfig())
    return grid, cands, q


def test_premium_recorded_in_meta():
    grid, cands, q = _setup()
    ids = list(range(1, 6))
    _, meta = greedy_assignment(cands, ids, grid, {},
                                pool_top_k_threshold=3, pool_q=q,
                                points_rule=jmlm_points, horizon_premium=4.0)
    assert meta["horizon_premium"] == 4.0
    assert meta["threshold"] == 3  # el threshold crudo se preserva para observabilidad


def test_zero_premium_is_backward_compatible():
    """premium=0 → resultado idéntico al comportamiento previo."""
    grid, cands, q = _setup()
    ids = list(range(1, 16))
    st = {pid: {"points_total": 5} for pid in ids}
    res_a, meta_a = greedy_assignment(cands, ids, grid, st,
                                      pool_top_k_threshold=6, pool_q=q,
                                      points_rule=jmlm_points)
    res_b, meta_b = greedy_assignment(cands, ids, grid, st,
                                      pool_top_k_threshold=6, pool_q=q,
                                      points_rule=jmlm_points, horizon_premium=0.0)
    assert [tuple(p["score"]) for _, p, _ in res_a] == [tuple(p["score"]) for _, p, _ in res_b]
    assert meta_a["exposure"] == meta_b["exposure"]


def test_unreachable_premium_falls_back_to_emax():
    """Premium altísimo (principio del torneo): listón inalcanzable en un partido →
    fallback E[max], que es el comportamiento diseñado (diversifica)."""
    grid, cands, q = _setup()
    ids = list(range(1, 16))
    st = {pid: {"points_total": 0} for pid in ids}
    res, meta = greedy_assignment(cands, ids, grid, st,
                                  pool_top_k_threshold=0, pool_q=q,
                                  points_rule=jmlm_points,
                                  horizon_premium=20.0)  # 2·√100
    assert len(res) == 15
    assert meta["objective"].startswith("e_max")
    assert meta["horizon_premium"] == 20.0


def test_moderate_premium_shifts_exposure_away_from_pure_anchor():
    """Premium moderado: el listón sube y el voraz necesita marcadores que paguen 6,
    no alcanza con el ancla — la exposición no puede ser 100% modal."""
    grid, cands, q = _setup()
    ids = list(range(1, 16))
    st = {pid: {"points_total": 10} for pid in ids}
    # threshold 12, premium 3 → 15: necesita el pick correcto para llegar (10+6=16 ≥ 15)
    res, meta = greedy_assignment(cands, ids, grid, st,
                                  pool_top_k_threshold=12, pool_q=q,
                                  points_rule=jmlm_points, horizon_premium=3.0)
    exposure = meta["exposure"]
    assert len(exposure) > 1  # cobertura, no monocultivo


# ---------- _matches_remaining ----------

def _fx(*kickoffs):
    return {"fase_grupos": [{"id": i, "kickoff_utc": k} for i, k in enumerate(kickoffs)],
            "eliminatorias": []}


def test_matches_remaining_counts_future_only():
    now = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    fx = _fx("2026-06-14T19:00:00.000Z",   # ya jugado
             "2026-06-15T11:00:00.000Z",   # ya arrancó
             "2026-06-15T19:00:00.000Z",   # hoy más tarde
             "2026-06-16T19:00:00.000Z")   # mañana
    assert _matches_remaining(fx, now) == 2


def test_matches_remaining_counts_dateless_knockouts():
    now = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    fx = {"fase_grupos": [{"id": 1, "kickoff_utc": "2026-06-14T19:00:00.000Z"}],
          "eliminatorias": [{"id": 200}, {"id": 201}]}  # sin fecha → cuentan
    assert _matches_remaining(fx, now) == 2


def test_matches_remaining_floor_is_one():
    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    fx = _fx("2026-06-14T19:00:00.000Z")
    assert _matches_remaining(fx, now) == 1

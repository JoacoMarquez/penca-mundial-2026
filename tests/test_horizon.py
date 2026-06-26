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


# ---------- protección de la líder (protect_theta) ----------

def _winner(score):
    gL, gV = score
    return "1" if gL > gV else ("2" if gL < gV else "X")


def test_protect_theta_none_is_backward_compatible():
    """protect_theta=None (default) → idéntico a no pasarlo: no cambia nada existente."""
    grid, cands, q = _setup()
    ids = list(range(1, 16))
    st = {pid: {"points_total": 10} for pid in ids}
    res_a, meta_a = greedy_assignment(cands, ids, grid, st, pool_top_k_threshold=12,
                                      pool_q=q, points_rule=jmlm_points, horizon_premium=3.0)
    res_b, meta_b = greedy_assignment(cands, ids, grid, st, pool_top_k_threshold=12,
                                      pool_q=q, points_rule=jmlm_points, horizon_premium=3.0,
                                      protect_theta=None)
    assert [tuple(p["score"]) for _, p, _ in res_a] == [tuple(p["score"]) for _, p, _ in res_b]
    assert meta_b["protected_pencas"] == []


def test_protect_leader_above_cutoff_gets_favorite_not_antifavorite():
    """Con premium alto, sin protección la líder es empujada a un anti-favorito; con
    protect_theta vuelve al favorito (ancla EV) y queda registrada como protegida."""
    grid, cands, q = _setup()  # favorito local (p_home=0.6)
    ids = list(range(1, 16))
    # Líder ENTRE el corte de hoy (10) y el proyectado (10+8=18) — el rango donde el premium
    # la trata como rezagada y la empuja al anti-favorito (caso real 1891: 161 entre 152 y 165.6).
    st = {pid: {"points_total": 8} for pid in ids}
    st[1] = {"points_total": 15}
    common = dict(pool_top_k_threshold=10, pool_q=q, points_rule=jmlm_points, horizon_premium=8.0)

    res_sq, _ = greedy_assignment(cands, ids, grid, st, **common)               # status quo
    res_pr, meta_pr = greedy_assignment(cands, ids, grid, st, protect_theta=0.4, **common)

    lead_sq = next(tuple(p["score"]) for pid, p, _ in res_sq if pid == 1)
    lead_pr = next(tuple(p["score"]) for pid, p, _ in res_pr if pid == 1)

    assert _winner(lead_sq) != "1"      # bug: la líder NO juega al favorito
    assert _winner(lead_pr) == "1"      # fix: la líder vuelve al favorito
    assert 1 in meta_pr["protected_pencas"]
    assert meta_pr["protect_theta"] == 0.4


def test_protect_theta_only_exempts_pencas_above_cutoff():
    """Una penca debajo del corte NO se protege (mantiene el premium completo → β intacto)."""
    grid, cands, q = _setup()
    ids = list(range(1, 16))
    st = {pid: {"points_total": 8} for pid in ids}   # todas debajo del corte (10)
    st[1] = {"points_total": 15}                       # sólo la líder arriba del corte
    _, meta = greedy_assignment(cands, ids, grid, st, pool_top_k_threshold=10, pool_q=q,
                                points_rule=jmlm_points, horizon_premium=8.0, protect_theta=0.4)
    assert meta["protected_pencas"] == [1]   # sólo la líder


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

"""Tests del asignador voraz genérico a N pencas (con repetición/exposición)."""

import numpy as np
import pytest

from src.model.poisson import score_grid, marginals
from src.meta.pool import pool_pick_distribution, PoolModelConfig
from src.strategy.portfolio import generate_candidates, picks_to_dicts
from src.strategy.assignment import greedy_assignment


@pytest.fixture
def favorite_match():
    g = score_grid(2.0, 0.8, lam12=0.08)
    m = marginals(g)
    return g, m


def _candidates(g, m, max_candidates=10):
    return picks_to_dicts(
        generate_candidates(g, m.p_home_win, m.p_away_win, max_candidates=max_candidates)
    )


# -------------------- generate_candidates --------------------

def test_candidates_are_distinct(favorite_match):
    g, m = favorite_match
    cands = generate_candidates(g, m.p_home_win, m.p_away_win, max_candidates=8)
    scores = [(c.score_local, c.score_visit) for c in cands]
    assert len(scores) == len(set(scores)), f"candidatos repetidos: {scores}"


def test_candidates_first_is_ev(favorite_match):
    g, m = favorite_match
    cands = generate_candidates(g, m.p_home_win, m.p_away_win, max_candidates=8)
    assert cands[0].objective == "ev"
    ev_max = max(c.e_points for c in cands)
    assert cands[0].e_points == pytest.approx(ev_max)


def test_candidates_respects_cap(favorite_match):
    g, m = favorite_match
    cands = generate_candidates(g, m.p_home_win, m.p_away_win, max_candidates=6)
    assert 1 <= len(cands) <= 6


# -------------------- greedy_assignment N=15 --------------------

def test_greedy_returns_n_assignments(favorite_match):
    g, m = favorite_match
    cands = _candidates(g, m)
    penca_ids = list(range(1, 16))  # 15 pencas
    pool_q = pool_pick_distribution(g, PoolModelConfig())
    result, meta = greedy_assignment(cands, penca_ids, g, {}, pool_top_k_threshold=10, pool_q=pool_q)
    assert len(result) == 15
    assert [pid for pid, _, _ in result] == penca_ids  # mismo orden de entrada


def test_greedy_picks_are_from_candidate_menu(favorite_match):
    g, m = favorite_match
    cands = _candidates(g, m)
    menu = {tuple(c["score"]) for c in cands}
    penca_ids = list(range(1, 16))
    result, _ = greedy_assignment(cands, penca_ids, g, {})
    for _, pick, _ in result:
        assert tuple(pick["score"]) in menu


def test_greedy_allows_repetition_with_15_pencas(favorite_match):
    """Con 15 pencas y ~8-10 candidatos, debe haber repetición (no 15 distintas)."""
    g, m = favorite_match
    cands = _candidates(g, m)
    penca_ids = list(range(1, 16))
    result, _ = greedy_assignment(cands, penca_ids, g, {})
    distinct = {tuple(pick["score"]) for _, pick, _ in result}
    assert 2 <= len(distinct) < 15, f"esperaba repetición, distinct={distinct}"


def test_greedy_exposure_sums_to_n(favorite_match):
    g, m = favorite_match
    cands = _candidates(g, m)
    penca_ids = list(range(1, 16))
    pool_q = pool_pick_distribution(g, PoolModelConfig())
    _, meta = greedy_assignment(cands, penca_ids, g, {}, pool_top_k_threshold=20, pool_q=pool_q)
    assert "exposure" in meta
    assert sum(meta["exposure"].values()) == 15


def test_greedy_leader_gets_ev_anchor(favorite_match):
    """La penca líder (más puntos) se procesa primero → toma el ancla EV."""
    g, m = favorite_match
    cands = _candidates(g, m)
    ev_score = tuple(cands[0]["score"])  # candidato 0 = ev
    penca_ids = [10, 20, 30]
    standings = {10: {"points_total": 50}, 20: {"points_total": 5}, 30: {"points_total": 1}}
    result, _ = greedy_assignment(cands, penca_ids, g, standings)  # e_max (sin threshold)
    by_pid = {pid: (tuple(pick["score"]), rank) for pid, pick, rank in result}
    assert by_pid[10][1] == 1  # líder es rank 1
    assert by_pid[10][0] == ev_score


def test_greedy_ranks_are_unique_and_complete(favorite_match):
    g, m = favorite_match
    cands = _candidates(g, m)
    penca_ids = list(range(1, 16))
    result, _ = greedy_assignment(cands, penca_ids, g, {})
    ranks = sorted(r for _, _, r in result)
    assert ranks == list(range(1, 16))


def test_greedy_emax_when_no_threshold(favorite_match):
    g, m = favorite_match
    cands = _candidates(g, m)
    result, meta = greedy_assignment(cands, list(range(1, 16)), g, {})
    assert meta["objective"] == "e_max"
    assert len(result) == 15


def test_greedy_empty_inputs():
    g = score_grid(1.5, 1.2)
    result, meta = greedy_assignment([], [1, 2, 3], g, {})
    assert result == []

"""Tests del backtest Monte Carlo de resultados (sin dependencia de archivos)."""

import pytest

from src.backtest.runner import BacktestMatch, run_montecarlo, run_sequential_mc


def _synthetic_matches() -> list[BacktestMatch]:
    """Set chico y variado: un favorito fuerte, un parejo, un favorito visitante."""
    return [
        BacktestMatch("m1", "A", "B", 2, 0, 0.62, 0.23, 0.15),
        BacktestMatch("m2", "C", "D", 1, 1, 0.38, 0.30, 0.32),
        BacktestMatch("m3", "E", "F", 0, 2, 0.20, 0.27, 0.53),
        BacktestMatch("m4", "G", "H", 1, 0, 0.50, 0.27, 0.23),
    ]


def test_montecarlo_metrics_in_range():
    r = run_montecarlo(_synthetic_matches(), n_sims=500, n_pencas=5, max_candidates=8)
    assert 0.0 <= r.p_at_least_one_wins <= 1.0
    assert 0.0 <= r.p_top_5_percent <= 1.0
    assert len(r.portfolio_max_points) == 500
    assert r.n_pencas == 5


def test_montecarlo_win_or_tie_geq_win():
    r = run_montecarlo(_synthetic_matches(), n_sims=500, n_pencas=10, max_candidates=8)
    assert r.p_at_least_one_wins_or_tie >= r.p_at_least_one_wins


def test_montecarlo_distinct_entries_capped_by_menu():
    """Con 15 pencas y menú de 8, las entradas distintas no superan el menú (resto = clones)."""
    r = run_montecarlo(_synthetic_matches(), n_sims=300, n_pencas=15, max_candidates=8)
    assert r.distinct_entries <= 8
    assert r.distinct_entries >= 1


def test_montecarlo_deterministic_with_seed():
    a = run_montecarlo(_synthetic_matches(), n_sims=400, n_pencas=5, seed=7)
    b = run_montecarlo(_synthetic_matches(), n_sims=400, n_pencas=5, seed=7)
    assert a.p_at_least_one_wins == b.p_at_least_one_wins


def test_montecarlo_diversification_beats_single_entry():
    """5 pencas diversificadas deberían ganar ≥ que una sola entrada EV (chalk)."""
    matches = _synthetic_matches()
    multi = run_montecarlo(matches, n_sims=3000, n_pencas=8, max_candidates=10, seed=1)
    single = run_montecarlo(matches, n_sims=3000, n_pencas=1, max_candidates=10, seed=1)
    assert multi.p_at_least_one_wins >= single.p_at_least_one_wins


# -------------------- backtest secuencial --------------------

def test_sequential_metrics_in_range():
    r = run_sequential_mc(_synthetic_matches(), n_sims=300, n_pencas=10, max_candidates=8)
    assert 0.0 <= r["p_win"] <= 1.0
    assert r["p_win_tie"] >= r["p_win"]
    assert r["n_pencas"] == 10


def test_sequential_deterministic_with_seed():
    a = run_sequential_mc(_synthetic_matches(), n_sims=300, n_pencas=10, seed=3)
    b = run_sequential_mc(_synthetic_matches(), n_sims=300, n_pencas=10, seed=3)
    assert a["p_win"] == b["p_win"]


def test_sequential_more_pencas_help():
    """En el régimen secuencial (standings + P(top-K)), más pencas no deberían perjudicar."""
    matches = _synthetic_matches()
    few = run_sequential_mc(matches, n_sims=800, n_pencas=3, max_candidates=8, seed=5)
    many = run_sequential_mc(matches, n_sims=800, n_pencas=12, max_candidates=8, seed=5)
    assert many["p_win"] >= few["p_win"]

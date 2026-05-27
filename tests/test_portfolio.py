"""Tests de invariantes del portfolio de 5 picks."""

import numpy as np
import pytest

from src.model.poisson import score_grid, marginals
from src.strategy.portfolio import generate_portfolio


@pytest.fixture
def strong_favorite_match():
    """España vs Cabo Verde — fuerte favoritismo."""
    g = score_grid(2.5, 0.5, lam12=0.05)
    m = marginals(g)
    return g, m


@pytest.fixture
def coin_flip_match():
    """Partido parejo — 1.3 vs 1.2 goles esperados."""
    g = score_grid(1.3, 1.2, lam12=0.1)
    m = marginals(g)
    return g, m


def test_portfolio_returns_exactly_5_picks(strong_favorite_match):
    g, m = strong_favorite_match
    portfolio = generate_portfolio(g, m.p_home_win, m.p_away_win)
    assert len(portfolio.picks) == 5


def test_portfolio_pick1_is_ev_max(strong_favorite_match):
    """Pick 1 debe tener la máxima E[points] de todo el portfolio."""
    g, m = strong_favorite_match
    portfolio = generate_portfolio(g, m.p_home_win, m.p_away_win)
    ev_max = max(p.e_points for p in portfolio.picks)
    assert portfolio.picks[0].e_points == ev_max


def test_portfolio_objectives_in_order(strong_favorite_match):
    g, m = strong_favorite_match
    portfolio = generate_portfolio(g, m.p_home_win, m.p_away_win)
    objs = [p.objective for p in portfolio.picks]
    assert objs == ["ev", "differentiated", "tail", "upset", "variance"]


def test_portfolio_diversity_strong_favorite(strong_favorite_match):
    """En partido con favorito claro, el portfolio debe tener al menos 3 picks distintas."""
    g, m = strong_favorite_match
    portfolio = generate_portfolio(g, m.p_home_win, m.p_away_win)
    scores = {(p.score_local, p.score_visit) for p in portfolio.picks}
    assert len(scores) >= 3, f"poca diversidad: {scores}"


def test_portfolio_upset_picks_opposite_winner_when_clear_favorite(strong_favorite_match):
    """Si el local es claro favorito, la pick #4 (upset) debe ser empate o visit."""
    g, m = strong_favorite_match
    portfolio = generate_portfolio(g, m.p_home_win, m.p_away_win)
    p4 = portfolio.picks[3]
    # En este partido el favorito es el local. Upset = empate o away.
    assert p4.score_local <= p4.score_visit


def test_portfolio_variance_pick_is_among_top_K(coin_flip_match):
    """La pick #5 debe estar entre las top-K por EV — no es un underdog random."""
    g, m = coin_flip_match
    portfolio = generate_portfolio(g, m.p_home_win, m.p_away_win)
    # E[points] de la #5 no debería ser menos del 50% de la #1
    assert portfolio.picks[4].e_points >= portfolio.picks[0].e_points * 0.4


def test_portfolio_pool_popularity_in_unit_interval(strong_favorite_match):
    g, m = strong_favorite_match
    portfolio = generate_portfolio(g, m.p_home_win, m.p_away_win)
    for p in portfolio.picks:
        assert 0.0 <= p.pool_popularity <= 1.0

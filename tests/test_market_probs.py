"""Tests del módulo de de-vig + agregación."""

import pytest

from src.model.market_probs import (
    BookQuote,
    aggregate,
    devig_proportional,
    devig_shin,
    overround,
)


def test_overround_typical_1x2():
    odds = {"H": 2.10, "D": 3.40, "A": 3.60}
    o = overround(odds)
    assert 1.03 < o < 1.08, f"overround típico de 1X2 está entre 3% y 8%, got {o:.3f}"


def test_devig_proportional_sums_to_one():
    odds = {"H": 2.10, "D": 3.40, "A": 3.60}
    probs = devig_proportional(odds)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_devig_shin_sums_to_one():
    odds = {"H": 2.10, "D": 3.40, "A": 3.60}
    probs = devig_shin(odds)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_devig_shin_vs_proportional_favorite_bias():
    """Shin corrige el favorite-longshot bias: da prob MAYOR al favorito y MENOR al
    longshot que proportional (los insiders apuestan al fav → su cuota subestima la prob real)."""
    odds = {"H": 1.50, "D": 4.00, "A": 7.00}  # favorito fuerte
    p_prop = devig_proportional(odds)
    p_shin = devig_shin(odds)
    assert p_shin["H"] > p_prop["H"], (
        f"Shin debería subir el favorito vs proportional: shin={p_shin['H']:.3f} prop={p_prop['H']:.3f}"
    )
    assert p_shin["A"] < p_prop["A"], (
        f"Shin debería bajar el longshot vs proportional: shin={p_shin['A']:.3f} prop={p_prop['A']:.3f}"
    )


def test_aggregate_weighted():
    q1 = BookQuote(book="pinnacle", market="1x2", probs={"H": 0.50, "D": 0.30, "A": 0.20})
    q2 = BookQuote(book="bet365",   market="1x2", probs={"H": 0.40, "D": 0.35, "A": 0.25})
    weights = {"pinnacle": 0.7, "bet365": 0.3}
    agg = aggregate([q1, q2], weights)
    # H: 0.7*0.50 + 0.3*0.40 = 0.35 + 0.12 = 0.47
    assert abs(agg["H"] - 0.47) < 1e-6
    assert abs(sum(agg.values()) - 1.0) < 1e-9


def test_aggregate_redistributes_when_book_missing():
    """Si una casa configurada no está en quotes, sus pesos se redistribuyen."""
    q1 = BookQuote(book="pinnacle", market="1x2", probs={"H": 0.50, "D": 0.30, "A": 0.20})
    weights = {"pinnacle": 0.5, "bet365": 0.3, "betfair_exchange": 0.2}
    agg = aggregate([q1], weights)
    # Con solo pinnacle, debería recuperar exactamente las probs de pinnacle
    assert abs(agg["H"] - 0.50) < 1e-6
    assert abs(agg["D"] - 0.30) < 1e-6
    assert abs(agg["A"] - 0.20) < 1e-6

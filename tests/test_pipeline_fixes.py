"""Tests de los fixes del pipeline: fallback de odds (#4) y publicación (#3)."""

import os

import src.agent.pipeline as pipeline
from src.agent.pipeline import (
    OddsSnapshot, MOCK_CONSTRAINTS, Phase, build_constraints_with_fallback, _publish_assignment,
    _should_publish,
)


def _empty_snapshot():
    return OddsSnapshot(match_id="105", fetched_at="2026-06-11T00:00:00Z", odds_by_book={})


# -------------------- #4 fallback de odds --------------------

def test_fallback_usa_version_previa(monkeypatch):
    prev = [{"version": 1, "odds_source": "live",
             "constraints": {"p_home": 0.6, "p_draw": 0.25, "p_away": 0.15, "p_over_2_5": 0.5}}]
    monkeypatch.setattr(pipeline, "load_previous_predictions", lambda mid: prev)
    c, src = build_constraints_with_fallback(_empty_snapshot(), {}, "105", "MEX vs RSA")
    assert src == "previous"
    assert (c.p_home_win, c.p_draw, c.p_away_win) == (0.6, 0.25, 0.15)
    assert c.p_over_2_5 == 0.5


def test_fallback_ignora_previa_que_era_mock(monkeypatch):
    # Una versión previa que ya fue mock NO debe reusarse → cae a mock de nuevo.
    prev = [{"version": 1, "odds_source": "mock",
             "constraints": {"p_home": 0.4, "p_draw": 0.3, "p_away": 0.3}}]
    monkeypatch.setattr(pipeline, "load_previous_predictions", lambda mid: prev)
    _, src = build_constraints_with_fallback(_empty_snapshot(), {}, "105", "MEX vs RSA")
    assert src == "mock"


def test_fallback_mock_si_no_hay_previa(monkeypatch):
    monkeypatch.setattr(pipeline, "load_previous_predictions", lambda mid: [])
    c, src = build_constraints_with_fallback(_empty_snapshot(), {}, "105", "MEX vs RSA")
    assert src == "mock"
    assert c is MOCK_CONSTRAINTS


def test_fallback_toma_la_mas_reciente(monkeypatch):
    # load_previous_predictions devuelve en orden ascendente; usamos la última (más nueva).
    prev = [
        {"version": 1, "odds_source": "live", "constraints": {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}},
        {"version": 2, "odds_source": "live", "constraints": {"p_home": 0.7, "p_draw": 0.2, "p_away": 0.1}},
    ]
    monkeypatch.setattr(pipeline, "load_previous_predictions", lambda mid: prev)
    c, src = build_constraints_with_fallback(_empty_snapshot(), {}, "105", "MEX vs RSA")
    assert src == "previous" and c.p_home_win == 0.7


# -------------------- #3 publicación --------------------

def test_publish_dry_run_ok(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    assignment = [(101, {"score": [1, 0]}, 1), (102, {"score": [2, 1]}, 2)]
    # Se publica en cualquier pasada (red de seguridad), no solo en T-30min.
    for ph in (Phase.T_24H, Phase.T_3H, Phase.T_30MIN):
        ok, detail = _publish_assignment("105", ph, assignment)
        assert ok is True and detail is None


# -------------------- guarda: nunca publicar MOCK al pool --------------------

def test_no_publica_constraints_mock():
    """Las constraints MOCK (inventadas) NO van al pool: meterían una pick sesgada al
    'home' cuando no hubo odds reales (raíz del caso 177)."""
    assert _should_publish("mock") is False


def test_publica_odds_reales_y_previas():
    """live (odds en vivo) y previous (odds reales de hace horas) sí se publican."""
    assert _should_publish("live") is True
    assert _should_publish("previous") is True

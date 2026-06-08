"""Tests de la lógica de ventana del scheduler (con catch-up)."""

from datetime import datetime, timedelta, timezone

import pytest

import src.agent.scheduler as sched
from src.agent.pipeline import Phase

NOW = datetime(2026, 6, 11, 12, 0, 0, tzinfo=timezone.utc)


def _fixtures(kickoff: datetime) -> dict:
    return {
        "fase_grupos": [{"id": "M1", "kickoff_utc": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ")}],
        "eliminatorias": [],
    }


@pytest.fixture
def no_runs(monkeypatch):
    """Ninguna fase corrió todavía."""
    monkeypatch.setattr(sched, "phase_already_ran", lambda mid, ph: False)


def test_t24h_emits_at_target(no_runs):
    out = sched.matches_in_window(_fixtures(NOW + timedelta(hours=24)), now=NOW)
    assert out == [("M1", Phase.T_24H)]


def test_t3h_is_latest_when_past_t24h(no_runs):
    # 3h antes del kickoff: target de T-24h ya pasó → la pasada relevante es T-3h
    out = sched.matches_in_window(_fixtures(NOW + timedelta(hours=3)), now=NOW)
    assert out == [("M1", Phase.T_3H)]


def test_catchup_emits_t30min_close_to_kickoff(no_runs):
    # 20 min antes del kickoff: ya pasamos el target de T-30min → debe emitir T-30min
    out = sched.matches_in_window(_fixtures(NOW + timedelta(minutes=20)), now=NOW)
    assert out == [("M1", Phase.T_30MIN)]


def test_missed_window_still_publishes(no_runs):
    # 25 min antes del kickoff = 5 min PASADO el target de T-30min (fuera del viejo ±2.5).
    # Con catch-up igual dispara T-30min (la que publica). Este es el fix.
    out = sched.matches_in_window(_fixtures(NOW + timedelta(minutes=25)), now=NOW)
    assert out == [("M1", Phase.T_30MIN)]


def test_no_rerun_and_no_fallback_when_latest_ran(monkeypatch):
    # T-30min YA corrió. Aunque T-3h/T-24h no hayan corrido, NO volvemos atrás.
    monkeypatch.setattr(sched, "phase_already_ran", lambda mid, ph: ph == Phase.T_30MIN)
    out = sched.matches_in_window(_fixtures(NOW + timedelta(minutes=20)), now=NOW)
    assert out == []


def test_nothing_after_kickoff(no_runs):
    out = sched.matches_in_window(_fixtures(NOW - timedelta(minutes=1)), now=NOW)
    assert out == []


def test_nothing_before_first_target(no_runs):
    # Faltan 30h → ni el target de T-24h llegó. No corre nada.
    out = sched.matches_in_window(_fixtures(NOW + timedelta(hours=30)), now=NOW)
    assert out == []


def test_one_phase_per_match_per_run(no_runs):
    # Cerca del kickoff, varias fases "due" pero se emite UNA sola (la más relevante).
    out = sched.matches_in_window(_fixtures(NOW + timedelta(minutes=20)), now=NOW)
    assert len(out) == 1

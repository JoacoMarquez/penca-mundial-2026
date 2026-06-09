"""Tests de la lógica de ventana del scheduler (con catch-up)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

import src.agent.scheduler as sched
from src.agent.pipeline import Phase

NOW = datetime(2026, 6, 11, 12, 0, 0, tzinfo=timezone.utc)


def _write_pred(base, match_id, version, phase, run_at):
    md = base / match_id
    md.mkdir(parents=True, exist_ok=True)
    fname = f"v{version}_{run_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    (md / fname).write_text(json.dumps({"phase": phase, "run_at": run_at.isoformat()}))


KICKOFF = datetime(2026, 6, 11, 19, 0, 0, tzinfo=timezone.utc)
STALE = datetime(2026, 5, 28, 0, 0, 0, tzinfo=timezone.utc)     # >72h antes del kickoff (test viejo)
RECENT = datetime(2026, 6, 11, 18, 30, 0, tzinfo=timezone.utc)  # dentro de la ventana real


def test_stale_phase_does_not_count(tmp_path, monkeypatch):
    """Una pasada vieja (test de mayo) NO debe marcar la fase como 'ya corrió'."""
    monkeypatch.setattr(sched, "PREDICTIONS_DIR", tmp_path)
    _write_pred(tmp_path, "105", 1, "T_30min", STALE)
    assert sched.phase_already_ran("105", Phase.T_30MIN, KICKOFF) is False


def test_recent_phase_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(sched, "PREDICTIONS_DIR", tmp_path)
    _write_pred(tmp_path, "105", 1, "T_30min", RECENT)
    assert sched.phase_already_ran("105", Phase.T_30MIN, KICKOFF) is True


def test_inaugural_match_not_blocked_by_stale_prediction(tmp_path, monkeypatch):
    """Escenario real: el 105 tiene un T_30min de prueba de mayo. NO debe saltearse la
    publicación del partido real del 11/6."""
    monkeypatch.setattr(sched, "PREDICTIONS_DIR", tmp_path)
    _write_pred(tmp_path, "105", 1, "T_30min", STALE)
    fx = {"fase_grupos": [{"id": "105", "kickoff_utc": "2026-06-11T19:00:00Z"}], "eliminatorias": []}
    now = datetime(2026, 6, 11, 18, 40, 0, tzinfo=timezone.utc)  # T-20min, target T-30 ya pasó
    out = sched.matches_in_window(fx, now=now)
    assert ("105", Phase.T_30MIN) in out  # la pasada REAL sí dispara, pese al archivo viejo


def _fixtures(kickoff: datetime) -> dict:
    return {
        "fase_grupos": [{"id": "M1", "kickoff_utc": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ")}],
        "eliminatorias": [],
    }


@pytest.fixture
def no_runs(monkeypatch):
    """Ninguna fase corrió todavía."""
    monkeypatch.setattr(sched, "phase_already_ran", lambda mid, ph, ko=None: False)


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
    monkeypatch.setattr(sched, "phase_already_ran", lambda mid, ph, ko=None: ph == Phase.T_30MIN)
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

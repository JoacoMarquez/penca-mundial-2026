"""Tests de:
  - P1: reintento barato de publicación (republish_pending + _retry_pending_publications).
  - P2: no publicar cuando la asignación cayó a e_max degradado (_is_degraded_emax).
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.agent import pipeline, scheduler
from src.agent.pipeline import _is_degraded_emax, republish_pending


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(pipeline, "_best_effort_alert", lambda *a, **k: None)
    monkeypatch.delenv("PENCA_BLOCK_DEGRADED_EMAX", raising=False)
    return tmp_path


def _make_snapshot(tmp_path):
    d = tmp_path / "pool_snapshots"
    d.mkdir(parents=True, exist_ok=True)
    (d / "185.json").write_text('{"entries": []}')


LATEST = {
    "phase": "T_3h",
    "published": False,
    "version": 2,
    "assignment": [
        {"penca_id": 1651, "score": [1, 0], "objective": "ev", "rank": 1},
        {"penca_id": 1652, "score": [2, 1], "objective": "tail", "rank": 2},
    ],
    "constraints": {"p_home": 0.5},
    "assignment_meta": {"objective": "p_top_k", "threshold": 206},
}


# ---------------- P2: _is_degraded_emax ----------------

def test_degraded_emax_blocks_when_no_cutoff_and_snapshots(tmp_data):
    _make_snapshot(tmp_data)
    assert _is_degraded_emax({"objective": "e_max", "threshold": None}) is True


def test_degraded_emax_allows_first_matchday_no_snapshots(tmp_data):
    # sin snapshots → es la 1ª fecha genuina sin pool → e_max legítimo, se publica
    assert _is_degraded_emax({"objective": "e_max", "threshold": None}) is False


def test_intentional_emax_ptopk0_not_blocked(tmp_data):
    _make_snapshot(tmp_data)
    # el e_max intencional de eliminatorias tiene threshold → NO se bloquea
    assert _is_degraded_emax({"objective": "e_max (P(top-K)=0)", "threshold": 206}) is False


def test_ptopk_not_blocked(tmp_data):
    _make_snapshot(tmp_data)
    assert _is_degraded_emax({"objective": "p_top_k", "threshold": 206}) is False


def test_degraded_emax_disabled_by_env(tmp_data, monkeypatch):
    monkeypatch.setenv("PENCA_BLOCK_DEGRADED_EMAX", "false")
    _make_snapshot(tmp_data)
    assert _is_degraded_emax({"objective": "e_max", "threshold": None}) is False


def test_degraded_emax_none_meta(tmp_data):
    assert _is_degraded_emax(None) is False


# ---------------- P1: republish_pending ----------------

def test_republish_success_dry_run_writes_new_version(tmp_data, monkeypatch):
    # DRY_RUN → NullPublisher → publish ok=True
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("PENCA_API_BASE_URL", "")
    monkeypatch.setenv("PENCA_API_KEY", "")
    md = pipeline.PREDICTIONS_DIR / "185"
    md.mkdir(parents=True)
    (md / "v1_a.json").write_text("{}")
    (md / "v2_a.json").write_text(json.dumps(LATEST))

    r = republish_pending({"id": 185, "home_name": "A", "away_name": "B"}, LATEST)
    assert r is True
    newest = json.loads(sorted(md.glob("v*_*.json"))[-1].read_text())
    assert newest["published"] is True
    assert newest["publish_retry"] is True
    assert newest["assignment"] == LATEST["assignment"]  # misma asignación, no recomputada


def test_republish_failure_writes_nothing(tmp_data, monkeypatch):
    monkeypatch.setattr(pipeline, "_publish_assignment", lambda *a, **k: (False, "500"))
    md = pipeline.PREDICTIONS_DIR / "185"
    md.mkdir(parents=True)
    (md / "v2_a.json").write_text(json.dumps(LATEST))
    before = len(list(md.glob("v*_*.json")))

    r = republish_pending({"id": 185, "home_name": "A", "away_name": "B"}, LATEST)
    assert r is False
    assert len(list(md.glob("v*_*.json"))) == before  # no versionó nada


def test_republish_empty_assignment_returns_none(tmp_data):
    r = republish_pending({"id": 185}, {"phase": "T_3h", "published": False, "assignment": []})
    assert r is None


# ---------------- P1: _retry_pending_publications (scheduler) ----------------

def test_retry_pending_only_future_unpublished(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(scheduler, "PREDICTIONS_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(pipeline, "republish_pending", lambda m, latest: calls.append(m["id"]))

    now = datetime.now(timezone.utc)
    future = (now + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    past = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    fx = {"eliminatorias": [
        {"id": 100, "kickoff_utc": future},   # futuro + published=False → reintenta
        {"id": 101, "kickoff_utc": future},   # futuro + published=True  → skip
        {"id": 102, "kickoff_utc": past},     # ya arrancó → skip
        {"id": 103, "kickoff_utc": future},   # futuro + published=None (MOCK/degradado) → skip
    ]}
    for mid, pub in [(100, False), (101, True), (102, False), (103, None)]:
        d = tmp_path / str(mid)
        d.mkdir()
        (d / "v1_a.json").write_text(json.dumps(
            {"phase": "T_3h", "published": pub, "assignment": [{"penca_id": 1, "score": [1, 0]}]}
        ))

    scheduler._retry_pending_publications(fx)
    assert calls == [100]

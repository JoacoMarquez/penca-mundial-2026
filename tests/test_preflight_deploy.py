"""Tests del pre-check de deploy (ventana de publicación)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.preflight_deploy import PRE_MIN, blocked_match

NOW = datetime(2026, 6, 25, 0, 30, 0, tzinfo=timezone.utc)


def _match(mid, ko):
    return {"id": mid, "kickoff_utc": ko}


def test_block_when_match_in_publish_window():
    # kickoff a las 01:00Z, now 00:30Z → 30 min antes → dentro de [-40m, +5m]
    matches = [_match("MATCH_A_05", "2026-06-25T01:00:00Z")]
    hit = blocked_match(matches, NOW)
    assert hit is not None
    assert hit[0] == "MATCH_A_05"


def test_allow_when_no_match_near():
    matches = [_match("MATCH_A_05", "2026-06-25T19:00:00Z")]  # faltan ~18h
    assert blocked_match(matches, NOW) is None


def test_allow_after_window_closes():
    # kickoff hace 10 min (now ya pasó kickoff + POST_MIN=5) → libre
    matches = [_match("MATCH_X", (NOW - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"))]
    assert blocked_match(matches, NOW) is None


def test_boundary_exactly_at_pre_edge_blocks():
    # now exactamente en kickoff − PRE_MIN → incluido (borde inferior)
    ko = (NOW + timedelta(minutes=PRE_MIN)).isoformat().replace("+00:00", "Z")
    assert blocked_match([_match("EDGE", ko)], NOW) is not None


def test_simultaneous_matches_first_hit_returned():
    # caso real jornada 3: dos partidos al mismo kickoff, basta con uno para bloquear
    matches = [
        _match("MATCH_A_05", "2026-06-25T01:00:00Z"),
        _match("MATCH_A_06", "2026-06-25T01:00:00Z"),
    ]
    hit = blocked_match(matches, NOW)
    assert hit is not None
    assert hit[0] in {"MATCH_A_05", "MATCH_A_06"}


def test_malformed_kickoff_is_skipped():
    matches = [_match("BAD", "no-es-fecha"), _match("MISSING", None)]
    assert blocked_match(matches, NOW) is None

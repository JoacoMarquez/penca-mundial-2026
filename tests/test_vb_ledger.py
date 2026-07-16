"""Tests de src/valuebet/ledger.py — round-trip, settlement, CLV, bankroll."""

from src.valuebet import ledger
from src.valuebet.types import Suggestion


def _sug(sid="abc123", stake=50.0, odds=2.2, mode="paper", status="open"):
    return Suggestion(
        id=sid, ts_utc="2026-07-16T12:00:00+00:00", mode=mode,
        legs=[{
            "quote": {"book": "supermatch", "sport": "soccer", "league": "l",
                      "event_id": "e1", "event_name": "A vs B",
                      "start_utc": "2026-07-17T12:00:00+00:00", "market": "1x2",
                      "outcome": "home", "decimal_odds": odds,
                      "fetched_utc": "2026-07-16T12:00:00+00:00"},
            "fair_prob": 0.50, "sharp_odds": 2.05, "edge": 0.10,
            "segment": "soccer|1x2|2.0-3.0",
        }],
        combined_odds=odds, combined_fair_prob=0.50, edge=0.10,
        stake_suggested=stake, bankroll_at_ts=1000.0, kelly_fraction_used=0.25,
        status=status,
    )


def test_roundtrip_versionado(tmp_path):
    p1 = ledger.append([_sug("a1")], tmp_path)
    p2 = ledger.append([_sug("a2")], tmp_path)
    assert p1 != p2 and p1.exists() and p2.exists()  # versiona, no sobreescribe
    all_s = ledger.load_all(tmp_path)
    assert [s.id for s in all_s] == ["a1", "a2"]
    assert all_s[0].legs[0]["quote"]["decimal_odds"] == 2.2


def test_settle_won_lost_void():
    s = ledger.settle(_sug(stake=50.0, odds=2.2), "won")
    assert abs(s.pnl - 60.0) < 1e-9
    s = ledger.settle(_sug(stake=50.0), "lost")
    assert s.pnl == -50.0
    s = ledger.settle(_sug(stake=50.0), "void")
    assert s.pnl == 0.0


def test_settle_resultado_invalido():
    try:
        ledger.settle(_sug(), "ganamo")
        raise AssertionError("debió tirar ValueError")
    except ValueError:
        pass


def test_clv():
    s = _sug(odds=2.2)
    ledger.apply_closing(s, [{
        "outcome_key": "e1|1x2|home", "closing_odds": 2.0,
        "closing_fair_prob": 0.52, "clv": 2.2 * 0.52 - 1,
    }])
    # clv = cuota tomada 2.2 * fair al cierre 0.52 - 1 = +14.4%
    assert abs(s.clv - 0.144) < 1e-9
    assert s.closing_fair_prob == 0.52


def test_clv_cierre_incompleto_no_computa():
    s = _sug()
    ledger.apply_closing(s, [{"outcome_key": "e1|1x2|home", "closing_odds": None,
                              "closing_fair_prob": None, "clv": None}])
    assert s.clv is None


def test_dedup():
    a = _sug("a1")
    keys = {"e1|1x2|home"}
    assert ledger.is_duplicate(keys, [a], 0.02, candidate_edge=0.11)       # +1pt < +2pts
    assert not ledger.is_duplicate(keys, [a], 0.02, candidate_edge=0.13)   # +3pts > +2pts
    assert not ledger.is_duplicate({"e2|1x2|home"}, [a], 0.02, 0.10)       # otro evento


def test_open_exposure_por_modo():
    a = _sug("a1", stake=50.0, mode="paper")
    b = _sug("b1", stake=30.0, mode="real")
    b.taken = True
    b.stake_real = 40.0
    total, parlay = ledger.open_exposure([a, b], "paper")
    assert total == 50.0 and parlay == 0.0
    total, parlay = ledger.open_exposure([a, b], "real")
    assert total == 40.0  # usa stake_real cuando taken


def test_bankroll_roundtrip(tmp_path):
    p = tmp_path / "bankroll.json"
    ledger.save_bankroll(1000.0, p)
    assert ledger.load_bankroll(p) == 1000.0

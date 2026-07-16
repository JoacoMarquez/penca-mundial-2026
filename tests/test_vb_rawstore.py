"""Tests de src/valuebet/rawstore.py — snapshots crudos por scan + retención."""

import json
from datetime import datetime, timedelta, timezone

from src.valuebet.rawstore import prune, save_snapshot
from src.valuebet.types import OddsQuote


def _q(odds=2.0):
    return OddsQuote(book="supermatch", sport="soccer", league="l", event_id="sm:1",
                     event_name="A vs B", start_utc="2026-07-17T12:00:00+00:00",
                     market="1x2", outcome="home", decimal_odds=odds,
                     fetched_utc="2026-07-16T12:00:00+00:00")


def test_save_snapshot_escribe_json(tmp_path):
    path = save_snapshot("supermatch", [_q(), _q(1.8)], base=tmp_path)
    assert path is not None and path.exists()
    rows = json.loads(path.read_text())
    assert len(rows) == 2
    assert rows[0]["event_name"] == "A vs B"
    assert rows[0]["decimal_odds"] == 2.0
    # layout: {base}/{book}/{YYYY-MM-DD}/{HHMMSS}.json
    assert path.parent.parent.name == "supermatch"


def test_prune_borra_dias_viejos(tmp_path):
    old = tmp_path / "supermatch" / "2020-01-01"
    old.mkdir(parents=True)
    (old / "120000.json").write_text("[]")
    recent_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    recent = tmp_path / "supermatch" / recent_date
    recent.mkdir(parents=True)
    (recent / "120000.json").write_text("[]")

    prune(base=tmp_path, days=30)
    assert not old.exists()
    assert recent.exists()

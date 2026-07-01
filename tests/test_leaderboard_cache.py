"""Tests del fetch de leaderboard con cache a disco (stale-while-error).

El endpoint /leaderboard cuelga a veces (timeout server-side). Estos tests fijan que ni el
dashboard ni el pipeline queden ciegos: el último leaderboard bueno se sirve stale desde
disco, con fallback al snapshot de pool más reciente y un techo de antigüedad.
"""

import json
import sys
import time
import types

import pytest

from src.utils import leaderboard as lb


ENTRIES = [
    {"penca_id": 1, "penca_name": "A", "points_total": 30},
    {"penca_id": 2, "penca_name": "B", "points_total": 20},
    {"penca_id": 3, "penca_name": "C", "points_total": 10},
]


@pytest.fixture(autouse=True)
def _tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


def _fake_httpx(status=200, payload=None, raises=None):
    """Módulo httpx falso para inyectar en sys.modules (deps reales no instaladas en CI local)."""
    mod = types.ModuleType("httpx")

    class _Resp:
        status_code = status

        def json(self):
            return payload or {}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            if raises:
                raise raises
            return _Resp()

    mod.Client = _Client
    return mod


def test_success_writes_cache_and_not_stale(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", _fake_httpx(payload={"entries": ENTRIES}))
    res = lb.fetch_leaderboard("http://api", "key")
    assert res["stale"] is False
    assert res["error"] is None
    assert len(res["entries"]) == 3
    # persistió a disco
    assert lb._cache_path().exists()
    cached = json.loads(lb._cache_path().read_text())
    assert cached["entries"] == ENTRIES


def test_api_down_serves_stale_from_cache(monkeypatch):
    lb._write_cache(ENTRIES)  # simula un fetch bueno previo
    monkeypatch.setitem(sys.modules, "httpx", _fake_httpx(raises=RuntimeError("timeout")))
    res = lb.fetch_leaderboard("http://api", "key")
    assert res["stale"] is True
    assert res["error"] == "timeout"
    assert len(res["entries"]) == 3
    assert res["age_seconds"] is not None


def test_api_down_no_config_serves_stale(monkeypatch):
    lb._write_cache(ENTRIES)
    # base/key vacíos → ni siquiera intenta la red, pero igual sirve el cache stale
    res = lb.fetch_leaderboard("", "")
    assert res["stale"] is True
    assert res["entries"] == ENTRIES


def test_snapshot_fallback_when_no_cache(_tmp_data_dir):
    # sin leaderboard_cache.json pero con un snapshot de pool → se usa el snapshot
    sdir = _tmp_data_dir / "pool_snapshots"
    sdir.mkdir()
    (sdir / "m50.json").write_text(json.dumps({"entries": ENTRIES}))
    res = lb.fetch_leaderboard("", "")
    assert res["stale"] is True
    assert len(res["entries"]) == 3


def test_max_stale_cap_drops_ancient_cache(monkeypatch, _tmp_data_dir):
    # cache con timestamp viejo (2h) y cap de 1h → no se usa
    old = time.time() - 2 * 3600
    lb._cache_path().write_text(json.dumps({"fetched_at": old, "entries": ENTRIES}))
    res = lb.fetch_leaderboard("", "", max_stale_seconds=3600)
    assert res["entries"] == []
    assert res["stale"] is False


def test_max_stale_cap_allows_recent_cache(monkeypatch, _tmp_data_dir):
    recent = time.time() - 600  # 10 min
    lb._cache_path().write_text(json.dumps({"fetched_at": recent, "entries": ENTRIES}))
    res = lb.fetch_leaderboard("", "", max_stale_seconds=3600)
    assert res["stale"] is True
    assert len(res["entries"]) == 3


def test_no_cache_no_snapshot_returns_empty():
    res = lb.fetch_leaderboard("", "")
    assert res["entries"] == []
    assert res["stale"] is False
    assert res["error"] == "API no configurada"


def test_empty_api_response_falls_back_to_cache(monkeypatch):
    lb._write_cache(ENTRIES)
    # la API responde 200 pero sin entries → tratamos como fallo y servimos stale
    monkeypatch.setitem(sys.modules, "httpx", _fake_httpx(payload={"entries": []}))
    res = lb.fetch_leaderboard("http://api", "key")
    assert res["stale"] is True
    assert len(res["entries"]) == 3

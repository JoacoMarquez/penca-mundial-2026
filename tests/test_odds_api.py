"""Tests del scraper The-Odds-API (segunda fuente de odds). Sin llamadas de red."""

from __future__ import annotations

import json
import time

from src.scrapers.odds_api import (
    _build_alias_map,
    extract_consensus_markets,
    fetch_world_cup_odds,
    map_game_to_match_id,
)


def _game(home, away, commence="2026-06-12T19:00:00Z", h_prices=(2.0,), d_prices=(3.0,), a_prices=(4.0,), totals=None):
    books = []
    for i in range(max(len(h_prices), len(d_prices), len(a_prices))):
        mkts = [{"key": "h2h", "outcomes": [
            {"name": home, "price": h_prices[i % len(h_prices)]},
            {"name": "Draw", "price": d_prices[i % len(d_prices)]},
            {"name": away, "price": a_prices[i % len(a_prices)]},
        ]}]
        if totals:
            mkts.append({"key": "totals", "outcomes": [
                {"name": "Over", "point": 2.5, "price": totals[0]},
                {"name": "Under", "point": 2.5, "price": totals[1]},
            ]})
        books.append({"key": f"book{i}", "markets": mkts})
    return {"home_team": home, "away_team": away, "commence_time": commence, "bookmakers": books}


# ---------- extracción / mediana ----------

def test_extract_consensus_takes_median():
    g = _game("Mexico", "South Korea",
              h_prices=(1.8, 2.0, 2.2),   # mediana 2.0
              d_prices=(3.0, 3.2, 3.4),   # mediana 3.2
              a_prices=(4.0, 4.5, 5.0),   # mediana 4.5
              totals=(1.9, 1.95))
    mk = extract_consensus_markets(g)
    assert mk["1x2"]["H"] == 2.0
    assert mk["1x2"]["D"] == 3.2
    assert mk["1x2"]["A"] == 4.5
    assert mk["ou_2_5"] == {"over": 1.9, "under": 1.95}


def test_extract_omits_missing_markets():
    g = _game("A", "B", totals=None)
    mk = extract_consensus_markets(g)
    assert "1x2" in mk
    assert "ou_2_5" not in mk


def test_extract_totals_picks_line_near_2_5():
    g = _game("A", "B")
    g["bookmakers"][0]["markets"].append({"key": "totals", "outcomes": [
        {"name": "Over", "point": 3.5, "price": 2.5},   # lejos de 2.5 → ignorar
        {"name": "Under", "point": 3.5, "price": 1.5},
    ]})
    mk = extract_consensus_markets(g)
    assert "ou_2_5" not in mk


# ---------- mapeo a match_id ----------

FIXTURES = {
    "fase_grupos": [
        {"id": 106, "home": "KOR", "away": "CZE", "kickoff_utc": "2026-06-12T02:00:00Z"},
        {"id": 107, "home": "CAN", "away": "BIH", "kickoff_utc": "2026-06-12T19:00:00Z"},
    ],
    "eliminatorias": [],
}
TEAMS = {
    "aliases": {
        "CAN": ["Canada"], "BIH": ["Bosnia & Herzegovina", "Bosnia and Herzegovina"],
        "KOR": ["South Korea"], "CZE": ["Czech Republic", "Czechia"],
    },
    "groups": {},
}


def test_map_game_to_match_id_by_codes():
    amap = _build_alias_map(TEAMS)
    g = _game("Canada", "Bosnia & Herzegovina")
    assert map_game_to_match_id(g, FIXTURES, amap) == 107


def test_map_handles_ampersand_alias():
    amap = _build_alias_map(TEAMS)
    assert amap["bosnia & herzegovina"] == "BIH"


def test_map_unknown_team_returns_none():
    amap = _build_alias_map(TEAMS)
    g = _game("Atlantis", "Wakanda")
    assert map_game_to_match_id(g, FIXTURES, amap) is None


def test_map_respects_home_away_orientation():
    """CZE vs KOR (invertido) no debe mapear al 106 (KOR vs CZE)."""
    amap = _build_alias_map(TEAMS)
    g = _game("Czech Republic", "South Korea")
    assert map_game_to_match_id(g, FIXTURES, amap) is None


# ---------- caché + guarda de presupuesto ----------

def test_fresh_cache_avoids_network(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    cache = tmp_path / "raw" / "odds_api" / "world_cup_latest.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({
        "fetched_at": time.time(), "remaining": 400, "games": [{"home_team": "X"}]}))
    # con caché fresca y sin forzar, devuelve la caché sin tocar la red
    games = fetch_world_cup_odds(api_key="fake", ttl_seconds=9999)
    assert games == [{"home_team": "X"}]


def test_budget_floor_blocks_refresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    cache = tmp_path / "raw" / "odds_api" / "world_cup_latest.json"
    cache.parent.mkdir(parents=True)
    # caché vieja (TTL vencido) pero con créditos por debajo del piso → no refresca,
    # devuelve la caché vieja en vez de gastar
    cache.write_text(json.dumps({
        "fetched_at": time.time() - 10_000, "remaining": 10, "games": [{"cached": True}]}))
    games = fetch_world_cup_odds(api_key="fake", ttl_seconds=60, budget_floor=40)
    assert games == [{"cached": True}]


def test_no_key_returns_none(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert fetch_world_cup_odds(api_key="") is None

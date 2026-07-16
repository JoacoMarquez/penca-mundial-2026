"""Pinnacle guest API, multi-deporte — fork parametrizado de src/scrapers/pinnacle.py.

FORK deliberado (no import): el scraper penca tiene SPORT_SOCCER y el normalizador 1X2
hardcodeados y no lo modificamos. Acá: sport ids configurables + normalizador que
soporta mercados 3-way (fútbol) y 2-way (tenis/básquet).

NOTA: Pinnacle bloquea redes uruguayas — correr desde el VPS NYC.
Smoke tests:  python -m src.valuebet.books.pinnacle_vb --list-sports
              python -m src.valuebet.books.pinnacle_vb --list-leagues soccer
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.valuebet.config import VBConfig
from src.valuebet.types import OddsQuote

log = logging.getLogger(__name__)

PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
# Public guest key del frontend de pinnacle.com (misma que usa el scraper penca; no es secret).
PINNACLE_API_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"

HEADERS = {
    "X-API-Key": PINNACLE_API_KEY,
    "Referer": "https://www.pinnacle.com/",
    "Origin": "https://www.pinnacle.com",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

# Deportes SIN empate en el mercado principal (moneyline 2-way)
TWO_WAY_SPORTS = {"tennis", "basketball"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{PINNACLE_BASE}{path}"
    with httpx.Client(timeout=15.0, headers=HEADERS) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def american_to_decimal(odds: float) -> float:
    if odds == 0:
        return 0.0
    if odds > 0:
        return odds / 100.0 + 1.0
    return 100.0 / abs(odds) + 1.0


def list_sports() -> list[dict]:
    return _get("/sports")


def list_leagues(sport_id: int) -> list[dict]:
    return _get(f"/sports/{sport_id}/leagues")


def get_matchups(league_id: int) -> list[dict]:
    return _get(f"/leagues/{league_id}/matchups")


def get_markets(league_id: int) -> list[dict]:
    return _get(f"/leagues/{league_id}/markets/straight", params={"primaryOnly": "false"})


# -------------------- normalización a OddsQuote --------------------

def normalize_league(
    sport: str,
    league_id: int,
    matchups_raw: list[dict],
    markets_raw: list[dict],
) -> list[OddsQuote]:
    """Matchups + markets crudos de una liga → lista plana de OddsQuote."""
    now = datetime.now(timezone.utc).isoformat()
    quotes: list[OddsQuote] = []

    matchup_info: dict[int, dict] = {}
    for m in matchups_raw:
        if m.get("type") != "matchup":
            continue
        parts = m.get("participants", [])
        home = next((p["name"] for p in parts if p.get("alignment") == "home"), None)
        away = next((p["name"] for p in parts if p.get("alignment") == "away"), None)
        if not (home and away):
            continue
        matchup_info[m["id"]] = {
            "event_name": f"{home} vs {away}",
            "start_utc": m["startTime"],
            "league": m.get("league", {}).get("name", str(league_id)),
        }

    for market in markets_raw:
        mid = market.get("matchupId")
        info = matchup_info.get(mid)
        if info is None or market.get("period", 0) != 0:
            continue

        prices = {
            p["designation"]: american_to_decimal(p["price"])
            for p in market.get("prices", [])
            if p.get("price") is not None
        }
        mtype = market.get("type")
        normalized: dict[str, float] = {}
        market_name = None

        if mtype == "moneyline":
            if sport in TWO_WAY_SPORTS:
                market_name = "moneyline"
                normalized = {"home": prices.get("home", 0.0), "away": prices.get("away", 0.0)}
            else:
                market_name = "1x2"
                normalized = {
                    "home": prices.get("home", 0.0),
                    "draw": prices.get("draw", 0.0),
                    "away": prices.get("away", 0.0),
                }
        elif mtype == "total":
            points = market.get("points")
            if points is None:
                continue
            market_name = f"total_{points}"
            normalized = {"over": prices.get("over", 0.0), "under": prices.get("under", 0.0)}
        # correct score llega por /markets/special — Fase 2

        if not market_name or any(v <= 1.0 for v in normalized.values()):
            continue
        for outcome, dec in normalized.items():
            quotes.append(OddsQuote(
                book="pinnacle", sport=sport, league=info["league"],
                event_id=f"pinn:{mid}", event_name=info["event_name"],
                start_utc=info["start_utc"], market=market_name,
                outcome=outcome, decimal_odds=dec, fetched_utc=now,
            ))
    return quotes


def fetch_quotes(cfg: VBConfig) -> list[OddsQuote]:
    """Todas las cuotas sharp de las ligas configuradas, todos los deportes."""
    out: list[OddsQuote] = []
    for sport in cfg.sports:
        for league_id in cfg.leagues.get(sport, []):
            try:
                matchups = get_matchups(league_id)
                markets = get_markets(league_id)
                out.extend(normalize_league(sport, league_id, matchups, markets))
            except Exception as e:
                log.warning("pinnacle %s liga %s falló: %s", sport, league_id, e)
    return out


# -------------------- CLI smoke test (correr desde el VPS) --------------------

if __name__ == "__main__":
    import argparse
    import json as _json

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-sports", action="store_true")
    ap.add_argument("--list-leagues", metavar="SPORT")
    args = ap.parse_args()

    if args.list_sports:
        for s in list_sports():
            print(f"  {s.get('id'):>4}  {s.get('name')}")
    elif args.list_leagues:
        from src.valuebet import config as vbconfig
        cfg = vbconfig.load()
        sid = cfg.sport_ids[args.list_leagues]
        leagues = list_leagues(sid)
        leagues.sort(key=lambda l: -(l.get("matchupCount") or 0))
        for l in leagues[:40]:
            print(f"  {l.get('id'):>6}  [{l.get('matchupCount', 0):>3} matchups]  {l.get('name')}")
    else:
        print(_json.dumps(list_sports(), indent=1)[:2000])

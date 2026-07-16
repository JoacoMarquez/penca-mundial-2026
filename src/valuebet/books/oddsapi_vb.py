"""The Odds API v4, multi-deporte — fork parametrizado de src/scrapers/odds_api.py.

FORK deliberado (no import): el scraper penca tiene SPORT_KEY hardcodeado al Mundial.
Conserva el skeleton probado: caché disco + TTL + piso de presupuesto (header
x-requests-remaining del free tier de 500 créditos/mes).

Roles en valuebet:
    1. Fallback de soft book si Supermatch está caído/inescrapeable (bookies soft
       de región eu incluidos en la respuesta).
    2. Fuente barata de closing lines para CLV cuando el evento arranca fuera del
       horario de scan (Pinnacle directo es gratis, pero desde Uruguay está
       bloqueado — en el VPS no, así que este rol es secundario).

Mapa de sport keys: https://the-odds-api.com/sports-odds-data/sports-apis.html
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.valuebet.config import VBConfig
from src.valuebet.types import OddsQuote, total_market

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
CACHE_TTL_DEFAULT = 90 * 60
BUDGET_FLOOR_DEFAULT = 40

# sport → sport_keys de The Odds API a consultar (ampliable por config futura)
SPORT_KEYS: dict[str, list[str]] = {
    "soccer": ["soccer_uefa_champs_league", "soccer_epl", "soccer_spain_la_liga",
               "soccer_conmebol_copa_libertadores"],
    "basketball": ["basketball_nba"],
    "tennis": ["tennis_atp_us_open", "tennis_wta_us_open"],
}

MARKET_MAP = {"h2h": None, "totals": None}  # h2h → 1x2/moneyline según deporte


def _cache_path(sport_key: str) -> Path:
    d = Path(os.environ.get("VALUEBET_DATA_DIR", "data/valuebet"))
    return d / "raw" / "odds_api" / f"{sport_key}_latest.json"


def fetch_sport_odds(
    sport_key: str,
    api_key: str | None = None,
    regions: str = "eu",
    markets: str = "h2h,totals",
    ttl_seconds: int = CACHE_TTL_DEFAULT,
    budget_floor: int = BUDGET_FLOOR_DEFAULT,
    force: bool = False,
) -> list[dict] | None:
    """Games con odds para un sport_key (cacheado). None si no hay key/falla."""
    api_key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return None

    cache = _cache_path(sport_key)
    cached = _read_cache(cache)
    now = time.time()

    if not force and cached and (now - cached.get("fetched_at", 0)) < ttl_seconds:
        return cached.get("games")
    if cached and cached.get("remaining") is not None and cached["remaining"] < budget_floor:
        log.warning("Odds-API: créditos bajos (%s < %s) — sirvo caché stale de %s",
                    cached["remaining"], budget_floor, sport_key)
        return cached.get("games")

    try:
        with httpx.Client(timeout=20.0) as c:
            r = c.get(
                f"{BASE_URL}/sports/{sport_key}/odds",
                params={"apiKey": api_key, "regions": regions,
                        "markets": markets, "oddsFormat": "decimal"},
            )
        remaining = int(r.headers.get("x-requests-remaining", -1))
        if r.status_code != 200:
            log.warning("Odds-API %d (%s): %s", r.status_code, sport_key, r.text[:200])
            return cached.get("games") if cached else None
        games = r.json()
        _write_cache(cache, {"fetched_at": now, "remaining": remaining, "games": games})
        log.info("Odds-API %s: %d juegos | créditos: %d", sport_key, len(games), remaining)
        return games
    except Exception as e:
        log.warning("Odds-API fetch falló (%s): %s", sport_key, e)
        return cached.get("games") if cached else None


def _read_cache(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_cache(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except Exception as e:
        log.warning("no pude escribir caché Odds-API: %s", e)


# -------------------- normalización a OddsQuote --------------------

def normalize_games(sport: str, sport_key: str, games: list[dict], bookie_filter: str | None = None) -> list[OddsQuote]:
    """Games de The Odds API → OddsQuote. Si bookie_filter, solo esa casa; si no,
    la MEJOR cuota soft por outcome (max entre casas — lo que un apostador puede tomar)."""
    now = datetime.now(timezone.utc).isoformat()
    out: list[OddsQuote] = []
    three_way = sport == "soccer"

    for g in games:
        event_name = f"{g.get('home_team')} vs {g.get('away_team')}"
        best: dict[tuple[str, str], tuple[float, str]] = {}  # (market, outcome) -> (odds, bookie)
        for bm in g.get("bookmakers", []):
            if bookie_filter and bm.get("key") != bookie_filter:
                continue
            for mk in bm.get("markets", []):
                if mk["key"] == "h2h":
                    market = "1x2" if three_way else "moneyline"
                    for oc in mk.get("outcomes", []):
                        name = oc["name"]
                        if name == g.get("home_team"):
                            outcome = "home"
                        elif name == g.get("away_team"):
                            outcome = "away"
                        elif name == "Draw":
                            outcome = "draw"
                        else:
                            continue
                        _keep_best(best, (market, outcome), oc["price"], bm["key"])
                elif mk["key"] == "totals":
                    for oc in mk.get("outcomes", []):
                        point = oc.get("point")
                        if point is None:
                            continue
                        market = total_market(point)
                        outcome = oc["name"].lower()  # Over/Under
                        _keep_best(best, (market, outcome), oc["price"], bm["key"])

        for (market, outcome), (odds, bookie) in best.items():
            out.append(OddsQuote(
                book=f"oddsapi:{bookie}", sport=sport, league=sport_key,
                event_id=f"oapi:{g['id']}", event_name=event_name,
                start_utc=g["commence_time"], market=market,
                outcome=outcome, decimal_odds=odds, fetched_utc=now,
            ))
    return out


def _keep_best(best: dict, key: tuple, odds: float, bookie: str) -> None:
    if odds and odds > 1.0 and (key not in best or odds > best[key][0]):
        best[key] = (odds, bookie)


def fetch_quotes(cfg: VBConfig) -> list[OddsQuote]:
    """Rol fallback: mejores cuotas soft por outcome para los deportes configurados."""
    out: list[OddsQuote] = []
    for sport in cfg.sports:
        for sk in SPORT_KEYS.get(sport, []):
            games = fetch_sport_odds(sk)
            if games:
                out.extend(normalize_games(sport, sk, games))
    return out

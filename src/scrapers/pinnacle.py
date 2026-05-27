"""Scraper de Pinnacle vía API guest pública (no oficial pero estable).

Endpoint base: https://guest.api.arcadia.pinnacle.com/0.1
Auth: X-API-Key header con la clave guest pública.

Mercados que extraemos:
    1X2 (moneyline)        — key 'm' o 's;0;m'
    Over/Under Totals      — key 't'
    BTTS                   — key 'g' (goals)
    Asian Handicap         — key 's'
    Correct Score          — key 'cs' (no siempre disponible)

NOTA: Pinnacle puede estar bloqueado en algunas redes (Uruguay incluida en muchos casos).
Desde el VPS NYC funciona sin problema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)


PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"

# Esta es la public guest API key — se ve en cualquier request del frontend de pinnacle.com.
# No es un secret personal, es de la app guest.
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

SPORT_SOCCER = 29


# -------------------- cliente bajo --------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{PINNACLE_BASE}{path}"
    with httpx.Client(timeout=15.0, headers=HEADERS) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def list_soccer_leagues() -> list[dict]:
    """Devuelve todas las ligas de fútbol disponibles en Pinnacle."""
    return _get(f"/sports/{SPORT_SOCCER}/leagues")


def find_world_cup_league_id() -> int | None:
    """Busca el league_id del Mundial 2026 por nombre.

    Pinnacle típicamente lo lista como 'FIFA World Cup' o similar.
    Históricamente el id de WorldCup ha sido 1980 pero puede cambiar por edición.
    """
    leagues = list_soccer_leagues()
    candidates = [
        l for l in leagues
        if any(kw in l.get("name", "").lower() for kw in ["world cup", "mundial"])
        and "u-" not in l.get("name", "").lower()   # excluir sub-X
        and "women" not in l.get("name", "").lower()  # excluir femenino
    ]
    if not candidates:
        return None
    # preferir el que tenga "2026" en el nombre si lo hay
    for c in candidates:
        if "2026" in c.get("name", ""):
            return c["id"]
    return candidates[0]["id"]


def get_matchups(league_id: int) -> list[dict]:
    return _get(f"/leagues/{league_id}/matchups")


def get_markets(league_id: int) -> list[dict]:
    """Markets straight (no específicos) para todos los matchups de la liga."""
    return _get(f"/leagues/{league_id}/markets/straight", params={"primaryOnly": "true"})


def get_special_markets(league_id: int) -> list[dict]:
    """Markets especiales: correct score, etc. (cuando disponibles)."""
    try:
        return _get(f"/leagues/{league_id}/markets/special")
    except httpx.HTTPStatusError:
        return []


# -------------------- normalización por partido --------------------

@dataclass(frozen=True)
class PinnacleMatch:
    matchup_id: int
    home: str
    away: str
    start_time_utc: str
    league_name: str


def parse_matchups(matchups_raw: list[dict]) -> list[PinnacleMatch]:
    out = []
    for m in matchups_raw:
        if m.get("type") != "matchup":   # excluir parlays etc
            continue
        parts = m.get("participants", [])
        home = next((p["name"] for p in parts if p.get("alignment") == "home"), None)
        away = next((p["name"] for p in parts if p.get("alignment") == "away"), None)
        if not (home and away):
            continue
        out.append(PinnacleMatch(
            matchup_id=m["id"],
            home=home,
            away=away,
            start_time_utc=m["startTime"],
            league_name=m.get("league", {}).get("name", ""),
        ))
    return out


def extract_match_markets(
    matchup_id: int,
    markets_raw: list[dict],
) -> dict[str, dict[str, float]]:
    """Para un matchup_id, devuelve dict { mercado_nombre → { outcome → odds } }.

    Mercados normalizados:
        1x2:     {"H": odds, "D": odds, "A": odds}
        ou_2_5:  {"over": odds, "under": odds}
        btts:    {"yes": odds, "no": odds}
        asian_handicap_0: {"H": odds, "A": odds}  (handicap 0, equivalente a draw-no-bet)
    """
    out: dict[str, dict[str, float]] = {}

    for market in markets_raw:
        if market.get("matchupId") != matchup_id:
            continue
        mtype = market.get("type")
        period = market.get("period", 0)
        if period != 0:
            continue  # solo full-game

        prices = {p["designation"]: p["price"] for p in market.get("prices", [])}

        if mtype == "moneyline":
            out["1x2"] = {
                "H": prices.get("home", 0.0),
                "D": prices.get("draw", 0.0),
                "A": prices.get("away", 0.0),
            }
        elif mtype == "total":
            points = market.get("points")
            if points is None:
                continue
            if abs(points - 2.5) < 1e-6:
                out["ou_2_5"] = {
                    "over": prices.get("over", 0.0),
                    "under": prices.get("under", 0.0),
                }
            elif abs(points - 3.5) < 1e-6:
                out["ou_3_5"] = {
                    "over": prices.get("over", 0.0),
                    "under": prices.get("under", 0.0),
                }
        elif mtype == "spread":
            points = market.get("points")
            if points is None:
                continue
            if abs(points) < 1e-6:
                out["asian_handicap_0"] = {
                    "H": prices.get("home", 0.0),
                    "A": prices.get("away", 0.0),
                }
        elif mtype == "team_total":
            # opcional: totales por equipo
            pass

    return out


# -------------------- mapeo a match_id de fixtures.yaml --------------------

def map_to_match_id(
    pinnacle_match: PinnacleMatch,
    fixtures: dict,
    team_aliases: dict[str, list[str]],
) -> str | None:
    """Identifica el match_id de fixtures.yaml que corresponde a un PinnacleMatch.

    Estrategia: matchear teams via aliases + fecha del partido (±24h).
    """
    home_code = _resolve_team_code(pinnacle_match.home, team_aliases)
    away_code = _resolve_team_code(pinnacle_match.away, team_aliases)
    if not home_code or not away_code:
        log.warning("No pude resolver códigos: home=%r away=%r", pinnacle_match.home, pinnacle_match.away)
        return None

    from datetime import datetime, timedelta
    pinn_kickoff = datetime.fromisoformat(pinnacle_match.start_time_utc.replace("Z", "+00:00"))

    for m in fixtures.get("fase_grupos", []) + fixtures.get("eliminatorias", []):
        if m.get("home") != home_code or m.get("away") != away_code:
            continue
        fix_kickoff = datetime.fromisoformat(m["kickoff_utc"].replace("Z", "+00:00"))
        if abs((fix_kickoff - pinn_kickoff).total_seconds()) < 24 * 3600:
            return m["id"]
    return None


def _resolve_team_code(name: str, aliases: dict[str, list[str]]) -> str | None:
    n = name.strip().lower()
    for code, names in aliases.items():
        if n == code.lower() or n in [x.lower() for x in names]:
            return code
    return None


# -------------------- CLI smoke test --------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("› Buscando Mundial 2026 en ligas de Pinnacle…")
    league_id = find_world_cup_league_id()
    print(f"  league_id = {league_id}")
    if not league_id:
        raise SystemExit(1)

    print("\n› Matchups:")
    matchups = parse_matchups(get_matchups(league_id))
    for m in matchups[:5]:
        print(f"  {m.matchup_id}  {m.home} vs {m.away}  ({m.start_time_utc})")

    print(f"\n› {len(matchups)} matchups totales")

    if matchups:
        print(f"\n› Markets para {matchups[0].home} vs {matchups[0].away}:")
        markets_raw = get_markets(league_id)
        m_data = extract_match_markets(matchups[0].matchup_id, markets_raw)
        for mkt, prices in m_data.items():
            print(f"  {mkt}: {prices}")

"""The-Odds-API — segunda fuente de odds (redundancia + consenso multi-libro).

Una sola llamada trae TODOS los partidos del Mundial con odds de ~31 casas. Tomamos la
MEDIANA por outcome (robusta a outliers) → un "libro consenso" que entra en la Capa 1
junto a Pinnacle. Si Pinnacle cae, este es el fallback; si los dos están, se promedian.

Presupuesto: el tier free son 500 créditos/mes y cada llamada cuesta (#regiones × #mercados).
Por eso:
  1. Caché en disco con TTL — todas las pasadas/partidos dentro de la ventana comparten
     una sola llamada.
  2. Guarda de presupuesto — si los créditos restantes (que la API reporta en cada
     respuesta) bajan del umbral, dejamos de llamar y caemos a Pinnacle solo.

Doc: https://the-odds-api.com/liveapi/guides/v4/
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

SPORT_KEY = "soccer_fifa_world_cup"
BASE_URL = "https://api.the-odds-api.com/v4"
CACHE_TTL_DEFAULT = 90 * 60          # 90 min
BUDGET_FLOOR_DEFAULT = 40            # si quedan menos créditos, no llamamos más


def _cache_path() -> Path:
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    return data_dir / "raw" / "odds_api" / "world_cup_latest.json"


def fetch_world_cup_odds(
    api_key: str | None = None,
    regions: str = "eu,us",
    markets: str = "h2h,totals",
    ttl_seconds: int = CACHE_TTL_DEFAULT,
    budget_floor: int = BUDGET_FLOOR_DEFAULT,
    force: bool = False,
) -> list[dict] | None:
    """Devuelve los partidos del Mundial con odds (cacheado). None si no hay key/falla.

    Estrategia: si la caché es fresca (< ttl), la usa. Si no, intenta refrescar — salvo
    que el presupuesto esté por debajo del piso, en cuyo caso devuelve la caché vieja
    (mejor odds de hace rato que nada) o None.
    """
    api_key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return None

    cache = _cache_path()
    cached = _read_cache(cache)
    now = time.time()

    if not force and cached and (now - cached.get("fetched_at", 0)) < ttl_seconds:
        return cached.get("games")

    # Guarda de presupuesto: si la última respuesta dejó pocos créditos, no gastamos más.
    if cached and cached.get("remaining") is not None and cached["remaining"] < budget_floor:
        log.warning(
            "Odds-API: créditos bajos (%s < %s) — uso caché de hace %.0f min, no refresco",
            cached["remaining"], budget_floor, (now - cached.get("fetched_at", 0)) / 60,
        )
        return cached.get("games")

    try:
        with httpx.Client(timeout=20.0) as c:
            r = c.get(
                f"{BASE_URL}/sports/{SPORT_KEY}/odds",
                params={"apiKey": api_key, "regions": regions,
                        "markets": markets, "oddsFormat": "decimal"},
            )
        remaining = int(r.headers.get("x-requests-remaining", -1))
        if r.status_code != 200:
            log.warning("Odds-API %d: %s", r.status_code, r.text[:200])
            return cached.get("games") if cached else None
        games = r.json()
        _write_cache(cache, {"fetched_at": now, "remaining": remaining, "games": games})
        log.info("Odds-API: %d partidos | créditos restantes: %d", len(games), remaining)
        return games
    except Exception as e:
        log.warning("Odds-API fetch falló: %s", e)
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


# ---------- extracción de mercados (mediana multi-libro) ----------

def _median(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x and x > 1.0)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def extract_consensus_markets(game: dict) -> dict[str, dict[str, float]]:
    """Mediana de odds decimales por outcome, sobre todas las casas del partido.

    Devuelve {"1x2": {"H","D","A"}, "ou_2_5": {"over","under"}} en formato odds_by_book.
    Outcomes faltantes se omiten (el de-vig / agregación los tolera).
    """
    home, away = game.get("home_team"), game.get("away_team")
    h, d, a = [], [], []
    over, under = [], []
    for bk in game.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] == "h2h":
                for o in mk["outcomes"]:
                    if o["name"] == home:
                        h.append(o["price"])
                    elif o["name"] == away:
                        a.append(o["price"])
                    elif o["name"] == "Draw":
                        d.append(o["price"])
            elif mk["key"] == "totals":
                # quedarse con la línea más cercana a 2.5
                for o in mk["outcomes"]:
                    if abs(o.get("point", 99) - 2.5) <= 0.25:
                        if o["name"] == "Over":
                            over.append(o["price"])
                        elif o["name"] == "Under":
                            under.append(o["price"])

    out: dict[str, dict[str, float]] = {}
    mh, md, ma = _median(h), _median(d), _median(a)
    if mh and md and ma:
        out["1x2"] = {"H": mh, "D": md, "A": ma}
    mo, mu = _median(over), _median(under)
    if mo and mu:
        out["ou_2_5"] = {"over": mo, "under": mu}
    return out


# ---------- mapeo a nuestros match_id ----------

def _build_alias_map(teams_data: dict) -> dict[str, str]:
    """{nombre_lower: code} desde aliases de teams.yaml."""
    out: dict[str, str] = {}
    for code, aliases in (teams_data.get("aliases") or {}).items():
        for a in (aliases if isinstance(aliases, list) else [aliases]):
            out[str(a).lower()] = code
    # también el nombre canónico de groups
    for _grp, teams in (teams_data.get("groups") or {}).items():
        for t in teams:
            if t.get("name") and t.get("code"):
                out[t["name"].lower()] = t["code"]
    return out


def map_game_to_match_id(game: dict, fixtures: dict, alias_map: dict[str, str]) -> Any | None:
    """Mapea un partido de Odds-API a un id de fixtures por códigos de equipo.

    Empareja home/away (en ese orden) y, si hay ambigüedad, por cercanía de kickoff.
    """
    hc = alias_map.get((game.get("home_team") or "").lower())
    ac = alias_map.get((game.get("away_team") or "").lower())
    if not hc or not ac:
        return None
    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    cands = [m for m in all_matches if m.get("home") == hc and m.get("away") == ac]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]["id"]
    # desempate por kickoff más cercano
    from datetime import datetime
    try:
        gt = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
        def _dist(m):
            try:
                return abs((datetime.fromisoformat(m["kickoff_utc"].replace("Z", "+00:00")) - gt).total_seconds())
            except Exception:
                return 1e18
        return min(cands, key=_dist)["id"]
    except Exception:
        return cands[0]["id"]


def markets_for_match(match_id: Any, fixtures: dict, teams_data: dict,
                      games: list[dict] | None = None) -> dict[str, dict[str, float]] | None:
    """Atajo: odds consenso de Odds-API para un match_id nuestro (o None)."""
    games = games if games is not None else fetch_world_cup_odds()
    if not games:
        return None
    alias_map = _build_alias_map(teams_data)
    for g in games:
        if str(map_game_to_match_id(g, fixtures, alias_map)) == str(match_id):
            mk = extract_consensus_markets(g)
            return mk or None
    return None

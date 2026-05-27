"""Cliente de API-Football (api-sports.io v3) para contexto de partidos.

Datos que obtenemos:
    - Lineups confirmadas / probables (T-3h, T-30min)
    - Lesionados por equipo
    - Forma reciente (últimos N partidos)
    - Head-to-head

Plan FREE: 100 req/día. Nuestro uso esperado:
    Por partido × 3 pasadas (T-24h, T-3h, T-30min) × ~5 endpoints = ~15 req
    104 partidos en 39 días = ~40 req/día promedio. Holgado dentro del límite.

NOTA: a la fecha (2026-05-27), los fixtures del Mundial 2026 todavía NO están cargados.
El módulo gracefully retorna {} si no hay data; cuando los fixtures aparezcan, automáticamente
empieza a alimentar Capa 4.
"""

from __future__ import annotations

import logging
import os
import unicodedata
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

import httpx

log = logging.getLogger(__name__)


API_BASE = "https://v3.football.api-sports.io"
WORLD_CUP_LEAGUE_ID = 1
WORLD_CUP_SEASON = 2026


def _key() -> str | None:
    k = os.environ.get("API_FOOTBALL_KEY", "")
    return k if k and not k.startswith("xxx") else None


def _get(path: str, params: dict[str, Any] | None = None) -> dict | None:
    key = _key()
    if not key:
        log.debug("API_FOOTBALL_KEY no configurada — skip")
        return None
    try:
        with httpx.Client(
            timeout=15.0,
            headers={"x-apisports-key": key},
        ) as c:
            r = c.get(f"{API_BASE}{path}", params=params or {})
        if r.status_code != 200:
            log.warning("API-Football %s → %d: %s", path, r.status_code, r.text[:200])
            return None
        return r.json()
    except Exception as e:
        log.warning("API-Football %s falló: %s", path, e)
        return None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


@lru_cache(maxsize=128)
def _team_name_aliases(team_name: str) -> set[str]:
    """Aliases conocidos para matchear nombres entre nuestras fuentes."""
    norm = _norm(team_name)
    return {norm, norm.replace("-", " "), norm.replace(" ", "")}


def find_fixture_for_match(
    home_name: str,
    away_name: str,
    kickoff_utc: datetime,
) -> dict | None:
    """Busca el fixture en API-Football que matchea con (home, away, kickoff ±1 día).

    Returns: dict del fixture o None si no se encuentra.
    """
    # API-Football: el filtro from/to combinado con season+league no anda bien.
    # Estrategia: consultar 3 días (ayer, hoy, mañana relativos al kickoff) por separado.
    candidates = []
    for delta_days in (-1, 0, 1):
        d = (kickoff_utc + timedelta(days=delta_days)).date().isoformat()
        data = _get("/fixtures", {
            "league": WORLD_CUP_LEAGUE_ID,
            "date": d,
        })
        if data and data.get("results", 0) > 0:
            candidates.extend(data.get("response", []))

    if not candidates:
        return None

    home_aliases = _team_name_aliases(home_name)
    away_aliases = _team_name_aliases(away_name)

    for r in candidates:
        teams = r["teams"]
        h_norm = _norm(teams["home"]["name"])
        a_norm = _norm(teams["away"]["name"])
        h_match = any(alias in h_norm or h_norm in alias for alias in home_aliases)
        a_match = any(alias in a_norm or a_norm in alias for alias in away_aliases)
        if h_match and a_match:
            return r
    return None


def get_team_injuries(team_id: int, season: int = WORLD_CUP_SEASON) -> list[dict]:
    """Lesionados de un equipo en una temporada. Devuelve lista cruda del endpoint."""
    data = _get("/injuries", {"team": team_id, "season": season})
    if not data:
        return []
    return data.get("response", [])


def get_team_recent_form(team_id: int, last_n: int = 5) -> list[dict]:
    """Últimos N partidos del equipo (cualquier competencia)."""
    data = _get("/fixtures", {"team": team_id, "last": last_n})
    if not data:
        return []
    return data.get("response", [])


def get_fixture_lineups(fixture_id: int) -> list[dict]:
    """Lineups (confirmadas) del partido. Disponibles típicamente ~1h antes del kickoff."""
    data = _get("/lineups", {"fixture": fixture_id})
    if not data:
        return []
    return data.get("response", [])


def get_head_to_head(team1_id: int, team2_id: int, last_n: int = 5) -> list[dict]:
    """H2H entre dos equipos."""
    data = _get("/fixtures/headtohead", {"h2h": f"{team1_id}-{team2_id}", "last": last_n})
    if not data:
        return []
    return data.get("response", [])


# ============ Agregación de contexto para Capa 4 ============

def collect_match_context(
    home_name: str,
    away_name: str,
    kickoff_utc: datetime,
    fetch_lineups: bool = False,
) -> dict:
    """Junta todo el contexto disponible para un partido. Lo pasa a Capa 4 LLM.

    Args:
        fetch_lineups: solo True en T-3h y T-30min (cuando ya hay lineups disponibles).

    Returns:
        dict con secciones agregadas. Strings legibles (no IDs raw) para que el LLM las use.
    """
    fixture = find_fixture_for_match(home_name, away_name, kickoff_utc)
    if not fixture:
        log.info("API-Football: no encontró fixture para %s vs %s en %s",
                 home_name, away_name, kickoff_utc.date())
        return {}

    fid = fixture["fixture"]["id"]
    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]
    home_actual = fixture["teams"]["home"]["name"]
    away_actual = fixture["teams"]["away"]["name"]

    out: dict[str, Any] = {
        "fixture_id": fid,
        "home_team_id": home_id,
        "away_team_id": away_id,
    }

    # Lesiones
    home_inj = get_team_injuries(home_id)
    away_inj = get_team_injuries(away_id)
    if home_inj:
        out["home_injuries"] = [
            f"{i['player']['name']} ({i.get('player', {}).get('reason') or 'lesión'})"
            for i in home_inj[:5]
        ]
    if away_inj:
        out["away_injuries"] = [
            f"{i['player']['name']} ({i.get('player', {}).get('reason') or 'lesión'})"
            for i in away_inj[:5]
        ]

    # Forma reciente
    home_form = get_team_recent_form(home_id, last_n=5)
    away_form = get_team_recent_form(away_id, last_n=5)
    if home_form:
        out["home_recent_form"] = _summarize_form(home_form, home_id)
    if away_form:
        out["away_recent_form"] = _summarize_form(away_form, away_id)

    # H2H
    h2h = get_head_to_head(home_id, away_id, last_n=5)
    if h2h:
        out["h2h_recent"] = _summarize_h2h(h2h, home_actual, away_actual)

    # Lineups (solo en T-3h / T-30min)
    if fetch_lineups:
        lineups = get_fixture_lineups(fid)
        if lineups:
            for lu in lineups:
                team_id = lu["team"]["id"]
                key = "home_lineup_change" if team_id == home_id else "away_lineup_change"
                formation = lu.get("formation", "?")
                coach = lu.get("coach", {}).get("name", "?")
                out[key] = f"Formación: {formation} (DT: {coach})"

    return out


def _summarize_form(fixtures: list[dict], team_id: int) -> str:
    """Convierte últimos 5 partidos en string tipo 'WWLDL (4-2 GF/GC)'."""
    results = []
    gf = gc = 0
    for f in fixtures:
        home_id = f["teams"]["home"]["id"]
        h_g = f["goals"]["home"]
        a_g = f["goals"]["away"]
        if h_g is None or a_g is None:
            continue
        if home_id == team_id:
            my, their = h_g, a_g
        else:
            my, their = a_g, h_g
        gf += my
        gc += their
        results.append("W" if my > their else ("D" if my == their else "L"))
    if not results:
        return "(sin data reciente)"
    return f"{''.join(results)} ({gf} GF / {gc} GC)"


def _summarize_h2h(fixtures: list[dict], home_name: str, away_name: str) -> str:
    """Resumen breve de los últimos enfrentamientos."""
    lines = []
    for f in fixtures[:3]:
        date = f["fixture"]["date"][:10]
        h, a = f["teams"]["home"]["name"], f["teams"]["away"]["name"]
        gh, ga = f["goals"]["home"], f["goals"]["away"]
        if gh is None or ga is None:
            continue
        lines.append(f"{h} {gh}-{ga} {a} ({date})")
    return "; ".join(lines) if lines else "(sin h2h reciente)"

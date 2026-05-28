"""Cliente de la API pública de ESPN para data de partidos.

Endpoints públicos, sin auth, sin anti-bot:
    /apis/site/v2/sports/soccer/fifa.world/scoreboard
    /apis/site/v2/sports/soccer/fifa.world/summary?event={id}
    /apis/site/v2/sports/soccer/fifa.world/teams
    /apis/site/v2/sports/soccer/fifa.world/teams/{id}
    /apis/site/v2/sports/soccer/fifa.world/teams/{id}/roster

Data útil:
    - Squad/roster de cada equipo (50+ jugadores con posición)
    - H2H histórico
    - News articles per partido
    - Odds DraftKings (otro bookmaker para combinar con Pinnacle)
    - Venue info (Estadio Banorte = Azteca renombrado por sponsor)

Limitaciones:
    - Lineups específicos pre-match: solo cuando los publica el equipo (~1h antes kickoff)
    - Injuries por jugador: no expuesto (usar Google News para eso)
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

import httpx

log = logging.getLogger(__name__)


ESPN_BASE_TEMPLATE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}"
DEFAULT_LEAGUE = "fifa.world"
# Variable global mutable: permite override para testing en otras ligas
_LEAGUE_CODE = DEFAULT_LEAGUE


def set_league(code: str) -> None:
    """Override del league code (ej: 'conmebol.libertadores' para testing)."""
    global _LEAGUE_CODE
    _LEAGUE_CODE = code
    _all_teams_cached.cache_clear()


def get_league() -> str:
    return _LEAGUE_CODE


def _get(path: str, params: dict[str, Any] | None = None) -> dict | None:
    ESPN_BASE = ESPN_BASE_TEMPLATE.format(league=_LEAGUE_CODE)
    try:
        with httpx.Client(timeout=10.0, headers={"User-Agent": "PencaMundial/1.0"}) as c:
            r = c.get(f"{ESPN_BASE}{path}", params=params or {})
        if r.status_code != 200:
            log.warning("ESPN %s → %d", path, r.status_code)
            return None
        return r.json()
    except Exception as e:
        log.warning("ESPN %s falló: %s", path, e)
        return None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


# ============ scoreboard / events ============

def get_scoreboard() -> list[dict]:
    """Próximos partidos del Mundial 2026."""
    data = _get("/scoreboard")
    return data.get("events", []) if data else []


@lru_cache(maxsize=1)
def _load_team_aliases() -> dict[str, list[str]]:
    """Lee config/teams.yaml para resolver nombres en español → inglés (que usa ESPN)."""
    try:
        from pathlib import Path
        import yaml
        path = Path(__file__).resolve().parents[2] / "config" / "teams.yaml"
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text()) or {}
        return data.get("aliases", {})
    except Exception as e:
        log.warning("No se pudo cargar teams.yaml aliases: %s", e)
        return {}


def _team_aliases(team_name: str) -> set[str]:
    """Para un team_name (cualquier idioma), retorna todos los aliases conocidos."""
    aliases = _load_team_aliases()
    n_target = _norm(team_name)
    out = {n_target}
    for code, names in aliases.items():
        if n_target == _norm(code) or n_target in (_norm(x) for x in names):
            out.add(_norm(code))
            for n in names:
                out.add(_norm(n))
            break
    return out


def find_event(home_name: str, away_name: str) -> dict | None:
    """Busca event por nombres de equipos. ESPN usa formato 'Away at Home' en inglés."""
    events = get_scoreboard()
    if not events:
        return None
    home_set = _team_aliases(home_name)
    away_set = _team_aliases(away_name)
    for e in events:
        name_norm = _norm(e.get("name", ""))
        h_match = any(alias in name_norm for alias in home_set if len(alias) >= 3)
        a_match = any(alias in name_norm for alias in away_set if len(alias) >= 3)
        if h_match and a_match:
            return e
    return None


def get_match_summary(event_id: str | int) -> dict | None:
    """Detalle completo del partido (boxscore, h2h, news, odds, rosters, etc)."""
    return _get("/summary", {"event": str(event_id)})


# ============ teams ============

@lru_cache(maxsize=1)
def _all_teams_cached() -> list[dict]:
    """Cachea la lista de 48 equipos del Mundial (cambia ~nunca)."""
    data = _get("/teams")
    if not data:
        return []
    leagues = data.get("sports", [{}])[0].get("leagues", [{}])
    return leagues[0].get("teams", []) if leagues else []


def find_team_id(team_name: str) -> int | None:
    """ESPN team ID para un nombre de equipo (case + accent insensitive)."""
    teams = _all_teams_cached()
    n_target = _norm(team_name)
    for entry in teams:
        info = entry.get("team", {})
        if (n_target == _norm(info.get("displayName", ""))
                or n_target == _norm(info.get("shortDisplayName", ""))
                or n_target == _norm(info.get("name", ""))):
            return int(info["id"])
    return None


def get_team_roster(team_id: int) -> list[dict]:
    """Squad de un equipo. Para el Mundial son los 23-26 convocados."""
    data = _get(f"/teams/{team_id}/roster")
    if not data:
        return []
    return data.get("athletes", [])


# ============ contexto agregado para Capa 4 ============

@dataclass(frozen=True)
class EspnMatchContext:
    event_id: str
    venue: str | None
    home_id: int | None
    away_id: int | None
    home_squad_names: list[str]
    away_squad_names: list[str]
    news_headlines: list[str]
    draftkings_odds: dict | None
    h2h_summary: str | None


def collect_match_context_espn(home_name: str, away_name: str) -> dict[str, Any]:
    """API de alto nivel — retorna dict compatible con MatchContext del LLM.

    Returns keys:
        - home_squad / away_squad: lista de jugadores convocados
        - news_summary: titulares ESPN
        - h2h_recent: resumen h2h
        - draftkings_summary: odds DraftKings como referencia
        - venue
    """
    event = find_event(home_name, away_name)
    if not event:
        return {}

    event_id = event["id"]
    summary = get_match_summary(event_id)
    if not summary:
        return {}

    out: dict[str, Any] = {"espn_event_id": event_id}

    # Venue
    game_info = summary.get("gameInfo", {})
    venue = game_info.get("venue", {}).get("fullName")
    if venue:
        out["venue"] = venue

    # Rosters de cada equipo (si están disponibles)
    rosters = summary.get("rosters", [])
    for r in rosters:
        team = r.get("team", {})
        roster_players = r.get("roster", [])
        if not roster_players:
            continue
        names = [p.get("athlete", {}).get("displayName") for p in roster_players[:25] if p.get("athlete")]
        names = [n for n in names if n]
        if _norm(team.get("displayName", "")) == _norm(home_name):
            out["home_squad_listed"] = ", ".join(names)
        else:
            out["away_squad_listed"] = ", ".join(names)

    # News (ESPN devuelve dict con "articles" key, no lista directa)
    news = summary.get("news", {})
    articles = news.get("articles", []) if isinstance(news, dict) else (news if isinstance(news, list) else [])
    if articles:
        headlines = []
        for n in articles[:5]:
            if isinstance(n, dict):
                h = n.get("headline") or n.get("title")
                desc = n.get("description") or ""
                if h:
                    line = f"[ESPN] {h}"
                    if desc:
                        line += f"  →  {desc[:150]}"
                    headlines.append(line)
        if headlines:
            out["espn_news"] = "\n".join(headlines)

    # H2H games
    h2h = summary.get("headToHeadGames", [])
    h2h_lines = []
    for group in h2h:
        if isinstance(group, dict) and "events" in group:
            for ev in group["events"][:3]:
                name = ev.get("name", "")
                date = ev.get("date", "")[:10]
                if name and date:
                    h2h_lines.append(f"{date}: {name}")
    if h2h_lines:
        out["h2h_recent_espn"] = " · ".join(h2h_lines[:3])

    # DraftKings odds — extraer ML home/away/draw + spread + OU
    pickcenter = summary.get("pickcenter", [])
    if pickcenter:
        p = pickcenter[0]
        ml_home = p.get('homeTeamOdds', {}).get('moneyLine')
        ml_away = p.get('awayTeamOdds', {}).get('moneyLine')
        # Buscar draw en el detalle de odds nested
        odds_list = summary.get("odds", [])
        ml_draw = None
        for od in odds_list:
            d_odds = od.get("drawOdds") or {}
            ml_draw = d_odds.get("moneyLine") or ml_draw

        if ml_home and ml_away:
            out["draftkings_odds"] = {
                "ml_home_american": ml_home,
                "ml_away_american": ml_away,
                "ml_draw_american": ml_draw,
                "spread": p.get('spread'),
                "over_under": p.get('overUnder'),
            }
            out["draftkings_summary"] = (
                f"spread={p.get('spread')}, OU={p.get('overUnder')}, "
                f"ML home/away/draw={ml_home}/{ml_away}/{ml_draw or '?'}"
            )

    # Standings del torneo (útil post-jornada para contexto motivacional)
    try:
        std = get_standings()
        if std:
            home_norm = _norm(home_name)
            away_norm = _norm(away_name)
            relevant_lines = []
            for group, entries in std.items():
                in_group = any(
                    _norm(e["team"]) == home_norm or _norm(e["team"]) == away_norm
                    for e in entries
                )
                if in_group:
                    relevant_lines.append(f"{group}:")
                    for e in entries:
                        marker = "→ " if _norm(e["team"]) in (home_norm, away_norm) else "  "
                        relevant_lines.append(
                            f"  {marker}{e['team']:<25} {e['w']}W-{e['d']}D-{e['l']}L  {e['points']}pts (GF {e['gf']}-{e['ga']} GA)"
                        )
                    break
            if relevant_lines:
                out["standings_context"] = "\n".join(relevant_lines)
    except Exception as e:
        log.warning("get_standings falló: %s", e)

    # Forma reciente (últimos 5)
    home_form = get_team_form_from_summary(summary, home_name)
    away_form = get_team_form_from_summary(summary, away_name)
    if home_form:
        out["home_recent_form_espn"] = home_form
    if away_form:
        out["away_recent_form_espn"] = away_form

    return out


def get_standings() -> dict[str, list[dict]] | None:
    """Tabla del torneo. Retorna {group_name: [{team, points, w, d, l, gd, ...}, ...]}.

    Útil para: 'ya clasificó / necesita ganar' como contexto para Capa 4.
    """
    data = _get("/standings")
    if not data:
        return None
    out: dict[str, list[dict]] = {}
    children = data.get("children", []) if isinstance(data, dict) else []
    for c in children:
        group_name = c.get("name", "?")
        std = c.get("standings", {})
        rows = []
        for entry in std.get("entries", []):
            team = entry.get("team", {})
            stats = {s.get("name"): s.get("value") for s in entry.get("stats", []) if isinstance(s, dict)}
            rows.append({
                "team": team.get("displayName"),
                "team_id": team.get("id"),
                "points": stats.get("points", 0),
                "w": stats.get("wins", 0),
                "d": stats.get("ties", 0),
                "l": stats.get("losses", 0),
                "gf": stats.get("pointsFor", 0),
                "ga": stats.get("pointsAgainst", 0),
                "rank": stats.get("rank", 0),
            })
        out[group_name] = rows
    return out


def get_team_form_from_summary(summary: dict, team_name: str) -> str | None:
    """Extrae 'forma reciente' del boxscore.form del summary endpoint.

    boxscore.form tiene los últimos partidos de cada equipo. Devuelve string tipo 'WWLDW'.
    """
    if not summary:
        return None
    boxscore = summary.get("boxscore", {})
    form_data = boxscore.get("form", [])
    n_target = _norm(team_name)
    for entry in form_data:
        if not isinstance(entry, dict):
            continue
        team = entry.get("team", {})
        if _norm(team.get("displayName", "")) != n_target and _norm(team.get("abbreviation", "")) != n_target:
            continue
        events = entry.get("events", [])
        if not events:
            return None
        results = []
        for ev in events[:5]:
            r = ev.get("gameResult")  # 'W' | 'L' | 'D' o similar
            if r:
                results.append(r[0])
        return "".join(results) if results else None
    return None


def get_team_squad(team_name: str) -> list[str] | None:
    """Squad list de un equipo (nombres). Útil si rosters del match no están poblados."""
    team_id = find_team_id(team_name)
    if not team_id:
        return None
    roster = get_team_roster(team_id)
    if not roster:
        return None
    names = []
    for a in roster:
        name = a.get("displayName")
        if name:
            pos = a.get("position", {}).get("abbreviation", "?")
            names.append(f"{name} ({pos})")
    return names[:30]

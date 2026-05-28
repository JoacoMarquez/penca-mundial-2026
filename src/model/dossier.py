"""Match Dossier — ficha estructurada por partido.

Consolida toda la info dispersa (Pinnacle, ESPN, Google News, weather, fixtures, teams.yaml)
en un objeto único que sirve como:
    1. Input estructurado al LLM Capa 4 (mejor razonamiento que texto crudo).
    2. Audit trail persistido en data/dossiers/{match_id}/v{N}.json.
    3. Display en Telegram (sección "Ficha del partido").
    4. Detección de info_edge_flags (mismatches entre nuestras señales y el mercado).

NO compite con Pinnacle como predictor — solo enriquece el contexto.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TeamProfile:
    name: str
    code: str | None = None              # FIFA 3-letter
    confederation: str | None = None
    elo_seed: int | None = None

    # Forma reciente
    recent_form: str | None = None       # ej "WWLDW" últimos 5
    last_n_gf: int | None = None         # goles a favor últimos N
    last_n_ga: int | None = None         # goles en contra últimos N

    # Squad
    squad_size: int | None = None
    squad_listed_count: int | None = None
    key_players_listed: list[str] = field(default_factory=list)

    # Lesiones/bajas (de noticias)
    reported_absences: list[str] = field(default_factory=list)
    notable_news_headlines: list[str] = field(default_factory=list)

    # Standings (cuando aplica)
    standings_rank: int | None = None
    standings_points: int | None = None
    standings_gf: int | None = None
    standings_ga: int | None = None
    qualification_status: str | None = None   # "qualified" | "in contention" | "eliminated"


@dataclass
class MatchDossier:
    # Identidad
    match_id: str | int
    match_label: str
    kickoff_utc: str
    kickoff_local_uy: str
    stage: str
    group: str | None = None

    # Sede / contexto
    venue_name: str | None = None
    venue_city: str | None = None
    venue_country: str | None = None
    venue_elevation_m: int | None = None
    venue_indoor: bool | None = None
    home_advantage_status: str = "home"      # "home" | "away" | "neutral"

    # Equipos
    home: TeamProfile = field(default_factory=lambda: TeamProfile(name="?"))
    away: TeamProfile = field(default_factory=lambda: TeamProfile(name="?"))

    # Match-specific
    rest_days_home: int | None = None
    rest_days_away: int | None = None
    importance_score: float | None = None    # 0..1 (eliminatoria > grupos > amistoso)

    # H2H
    h2h_summary: str | None = None
    h2h_count: int | None = None

    # Market consensus
    market_p_home: float | None = None
    market_p_draw: float | None = None
    market_p_away: float | None = None
    market_e_goals_home: float | None = None
    market_e_goals_away: float | None = None
    market_sources: list[str] = field(default_factory=list)
    draftkings_odds: dict | None = None

    # Clima
    weather_summary: str | None = None
    weather_details: dict | None = None

    # Detección de info edge: dónde nuestras señales chocan con el mercado
    info_edge_flags: list[str] = field(default_factory=list)

    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sources_used: list[str] = field(default_factory=list)


# ============ builders ============

def _kickoff_uy(kickoff_iso: str) -> str:
    dt = datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00"))
    dt_uy = dt - timedelta(hours=3)
    return dt_uy.strftime("%a %d/%m %H:%M UY")


def _parse_form_for_gf_ga(form: str | None, recent_results: list[dict] | None) -> tuple[int | None, int | None]:
    """Si tenemos resultados crudos, calculamos GF/GA. Si solo tenemos 'WWLDW', no."""
    if not recent_results:
        return (None, None)
    gf = sum(r.get("goals_scored", 0) for r in recent_results[:5])
    ga = sum(r.get("goals_conceded", 0) for r in recent_results[:5])
    return (gf, ga)


def _compute_rest_days(
    team_code_or_name: str,
    target_kickoff_utc: datetime,
    all_fixtures: list[dict],
) -> int | None:
    """Días desde el último partido del equipo antes del target.

    Útil en eliminatorias y cuando hay calendarios apretados.
    """
    if not all_fixtures:
        return None
    prev_dates = []
    for m in all_fixtures:
        ko = m.get("kickoff_utc")
        if not ko:
            continue
        ko_dt = datetime.fromisoformat(ko.replace("Z", "+00:00"))
        if ko_dt >= target_kickoff_utc:
            continue
        teams = (m.get("home"), m.get("away"), m.get("home_name"), m.get("away_name"))
        if team_code_or_name in (t for t in teams if t):
            prev_dates.append(ko_dt)
    if not prev_dates:
        return None
    last = max(prev_dates)
    return (target_kickoff_utc - last).days


def _importance_score(stage: str) -> float:
    mapping = {
        "group": 0.55,
        "round_of_32": 0.7,
        "round_of_16": 0.78,
        "quarter": 0.85,
        "semi": 0.93,
        "final": 1.0,
        "third_place": 0.6,
    }
    return mapping.get((stage or "").lower(), 0.5)


def _detect_info_edge_flags(
    dossier: MatchDossier,
) -> list[str]:
    """Detecta mismatches obvios entre nuestras señales y el mercado."""
    flags = []
    p_h = dossier.market_p_home or 0
    p_a = dossier.market_p_away or 0

    # Flag 1: Elo dice una cosa, mercado otra
    if dossier.home.elo_seed and dossier.away.elo_seed:
        # Diferencial Elo → win prob aproximada
        elo_diff = dossier.home.elo_seed + 80 - dossier.away.elo_seed  # home advantage +80
        elo_p_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400))
        gap = p_h - elo_p_home
        if abs(gap) > 0.15:
            flags.append(
                f"Mercado vs Elo: mercado dice P(local)={p_h:.0%} pero Elo {elo_p_home:.0%} "
                f"(diff {gap:+.0%}). Mercado puede tener info no capturada en Elo."
            )

    # Flag 2: forma reciente muy mala vs alta probabilidad de victoria
    if dossier.home.recent_form and dossier.home.recent_form.startswith("LL"):
        if p_h > 0.55:
            flags.append(
                f"Forma local pobre ({dossier.home.recent_form}) pero mercado lo da fav {p_h:.0%} — "
                f"chequear si hay razones (rivales más débiles ahora, info no pública)."
            )
    if dossier.away.recent_form and dossier.away.recent_form.startswith("LL"):
        if p_a > 0.45:
            flags.append(
                f"Forma visit pobre ({dossier.away.recent_form}) pero mercado P(visit)={p_a:.0%}."
            )

    # Flag 3: muchas bajas reportadas
    n_home_abs = len(dossier.home.reported_absences)
    n_away_abs = len(dossier.away.reported_absences)
    if n_home_abs >= 3:
        flags.append(f"{dossier.home.name}: {n_home_abs} bajas reportadas (chequear impacto en λ)")
    if n_away_abs >= 3:
        flags.append(f"{dossier.away.name}: {n_away_abs} bajas reportadas")

    # Flag 4: descanso desigual
    if dossier.rest_days_home and dossier.rest_days_away:
        if abs(dossier.rest_days_home - dossier.rest_days_away) >= 3:
            flags.append(
                f"Descanso desigual: home {dossier.rest_days_home}d vs away {dossier.rest_days_away}d. "
                f"El más descansado tiene ventaja."
            )

    return flags


def build_dossier(
    match: dict,
    constraints: dict,                       # del modelo Poisson
    teams_data: dict | None = None,         # config/teams.yaml
    fapi_ctx: dict | None = None,
    espn_ctx: dict | None = None,
    news_ctx: dict | None = None,
    weather_ctx: dict | None = None,
    all_fixtures: list[dict] | None = None,
    venues_data: dict | None = None,
) -> MatchDossier:
    """Construye el dossier consolidado a partir de todas las fuentes."""
    fapi_ctx = fapi_ctx or {}
    espn_ctx = espn_ctx or {}
    news_ctx = news_ctx or {}

    home_code = match.get("home", "?")
    away_code = match.get("away", "?")
    home_name = match.get("home_name") or home_code
    away_name = match.get("away_name") or away_code

    kickoff_dt = datetime.fromisoformat(match["kickoff_utc"].replace("Z", "+00:00"))

    # Resolver perfiles de equipo desde teams.yaml
    def _team_info(code: str) -> dict:
        if not teams_data:
            return {}
        for group_letter, teams in teams_data.get("groups", {}).items():
            for t in teams:
                if t.get("code") == code:
                    return t
        return {}

    home_meta = _team_info(home_code)
    away_meta = _team_info(away_code)

    # Resolver venue
    venue_name = match.get("venue") or espn_ctx.get("venue")
    venue_data = {}
    if venues_data and venue_name:
        name_to_key = venues_data.get("name_to_key", {})
        key = name_to_key.get(venue_name)
        if not key:
            for k, v in venues_data.get("venues", {}).items():
                if v.get("name", "").lower() in venue_name.lower():
                    key = k
                    break
        if key:
            venue_data = venues_data.get("venues", {}).get(key, {})

    # TeamProfile builders
    home_news_text = news_ctx.get("home_news_summary", "")
    home_headlines = [l for l in home_news_text.split("\n") if l.strip().startswith("[")][:3]

    away_news_text = news_ctx.get("away_news_summary", "")
    away_headlines = [l for l in away_news_text.split("\n") if l.strip().startswith("[")][:3]

    home_profile = TeamProfile(
        name=home_name,
        code=home_code,
        confederation=home_meta.get("confed"),
        elo_seed=home_meta.get("elo_seed"),
        recent_form=fapi_ctx.get("home_recent_form") or espn_ctx.get("home_recent_form_espn"),
        squad_listed_count=len((espn_ctx.get("home_squad_listed") or "").split(",")) if espn_ctx.get("home_squad_listed") else None,
        reported_absences=fapi_ctx.get("home_injuries") or [],
        notable_news_headlines=home_headlines,
    )

    away_profile = TeamProfile(
        name=away_name,
        code=away_code,
        confederation=away_meta.get("confed"),
        elo_seed=away_meta.get("elo_seed"),
        recent_form=fapi_ctx.get("away_recent_form") or espn_ctx.get("away_recent_form_espn"),
        squad_listed_count=len((espn_ctx.get("away_squad_listed") or "").split(",")) if espn_ctx.get("away_squad_listed") else None,
        reported_absences=fapi_ctx.get("away_injuries") or [],
        notable_news_headlines=away_headlines,
    )

    # Rest days
    rest_home = _compute_rest_days(home_code, kickoff_dt, all_fixtures or [])
    rest_away = _compute_rest_days(away_code, kickoff_dt, all_fixtures or [])

    # Sources used
    sources = []
    if fapi_ctx: sources.append("api-football")
    if espn_ctx: sources.append("espn")
    if news_ctx: sources.append("google_news")
    if weather_ctx: sources.append("openweathermap")
    if constraints: sources.append("pinnacle")

    dossier = MatchDossier(
        match_id=match["id"],
        match_label=f"{home_name} vs {away_name}",
        kickoff_utc=match["kickoff_utc"],
        kickoff_local_uy=_kickoff_uy(match["kickoff_utc"]),
        stage=match.get("stage", "?"),
        group=match.get("group"),
        venue_name=venue_name,
        venue_city=venue_data.get("city"),
        venue_country=venue_data.get("country"),
        venue_elevation_m=venue_data.get("elevation_m"),
        venue_indoor=venue_data.get("indoor"),
        home_advantage_status="neutral" if venue_data.get("country") not in (home_meta.get("country"), None) else "home",
        home=home_profile,
        away=away_profile,
        rest_days_home=rest_home,
        rest_days_away=rest_away,
        importance_score=_importance_score(match.get("stage", "")),
        h2h_summary=fapi_ctx.get("h2h_recent") or espn_ctx.get("h2h_recent_espn"),
        market_p_home=constraints.get("p_home"),
        market_p_draw=constraints.get("p_draw"),
        market_p_away=constraints.get("p_away"),
        market_e_goals_home=constraints.get("e_goals_L"),
        market_e_goals_away=constraints.get("e_goals_V"),
        market_sources=["pinnacle"],
        draftkings_odds=espn_ctx.get("draftkings_odds"),
        weather_summary=weather_ctx.get("summary") if weather_ctx else None,
        weather_details=weather_ctx,
        sources_used=sources,
    )

    # Computar info_edge_flags al final
    dossier.info_edge_flags = _detect_info_edge_flags(dossier)
    return dossier


# ============ formatters ============

def dossier_to_llm_context_text(dossier: MatchDossier) -> str:
    """Formatea el dossier como texto estructurado para el prompt del LLM.

    Mejor que pasarle texto crudo de noticias. Lo presenta como ficha clara.
    """
    lines = [
        f"=== FICHA DEL PARTIDO ===",
        f"Partido: {dossier.match_label}",
        f"Etapa: {dossier.stage}  |  Importancia: {dossier.importance_score:.0%}" if dossier.importance_score else f"Etapa: {dossier.stage}",
        f"Kickoff: {dossier.kickoff_local_uy}",
    ]
    if dossier.venue_name:
        venue_line = f"Sede: {dossier.venue_name}"
        if dossier.venue_city: venue_line += f" ({dossier.venue_city})"
        if dossier.venue_elevation_m and dossier.venue_elevation_m > 1500:
            venue_line += f"  ⚠️ altura {dossier.venue_elevation_m}m"
        if dossier.venue_indoor:
            venue_line += "  (indoor)"
        lines.append(venue_line)
    lines.append(f"Ventaja de localía: {dossier.home_advantage_status}")
    lines.append("")

    # Equipos lado a lado
    def team_section(t: TeamProfile, label: str) -> list[str]:
        ls = [f"--- {label}: {t.name} ---"]
        if t.elo_seed: ls.append(f"  Elo: {t.elo_seed}")
        if t.recent_form: ls.append(f"  Forma reciente (últ 5): {t.recent_form}")
        if t.squad_listed_count: ls.append(f"  Squad listado: {t.squad_listed_count} jugadores")
        if t.reported_absences:
            ls.append(f"  Bajas reportadas:")
            for a in t.reported_absences[:5]:
                ls.append(f"    · {a}")
        if t.notable_news_headlines:
            ls.append(f"  Noticias destacadas:")
            for h in t.notable_news_headlines[:3]:
                ls.append(f"    · {h[:120]}")
        return ls

    lines.extend(team_section(dossier.home, "LOCAL"))
    lines.append("")
    lines.extend(team_section(dossier.away, "VISITANTE"))
    lines.append("")

    # Match-specific
    if dossier.rest_days_home is not None or dossier.rest_days_away is not None:
        lines.append(f"Descanso: home {dossier.rest_days_home or '?'}d / away {dossier.rest_days_away or '?'}d")
    if dossier.h2h_summary:
        lines.append(f"H2H: {dossier.h2h_summary[:200]}")
    if dossier.weather_summary:
        lines.append(f"Clima: {dossier.weather_summary}")

    # Mercado (de referencia)
    if dossier.market_p_home is not None:
        lines.append("")
        lines.append("--- MERCADO (Pinnacle) ---")
        lines.append(
            f"  P(local) {dossier.market_p_home:.0%}  ·  "
            f"P(empate) {(dossier.market_p_draw or 0):.0%}  ·  "
            f"P(visit) {(dossier.market_p_away or 0):.0%}"
        )
        lines.append(
            f"  E[goles]: {(dossier.market_e_goals_home or 0):.2f} - {(dossier.market_e_goals_away or 0):.2f}"
        )

    # Flags
    if dossier.info_edge_flags:
        lines.append("")
        lines.append("--- FLAGS A REVISAR ---")
        for f in dossier.info_edge_flags:
            lines.append(f"  ⚠️ {f}")

    return "\n".join(lines)


def save_dossier_json(dossier: MatchDossier, data_dir: Path) -> Path:
    """Persiste el dossier como JSON versionado."""
    match_dir = data_dir / "dossiers" / str(dossier.match_id)
    match_dir.mkdir(parents=True, exist_ok=True)
    existing = list(match_dir.glob("v*.json"))
    n = len(existing) + 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = match_dir / f"v{n}_{ts}.json"
    path.write_text(json.dumps(asdict(dossier), indent=2, ensure_ascii=False, default=str))
    return path

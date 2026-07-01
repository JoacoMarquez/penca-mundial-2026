"""Pipeline orquestador por partido.

Tres fases:
    T-24h    — investigación exhaustiva, genera v1, notifica al usuario.
    T-3h     — verifica alineaciones probables, diff vs v1, notifica si cambia.
    T-30min  — alineaciones confirmadas, genera v_final, publica vía API, notifica lock-in.

Cada pasada escribe data/predictions/{match_id}/v{N}_{ts}.json (versionado, nunca sobreescribe).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from src.meta.pool import PoolModelConfig
from src.model.dossier import (
    MatchDossier, build_dossier, dossier_to_llm_context_text, save_dossier_json,
)
from src.model.market_probs import BookQuote, aggregate, devig
from src.model.poisson import MarketConstraints, fit_params, marginals, score_grid
from src.model.qualitative import (
    MAX_ABS_ADJUSTMENT_SPECULATIVE,
    MatchContext,
    adjust_with_llm,
    apply_to_lambdas,
)
from src.notifier.telegram import TelegramConfig, TelegramNotifier
from src.strategy.portfolio import (
    PortfolioResult, generate_portfolio, generate_candidates, picks_to_dicts,
)
from src.strategy.assignment import (
    fetch_my_pencas_standings,
    fetch_pool_top_k_threshold,
    greedy_assignment,
)
from src.publisher.penca_api import (
    PredictionPayload,
    get_publisher_from_env,
)
from src.utils.versions import sort_versions

log = logging.getLogger(__name__)


class Phase(str, Enum):
    T_24H = "T_24h"
    T_3H = "T_3h"
    T_30MIN = "T_30min"


# -------------------- helpers de I/O --------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
PREDICTIONS_DIR = DATA_DIR / "predictions"


def load_fixtures() -> dict:
    with open(PROJECT_ROOT / "config" / "fixtures.yaml") as f:
        return yaml.safe_load(f)


def load_teams() -> dict:
    with open(PROJECT_ROOT / "config" / "teams.yaml") as f:
        return yaml.safe_load(f)


def load_books_config() -> dict:
    with open(PROJECT_ROOT / "config" / "books.yaml") as f:
        return yaml.safe_load(f)


def _matches_remaining(fixtures: dict, now: datetime | None = None) -> int:
    """Partidos del torneo que todavía no arrancaron (incluye el actual si no es kickoff).

    Insumo del cutoff con horizonte: β·√(restantes). Mínimo 1 para no anular el premium
    en el último partido (que igual tiende a cero solo).
    """
    now = now or datetime.now(timezone.utc)
    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    count = 0
    for m in all_matches:
        ko = m.get("kickoff_utc")
        if not ko:
            count += 1  # eliminatoria sin fecha todavía → falta jugarse
            continue
        try:
            if datetime.fromisoformat(str(ko).replace("Z", "+00:00")) > now:
                count += 1
        except Exception:
            count += 1
    return max(count, 1)


def find_match(fixtures: dict, match_id: str) -> dict:
    for m in fixtures.get("fase_grupos", []):
        if str(m["id"]) == str(match_id):
            return m
    for m in fixtures.get("eliminatorias", []):
        if str(m["id"]) == str(match_id):
            return m
    raise ValueError(f"match_id {match_id} no encontrado en fixtures")


def next_version_path(match_id: str) -> tuple[Path, int]:
    """Devuelve la siguiente versión disponible: vN_<timestamp>.json."""
    match_dir = PREDICTIONS_DIR / str(match_id)
    match_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(match_dir.glob("v*_*.json"))
    n = len(existing) + 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return match_dir / f"v{n}_{ts}.json", n


def load_previous_predictions(match_id: str) -> list[dict]:
    match_dir = PREDICTIONS_DIR / str(match_id)
    if not match_dir.exists():
        return []
    out = []
    for p in sort_versions(match_dir.glob("v*_*.json")):
        out.append(json.loads(p.read_text()))
    return out


# -------------------- scrapers (interfaz; impl en src/scrapers/) --------------------

@dataclass(frozen=True)
class OddsSnapshot:
    """Snapshot de odds de UN partido en UN momento, de TODAS las casas configuradas."""
    match_id: str
    fetched_at: str  # ISO 8601 UTC
    # Para cada casa, dict de mercado → dict outcome → odds
    # Ejemplo: {"pinnacle": {"1x2": {"H": 2.10, "D": 3.40, "A": 3.60}, "btts": {...}}}
    odds_by_book: dict[str, dict[str, dict[str, float]]]


def fetch_odds(match_id: str) -> OddsSnapshot:
    """Fetcha odds en vivo de las casas configuradas. Si una falla, log + sigue con las que respondan."""
    fixtures = load_fixtures()
    match = find_match(fixtures, match_id)

    odds_by_book: dict[str, dict[str, dict[str, float]]] = {}

    # Pinnacle
    try:
        from src.scrapers.pinnacle import (
            extract_match_markets, find_world_cup_league_id, get_markets,
            get_matchups, map_to_match_id, parse_matchups,
        )
        league_id = find_world_cup_league_id()
        if league_id:
            matchups = parse_matchups(get_matchups(league_id))
            markets_raw = get_markets(league_id, primary_only=False)
            teams_data = load_teams()
            aliases = teams_data.get("aliases", {})
            for pm in matchups:
                mapped = map_to_match_id(pm, fixtures, aliases)
                if mapped is not None and str(mapped) == str(match_id):
                    pinnacle_markets = extract_match_markets(pm.matchup_id, markets_raw)
                    if pinnacle_markets:
                        odds_by_book["pinnacle"] = pinnacle_markets
                    break
            if "pinnacle" not in odds_by_book:
                log.warning("Pinnacle: no encontré matchup para match_id=%s", match_id)
    except Exception as e:
        log.exception("Pinnacle fetch falló: %s", e)

    # The-Odds-API — segunda fuente: consenso (mediana) de ~31 casas en una sola llamada
    # cacheada. Redundancia ante caída de Pinnacle + señal multi-libro real.
    try:
        from src.scrapers.odds_api import markets_for_match
        teams_data = load_teams()
        consensus = markets_for_match(match_id, fixtures, teams_data)
        if consensus:
            odds_by_book["odds_api_consensus"] = consensus
        else:
            log.info("Odds-API: sin consenso para match_id=%s", match_id)
    except Exception as e:
        log.warning("Odds-API fetch falló: %s", e)

    # Bet365 — TODO: implementar con Playwright

    if not odds_by_book:
        # NO inventamos odds acá. El caller decide el fallback (versión previa con odds
        # reales, y solo como último recurso un mock + alerta). Publicar con cuotas
        # inventadas a T-30min sería peor que usar las reales de hace unas horas.
        log.error("Ninguna casa devolvió odds para %s — odds_by_book vacío", match_id)

    return OddsSnapshot(
        match_id=match_id,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        odds_by_book=odds_by_book,
    )


# -------------------- modelo --------------------

def build_market_constraints(snapshot: OddsSnapshot, books_config: dict) -> MarketConstraints:
    """Aplica de-vig + agregación ponderada y produce las constraints del modelo."""
    weights = {b: c["weight"] for b, c in books_config["books"].items() if c.get("enabled", True)}
    devig_methods = books_config["devig"]

    # 1X2
    quotes_1x2: list[BookQuote] = []
    quotes_ou25: list[BookQuote] = []
    quotes_btts: list[BookQuote] = []
    for book, markets in snapshot.odds_by_book.items():
        if "1x2" in markets:
            probs = devig(markets["1x2"], method=devig_methods.get("1x2", "proportional"))
            quotes_1x2.append(BookQuote(book=book, market="1x2", probs=probs))
        if "ou_2_5" in markets:
            probs = devig(markets["ou_2_5"], method=devig_methods.get("over_under", "proportional"))
            quotes_ou25.append(BookQuote(book=book, market="ou_2_5", probs=probs))
        if "btts" in markets:
            probs = devig(markets["btts"], method=devig_methods.get("btts", "proportional"))
            quotes_btts.append(BookQuote(book=book, market="btts", probs=probs))

    p_1x2 = aggregate(quotes_1x2, weights)
    p_ou25 = aggregate(quotes_ou25, weights) if quotes_ou25 else None
    p_btts = aggregate(quotes_btts, weights) if quotes_btts else None

    return MarketConstraints(
        p_home_win=p_1x2["H"],
        p_draw=p_1x2["D"],
        p_away_win=p_1x2["A"],
        p_over_2_5=p_ou25["over"] if p_ou25 else None,
        p_btts=p_btts["yes"] if p_btts else None,
    )


# Constraints neutras de último recurso (solo si no hay odds en vivo NI versión previa).
MOCK_CONSTRAINTS = MarketConstraints(p_home_win=0.40, p_draw=0.30, p_away_win=0.30)


def _best_effort_alert(context: str, message: str) -> None:
    """Manda una alerta a Telegram, best-effort (no rompe el pipeline si Telegram falla)."""
    try:
        TelegramNotifier(TelegramConfig.from_env()).send_error(context, message)
    except Exception:
        pass


def build_constraints_with_fallback(
    snapshot: OddsSnapshot, books_config: dict, match_id: str, match_label: str,
) -> tuple[MarketConstraints, str]:
    """Constraints del mercado, con fallback si no hay odds en vivo.

    Orden: (1) odds en vivo; (2) constraints de la última versión previa con odds reales
    (de hace horas); (3) último recurso: MOCK + alerta fuerte. Nunca publicamos sobre odds
    inventadas si tenemos unas reales de antes. Devuelve (constraints, source).
    """
    if snapshot.odds_by_book:
        return build_market_constraints(snapshot, books_config), "live"

    for prev in reversed(load_previous_predictions(match_id)):
        c = prev.get("constraints") or {}
        if c.get("p_home") is not None and prev.get("odds_source") != "mock":
            log.warning("Sin odds en vivo para %s — uso constraints de la versión previa v%s",
                        match_id, prev.get("version"))
            _best_effort_alert(
                f"odds caídas {match_label}",
                "No hubo odds en vivo; uso las de la versión anterior (reales, de hace horas).",
            )
            return MarketConstraints(
                p_home_win=c["p_home"], p_draw=c["p_draw"], p_away_win=c["p_away"],
                p_over_2_5=c.get("p_over_2_5"), p_btts=c.get("p_btts"),
            ), "previous"

    log.error("Sin odds en vivo NI previas para %s — usando MOCK", match_id)
    _best_effort_alert(
        f"⚠️ SIN ODDS {match_label}",
        "No hubo odds en vivo ni versión previa. Uso MOCK neutro — REVISAR manualmente.",
    )
    return MOCK_CONSTRAINTS, "mock"


def _should_publish(odds_source: str) -> bool:
    """¿Esta pick puede ir al pool?

    NO publicamos cuando las constraints cayeron a MOCK (40/30/30 inventadas): meterían
    al pool una pick sesgada al "home" cuando no hubo señal real (típicamente un knockout
    cuyo bracket no estaba resuelto al correr la pasada, o todas las casas caídas sin
    versión previa). Mejor no publicar nada y alertar que cargar una pick basura.
    `live` y `previous` (odds reales, aunque sean de hace horas) sí se publican.
    """
    return odds_source != "mock"


# -------------------- pipeline --------------------

@dataclass
class PipelineRun:
    match_id: str
    phase: Phase
    version: int
    run_at: str
    constraints: dict[str, Any]
    portfolio: dict[str, Any]
    odds_snapshot: dict[str, Any]
    qualitative_adjustment: dict[str, Any] | None = None
    assignment: list[dict[str, Any]] | None = None
    assignment_meta: dict[str, Any] | None = None
    tipster_consensus: dict[str, Any] | None = None
    dossier_summary_text: str | None = None    # versión resumida del dossier para Telegram
    odds_anomaly: dict[str, Any] | None = None  # flag si las odds se movieron >5pp vs versión previa
    odds_source: str = "live"          # live | previous | mock — de dónde salieron las constraints
    published: bool | None = None      # solo T-30min: True=publicó ok, False=falló (→ retry), None=n/a
    home_xi: list[str] | None = None   # XI confirmado (si lo hubo) — para auditar λ vs alineación
    away_xi: list[str] | None = None


def run_match_pipeline(match_id: str, phase: Phase) -> PipelineRun:
    """Ejecuta una pasada de la pipeline para un partido en una fase dada."""
    log.info("pipeline START | match=%s phase=%s", match_id, phase.value)

    # Los fixtures reales traen IDs enteros (105…); normalizamos a str para rutas y payload.
    match_id = str(match_id)
    fixtures = load_fixtures()
    match = find_match(fixtures, match_id)

    # 1. Scrape odds
    snapshot = fetch_odds(match_id)

    # 2. Construir constraints (con fallback a la versión previa si no hay odds en vivo)
    books_config = load_books_config()
    constraints, odds_source = build_constraints_with_fallback(
        snapshot, books_config, match_id, _format_match_label(match)
    )

    # 3. Fit Poisson bivariada + score grid
    lam_L, lam_V, lam12 = fit_params(constraints)
    grid = score_grid(lam_L, lam_V, lam12, max_goals=7)
    m = marginals(grid)

    # 3-ANOMALY: Comparar odds vs versión previa. Si se mueve >5pp, flag + alerta.
    odds_anomaly = None
    try:
        from src.model.anomaly import compare_market_probs
        previous_versions = load_previous_predictions(match_id)
        if previous_versions:
            prev_constraints = previous_versions[-1].get("constraints", {})
            current_probs = {
                "p_home": constraints.p_home_win,
                "p_draw": constraints.p_draw,
                "p_away": constraints.p_away_win,
            }
            anomaly = compare_market_probs(prev_constraints, current_probs)
            if anomaly:
                odds_anomaly = {
                    "delta_home_pp": anomaly.delta_home_pp,
                    "delta_draw_pp": anomaly.delta_draw_pp,
                    "delta_away_pp": anomaly.delta_away_pp,
                    "max_abs_delta_pp": anomaly.max_abs_delta_pp,
                    "direction": anomaly.direction,
                    "summary": anomaly.summary,
                }
                log.warning("ODDS ANOMALY: %s", anomaly.summary)
                # Alerta Telegram inmediata
                try:
                    from src.notifier.telegram import TelegramNotifier, TelegramConfig
                    notif = TelegramNotifier(TelegramConfig.from_env())
                    label = _format_match_label(match)
                    notif.send_error(
                        f"odds anomaly {label}",
                        anomaly.summary,
                    )
                except Exception:
                    pass
    except Exception as e:
        log.exception("anomaly check falló: %s", e)

    # 3a. Capa 3: tipsters consensus (si hay ANTHROPIC_API_KEY y solo en T-24h para ahorrar costos)
    tipster_consensus: dict | None = None
    if phase == Phase.T_24H and os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-xxx"):
        try:
            from src.scrapers.tipsters_runner import collect_tipster_consensus
            kickoff_dt = datetime.fromisoformat(match["kickoff_utc"].replace("Z", "+00:00"))
            tipster_consensus = collect_tipster_consensus(
                home=match.get("home_name") or match.get("home", "?"),
                away=match.get("away_name") or match.get("away", "?"),
                kickoff_utc=kickoff_dt,
                market_p_home=constraints.p_home_win,
                market_p_away=constraints.p_away_win,
            )
            log.info("Capa 3 tipsters: %d picks recolectadas", tipster_consensus.get("n_tipsters_picked", 0))
        except Exception as e:
            log.exception("Capa 3 falló — sigo sin tipsters: %s", e)

    # 3b. Capa 4: ajuste cualitativo con LLM (si hay ANTHROPIC_API_KEY)
    qualitative_summary: dict | None = None
    if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-xxx"):
        try:
            # Recolectar contexto de partido: lineups/lesiones (api-football, si plan) + noticias + clima
            try:
                from src.scrapers.football_api import collect_match_context
                from src.scrapers.weather import get_weather_for_match
                from src.scrapers.news import collect_news_context
                from src.scrapers.espn import collect_match_context_espn
                kickoff_dt = datetime.fromisoformat(match["kickoff_utc"].replace("Z", "+00:00"))
                fetch_lineups = phase in (Phase.T_3H, Phase.T_30MIN)
                home_name = match.get("home_name") or match.get("home", "?")
                away_name = match.get("away_name") or match.get("away", "?")
                fapi_ctx = collect_match_context(
                    home_name=home_name, away_name=away_name,
                    kickoff_utc=kickoff_dt, fetch_lineups=fetch_lineups,
                )
                weather_ctx = get_weather_for_match(match.get("venue"), kickoff_dt)
                news_ctx = collect_news_context(home_name, away_name, lang="es")
                espn_ctx = collect_match_context_espn(home_name, away_name)
            except Exception as e:
                log.warning("fetch contexto falló: %s", e)
                fapi_ctx = {}
                weather_ctx = None
                news_ctx = {}
                espn_ctx = {}

            # Fallback de alineación: si api-football (plan FREE) no trajo el XI,
            # buscamos el once confirmado en otras fuentes para T-3h/T-30min.
            # Sin XI confiable, Capa 4 trabaja a ciegas sobre rumores de noticias.
            if fetch_lineups and not fapi_ctx.get("home_xi") and not fapi_ctx.get("away_xi"):
                # 1) ESPN (ya scrapeado arriba, sin Playwright). Trae los starters
                #    cuando la alineación está anunciada (~1h antes del kickoff).
                if espn_ctx.get("home_xi") or espn_ctx.get("away_xi"):
                    for k in ("home_xi", "away_xi", "home_lineup_change",
                              "away_lineup_change", "lineup_confirmed", "lineup_source"):
                        if espn_ctx.get(k) is not None:
                            fapi_ctx.setdefault(k, espn_ctx[k])
                    log.info("Alineación vía ESPN: confirmed=%s, xi_local=%d xi_visit=%d",
                             espn_ctx.get("lineup_confirmed"),
                             len(espn_ctx.get("home_xi") or []),
                             len(espn_ctx.get("away_xi") or []))
                # 2) SofaScore (browser real) como segundo fallback si ESPN no trajo XI.
                if not fapi_ctx.get("home_xi") and not fapi_ctx.get("away_xi"):
                    try:
                        from src.scrapers.sofascore import collect_match_context_sofascore
                        sofa_ctx = collect_match_context_sofascore(home_name, away_name, kickoff_dt)
                        if sofa_ctx:
                            # No pisar nada que api-football ya haya provisto.
                            for k, v in sofa_ctx.items():
                                fapi_ctx.setdefault(k, v)
                            log.info("Alineación vía SofaScore: confirmed=%s, xi_local=%d xi_visit=%d",
                                     sofa_ctx.get("lineup_confirmed"),
                                     len(sofa_ctx.get("home_xi") or []),
                                     len(sofa_ctx.get("away_xi") or []))
                    except Exception as e:
                        log.warning("fallback SofaScore falló: %s", e)

            # Construir Match Dossier consolidado (todas las fuentes en una ficha estructurada)
            try:
                import yaml
                teams_yaml = yaml.safe_load((PROJECT_ROOT / "config" / "teams.yaml").read_text())
                venues_yaml = yaml.safe_load((PROJECT_ROOT / "config" / "venues.yaml").read_text())
            except Exception:
                teams_yaml, venues_yaml = None, None
            constraints_dict = {
                "p_home": constraints.p_home_win,
                "p_draw": constraints.p_draw,
                "p_away": constraints.p_away_win,
                "e_goals_L": m.expected_goals_L,
                "e_goals_V": m.expected_goals_V,
            }
            all_fixtures_list = (
                (fixtures.get("fase_grupos") or [])
                + (fixtures.get("eliminatorias") or [])
            )
            dossier = build_dossier(
                match=match,
                constraints=constraints_dict,
                teams_data=teams_yaml,
                fapi_ctx=fapi_ctx,
                espn_ctx=espn_ctx,
                news_ctx=news_ctx,
                weather_ctx=weather_ctx,
                all_fixtures=all_fixtures_list,
                venues_data=venues_yaml,
            )
            try:
                save_dossier_json(dossier, DATA_DIR)
            except Exception as e:
                log.warning("save_dossier_json falló: %s", e)
            structured_context = dossier_to_llm_context_text(dossier)
            # Versión compacta para Telegram (HTML)
            from src.model.dossier import dossier_to_telegram_summary
            dossier_summary_for_telegram = dossier_to_telegram_summary(dossier)

            # ¿Conseguimos XI confirmado de alguna fuente? Si a T-3h/T-30min nadie lo trajo,
            # la Capa 4 corre "a ciegas" sobre rumores → la marcamos como especulativa para
            # endurecer el bound del ajuste (±0.10) y avisar antes del kickoff.
            no_confirmed_xi = bool(
                fetch_lineups and not (fapi_ctx.get("home_xi") or fapi_ctx.get("away_xi"))
            )
            if no_confirmed_xi:
                log.warning(
                    "NO XI CONFIRMADO para %s vs %s (%s) — Capa 4 a ciegas, bound reducido a ±%.2f. "
                    "Fuentes intentadas: api-football/espn/sofascore.",
                    dossier.home.name, dossier.away.name, phase.value,
                    MAX_ABS_ADJUSTMENT_SPECULATIVE,
                )

            # Overrides manuales de disponibilidad (config/player_overrides.yaml).
            # Corrigen falsos positivos de prensa que la Capa 4 podría tomar como baja.
            from src.model.overrides import team_overrides, filter_available
            home_avail, home_out = team_overrides(match.get("home"))
            away_avail, away_out = team_overrides(match.get("away"))
            home_inj = filter_available(dossier.home.reported_absences, home_avail) + list(home_out)
            away_inj = filter_available(dossier.away.reported_absences, away_avail) + list(away_out)

            ctx = MatchContext(
                home_team=dossier.home.name,
                away_team=dossier.away.name,
                kickoff_local=dossier.kickoff_local_uy,
                stage=match.get("stage", "group"),
                market_p_home=constraints.p_home_win,
                market_p_draw=constraints.p_draw,
                market_p_away=constraints.p_away_win,
                market_e_goals_L=m.expected_goals_L,
                market_e_goals_V=m.expected_goals_V,
                home_recent_form=dossier.home.recent_form,
                away_recent_form=dossier.away.recent_form,
                home_injuries=home_inj or None,
                away_injuries=away_inj or None,
                home_available=home_avail or None,
                away_available=away_avail or None,
                no_confirmed_xi=no_confirmed_xi,
                home_lineup_change=fapi_ctx.get("home_lineup_change"),
                away_lineup_change=fapi_ctx.get("away_lineup_change"),
                home_xi=fapi_ctx.get("home_xi"),
                away_xi=fapi_ctx.get("away_xi"),
                lineup_confirmed=bool(fapi_ctx.get("lineup_confirmed")),
                h2h_recent=dossier.h2h_summary,
                weather=dossier.weather_summary,
                motivation_notes=structured_context,
                home_news_summary=_combine_news(
                    news_ctx.get("home_news_summary"),
                    espn_ctx.get("espn_news"),
                ),
                away_news_summary=news_ctx.get("away_news_summary"),
            )
            adj = adjust_with_llm(ctx)
            new_L, new_V = apply_to_lambdas(lam_L, lam_V, adj)
            log.info("qualitative adj | λL %.2f→%.2f  λV %.2f→%.2f", lam_L, new_L, lam_V, new_V)
            lam_L, lam_V = new_L, new_V
            # Re-fit grid con λs ajustadas
            grid = score_grid(lam_L, lam_V, lam12, max_goals=7)
            m = marginals(grid)
            qualitative_summary = {
                "delta_lambda_L": adj.delta_lambda_L,
                "delta_lambda_V": adj.delta_lambda_V,
                "reasoning": adj.reasoning,
                "confidence": adj.confidence,
            }
        except Exception as e:
            log.exception("Capa 4 (qualitative) falló — sigo sin ajuste: %s", e)
    else:
        log.info("ANTHROPIC_API_KEY no configurada — skip Capa 4")

    # 4. Generar el portfolio (menú de objetivos canónicos para mostrar/persistir)
    # Config del pool: calibrada por ranking-inversion si hay jornadas jugadas, prior si no.
    from src.meta.calibration import get_pool_config
    pool_config = get_pool_config()
    portfolio = generate_portfolio(
        grid,
        market_p_home=constraints.p_home_win,
        market_p_away=constraints.p_away_win,
        pool_config=pool_config,
    )

    # 4b. Asignación adaptativa a N pencas (voraz, con repetición/exposición).
    #     Penca líder → ancla conservadora; las demás cubren outcomes aún no ganados.
    from src.utils.env import get_int_list
    penca_ids = get_int_list("PENCA_IDS")
    standings = fetch_my_pencas_standings(
        api_base_url=os.environ.get("PENCA_API_BASE_URL", ""),
        api_key=os.environ.get("PENCA_API_KEY", ""),
        my_penca_ids=penca_ids,
    ) if penca_ids else {}
    assignment_list: list[tuple[int, dict, int | None]] = []
    assignment_meta: dict[str, Any] = {}
    if penca_ids:
        # Menú de candidatos: cap chico a propósito. Con el objetivo e_max (sin datos de
        # pool, ej. primera fecha) un menú grande hace que el voraz gaste pencas en
        # marcadores absurdos (4-3, 4-2) para "cubrir" colas de muchos goles. 8 cubre lo
        # sensato sin diluir. Tunear con Monte Carlo del pool cuando esté.
        max_candidates = int(os.environ.get("PENCA_MAX_CANDIDATES", "10"))
        candidate_dicts = picks_to_dicts(
            generate_candidates(
                grid,
                market_p_home=constraints.p_home_win,
                market_p_away=constraints.p_away_win,
                pool_config=pool_config,
                max_candidates=max_candidates,
            )
        )
        try:
            from src.meta.pool import pool_pick_distribution
            pool_q_for_assignment = pool_pick_distribution(grid, pool_config)
            top_k_threshold = fetch_pool_top_k_threshold(
                api_base_url=os.environ.get("PENCA_API_BASE_URL", ""),
                api_key=os.environ.get("PENCA_API_KEY", ""),
                k=3,
            )
            # Cutoff con horizonte: el corte que importa es el del FINAL del torneo, no
            # el de hoy. Premium = β·√(partidos restantes) — backtest (Euro 2024, pool
            # 418, 5 semillas): β=2.0 da +6pp de P(ganar) vs cutoff miope (β=0 lo apaga).
            horizon_beta = float(os.environ.get("PENCA_HORIZON_BETA", "2.0"))
            horizon_premium = 0.0
            if horizon_beta > 0 and top_k_threshold is not None:
                import math
                horizon_premium = horizon_beta * math.sqrt(_matches_remaining(fixtures))
            # Protección de la líder: exime del premium a las pencas YA sobre el corte (que
            # el premium, de otro modo, empuja a anti-favoritos por tratarlas como rezagadas).
            # θ = colchón sobre el corte (en unidades de premium) para considerar "a salvo".
            # Backtest sembrado con el pago real (20k/10k/5k): θ≈0.4 maximiza E[premio] (+22%)
            # robusto a la calibración del campo. PENCA_PROTECT_THETA="" o "off" lo apaga.
            _pt_raw = os.environ.get("PENCA_PROTECT_THETA", "0.4").strip().lower()
            protect_theta = None if _pt_raw in ("", "off", "none") else float(_pt_raw)
            log.info(
                "Asignación: %d pencas, %d candidatos, pool top-3 threshold=%s, horizonte=+%.1f, protect_theta=%s",
                len(penca_ids), len(candidate_dicts), top_k_threshold, horizon_premium, protect_theta,
            )
            assignment_list, assignment_meta = greedy_assignment(
                candidate_dicts, penca_ids, grid, standings,
                pool_top_k_threshold=top_k_threshold,
                pool_q=pool_q_for_assignment,
                horizon_premium=horizon_premium,
                protect_theta=protect_theta,
            )
        except Exception as e:
            log.exception("greedy_assignment falló, usando mapeo cíclico: %s", e)
            assignment_list = [
                (pid, candidate_dicts[i % len(candidate_dicts)], None)
                for i, pid in enumerate(penca_ids)
            ]

    # 5. Publicar en CADA pasada (upsert) ANTES de persistir — red de seguridad: si una
    #    pasada tardía falla o el droplet se cae cerca del kickoff, ya hay algo publicado.
    #    El T-30min con published=False no cuenta como corrido → el scheduler lo reintenta.
    published: bool | None = None
    publish_detail: str | None = None
    if assignment_list and _is_degraded_emax(assignment_meta):
        # El objetivo cayó a e_max miope por FALTA de cutoff del pool (leaderboard caído Y
        # snapshot vencido >24h). Esas picks gastan pencas en colas de goles (4-3/4-2) — el
        # síntoma de Bélgica. Preferimos NO publicar y avisar antes que pisar el pool con picks
        # degradadas; las picks previas (si las hubo) quedan. published=None → no se reintenta.
        log.error(
            "objetivo e_max DEGRADADO (sin cutoff del pool) para %s — NO se publica", match_id,
        )
        _best_effort_alert(
            f"⚠️ Modo degradado no publicado {_format_match_label(match)}",
            "El leaderboard del pool no está disponible (ni fresco ni snapshot ≤24h) → la "
            "asignación cayó a e_max miope (marcadores de cola tipo 4-3). NO se publicó para no "
            "cargar picks degradadas al pool. Cargá a mano si el partido es inminente, o esperá "
            "a que vuelva la API del pool.",
        )
    elif assignment_list and not _should_publish(odds_source):
        # Backstop final: las constraints fueron MOCK → no tocamos el pool. Persistimos la
        # versión local (para auditoría) y alertamos fuerte, pero NO publicamos una pick
        # sobre probabilidades inventadas. published queda None (no es un fallo a reintentar).
        log.error(
            "odds_source=%s para %s — NO se publica (picks sobre prob. inventadas)",
            odds_source, match_id,
        )
        _best_effort_alert(
            f"⚠️ MOCK no publicado {_format_match_label(match)}",
            "Las constraints cayeron a MOCK (sin odds reales ni versión previa). NO se "
            "publicó al pool para no cargar una pick sesgada al local. Revisar: ¿el "
            "partido tiene equipos resueltos? ¿odds caídas?",
        )
    elif assignment_list:
        published, publish_detail = _publish_assignment(match_id, phase, assignment_list)
        if published is False:
            tail = (" — el scheduler reintentará." if phase == Phase.T_30MIN
                    else " — se reintenta en la próxima pasada.")
            _best_effort_alert(
                f"publish {phase.value} {_format_match_label(match)}",
                (publish_detail or "publicación falló") + tail,
            )

    # 6. Persistir versionado
    output_path, version = next_version_path(match_id)
    run = PipelineRun(
        match_id=match_id,
        phase=phase,
        version=version,
        run_at=datetime.now(timezone.utc).isoformat(),
        constraints={
            "p_home": constraints.p_home_win,
            "p_draw": constraints.p_draw,
            "p_away": constraints.p_away_win,
            "p_over_2_5": constraints.p_over_2_5,
            "p_btts": constraints.p_btts,
            "lambda_L": lam_L, "lambda_V": lam_V, "lambda_12": lam12,
            "e_goals_L": m.expected_goals_L, "e_goals_V": m.expected_goals_V,
        },
        portfolio=portfolio.to_dict(),
        odds_snapshot=asdict(snapshot),
        qualitative_adjustment=qualitative_summary,
        assignment=[
            {"penca_id": pid, "rank": rank, "objective": pick["objective"], "score": pick["score"]}
            for pid, pick, rank in assignment_list
        ],
        assignment_meta=assignment_meta or None,
        tipster_consensus=tipster_consensus,
        dossier_summary_text=locals().get("dossier_summary_for_telegram"),
        odds_anomaly=locals().get("odds_anomaly"),
        odds_source=odds_source,
        published=published,
        home_xi=(locals().get("fapi_ctx") or {}).get("home_xi"),
        away_xi=(locals().get("fapi_ctx") or {}).get("away_xi"),
    )
    output_path.write_text(json.dumps(asdict(run), indent=2, default=str))
    log.info("pipeline DONE | v=%d phase=%s odds=%s published=%s",
             version, phase.value, odds_source, published)

    # 7. Notificar según la fase
    _notify_and_publish(match, run, portfolio, phase)

    return run


# -------------------- notify + publish --------------------

def _combine_news(*chunks: str | None) -> str | None:
    """Junta varias secciones de noticias (Google News + ESPN) en un solo string."""
    parts = [c for c in chunks if c]
    if not parts:
        return None
    return "\n\n".join(parts)


def _format_match_label(match: dict, teams_data: dict | None = None) -> str:
    """Prefiere nombres completos en español del fixtures (home_name/away_name)."""
    home = match.get("home_name") or match.get("home") or "?"
    away = match.get("away_name") or match.get("away") or "?"
    return f"{home} vs {away}"


def _format_kickoff_local(match: dict) -> str:
    """Convierte el kickoff UTC a hora de Uruguay (UTC-3) para display."""
    from datetime import timedelta
    dt = datetime.fromisoformat(match["kickoff_utc"].replace("Z", "+00:00"))
    dt_uy = dt - timedelta(hours=3)
    return dt_uy.strftime("%a %d/%m %H:%M") + " UY"


def _publish_assignment(match_id: str, phase: Phase, assignment_list) -> tuple[bool, str | None]:
    """Publica las picks asignadas vía la API (upsert). Devuelve (todas_ok, detalle_o_None).

    Se publica en CADA pasada como red de seguridad: si una pasada tardía falla o el droplet
    se cae cerca del kickoff, ya hay una predicción publicada. La API hace upsert por
    (partido, penca), así que el estado final es el del T-30min si llega a correr.
    """
    publisher = get_publisher_from_env()
    payloads = [
        PredictionPayload(
            match_id=match_id, penca_id=str(pid),
            score_local=pick["score"][0], score_visit=pick["score"][1],
        )
        for pid, pick, _rank in assignment_list
    ]
    results = publisher.publish_batch(payloads)
    failed = [r for r in results if not r.ok]
    if not failed:
        log.info("publish %s OK | %d picks", phase.value, len(results))
        return True, None
    detail = f"{len(failed)}/{len(results)} fallaron: {failed[0].detail}"
    log.error("publish %s FALLÓ | %s", phase.value, detail)
    return False, detail


def _pool_snapshots_exist() -> bool:
    """¿Alguna vez hubo datos del pool? (snapshots en disco). Distingue 'leaderboard caído y
    snapshot vencido' (degradado, ya hubo pool) de la 1ª fecha genuina sin pool (e_max legítimo)."""
    try:
        d = DATA_DIR / "pool_snapshots"
        return d.exists() and any(d.glob("*.json"))
    except Exception:
        return False


def _is_degraded_emax(assignment_meta: dict | None) -> bool:
    """True si la asignación cayó a e_max miope por FALTA de cutoff del pool (leaderboard caído
    Y snapshot vencido). Distinto del e_max intencional de eliminatorias
    (objective='e_max (P(top-K)=0)', que SÍ tuvo cutoff). Solo bloquea si ya hubo pool (existen
    snapshots): en la 1ª fecha sin pool, e_max es legítimo y se publica. Desactivable con
    PENCA_BLOCK_DEGRADED_EMAX=false."""
    if os.environ.get("PENCA_BLOCK_DEGRADED_EMAX", "true").strip().lower() in ("false", "0", "off", "no"):
        return False
    meta = assignment_meta or {}
    # Solo el e_max "pelado" (sin cutoff); el intencional es "e_max (P(top-K)=0)" con threshold.
    if meta.get("objective") != "e_max" or meta.get("threshold") is not None:
        return False
    return _pool_snapshots_exist()


def republish_pending(match: dict, latest: dict) -> bool | None:
    """Reintenta SOLO la publicación de una asignación YA calculada (sin re-investigar).

    Cierra el gap del scheduler: una pasada T-24h/T-3h que quedó con published=False no se
    reintenta hasta la ventana T-30min (~2.5h). Acá reenviamos los POSTs de la última asignación
    (upsert idempotente). En éxito, versiona un archivo nuevo idéntico salvo published=True +
    publish_retry=True (respeta 'nunca sobreescribir'). Devuelve True/False/None.
    """
    match_id = str(match["id"])
    assignment = latest.get("assignment") or []
    if not assignment:
        return None
    try:
        phase = Phase(latest.get("phase"))
    except ValueError:
        phase = Phase.T_30MIN
    assignment_list = [
        (a["penca_id"], {"score": a["score"], "objective": a.get("objective")}, a.get("rank"))
        for a in assignment
    ]
    published, detail = _publish_assignment(match_id, phase, assignment_list)
    if published:
        new = dict(latest)
        new["published"] = True
        new["publish_retry"] = True
        new["run_at"] = datetime.now(timezone.utc).isoformat()
        out_path, version = next_version_path(match_id)
        new["version"] = version
        out_path.write_text(json.dumps(new, indent=2, default=str))
        log.info("republish OK (reintento) | match=%s v=%d picks=%d", match_id, version, len(assignment_list))
        _best_effort_alert(
            f"✅ Publicado (reintento) {_format_match_label(match)}",
            f"La API del pool respondió — se publicaron {len(assignment_list)} picks que habían "
            "fallado en una pasada anterior.",
        )
    else:
        log.info("republish sigue fallando | match=%s | %s", match_id, detail)
    return published


def _notify_and_publish(
    match: dict, run: PipelineRun, portfolio: PortfolioResult, phase: Phase,
) -> None:
    """Telegram = SOLO avisos (no pronósticos). En la pasada final (T-30min):
    1) confirma con una línea que se publicaron las picks, y
    2) corre las banderas pre-partido (concentración / λ-vs-XI) y avisa si saltan.

    T-24h y T-3h no notifican nada (las picks viven en la web/dashboard).
    """
    if phase != Phase.T_30MIN:
        return
    try:
        notifier = TelegramNotifier(TelegramConfig.from_env())
    except RuntimeError as e:
        log.warning("Telegram no configurado: %s", e)
        return

    label = _format_match_label(match)
    assignment = run.assignment or []

    # Banderas pre-partido (algo para revisar antes del kickoff)
    try:
        from src.agent.alerts import (
            check_concentration, check_lambda_vs_xi, check_no_confirmed_xi, render_flags,
        )
        flags = [f for f in (
            check_concentration([a.get("score") for a in assignment], len(assignment)),
            check_lambda_vs_xi(
                run.qualitative_adjustment, run.home_xi, run.away_xi,
                match.get("home_name") or "local", match.get("away_name") or "visitante",
            ),
            check_no_confirmed_xi(run.qualitative_adjustment, run.home_xi, run.away_xi),
        ) if f]
        if flags:
            notifier.send_alert(label, render_flags(flags))
    except Exception:
        log.exception("banderas pre-partido fallaron (no bloqueante)")

    # Confirmación mínima de publicación
    if run.published:
        try:
            notifier.send_publish_ok(label, len(assignment))
        except Exception:
            log.warning("no se pudo enviar confirmación de publicación")
    elif not assignment:
        log.warning("Sin asignación — no se publicó (¿PENCA_IDS vacío?)")


# -------------------- CLI --------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

    ap = argparse.ArgumentParser(description="Run pipeline para un partido específico")
    ap.add_argument("match_id", help="ID del partido (ej: MATCH_A_01)")
    ap.add_argument("--phase", choices=[p.value for p in Phase], default=Phase.T_24H.value)
    args = ap.parse_args()

    run = run_match_pipeline(args.match_id, Phase(args.phase))
    print(json.dumps(run.portfolio, indent=2))

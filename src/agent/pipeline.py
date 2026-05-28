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
from src.model.qualitative import MatchContext, adjust_with_llm, apply_to_lambdas
from src.notifier.telegram import TelegramConfig, TelegramNotifier
from src.strategy.portfolio import PortfolioResult, generate_portfolio
from src.strategy.assignment import (
    fetch_my_pencas_standings,
    fetch_pool_top_k_threshold,
    optimal_assignment,
    optimal_assignment_p_top_k,
)
from src.publisher.penca_api import (
    PredictionPayload,
    get_publisher_from_env,
)

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
    match_dir = PREDICTIONS_DIR / match_id
    match_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(match_dir.glob("v*_*.json"))
    n = len(existing) + 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return match_dir / f"v{n}_{ts}.json", n


def load_previous_predictions(match_id: str) -> list[dict]:
    match_dir = PREDICTIONS_DIR / match_id
    if not match_dir.exists():
        return []
    out = []
    for p in sorted(match_dir.glob("v*_*.json")):
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

    # Bet365 — TODO: implementar con Playwright
    # Betfair — desactivado por decisión del usuario

    if not odds_by_book:
        log.error("Ninguna casa devolvió odds para %s — usando MOCK fallback", match_id)
        odds_by_book = {
            "pinnacle": {
                "1x2": {"H": 2.10, "D": 3.40, "A": 3.50},
                "ou_2_5": {"over": 2.00, "under": 1.85},
            }
        }

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


def run_match_pipeline(match_id: str, phase: Phase) -> PipelineRun:
    """Ejecuta una pasada de la pipeline para un partido en una fase dada."""
    log.info("pipeline START | match=%s phase=%s", match_id, phase.value)

    fixtures = load_fixtures()
    match = find_match(fixtures, match_id)

    # 1. Scrape odds
    snapshot = fetch_odds(match_id)

    # 2. Construir constraints del mercado
    books_config = load_books_config()
    constraints = build_market_constraints(snapshot, books_config)

    # 3. Fit Poisson bivariada + score grid
    lam_L, lam_V, lam12 = fit_params(constraints)
    grid = score_grid(lam_L, lam_V, lam12, max_goals=7)
    m = marginals(grid)

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
                home_injuries=dossier.home.reported_absences or None,
                away_injuries=dossier.away.reported_absences or None,
                home_lineup_change=fapi_ctx.get("home_lineup_change"),
                away_lineup_change=fapi_ctx.get("away_lineup_change"),
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

    # 4. Generar las 5 picks
    portfolio = generate_portfolio(
        grid,
        market_p_home=constraints.p_home_win,
        market_p_away=constraints.p_away_win,
        pool_config=PoolModelConfig(),
    )

    # 4b. Asignación adaptativa: penca con más puntos → estrategia más conservadora
    penca_ids_raw = [x.strip() for x in os.environ.get("PENCA_IDS", "").split(",") if x.strip()]
    try:
        penca_ids = [int(x) for x in penca_ids_raw]
    except ValueError:
        penca_ids = []
    standings = fetch_my_pencas_standings(
        api_base_url=os.environ.get("PENCA_API_BASE_URL", ""),
        api_key=os.environ.get("PENCA_API_KEY", ""),
        my_penca_ids=penca_ids,
    ) if len(penca_ids) == 5 else {}
    assignment_list: list[tuple[int, dict, int | None]] = []
    assignment_meta: dict[str, Any] = {}
    if len(penca_ids) == 5:
        try:
            from src.meta.pool import pool_pick_distribution, PoolModelConfig as _PC
            pool_q_for_assignment = pool_pick_distribution(grid, _PC())
            top_k_threshold = fetch_pool_top_k_threshold(
                api_base_url=os.environ.get("PENCA_API_BASE_URL", ""),
                api_key=os.environ.get("PENCA_API_KEY", ""),
                k=3,
            )
            log.info("Asignación: pool top-3 threshold=%s", top_k_threshold)
            assignment_list, assignment_meta = optimal_assignment_p_top_k(
                portfolio.to_dict()["picks"], penca_ids, grid, standings,
                pool_top_k_threshold=top_k_threshold,
                pool_q=pool_q_for_assignment,
            )
        except Exception as e:
            log.exception("optimal_assignment falló, usando mapeo fijo: %s", e)
            assignment_list = [
                (pid, pick, None)
                for pid, pick in zip(penca_ids, portfolio.to_dict()["picks"])
            ]

    # 5. Persistir versionado
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
    )
    output_path.write_text(json.dumps(asdict(run), indent=2, default=str))
    log.info("pipeline DONE | v=%d wrote=%s", version, output_path)

    # 6. Notificar y publicar según la fase
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


def _notify_and_publish(
    match: dict, run: PipelineRun, portfolio: PortfolioResult, phase: Phase
) -> None:
    """Manda la notif por Telegram. Publica solo en T_30MIN (lock-in)."""
    try:
        notifier = TelegramNotifier(TelegramConfig.from_env())
    except RuntimeError as e:
        log.warning("Telegram no configurado: %s", e)
        notifier = None

    label = _format_match_label(match)
    kickoff_local = _format_kickoff_local(match)
    model_summary = {
        "p_home": run.constraints["p_home"],
        "p_draw": run.constraints["p_draw"],
        "p_away": run.constraints["p_away"],
        "e_goals_L": run.constraints["e_goals_L"],
        "e_goals_V": run.constraints["e_goals_V"],
    }
    picks = run.portfolio["picks"]

    # Anotar picks con info de asignación (penca_id real, rank entre tus pencas)
    if run.assignment:
        assigned_by_obj = {a["objective"]: a for a in run.assignment}
        picks_annotated = []
        for p in picks:
            assignment_info = assigned_by_obj.get(p["objective"])
            ann = dict(p)
            if assignment_info:
                ann["assigned_penca_id"] = assignment_info["penca_id"]
                ann["assigned_rank"] = assignment_info["rank"]
            picks_annotated.append(ann)
    else:
        picks_annotated = list(picks)

    if phase == Phase.T_24H and notifier:
        notifier.send_t24h_picks(
            label, kickoff_local, picks_annotated, model_summary,
            qualitative=run.qualitative_adjustment,
            assignment_meta=run.assignment_meta,
            tipster_consensus=run.tipster_consensus,
            dossier_summary=run.dossier_summary_text,
        )

    elif phase == Phase.T_3H and notifier:
        # Diff vs versión anterior
        prev_picks = _last_picks_from_predictions(run.match_id, exclude_version=run.version)
        if prev_picks:
            diffs = _compute_diffs(prev_picks, picks)
            if diffs:
                notifier.send_diff(label, "T-3h: alineaciones probables", diffs)

    elif phase == Phase.T_30MIN:
        pubpsher = get_publisher_from_env()
        if run.assignment:
            payloads = [
                PredictionPayload(
                    match_id=run.match_id,
                    penca_id=str(a["penca_id"]),
                    score_local=a["score"][0],
                    score_visit=a["score"][1],
                )
                for a in run.assignment
            ]
            results = pubpsher.publish_batch(payloads)
            failed = [r for r in results if not r.ok]
            if failed and notifier:
                notifier.send_error(
                    "publish T-30min",
                    f"{len(failed)}/{len(results)} fallaron: {failed[0].detail}",
                )
        else:
            log.warning("Sin asignación — skip publish")

        if notifier:
            notifier.send_lockin(label, picks_annotated)


def _last_picks_from_predictions(match_id: str, exclude_version: int) -> list[dict] | None:
    """Lee la versión más reciente anterior a `exclude_version` para hacer diff."""
    match_dir = PREDICTIONS_DIR / match_id
    if not match_dir.exists():
        return None
    versions = sorted(match_dir.glob("v*_*.json"))
    for f in reversed(versions):
        try:
            data = json.loads(f.read_text())
            if data.get("version") != exclude_version:
                return data.get("portfolio", {}).get("picks")
        except Exception:
            continue
    return None


def _compute_diffs(old_picks: list[dict], new_picks: list[dict]) -> list[dict]:
    """Devuelve picks que cambiaron entre versiones."""
    diffs = []
    for old, new in zip(old_picks, new_picks):
        if old["score"] != new["score"]:
            diffs.append({
                "penca_index": new["penca_index"],
                "old_score": old["score"],
                "new_score": new["score"],
                "reason": f"E[pts] {old['e_points']:.2f} → {new['e_points']:.2f}",
            })
    return diffs


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

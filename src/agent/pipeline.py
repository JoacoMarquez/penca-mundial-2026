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
from src.model.market_probs import BookQuote, aggregate, devig
from src.model.poisson import MarketConstraints, fit_params, marginals, score_grid
from src.strategy.portfolio import PortfolioResult, generate_portfolio

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
        if m["id"] == match_id:
            return m
    for m in fixtures.get("eliminatorias", []):
        if m["id"] == match_id:
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
    """STUB. Las implementaciones reales viven en src/scrapers/{pinnacle,bet365,betfair}.py
    y se ensamblan acá. Por ahora retornamos un mock para que el pipeline corra end-to-end."""
    log.warning("fetch_odds: usando MOCK ODDS — implementar scrapers reales en src/scrapers/")
    return OddsSnapshot(
        match_id=match_id,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        odds_by_book={
            "pinnacle": {
                "1x2": {"H": 2.10, "D": 3.40, "A": 3.50},
                "ou_2_5": {"over": 2.00, "under": 1.85},
                "btts": {"yes": 1.95, "no": 1.85},
            },
            "betfair_exchange": {
                "1x2": {"H": 2.12, "D": 3.45, "A": 3.55},
                "ou_2_5": {"over": 2.05, "under": 1.82},
                "btts": {"yes": 1.97, "no": 1.83},
            },
            "bet365": {
                "1x2": {"H": 2.05, "D": 3.30, "A": 3.40},
                "ou_2_5": {"over": 1.95, "under": 1.80},
                "btts": {"yes": 1.90, "no": 1.80},
            },
        },
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

    # 4. Generar las 5 picks
    portfolio = generate_portfolio(
        grid,
        market_p_home=constraints.p_home_win,
        market_p_away=constraints.p_away_win,
        pool_config=PoolModelConfig(),
    )

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
    )
    output_path.write_text(json.dumps(asdict(run), indent=2, default=str))
    log.info("pipeline DONE | v=%d wrote=%s", version, output_path)

    return run


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

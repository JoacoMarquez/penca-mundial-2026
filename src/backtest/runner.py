"""Backtest harness.

Carga resultados + closing odds históricos de un torneo (Euro 2024, Copa América 2024, Mundial 2022)
y simula:
    1. Que corremos nuestra pipeline en cada partido (con las closing odds como input).
    2. Un pool sintético de N jugadores chalk-biased.
    3. Para cada simulación del torneo: ¿cuántas veces nuestras 5 picks ganan el pool?

Métricas:
    P(al menos una de nuestras 5 termina #1 del pool sintético)
    P(comparado con 5x chalk benchmark)
    Distribución de puntos por penca

Data esperada en data/backtest/{tournament}/:
    matches.csv          # match_id, home_team, away_team, date, group, actual_home_score, actual_away_score
    closing_odds.csv     # match_id, book, market, outcome, odds, fetched_at
    (opcional) tipster_picks.csv

Estos CSVs los puede llenar otro scraper o cargarse a mano desde fuentes públicas
(OddsPortal historical, FBref).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.meta.pool import PoolModelConfig, pool_pick_distribution
from src.model.market_probs import BookQuote, aggregate, devig
from src.model.poisson import (
    MarketConstraints,
    fit_params,
    jmlm_points,
    marginals,
    score_grid,
)
from src.strategy.portfolio import generate_portfolio

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestMatch:
    match_id: str
    home: str
    away: str
    actual_home_score: int
    actual_away_score: int
    market_p_home: float
    market_p_draw: float
    market_p_away: float
    market_p_over_2_5: float | None = None
    market_p_btts: float | None = None


@dataclass(frozen=True)
class BacktestResult:
    n_matches: int
    n_simulations: int
    portfolio_points_by_penca: list[list[int]]   # [n_sims][5_pencas] — total points
    pool_top_points: list[int]                   # [n_sims] — max score del pool sintético
    p_at_least_one_wins: float                   # P(max(portfolio) > max(pool))
    p_chalk_baseline_wins: float                 # P(chalk también ganaría)


def load_backtest_data(tournament: str, data_root: Path) -> list[BacktestMatch]:
    """Carga los partidos del torneo. Asume CSVs preparados en data/backtest/{tournament}/.

    Para empezar rápido se puede cargar a mano un sample reducido de Euro 2024.
    """
    import csv

    tdir = data_root / tournament
    matches_path = tdir / "matches.csv"
    if not matches_path.exists():
        raise FileNotFoundError(f"No existe {matches_path}. Crearlo con columnas: "
                                "match_id,home,away,actual_home_score,actual_away_score,"
                                "market_p_home,market_p_draw,market_p_away,market_p_over_2_5,market_p_btts")

    out = []
    with open(matches_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(BacktestMatch(
                match_id=row["match_id"],
                home=row["home"], away=row["away"],
                actual_home_score=int(row["actual_home_score"]),
                actual_away_score=int(row["actual_away_score"]),
                market_p_home=float(row["market_p_home"]),
                market_p_draw=float(row["market_p_draw"]),
                market_p_away=float(row["market_p_away"]),
                market_p_over_2_5=float(row["market_p_over_2_5"]) if row.get("market_p_over_2_5") else None,
                market_p_btts=float(row["market_p_btts"]) if row.get("market_p_btts") else None,
            ))
    return out


def simulate_pool(
    n_players: int,
    matches: list[BacktestMatch],
    pool_config: PoolModelConfig,
    rng: np.random.Generator,
    chalk_concentration: float = 0.70,    # 70% pican el modal de Q (chalk puro)
    top_k_sampling: int = 5,              # los que no son chalk sampleamos del top-K Q (no de Q completa)
) -> np.ndarray:
    """Simula `n_players` modelando humanos reales.

    Modelo:
        - chalk_concentration% pican el modal de Q (es lo que hace la mayoría)
        - (1 - chalk_concentration)% sampleamos del top-K de Q (los marcadores más populares,
          NO scorelines random — los humanos no juegan 5-1, juegan 2-1 o 1-1).
    """
    n_matches = len(matches)
    points = np.zeros((n_players, n_matches), dtype=int)

    for j, match in enumerate(matches):
        constraints = MarketConstraints(
            p_home_win=match.market_p_home,
            p_draw=match.market_p_draw,
            p_away_win=match.market_p_away,
            p_over_2_5=match.market_p_over_2_5,
            p_btts=match.market_p_btts,
        )
        lam_L, lam_V, lam12 = fit_params(constraints)
        grid = score_grid(lam_L, lam_V, lam12)
        pool_q = pool_pick_distribution(grid, pool_config)

        n = pool_q.shape[0]
        flat_q = pool_q.flatten()
        modal_idx = int(np.argmax(flat_q))

        # Top-K de Q para los "sampleadores": tomamos los K más probables y normalizamos
        top_idx = np.argsort(flat_q)[::-1][:top_k_sampling]
        top_p = flat_q[top_idx]
        top_p = top_p / top_p.sum()

        actual = (match.actual_home_score, match.actual_away_score)

        for i in range(n_players):
            if rng.random() < chalk_concentration:
                idx = modal_idx
            else:
                idx = int(rng.choice(top_idx, p=top_p))
            pgL, pgV = idx // n, idx % n
            points[i, j] = jmlm_points((pgL, pgV), actual)

    return points


def run_backtest(
    matches: list[BacktestMatch],
    n_simulations: int = 100,
    n_pool_players: int = 150,
    seed: int = 42,
) -> BacktestResult:
    """Corre el backtest completo. Para cada simulación:
        1. Genera nuestras 5 picks con la pipeline actual.
        2. Simula un pool de n_pool_players chalk-biased.
        3. Compara nuestros mejores puntos con el top del pool.
    """
    rng = np.random.default_rng(seed)

    # Las 5 picks son determinísticas dado el modelo — no varían entre sims
    portfolio_picks_by_match = []
    for match in matches:
        constraints = MarketConstraints(
            p_home_win=match.market_p_home,
            p_draw=match.market_p_draw,
            p_away_win=match.market_p_away,
            p_over_2_5=match.market_p_over_2_5,
            p_btts=match.market_p_btts,
        )
        lam_L, lam_V, lam12 = fit_params(constraints)
        grid = score_grid(lam_L, lam_V, lam12)
        portfolio = generate_portfolio(grid, match.market_p_home, match.market_p_away)
        actual = (match.actual_home_score, match.actual_away_score)
        pick_points = [
            jmlm_points((p.score_local, p.score_visit), actual)
            for p in portfolio.picks
        ]
        portfolio_picks_by_match.append(pick_points)

    # Suma por penca
    portfolio_total_by_penca = np.array(portfolio_picks_by_match).sum(axis=0)  # shape (5,)

    # Para benchmark chalk: pick EV-only en cada partido (sin diversificación)
    chalk_picks_by_match = [pps[0] for pps in portfolio_picks_by_match]   # penca 1 = EV pure ≈ chalk
    chalk_total = sum(chalk_picks_by_match)

    # Simular pool
    pool_results = []
    for s in range(n_simulations):
        pool_pts = simulate_pool(n_pool_players, matches, PoolModelConfig(), rng)
        pool_total = pool_pts.sum(axis=1)   # shape (n_players,)
        pool_results.append(int(pool_total.max()))

    pool_results_arr = np.array(pool_results)
    p_at_least_one = float((portfolio_total_by_penca.max() > pool_results_arr).mean())
    p_chalk = float((chalk_total > pool_results_arr).mean())

    return BacktestResult(
        n_matches=len(matches),
        n_simulations=n_simulations,
        portfolio_points_by_penca=[portfolio_total_by_penca.tolist()] * n_simulations,
        pool_top_points=pool_results,
        p_at_least_one_wins=p_at_least_one,
        p_chalk_baseline_wins=p_chalk,
    )


# -------------------- CLI --------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("tournament", help="ej: euro_2024")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--pool-size", type=int, default=150)
    args = ap.parse_args()

    matches = load_backtest_data(args.tournament, Path("data/backtest"))
    print(f"› Cargados {len(matches)} partidos de {args.tournament}")
    result = run_backtest(matches, n_simulations=args.sims, n_pool_players=args.pool_size)
    print(f"\n📊 Resultado del backtest:")
    print(f"   Partidos:                 {result.n_matches}")
    print(f"   Simulaciones del pool:    {result.n_simulations}")
    print(f"   Puntos por penca (1..5):  {result.portfolio_points_by_penca[0]}")
    print(f"   Puntos top del pool:      mediana={np.median(result.pool_top_points):.0f}  max={max(result.pool_top_points)}")
    print(f"")
    print(f"   P(al menos 1 penca gana al pool):  {result.p_at_least_one_wins:.1%}")
    print(f"   P(chalk solo gana al pool):        {result.p_chalk_baseline_wins:.1%}")
    print(f"   Uplift vs chalk:                   {result.p_at_least_one_wins - result.p_chalk_baseline_wins:+.1%}")

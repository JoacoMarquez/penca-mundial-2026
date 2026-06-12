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
from src.strategy.portfolio import generate_candidates, picks_to_dicts
from src.strategy.assignment import greedy_assignment

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
    n_pool_players: int
    n_pencas: int                                # cuántas pencas jugamos
    max_candidates: int                          # tamaño del menú de candidatos
    distinct_entries: int                        # entradas torneo-largas ÚNICAS entre las N pencas
    portfolio_points_by_penca: list[list[int]]   # [n_sims][n_pencas] — total points
    pool_top_points: list[int]                   # [n_sims] — max score del pool sintético
    pool_median_points: list[float]              # mediana del pool por sim
    portfolio_max_points: list[int]              # max de nuestras N pencas por sim
    portfolio_percentile_in_pool: list[float]    # en qué percentil cae nuestro max (0-100)
    p_at_least_one_wins: float                   # P(max(portfolio) > max(pool))
    p_top_1_percent: float                       # P(max(portfolio) está en top 1% del pool)
    p_top_5_percent: float                       # P(top 5%)
    p_top_10_percent: float                      # P(top 10%)
    p_chalk_baseline_wins: float                 # P(chalk también ganaría)
    p_at_least_one_wins_or_tie: float = 0.0      # P(max(portfolio) >= max(pool)) — incluye empates


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


def _precompute_matches(
    matches: list[BacktestMatch], pool_config: PoolModelConfig, max_candidates: int,
    with_pts_matrix: bool = False,
) -> list[dict]:
    """Precomputa por partido (1 vez): grilla, distribución del pool y menú de candidatos.

    Caro pero se hace UNA vez y se reusa entre todas las sims y todos los N del barrido.
    Si `with_pts_matrix`, agrega la matriz de puntos pick×resultado (para el Monte Carlo
    de resultados, donde el desenlace también se samplea).
    """
    pre = []
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
        n = grid.shape[0]
        nn = n * n
        pool_q = pool_pick_distribution(grid, pool_config)
        flat_q = pool_q.flatten()
        modal_idx = int(np.argmax(flat_q))
        top_idx = np.argsort(flat_q)[::-1][:5]
        top_p = flat_q[top_idx]
        top_p = top_p / top_p.sum()
        actual = (match.actual_home_score, match.actual_away_score)
        # Puntos que da CADA outcome (como pick) contra el resultado real — vector plano.
        pts_for_pick = np.array(
            [jmlm_points((k // n, k % n), actual) for k in range(nn)], dtype=int
        )
        cand_dicts = picks_to_dicts(
            generate_candidates(grid, match.market_p_home, match.market_p_away, max_candidates=max_candidates)
        )
        entry = {
            "grid": grid, "n": n, "modal_idx": modal_idx, "top_idx": top_idx,
            "top_p": top_p, "pts_for_pick": pts_for_pick, "cand_dicts": cand_dicts,
        }
        if with_pts_matrix:
            # grilla normalizada (para samplear el resultado) + matriz puntos[pick, resultado]
            flat_grid = grid.flatten()
            entry["flat_grid"] = flat_grid / flat_grid.sum()
            pts_matrix = np.zeros((nn, nn), dtype=np.int16)
            for pk in range(nn):
                pick = (pk // n, pk % n)
                for ok in range(nn):
                    pts_matrix[pk, ok] = jmlm_points(pick, (ok // n, ok % n))
            entry["pts_matrix"] = pts_matrix
        pre.append(entry)
    return pre


def _simulate_pool_fast(
    pre: list[dict], n_players: int, n_sims: int, rng: np.random.Generator,
    chalk_concentration: float = 0.70,
) -> np.ndarray:
    """Pool vectorizado: devuelve totales por (sim, jugador). Independiente de N."""
    pool_total = np.zeros((n_sims, n_players), dtype=int)
    for p in pre:
        is_chalk = rng.random((n_sims, n_players)) < chalk_concentration
        sampled = rng.choice(p["top_idx"], size=(n_sims, n_players), p=p["top_p"])
        idx = np.where(is_chalk, p["modal_idx"], sampled)
        pool_total += p["pts_for_pick"][idx]
    return pool_total


def _portfolio_for_n(pre: list[dict], n_pencas: int) -> tuple[np.ndarray, int, int, int]:
    """Asigna N pencas con el voraz (e_max) partido a partido. Determinístico.

    Returns: (total_by_penca, portfolio_max, distinct_entries, chalk_total).
    """
    penca_ids = list(range(1, n_pencas + 1))
    points_by_match, picks_by_match = [], []
    chalk_total = 0
    for p in pre:
        n = p["n"]
        assignment, _ = greedy_assignment(p["cand_dicts"], penca_ids, p["grid"], {})
        by_pid = {pid: pick for pid, pick, _ in assignment}
        row_pts, row_picks = [], []
        for pid in penca_ids:
            gL, gV = int(by_pid[pid]["score"][0]), int(by_pid[pid]["score"][1])
            row_pts.append(int(p["pts_for_pick"][gL * n + gV]))
            row_picks.append((gL, gV))
        points_by_match.append(row_pts)
        picks_by_match.append(row_picks)
        egL, egV = int(p["cand_dicts"][0]["score"][0]), int(p["cand_dicts"][0]["score"][1])
        chalk_total += int(p["pts_for_pick"][egL * n + egV])

    total_by_penca = np.array(points_by_match).sum(axis=0)
    sequences = {tuple(picks_by_match[j][i] for j in range(len(pre))) for i in range(n_pencas)}
    return total_by_penca, int(total_by_penca.max()), len(sequences), chalk_total


def _build_result(
    pre, n_simulations, n_pool_players, n_pencas, max_candidates,
    pool_total, total_by_penca, portfolio_max, distinct_entries, chalk_total,
) -> BacktestResult:
    pool_top = pool_total.max(axis=1)
    pool_median = np.median(pool_total, axis=1)
    percentile = (pool_total < portfolio_max).mean(axis=1) * 100
    return BacktestResult(
        n_matches=len(pre),
        n_simulations=n_simulations,
        n_pool_players=n_pool_players,
        n_pencas=n_pencas,
        max_candidates=max_candidates,
        distinct_entries=distinct_entries,
        portfolio_points_by_penca=[total_by_penca.tolist()] * n_simulations,
        pool_top_points=pool_top.tolist(),
        pool_median_points=pool_median.tolist(),
        portfolio_max_points=[portfolio_max] * n_simulations,
        portfolio_percentile_in_pool=percentile.tolist(),
        p_at_least_one_wins=float((portfolio_max > pool_top).mean()),
        p_top_1_percent=float((percentile >= 99).mean()),
        p_top_5_percent=float((percentile >= 95).mean()),
        p_top_10_percent=float((percentile >= 90).mean()),
        p_chalk_baseline_wins=float((chalk_total > pool_top).mean()),
    )


def run_backtest(
    matches: list[BacktestMatch],
    n_simulations: int = 100,
    n_pool_players: int = 150,
    seed: int = 42,
    n_pencas: int = 5,
    max_candidates: int = 8,
) -> BacktestResult:
    """Backtest para N pencas usando el camino real (candidates + greedy_assignment).

    Nota: usa el objetivo e_max (sin standings — régimen "primera fecha / sin datos de
    pool"). Es un LOWER BOUND del valor de N: el objetivo P(top-K), que diferencia más
    a las pencas extra durante el torneo, no se modela acá (requiere standings secuenciales).
    """
    rng = np.random.default_rng(seed)
    pre = _precompute_matches(matches, PoolModelConfig(), max_candidates)
    pool_total = _simulate_pool_fast(pre, n_pool_players, n_simulations, rng)
    total_by_penca, portfolio_max, distinct_entries, chalk_total = _portfolio_for_n(pre, n_pencas)
    return _build_result(
        pre, n_simulations, n_pool_players, n_pencas, max_candidates,
        pool_total, total_by_penca, portfolio_max, distinct_entries, chalk_total,
    )


def sweep_n(
    matches: list[BacktestMatch],
    n_values: Iterable[int],
    max_candidates: int = 8,
    n_simulations: int = 200,
    n_pool_players: int = 150,
    seed: int = 42,
) -> list[BacktestResult]:
    """Barre N pencas reusando precompute + pool (mismo pool para todos los N)."""
    rng = np.random.default_rng(seed)
    pre = _precompute_matches(matches, PoolModelConfig(), max_candidates)
    pool_total = _simulate_pool_fast(pre, n_pool_players, n_simulations, rng)
    results = []
    for n in n_values:
        total_by_penca, portfolio_max, distinct_entries, chalk_total = _portfolio_for_n(pre, n)
        results.append(_build_result(
            pre, n_simulations, n_pool_players, n, max_candidates,
            pool_total, total_by_penca, portfolio_max, distinct_entries, chalk_total,
        ))
    return results


# -------------------- Monte Carlo de RESULTADOS --------------------
# La diferencia clave: acá el desenlace de cada partido también se samplea (desde la grilla
# del modelo), en vez de usar el único resultado histórico. Portfolio Y pool se puntúan
# contra el MISMO torneo sampleado. Expone la diversificación a miles de mundos posibles —
# incluidos los batacazos — que es donde la diversificación cobra (o no).

def _mc_sample(
    pre: list[dict], n_players: int, n_sims: int, rng: np.random.Generator,
    chalk_concentration: float = 0.70,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Samplea n_sims torneos. Devuelve (outcomes_por_partido, pool_totals, chalk_total).

    pool_totals y chalk_total NO dependen de N → se computan una vez y se reusan en el barrido.
    """
    outcomes = []
    pool_totals = np.zeros((n_sims, n_players), dtype=int)
    chalk_total = np.zeros(n_sims, dtype=int)
    for p in pre:
        n = p["n"]
        nn = n * n
        pts = p["pts_matrix"]
        o = rng.choice(nn, size=n_sims, p=p["flat_grid"])   # resultado sampleado por sim
        outcomes.append(o)
        # Pool: cada jugador chalk o sampler del top-K, puntuado contra el resultado sampleado
        is_chalk = rng.random((n_sims, n_players)) < chalk_concentration
        sampled = rng.choice(p["top_idx"], size=(n_sims, n_players), p=p["top_p"])
        pool_pick = np.where(is_chalk, p["modal_idx"], sampled)
        pool_totals += pts[pool_pick, o[:, None]]
        # Benchmark chalk = ancla EV (candidato 0) jugada siempre
        ev_idx = int(p["cand_dicts"][0]["score"][0]) * n + int(p["cand_dicts"][0]["score"][1])
        chalk_total += pts[ev_idx, o]
    return outcomes, pool_totals, chalk_total


def _mc_portfolio(pre: list[dict], outcomes: list[np.ndarray], n_pencas: int) -> tuple[np.ndarray, int]:
    """Puntúa las N pencas contra los torneos sampleados. Devuelve (our_totals, distinct_entries)."""
    penca_ids = list(range(1, n_pencas + 1))
    n_sims = len(outcomes[0])
    our_totals = np.zeros((n_sims, n_pencas), dtype=int)
    per_match_picks = []
    for j, p in enumerate(pre):
        n = p["n"]
        pts = p["pts_matrix"]
        assignment, _ = greedy_assignment(p["cand_dicts"], penca_ids, p["grid"], {})
        by_pid = {pid: pick for pid, pick, _ in assignment}
        idx = np.array([
            int(by_pid[pid]["score"][0]) * n + int(by_pid[pid]["score"][1]) for pid in penca_ids
        ])
        per_match_picks.append(idx)
        our_totals += pts[idx][:, outcomes[j]].T   # (n_sims, n_pencas)
    sequences = {tuple(per_match_picks[j][i] for j in range(len(pre))) for i in range(n_pencas)}
    return our_totals, len(sequences)


def _build_mc_result(
    pre, n_sims, n_pool_players, n_pencas, max_candidates,
    our_totals, pool_totals, chalk_total, distinct_entries,
) -> BacktestResult:
    portfolio_max = our_totals.max(axis=1)
    pool_top = pool_totals.max(axis=1)
    percentile = (pool_totals < portfolio_max[:, None]).mean(axis=1) * 100
    return BacktestResult(
        n_matches=len(pre),
        n_simulations=n_sims,
        n_pool_players=n_pool_players,
        n_pencas=n_pencas,
        max_candidates=max_candidates,
        distinct_entries=distinct_entries,
        portfolio_points_by_penca=[np.round(our_totals.mean(axis=0), 1).tolist()],
        pool_top_points=pool_top.tolist(),
        pool_median_points=np.median(pool_totals, axis=1).tolist(),
        portfolio_max_points=portfolio_max.tolist(),
        portfolio_percentile_in_pool=percentile.tolist(),
        p_at_least_one_wins=float((portfolio_max > pool_top).mean()),
        p_top_1_percent=float((percentile >= 99).mean()),
        p_top_5_percent=float((percentile >= 95).mean()),
        p_top_10_percent=float((percentile >= 90).mean()),
        p_chalk_baseline_wins=float((chalk_total > pool_top).mean()),
        p_at_least_one_wins_or_tie=float((portfolio_max >= pool_top).mean()),
    )


def run_montecarlo(
    matches: list[BacktestMatch],
    n_sims: int = 2000,
    n_pool_players: int = 150,
    seed: int = 42,
    n_pencas: int = 5,
    max_candidates: int = 8,
) -> BacktestResult:
    """Monte Carlo de resultados para N pencas (samplea el desenlace de cada partido)."""
    rng = np.random.default_rng(seed)
    pre = _precompute_matches(matches, PoolModelConfig(), max_candidates, with_pts_matrix=True)
    outcomes, pool_totals, chalk_total = _mc_sample(pre, n_pool_players, n_sims, rng)
    our_totals, distinct = _mc_portfolio(pre, outcomes, n_pencas)
    return _build_mc_result(
        pre, n_sims, n_pool_players, n_pencas, max_candidates,
        our_totals, pool_totals, chalk_total, distinct,
    )


def sweep_n_montecarlo(
    matches: list[BacktestMatch],
    n_values: Iterable[int],
    max_candidates: int = 8,
    n_sims: int = 2000,
    n_pool_players: int = 150,
    seed: int = 42,
) -> list[BacktestResult]:
    """Barre N con el Monte Carlo de resultados (mismos torneos + pool para todos los N)."""
    rng = np.random.default_rng(seed)
    pre = _precompute_matches(matches, PoolModelConfig(), max_candidates, with_pts_matrix=True)
    outcomes, pool_totals, chalk_total = _mc_sample(pre, n_pool_players, n_sims, rng)
    results = []
    for n in n_values:
        our_totals, distinct = _mc_portfolio(pre, outcomes, n)
        results.append(_build_mc_result(
            pre, n_sims, n_pool_players, n, max_candidates,
            our_totals, pool_totals, chalk_total, distinct,
        ))
    return results


# -------------------- Backtest SECUENCIAL (standings + P(top-K)) --------------------
# Diferencia con run_montecarlo: procesa los partidos en orden y, en cada uno, asigna las
# N pencas con el objetivo P(top-3) usando los standings ACTUALES (que evolucionan por sim).
# Esto es lo que hace producción de la 2ª fecha en adelante — y es donde las pencas 11-15
# ganan valor (reciben desvíos de remontada dirigidos según cómo venís en la tabla).

def _greedy_alloc_fast(
    cand_pts_grid: np.ndarray,   # (K, G) puntos de cada candidato vs cada outcome de la grilla
    grid_probs: np.ndarray,      # (G,) probabilidad de cada outcome
    current_pts: np.ndarray,     # (N,) puntos acumulados de cada penca
    threshold: float | None,     # cutoff top-K del pool (total acumulado) o None → e_max
    modal_gain: np.ndarray,      # (G,) ganancia del pick modal del pool por outcome
    exact_probs: np.ndarray | None = None,  # (K,) P(exacto) de cada candidato
    w_exact: float = 0.0,        # peso del desempate por exactos en el objetivo secundario
) -> np.ndarray:
    """Voraz vectorizado: asigna cada penca al candidato que más sube P(top-K) (o E[max]).

    Con w_exact > 0, el objetivo secundario es E[max] + w·P(exacto): entre candidatos
    casi equivalentes en puntos, prefiere el que acumula más probabilidad de marcador
    exacto — la moneda del desempate del torneo.
    """
    K, G = cand_pts_grid.shape
    N = current_pts.shape[0]
    order = np.argsort(-current_pts, kind="stable")   # líder primero
    use_topk = threshold is not None
    thr = (threshold + modal_gain) if use_topk else None
    tie_bonus = (w_exact * exact_probs) if (w_exact and exact_probs is not None) else 0.0
    best_final = np.full(G, -1e9)
    assigned = np.empty(N, dtype=int)
    for pid in order:
        bf = np.maximum(best_final[None, :], current_pts[pid] + cand_pts_grid)  # (K, G)
        emax = bf @ grid_probs + tie_bonus   # (K,)
        if use_topk:
            key = ((bf >= thr[None, :]) @ grid_probs) * 1e6 + emax   # P(top-K) domina
        else:
            key = emax
        c = int(np.argmax(key))
        assigned[pid] = c
        best_final = bf[c]
    return assigned


def run_sequential_mc(
    matches: list[BacktestMatch],
    n_sims: int = 1000,
    n_pool_players: int = 150,
    seed: int = 42,
    n_pencas: int = 5,
    max_candidates: int = 10,
    chalk_concentration: float = 0.70,
    top_k: int = 3,
    w_exact: float = 0.0,
    horizon_alpha: float = 0.0,
    horizon_beta: float = 0.0,
) -> dict:
    """Monte Carlo SECUENCIAL: standings evolucionan jornada a jornada, asignación P(top-K).

    Resuelve el primer desempate del torneo (más marcadores exactos): p_win_resolved
    cuenta victoria por puntos O por exactos en caso de empate de puntos; el empate
    residual (puntos y exactos iguales) suma 0.5.
    """
    rng = np.random.default_rng(seed)
    pre = _precompute_matches(matches, PoolModelConfig(), max_candidates, with_pts_matrix=True)
    M = []
    for p in pre:
        n = p["n"]
        pts = p["pts_matrix"]
        cand_idx = np.array(
            [int(c["score"][0]) * n + int(c["score"][1]) for c in p["cand_dicts"]]
        )
        # Varianza por partido de los puntos de UN jugador del pool (para proyectar
        # cuánto se va a estirar el cutoff con los partidos que faltan).
        modal_row = pts[p["modal_idx"], :].astype(float)              # (G,)
        top_rows = pts[p["top_idx"], :].astype(float)                 # (T, G)
        c_ = chalk_concentration
        e_o = c_ * modal_row + (1 - c_) * (p["top_p"] @ top_rows)      # E[pts|outcome]
        e2_o = c_ * modal_row**2 + (1 - c_) * (p["top_p"] @ top_rows**2)
        mean_pts = float(p["flat_grid"] @ e_o)
        var_pts = float(p["flat_grid"] @ e2_o) - mean_pts**2
        M.append({
            "cand_pts_grid": pts[cand_idx, :].astype(float),
            "grid_probs": p["flat_grid"],
            "modal_gain": pts[p["modal_idx"], :].astype(float),
            "cand_idx": cand_idx,
            "exact_probs": p["flat_grid"][cand_idx],
            "pts": pts, "modal_idx": p["modal_idx"], "top_idx": p["top_idx"],
            "top_p": p["top_p"], "flat_grid": p["flat_grid"], "G": n * n,
            "pool_var": max(var_pts, 0.0),
        })

    # Suma de varianza RESTANTE (incluyendo el partido actual) en cada índice.
    suffix_var = np.zeros(len(M) + 1)
    for t in range(len(M) - 1, -1, -1):
        suffix_var[t] = suffix_var[t + 1] + M[t]["pool_var"]

    N, Pn = n_pencas, n_pool_players
    win = win_tie = 0
    win_resolved = 0.0
    pmax_acc = ptop_acc = 0.0
    for _ in range(n_sims):
        our_pts = np.zeros(N)
        pool_pts = np.zeros(Pn)
        our_ex = np.zeros(N, dtype=int)
        pool_ex = np.zeros(Pn, dtype=int)
        for t, m in enumerate(M):
            # Cutoff top-K del pool con los standings ACTUALES (antes de este partido)
            if Pn >= top_k:
                threshold = float(np.partition(pool_pts, -top_k)[-top_k])
            else:
                threshold = float(pool_pts.max()) if Pn else 0.0
            # HORIZONTE: premium = α·√(varianza restante del pool). Temprano (mucho por
            # jugar) sube el listón → fuerza picks más agresivos; →0 al final del torneo.
            if horizon_alpha > 0:
                threshold += horizon_alpha * float(np.sqrt(suffix_var[t]))
            # Forma de PRODUCCIÓN: β·√(partidos restantes) — no requiere odds futuras
            # (absorbe σ̄ en β). Validar acá que rinde como la forma α.
            if horizon_beta > 0:
                threshold += horizon_beta * float(np.sqrt(len(M) - t))
            assigned = _greedy_alloc_fast(
                m["cand_pts_grid"], m["grid_probs"], our_pts, threshold, m["modal_gain"],
                exact_probs=m["exact_probs"], w_exact=w_exact,
            )
            o = int(rng.choice(m["G"], p=m["flat_grid"]))         # resultado real sampleado
            our_pts = our_pts + m["cand_pts_grid"][assigned, o]   # nuestras pencas suman
            our_ex = our_ex + (m["cand_idx"][assigned] == o)
            # Pool: cada jugador chalk o sampler, suma vs el resultado
            is_chalk = rng.random(Pn) < chalk_concentration
            sampled = rng.choice(m["top_idx"], size=Pn, p=m["top_p"])
            pool_pick = np.where(is_chalk, m["modal_idx"], sampled)
            pool_pts = pool_pts + m["pts"][pool_pick, o]
            pool_ex = pool_ex + (pool_pick == o)
        pmax, ptop = our_pts.max(), pool_pts.max()
        win += pmax > ptop
        win_tie += pmax >= ptop
        # Desempate del torneo: puntos, después exactos. Residual → 0.5.
        our_best_ex = int(our_ex[our_pts == pmax].max())
        pool_best_ex = int(pool_ex[pool_pts == ptop].max())
        if (pmax, our_best_ex) > (ptop, pool_best_ex):
            win_resolved += 1.0
        elif (pmax, our_best_ex) == (ptop, pool_best_ex):
            win_resolved += 0.5
        pmax_acc += pmax
        ptop_acc += ptop

    return {
        "n_pencas": N,
        "max_candidates": max_candidates,
        "n_sims": n_sims,
        "w_exact": w_exact,
        "horizon_alpha": horizon_alpha,
        "p_win": win / n_sims,
        "p_win_tie": win_tie / n_sims,
        "p_win_resolved": win_resolved / n_sims,
        "portfolio_max_mean": pmax_acc / n_sims,
        "pool_top_mean": ptop_acc / n_sims,
    }


def sweep_sequential(
    matches: list[BacktestMatch],
    n_values: Iterable[int],
    max_candidates: int = 10,
    n_sims: int = 1000,
    n_pool_players: int = 150,
    seed: int = 42,
) -> list[dict]:
    """Barre N con el backtest secuencial (mismo seed → mismos torneos/pool para todos los N)."""
    return [
        run_sequential_mc(
            matches, n_sims=n_sims, n_pool_players=n_pool_players,
            seed=seed, n_pencas=n, max_candidates=max_candidates,
        )
        for n in n_values
    ]


# -------------------- CLI --------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("tournament", help="ej: euro_2024")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--pool-size", type=int, default=150)
    ap.add_argument("--pencas", type=int, default=5, help="N pencas a jugar")
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--sweep", default="", help="CSV de N a barrer, ej: 5,10,15,20")
    ap.add_argument("--mc", action="store_true",
                    help="Usa el Monte Carlo de RESULTADOS (samplea desenlaces, no usa el histórico)")
    ap.add_argument("--seq", action="store_true",
                    help="Backtest SECUENCIAL: standings evolucionan + objetivo P(top-3) por partido")
    ap.add_argument("--sweep-w", default="",
                    help="CSV de pesos w_exact a barrer en el secuencial (desempate por exactos), ej: 0,1,2,5,10")
    ap.add_argument("--sweep-horizon", default="",
                    help="CSV de horizon_alpha a barrer (cutoff con varianza restante), ej: 0,0.25,0.5,1,2")
    args = ap.parse_args()

    if args.sweep_w:
        matches_w = load_backtest_data(args.tournament, Path("data/backtest"))
        n_sims = args.sims if args.sims != 100 else 1500
        print(f"\n📊 Barrido w_exact (desempate por exactos) | N={args.pencas} pool={args.pool_size} sims={n_sims}")
        print(f"   {'w':>5}  {'P(gana pts)':>11}  {'P(gana c/desempate)':>19}  {'nuestro max':>11}")
        for w in [float(x) for x in args.sweep_w.split(",")]:
            r = run_sequential_mc(
                matches_w, n_sims=n_sims, n_pool_players=args.pool_size,
                n_pencas=args.pencas, max_candidates=args.max_candidates, w_exact=w,
            )
            print(f"   {w:>5.1f}  {r['p_win']:>10.1%}  {r['p_win_resolved']:>18.1%}  {r['portfolio_max_mean']:>11.1f}")
        import sys as _sys
        _sys.exit(0)

    if args.sweep_horizon:
        matches_h = load_backtest_data(args.tournament, Path("data/backtest"))
        n_sims = args.sims if args.sims != 100 else 1500
        print(f"\n📊 Barrido horizon_alpha (cutoff con varianza restante) | N={args.pencas} pool={args.pool_size} sims={n_sims}")
        print(f"   {'α':>5}  {'P(gana pts)':>11}  {'P(gana c/desempate)':>19}  {'nuestro max':>11}  {'pool top':>9}")
        for a in [float(x) for x in args.sweep_horizon.split(",")]:
            r = run_sequential_mc(
                matches_h, n_sims=n_sims, n_pool_players=args.pool_size,
                n_pencas=args.pencas, max_candidates=args.max_candidates, horizon_alpha=a,
            )
            print(f"   {a:>5.2f}  {r['p_win']:>10.1%}  {r['p_win_resolved']:>18.1%}  {r['portfolio_max_mean']:>11.1f}  {r['pool_top_mean']:>9.1f}")
        import sys as _sys
        _sys.exit(0)

    matches = load_backtest_data(args.tournament, Path("data/backtest"))
    print(f"› Cargados {len(matches)} partidos de {args.tournament}")

    if args.sweep:
        n_values = [int(x) for x in args.sweep.split(",")]
        if args.seq:
            n_sims = args.sims if args.sims != 100 else 1000
            results = sweep_sequential(
                matches, n_values, max_candidates=args.max_candidates,
                n_sims=n_sims, n_pool_players=args.pool_size,
            )
            print(f"\n📊 Barrido SECUENCIAL (standings + P(top-3)) (pool={args.pool_size}, sims={n_sims}, menú={args.max_candidates})")
            print(f"   {'N':>4}  {'P(gana)':>9}  {'P(g/empat)':>10}  {'nuestro max':>11}  {'pool top':>9}")
            print(f"   {'-'*4}  {'-'*9}  {'-'*10}  {'-'*11}  {'-'*9}")
            for r in results:
                print(f"   {r['n_pencas']:>4}  {r['p_win']:>8.1%}  {r['p_win_tie']:>9.1%}  "
                      f"{r['portfolio_max_mean']:>11.1f}  {r['pool_top_mean']:>9.1f}")
            import sys as _sys
            _sys.exit(0)
        if args.mc:
            n_sims = args.sims if args.sims != 100 else 2000
            results = sweep_n_montecarlo(
                matches, n_values, max_candidates=args.max_candidates,
                n_sims=n_sims, n_pool_players=args.pool_size,
            )
            print(f"\n📊 Barrido MONTE CARLO de resultados (pool={args.pool_size}, sims={n_sims}, menú={args.max_candidates})")
            print(f"   {'N':>4}  {'distintas':>9}  {'P(gana)':>9}  {'P(g/empat)':>10}  {'P(top5%)':>9}  {'vs chalk':>9}")
            print(f"   {'-'*4}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*9}")
            chalk_ref = results[0].p_chalk_baseline_wins
            for r in results:
                print(f"   {r.n_pencas:>4}  {r.distinct_entries:>9}  {r.p_at_least_one_wins:>8.1%}  "
                      f"{r.p_at_least_one_wins_or_tie:>9.1%}  {r.p_top_5_percent:>8.1%}  "
                      f"{r.p_at_least_one_wins - chalk_ref:>+8.1%}")
            print(f"\n   Benchmark chalk (1 entrada EV): gana {chalk_ref:.1%}")
        else:
            results = sweep_n(
                matches, n_values, max_candidates=args.max_candidates,
                n_simulations=args.sims, n_pool_players=args.pool_size,
            )
            print(f"\n📊 Barrido ESTÁTICO (1 resultado histórico) (pool={args.pool_size}, sims={args.sims}, menú={args.max_candidates})")
            print(f"   {'N':>4}  {'distintas':>9}  {'P(gana)':>9}  {'P(top5%)':>9}  {'vs chalk':>9}")
            print(f"   {'-'*4}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
            chalk_ref = results[0].p_chalk_baseline_wins
            for r in results:
                print(f"   {r.n_pencas:>4}  {r.distinct_entries:>9}  {r.p_at_least_one_wins:>8.1%}  "
                      f"{r.p_top_5_percent:>8.1%}  {r.p_at_least_one_wins - chalk_ref:>+8.1%}")
            print(f"\n   Benchmark chalk (1 entrada EV): {chalk_ref:.1%}")
        import sys as _sys
        _sys.exit(0)

    result = run_backtest(
        matches, n_simulations=args.sims, n_pool_players=args.pool_size,
        n_pencas=args.pencas, max_candidates=args.max_candidates,
    )
    print(f"\n📊 Resultado del backtest:")
    print(f"   Partidos:                 {result.n_matches}")
    print(f"   Pool size:                {result.n_pool_players}")
    print(f"   Simulaciones:             {result.n_simulations}")
    print()
    print(f"   Puntos por penca (1..5):  {result.portfolio_points_by_penca[0]}")
    print(f"   Nuestro MAX:              {result.portfolio_max_points[0]}")
    print()
    print(f"   Pool mediana:             {np.median(result.pool_median_points):.0f}")
    print(f"   Pool top mediana:         {np.median(result.pool_top_points):.0f}")
    print(f"   Pool top max:             {max(result.pool_top_points)}")
    print()
    print(f"   Percentil promedio nuestro:  {np.mean(result.portfolio_percentile_in_pool):.1f}")
    print()
    print(f"   P(ganamos el pool):           {result.p_at_least_one_wins:.1%}")
    print(f"   P(top 1%):                    {result.p_top_1_percent:.1%}")
    print(f"   P(top 5%):                    {result.p_top_5_percent:.1%}")
    print(f"   P(top 10%):                   {result.p_top_10_percent:.1%}")
    print()
    print(f"   Benchmark chalk solo gana:    {result.p_chalk_baseline_wins:.1%}")
    print(f"   Uplift vs chalk:              {result.p_at_least_one_wins - result.p_chalk_baseline_wins:+.1%}")

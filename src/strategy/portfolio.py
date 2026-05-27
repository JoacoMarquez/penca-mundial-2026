"""Generación de las 5 picks por partido — portfolio de perturbaciones óptimas.

Cada penca apunta a un objetivo distinto sobre el mismo modelo probabilístico:
    1. EV puro: argmax E[my_pts | pick]
    2. Differentiated EV: argmax (E[my_pts | pick] * (1 - Q(pick))^β) — castiga picks populares
    3. Tail-max: argmax E[my_pts | pick, top-K% scoring outcomes] — apuesta a partidos con muchos goles
    4. Conditional alternative: argmax E[my_pts | pick, ganador opuesto a favorito]
    5. Variance-max forzando diversidad: entre top-K por EV no usadas, max Var[points]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model.poisson import jmlm_points
from src.meta.pool import PoolModelConfig, pool_pick_distribution


@dataclass(frozen=True)
class PickWithMetrics:
    """Una pick con sus métricas para auditing/notif."""
    score_local: int
    score_visit: int
    objective: str           # "ev" | "differentiated" | "tail" | "upset" | "variance"
    e_points: float          # E[my_pts | grid]
    e_pool_points: float     # E[pool_typical_pts | grid]
    uplift: float            # e_points - e_pool_points
    variance: float          # Var[my_pts | grid]
    pool_popularity: float   # Q(esta pick)


MIN_REALISTIC_PROB = 0.005   # 0.5% — scorelines más improbables que esto se descartan como candidatos
MAX_REALISTIC_GOALS = 5      # ningún equipo va a meter 6+ goles — descarta candidatos absurdos
DIFFERENTIATED_TOP_K = 8     # differentiated solo elige entre los top-K más probables (evita 5-0)
TAIL_TOP_K = 8               # idem para tail


def all_picks_with_metrics(
    grid: np.ndarray,
    pool_q: np.ndarray,
    points_rule=jmlm_points,
    restrict_realistic: bool = True,
) -> list[dict]:
    """Pre-calcula métricas para cada scoreline posible. Retorna lista ordenada por E[pts] desc.

    Si `restrict_realistic`, descarta scorelines con P < MIN_REALISTIC_PROB o goles > MAX_REALISTIC_GOALS.
    Esto evita picks absurdas como 7-0 cuando differentiated/tail las prefieren por la matemática.
    """
    n = grid.shape[0]
    metrics = []

    # Pre-compute pool average points por outcome (1 vez)
    pool_pts_by_actual = np.zeros((n, n))
    for agL in range(n):
        for agV in range(n):
            for pgL in range(n):
                for pgV in range(n):
                    pool_pts_by_actual[agL, agV] += pool_q[pgL, pgV] * points_rule((pgL, pgV), (agL, agV))

    e_pool = float((grid * pool_pts_by_actual).sum())

    for pgL in range(n):
        for pgV in range(n):
            if restrict_realistic:
                if pgL > MAX_REALISTIC_GOALS or pgV > MAX_REALISTIC_GOALS:
                    continue
                # Probabilidad de que ese marcador efectivamente ocurra
                if grid[pgL, pgV] < MIN_REALISTIC_PROB:
                    continue
            pick = (pgL, pgV)
            pts_by_actual = np.array([
                [points_rule(pick, (agL, agV)) for agV in range(n)]
                for agL in range(n)
            ])
            e_pts = float((grid * pts_by_actual).sum())
            e_pts_sq = float((grid * pts_by_actual ** 2).sum())
            var = max(e_pts_sq - e_pts ** 2, 0.0)

            metrics.append({
                "pick": pick,
                "e_points": e_pts,
                "e_pool_points": e_pool,    # mismo para todos (no depende de mi pick)
                "uplift": e_pts - e_pool,
                "variance": var,
                "pool_popularity": float(pool_q[pgL, pgV]),
                "p_scoreline": float(grid[pgL, pgV]),
            })

    metrics.sort(key=lambda d: -d["e_points"])
    return metrics


def winner_of(pick: tuple[int, int]) -> str:
    """1 / X / 2."""
    gL, gV = pick
    if gL > gV: return "1"
    if gL < gV: return "2"
    return "X"


def pick_ev(metrics: list[dict]) -> dict:
    """Objetivo 1: argmax E[my_pts]."""
    return max(metrics, key=lambda d: d["e_points"])


def pick_differentiated(
    metrics: list[dict],
    beta: float = 2.0,
    excluded: set[tuple[int, int]] | None = None,
    top_k: int = DIFFERENTIATED_TOP_K,
) -> dict:
    """Objetivo 2: argmax E[my_pts] * (1 - Q(pick))^β, RESTRICTO a top-K más probables.

    Castiga picks populares pero solo dentro de scorelines realistas (top-K por P(scoreline)).
    Sin top_k cap, esto preferiría 5-0/6-1 — matemáticamente óptimo pero no humanos.
    """
    excluded = excluded or set()
    # Top-K por probabilidad de marcador (no por E[points])
    by_prob = sorted(metrics, key=lambda d: -d.get("p_scoreline", 0.0))
    candidates = [d for d in by_prob[:top_k] if d["pick"] not in excluded]
    if not candidates:
        candidates = [d for d in metrics if d["pick"] not in excluded]

    def score(d: dict) -> float:
        return d["e_points"] * (1.0 - d["pool_popularity"]) ** beta
    return max(candidates or metrics, key=score)


def pick_tail(
    grid: np.ndarray,
    pool_q: np.ndarray,
    top_k_percent: float = 0.10,
    points_rule=jmlm_points,
    excluded: set[tuple[int, int]] | None = None,
) -> dict:
    """Objetivo 3: argmax E[my_pts | pick, top-K% outcomes por total de goles].

    Condicional en outcomes con muchos goles (cola alta), qué pick rinde más.
    """
    n = grid.shape[0]

    # Identificar el top-K% outcomes por total de goles (con desempate por probabilidad)
    outcomes = []
    for agL in range(n):
        for agV in range(n):
            outcomes.append(((agL, agV), agL + agV, grid[agL, agV]))
    # ordenar por goles totales desc, luego prob desc
    outcomes.sort(key=lambda t: (-t[1], -t[2]))
    k = max(1, int(len(outcomes) * top_k_percent))
    tail_outcomes = outcomes[:k]
    tail_mass = sum(t[2] for t in tail_outcomes)
    if tail_mass == 0:
        # Fallback raro: cae a EV-max
        return pick_ev(all_picks_with_metrics(grid, pool_q, points_rule))

    excluded = excluded or set()
    # Construir set de picks realistas: top-K por P(scoreline) global
    flat = []
    for pgL in range(n):
        for pgV in range(n):
            if pgL <= MAX_REALISTIC_GOALS and pgV <= MAX_REALISTIC_GOALS:
                flat.append((grid[pgL, pgV], (pgL, pgV)))
    flat.sort(reverse=True)
    top_k_picks = {pick for _, pick in flat[:TAIL_TOP_K]}

    # E[my_pts | pick, tail] = (1/tail_mass) * sum_{tail} P(actual) * points(pick, actual)
    best = None
    for pgL in range(n):
        for pgV in range(n):
            pick = (pgL, pgV)
            if pick not in top_k_picks:
                continue
            if pick in excluded:
                continue
            e = sum(
                p * points_rule(pick, actual) for actual, _, p in tail_outcomes
            ) / tail_mass
            if best is None or e > best[0]:
                # también guardo métricas estándar para consistencia
                e_pts = sum(grid[agL, agV] * points_rule(pick, (agL, agV)) for agL in range(n) for agV in range(n))
                e_pool = 0.0
                for agL in range(n):
                    for agV in range(n):
                        for ppL in range(n):
                            for ppV in range(n):
                                e_pool += grid[agL, agV] * pool_q[ppL, ppV] * points_rule((ppL, ppV), (agL, agV))
                e_pts_sq = sum(grid[agL, agV] * points_rule(pick, (agL, agV)) ** 2 for agL in range(n) for agV in range(n))
                var = max(e_pts_sq - e_pts ** 2, 0.0)
                best = (e, {
                    "pick": pick,
                    "e_points": e_pts,
                    "e_points_in_tail": e,
                    "e_pool_points": e_pool,
                    "uplift": e_pts - e_pool,
                    "variance": var,
                    "pool_popularity": float(pool_q[pgL, pgV]),
                })
    return best[1]


def pick_upset(
    metrics: list[dict],
    market_p_home: float,
    market_p_away: float,
) -> dict:
    """Objetivo 4: si el favorito tiene 1X2 > 0.50, jugar el ganador opuesto.

    Si la pick EV-max ya implica al underdog, ya hay variance natural — caemos al mejor pick alternativo:
    el segundo-más-EV con ganador contrario al EV-max.
    """
    if market_p_home >= 0.50 and market_p_home > market_p_away:
        # favorito = local. El upset es: empate o visitante.
        candidates = [d for d in metrics if winner_of(d["pick"]) in ("X", "2")]
    elif market_p_away >= 0.50 and market_p_away > market_p_home:
        # favorito = visitante. Upset = empate o local.
        candidates = [d for d in metrics if winner_of(d["pick"]) in ("X", "1")]
    else:
        # partido parejo. No hay favorito claro. El "upset alternativo" es el mejor empate.
        candidates = [d for d in metrics if winner_of(d["pick"]) == "X"]

    if not candidates:
        return metrics[0]  # fallback al EV-max
    return candidates[0]   # ya están ordenados por e_points desc


def pick_variance_diversified(
    metrics: list[dict],
    already_chosen: set[tuple[int, int]],
    top_k: int = 8,
) -> dict:
    """Objetivo 5: entre el top-K por EV que NO estén en `already_chosen`, max Var[my_pts]."""
    candidates = [d for d in metrics if d["pick"] not in already_chosen][:top_k]
    if not candidates:
        # fallback: cualquier pick no usada
        candidates = [d for d in metrics if d["pick"] not in already_chosen]
    if not candidates:
        # extremo: si todos los picks ya fueron usados (no debería pasar)
        return metrics[-1]
    return max(candidates, key=lambda d: d["variance"])


@dataclass(frozen=True)
class PortfolioResult:
    picks: list[PickWithMetrics]   # exactamente 5

    def to_dict(self) -> dict:
        return {
            "picks": [
                {
                    "penca_index": i + 1,
                    "objective": p.objective,
                    "score": [p.score_local, p.score_visit],
                    "e_points": round(p.e_points, 3),
                    "e_pool_points": round(p.e_pool_points, 3),
                    "uplift": round(p.uplift, 3),
                    "variance": round(p.variance, 3),
                    "pool_popularity": round(p.pool_popularity, 4),
                }
                for i, p in enumerate(self.picks)
            ]
        }


def generate_portfolio(
    grid: np.ndarray,
    market_p_home: float,
    market_p_away: float,
    pool_config: PoolModelConfig | None = None,
    points_rule=jmlm_points,
) -> PortfolioResult:
    """Genera las 5 picks ejecutando los 5 objetivos en orden.

    Permite que objetivo 4 y 5 caigan a fallbacks si las picks anteriores ya cubrieron sus targets.
    """
    if pool_config is None:
        pool_config = PoolModelConfig()
    pool_q = pool_pick_distribution(grid, pool_config)

    metrics = all_picks_with_metrics(grid, pool_q, points_rule)

    # 1: EV puro
    m1 = pick_ev(metrics)
    # 2: differentiated EV — excluyendo el #1
    m2 = pick_differentiated(metrics, excluded={m1["pick"]})
    # 3: tail-max — excluyendo los anteriores
    m3 = pick_tail(grid, pool_q, points_rule=points_rule, excluded={m1["pick"], m2["pick"]})
    # 4: upset / conditional alternative — el upset filter ya restringe ganador, dedupe se hace post
    m4 = pick_upset(metrics, market_p_home, market_p_away)

    # Si 4 coincide con 1, intentamos el segundo mejor "alternativo"
    if m4["pick"] == m1["pick"]:
        already = {m1["pick"], m2["pick"], m3["pick"]}
        alt_metrics = [d for d in metrics if d["pick"] not in already]
        # buscar el siguiente con winner opuesto al de m1
        m1_winner = winner_of(m1["pick"])
        opp_metrics = [d for d in alt_metrics if winner_of(d["pick"]) != m1_winner]
        m4 = opp_metrics[0] if opp_metrics else alt_metrics[0]

    # 5: variance-max diversified (entre top-K que no estén en {1,2,3,4})
    already = {m["pick"] for m in (m1, m2, m3, m4)}
    m5 = pick_variance_diversified(metrics, already)

    chosen = [m1, m2, m3, m4, m5]
    objectives = ["ev", "differentiated", "tail", "upset", "variance"]

    picks = []
    for obj, m in zip(objectives, chosen):
        gL, gV = m["pick"]
        picks.append(PickWithMetrics(
            score_local=gL,
            score_visit=gV,
            objective=obj,
            e_points=m["e_points"],
            e_pool_points=m["e_pool_points"],
            uplift=m["uplift"],
            variance=m["variance"],
            pool_popularity=m["pool_popularity"],
        ))

    return PortfolioResult(picks=picks)


if __name__ == "__main__":
    # Smoke: España vs Cabo Verde
    from src.model.poisson import score_grid, marginals
    g = score_grid(2.5, 0.5, 0.05)
    m = marginals(g)
    print(f"Mercado: P(home)={m.p_home_win:.3f}  P(draw)={m.p_draw:.3f}  P(away)={m.p_away_win:.3f}")
    portfolio = generate_portfolio(g, m.p_home_win, m.p_away_win)
    for p in portfolio.picks:
        print(
            f"  [{p.objective:14s}] {p.score_local}-{p.score_visit}  "
            f"E[pts]={p.e_points:.2f}  uplift={p.uplift:+.2f}  Var={p.variance:.2f}  pop={p.pool_popularity:.3f}"
        )

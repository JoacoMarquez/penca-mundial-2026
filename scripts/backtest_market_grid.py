"""Backtest del refinamiento con mercados ricos (src/clausura/market_grid.py).

Supermatch no expone odds históricas, así que el experimento es de RECUPERACIÓN DE
MODELO, el mismo diseño que --experimento-rivales:

  1. **Verdad por partido**: mezcla (1−ε)·Poisson(λ de ratings out-of-sample) +
     ε·distribución empírica de marcadores de las temporadas PREVIAS. El ε se ajusta
     por MLE sobre las temporadas previas (out-of-sample respecto de la evaluada):
     captura lo que el Poisson independiente no puede — el exceso real de 0-0/1-1
     de esta liga — que es exactamente la información que un mercado bien calibrado
     cotiza y el 1X2 solo no revela.
  2. **Mercados sintéticos desde la verdad** con vig realista (overrounds medidos en
     la fixture real del 2026-08-04: 1X2 ~1.11, over ~1.20, goles ~1.20, marcador
     exacto ~1.25 con top-12 celdas + 'otro') y ruido lognormal σ=2% en las cuotas.
  3. **Brazo A (statu quo)**: λ del 1X2+over de-vigueado, blend 70/30 con ratings →
     Poisson. **Brazo B**: la misma grilla refinada con goles exactos + marcador
     exacto (refine_grid). Ambos ven los mismos mercados con el mismo ruido.

Métricas por temporada (4 evaluables):
  * KL(verdad ‖ grilla) media — ¿quién recupera mejor la distribución?
  * log-loss del RESULTADO REAL — la verdad es sintética, pero los 598 resultados
    no: si ε capta estructura real, B tiene que predecir mejor los partidos reales.
  * pick óptimo: % de partidos donde cambia y ΔE[pts] bajo la verdad.
  * E[premio] de los portfolios de 12 participaciones construidos con cada grilla,
    evaluados bajo la verdad con semilla fresca (común a ambos brazos).

Uso:
    python -m scripts.backtest_market_grid [--sims 800] [--eval-sims 4000]
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from src.clausura.economics import (
    MAX_GOALS,
    PrizeConfig,
    SeasonSimulator,
    SimConfig,
    flatten_grid,
    index_score,
    score_index,
)
from src.clausura.historical import PartidoHistorico, load_dataset
from src.clausura.market_grid import GOAL_BUCKETS, refine_grid
from src.clausura.odds import EventOdds
from src.clausura.picks import MARKET_WEIGHT, market_lambdas
from src.clausura.pool import PoolConfig, pool_distribution
from src.clausura.ratings import fit_ratings
from src.clausura.scoring import expected_points_grid
from src.clausura.strategy import EVAL_SEED_OFFSET, build_portfolio
from src.model.poisson import score_grid

log = logging.getLogger(__name__)

SIDE = MAX_GOALS + 1
LIGA_LAM = (1.28, 1.11)
OR_1X2, OR_OVER, OR_GOALS, OR_SCORE = 1.11, 1.20, 1.20, 1.25
NOISE_SIGMA = 0.02
TOP_SCORE_CELLS = 12


# -------------------- verdad: Poisson + inflación empírica --------------------

def empirical_dist(partidos: list[PartidoHistorico]) -> np.ndarray:
    """Distribución empírica de marcadores (suavizado Laplace 0.5)."""
    c = np.full((SIDE, SIDE), 0.5)
    for p in partidos:
        c[min(p.goles_local, MAX_GOALS), min(p.goles_visitante, MAX_GOALS)] += 1
    return c / c.sum()


def fit_epsilon(previas: list[PartidoHistorico], ratings) -> float:
    """MLE de ε en (1−ε)·Poisson(λ del partido) + ε·empírica, sobre las previas.

    CLAVE: el Poisson de la mezcla es POR PARTIDO (λ de ratings), no el de liga —
    si no, ε absorbe la heterogeneidad entre partidos en vez del exceso de
    empates/estructura que el Poisson independiente no capta, y clava en el tope.
    (Los λ de las previas usan ratings ajustados sobre esas mismas previas: leve
    sesgo a favor del Poisson → ε conservador, que es el lado seguro acá.)
    """
    emp = empirical_dist(previas)
    bases = [score_grid(*ratings.lambdas(p.local, p.visitante), 0.0, max_goals=MAX_GOALS)
             for p in previas]
    cells = [(min(p.goles_local, MAX_GOALS), min(p.goles_visitante, MAX_GOALS))
             for p in previas]
    lls = []
    grid_eps = np.linspace(0.0, 0.6, 25)
    for eps in grid_eps:
        ll = sum(np.log((1 - eps) * b[c] + eps * emp[c]) for b, c in zip(bases, cells))
        lls.append(ll)
    return float(grid_eps[int(np.argmax(lls))])


def truth_grid(lam_l: float, lam_v: float, eps: float, emp: np.ndarray) -> np.ndarray:
    g = (1 - eps) * score_grid(lam_l, lam_v, 0.0, max_goals=MAX_GOALS) + eps * emp
    return g / g.sum()


# -------------------- mercados sintéticos con vig --------------------

def _vig(probs: dict[str, float], overround: float, rng) -> dict[str, float]:
    """probs → cuotas con overround proporcional + ruido lognormal."""
    out = {}
    for k, p in probs.items():
        odds = 1.0 / max(p * overround, 1e-9) * float(np.exp(rng.normal(0, NOISE_SIGMA)))
        out[k] = max(odds, 1.01)
    return out


def synth_odds(truth: np.ndarray, rng) -> EventOdds:
    """EventOdds sintéticas desde la grilla verdad (lo que cotizaría la casa)."""
    ph = float(np.tril(truth, -1).sum())        # local gana: gL > gV
    pa = float(np.triu(truth, 1).sum())
    pd = float(np.trace(truth))
    over = float(sum(truth[i, j] for i in range(SIDE) for j in range(SIDE) if i + j > 2.5))

    marg_h = truth.sum(axis=1)
    marg_a = truth.sum(axis=0)
    goals = lambda m: dict(zip(GOAL_BUCKETS, [m[0], m[1], m[2], float(m[3:].sum())]))

    flat = flatten_grid(truth)
    top = np.argsort(-flat)[:TOP_SCORE_CELLS]
    cs_probs = {f"{gl}:{gv}": float(flat[i]) for i in top
                for gl, gv in [index_score(int(i))]}
    cs_probs["otro"] = max(1.0 - sum(cs_probs.values()), 1e-6)

    return EventOdds(
        event_id="synth", home="H", away="A", start_utc="", fetched_utc="",
        x1x2=_vig({"home": ph, "draw": pd, "away": pa}, OR_1X2, rng),
        totals={"2.5": _vig({"over": over, "under": 1 - over}, OR_OVER, rng)},
        correct_score=_vig(cs_probs, OR_SCORE, rng),
        home_goals=_vig(goals(marg_h), OR_GOALS, rng),
        away_goals=_vig(goals(marg_a), OR_GOALS, rng),
    )


# -------------------- brazos --------------------

def arm_grids(odds: list[EventOdds], lams_rt: list[tuple[float, float]],
              rich: bool) -> list[np.ndarray]:
    """Brazo A (rich=False): 1X2+over → Poisson blend. Brazo B: + refine_grid."""
    out = []
    for o, (rl, rv) in zip(odds, lams_rt):
        mkt = market_lambdas(o)
        lam_l = MARKET_WEIGHT * mkt[0] + (1 - MARKET_WEIGHT) * rl
        lam_v = MARKET_WEIGHT * mkt[1] + (1 - MARKET_WEIGHT) * rv
        g = score_grid(lam_l, lam_v, 0.0, max_goals=MAX_GOALS)
        if rich:
            g, _ = refine_grid(g, o)
        out.append(g)
    return out


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float((p * np.log((p + 1e-15) / (q + 1e-15))).sum())


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--participaciones", type=int, default=12)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--eval-sims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()

    data = load_dataset()
    temporadas: dict[str, list[PartidoHistorico]] = {}
    for p in data:
        temporadas.setdefault(p.campeonato, []).append(p)
    orden = sorted(temporadas, key=lambda t: min(p.inicio_utc for p in temporadas[t]))

    filas = []
    for nombre in orden:
        previas = [p for t in orden[: orden.index(nombre)] for p in temporadas[t]]
        if len(previas) < 100:
            print(f"(salteada {nombre}: sin temporadas previas)")
            continue
        partidos = sorted(temporadas[nombre], key=lambda p: p.inicio_utc)
        ratings = fit_ratings(previas)
        eps = fit_epsilon(previas, ratings)
        emp = empirical_dist(previas)
        rng = np.random.default_rng(args.seed + hash(nombre) % 10_000)

        lams_rt = [ratings.lambdas(p.local, p.visitante) for p in partidos]
        truths = [truth_grid(rl, rv, eps, emp) for rl, rv in lams_rt]
        odds = [synth_odds(t, rng) for t in truths]

        grids_a = arm_grids(odds, lams_rt, rich=False)
        grids_b = arm_grids(odds, lams_rt, rich=True)

        real = [(min(p.goles_local, MAX_GOALS), min(p.goles_visitante, MAX_GOALS))
                for p in partidos]
        kl_a = float(np.mean([kl(t, g) for t, g in zip(truths, grids_a)]))
        kl_b = float(np.mean([kl(t, g) for t, g in zip(truths, grids_b)]))
        ll_a = float(np.mean([-np.log(g[c]) for g, c in zip(grids_a, real)]))
        ll_b = float(np.mean([-np.log(g[c]) for g, c in zip(grids_b, real)]))

        cambios, delta_pts = 0, 0.0
        for ga, gb, t, p in zip(grids_a, grids_b, truths, partidos):
            evs_a = [expected_points_grid(index_score(i), ga, p.preferencial)
                     for i in range(SIDE * SIDE)]
            evs_b = [expected_points_grid(index_score(i), gb, p.preferencial)
                     for i in range(SIDE * SIDE)]
            pa_, pb_ = index_score(int(np.argmax(evs_a))), index_score(int(np.argmax(evs_b)))
            if pa_ != pb_:
                cambios += 1
                delta_pts += (expected_points_grid(pb_, t, p.preferencial)
                              - expected_points_grid(pa_, t, p.preferencial))

        # portfolios con cada grilla, evaluados bajo la verdad (pool y sorteos comunes)
        fechas = [p.fecha_id for p in partidos]
        pref = [p.preferencial for p in partidos]
        sim = SimConfig(n_sims=args.sims, seed=args.seed)
        port_a = build_portfolio(grids_a, fechas, pref, args.participaciones, sim=sim)
        port_b = build_portfolio(grids_b, fechas, pref, args.participaciones, sim=sim)

        pool_truth = [pool_distribution(t, PoolConfig()) for t in truths]
        ev_sim = SeasonSimulator(truths, fechas, pref, pool_truth, PrizeConfig(),
                                 SimConfig(n_sims=args.eval_sims,
                                           seed=args.seed + EVAL_SEED_OFFSET))
        ev_sim.load_picks(port_a.picks)
        e_a = ev_sim.e_premio_total()
        ev_sim.load_picks(port_b.picks)
        e_b = ev_sim.e_premio_total()

        filas.append((nombre, eps, kl_a, kl_b, ll_a, ll_b,
                      cambios, len(partidos), delta_pts, e_a, e_b))
        print(f"\n=== {nombre} (ε={eps:.2f}, {len(partidos)} partidos) ===")
        print(f"  KL(verdad‖grilla):  A {kl_a:.4f} → B {kl_b:.4f}  ({kl_b - kl_a:+.4f})")
        print(f"  log-loss real:      A {ll_a:.4f} → B {ll_b:.4f}  ({ll_b - ll_a:+.4f})")
        print(f"  pick óptimo: cambia en {cambios}/{len(partidos)} · "
              f"ΔE[pts] bajo verdad {delta_pts:+.2f} total")
        print(f"  E[premio] portfolio: A ${e_a:,.0f} → B ${e_b:,.0f}  ({e_b - e_a:+,.0f})")

    if not filas:
        return
    d_ll = [f[5] - f[4] for f in filas]
    d_ep = [f[10] - f[9] for f in filas]
    print(f"\n=== resumen ({len(filas)} temporadas) ===")
    print(f"  Δ log-loss real medio: {np.mean(d_ll):+.4f} "
          f"(negativo = el refinado predice mejor los resultados REALES)")
    print(f"  Δ KL medio:            {np.mean([f[3] - f[2] for f in filas]):+.4f}")
    print(f"  Δ E[premio] medio:     {np.mean(d_ep):+,.0f} ± "
          f"{np.std(d_ep, ddof=1) / np.sqrt(len(d_ep)):,.0f} (se)")


if __name__ == "__main__":
    main()

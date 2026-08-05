"""Backtest del portfolio contra temporadas reales de la Primera uruguaya.

Regla de trabajo #1 del proyecto: nada de estrategia entra sin backtest. Acá el
dataset son las 5 temporadas que devuelve el penca-api (598 partidos, 2024-2026),
y el experimento es contrafactual: construimos el portfolio con el modelo, lo
"jugamos" contra el resultado REAL de cada partido, y liquidamos los premios como
el reglamento — con un pool rival simulado, porque los picks ajenos históricos no
son públicos.

Tres limitaciones que hay que tener presentes al leer los números (regla #3,
honestidad sobre incertidumbre):

  * **Sin odds históricas.** Supermatch no expone las cuotas de partidos pasados, así
    que el λ de cada partido sale del modelo ataque/defensa ajustado a las temporadas
    ANTERIORES (out-of-sample estricto), no del mercado. Eso subestima nuestro edge
    real: en producción el λ viene del mercado, que es más informativo que el rating.
  * **Pool sintético.** Los picks ajenos históricos no son públicos, así que los
    rivales se sortean del modelo de pool. Si el pool real es más chalk que el modelo,
    nuestro edge por diferenciación sube; si es más disperso, baja.
  * **Muestra chica para el premio.** El premio grande es todo-o-nada sobre $350.000:
    con 5 temporadas, la diferencia de premio entre estrategias es casi toda ruido.
    Por eso el reporte separa la métrica de PUNTOS (baja varianza, informativa con
    n=5) del premio realizado, y agrega E[premio] esperado bajo el simulador, que
    promedia sobre miles de escenarios en vez de sobre 5.

Uso:
    python -m src.clausura.backtest                 # todas las temporadas
    python -m src.clausura.backtest --temporada "Torneo Apertura 2026"
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np

from src.clausura.economics import (
    MAX_GOALS,
    PrizeConfig,
    SeasonSimulator,
    SimConfig,
    flatten_grid,
    index_score,
    points_matrix,
    score_index,
)
from src.clausura.historical import PartidoHistorico, load_dataset
from src.clausura.pool import PoolConfig, pool_distribution
from src.clausura.ratings import TeamRatings, fit_ratings
from src.clausura.strategy import (
    baseline_chalk,
    baseline_ev,
    baseline_random_diverse,
    build_portfolio,
)
from src.model.poisson import score_grid

log = logging.getLogger(__name__)

# Prior de liga estimado del histórico (598 partidos): E[gL]=1.28, E[gV]=1.11.
# La covarianza observada es ~0 (corr +0.002), así que λ12=0: Poisson independiente
# describe bien esta liga y no hace falta el término bivariado.
LIGA_LAM_LOCAL = 1.28
LIGA_LAM_VISITA = 1.11
LIGA_LAM12 = 0.0


@dataclass
class BacktestResult:
    temporada: str
    n_partidos: int
    puntos: dict[str, float]        # estrategia → puntos de la mejor participación
    premio: dict[str, float]        # estrategia → premio cobrado contra el resultado REAL
    exactos: dict[str, float]       # estrategia → exactos de la mejor participación
    esperado: dict[str, float]      # estrategia → E[premio] bajo el simulador


def build_grids(
    partidos: list[PartidoHistorico],
    ratings: TeamRatings | None = None,
) -> list[np.ndarray]:
    """Grilla por partido desde el modelo ataque/defensa.

    Sin `ratings` cae al prior de liga plano (todos los partidos idénticos), que sirve
    de control: en ese régimen no hay señal por partido y la optimización no puede
    hacer más que repartir candidatos, así que cualquier ventaja del portfolio sobre
    la diversidad al azar tiene que venir del rating.
    """
    if ratings is None:
        g = score_grid(LIGA_LAM_LOCAL, LIGA_LAM_VISITA, LIGA_LAM12, max_goals=MAX_GOALS)
        return [g for _ in partidos]

    grids = []
    for p in partidos:
        lam_l, lam_v = ratings.lambdas(p.local, p.visitante)
        grids.append(score_grid(lam_l, lam_v, LIGA_LAM12, max_goals=MAX_GOALS))
    return grids


def actual_indices(partidos: list[PartidoHistorico]) -> np.ndarray:
    """Resultados reales como índices de score (truncados a la grilla de trabajo)."""
    return np.array([
        score_index(min(p.goles_local, MAX_GOALS), min(p.goles_visitante, MAX_GOALS))
        for p in partidos
    ])


def realized_prizes(
    picks: np.ndarray,
    partidos: list[PartidoHistorico],
    grids: list[np.ndarray],
    pool_qs: list[np.ndarray],
    prize: PrizeConfig,
    sim: SimConfig,
) -> tuple[float, float, float]:
    """Liquida el portfolio contra el resultado REAL, con pool rival simulado.

    Devuelve (premio_cobrado, puntos_mejor_participacion, exactos_mejor).
    """
    simulator = SeasonSimulator(grids, [p.fecha_id for p in partidos],
                               [p.preferencial for p in partidos], pool_qs, prize, sim)

    # Reemplazamos los resultados sorteados por el resultado REAL, repetido en todas
    # las simulaciones: la incertidumbre que queda es solo la de los picks rivales.
    real = actual_indices(partidos)
    simulator.actual = np.repeat(real[:, None], sim.n_sims, axis=1)

    # Los acumulados de rivales se calcularon con los resultados sorteados: recalcular.
    # Semilla desplazada: el portfolio se optimizó contra los sorteos de sim.seed y
    # evaluarlo sobre los mismos le daría ventaja espuria (winner's curse).
    from src.clausura.strategy import EVAL_SEED_OFFSET
    rng = np.random.default_rng(sim.seed + EVAL_SEED_OFFSET)
    simulator.rivals_total = np.zeros((sim.n_rivales, sim.n_sims), dtype=np.int32)
    simulator.rivals_fecha = np.zeros(
        (simulator.n_fechas, sim.n_rivales, sim.n_sims), dtype=np.int32
    )
    for m in range(simulator.n_matches):
        rp = rng.choice(len(pool_qs[m]), size=(sim.n_rivales, sim.n_sims), p=pool_qs[m])
        pts = simulator.pm[m][rp, simulator.actual[m][None, :]]
        simulator.rivals_total += pts
        simulator.rivals_fecha[simulator.match_fecha[m]] += pts

    simulator.load_picks(picks)
    res = simulator.result()

    puntos = float(simulator.mine_total.max(axis=0).mean())
    mejor = int(np.argmax(simulator.mine_total[:, 0]))
    exactos = float(np.sum(picks[mejor] == real))
    return res.e_premio_total, puntos, exactos


def run_temporada(
    partidos: list[PartidoHistorico],
    n_participaciones: int = 5,
    n_sims: int = 400,
    n_rivales: int = 151,
    ratings: TeamRatings | None = None,
) -> BacktestResult:
    grids = build_grids(partidos, ratings)
    fechas = [p.fecha_id for p in partidos]
    pref = [p.preferencial for p in partidos]
    pool_cfg = PoolConfig()
    pool_qs = [pool_distribution(g, pool_cfg) for g in grids]

    prize = PrizeConfig()
    sim = SimConfig(n_sims=n_sims, n_rivales=n_rivales)

    estrategias = {
        "chalk": baseline_chalk(grids, n_participaciones),
        "ev": baseline_ev(grids, pref, n_participaciones),
        "diverso_azar": baseline_random_diverse(grids, pref, n_participaciones),
    }

    # El portfolio optimizado se construye con el simulador (sin ver resultados reales)
    port = build_portfolio(
        grids=grids, fecha_de_partido=fechas, preferencial=pref,
        n_participaciones=n_participaciones, pool_cfg=pool_cfg,
        prize=prize, sim=SimConfig(n_sims=n_sims, n_rivales=n_rivales),
    )
    estrategias["portfolio"] = port.picks

    # E[premio] con semilla de evaluación INDEPENDIENTE de la de optimización
    # (misma para todas las estrategias → la comparación queda pareada)
    from src.clausura.strategy import EVAL_SEED_OFFSET
    eval_sim = SimConfig(n_sims=n_sims, n_rivales=n_rivales,
                         seed=sim.seed + EVAL_SEED_OFFSET)

    puntos, premio, exactos, esperado = {}, {}, {}, {}
    for nombre, picks in estrategias.items():
        pr, pts, ex = realized_prizes(picks, partidos, grids, pool_qs, prize, sim)
        premio[nombre], puntos[nombre], exactos[nombre] = pr, pts, ex
        # E[premio] bajo el simulador (miles de escenarios, no los 5 de la muestra)
        s = SeasonSimulator(grids, fechas, pref, pool_qs, prize, eval_sim)
        s.load_picks(picks)
        esperado[nombre] = s.e_premio_total()

    return BacktestResult(
        temporada=partidos[0].campeonato,
        n_partidos=len(partidos),
        puntos=puntos, premio=premio, exactos=exactos, esperado=esperado,
    )


# -------------------- experimento: rivales empíricos vs pool i.i.d. --------------------
#
# Valida el cambio de 2026-08-04 (src/clausura/rivals.py): sembrar el simulador con
# la tabla real y rivales por participación, en vez de re-sortear un pool i.i.d.
#
# Los picks ajenos históricos no son públicos, así que el experimento es de
# recuperación de modelo: se genera un pool "verdad" con heterogeneidad realista
# (estilos γ lognormales, ausentismo Beta), se juega la temporada REAL hasta la
# fecha k, y ambos brazos ven EXACTAMENTE lo mismo que veríamos en producción
# (picks públicos de las fechas jugadas + tabla). Difieren solo en el modelo:
#
#   A (statu quo): Q agregada empírica + rivales re-sorteados i.i.d.
#   B (nuevo):     RivalModel ajustado (γ, p_show, standings reales)
#
# Los dos portfolios resultantes se evalúan contra el MISMO pool verdad (sus picks
# futuros reales, common random numbers) → la diferencia es solo calidad de decisión.

@dataclass
class ExperimentoRep:
    e_premio_a: float
    e_premio_b: float
    realizado_a: float
    realizado_b: float
    corr_gamma: float


def _fechas_ordenadas(partidos: list[PartidoHistorico]) -> list[int]:
    inicio = {}
    for p in partidos:
        inicio[p.fecha_id] = min(inicio.get(p.fecha_id, p.inicio_utc), p.inicio_utc)
    return sorted(inicio, key=inicio.get)


def _ground_truth_pool(rng, prior_qs, n_rivales):
    """Pool verdad: (picks (R, n_matches), show (R, n_matches), gamma (R,))."""
    from src.clausura.rivals import _tilted_sample
    gamma = np.exp(rng.normal(0.0, 0.6, n_rivales))
    p_show = rng.beta(9.0, 1.0, n_rivales)
    n_matches = len(prior_qs)
    picks = np.zeros((n_rivales, n_matches), dtype=np.int64)
    for m in range(n_matches):
        picks[:, m] = _tilted_sample(prior_qs[m], gamma, rng)
    show = rng.random((n_rivales, n_matches)) < p_show[:, None]
    return picks, show, gamma


def run_experimento_rivales(
    partidos: list[PartidoHistorico],
    ratings: TeamRatings | None,
    n_participaciones: int = 12,
    n_sims: int = 800,
    n_rivales: int = 151,
    obs_fechas: int = 5,
    reps: int = 3,
    eval_sims: int = 4000,
    base_seed: int = 20260804,
) -> list[ExperimentoRep]:
    from src.clausura.pool_snapshot import blended_q
    from src.clausura.rivals import RivalModel, build_rival_model_from_arrays

    partidos = sorted(partidos, key=lambda p: p.inicio_utc)
    fechas = _fechas_ordenadas(partidos)
    if obs_fechas >= len(fechas):
        raise ValueError(f"obs_fechas={obs_fechas} pero la temporada tiene {len(fechas)}")
    pasadas = set(fechas[:obs_fechas])

    fecha_de = [p.fecha_id for p in partidos]
    pref = [p.preferencial for p in partidos]
    real = actual_indices(partidos)
    played = np.array([p.fecha_id in pasadas for p in partidos])

    model_grids = build_grids(partidos, ratings)
    prior_qs = [pool_distribution(g, PoolConfig()) for g in model_grids]
    # grillas para optimizar a mitad de temporada: pasado = resultado real (delta)
    grids_mid = list(model_grids)
    for i in np.flatnonzero(played):
        d = np.zeros_like(model_grids[i])
        d[index_score(int(real[i]))] = 1.0
        grids_mid[i] = d

    prize = PrizeConfig()

    # picks propios del pasado: portfolio de arranque de temporada (sin info del pool),
    # compartido por ambos brazos → la diferencia queda solo en las fechas futuras
    log.warning("experimento rivales: portfolio de arranque (%d participaciones)…",
                n_participaciones)
    prelim = build_portfolio(
        grids=model_grids, fecha_de_partido=fecha_de, preferencial=pref,
        n_participaciones=n_participaciones, prize=prize,
        sim=SimConfig(n_sims=n_sims, n_rivales=n_rivales, seed=base_seed),
    )

    reps_out: list[ExperimentoRep] = []
    for rep in range(reps):
        rng = np.random.default_rng(base_seed + 1000 * (rep + 1))
        gt_picks, gt_show, gt_gamma = _ground_truth_pool(rng, prior_qs, n_rivales)

        # lo observable en producción: picks públicos de las fechas jugadas + tabla
        known_obs = np.where(gt_show & played[None, :], gt_picks, -1)
        pm_normal, pm_pref = points_matrix(False), points_matrix(True)
        puntos_reales = np.zeros(n_rivales, dtype=np.int64)
        for i in np.flatnonzero(played):
            pm = pm_pref if pref[i] else pm_normal
            has = known_obs[:, i] >= 0
            puntos_reales[has] += pm[known_obs[has, i], real[i]]

        # Q empírica agregada (igual que producción statu quo)
        pool_qs_obs = list(prior_qs)
        for i in np.flatnonzero(played):
            counts = np.bincount(known_obs[known_obs[:, i] >= 0, i], minlength=len(prior_qs[i]))
            pool_qs_obs[i] = blended_q(prior_qs[i], counts.astype(float))

        comun = dict(
            grids=grids_mid, fecha_de_partido=fecha_de, preferencial=pref,
            n_participaciones=n_participaciones, prize=prize,
            frozen_picks=prelim.picks, frozen_mask=played, pool_qs=pool_qs_obs,
        )
        log.warning("  rep %d/%d: brazo A (Q agregada i.i.d.)…", rep + 1, reps)
        port_a = build_portfolio(
            sim=SimConfig(n_sims=n_sims, n_rivales=n_rivales, seed=base_seed + rep), **comun)

        log.warning("  rep %d/%d: brazo B (RivalModel empírico)…", rep + 1, reps)
        model_fit = build_rival_model_from_arrays(
            known_obs, played, pool_qs_obs, pref, real, puntos_reales)
        port_b = build_portfolio(
            sim=SimConfig(n_sims=n_sims, n_rivales=n_rivales, seed=base_seed + rep),
            rivals=model_fit, **comun)

        # evaluación: pool verdad completo (picks futuros reales), semilla fresca,
        # mismos sorteos para A y B
        gt_model = RivalModel(
            known_picks=np.where(gt_show, gt_picks, -1),
            played_mask=np.ones(len(partidos), dtype=bool),   # no observado ⇒ no cargó
            gamma=gt_gamma, p_show=np.ones(n_rivales),
            residuo=np.zeros(n_rivales, dtype=np.int64),
        )
        ev = SeasonSimulator(grids_mid, fecha_de, pref, prior_qs, prize,
                             SimConfig(n_sims=eval_sims, n_rivales=n_rivales,
                                       seed=base_seed + 777 * (rep + 1)),
                             rivals=gt_model)
        ev.load_picks(port_a.picks)
        e_a = ev.e_premio_total()
        ev.load_picks(port_b.picks)
        e_b = ev.e_premio_total()

        # premio realizado contra el resultado real completo (una muestra, ruidoso)
        grids_final = []
        for i in range(len(partidos)):
            d = np.zeros_like(model_grids[i])
            d[index_score(int(real[i]))] = 1.0
            grids_final.append(d)
        rl = SeasonSimulator(grids_final, fecha_de, pref, prior_qs, prize,
                             SimConfig(n_sims=1, n_rivales=n_rivales, seed=1), rivals=gt_model)
        rl.load_picks(port_a.picks)
        r_a = rl.e_premio_total()
        rl.load_picks(port_b.picks)
        r_b = rl.e_premio_total()

        corr = float(np.corrcoef(np.log(model_fit.gamma), np.log(gt_gamma))[0, 1])
        reps_out.append(ExperimentoRep(e_a, e_b, r_a, r_b, corr))
        log.warning("  rep %d/%d: E[premio] A $%.0f · B $%.0f · Δ %+.0f · "
                    "corr(logγ) %.2f", rep + 1, reps, e_a, e_b, e_b - e_a, corr)
    return reps_out


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporada", help="nombre exacto; default: todas")
    ap.add_argument("--participaciones", type=int, default=5)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--rivales", type=int, default=151)
    ap.add_argument("--sin-ratings", action="store_true",
                    help="control: prior de liga plano, todos los partidos idénticos")
    ap.add_argument("--experimento-rivales", action="store_true",
                    help="A/B: pool i.i.d. (statu quo) vs RivalModel empírico")
    ap.add_argument("--obs-fechas", type=int, default=5,
                    help="fechas ya jugadas/observadas en el experimento (default 5)")
    ap.add_argument("--reps", type=int, default=3,
                    help="repeticiones del pool verdad por temporada (default 3)")
    ap.add_argument("--eval-sims", type=int, default=4000)
    args = ap.parse_args()

    data = load_dataset()
    temporadas: dict[str, list[PartidoHistorico]] = {}
    for p in data:
        temporadas.setdefault(p.campeonato, []).append(p)

    # Orden cronológico real para el split out-of-sample
    orden = sorted(temporadas, key=lambda t: min(p.inicio_utc for p in temporadas[t]))
    objetivo = [args.temporada] if args.temporada else orden

    costo = args.participaciones * PrizeConfig().costo_participacion
    resultados = []
    for nombre in objetivo:
        partidos = temporadas[nombre]
        # Ratings ajustados SOLO con las temporadas anteriores a esta (sin look-ahead)
        previas = [p for t in orden[: orden.index(nombre)] for p in temporadas[t]]
        ratings = None
        if not args.sin_ratings and len(previas) >= 100:
            ratings = fit_ratings(previas)
        if not args.sin_ratings and ratings is None:
            print(f"\n(salteada {nombre}: no hay temporadas previas para ajustar ratings)")
            continue

        if args.experimento_rivales:
            logging.getLogger("src.clausura").setLevel(logging.WARNING)
            reps = run_experimento_rivales(
                partidos, ratings, args.participaciones, args.sims, args.rivales,
                args.obs_fechas, args.reps, args.eval_sims,
            )
            deltas = [x.e_premio_b - x.e_premio_a for x in reps]
            print(f"\n=== {nombre} — rivales empíricos vs pool i.i.d. "
                  f"({args.obs_fechas} fechas observadas, {args.reps} reps) ===")
            print(f"{'rep':>4s}  {'E[premio] A':>12s}  {'E[premio] B':>12s}  "
                  f"{'Δ (B−A)':>10s}  {'real A':>9s}  {'real B':>9s}  {'corr γ':>7s}")
            for j, x in enumerate(reps):
                print(f"{j + 1:>4d}  ${x.e_premio_a:11,.0f}  ${x.e_premio_b:11,.0f}  "
                      f"{x.e_premio_b - x.e_premio_a:+10,.0f}  ${x.realizado_a:8,.0f}  "
                      f"${x.realizado_b:8,.0f}  {x.corr_gamma:7.2f}")
            se = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
            print(f"   Δ medio {np.mean(deltas):+,.0f} ± {se:,.0f} (se pareado)")
            continue

        r = run_temporada(partidos, args.participaciones, args.sims, args.rivales, ratings)
        resultados.append(r)
        print(f"\n=== {nombre} ({r.n_partidos} partidos, "
              f"ratings de {len(previas)} partidos previos) ===")
        print(f"{'estrategia':>14s}  {'pts mejor':>9s}  {'exactos':>7s}  "
              f"{'premio real':>12s}  {'E[premio]':>11s}")
        for k in r.premio:
            print(f"{k:>14s}  {r.puntos[k]:9.1f}  {r.exactos[k]:7.0f}  "
                  f"${r.premio[k]:11,.0f}  ${r.esperado[k]:10,.0f}")

    if not resultados:
        return

    print(f"\n=== promedio de {len(resultados)} temporadas (costo ${costo:,.0f}) ===")
    print(f"{'estrategia':>14s}  {'pts':>7s}  {'exactos':>7s}  {'premio real':>12s}  "
          f"{'E[premio]':>11s}  {'ROI sobre E':>11s}")
    for k in resultados[0].premio:
        pts = np.mean([r.puntos[k] for r in resultados])
        ex = np.mean([r.exactos[k] for r in resultados])
        pr = np.mean([r.premio[k] for r in resultados])
        esp = np.mean([r.esperado[k] for r in resultados])
        print(f"{k:>14s}  {pts:7.1f}  {ex:7.1f}  ${pr:11,.0f}  ${esp:10,.0f}  "
              f"{(esp - costo) / costo:+10.1%}")
    print("\nNota: 'premio real' con n=%d es casi todo ruido (premio todo-o-nada de "
          "$350k).\nLa columna informativa es E[premio], que promedia miles de "
          "escenarios." % len(resultados))


if __name__ == "__main__":
    main()

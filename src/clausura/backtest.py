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
    rng = np.random.default_rng(sim.seed)
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

    puntos, premio, exactos, esperado = {}, {}, {}, {}
    for nombre, picks in estrategias.items():
        pr, pts, ex = realized_prizes(picks, partidos, grids, pool_qs, prize, sim)
        premio[nombre], puntos[nombre], exactos[nombre] = pr, pts, ex
        # E[premio] bajo el simulador (miles de escenarios, no los 5 de la muestra)
        s = SeasonSimulator(grids, fechas, pref, pool_qs, prize, sim)
        s.load_picks(picks)
        esperado[nombre] = s.e_premio_total()

    return BacktestResult(
        temporada=partidos[0].campeonato,
        n_partidos=len(partidos),
        puntos=puntos, premio=premio, exactos=exactos, esperado=esperado,
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporada", help="nombre exacto; default: todas")
    ap.add_argument("--participaciones", type=int, default=5)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--rivales", type=int, default=151)
    ap.add_argument("--sin-ratings", action="store_true",
                    help="control: prior de liga plano, todos los partidos idénticos")
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

"""¿Vale plata modelar el ausentismo por FECHA en vez de por partido?

El dato que motiva esto (snapshot real F1, 727 filas): 630 rivales (87%) cargaron
los 7 partidos observables, contra el 56% de planillas completas que predice el
Bernoulli independiente por partido (0.92⁷). El modelo viejo reparte los agujeros
entre todos → el máximo del pool por fecha simulado queda más bajo que el real →
los 15 premios de $10.000 se modelan más ganables de lo que son.

Experimento de recuperación de modelo (mismo diseño que el del RivalModel,
backtest.run_experimento_rivales): se genera un pool "verdad" con ausentismo por
fecha (lo realista), se juega la temporada real hasta la fecha k, y los dos
brazos ven EXACTAMENTE lo mismo (picks públicos de lo jugado + tabla) y fitean el
MISMO RivalModel. Difieren solo en cómo el simulador sortea los futuros:

  A (viejo):  show ~ Bernoulli(p_show) independiente por partido
  B (nuevo):  show sorteado una vez por (rival, fecha), compartido (SimConfig.show_por_fecha)

Ambos portfolios se evalúan contra el MISMO pool verdad (sus picks/cargas futuras
reales, semilla fresca, sorteos comunes) → la diferencia es solo calidad de decisión.

Uso:
    .venv/bin/python scripts/backtest_show_fecha.py --sims 800 --reps 3
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.clausura.backtest import (  # noqa: E402
    _fechas_ordenadas,
    actual_indices,
    build_grids,
)
from src.clausura.economics import (  # noqa: E402
    PrizeConfig,
    SeasonSimulator,
    SimConfig,
    index_score,
    points_matrix,
)
from src.clausura.historical import load_dataset  # noqa: E402
from src.clausura.pool import PoolConfig, pool_distribution  # noqa: E402
from src.clausura.pool_snapshot import blended_q  # noqa: E402
from src.clausura.ratings import fit_ratings  # noqa: E402
from src.clausura.rivals import RivalModel, _tilted_sample, build_rival_model_from_arrays  # noqa: E402
from src.clausura.strategy import build_portfolio  # noqa: E402


def _ground_truth_fecha(rng, prior_qs, fecha_de, n_rivales):
    """Pool verdad con ausentismo POR FECHA: (picks, show, gamma).

    Igual que backtest._ground_truth_pool salvo el show: un Bernoulli por
    (rival, fecha) replicado en sus partidos — la estructura que el snapshot
    real de la F1 muestra (87% de planillas completas).
    """
    gamma = np.exp(rng.normal(0.0, 0.6, n_rivales))
    p_show = rng.beta(9.0, 1.0, n_rivales)
    n_matches = len(prior_qs)
    picks = np.zeros((n_rivales, n_matches), dtype=np.int64)
    for m in range(n_matches):
        picks[:, m] = _tilted_sample(prior_qs[m], gamma, rng)
    fechas = sorted(set(fecha_de))
    show_fecha = {f: rng.random(n_rivales) < p_show for f in fechas}
    show = np.stack([show_fecha[fecha_de[m]] for m in range(n_matches)], axis=1)
    return picks, show, gamma


def run_experimento(partidos, ratings, n_participaciones=12, n_sims=800,
                    n_rivales=151, obs_fechas=5, reps=3, eval_sims=4000,
                    base_seed=20260812):
    partidos = sorted(partidos, key=lambda p: p.inicio_utc)
    fechas = _fechas_ordenadas(partidos)
    pasadas = set(fechas[:obs_fechas])

    fecha_de = [p.fecha_id for p in partidos]
    pref = [p.preferencial for p in partidos]
    real = actual_indices(partidos)
    played = np.array([p.fecha_id in pasadas for p in partidos])

    model_grids = build_grids(partidos, ratings)
    prior_qs = [pool_distribution(g, PoolConfig()) for g in model_grids]
    grids_mid = list(model_grids)
    for i in np.flatnonzero(played):
        d = np.zeros_like(model_grids[i])
        d[index_score(int(real[i]))] = 1.0
        grids_mid[i] = d

    prize = PrizeConfig()
    prelim = build_portfolio(
        grids=model_grids, fecha_de_partido=fecha_de, preferencial=pref,
        n_participaciones=n_participaciones, prize=prize,
        sim=SimConfig(n_sims=n_sims, n_rivales=n_rivales, seed=base_seed))

    out = []
    for rep in range(reps):
        rng = np.random.default_rng(base_seed + 1000 * (rep + 1))
        gt_picks, gt_show, _ = _ground_truth_fecha(rng, prior_qs, fecha_de, n_rivales)

        known_obs = np.where(gt_show & played[None, :], gt_picks, -1)
        pm_n, pm_p = points_matrix(False), points_matrix(True)
        puntos_reales = np.zeros(n_rivales, dtype=np.int64)
        for i in np.flatnonzero(played):
            pm = pm_p if pref[i] else pm_n
            has = known_obs[:, i] >= 0
            puntos_reales[has] += pm[known_obs[has, i], real[i]]

        pool_qs_obs = list(prior_qs)
        for i in np.flatnonzero(played):
            counts = np.bincount(known_obs[known_obs[:, i] >= 0, i],
                                 minlength=len(prior_qs[i]))
            pool_qs_obs[i] = blended_q(prior_qs[i], counts.astype(float))

        model_fit = build_rival_model_from_arrays(
            known_obs, played, pool_qs_obs, pref, real, puntos_reales)

        comun = dict(
            grids=grids_mid, fecha_de_partido=fecha_de, preferencial=pref,
            n_participaciones=n_participaciones, prize=prize,
            frozen_picks=prelim.picks, frozen_mask=played, pool_qs=pool_qs_obs,
            rivals=model_fit,
        )
        port_a = build_portfolio(sim=SimConfig(
            n_sims=n_sims, n_rivales=n_rivales, seed=base_seed + rep,
            show_por_fecha=False), **comun)
        port_b = build_portfolio(sim=SimConfig(
            n_sims=n_sims, n_rivales=n_rivales, seed=base_seed + rep,
            show_por_fecha=True), **comun)

        # evaluación: pool verdad completo (cargas futuras reales por fecha),
        # semilla fresca, mismos sorteos para A y B
        gt_model = RivalModel(
            known_picks=np.where(gt_show, gt_picks, -1),
            played_mask=np.ones(len(partidos), dtype=bool),
            gamma=np.ones(n_rivales), p_show=np.ones(n_rivales),
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
        out.append((e_a, e_b))
        print(f"  rep {rep + 1}/{reps}: A ${e_a:,.0f} · B ${e_b:,.0f} · "
              f"Δ {e_b - e_a:+,.0f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participaciones", type=int, default=12)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--rivales", type=int, default=151)
    ap.add_argument("--obs-fechas", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--eval-sims", type=int, default=4000)
    ap.add_argument("--temporada", default=None)
    args = ap.parse_args()

    data = load_dataset()
    temporadas: dict[str, list] = {}
    for p in data:
        temporadas.setdefault(p.campeonato, []).append(p)
    orden = sorted(temporadas, key=lambda t: min(p.inicio_utc for p in temporadas[t]))
    objetivo = [args.temporada] if args.temporada else orden

    deltas = []
    for nombre in objetivo:
        previas = [p for t in orden[: orden.index(nombre)] for p in temporadas[t]]
        if len(previas) < 100:
            print(f"(salteada {nombre}: sin historia previa para ratings)")
            continue
        print(f"\n=== {nombre} — show por fecha vs por partido ===")
        reps = run_experimento(
            temporadas[nombre], fit_ratings(previas), args.participaciones,
            args.sims, args.rivales, args.obs_fechas, args.reps, args.eval_sims)
        deltas += [b - a for a, b in reps]

    d = np.array(deltas)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    print(f"\nΔ medio (B−A): {d.mean():+,.0f} ± {se:,.0f} "
          f"(se pareado, {len(d)} reps, a favor en {(d > 0).sum()}/{len(d)})")


if __name__ == "__main__":
    main()

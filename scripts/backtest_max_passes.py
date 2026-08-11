"""¿Cuántas pasadas de ascenso conviene, ahora que hay 19.200 sorteos?

`max_passes` es el MISMO lever que el tamaño del menú: cada pasada extra son más
comparaciones de dos estimaciones Monte Carlo, o sea más boletos para que el ruido
gane el argmax, pero también más chances de encontrar la mejora real. Con pocos
sorteos gana lo primero, con muchos lo segundo — y el signo del menú se dio vuelta
justamente al subir sorteos (ver strategy.py, K_EV).

La decisión vigente (`max_passes=3`) viene de la auditoría del 2026-08-08, que midió
3→6 como 85% overfitting **explícitamente "sin subir sorteos"**. Desde entonces los
sorteos pasaron de 2.400 a 19.200: ocho veces más. Toca volver a medir.

WARM START. Va en el barrido porque se pisa con las pasadas: arrancar desde la
planilla previa es, en efecto, empezar con pasadas ya hechas. Se modela como en
producción — el rerun de cierre warm-startea desde la planilla de la mañana, que acá
es el brazo base (3 pasadas, sin warm start).

Igual que el barrido de menú: sorteos comunes, dos verdades del pool (la calibrada
contra 4.791 picks reales y la vieja como control conservador), y se exige ganar en
las dos.
"""
import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_AQUI = pathlib.Path(__file__).resolve().parent

from src.clausura.economics import PrizeConfig, SimConfig  # noqa: E402
from src.clausura.pool import PoolConfig, pool_distribution  # noqa: E402
from src.clausura.strategy import EvaluadorPortfolio, build_portfolio  # noqa: E402

PASSES = (3, 6, 10)
BASE_PASSES = 3                      # producción hoy


def q_verdad(pred_grids):
    d = json.load(open(_AQUI / "bias_calibrado.json"))
    c = PoolConfig(chalk_strength=d["chalk"], temperature=1.0,
                   default_bias=d["default_bias"],
                   popular_bias={tuple(map(int, k.split("-"))): v
                                 for k, v in d["bias"].items()})
    return [pool_distribution(g, c) for g in pred_grids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=19200)
    ap.add_argument("--eval-seeds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--participaciones", type=int, default=12)
    a = ap.parse_args()

    from src.clausura.api import PencaApiClient
    from src.clausura.odds import fetch_primera_odds
    from src.clausura.picks import (
        build_season_grids, ensure_ratings, flat_eventos, load_config, match_odds,
    )

    cfg = load_config()
    eventos = flat_eventos(cfg)
    resultados = {}
    with PencaApiClient() as api:
        for _, f in cfg["fechas"].items():
            for e in api._get(f"/front/campeonatos/fechas/{f['fecha_id']}/eventos"):
                r = e.get("resultado") or {}
                gl, gv = r.get("golesEquipoLocal"), r.get("golesEquipoVisitante")
                if gl is not None and gv is not None:
                    resultados[e["id"]] = (int(gl), int(gv))
    ratings = ensure_ratings()
    try:
        odds_by_evento = match_odds(eventos, fetch_primera_odds())
    except Exception as e:
        print(f"sin odds ({e})", flush=True)
        odds_by_evento = {}

    grids, _, pred_grids, _ = build_season_grids(
        eventos, ratings, odds_by_evento, resultados)
    fecha_de = [ev["fecha_n"] for ev in eventos]
    pref = [bool(ev["preferencial"]) for ev in eventos]
    prize = PrizeConfig()

    qs = {"verdad": q_verdad(pred_grids),
          "control": [pool_distribution(g, PoolConfig()) for g in pred_grids]}
    # la creencia del optimizador es siempre la de producción; acá se barre la BÚSQUEDA
    creencia = [pool_distribution(g, PoolConfig()) for g in pred_grids]

    brazos = [(p, w) for p, w in itertools.product(PASSES, (False, True))]
    print(f"{len(eventos)} eventos · {len(resultados)} jugados · "
          f"{len(brazos)} brazos × {a.reps} reps · {a.sims} sorteos\n", flush=True)

    acum = {b: {"verdad": [], "control": []} for b in brazos}
    for rep in range(a.reps):
        semilla = 20260811 + 7919 * rep
        sim = SimConfig(n_sims=a.sims, n_rivales=718, seed=semilla)

        def planilla(passes, warm):
            return np.asarray(build_portfolio(
                grids=grids, fecha_de_partido=fecha_de, preferencial=pref,
                n_participaciones=a.participaciones, prize=prize, pool_qs=creencia,
                sim=sim, max_passes=passes,
                warm_start=base if warm else None).picks)

        t0 = time.time()
        base = None
        base = planilla(BASE_PASSES, False)
        evs = {k: EvaluadorPortfolio(grids, fecha_de, pref, q, prize, sim)
               for k, q in qs.items()}
        print(f"--- rep {rep + 1} (base en {time.time() - t0:.0f}s) ---", flush=True)
        for passes, warm in brazos:
            if (passes, warm) == (BASE_PASSES, False):
                for k in acum[(passes, warm)]:
                    acum[(passes, warm)][k].append(0.0)
                continue
            t1 = time.time()
            p = planilla(passes, warm)
            fila = []
            for k, ev in evs.items():
                c = ev.comparar(base, p, n_seeds=a.eval_seeds)
                acum[(passes, warm)][k].append(c.delta)
                fila.append(f"{k} {c.delta:+,.0f} ± {c.se:,.0f}")
            print(f"  {passes:>2} pasadas · warm {str(warm):<5} ({time.time()-t1:>3.0f}s)   "
                  + "   ·   ".join(fila), flush=True)

    print(f"\n{'='*74}\nRESUMEN — Δ E[premio] vs producción ({BASE_PASSES} pasadas, sin warm)\n{'='*74}")
    print(f"  {'pasadas':>8} {'warm':>6}   {'verdad = Q calibrada':>24}   {'control = Q vieja':>22}")
    for b in sorted(brazos, key=lambda x: -float(np.mean(acum[x]["control"]))):
        v, c = acum[b]["verdad"], acum[b]["control"]
        se = lambda xs: float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0
        marca = "  ← producción hoy" if b == (BASE_PASSES, False) else ""
        print(f"  {b[0]:>8} {str(b[1]):>6}   {np.mean(v):>+15,.0f} ± {se(v):>6,.0f}"
              f"   {np.mean(c):>+13,.0f} ± {se(c):>6,.0f}{marca}")


if __name__ == "__main__":
    main()

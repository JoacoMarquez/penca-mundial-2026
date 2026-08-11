"""Barrido conjunto: fuerza de chalk × tamaño del menú.

POR QUÉ JUNTAS. Las dos son la misma perilla por caminos distintos. `chalk_strength`
empuja a diferenciarse DENTRO de los candidatos disponibles; el menú decide CUÁNTOS
candidatos hay. Medir una con la otra fija es calibrar el acelerador con el freno de
mano puesto.

Y hay una razón concreta para re-abrir el menú: se fijó en (3, 0) el 2026-08-08 con
+$9.737 ± 859 (16/16 reps), pero esa medición usó la Q vieja — la que subestimaba
cuánto se amontona el pool. Con el pool más disperso de lo real, los candidatos por
HUECO se evalúan con una vara que les juega en contra: si el modelo cree que hay poca
compañía en los marcadores populares, escaparse rinde menos de lo que rinde.

VERDAD DEL POOL. Fija en todos los brazos, y NO es la perilla que se barre: es la Q
calibrada contra 4.791 picks reales (log-loss leave-one-match-out 2.167 vs 2.271).
Se reporta también contra la Q vieja como control conservador.

Todos los brazos se comparan contra producción de hoy (chalk 1.0, menú (3,0)) con
SORTEOS COMUNES y las mismas semillas.
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
_aqui = lambda n: _AQUI / n

from src.clausura import strategy  # noqa: E402
from src.clausura.economics import PrizeConfig, SimConfig  # noqa: E402
from src.clausura.pool import PoolConfig, pool_distribution  # noqa: E402
from src.clausura.strategy import EvaluadorPortfolio, build_portfolio  # noqa: E402

CHALKS = (1.0, 1.5, 2.2, 3.0)
MENUS = ((3, 0), (5, 0), (3, 3))
BASE = (1.0, (3, 0))                      # producción hoy


def q_verdad(pred_grids):
    d = json.load(open(_aqui("bias_calibrado.json")))
    c = PoolConfig(chalk_strength=d["chalk"], temperature=1.0,
                   default_bias=d["default_bias"],
                   popular_bias={tuple(map(int, k.split("-"))): v
                                 for k, v in d["bias"].items()})
    return [pool_distribution(g, c) for g in pred_grids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=9600)
    ap.add_argument("--eval-seeds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--participaciones", type=int, default=12)
    ap.add_argument("--chalks", default="")   # "1.0,2.2"
    ap.add_argument("--menus", default="")    # "3-0,5-0"
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

    chalks = tuple(float(x) for x in a.chalks.split(',')) if a.chalks else CHALKS
    menus = (tuple(tuple(int(v) for v in m.split('-')) for m in a.menus.split(','))
             if a.menus else MENUS)

    qs_verdad = q_verdad(pred_grids)
    qs_control = [pool_distribution(g, PoolConfig()) for g in pred_grids]
    # creencia del optimizador por valor de chalk (la tabla de sesgos NO se toca:
    # midió no aportar nada, +2.251 ± 761 contra +3.036 ± 327 del chalk solo)
    # BASE siempre entra aunque no se la pida entre los brazos: es el punto de
    # comparación de todos, y sin su creencia el primer build tira KeyError.
    creencia = {c: [pool_distribution(g, PoolConfig(chalk_strength=c))
                    for g in pred_grids]
                for c in sorted(set(chalks) | {BASE[0]})}

    brazos = [(c, m) for c, m in itertools.product(chalks, menus)]
    print(f"{len(eventos)} eventos · {len(resultados)} jugados · "
          f"{len(brazos)} brazos × {a.reps} reps · {a.sims} sorteos\n", flush=True)

    acum = {b: {"verdad": [], "control": []} for b in brazos}
    for rep in range(a.reps):
        semilla = 20260810 + 7919 * rep
        sim = SimConfig(n_sims=a.sims, n_rivales=718, seed=semilla)

        def planilla(chalk, menu):
            strategy.K_EV, strategy.K_HUECO = menu
            try:
                return np.asarray(build_portfolio(
                    grids=grids, fecha_de_partido=fecha_de, preferencial=pref,
                    n_participaciones=a.participaciones, prize=prize,
                    pool_qs=creencia[chalk], sim=sim).picks)
            finally:
                strategy.K_EV, strategy.K_HUECO = 3, 0

        t0 = time.time()
        base = planilla(*BASE)
        evs = {
            "verdad": EvaluadorPortfolio(grids, fecha_de, pref, qs_verdad, prize, sim),
            "control": EvaluadorPortfolio(grids, fecha_de, pref, qs_control, prize, sim),
        }
        print(f"--- rep {rep + 1} (base en {time.time() - t0:.0f}s) ---", flush=True)
        for chalk, menu in brazos:
            if (chalk, menu) == BASE:
                acum[(chalk, menu)]["verdad"].append(0.0)
                acum[(chalk, menu)]["control"].append(0.0)
                continue
            p = planilla(chalk, menu)
            fila = []
            for k, ev in evs.items():
                c = ev.comparar(base, p, n_seeds=a.eval_seeds)
                acum[(chalk, menu)][k].append(c.delta)
                fila.append(f"{k} {c.delta:+,.0f} ± {c.se:,.0f}")
            print(f"  chalk {chalk:<4} menú {menu}   " + "   ·   ".join(fila), flush=True)

    print(f"\n{'='*72}\nRESUMEN — Δ E[premio] vs producción (chalk 1.0, menú (3,0))\n{'='*72}")
    print(f"  {'chalk':>6} {'menú':>8}   {'verdad = Q calibrada':>24}   {'control = Q vieja':>22}")
    for b in sorted(brazos, key=lambda x: -float(np.mean(acum[x]["verdad"]))):
        v, c = acum[b]["verdad"], acum[b]["control"]
        se_v = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        se_c = float(np.std(c, ddof=1) / np.sqrt(len(c))) if len(c) > 1 else 0.0
        marca = "  ← producción hoy" if b == BASE else ""
        print(f"  {b[0]:>6} {str(b[1]):>8}   {np.mean(v):>+15,.0f} ± {se_v:>6,.0f}"
              f"   {np.mean(c):>+13,.0f} ± {se_c:>6,.0f}{marca}")


if __name__ == "__main__":
    main()

"""Cap de diversidad re-medido contra las fechas REALES del Clausura (F2-F8).

El A/B del 2026-09-04 dejó las dos varas discrepando en signo: por E[premio] el
cap pierde (−$437 / −$125), contra resultados reales gana (−7,6pp / −5,5pp en
P(azar ≥ nuestro máx)) — pero con n efectivo 3, porque las 15 fechas de una
temporada histórica comparten planilla. La salida que dejó escrita la memoria
(cap_diversidad_rechazado) es acumular fechas del Clausura EN VIVO: cada fecha
trae planilla nueva y rompe el agrupamiento.

Por fecha, con lo que se sabía antes de cada partido:

  * **Grillas**: mercado pre-cierre (cuotas versionadas; 1X2 + over → Poisson
    bivariado, como market_lambdas). La F1 queda afuera (sin cuotas antes del 13/8).
  * **Pool**: Q empírica de los picks REALES de los rivales en esa fecha (90%)
    suavizada con el prior del modelo (10%). Es información post-cierre: sesga los
    dos brazos igual y es lo más parecido al snapshot que ve producción.
  * **Objetivo**: build_portfolio sobre los 8 partidos de la fecha, sin especiales
    ni standings. Con una sola fecha, el "premio grande" es ganar ese bloque: un
    objetivo de cola pura, el mismo para todos los brazos.

Brazos (idénticos salvo menú y cap):
    A control  K_EV=5 sin cap   (producción)
    C menú     K_EV=8 sin cap
    B cap      K_EV=5 cap=3
    D cap+menú K_EV=8 cap=2

Cap = ningún marcador en más de `cap` filas de un mismo partido; se respeta en la
inicialización y en cada movimiento. OJO (lección del 4/8): por E[premio] un brazo
con cap NO PUEDE dar positivo — es un óptimo restringido del mismo objetivo. Esa
columna se reporta para ver el costo, no como veredicto. El veredicto es la
columna real: contra los resultados y los picks reales del pool de esa fecha.

Métricas reales por fecha y brazo: máximo de nuestras 12, percentil de ese máximo
en el pool real, P(máx de 12 rivales al azar ≥ nuestro máx) (hipergeométrica exacta,
menor es mejor), premio de fecha que habría cobrado, y batacazos cubiertos.

Uso (VPS, env cargado; ~1-2 h a 19.200 sorteos):
    PYTHONPATH=/opt/penca python -m scripts.ab_cap_diversidad_vivo [--sims 19200] [--seeds 2]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass

import numpy as np

from src.clausura.economics import (
    N_SCORES, PrizeConfig, SeasonSimulator, SimConfig, index_score, points_matrix,
    score_index,
)
from src.clausura.picks import flat_eventos, load_config
from src.clausura.pool import PoolConfig, pool_distribution
from src.clausura.pool_snapshot import load_latest_snapshot
from src.clausura.postmortem import resultados_de_fecha
from src.clausura.rivals import mis_numeros_env
from src.clausura.strategy import EVAL_SEED_OFFSET, build_candidates
from scripts.premios_fecha_calibracion import _idx, grilla_pre_cierre, odds_snapshots

log = logging.getLogger(__name__)

N_PART = 12
BRAZOS = {"A": (5, None), "C": (8, None), "B": (5, 3), "D": (8, 2)}


@dataclass
class Fecha:
    n: int
    eventos: list[dict]
    grids: list[np.ndarray]          # (SIDE, SIDE)
    qs: list[np.ndarray]             # N_SCORES
    reales: list[int]                # índice del resultado real
    riv_picks: np.ndarray            # (R, 8) índice, -1 = no cargó


def armar_fecha(n: int, cfg, snaps, pool, mis) -> Fecha | None:
    evs = [e for e in flat_eventos(cfg) if e["fecha_n"] == n]
    res, _ = resultados_de_fecha(cfg, n)
    evs = [e for e in evs if e["evento_id"] in res]
    grids, qs, reales = [], [], []
    riv = np.full((len(pool), len(evs)), -1, dtype=np.int64)
    for j, e in enumerate(evs):
        g = grilla_pre_cierre(e, snaps)
        if g is None:
            log.warning("fecha %d sin cuotas pre-cierre — afuera", n)
            return None
        g = g.reshape(int(math.isqrt(g.size)), -1)
        grids.append(g)
        emp = np.zeros(N_SCORES)
        for r, p in enumerate(pool):
            pk = p["picks"].get(str(e["evento_id"]))
            if pk:
                riv[r, j] = _idx(*pk)
                emp[riv[r, j]] += 1
        prior = pool_distribution(g, PoolConfig())
        q = 0.9 * emp / max(emp.sum(), 1) + 0.1 * prior
        qs.append(q / q.sum())
        reales.append(_idx(*res[e["evento_id"]]))
    return Fecha(n, evs, grids, qs, reales, riv)


def construir(f: Fecha, k_ev: int, cap: int | None, sims: int, seed: int,
              n_rivales: int, max_passes: int = 6) -> tuple[np.ndarray, float]:
    """Ascenso por coordenadas de build_portfolio con cap opcional → (picks, E[premio] oos)."""
    pref = [bool(e["preferencial"]) for e in f.eventos]
    fechas = [f.n] * len(f.eventos)
    cands = [[score_index(*c.pick) for c in
              sorted(build_candidates(g, q, p, k_ev=k_ev), key=lambda c: -c.e_points)]
             for g, q, p in zip(f.grids, f.qs, pref)]
    sim_cfg = SimConfig(n_sims=sims, n_rivales=n_rivales, seed=seed)
    sim = SeasonSimulator(f.grids, fechas, pref, f.qs, PrizeConfig(), sim_cfg, None)

    picks = np.zeros((N_PART, len(f.eventos)), dtype=np.int64)
    for m, cs in enumerate(cands):
        if cap is None:
            picks[:, m] = cs[0]
        else:
            # la fila 0 es el ancla EV; el resto llena en orden de E[pts] sin pasar el cap
            uso: dict[int, int] = {}
            for i in range(N_PART):
                c = next((c for c in cs if uso.get(c, 0) < cap), cs[i % len(cs)])
                picks[i, m] = c
                uso[c] = uso.get(c, 0) + 1
    sim.load_picks(picks)
    actual = sim.e_premio_total()

    for _ in range(max_passes):
        mejoras = 0
        for i in range(1, N_PART):
            for m, cs in enumerate(cands):
                orig = int(sim.picks[i, m])
                col = sim.picks[:, m]
                mejor, mejor_val = orig, actual
                for c in cs:
                    if c == orig:
                        continue
                    if cap is not None and int((col == c).sum()) >= cap:
                        continue
                    sim.set_pick(i, m, c)
                    val = sim.e_premio_total()
                    if val > mejor_val:
                        mejor, mejor_val = c, val
                sim.set_pick(i, m, mejor)
                if mejor != orig:
                    mejoras += 1
                    actual = mejor_val
        if mejoras == 0:
            break

    ev = SeasonSimulator(f.grids, fechas, pref, f.qs, PrizeConfig(),
                         SimConfig(n_sims=sims, n_rivales=n_rivales,
                                   seed=seed + EVAL_SEED_OFFSET), None)
    ev.load_picks(sim.picks)
    return sim.picks.copy(), float(ev.result().e_premio_total)


def p_azar_gana(pool_pts: np.ndarray, nuestro: int, k: int = N_PART) -> float:
    """P(máx de k rivales al azar, sin reposición, ≥ nuestro máx). Exacta."""
    n, m = len(pool_pts), int((pool_pts >= nuestro).sum())
    if m == 0:
        return 0.0
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def evaluar_real(f: Fecha, picks: np.ndarray) -> dict:
    pref = [bool(e["preferencial"]) for e in f.eventos]
    mios = np.zeros(N_PART, dtype=np.int64)
    riv = np.zeros(f.riv_picks.shape[0], dtype=np.int64)
    batacazos = cubiertos = 0
    for m, real in enumerate(f.reales):
        P = points_matrix(pref[m])
        mios += P[picks[:, m], real]
        ok = f.riv_picks[:, m] >= 0
        riv[ok] += P[f.riv_picks[ok, m], real]
        # batacazo: el resultado real tenía < 5% del pool con su signo (L/E/V)
        signo = np.sign(np.subtract(*index_score(real)))
        q_signo = sum(f.qs[m][i] for i in range(N_SCORES)
                      if np.sign(np.subtract(*index_score(i))) == signo)
        if q_signo < 0.15:
            batacazos += 1
            cubiertos += int(any(np.sign(np.subtract(*index_score(int(p)))) == signo
                                 for p in picks[:, m]))
    top = int(mios.max())
    tope = max(top, int(riv.max()))
    n_m, n_r = int((mios == tope).sum()), int((riv == tope).sum())
    return {
        "max": top, "max_rival": int(riv.max()),
        "pct": float((riv < top).mean() + 0.5 * (riv == top).mean()),
        "p_azar": p_azar_gana(riv, top),
        "premio": 10_000.0 * n_m / (n_m + n_r) if top == tope else 0.0,
        "distintos": float(np.mean([len(set(picks[:, m])) for m in range(picks.shape[1])])),
        "batacazos": batacazos, "cubiertos": cubiertos,
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=19_200)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--hasta", type=int, default=8)
    ap.add_argument("--out", default=None, help="JSON con las filas crudas")
    args = ap.parse_args()

    cfg = load_config()
    mis = sorted(mis_numeros_env())
    pool = [p for p in load_latest_snapshot()["participaciones"] if p["numero"] not in set(mis)]
    snaps = odds_snapshots()

    filas = []
    for n in range(1, args.hasta + 1):
        f = armar_fecha(n, cfg, snaps, pool, mis)
        if f is None:
            continue
        for s in range(args.seeds):
            for brazo, (k_ev, cap) in BRAZOS.items():
                picks, e_premio = construir(f, k_ev, cap, args.sims, 1000 * n + s,
                                            len(pool))
                r = evaluar_real(f, picks) | {"fecha": n, "seed": s, "brazo": brazo,
                                              "e_premio": e_premio}
                filas.append(r)
                print(f"F{n} s{s} {brazo}: E ${e_premio:>7,.0f} · máx {r['max']:>2} "
                      f"(rival {r['max_rival']}) pct {r['pct']:.3f} "
                      f"azar≥ {r['p_azar']:.3f} · ${r['premio']:,.0f} · "
                      f"{r['distintos']:.2f} dist · bat {r['cubiertos']}/{r['batacazos']}",
                      flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(filas, fh)

    print("\n=== Δ vs A (pareado por fecha y semilla) ===")
    base = {(r["fecha"], r["seed"]): r for r in filas if r["brazo"] == "A"}
    for brazo in ("C", "B", "D"):
        rs = [r for r in filas if r["brazo"] == brazo]
        d_e = np.array([r["e_premio"] - base[(r["fecha"], r["seed"])]["e_premio"] for r in rs])
        d_az = np.array([r["p_azar"] - base[(r["fecha"], r["seed"])]["p_azar"] for r in rs])
        d_max = np.array([r["max"] - base[(r["fecha"], r["seed"])]["max"] for r in rs])
        # por FECHA (promedio de semillas): la unidad independiente es la fecha
        fechas = sorted({r["fecha"] for r in rs})
        az_f = np.array([np.mean([d for d, r in zip(d_az, rs) if r["fecha"] == fe])
                         for fe in fechas])
        mejor = int((az_f < 0).sum())
        se = az_f.std(ddof=1) / math.sqrt(len(az_f)) if len(az_f) > 1 else float("nan")
        print(f"{brazo}: ΔE[premio] {d_e.mean():+,.0f} ± {d_e.std(ddof=1)/math.sqrt(len(d_e)):,.0f}"
              f" · Δ azar≥ {az_f.mean():+.3f} ± {se:.3f} (mejor en {mejor}/{len(fechas)} fechas)"
              f" · Δ máx {d_max.mean():+.2f} pts"
              f" · premios ${sum(r['premio'] for r in rs)/args.seeds:,.0f}"
              f" vs A ${sum(b['premio'] for b in base.values())/args.seeds:,.0f}"
              f" · batacazos {sum(r['cubiertos'] for r in rs)/args.seeds:.1f}"
              f" vs {sum(b['cubiertos'] for b in base.values())/args.seeds:.1f}"
              f" de {sum(b['batacazos'] for b in base.values())/args.seeds:.0f}")


if __name__ == "__main__":
    main()

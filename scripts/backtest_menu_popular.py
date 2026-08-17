"""¿Ofrecerle al optimizador los marcadores POPULARES del pool mejora el E[premio]?

EL MODO DE FALLA (cierre de la Fecha 2, 2026-08-16): el chequeo de asignación del
postmortem dio tres días seguidos "12 rivales al azar nos ganan el 96%", con la
diversidad bien (corr 0.08-0.14 entre filas) y el NIVEL bajo (25.3 vs 29.2 del pool).
El patrón concreto en los partidos:
  * Peñarol 2-1 Central Español: 68% del pool en 2-0/3-0/3-1, nosotros 12/12 en
    1-0/2-0 (0% en la goleada del favorito);
  * Progreso 0-2 Maldonado: 9/12 con el ganador pero 0 en el 0-2 que tenía el 15%
    del pool (nuestras perturbaciones de visita van al 0-1/1-2 "raros");
  * Defensor 0-4 Liverpool: 2/12 en la V, con el pool en 31%.
El top-K_EV por E[pts] del kernel aditivo se queda en goles bajos. K_COBERTURA
(mejor por E[pts] de cada desenlace) ya se midió NEUTRA el 12/8 porque ofrece el
0-1/1-1, no lo que el pool juega. K_POPULAR ofrece los top-M por pool_q.

Misma escalera que backtest_cobertura.py:
  ETAPA 1 (barata): ¿el optimizador TOMA los candidatos populares? Si las planillas
  salen idénticas no hay nada que medir. Métrica secundaria: cobertura L/E/V y goles
  medios por pick antes/después.
  ETAPA 2 (cara): A/B de E[premio] con sorteos comunes, dos verdades del pool.
  ETAPA 3: contra RESULTADOS REALES (4 temporadas walk-forward): puntos de la mejor
  planilla, batacazos cubiertos, y la vara del chequeo de asignación —¿la mejor
  planilla nuestra le gana al máximo de N rivales simulados al azar?— que es el
  objetivo real y lo que las etapas 1-2 no miran directamente.

Reproducir: python scripts/backtest_menu_popular.py --etapa 1|2|3 [--k 2,3]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_AQUI = pathlib.Path(__file__).resolve().parent

from src.clausura import strategy  # noqa: E402
from src.clausura.economics import (  # noqa: E402
    PrizeConfig, SimConfig, flatten_grid, index_score,
)
from src.clausura.pool import PoolConfig, pool_distribution  # noqa: E402
from src.clausura.strategy import EvaluadorPortfolio, build_portfolio  # noqa: E402

# reuso del harness de cobertura (mismo entorno, mismas métricas)
from backtest_cobertura import cargar_entorno, cobertura_por_evento, lado_de, q_verdad  # noqa: E402

# Piso de partidos para que una temporada cuente en la etapa 3 (ver el guard abajo).
MIN_PARTIDOS_TEMPORADA = 60


def planilla_con(k_popular, **kw):
    prev = strategy.K_POPULAR
    strategy.K_POPULAR = k_popular
    try:
        return np.asarray(build_portfolio(**kw).picks)
    finally:
        strategy.K_POPULAR = prev


def goles_medios(picks):
    return float(np.mean([sum(index_score(int(i))) for i in picks.ravel()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--k", default="2,3", help="valores de K_POPULAR a probar")
    ap.add_argument("--sims", type=int, default=19200)
    ap.add_argument("--eval-seeds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--participaciones", type=int, default=12)
    a = ap.parse_args()
    ks = [int(x) for x in a.k.split(",")]

    if a.etapa == 3:
        return etapa3_historica(a, ks)

    eventos, resultados, grids, pred_grids = cargar_entorno()
    fecha_de = [ev["fecha_n"] for ev in eventos]
    pref = [bool(ev["preferencial"]) for ev in eventos]
    prize = PrizeConfig()
    creencia = [pool_distribution(g, PoolConfig()) for g in pred_grids]
    kw = dict(grids=grids, fecha_de_partido=fecha_de, preferencial=pref,
              n_participaciones=a.participaciones, prize=prize, pool_qs=creencia)

    if a.etapa == 1:
        sim = SimConfig(n_sims=a.sims, n_rivales=718, seed=20260812)
        t0 = time.time()
        base = planilla_con(0, sim=sim, **kw)
        print(f"base (K_POPULAR=0): goles medios por pick {goles_medios(base):.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        cb = cobertura_por_evento(base, eventos, resultados)
        nom = {ev["evento_id"]: f"{ev['local']} vs {ev['visitante']}" for ev in eventos}
        for k in ks:
            t0 = time.time()
            con = planilla_con(k, sim=sim, **kw)
            dif = int((base != con).sum())
            print(f"\nK_POPULAR={k}: picks distintos {dif} de {base.size} · goles medios "
                  f"{goles_medios(con):.2f} ({time.time()-t0:.0f}s)")
            if dif == 0:
                print("  el optimizador NO tomó ningún candidato popular.")
                continue
            cc = cobertura_por_evento(con, eventos, resultados)
            print(f"  {'partido':<40}{'antes L/E/V':>14}{'después L/E/V':>16}")
            for eid in cb:
                b, c = cb[eid], cc[eid]
                if b != c:
                    print(f"  {nom[eid]:<40}{b[1]:>4}/{b[0]}/{b[-1]}"
                          f"{c[1]:>8}/{c[0]}/{c[-1]}   ← cambió")
            # qué marcadores nuevos entraron (los que están en `con` y no en `base`)
            nuevos = {}
            for m, ev in enumerate(eventos):
                if ev["evento_id"] in resultados:
                    continue
                sb = {index_score(int(x)) for x in base[:, m]}
                sc = {index_score(int(x)) for x in con[:, m]}
                if sc - sb:
                    nuevos[nom[ev["evento_id"]]] = sorted(sc - sb)
            for n, s in nuevos.items():
                print(f"  + {n:<38} {', '.join(f'{g}-{v}' for g, v in s)}")
        print("\n  si hubo cambios: correr --etapa 2")
        return

    # ---- etapa 2: el A/B en plata ----
    qs = {"verdad": q_verdad(pred_grids), "control": creencia}
    acum = {k: {v: [] for v in qs} for k in ks}
    for rep in range(a.reps):
        sim = SimConfig(n_sims=a.sims, n_rivales=718, seed=20260812 + 7919 * rep)
        t0 = time.time()
        base = planilla_con(0, sim=sim, **kw)
        for k in ks:
            con = planilla_con(k, sim=sim, **kw)
            fila = []
            for v, q in qs.items():
                ev = EvaluadorPortfolio(grids, fecha_de, pref, q, prize, sim)
                c = ev.comparar(base, con, n_seeds=a.eval_seeds)
                acum[k][v].append(c.delta)
                fila.append(f"{v} {c.delta:+,.0f} ± {c.se:,.0f}")
            print(f"rep {rep+1} K={k} ({time.time()-t0:.0f}s)   " + "   ·   ".join(fila),
                  flush=True)

    print(f"\n{'='*66}\nRESUMEN — Δ E[premio] de K_POPULAR vs producción (0)\n{'='*66}")
    for k in ks:
        for v, xs in acum[k].items():
            se = float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0
            print(f"  K={k} {v:<10} {np.mean(xs):>+12,.0f} ± {se:,.0f}   ({len(xs)} reps)")


def _p_azar_gana(pool_max_samples, nuestro_max):
    """Fracción de sorteos en que el máx de N rivales al azar ≥ nuestro máx."""
    return float(np.mean(np.asarray(pool_max_samples) >= nuestro_max))


def etapa3_historica(a, ks):
    """Contra RESULTADOS REALES, walk-forward, con la vara del chequeo de asignación.

    Además de puntos y batacazos (como en cobertura), simula rivales i.i.d. ∝ Q^γ
    contra los resultados reales (mismo mecanismo que pool_pit) y mide P(máx de N
    rivales al azar ≥ nuestra mejor planilla): la métrica del postmortem, en el
    backtest. Menor es mejor.
    """
    from src.clausura.backtest import build_grids, realized_prizes, actual_indices
    from src.clausura.intermedio import load_dataset_completo
    from src.clausura.ratings import fit_ratings
    from src.clausura.scoring import supermatch_points

    partidos = load_dataset_completo()
    temporadas, orden = {}, []
    for pt in partidos:
        if pt.campeonato_id not in temporadas:
            temporadas[pt.campeonato_id] = []
            orden.append(pt.campeonato_id)
        temporadas[pt.campeonato_id].append(pt)

    prize = PrizeConfig()
    variantes = [0] + ks
    tot = {k: {"pts_max": [], "premio": [], "bat_cub": 0, "bat_tot": 0, "p_azar": [],
               "pts_med": []} for k in variantes}
    rng = np.random.default_rng(20260816)

    for i, cid in enumerate(orden):
        previas = [q for c in orden[:i] for q in temporadas[c]]
        if len(previas) < 120:
            continue
        ps = temporadas[cid]
        # Una temporada con casi ningún partido (típico: falta el dataset generado,
        # p.ej. data/processed/intermedio_2026.json, que NO está versionado) entra al
        # promedio como una fila de 3 puntos idéntica en los tres brazos y arrastra el
        # RESUMEN. Pasó en la corrida del 2026-08-17: 4 temporadas reportadas, 3
        # utilizables. Mejor saltarla y decirlo que promediar ruido en silencio.
        if len(ps) < MIN_PARTIDOS_TEMPORADA:
            print(f"  ⏭️  {ps[0].campeonato[:34]}: {len(ps)} partidos — temporada "
                  f"degenerada, se saltea (¿falta correr src.clausura.intermedio?)",
                  flush=True)
            continue
        ratings = fit_ratings(previas)
        grids = build_grids(ps, ratings)
        fechas = [q.fecha_id for q in ps]
        pref = [q.preferencial for q in ps]
        qs = [pool_distribution(g, PoolConfig()) for g in grids]
        real = actual_indices(ps)
        sim = SimConfig(n_sims=a.sims, n_rivales=718, seed=20260812)
        eval_sim = SimConfig(n_sims=2400, n_rivales=718, seed=20260812)

        # rivales simulados ∝ Q contra los resultados reales → distribución del máx
        # de N tickets al azar (2000 sorteos de N tickets)
        n_riv, n_part = 4000, a.participaciones
        pts_riv = np.zeros(n_riv)
        for m, q in enumerate(ps):
            r = index_score(int(real[m]))
            picks_m = rng.choice(36, size=n_riv, p=qs[m] / qs[m].sum())
            pts_riv += np.array([supermatch_points(index_score(int(x)), r, q.preferencial)
                                 for x in picks_m])
        max_azar = np.array([pts_riv[rng.choice(n_riv, n_part, replace=False)].max()
                             for _ in range(2000)])

        print(f"\n  {ps[0].campeonato[:34]}  (pool sim: mediana {np.median(pts_riv):.0f}, "
              f"máx de {n_part} al azar ~{max_azar.mean():.0f})")
        for k in variantes:
            picks = planilla_con(k, grids=grids, fecha_de_partido=fechas, preferencial=pref,
                                 n_participaciones=n_part, prize=prize, pool_qs=qs, sim=sim)
            premio, _, _ = realized_prizes(picks, ps, grids, qs, prize, eval_sim)
            pts = np.zeros(n_part)
            bat_cub = bat_tot = 0
            for m, q in enumerate(ps):
                r = index_score(int(real[m]))
                p_lado = flatten_grid(grids[m])
                lado_r = lado_de(real[m])
                prob_lado = sum(p_lado[j] for j in range(36) if lado_de(j) == lado_r)
                es_batacazo = prob_lado < 0.25
                bat_tot += es_batacazo
                cubierto = False
                for kk in range(n_part):
                    pk = index_score(int(picks[kk, m]))
                    pts[kk] += supermatch_points(pk, r, q.preferencial)
                    if lado_de(picks[kk, m]) == lado_r:
                        cubierto = True
                bat_cub += es_batacazo and cubierto
            d = tot[k]
            p_az = _p_azar_gana(max_azar, pts.max())
            d["pts_max"].append(pts.max()); d["pts_med"].append(pts.mean())
            d["premio"].append(premio); d["p_azar"].append(p_az)
            d["bat_cub"] += bat_cub; d["bat_tot"] += bat_tot
            print(f"    K={k}: mejor {pts.max():.0f} pts (media {pts.mean():.1f}) · "
                  f"azar nos gana {p_az:.0%} · batacazos {bat_cub}/{bat_tot} · "
                  f"premio ${premio:,.0f}", flush=True)

    n = len(tot[0]["pts_max"])
    print(f"\n{'='*70}\nRESUMEN vs REALIDAD ({n} temporadas)\n{'='*70}")
    for k in variantes:
        d = tot[k]
        print(f"  K_POPULAR={k}: mejor planilla {np.mean(d['pts_max']):.1f} pts · media "
              f"{np.mean(d['pts_med']):.1f} · azar nos gana {np.mean(d['p_azar']):.0%} · "
              f"batacazos {d['bat_cub']}/{d['bat_tot']} · premio medio "
              f"${np.mean(d['premio']):,.0f}")


if __name__ == "__main__":
    main()

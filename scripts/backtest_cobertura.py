"""¿Garantizar el mejor marcador de cada desenlace mejora el E[premio]?

EL MODO DE FALLA (Fecha 2, medido el 2026-08-12): en Peñarol vs Central Español el
mejor marcador de visitante está en el puesto 19 por E[pts] y el empate en el 7. El
menú corta en 5 → las 12 participaciones salen con CERO exposición al 43% de los
desenlaces. Y la Fecha 1 realizó exactamente eso: 0/12 picks de visitante en los 3
batacazos.

LO YA MEDIDO QUE ACOTA LA EXPECTATIVA:
  * menú de 8 (que SÍ incluía el empate, puesto 7) midió igual que 5 → la mitad
    "empate" de esta idea tiene evidencia previa en contra;
  * la rama de hueco ofrecía rareza (5-1, 5-0) y el optimizador JAMÁS la tomó
    ((5,3) ≡ (5,0) bit a bit) → ofrecer candidatos no obliga a usarlos.

Por eso ETAPA 1 (barata): ¿el optimizador siquiera TOMA los candidatos de cobertura?
Si las planillas salen idénticas, no hay nada que medir. ETAPA 2 (cara): el A/B de
E[premio] con sorteos comunes y las dos verdades de siempre.

Métrica secundaria de la etapa 1: cuántas participaciones cubren cada desenlace por
partido, antes y después. Si la cobertura sube pero el E[premio] no, eso también
informa: el modelo cree que esa exposición no vale — y la discusión pasa a ser si le
creemos al modelo del pool o a la Fecha 1.
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

from src.clausura import strategy  # noqa: E402
from src.clausura.economics import (  # noqa: E402
    PrizeConfig, SimConfig, flatten_grid, index_score,
)
from src.clausura.pool import PoolConfig, pool_distribution  # noqa: E402
from src.clausura.strategy import EvaluadorPortfolio, build_portfolio  # noqa: E402


def q_verdad(pred_grids):
    d = json.load(open(_AQUI / "bias_calibrado.json"))
    c = PoolConfig(chalk_strength=d["chalk"], temperature=1.0,
                   default_bias=d["default_bias"],
                   popular_bias={tuple(map(int, k.split("-"))): v
                                 for k, v in d["bias"].items()})
    return [pool_distribution(g, c) for g in pred_grids]


def cargar_entorno():
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
    try:
        odds_by_evento = match_odds(eventos, fetch_primera_odds())
    except Exception as e:
        print(f"sin odds ({e})", flush=True)
        odds_by_evento = {}
    grids, _, pred_grids, _ = build_season_grids(
        eventos, ensure_ratings(), odds_by_evento, resultados)
    return eventos, resultados, grids, pred_grids


def lado_de(idx):
    gl, gv = index_score(int(idx))
    return (gl > gv) - (gl < gv)


def cobertura_por_evento(picks, eventos, resultados):
    """{evento_id: {1: n, 0: n, -1: n}} solo para partidos sin resultado."""
    out = {}
    for m, ev in enumerate(eventos):
        if ev["evento_id"] in resultados:
            continue
        c = {1: 0, 0: 0, -1: 0}
        for k in range(picks.shape[0]):
            c[lado_de(picks[k, m])] += 1
        out[ev["evento_id"]] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--sims", type=int, default=19200)
    ap.add_argument("--eval-seeds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--participaciones", type=int, default=12)
    a = ap.parse_args()

    if a.etapa == 3:
        return etapa3_historica(a)

    eventos, resultados, grids, pred_grids = cargar_entorno()
    fecha_de = [ev["fecha_n"] for ev in eventos]
    pref = [bool(ev["preferencial"]) for ev in eventos]
    prize = PrizeConfig()
    creencia = [pool_distribution(g, PoolConfig()) for g in pred_grids]

    def planilla(cob, sim):
        strategy.K_COBERTURA = cob
        try:
            return np.asarray(build_portfolio(
                grids=grids, fecha_de_partido=fecha_de, preferencial=pref,
                n_participaciones=a.participaciones, prize=prize,
                pool_qs=creencia, sim=sim).picks)
        finally:
            strategy.K_COBERTURA = False

    if a.etapa == 1:
        sim = SimConfig(n_sims=a.sims, n_rivales=718, seed=20260812)
        t0 = time.time()
        base = planilla(False, sim)
        con = planilla(True, sim)
        dif = int((base != con).sum())
        print(f"\npicks distintos: {dif} de {base.size}  ({time.time()-t0:.0f}s)")
        if dif == 0:
            print("el optimizador NO tomó ningún candidato de cobertura — no hay nada "
                  "que medir en E[premio]. Fin.")
            return
        cb, cc = (cobertura_por_evento(x, eventos, resultados) for x in (base, con))
        nom = {ev["evento_id"]: f"{ev['local']} vs {ev['visitante']}" for ev in eventos}
        print(f"\n  {'partido':<40}{'antes L/E/V':>14}{'después L/E/V':>16}")
        for eid in cb:
            b, c = cb[eid], cc[eid]
            if b != c:
                print(f"  {nom[eid]:<40}{b[1]:>4}/{b[0]}/{b[-1]}"
                      f"{c[1]:>8}/{c[0]}/{c[-1]}   ← cambió")
        sin_v_antes = sum(1 for v in cb.values() if v[-1] == 0)
        sin_v_desp = sum(1 for v in cc.values() if v[-1] == 0)
        print(f"\n  partidos abiertos sin NINGÚN pick de visitante: "
              f"{sin_v_antes} → {sin_v_desp} (de {len(cb)})")
        print("\n  hay señal: correr --etapa 2")
        return

    # ---- etapa 2: el A/B en plata ----
    qs = {"verdad": q_verdad(pred_grids), "control": creencia}
    acum = {k: [] for k in qs}
    for rep in range(a.reps):
        sim = SimConfig(n_sims=a.sims, n_rivales=718, seed=20260812 + 7919 * rep)
        t0 = time.time()
        base = planilla(False, sim)
        con = planilla(True, sim)
        fila = []
        for k, q in qs.items():
            ev = EvaluadorPortfolio(grids, fecha_de, pref, q, prize, sim)
            c = ev.comparar(base, con, n_seeds=a.eval_seeds)
            acum[k].append(c.delta)
            fila.append(f"{k} {c.delta:+,.0f} ± {c.se:,.0f}")
        print(f"rep {rep+1} ({time.time()-t0:.0f}s)   " + "   ·   ".join(fila), flush=True)

    print(f"\n{'='*66}\nRESUMEN — Δ E[premio] de cobertura vs producción (5,0)\n{'='*66}")
    for k, xs in acum.items():
        se = float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0
        print(f"  {k:<10} {np.mean(xs):>+12,.0f} ± {se:,.0f}   ({len(xs)} reps)")


def etapa3_historica(a):
    """El test que el Monte Carlo no puede hacer: contra RESULTADOS REALES.

    Las etapas 1-2 evalúan bajo el modelo, y el argumento a favor de la cobertura es
    que el modelo subestima los batacazos (Fecha 1: 3 batacazos, 0/12 cubiertos).
    Acá se juegan las temporadas históricas walk-forward —ratings ajustados solo con
    lo anterior— y se liquida contra lo que de verdad pasó, batacazos incluidos.

    Métricas libres de modelo: puntos reales y batacazos cubiertos (resultado cuyo
    lado tenía P<25% en la grilla; ¿alguna de las 12 lo llevaba?). El premio realizado
    usa pool simulado (los picks ajenos históricos no existen) y es TODO-O-NADA con
    n=4 temporadas: se reporta, pero la señal está en las otras dos.
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
    tot = {False: {"pts_max": [], "premio": [], "bat_cub": 0, "bat_tot": 0},
           True: {"pts_max": [], "premio": [], "bat_cub": 0, "bat_tot": 0}}

    for i, cid in enumerate(orden):
        previas = [q for c in orden[:i] for q in temporadas[c]]
        if len(previas) < 120:
            continue
        ps = temporadas[cid]
        ratings = fit_ratings(previas)
        grids = build_grids(ps, ratings)
        fechas = [q.fecha_id for q in ps]
        pref = [q.preferencial for q in ps]
        qs = [pool_distribution(g, PoolConfig()) for g in grids]
        real = actual_indices(ps)
        sim = SimConfig(n_sims=a.sims, n_rivales=718, seed=20260812)
        eval_sim = SimConfig(n_sims=2400, n_rivales=718, seed=20260812)

        fila = [ps[0].campeonato[:28]]
        for cob in (False, True):
            strategy.K_COBERTURA = cob
            try:
                picks = np.asarray(build_portfolio(
                    grids=grids, fecha_de_partido=fechas, preferencial=pref,
                    n_participaciones=a.participaciones, prize=prize,
                    pool_qs=qs, sim=sim).picks)
            finally:
                strategy.K_COBERTURA = False
            premio, _, _ = realized_prizes(picks, ps, grids, qs, prize, eval_sim)

            # puntos reales por participación (kernel exacto, sin pool)
            pts = np.zeros(a.participaciones)
            bat_cub = bat_tot = 0
            for m, q in enumerate(ps):
                r = index_score(int(real[m]))
                p_lado = flatten_grid(grids[m])
                lado_r = lado_de(real[m])
                prob_lado = sum(p_lado[j] for j in range(36) if lado_de(j) == lado_r)
                es_batacazo = prob_lado < 0.25
                bat_tot += es_batacazo
                cubierto = False
                for k in range(a.participaciones):
                    pk = index_score(int(picks[k, m]))
                    pts[k] += supermatch_points(pk, r, q.preferencial)
                    if lado_de(picks[k, m]) == lado_r:
                        cubierto = True
                bat_cub += es_batacazo and cubierto
            d = tot[cob]
            d["pts_max"].append(pts.max()); d["premio"].append(premio)
            d["bat_cub"] += bat_cub; d["bat_tot"] += bat_tot
            fila.append(f"{'CON' if cob else 'sin'}: mejor {pts.max():.0f} pts · "
                        f"batacazos {bat_cub}/{bat_tot} · premio ${premio:,.0f}")
        print(f"  {fila[0]:<30} {fila[1]}\n  {'':<30} {fila[2]}", flush=True)

    print(f"\n{'='*70}\nRESUMEN vs REALIDAD ({len(tot[False]['pts_max'])} temporadas)\n{'='*70}")
    for cob in (False, True):
        d = tot[cob]
        print(f"  {'CON cobertura' if cob else 'sin cobertura'}: "
              f"mejor planilla {np.mean(d['pts_max']):.1f} pts de media · "
              f"batacazos cubiertos {d['bat_cub']}/{d['bat_tot']} · "
              f"premio medio ${np.mean(d['premio']):,.0f}")


if __name__ == "__main__":
    main()

"""Calibración del pool del Clausura 2026 con la temporada entera → prior de la próxima penca.

La concentración del pool vigente (PoolConfig.chalk_strength=2,2) salió de 4.791 picks
de 7 partidos de la F1. Con el replay (scripts/replay_dataset.py) hay ~45.000 picks de
64 partidos. Este script:

  1. Reajusta chalk por máxima verosimilitud contra los picks reales (mismo
     pool_distribution de producción, grilla de producción pre-cierre), con validación
     dejando UNA FECHA afuera, y por fecha para ver si el pool se mueve en la temporada.
  2. Mide el comportamiento de los rivales por fecha: carga (abandono), tasa de exactos
     del pool, cuánto juega al favorito, goles medios del pick.
  3. Especiales: campeón y goleador elegidos, % sin especiales.
  4. Mide en PLATA qué chalk conviene usar en el optimizador, con la verdad de los
     rivales REALES (premio de fecha, F1-F8): la lección del 11/8 es que chalk es una
     perilla de diferenciación, no una creencia, así que el MLE no alcanza para elegirlo.

Salida: data/pool_prior/clausura2026.json (prior para arrancar la próxima penca).

Resultado del 2/10 (42.044 picks, 64 partidos):
  * chalk MLE 1,5, ESTABLE en las 8 fechas (1,4-1,5); log-loss dejando una fecha
    afuera 2,253 vs 2,310 con 2,2.
  * Producción ya juega cerca de eso: la temperatura que calibra online (1,2-2,0 desde
    la F2) deja la concentración efectiva chalk/T en 1,1-1,8. El 2,2 solo pesó al
    ARRANCAR (F1 con T=1,0 → 2,2 efectivo).
  * En plata (premio de fecha F1-F8, rivales reales, T=1): 1,5 +$253 vs 2,2 (7/8
    fechas), 3,0 ≈ 2,2. Chico, y ya capturado por la temperatura online.
  ⇒ Para la PRÓXIMA penca arrancar con concentración efectiva 1,5 (chalk 1,5, T=1),
    no 2,2: es lo que el pool del Clausura jugó todas las fechas.
  * Rivales: carga 94% → 81% de F1 a F8 (−1,6 pp por fecha), exactos del pool 7-14%
    por fecha, ~59% de picks al favorito, 2,2 goles por pick. 40% sin especiales;
    campeón Peñarol 69% / Nacional 27%.

    python -m scripts.pool_prior [--sims 19200] [--eval 20000]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

from src.clausura.economics import N_SCORES, index_score
from src.clausura.pool import PoolConfig, pool_distribution
from scripts.replay_clausura import DATASET, armar_verdad, ascenso, cargar, evaluar

OUT = Path("data/pool_prior/clausura2026.json")
CHALKS = np.round(np.arange(1.0, 4.01, 0.1), 2)


def loglik(f, chalk: float, temperatura: float = 1.0) -> tuple[float, int]:
    ll, n = 0.0, 0
    for m, g in enumerate(f.prod):
        q = pool_distribution(g, PoolConfig(chalk_strength=chalk, temperature=temperatura))
        pk = f.riv[:, m]
        pk = pk[pk >= 0]
        ll += float(np.log(np.maximum(q[pk], 1e-12)).sum())
        n += len(pk)
    return ll, n


def ajuste(fechas, excluir: int | None = None) -> float:
    tot = {c: sum(loglik(f, c)[0] for k, f in fechas.items() if k != excluir) for c in CHALKS}
    return float(max(tot, key=tot.get))


def comportamiento(fechas) -> list[dict]:
    out = []
    for n, f in fechas.items():
        R = f.riv.shape[0]
        carga = (f.riv >= 0).any(axis=1)
        ex, fav, goles, tot = 0, 0, 0.0, 0
        for m, g in enumerate(f.prod):
            pk = f.riv[carga & (f.riv[:, m] >= 0), m]
            ex += int((pk == f.real[m]).sum())
            p_loc = sum(g[i, j] for i in range(g.shape[0]) for j in range(g.shape[1]) if i > j)
            p_vis = sum(g[i, j] for i in range(g.shape[0]) for j in range(g.shape[1]) if j > i)
            sc = np.array([index_score(int(x)) for x in pk])
            if len(sc):
                lado = np.sign(sc[:, 0] - sc[:, 1])
                fav += int((lado == (1 if p_loc >= p_vis else -1)).sum())
                goles += float(sc.sum())
            tot += len(pk)
        out.append({"fecha": n, "rivales": R, "cargaron": round(float(carga.mean()), 3),
                    "exactos": round(ex / max(tot, 1), 4),
                    "al_favorito": round(fav / max(tot, 1), 3),
                    "goles_pick": round(goles / max(tot, 1), 2)})
    return out


def especiales(d: dict) -> dict:
    mis = set(d["mis_numeros"])
    riv = [p for p in d["participaciones"] if p["numero"] not in mis]
    camp = collections.Counter(p["campeon"] for p in riv if p.get("campeon"))
    gol = collections.Counter(p["goleador"] for p in riv if p.get("goleador"))
    nc = sum(camp.values())
    ng = sum(gol.values())
    return {"rivales": len(riv),
            "sin_campeon": round(1 - nc / len(riv), 3), "sin_goleador": round(1 - ng / len(riv), 3),
            "campeon": {k: round(v / nc, 3) for k, v in camp.most_common(8)},
            "goleador": {k: round(v / ng, 3) for k, v in gol.most_common(8)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=19_200)
    ap.add_argument("--eval", type=int, default=20_000)
    ap.add_argument("--chalks", default="1.5,2.2,3.0")
    a = ap.parse_args()
    fechas, d = cargar(DATASET)

    # 1. MLE global, por fecha, y leave-one-fecha-out
    global_ = ajuste(fechas)
    por_fecha = {n: ajuste({n: f}) for n, f in fechas.items()}
    lofo = []
    for n, f in fechas.items():
        c = ajuste(fechas, excluir=n)
        ll_fit, k = loglik(f, c)
        ll_22, _ = loglik(f, 2.2)
        lofo.append({"fecha": n, "chalk_ajustado": c, "ll_ajustado": ll_fit / k, "ll_2.2": ll_22 / k})
    ll_fit = np.mean([x["ll_ajustado"] for x in lofo])
    ll_22 = np.mean([x["ll_2.2"] for x in lofo])
    n_picks = sum(loglik(f, 2.2)[1] for f in fechas.values())
    print(f"== pool: {n_picks:,} picks de {sum(len(f.eventos) for f in fechas.values())} partidos ==")
    print(f"chalk MLE global {global_} (producción 2,2)")
    print("por fecha: " + " · ".join(f"F{n} {c}" for n, c in por_fecha.items()))
    print(f"log-loss por pick dejando una fecha afuera: ajustado {-ll_fit:.4f} vs 2,2 {-ll_22:.4f}")

    # 2. comportamiento
    comp = comportamiento(fechas)
    print("\n== rivales por fecha ==")
    for c in comp:
        print(f"F{c['fecha']}: cargaron {c['cargaron']:.0%} · exactos {c['exactos']:.1%} · "
              f"al favorito {c['al_favorito']:.0%} · goles/pick {c['goles_pick']}")

    # 3. especiales
    esp = especiales(d)
    print(f"\n== especiales ==\nsin campeón {esp['sin_campeon']:.0%} · sin goleador {esp['sin_goleador']:.0%}")
    print("campeón:", esp["campeon"])
    print("goleador:", esp["goleador"])

    # 4. plata: qué chalk usar en el optimizador, con rivales reales
    print("\n== chalk del optimizador, verdad = rivales REALES (premio de fecha F1-F8) ==")
    verdades = {n: armar_verdad(f, a.eval, 9000 + n) for n, f in fechas.items()}
    plata = {}
    base = None
    for c in sorted({float(x) for x in a.chalks.split(",")} | {global_}):
        filas = []
        for n, f in fechas.items():
            q_orig = f.q_modelo
            f.q_modelo = [pool_distribution(g, PoolConfig(chalk_strength=c)) for g in f.prod]
            filas.append(evaluar(f, ascenso(f, 12, 5, a.sims, 100 + n), verdades[n]))
            f.q_modelo = q_orig
        e = sum(r["e_real"] for r in filas)
        plata[c] = {"e_premio_fechas": round(e), "por_fecha": [round(r["e_real"]) for r in filas]}
        if c == 2.2:
            base = filas
        print(f"chalk {c}: ${e:,.0f}", flush=True)
    if base is not None:
        for c, v in plata.items():
            dd = np.array(v["por_fecha"]) - np.array(plata[2.2]["por_fecha"])
            v["delta_vs_2.2"] = int(dd.sum())
            v["mejor_en"] = int((dd > 0).sum())
            print(f"  {c}: Δ vs 2,2 ${dd.sum():+,.0f} (mejor en {(dd > 0).sum()}/{len(dd)} fechas)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "fuente": "Clausura 2026, picks reales F1-F8 (replay_dataset), grilla de producción pre-cierre",
        "picks": n_picks,
        "chalk_mle": global_, "chalk_por_fecha": por_fecha,
        "logloss_lofo": {"ajustado": round(-ll_fit, 4), "2.2": round(-ll_22, 4)},
        "comportamiento": comp, "especiales": esp, "chalk_en_plata": plata,
    }, ensure_ascii=False, indent=1))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()

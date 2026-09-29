"""K_EV 5 vs 8 con el PIPELINE DE PRODUCCIÓN y el estado real de la carrera.

El A/B por fecha del 29/9 (scripts/ab_cap_diversidad_vivo.py) midió el menú de 8
en +$3.613 ± 507 con un objetivo de "ganar un bloque" — parecido a ir muy atrás —
contra +$565 del arnés de temporada del 4/9 y la meseta 5≈8 del 11/8. Ese arnés no
tiene standings ni RivalModel, así que no alcanza para tocar producción. Esto sí:
picks.run() entero (ratings, cuotas, RivalModel con los puntos vivos del ranking,
especiales congelados, frozen de lo jugado) sobre la fecha objetivo, una vez por
brazo, con la MISMA semilla de optimización por rep, en frío (sin warm start: la
cadena de planillas se armó con K_EV=5 y favorecería al control), sin guardar ni
notificar.

Las dos matrices se liquidan con el evaluador de la corrida A (las grillas, el pool
y los rivales no dependen del menú) con sorteos FRESCOS y comunes: Δ = B − A.

Uso (VPS, env cargado; ~30-45 min por corrida, 2 por rep):
    PYTHONPATH=/opt/penca python -m scripts.ab_menu_produccion --fecha 9 --reps 2
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math

import numpy as np

from src.clausura import picks as picks_mod
from src.clausura import strategy
from src.clausura.economics import SimConfig

log = logging.getLogger(__name__)


def correr(fecha: int, k_ev: int, seed: int, sims: int, n_part: int) -> dict:
    strategy.K_EV = k_ev
    picks_mod.SimConfig = functools.partial(SimConfig, seed=seed)
    ctx: dict = {}
    picks_mod.run(fecha, n_part, telegram=False, n_sims=sims,
                  usar_warm_start=False, guardar=False, contexto=ctx)
    return ctx


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", type=int, required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sims", type=int, default=19_200)
    ap.add_argument("--participaciones", type=int, default=12)
    ap.add_argument("--k-control", type=int, default=5)
    ap.add_argument("--k-brazo", type=int, default=8)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    filas = []
    for rep in range(args.reps):
        seed = 20260807 + 7919 * rep
        a = correr(args.fecha, args.k_control, seed, args.sims, args.participaciones)
        b = correr(args.fecha, args.k_brazo, seed, args.sims, args.participaciones)
        ev = a["evaluador"]
        comp = ev.comparar(a["portfolio"].picks, b["portfolio"].picks,
                           n_seeds=args.eval_seeds)
        pa, pb = a["portfolio"].picks, b["portfolio"].picks
        cols = [m for m in range(pa.shape[1]) if not np.array_equal(pa[:, m], pb[:, m])]
        dist = lambda p: float(np.mean([len(set(p[:, m])) for m in cols])) if cols else 0.0
        fila = {"rep": rep, "delta": comp.delta, "se": comp.se,
                "valor_a": comp.valor_a, "valor_b": comp.valor_b,
                "partidos_distintos": len(cols),
                "distintos_a": dist(pa), "distintos_b": dist(pb)}
        filas.append(fila)
        print(f"rep {rep}: A(K={args.k_control}) ${comp.valor_a:,.0f} · "
              f"B(K={args.k_brazo}) ${comp.valor_b:,.0f} · Δ {comp.delta:+,.0f} ± {comp.se:,.0f}"
              f" · {len(cols)} partidos cambian · marcadores distintos "
              f"{fila['distintos_a']:.2f} → {fila['distintos_b']:.2f}", flush=True)

    d = np.array([f["delta"] for f in filas])
    se = (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else filas[0]["se"]
    print(f"\nΔ E[premio] K_EV {args.k_control}→{args.k_brazo}: {d.mean():+,.0f} ± {se:,.0f}"
          f" ({len(d)} reps; umbral de acción $2.000)")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(filas, fh)


if __name__ == "__main__":
    main()

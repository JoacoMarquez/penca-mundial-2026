"""Ajusta el modelo del pool (Q: γ, w_banda; participación; especiales) con una penca pasada.

Usa las probabilidades walk-forward del modelo de producción al momento de cada partido
(lo que cualquiera "veía" antes de cargar), así el γ ajustado es consistente con el
modelo que después alimenta la simulación.

Uso:  python -m src.lub.calibrar [--penca 37]   → data/lub/pool_model_{penca}.json
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from src.lub.pit import perfiles_desde_pool, sorteo_real
from src.lub.pool import PoolModel, fit_q, load_pool, picks_por_evento


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penca", type=int, default=37)
    a = ap.parse_args()
    pool = load_pool(f"data/lub/pool_{a.penca}.json")
    so, eventos = sorteo_real(n_sims=1)
    byev = picks_por_evento(pool)
    P, C = [], []
    for j, s in enumerate(so.slots):
        for c in byev.get(s.evento_id, []):
            P.append(so.probs[0, j]); C.append(c)
    g, w, ll = fit_q(np.array(P), np.array(C))
    F = pool["participaciones"]
    fase = collections.defaultdict(list)
    for s in so.slots:
        fase[s.fase].append(len(byev.get(s.evento_id, [])) / len(F))
    camp = collections.Counter(f["campeon"] for f in F if f["campeon"])
    n_camp = sum(camp.values())
    pm = PoolModel(gamma=g, w_banda=w, participacion={k: float(np.mean(v)) for k, v in fase.items()},
                   campeon={k: v / n_camp for k, v in camp.items()})
    # perfiles de actividad/estilo por rival (agregados: sin números de participación),
    # para que el VPS no necesite el crudo de 4,7 MB
    probs_ev = {s.evento_id: so.probs[0, j] for j, s in enumerate(so.slots)}
    perf = perfiles_desde_pool(pool, eventos, probs_ev, pm)
    d = pm.to_json()
    d["perfiles"] = {"part": np.round(perf.part, 3).tolist(), "esp": perf.esp.astype(int).tolist(),
                     "gamma": np.round(perf.gamma, 3).tolist()}
    out = f"data/lub/pool_model_{a.penca}.json"
    json.dump(d, open(out, "w"), ensure_ascii=False)
    print(f"γ={g:.2f} w_banda={np.round(w, 2)} ll={ll:.3f} → {out}")


if __name__ == "__main__":
    main()

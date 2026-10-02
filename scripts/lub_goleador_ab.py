"""A/B: especiales optimizados con el goleador CONJUNTO (temporada) vs INDEPENDIENTE.

Brazo A = producción nueva (goleador simulado sobre los mismos sorteos de la
temporada). Brazo B = mismo P(goleador) marginal pero sorteado independiente de la
temporada (lo que hacía el prior a mano). Se liquidan los dos con la MISMA verdad —el
modelo conjunto— en sorteos independientes de la optimización, con los mismos picks
de la fecha (los de A) para aislar los especiales.

    python -m scripts.lub_goleador_ab --sims 4000 --reps 5
"""
import argparse
from datetime import datetime, timezone

import numpy as np

from src.lub.data import DATA_DIR, load_temporadas
from src.lub.goleador import cargar_candidatos, simular_goleador
from src.lub.model import PARAMS_PROD, fit
from src.lub.picks import (CAMPEON_EXP, GOLEADOR_EXP, KAPPA_POOL, TEMPORADA, correr, fecha_actual,
                           slots_temporada)
from src.lub.pool import PoolModel, perfiles_de_json
from src.lub.portfolio import Evaluador, evaluar
from src.lub.scoring import clase
from src.lub.season import Config, simular, simular_rivales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--rivales", type=int, default=320)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    import json
    now = datetime.now(timezone.utc)
    cands = cargar_candidatos()
    nombres = [c.nombre for c in cands]
    partidos = load_temporadas()
    rt = fit(partidos, now, TEMPORADA, PARAMS_PROD)

    pl_a = correr(a.k, a.sims, a.rivales, refrescar=False, now=now)
    so0 = simular(rt, slots_temporada(partidos), Config(n_sims=a.sims, n_rivales=a.rivales))
    g0, _ = simular_goleador(so0.partidos, so0.equipos, cands, seed=Config().seed + 77)
    p_marg = np.bincount(g0, minlength=len(cands)) / a.sims + 1e-5
    p_marg /= p_marg.sum()
    pl_b = correr(a.k, a.sims, a.rivales, refrescar=False, now=now, goleador=(nombres, p_marg))
    print("A conjunto:  ", list(zip(pl_a["campeon"], pl_a["goleador"])))
    print("B independ.:", list(zip(pl_b["campeon"], pl_b["goleador"])))

    actuales = fecha_actual(partidos, now)
    ids = [p.evento_id for p in actuales]
    picks = np.array([[clase(*f["picks"][e]) for f in pl_a["partidos"]] for e in range(a.k)])
    pm_json = json.loads((DATA_DIR / "pool_model_37.json").read_text())
    pm = PoolModel.from_json(pm_json)
    deltas, sin_gol = [], []
    for r in range(a.reps):
        cfg = Config(n_sims=a.sims, n_rivales=a.rivales, seed=777_000 + r)
        so = simular(rt, slots_temporada(partidos), cfg)
        gol, _ = simular_goleador(so.partidos, so.equipos, cands, seed=cfg.seed + 77)
        p_camp = np.bincount(so.campeon, minlength=len(so.equipos)) / a.sims
        shares = np.power(p_camp + 1e-4, CAMPEON_EXP); shares /= shares.sum()
        gsh = np.power(p_marg, GOLEADOR_EXP); gsh /= gsh.sum()
        simular_rivales(so, cfg, pm.q, shares, gol, gsh, perfiles=perfiles_de_json(pm_json), kappa=KAPPA_POOL)
        idx = [[j for j, s in enumerate(so.slots) if s.evento_id == e][0] for e in ids]
        ev = Evaluador(so, a.k, idx, seed=8, goleador_real=gol, n_goleadores=len(cands))
        res = []
        for pl in (pl_a, pl_b):
            camp = np.array([so.equipos.index(c) for c in pl["campeon"]])
            g = np.array([nombres.index(n) for n in pl["goleador"]])
            res.append(evaluar(ev, picks, camp, g))
        camp = np.array([so.equipos.index(c) for c in pl_a["campeon"]])
        sin = evaluar(ev, picks, camp, np.full(a.k, -1))
        sin_gol.append((res[0]["_tot"] - sin["_tot"]).mean())
        d = res[0]["_tot"] - res[1]["_tot"]
        deltas.append(d.mean())
        print(f"rep {r}: A ${res[0]['e_premio']:,.0f}  B ${res[1]['e_premio']:,.0f}  "
              f"Δ ${d.mean():+,.0f} ± {d.std() / np.sqrt(len(d)):,.0f}", flush=True)
    dd = np.array(deltas)
    print(f"\nΔ conjunto − independiente: ${dd.mean():+,.0f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):,.0f} "
          f"({len(dd)} reps)")
    sg = np.array(sin_gol)
    print(f"Δ con goleador − sin goleador (lo que pasaba sin menú): ${sg.mean():+,.0f} ± "
          f"{sg.std(ddof=1) / np.sqrt(len(sg)):,.0f}")


if __name__ == "__main__":
    main()

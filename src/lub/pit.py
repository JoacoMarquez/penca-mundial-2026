"""PIT del pool: ¿el modelo de rivales genera un máximo tan alto como el real?

Re-juega la LUB 25/26 con los RESULTADOS REALES (sin azar en los partidos) y rivales
simulados (Q + perfiles de actividad bootstrapeados de la misma penca), y ubica:

  - el máximo real de la temporada (578) en la distribución simulada del máximo;
  - el máximo real de cada fecha en la distribución simulada de su máximo.

PIT ≈ 0,5 ⇒ calibrado. PIT alto (el real queda en la cola alta) ⇒ el modelo subestima
la vara para ganar y el optimizador jugaría demasiado conservador. Es la misma prueba
que `src/clausura/pool_pit.py` (ver memoria del cap de diversidad).

Ojo: los perfiles y Q se ajustan con la misma temporada que se testea — el PIT mide
si la ESTRUCTURA del modelo (i.i.d. dado el perfil) reproduce la cola, no poder
predictivo fuera de muestra.

Uso:  python -m src.lub.pit
"""

from __future__ import annotations

import collections
import json
from datetime import datetime

import numpy as np

from src.lub.data import load_temporadas
from src.lub.model import Params, fit, probs_partido
from src.lub.pool import PoolModel, load_pool, picks_por_evento
from src.lub.scoring import clase, puntos
from src.lub.season import KAPPA_POOL, Config, Perfiles, Slot, Sorteo, ordenar_por_fecha, simular_rivales

from src.lub.model import PARAMS_PROD as PARAMS


def perfiles_desde_pool(pool: dict, eventos: dict, probs_evento: dict | None = None,
                        pm: PoolModel | None = None, min_picks: int = 60) -> Perfiles:
    """Perfiles de actividad (y estilo γ_i si se pasan probs_evento y pm) de cada rival real."""
    from scipy.optimize import minimize_scalar
    n_por_fase = collections.Counter(e.fase for e in eventos.values() if e.pts_local is not None)
    part, esp, gam = [], [], []
    w10 = np.concatenate([pm.w_banda, pm.w_banda]) if pm is not None else None
    for f in pool["participaciones"]:
        c = collections.Counter(eventos[p["evento_id"]].fase for p in f["picks"] if p["evento_id"] in eventos)
        part.append([c[ph] / max(n_por_fase[ph], 1) for ph in ("regular", "liguilla", "playoff")])
        esp.append(f["campeon"] is not None)
        g = pm.gamma if pm is not None else 1.0
        if probs_evento is not None and pm is not None:
            pk = [(probs_evento[p["evento_id"]], clase(p["local"], p["visitante"])) for p in f["picks"]
                  if p["evento_id"] in probs_evento and p["local"] != p["visitante"]]
            if len(pk) >= min_picks:
                P = np.log(np.clip(np.array([x[0] for x in pk]), 1e-9, 1)); C = np.array([x[1] for x in pk])

                def nll(gg):
                    s = gg * P + np.log(w10); s = s - s.max(1, keepdims=True)
                    return -(s[np.arange(len(C)), C] - np.log(np.exp(s).sum(1))).mean()
                g = minimize_scalar(nll, bounds=(0.0, 6.0), method="bounded").x
        gam.append(g)
    return Perfiles(part=np.clip(np.array(part), 0, 1), esp=np.array(esp), gamma=np.array(gam))


def sorteo_real(temporada: str = "LUB 25/26", n_sims: int = 2000) -> tuple[Sorteo, dict]:
    ps = load_temporadas()
    evs = sorted([p for p in ps if p.temporada == temporada and p.pts_local is not None],
                 key=lambda p: p.inicio_utc)
    cache: dict[str, object] = {}
    probs, clases, slots = [], [], []
    for e in evs:
        d = e.inicio_utc[:10]
        if d not in cache:
            cache[d] = fit(ps, datetime.fromisoformat(e.inicio_utc), temporada, PARAMS)
        probs.append(probs_partido(cache[d], e.local, e.visitante))
        clases.append(clase(e.pts_local, e.pts_visitante))
        slots.append(Slot(e.fecha_nombre, e.fase, e.local, e.visitante, e.preferencial, e.evento_id,
                          (e.pts_local, e.pts_visitante)))
    fechas = list(dict.fromkeys(s.fecha for s in slots))
    fidx = {f: i for i, f in enumerate(fechas)}
    S = n_sims
    equipos = sorted({s.local for s in slots} | {s.visitante for s in slots})
    campeon = "Peñarol" if temporada == "LUB 25/26" else None
    so = Sorteo(slots=slots, fechas=fechas, fecha_de_slot=np.array([fidx[s.fecha] for s in slots]),
                jugado=np.ones((S, len(slots)), bool),
                clase=np.tile(np.array(clases, np.int8), (S, 1)),
                probs=np.tile(np.array(probs, np.float32)[None], (S, 1, 1)),
                pref=np.array([s.preferencial for s in slots]),
                campeon=np.full(S, equipos.index(campeon)), equipos=equipos)
    return ordenar_por_fecha(so), {e.evento_id: e for e in evs}


def main() -> None:
    pool = load_pool("data/lub/pool_37.json")
    pm = PoolModel.from_json(json.load(open("data/lub/pool_model_37.json")))
    so, eventos = sorteo_real()
    probs_ev = {sl.evento_id: so.probs[0, j] for j, sl in enumerate(so.slots)}
    perf = perfiles_desde_pool(pool, eventos, probs_ev, pm)
    shares = np.array([pm.campeon.get(e, 0.0) for e in so.equipos]) + 1e-6
    shares /= shares.sum()
    # goleador: 53/187 de los que cargaron acertaron (Vescovi) → índice 0 = el real
    gol_real = np.zeros(so.clase.shape[0], int)
    gol_sh = np.array([53 / 187, 1 - 53 / 187])
    cfg = Config(n_sims=so.clase.shape[0], n_rivales=len(pool["participaciones"]))
    import os
    k, sp = float(os.environ.get('KAPPA', KAPPA_POOL)), float(os.environ.get('KAPPA_SPREAD', 0.0))
    tot = simular_rivales(so, cfg, pm.q, shares, gol_real, gol_sh, perf, kappa=k, kappa_spread=sp)
    print(f'κ={k} ± {sp}')

    # reales
    real_tot = max(f["puntos_totales"] for f in pool["participaciones"])
    by_fecha: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    for f in pool["participaciones"]:
        for p in f["picks"]:
            e = eventos.get(p["evento_id"])
            if e is not None:
                by_fecha[e.fecha_nombre][f["participacion_id"]] += puntos((p["local"], p["visitante"]),
                                                                         (e.pts_local, e.pts_visitante), e.preferencial)
    pit_tot = (so.riv_total_max < real_tot).mean() + 0.5 * (so.riv_total_max == real_tot).mean()
    print(f"TEMPORADA: máx real {real_tot} | simulado mediana {np.median(so.riv_total_max):.0f} "
          f"[p5 {np.percentile(so.riv_total_max, 5):.0f}, p95 {np.percentile(so.riv_total_max, 95):.0f}] → PIT {pit_tot:.2f}")
    print(f"  mediana real {np.median([f['puntos_totales'] for f in pool['participaciones']]):.0f} "
          f"vs simulada {np.median(tot):.0f}")
    pits = []
    for i, fe in enumerate(so.fechas):
        rm = max(by_fecha[fe].values()) if by_fecha[fe] else 0
        sm = so.riv_fecha_max[:, i]
        pits.append((sm < rm).mean() + 0.5 * (sm == rm).mean())
    pits = np.array(pits)
    print(f"FECHAS ({len(pits)}): PIT medio {pits.mean():.2f}  (0,5 = calibrado)  "
          f"cuartiles {np.round(np.percentile(pits, [25, 50, 75]), 2)}")
    print("  reales vs simulados (mediana) por fecha:",
          [(max(by_fecha[fe].values()) if by_fecha[fe] else 0, int(np.median(so.riv_fecha_max[:, i])))
           for i, fe in enumerate(so.fechas)][:12], "...")


if __name__ == "__main__":
    main()

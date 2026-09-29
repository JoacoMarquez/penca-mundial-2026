"""Pipeline de la penca LUB: datos → ratings → cuotas → temporada → pool → portfolio → planilla.

Uso:
    python -m src.lub.picks [--k 12] [--sims 4000] [--rivales 320] [--no-guardar] [--telegram]
    python -m src.lub.picks --sweep 1,4,8,12,16,20     # E[premio] vs cantidad de participaciones

La planilla se versiona en data/lub/planillas/v{N}_{ts}.json (nunca se pisa) y se
imprime lista para cargar a mano: por participación, el marcador de cada partido y
los especiales.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.lub import odds as lub_odds
from src.lub.data import CAMPEONATO_ID, DATA_DIR, Partido, fetch_temporadas, load_temporadas, save_temporadas
from src.lub.model import PARAMS_PROD, fit
from src.lub.pool import PoolModel, perfiles_de_json
from src.lub.portfolio import Evaluador, evaluar, optimizar
from src.lub.scoring import BANDAS, N_BANDAS, marcador_de_clase
from src.lub.season import KAPPA_POOL, Config, Slot, simular, simular_rivales

log = logging.getLogger(__name__)

TEMPORADA = "LUB 26/27"
PRECIO = 200.0
MARKET_WEIGHT = 0.7          # igual que el Clausura: 70% mercado / 30% ratings
TOTAL_DEFAULT = 152.0        # total medio de la LUB (24/25–25/26) si no hay línea de total
CAMPEON_EXP = 2.5            # shares del pool por campeón ∝ P(campeón)^2,5: el top-3 junta ~90% como en 25/26


def _now() -> datetime:
    return datetime.now(timezone.utc)


def slots_temporada(partidos: list[Partido]) -> list[Slot]:
    """Fase regular 26/27: las fechas publicadas + la segunda rueda espejada (localía invertida)."""
    cur = sorted([p for p in partidos if p.temporada == TEMPORADA and p.fase == "regular"],
                 key=lambda p: (int(p.fecha_nombre.split()[-1]), p.inicio_utc))
    publicadas = {p.fecha_nombre for p in cur}
    slots = [Slot(p.fecha_nombre, "regular", p.local, p.visitante, p.preferencial, p.evento_id,
                  (p.pts_local, p.pts_visitante) if p.pts_local is not None else None) for p in cur]
    n_pub = len(publicadas)
    if n_pub < 22:
        primera = [s for s in slots if int(s.fecha.split()[-1]) <= 11]
        for s in primera:
            n = int(s.fecha.split()[-1]) + 11
            if f"Fecha {n}" not in publicadas:
                slots.append(Slot(f"Fecha {n}", "regular", s.visitante, s.local, s.preferencial))
    return slots


def fecha_actual(partidos: list[Partido], now: datetime) -> list[Partido]:
    """Partidos de la próxima fecha con pronóstico todavía abierto."""
    abiertos = sorted([p for p in partidos if p.temporada == TEMPORADA and p.pts_local is None
                       and datetime.fromisoformat(p.cierre_utc) > now], key=lambda p: p.inicio_utc)
    if not abiertos:
        return []
    f = abiertos[0].fecha_nombre
    return [p for p in abiertos if p.fecha_nombre == f]


def correr(k: int, n_sims: int, n_rivales: int, refrescar: bool = True, now: datetime | None = None,
           campeon_fijo: list[str] | None = None) -> dict:
    now = now or _now()
    if refrescar:
        save_temporadas(fetch_temporadas())
    partidos = load_temporadas()
    rt = fit(partidos, now, TEMPORADA, PARAMS_PROD)
    jugados = collections.Counter()
    for p in partidos:
        if p.temporada == TEMPORADA and p.pts_local is not None:
            jugados[p.local] += 1; jugados[p.visitante] += 1

    actuales = fecha_actual(partidos, now)
    # cuotas de mercado para la fecha actual (si Supermatch ya las publicó)
    lineas = lub_odds.fetch_lub() if actuales else []
    mu_override, totales, fuentes = {}, {}, {}
    for p in actuales:
        ms = int(datetime.fromisoformat(p.inicio_utc).timestamp() * 1000)
        lp = lub_odds.match_fixture(lineas, p.local, p.visitante, ms)
        mu_mod = rt.mu(p.local, p.visitante)
        mu_mkt = lp.mu_mercado(rt.sigma_pred) if lp else None
        if mu_mkt is not None:
            mu_override[p.evento_id] = MARKET_WEIGHT * mu_mkt + (1 - MARKET_WEIGHT) * mu_mod
            fuentes[p.evento_id] = f"mercado {mu_mkt:+.1f} / ratings {mu_mod:+.1f}"
        else:
            fuentes[p.evento_id] = f"ratings {mu_mod:+.1f} (sin cuota)"
        totales[p.evento_id] = (lp.total_esperado() if lp else None) or TOTAL_DEFAULT

    cfg = Config(n_sims=n_sims, n_rivales=n_rivales)
    so = simular(rt, slots_temporada(partidos), cfg, dict(jugados), mu_override)
    p_camp = np.bincount(so.campeon, minlength=len(so.equipos)) / n_sims

    pm_json = json.loads((DATA_DIR / "pool_model_37.json").read_text())
    pm = PoolModel.from_json(pm_json)
    shares = np.power(p_camp + 1e-4, CAMPEON_EXP)
    shares /= shares.sum()
    simular_rivales(so, cfg, pm.q, shares, perfiles=perfiles_de_json(pm_json), kappa=KAPPA_POOL)

    ids_actual = {p.evento_id for p in actuales}
    slots_idx = [j for j, s in enumerate(so.slots) if s.evento_id in ids_actual]
    ev = Evaluador(so, k, slots_idx)
    orden_camp = list(np.argsort(-p_camp))
    opts = [int(t) for t in orden_camp if p_camp[t] >= 0.01]
    especiales_libres = not any(p.temporada == TEMPORADA and datetime.fromisoformat(p.inicio_utc) <= now
                                for p in partidos) and campeon_fijo is None
    init = None
    if campeon_fijo:
        init = np.array([so.equipos.index(t) for t in campeon_fijo])
    port = optimizar(ev, opts, campeon_init=init, especiales_libres=especiales_libres)

    # fuera de muestra: mismo portfolio, sorteos nuevos (temporada + rivales + política)
    cfg2 = Config(n_sims=n_sims, n_rivales=n_rivales, seed=cfg.seed + 1000)
    so2 = simular(rt, slots_temporada(partidos), cfg2, dict(jugados), mu_override)
    simular_rivales(so2, cfg2, pm.q, shares, perfiles=perfiles_de_json(pm_json), kappa=KAPPA_POOL)
    idx2 = [j for j, s in enumerate(so2.slots) if s.evento_id in ids_actual]
    orden1 = [so.slots[j].evento_id for j in slots_idx]
    orden2 = [so2.slots[j].evento_id for j in idx2]
    perm = [orden2.index(e) for e in orden1]
    ev2 = Evaluador(so2, k, [idx2[i] for i in perm], seed=8)
    oos = evaluar(ev2, port.picks_actual, port.campeon, port.goleador)

    # planilla
    by_slot = {so.slots[j].evento_id: (i, j) for i, j in enumerate(slots_idx)}
    filas = []
    for p in actuales:
        i, j = by_slot[p.evento_id]
        probs = so.probs[:, j].mean(0)
        filas.append({
            "evento_id": p.evento_id, "local": p.local, "visitante": p.visitante,
            "cierre_utc": p.cierre_utc, "preferencial": p.preferencial, "fuente": fuentes[p.evento_id],
            "p_local": round(float(probs[:N_BANDAS].sum()), 3),
            "probs_clase": [round(float(x), 3) for x in probs],
            "picks": [list(marcador_de_clase(int(port.picks_actual[e, i]), totales[p.evento_id]))
                      for e in range(k)],
        })
    return {
        "generado_utc": now.isoformat(), "temporada": TEMPORADA, "k": k, "n_sims": n_sims,
        "n_rivales": n_rivales, "fecha": actuales[0].fecha_nombre if actuales else None,
        "partidos": filas,
        "campeon": [so.equipos[int(t)] for t in port.campeon],
        "especiales_libres": especiales_libres,
        "p_campeon": {so.equipos[t]: round(float(p_camp[t]), 3) for t in orden_camp},
        "e_premio": round(port.e_premio), "oos": {kk: round(v, 3) for kk, v in oos.items() if not kk.startswith("_")}, "_oos_tot": oos["_tot"], "detalle": {kk: round(v, 1) if isinstance(v, float) else v
                                                       for kk, v in port.detalle.items()},
        "costo": k * PRECIO,
        "ratings": {e: round(v, 1) for e, v in sorted(rt.r.items(), key=lambda kv: -kv[1])},
    }


def formatear(pl: dict) -> str:
    o = pl["oos"]
    L = [f"🏀 Penca LUB — {pl['fecha']} ({pl['k']} participaciones)",
         f"E[premio] ${o['e_premio']:,.0f} ± {o['se']:,.0f} (costo ${pl['costo']:,.0f}) · penca ${o['e_penca']:,.0f} "
         f"(P {o['p_penca']:.1%}) · fechas ${o['e_fechas']:,.0f}"]
    for f in pl["partidos"]:
        L.append(f"\n{'⭐ ' if f['preferencial'] else ''}{f['local']} vs {f['visitante']} "
                 f"(P local {f['p_local']:.0%}; {f['fuente']})")
        cnt = collections.Counter(tuple(x) for x in f["picks"])
        L.append("   " + " · ".join(f"{a}-{b} ×{n}" for (a, b), n in cnt.most_common()))
    if pl["especiales_libres"]:
        L.append("\nCampeón: " + ", ".join(f"{t} ×{n}" for t, n in collections.Counter(pl["campeon"]).most_common()))
    return "\n".join(L)


def guardar(pl: dict) -> Path:
    d = DATA_DIR / "planillas"
    d.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name[1:].split("_")[0]) for p in d.glob("v*_*.json")] or [0])
    path = d / f"v{n}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(pl, ensure_ascii=False, indent=1))
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=int(os.environ.get("LUB_K", 12)))
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--rivales", type=int, default=320)
    ap.add_argument("--sweep", help="lista de K para medir E[premio] vs participaciones")
    ap.add_argument("--no-refrescar", action="store_true")
    ap.add_argument("--no-guardar", action="store_true")
    ap.add_argument("--telegram", action="store_true")
    a = ap.parse_args()
    if a.sweep:
        for i, k in enumerate(int(x) for x in a.sweep.split(",")):
            pl = correr(k, a.sims, a.rivales, refrescar=(i == 0 and not a.no_refrescar))
            d = pl["oos"]
            net = pl["_oos_tot"] - pl["costo"]
            print(f"K={k:2d} DIST  P(perder plata) {(net < 0).mean():.1%}  P(ganar penca) {d['p_penca']:.1%} "
                  f"(entero {d['p_penca_entero']:.1%})  P(≥1 fecha) {d['p_alguna_fecha']:.1%}  "
                  f"fechas ganadas E {d['e_n_fechas']:.2f}  neto p10/p50/p90 "
                  f"{np.percentile(net, 10):+,.0f} / {np.percentile(net, 50):+,.0f} / {np.percentile(net, 90):+,.0f}", flush=True)
            print(f"K={k:2d}  E[premio] OOS ${d['e_premio']:>7,.0f} ± {d['se']:,.0f} (in-sample ${pl['e_premio']:,})  "
                  f"costo ${pl['costo']:>5,.0f}  neto ${d['e_premio'] - pl['costo']:>+7,.0f}  "
                  f"(penca ${d['e_penca']:,.0f}, P {d['p_penca']:.1%}; fechas ${d['e_fechas']:,.0f})", flush=True)
        return
    pl = correr(a.k, a.sims, a.rivales, refrescar=not a.no_refrescar)
    txt = formatear(pl)
    print(txt)
    pl.pop("_oos_tot", None)
    if not a.no_guardar:
        print("→", guardar(pl))
    if a.telegram:
        from src.notifier.telegram import TelegramNotifier
        TelegramNotifier().send(txt)


if __name__ == "__main__":
    main()

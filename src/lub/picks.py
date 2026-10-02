"""Pipeline de la penca LUB: datos → ratings → cuotas → temporada → pool → portfolio → planilla.

Uso:
    python -m src.lub.picks [--k 12] [--sims 4000] [--rivales 320] [--no-guardar] [--telegram]
    python -m src.lub.picks --ventana-h 24 --telegram   # timer diario: lo que cierra hoy
    python -m src.lub.picks --sweep 1,4,8,12,16,20     # E[premio] vs cantidad de participaciones

Timer diario (--ventana-h): optimiza la fecha ENTERA que sigue abierta (el premio de
fecha depende de todos sus partidos) pero la planilla marca "hoy" solo los que cierran
dentro de la ventana, que son los que hay que cargar. Con la ventana de 24h y una
corrida por día cada partido se carga exactamente una vez, la mañana en que cierra y
con la cuota más madura; los que cierran otro día salen provisorios y se re-optimizan
en su propia mañana. Si ese día no cierra nada, sale en silencio. Las fechas de la LUB
se solapan (en la 25/26 la F3 arrancó antes de que terminara la F2), así que un mismo
día puede traer partidos de dos fechas: se corre una optimización por fecha.

Participaciones: LUB_MIS_PARTICIPACIONES (lista ordenada del env) fija K y el orden
de las columnas; sin ella, K sale de --k / LUB_K y las columnas son P1..PK.

La planilla se versiona en data/lub/planillas/v{N}_{ts}.json (nunca se pisa) y se
imprime lista para cargar a mano: por participación, el marcador de cada partido y
los especiales.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import logging
import os
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.lub import odds as lub_odds
from src.lub.data import (CAMPEONATO_ID, DATA_DIR, TZ_UY, Partido, fetch_especiales_propios,
                          fetch_opciones_goleador, fetch_temporadas, load_temporadas,
                          mis_numeros_env, save_temporadas)
from src.lub.goleador import cargar_candidatos, simular_goleador
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
# Goleador uruguayo (25 pts): con data/lub/goleador_candidatos.json (src.lub.goleador,
# estadísticas de Genius) se simula CONJUNTO con la temporada y no hace falta el menú
# del API (que da 500 sin sesión). Sin ese archivo, P(goleador) es un prior ARMADO A
# MANO sobre el menú real: {nombre: prob}; sin ese archivo o sin menú, no se asigna. Shares del pool ∝ prior^1: de la 25/26
# solo sabemos que el 28% de los que cargaron fue al que ganó (Vescovi), sin dato de
# qué tan concentrado estaba el resto — no inventar un exponente.
GOLEADOR_PRIOR = DATA_DIR / "goleador_prior.json"
GOLEADOR_EXP = 1.0


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


def _abiertos(partidos: list[Partido], now: datetime) -> list[Partido]:
    return sorted([p for p in partidos if p.temporada == TEMPORADA and p.pts_local is None
                   and datetime.fromisoformat(p.cierre_utc) > now], key=lambda p: p.inicio_utc)


def fecha_actual(partidos: list[Partido], now: datetime, fecha: str | None = None) -> list[Partido]:
    """Partidos abiertos de `fecha` (default: la del próximo cierre)."""
    abiertos = _abiertos(partidos, now)
    if not abiertos:
        return []
    f = fecha or abiertos[0].fecha_nombre
    return [p for p in abiertos if p.fecha_nombre == f]


def fechas_que_cierran(partidos: list[Partido], now: datetime, ventana_h: float) -> list[str]:
    """Fechas con algún partido cuyo pronóstico cierra dentro de la ventana, en orden."""
    lim = now + timedelta(hours=ventana_h)
    out: list[str] = []
    for p in _abiertos(partidos, now):
        if datetime.fromisoformat(p.cierre_utc) <= lim and p.fecha_nombre not in out:
            out.append(p.fecha_nombre)
    return out


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", " ").split())


def cargar_goleador(opciones: list[str] | None,
                    path: Path | None = None) -> tuple[list[str], np.ndarray] | None:
    """(nombres del menú, P(goleador) sobre el menú) o None si falta el menú o el prior.

    El prior se matchea por nombre normalizado (sin tildes ni mayúsculas); lo que no
    matchea se loguea, porque un prior que no cubre al favorito real es peor que
    ninguno."""
    path = path or GOLEADOR_PRIOR
    if not opciones or not path.exists():
        return None
    prior = {_norm(n): float(v) for n, v in json.loads(path.read_text()).items()}
    p = np.array([prior.get(_norm(n), 0.0) for n in opciones])
    sueltos = set(prior) - {_norm(n) for n in opciones}
    if sueltos:
        log.warning("prior de goleador con nombres fuera del menú: %s", sorted(sueltos))
    if p.sum() <= 0:
        return None
    p = p + 1e-4
    return list(opciones), p / p.sum()


def _especiales_idx(nombres: list[str | None], universo: list[str], k: int) -> np.ndarray:
    """Índices en `universo` (−1 = no cargado / desconocido: nunca acierta)."""
    idx = {_norm(u): i for i, u in enumerate(universo)}
    out = [idx.get(_norm(n), -1) if n else -1 for n in nombres[:k]]
    return np.array(out + [-1] * (k - len(out)), int)


def correr(k: int, n_sims: int, n_rivales: int, refrescar: bool = True, now: datetime | None = None,
           campeon_fijo: list[str | None] | None = None, fecha: str | None = None,
           ventana_h: float | None = None, goleador: tuple[list[str], np.ndarray] | None = None,
           goleador_fijo: list[str | None] | None = None) -> dict:
    """Optimiza la fecha abierta `fecha` (default: la del próximo cierre).

    campeon_fijo / goleador_fijo: especiales ya cargados (desde que arranca la temporada
    no se pueden cambiar); entran a la valuación pero no se optimizan."""
    now = now or _now()
    if refrescar:
        save_temporadas(fetch_temporadas())
    partidos = load_temporadas()
    rt = fit(partidos, now, TEMPORADA, PARAMS_PROD)
    jugados = collections.Counter()
    for p in partidos:
        if p.temporada == TEMPORADA and p.pts_local is not None:
            jugados[p.local] += 1; jugados[p.visitante] += 1

    actuales = fecha_actual(partidos, now, fecha)
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
    # goleador: CONJUNTO con la temporada si hay candidatos con estadísticas
    # (src.lub.goleador: puntos totales = ppg × partidos que juega su equipo en ESTE
    # sorteo); si no, el prior a mano sobre el menú, independiente de la temporada.
    gol_nombres, gol_real, gol_real2, gol_shares = None, None, None, None
    cands = cargar_candidatos() if goleador is None else None
    if cands:
        gol_nombres = [c.nombre for c in cands]
        gol_real, _ = simular_goleador(so.partidos, so.equipos, cands, seed=cfg.seed + 77)
        p_gol = np.bincount(gol_real, minlength=len(cands)) / n_sims + 1e-5
        p_gol /= p_gol.sum()
    elif goleador is not None:
        gol_nombres, p_gol = goleador
        gol_real = np.random.default_rng(cfg.seed + 77).choice(len(p_gol), size=n_sims, p=p_gol)
        gol_real2 = np.random.default_rng(cfg.seed + 1077).choice(len(p_gol), size=n_sims, p=p_gol)
    if gol_nombres is not None:
        gol_shares = np.power(p_gol, GOLEADOR_EXP)
        gol_shares /= gol_shares.sum()
    simular_rivales(so, cfg, pm.q, shares, gol_real, gol_shares,
                    perfiles=perfiles_de_json(pm_json), kappa=KAPPA_POOL)

    ids_actual = {p.evento_id for p in actuales}
    slots_idx = [j for j, s in enumerate(so.slots) if s.evento_id in ids_actual]
    ev = Evaluador(so, k, slots_idx, goleador_real=gol_real, n_goleadores=len(gol_nombres or []))
    orden_camp = list(np.argsort(-p_camp))
    opts = [int(t) for t in orden_camp if p_camp[t] >= 0.01]
    especiales_libres = not any(p.temporada == TEMPORADA and datetime.fromisoformat(p.inicio_utc) <= now
                                for p in partidos) and campeon_fijo is None
    init = _especiales_idx(campeon_fijo, so.equipos, k) if campeon_fijo is not None else None
    gol_opts, gol_init = None, None
    if gol_nombres is not None:
        gol_opts = [int(g) for g in np.argsort(-p_gol) if p_gol[g] >= 0.01] or [int(p_gol.argmax())]
        if goleador_fijo is not None:
            gol_init = _especiales_idx(goleador_fijo, gol_nombres, k)
    port = optimizar(ev, opts, campeon_init=init, especiales_libres=especiales_libres,
                     goleador_opts=gol_opts, goleador_init=gol_init)

    # fuera de muestra: mismo portfolio, sorteos nuevos (temporada + rivales + política)
    cfg2 = Config(n_sims=n_sims, n_rivales=n_rivales, seed=cfg.seed + 1000)
    so2 = simular(rt, slots_temporada(partidos), cfg2, dict(jugados), mu_override)
    if cands:
        gol_real2, _ = simular_goleador(so2.partidos, so2.equipos, cands, seed=cfg2.seed + 77)
    simular_rivales(so2, cfg2, pm.q, shares, gol_real2, gol_shares,
                    perfiles=perfiles_de_json(pm_json), kappa=KAPPA_POOL)
    idx2 = [j for j, s in enumerate(so2.slots) if s.evento_id in ids_actual]
    orden1 = [so.slots[j].evento_id for j in slots_idx]
    orden2 = [so2.slots[j].evento_id for j in idx2]
    perm = [orden2.index(e) for e in orden1]
    ev2 = Evaluador(so2, k, [idx2[i] for i in perm], seed=8, goleador_real=gol_real2,
                    n_goleadores=len(gol_nombres or []))
    oos = evaluar(ev2, port.picks_actual, port.campeon, port.goleador)

    # planilla
    by_slot = {so.slots[j].evento_id: (i, j) for i, j in enumerate(slots_idx)}
    lim = now + timedelta(hours=ventana_h) if ventana_h is not None else None
    filas = []
    for p in actuales:
        i, j = by_slot[p.evento_id]
        probs = so.probs[:, j].mean(0)
        filas.append({
            "evento_id": p.evento_id, "fecha": p.fecha_nombre, "local": p.local, "visitante": p.visitante,
            "cierre_utc": p.cierre_utc, "preferencial": p.preferencial, "fuente": fuentes[p.evento_id],
            "hoy": lim is None or datetime.fromisoformat(p.cierre_utc) <= lim,
            "p_local": round(float(probs[:N_BANDAS].sum()), 3),
            "probs_clase": [round(float(x), 3) for x in probs],
            "picks": [list(marcador_de_clase(int(port.picks_actual[e, i]), totales[p.evento_id]))
                      for e in range(k)],
        })
    return {
        "generado_utc": now.isoformat(), "temporada": TEMPORADA, "k": k, "n_sims": n_sims,
        "n_rivales": n_rivales, "fecha": actuales[0].fecha_nombre if actuales else None,
        "partidos": filas,
        "campeon": [so.equipos[int(t)] if t >= 0 else None for t in port.campeon],
        "goleador": ([gol_nombres[int(g)] if g >= 0 else None for g in port.goleador]
                     if port.goleador is not None else None),
        "p_goleador": ({gol_nombres[g]: round(float(p_gol[g]), 3) for g in np.argsort(-p_gol)[:8]}
                       if gol_nombres is not None else None),
        "especiales_libres": especiales_libres,
        "p_campeon": {so.equipos[t]: round(float(p_camp[t]), 3) for t in orden_camp},
        "e_premio": round(port.e_premio), "oos": {kk: round(v, 3) for kk, v in oos.items() if not kk.startswith("_")}, "_oos_tot": oos["_tot"], "detalle": {kk: round(v, 1) if isinstance(v, float) else v
                                                       for kk, v in port.detalle.items()},
        "costo": k * PRECIO,
        "ratings": {e: round(v, 1) for e, v in sorted(rt.r.items(), key=lambda kv: -kv[1])},
    }


def combinar(pls: list[dict], numeros: list[int], ventana_h: float | None) -> dict:
    """Una planilla por corrida aunque el día traiga dos fechas: partidos concatenados
    y el resumen de cada optimización en `corridas`. Los especiales salen de la
    primera (son los mismos en todas: temporada, no fecha)."""
    base = dict(pls[0])
    base["partidos"] = [f for pl in pls for f in pl["partidos"]]
    base["fecha"] = " + ".join(pl["fecha"] for pl in pls)
    base["corridas"] = [{"fecha": pl["fecha"], "oos": pl["oos"], "e_premio": pl["e_premio"],
                         "detalle": pl["detalle"]} for pl in pls]
    base["numeros"] = numeros
    base["ventana_h"] = ventana_h
    base.pop("_oos_tot", None)
    return base


def etiquetas(pl: dict) -> list[str]:
    """Rótulo de cada columna: los 3 últimos dígitos del número (como el Clausura) o P1..PK."""
    nums = pl.get("numeros") or []
    return [f"…{nums[e] % 1000}" if e < len(nums) else f"P{e + 1}" for e in range(pl["k"])]


def _hora_uy(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(TZ_UY).strftime("%d/%m %H:%M")


def formatear(pl: dict) -> str:
    """Telegram (HTML): lo que hay que cargar HOY, participación por participación —
    la web se carga de a una participación, así que ese es el orden útil."""
    e = html.escape
    hoy = [f for f in pl["partidos"] if f.get("hoy", True)]
    L = [f"<b>🏀 Penca LUB — {e(pl['fecha'])}</b> ({pl['k']} participaciones)"]
    for c in pl.get("corridas") or [{"fecha": pl["fecha"], "oos": pl["oos"]}]:
        o = c["oos"]
        L.append(f"{e(c['fecha'])}: E[premio] ${o['e_premio']:,.0f} ± {o['se']:,.0f} · penca "
                 f"${o['e_penca']:,.0f} (P {o['p_penca']:.1%}) · fechas ${o['e_fechas']:,.0f}")
    L.append(f"Costo ${pl['costo']:,.0f}")
    if not hoy:
        return "\n".join(L + ["\nNada para cargar hoy."])
    L.append("\n<b>Cargar hoy:</b>")
    for n, f in enumerate(hoy, 1):
        L.append(f"{n}. {'⭐ ' if f['preferencial'] else ''}{e(f['local'])} vs {e(f['visitante'])} — "
                 f"cierra {_hora_uy(f['cierre_utc'])} (P local {f['p_local']:.0%}; {e(f['fuente'])})")
    L.append("")
    for col, lab in enumerate(etiquetas(pl)):
        picks = " · ".join(f"{f['picks'][col][0]}-{f['picks'][col][1]}" for f in hoy)
        L.append(f"<code>{lab:>5}</code> {picks}")
    provisorios = [f for f in pl["partidos"] if not f.get("hoy", True)]
    if provisorios:
        L.append(f"\n<i>{len(provisorios)} partido(s) de la fecha cierran otro día: salen en su propia planilla.</i>")
    if pl["especiales_libres"]:
        L.append("\n<b>Especiales (antes del primer partido):</b>")
        gol = pl.get("goleador") or [None] * pl["k"]
        for lab, camp, g in zip(etiquetas(pl), pl["campeon"], gol):
            L.append(f"<code>{lab:>5}</code> {e(camp or '—')}" + (f" · {e(g)}" if g else ""))
        if not pl.get("goleador"):
            L.append("<i>Goleador sin asignar: falta el menú del API o data/lub/goleador_prior.json.</i>")
    return "\n".join(L)


def guardar(pl: dict) -> Path:
    d = DATA_DIR / "planillas"
    d.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name[1:].split("_")[0]) for p in d.glob("v*_*.json")] or [0])
    path = d / f"v{n}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(pl, ensure_ascii=False, indent=1))
    return path


def ultima_planilla(solo_libres: bool = False) -> dict | None:
    """La planilla más nueva (por número de versión); con solo_libres, la última que
    todavía asignaba especiales — de ahí salen los que se cargaron si el API no los da."""
    d = DATA_DIR / "planillas"
    for p in sorted(d.glob("v*_*.json"), key=lambda p: -int(p.name[1:].split("_")[0])):
        pl = json.loads(p.read_text())
        if not solo_libres or pl.get("especiales_libres"):
            pl["_archivo"] = p.name
            return pl
    return None


def especiales_cargados(numeros: list[int]) -> tuple[list | None, list | None]:
    """(campeones, goleadores) ya cargados, para valuar desde que arranca la temporada.

    Primero el API (la verdad, pública desde el primer partido); si no, la última
    planilla que los asignaba — asumir que se cargó lo que dijo es la mejor apuesta y
    es lo mismo que haría el usuario."""
    if numeros:
        try:
            esp = fetch_especiales_propios(numeros)
        except Exception as ex:                      # el API caído no frena la planilla
            log.warning("no pude leer los especiales propios: %s", ex)
            esp = None
        if esp:
            return [c for c, _ in esp], [g for _, g in esp]
    pl = ultima_planilla(solo_libres=True)
    if pl:
        return pl.get("campeon"), pl.get("goleador")
    return None, None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None,
                    help="participaciones (default: cuántos números hay en LUB_MIS_PARTICIPACIONES, si no LUB_K o 12)")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--rivales", type=int, default=320)
    ap.add_argument("--sweep", help="lista de K para medir E[premio] vs participaciones")
    ap.add_argument("--ventana-h", type=float, default=None,
                    help="modo timer: solo las fechas con cierres dentro de estas horas; sin cierres, sale en silencio")
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

    numeros = mis_numeros_env()
    k = a.k or len(numeros) or int(os.environ.get("LUB_K", 12))
    if numeros and k != len(numeros):
        raise SystemExit(f"--k {k} no coincide con los {len(numeros)} números de LUB_MIS_PARTICIPACIONES")
    now = _now()
    if not a.no_refrescar:
        save_temporadas(fetch_temporadas())
    partidos = load_temporadas()
    if a.ventana_h is not None:
        fechas = fechas_que_cierran(partidos, now, a.ventana_h)
        if not fechas:
            log.info("no cierra ningún partido en las próximas %.0fh — nada que hacer", a.ventana_h)
            return
    else:
        abiertos = fecha_actual(partidos, now)
        if not abiertos:
            log.info("no hay partidos abiertos")
            return
        fechas = [abiertos[0].fecha_nombre]

    arranco = any(p.temporada == TEMPORADA and datetime.fromisoformat(p.inicio_utc) <= now for p in partidos)
    camp_fijo, gol_fijo = especiales_cargados(numeros) if arranco else (None, None)
    if arranco and camp_fijo is None:
        log.warning("temporada arrancada sin especiales conocidos: se valúa como si no acertaran")
        camp_fijo = [None] * k
    goleador = None
    try:
        goleador = cargar_goleador(fetch_opciones_goleador())
    except Exception as ex:
        log.warning("no pude leer el menú de goleador: %s", ex)

    pls = [correr(k, a.sims, a.rivales, refrescar=False, now=now, campeon_fijo=camp_fijo, fecha=f,
                  ventana_h=a.ventana_h, goleador=goleador, goleador_fijo=gol_fijo) for f in fechas]
    pl = combinar(pls, numeros, a.ventana_h)
    txt = formatear(pl)
    print(txt)
    if not a.no_guardar:
        print("→", guardar(pl))
    if a.telegram:
        from src.notifier.telegram import TelegramConfig, TelegramNotifier
        TelegramNotifier(TelegramConfig.from_env()).send(txt)


if __name__ == "__main__":
    main()

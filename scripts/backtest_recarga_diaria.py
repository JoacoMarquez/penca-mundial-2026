"""¿Cuánto vale cargar los partidos de cada día con la planilla de ESE día?

LA PREGUNTA. Una fecha del Clausura se juega viernes-domingo. Se puede cargar todo el
viernes de una, o cargar los partidos de cada día con la corrida de la mañana de ese
día. La segunda opción usa una planilla que YA SABE los resultados de los días
anteriores; la primera no. ¿Cuánta plata hay en esa diferencia?

POR QUÉ PODRÍA HABER ALGO. No son las cuotas: dos días de mercado uruguayo mueven poco
y el 70/30 con ratings lo amortigua más. El canal es OTRO — el objetivo del optimizador
es E[premio], no puntos, así que depende de DÓNDE ESTAMOS parados. Después del sábado
nuestra distancia al líder cambió, y con ella cuánto conviene diferenciarse el domingo:
si el sábado salió bien conviene parecerse más al pool (defender), si salió mal conviene
abrirse. Una planilla de domingo elegida el viernes está optimizada contra una posición
competitiva que a esa altura ya venció.

DISEÑO. Tres brazos, todos evaluados bajo la MISMA verdad y con sorteos comunes:

    A   — planilla del viernes: se optimiza sin saber nada del día 1.
    B0  — control de PULIDO: re-optimiza los picks del día 2 con warm start desde A,
          pero SIN saber el resultado del día 1 (mismas grillas que A).
    Bs  — planilla del día 2: re-optimiza los picks del día 2 con warm start desde A
          y con el día 1 YA RESUELTO (grillas degeneradas en el resultado sorteado).

El control B0 es lo que hace interpretable el número. Bs recibe pasadas extra de ascenso
igual que cualquier rerun, y ese pulido vale plata por sí solo (+$1.071 medidos el
2026-08-11, que fue lo que llevó max_passes a 6). Sin B0, todo ese pulido se le
atribuiría a la información del día 1. Entonces:

    valor de la INFORMACIÓN = Bs − B0      ← lo que responde la pregunta
    valor del PULIDO         = B0 − A      ← el mismo canal del rerun, ya conocido

Se sortea el día 1 S veces (escenarios). Para cada escenario se re-optimiza y se
evalúan los tres brazos bajo las grillas condicionadas a ESE resultado, con sorteos
comunes. El Δ que se reporta es el promedio sobre escenarios, y su SE sale de la
dispersión ENTRE escenarios — que es la incertidumbre que importa: no sabemos qué
sábado nos va a tocar.

Los picks del día 1 están congelados en los tres brazos (ya se cargaron), así que
dentro de un escenario los tres tienen exactamente los mismos puntos del día 1: la
única diferencia es cómo se juega el día 2 sabiendo eso.
"""
import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_AQUI = pathlib.Path(__file__).resolve().parent

from src.clausura.api import TZ_UY  # noqa: E402
from src.clausura.economics import MAX_GOALS, PrizeConfig, SimConfig  # noqa: E402
from src.clausura.pool import PoolConfig, pool_distribution  # noqa: E402
from src.clausura.strategy import EvaluadorPortfolio, build_portfolio  # noqa: E402


def q_verdad(pred_grids):
    """Pool calibrado con los 4.791 picks reales — la misma verdad del resto de los
    experimentos (scripts/bias_calibrado.json)."""
    d = json.load(open(_AQUI / "bias_calibrado.json"))
    c = PoolConfig(chalk_strength=d["chalk"], temperature=1.0,
                   default_bias=d["default_bias"],
                   popular_bias={tuple(map(int, k.split("-"))): v
                                 for k, v in d["bias"].items()})
    return [pool_distribution(g, c) for g in pred_grids]


def dia_uy(ev) -> str:
    return datetime.fromisoformat(ev["inicio_utc"]).astimezone(TZ_UY).date().isoformat()


def sortear_resultado(grid, rng) -> tuple[int, int]:
    """Un marcador sorteado de la grilla (la verdad del experimento)."""
    plano = np.asarray(grid, dtype=float).ravel()
    plano = plano / plano.sum()
    k = rng.choice(plano.size, p=plano)
    return int(k // (MAX_GOALS + 1)), int(k % (MAX_GOALS + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=19200)
    ap.add_argument("--escenarios", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--participaciones", type=int, default=12)
    ap.add_argument("--fecha", type=int, default=None,
                    help="fecha objetivo (default: la primera sin resultados)")
    ap.add_argument("--out", default=None, help="JSON con el detalle")
    a = ap.parse_args()

    from src.clausura.api import PencaApiClient
    from src.clausura.odds import fetch_primera_odds
    from src.clausura.picks import (
        build_season_grids, ensure_ratings, flat_eventos, load_config, match_odds,
    )

    cfg = load_config()
    eventos = flat_eventos(cfg)
    idx_of = {ev["evento_id"]: i for i, ev in enumerate(eventos)}

    resultados = {}
    with PencaApiClient() as api:
        for _, f in cfg["fechas"].items():
            for e in api._get(f"/front/campeonatos/fechas/{f['fecha_id']}/eventos"):
                r = e.get("resultado") or {}
                gl, gv = r.get("golesEquipoLocal"), r.get("golesEquipoVisitante")
                if gl is not None and gv is not None:
                    resultados[e["id"]] = (int(gl), int(gv))
    ratings = ensure_ratings()
    try:
        odds_by_evento = match_odds(eventos, fetch_primera_odds())
    except Exception as e:
        print(f"sin odds ({e})", flush=True)
        odds_by_evento = {}

    grids, _, pred_grids, _ = build_season_grids(
        eventos, ratings, odds_by_evento, resultados)
    fecha_de = [ev["fecha_n"] for ev in eventos]
    pref = [bool(ev["preferencial"]) for ev in eventos]
    prize = PrizeConfig()
    qs = [pool_distribution(g, PoolConfig()) for g in pred_grids]
    qv = q_verdad(pred_grids)

    # ---------- fecha objetivo y partición en días ----------
    # Fecha objetivo = la que se está jugando (la primera con partidos pendientes).
    # NO se exige que esté entera por jugar: los partidos ya resueltos de esa fecha
    # son parte del condicionamiento, igual que en la vida real. Lo que se necesita es
    # que queden AL MENOS DOS DÍAS pendientes — el anteúltimo se sortea (día 1) y el
    # último es el que se decide cargar esa mañana (día 2).
    #
    # OJO con el config: la AUF reprograma tarde y las fechas futuras vienen todas
    # con el mismo día por defecto (F3 en adelante figuran 8 partidos un viernes), así
    # que apuntar a una fecha "virgen" mide una estructura que no existe.
    por_fecha = defaultdict(list)
    for ev in eventos:
        por_fecha[ev["fecha_n"]].append(ev)
    # Se pide 2+ DÍAS pendientes, no 1+ partido: si no, la F1 gana la selección por el
    # Torque-Peñarol suspendido al 2/9, que es un partido suelto y no un fin de semana.
    def dias_pendientes(evs):
        return {dia_uy(ev) for ev in evs if ev["evento_id"] not in resultados}

    pendientes = sorted(n for n, evs in por_fecha.items()
                        if len(dias_pendientes(evs)) >= 2)
    target = a.fecha or (pendientes[0] if pendientes else None)
    if target is None:
        print("no hay fechas sin jugar")
        return
    del_target = [ev for ev in eventos if ev["fecha_n"] == target]
    # Solo los PENDIENTES se parten en días: los ya jugados quedan fijos en su
    # resultado real y entran al condicionamiento por el mismo camino que el día 1.
    por_dia = defaultdict(list)
    for ev in del_target:
        if ev["evento_id"] not in resultados:
            por_dia[dia_uy(ev)].append(ev)
    dias = sorted(por_dia)
    ya_jugados = [ev for ev in del_target if ev["evento_id"] in resultados]
    if len(dias) < 2:
        print(f"a la fecha {target} le queda un solo día pendiente — nada que medir. "
              f"Probá con --fecha N (necesita 2+ días por jugar).")
        return
    if ya_jugados:
        print(f"(la fecha {target} ya tiene {len(ya_jugados)} partido(s) jugado(s): "
              f"quedan fijos en su resultado real)", flush=True)
    # día 2 = el ÚLTIMO día (el que se cargaría esa mañana); día 1 = todo lo anterior
    dia2 = por_dia[dias[-1]]
    dia1 = [ev for d in dias[:-1] for ev in por_dia[d]]

    i_dia1 = [idx_of[ev["evento_id"]] for ev in dia1]
    i_dia2 = [idx_of[ev["evento_id"]] for ev in dia2]

    print(f"Fecha {target} · {len(del_target)} partidos en {len(dias)} días", flush=True)
    print(f"  día 1 ({', '.join(dias[:-1])}): {len(dia1)} partidos — se cargan igual "
          f"en los dos mundos", flush=True)
    print(f"  día 2 ({dias[-1]}): {len(dia2)} partidos — ACÁ está la decisión", flush=True)
    for ev in dia2:
        print(f"      · {ev['local']} vs {ev['visitante']}", flush=True)
    print(f"\n{a.escenarios} escenarios · {a.sims} sorteos · {a.seeds} semillas de "
          f"evaluación · {a.participaciones} participaciones\n", flush=True)

    # ---------- brazo A: la planilla del viernes ----------
    sim0 = SimConfig(n_sims=a.sims, n_rivales=718, seed=20260815)
    t0 = time.time()
    A = np.asarray(build_portfolio(
        grids=grids, fecha_de_partido=fecha_de, preferencial=pref,
        n_participaciones=a.participaciones, prize=prize, pool_qs=qs, sim=sim0).picks)
    print(f"A (planilla del viernes) lista ({time.time()-t0:.0f}s)", flush=True)

    # Solo se re-optimiza el día 2 de la fecha objetivo: todo lo demás queda como en A.
    # Así el contraste aísla la decisión que se está estudiando.
    mask = np.ones(len(eventos), dtype=bool)
    for i in i_dia2:
        mask[i] = False

    # ---------- brazo B0: control de pulido (sin saber el día 1) ----------
    t0 = time.time()
    B0 = np.asarray(build_portfolio(
        grids=grids, fecha_de_partido=fecha_de, preferencial=pref,
        n_participaciones=a.participaciones, prize=prize, pool_qs=qs,
        sim=SimConfig(n_sims=a.sims, n_rivales=718, seed=20260815 + 104729),
        frozen_picks=A, frozen_mask=mask, warm_start=A).picks)
    print(f"B0 (control de pulido) listo · {int((A != B0).sum())} picks distintos de A "
          f"({time.time()-t0:.0f}s)\n", flush=True)

    rng = np.random.default_rng(20260815)
    filas = []
    for s in range(a.escenarios):
        t0 = time.time()
        # 1) se sortea el día 1 y se lo vuelve determinístico en las grillas
        res_s = dict(resultados)
        for ev, i in zip(dia1, i_dia1):
            res_s[ev["evento_id"]] = sortear_resultado(pred_grids[i], rng)
        grids_s, _, pred_s, _ = build_season_grids(
            eventos, ratings, odds_by_evento, res_s)
        qs_s = [pool_distribution(g, PoolConfig()) for g in pred_s]
        qv_s = q_verdad(pred_s)

        # 2) la planilla del día 2, optimizada SABIENDO el día 1
        Bs = np.asarray(build_portfolio(
            grids=grids_s, fecha_de_partido=fecha_de, preferencial=pref,
            n_participaciones=a.participaciones, prize=prize, pool_qs=qs_s,
            sim=SimConfig(n_sims=a.sims, n_rivales=718, seed=20260815 + 7919 * (s + 1)),
            frozen_picks=A, frozen_mask=mask, warm_start=A).picks)

        # 3) los tres brazos, misma verdad condicionada, sorteos comunes
        ev_s = EvaluadorPortfolio(
            grids_s, fecha_de, pref, qv_s, prize,
            SimConfig(n_sims=a.sims, n_rivales=718, seed=20260815 + 31 * (s + 1)))
        info = ev_s.comparar(B0, Bs, n_seeds=a.seeds)     # información del día 1
        pulido = ev_s.comparar(A, B0, n_seeds=a.seeds)    # pasadas extra
        # El total NO se mide: sale exacto de los otros dos. `comparar` deriva sus
        # semillas de la config del evaluador, así que las tres comparaciones usarían
        # los MISMOS sorteos y (Bs−A) ≡ (Bs−B0) + (B0−A) sorteo a sorteo. Medirlo
        # aparte sería un tercio más de CPU para reproducir una identidad algebraica.
        total_delta = info.delta + pulido.delta

        marcador = " ".join(f"{r[0]}-{r[1]}" for r in
                            (res_s[ev["evento_id"]] for ev in dia1))
        filas.append({
            "escenario": s + 1, "dia1": marcador,
            "info": info.delta, "info_se": info.se,
            "pulido": pulido.delta, "pulido_se": pulido.se,
            "total": total_delta,
            "picks_movidos": int((B0 != Bs).sum()),
        })
        print(f"  esc {s+1:>2}: día1 [{marcador}] · info {info.delta:+8,.0f} ± "
              f"{info.se:>5,.0f} · pulido {pulido.delta:+8,.0f} · total "
              f"{total_delta:+8,.0f} · {filas[-1]['picks_movidos']:>2} picks "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---------- resumen ----------
    def resumen(clave):
        v = np.array([f[clave] for f in filas], dtype=float)
        # SE ENTRE escenarios: la incertidumbre relevante es qué sábado toca,
        # no el ruido Monte Carlo dentro de un escenario dado.
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), int((v > 0).sum()), len(v)

    print(f"\n{'='*78}\nRESULTADO ({len(filas)} escenarios)\n{'='*78}")
    for clave, etiqueta in (("info", "INFORMACIÓN del día 1 (Bs − B0)"),
                            ("pulido", "pulido de las pasadas extra (B0 − A)"),
                            ("total", "total realizado (Bs − A)")):
        m, se, pos, n = resumen(clave)
        veredicto = "SIGNIFICATIVO" if abs(m) > 2 * se else "nulo"
        print(f"  {etiqueta:<36} {m:+9,.0f} ± {se:>6,.0f}  "
              f"{pos}/{n} positivos  [{veredicto}]")
    m, se, _, _ = resumen("info")
    print(f"\n  mde_80 de la información = {2.84 * se:,.0f} "
          f"(piso de acción del gate: $2.000)")

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps({
            "fecha": target, "dias": dias, "n_dia1": len(dia1), "n_dia2": len(dia2),
            "sims": a.sims, "escenarios": a.escenarios, "seeds": a.seeds,
            "participaciones": a.participaciones, "filas": filas,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\ndetalle → {a.out}")


if __name__ == "__main__":
    main()

"""¿El piso de $2.000 del aviso por valor nos hace perder mejoras reales?

CÓMO SE LLEGÓ A ESTA PREGUNTA. El umbral tiene dos condiciones: Δ > 2·SE y Δ > $2.000.
Los dos únicos casos reales que pasaron por la compuerta (2026-08-10) midieron
−191 ± 106 y −456 ± 171: con SE ~150, la condición de ruido pide ~$300 y el piso pide
$2.000. **El piso muerde primero, por 6x.** O sea que la parte que sospechábamos
vencida por el cambio de sorteos ya es casi inerte — a 19.200 el optimizador quedó tan
estable que el test de ruido no filtra nada.

Entonces la pregunta no es "¿el umbral quedó alto?" sino "¿cuánta plata cae entre el
piso de ruido y los $2.000?". Eso depende de la DISTRIBUCIÓN del Δ verdadero en
situaciones de rerun reales, que es lo que mide este script.

DISEÑO. Se replica la situación de producción: planilla de la mañana, y reruns que
warm-startean desde ella con los mismos insumos (que es el caso dominante — las cuotas
casi no se mueven intradía en esta liga). Para cada par:

  * Δ MEDIDO: 5 semillas, que es lo que ve la compuerta en producción.
  * Δ VERDADERO: muchas más semillas, como referencia.

Con eso se compara qué realiza cada política de umbral: cuánta plata captura y cuántas
recargas manuales pide a cambio. Una recarga que no paga no es gratis — son 12 planillas
a mano, con su riesgo de tipeo.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_AQUI = pathlib.Path(__file__).resolve().parent

from src.clausura.economics import PrizeConfig, SimConfig  # noqa: E402
from src.clausura.pool import PoolConfig, pool_distribution  # noqa: E402
from src.clausura.rerun_cierre import UMBRAL_ABS, UMBRAL_SE  # noqa: E402
from src.clausura.strategy import EvaluadorPortfolio, build_portfolio  # noqa: E402

PISOS = (0.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0)


def q_verdad(pred_grids):
    d = json.load(open(_AQUI / "bias_calibrado.json"))
    c = PoolConfig(chalk_strength=d["chalk"], temperature=1.0,
                   default_bias=d["default_bias"],
                   popular_bias={tuple(map(int, k.split("-"))): v
                                 for k, v in d["bias"].items()})
    return [pool_distribution(g, c) for g in pred_grids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=19200)
    ap.add_argument("--pares", type=int, default=8)
    ap.add_argument("--seeds-medido", type=int, default=5)    # lo que ve producción
    ap.add_argument("--seeds-verdad", type=int, default=20)
    ap.add_argument("--participaciones", type=int, default=12)
    a = ap.parse_args()

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

    print(f"{len(eventos)} eventos · {a.pares} pares de rerun · {a.sims} sorteos\n"
          f"umbral vigente: Δ > {UMBRAL_SE}·SE y Δ > ${UMBRAL_ABS:,.0f}\n", flush=True)

    sim0 = SimConfig(n_sims=a.sims, n_rivales=718, seed=20260812)
    manana = np.asarray(build_portfolio(
        grids=grids, fecha_de_partido=fecha_de, preferencial=pref,
        n_participaciones=a.participaciones, prize=prize, pool_qs=qs, sim=sim0).picks)

    filas = []
    for k in range(a.pares):
        semilla = 20260812 + 7919 * (k + 1)
        sim = SimConfig(n_sims=a.sims, n_rivales=718, seed=semilla)
        t0 = time.time()
        # el rerun warm-startea desde la planilla de la mañana, mismos insumos
        rerun = np.asarray(build_portfolio(
            grids=grids, fecha_de_partido=fecha_de, preferencial=pref,
            n_participaciones=a.participaciones, prize=prize, pool_qs=qs, sim=sim,
            warm_start=manana).picks)
        distintos = int((manana != rerun).sum())
        ev = EvaluadorPortfolio(grids, fecha_de, pref, qv, prize, sim)
        med = ev.comparar(manana, rerun, n_seeds=a.seeds_medido)
        ver = ev.comparar(manana, rerun, n_seeds=a.seeds_verdad)
        filas.append({"picks": distintos, "med": med.delta, "se": med.se,
                      "verdad": ver.delta, "se_v": ver.se})
        print(f"  par {k+1}: {distintos:>3} picks distintos · "
              f"medido {med.delta:+8,.0f} ± {med.se:>5,.0f} · "
              f"verdad {ver.delta:+8,.0f} ± {ver.se:>5,.0f}  ({time.time()-t0:.0f}s)",
              flush=True)

    print(f"\n{'='*78}\n¿QUÉ REALIZA CADA PISO?  (n={len(filas)} reruns)\n{'='*78}")
    print(f"  {'piso':>8} {'avisa':>7} {'plata realizada':>18} {'por recarga pedida':>20}")
    for piso in PISOS:
        avisa = [f for f in filas
                 if f["med"] > piso and f["med"] > UMBRAL_SE * max(f["se"], 1e-9)]
        realizado = sum(f["verdad"] for f in avisa)
        por = realizado / len(avisa) if avisa else 0.0
        marca = "  ← vigente" if piso == UMBRAL_ABS else ""
        print(f"  ${piso:>7,.0f} {len(avisa):>4}/{len(filas):<3} {realizado:>+17,.0f}"
              f" {por:>+19,.0f}{marca}")

    v = np.array([f["verdad"] for f in filas])
    print(f"\n  Δ verdadero: media {v.mean():+,.0f} · mediana {np.median(v):+,.0f} · "
          f"máx {v.max():+,.0f} · positivos {int((v > 0).sum())}/{len(v)}")
    print(f"  SE del Δ medido con {a.seeds_medido} semillas: "
          f"{np.mean([f['se'] for f in filas]):,.0f} de promedio "
          f"→ la condición de ruido pide ~${UMBRAL_SE * np.mean([f['se'] for f in filas]):,.0f}")


if __name__ == "__main__":
    main()

"""¿El menú tiene que cambiar cuando vamos ABAJO en la tabla? Pre-decidirlo hoy.

EL HUECO (auditoría 13/8, eje metodología). Todas las decisiones de menú —K_EV=5,
K_COBERTURA=False, K_HUECO=0— se midieron con el standing actual (déficit ~0), y
`medida_con` no registra el standing como supuesto. La respuesta de varianza al
déficit EMERGE del E[premio] (el simulador ancla los puntos reales), pero el menú
solo ofrece marcadores pegados al modo: si a −25 conviene cobertura, el optimizador
no tiene con qué expresarlo y ninguna alarma va a sonar.

La alternativa a este experimento es improvisar la regla en la fecha 10, mirando la
tabla con 40 puntos de bronca encima. Mejor decidirla hoy con la cabeza fría.

## Método

La perilla `SimConfig.handicap_propio` le resta Δ puntos al total de temporada de
nuestras 12 participaciones (solo el premio grande; los premios por fecha no se
tocan — el déficit modela puntos ya perdidos en fechas liquidadas). Para cada
Δ ∈ {0, 25, 40}:

    brazo A: menú de producción (5,0)          optimizado BAJO el déficit
    brazo B: (5,0) + candidatos de cobertura   optimizado BAJO el déficit
    Δ E[premio] = B − A, sorteos comunes, semilla de evaluación independiente,
                  bajo las DOS verdades del pool de siempre (control + calibrada)

El punto Δ=0 REPLICA la medición existente de cobertura (+153 ± 499 / +450 ± 732,
config/decisiones.yaml) — es el ancla que valida el harness. La pregunta es si el
signo o el tamaño cambian con Δ ≥ 25.

Métrica secundaria: el perfil de diversificación del brazo A solo (picks de
visitante y de empate en partidos abiertos, por déficit). Si el optimizador ya se
diversifica solo dentro del menú (5,0) cuando va perdiendo, la regla condicional
sobra; si NO cambia nada aunque pierda, el menú lo tiene maniatado y la cobertura
es la única válvula.

## Qué sale de acá

Una REGLA PRE-REGISTRADA en config/decisiones.yaml, del estilo "si el mejor de
nuestras participaciones está a X o más del líder, prender K_COBERTURA" — o el
rechazo documentado de que ni con −40 hace falta.

Mismo entorno que scripts/backtest_cobertura.py (pool modelado, sin RivalModel):
comparable con la medición de cobertura existente, que es el punto de referencia.

Uso:
    python -m scripts.backtest_deficit                      # Δ ∈ {0,25,40}, 2 reps
    python -m scripts.backtest_deficit --deficits 0 40 --reps 1 --sims 4800   # rápido
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.clausura import strategy  # noqa: E402
from src.clausura.economics import PrizeConfig, SimConfig, index_score  # noqa: E402
from src.clausura.pool import PoolConfig, pool_distribution  # noqa: E402
from src.clausura.strategy import EvaluadorPortfolio, build_portfolio  # noqa: E402
from scripts.backtest_cobertura import cargar_entorno, q_verdad  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "experimentos" / "deficit_menu.json"


def lado_de(idx: int) -> int:
    gl, gv = index_score(int(idx))
    return (gl > gv) - (gl < gv)


def perfil(picks: np.ndarray, eventos, resultados) -> dict:
    """Cuánta varianza expresa el portfolio en los partidos ABIERTOS."""
    abiertos = [m for m, ev in enumerate(eventos) if ev["evento_id"] not in resultados]
    lados = np.array([[lado_de(picks[k, m]) for m in abiertos]
                      for k in range(picks.shape[0])])
    n = picks.shape[0] * len(abiertos)
    return {
        "picks_visitante": int((lados == -1).sum()),
        "picks_empate": int((lados == 0).sum()),
        "celdas": n,
        "partidos_sin_visitante": int(sum((lados[:, j] != -1).all()
                                          for j in range(len(abiertos)))),
        "partidos_abiertos": len(abiertos),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deficits", type=int, nargs="+", default=[0, 25, 40])
    ap.add_argument("--sims", type=int, default=19200)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--eval-seeds", type=int, default=6)
    ap.add_argument("--participaciones", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true", help="no persistir el resultado")
    a = ap.parse_args()

    eventos, resultados, grids, pred_grids = cargar_entorno()
    fecha_de = [ev["fecha_n"] for ev in eventos]
    pref = [bool(ev["preferencial"]) for ev in eventos]
    prize = PrizeConfig()
    creencia = [pool_distribution(g, PoolConfig()) for g in pred_grids]
    verdades = {"control": creencia, "calibrada": q_verdad(pred_grids)}

    def planilla(cob: bool, sim: SimConfig) -> np.ndarray:
        strategy.K_COBERTURA = cob
        try:
            return np.asarray(build_portfolio(
                grids=grids, fecha_de_partido=fecha_de, preferencial=pref,
                n_participaciones=a.participaciones, prize=prize,
                pool_qs=creencia, sim=sim).picks)
        finally:
            strategy.K_COBERTURA = False

    resultados_exp: dict[int, dict] = {}
    for deficit in a.deficits:
        acum = {k: [] for k in verdades}
        perfiles = []
        for rep in range(a.reps):
            sim = SimConfig(n_sims=a.sims, n_rivales=718,
                            seed=20260813 + 7919 * rep, handicap_propio=deficit)
            t0 = time.time()
            base = planilla(False, sim)
            con = planilla(True, sim)
            perfiles.append(perfil(base, eventos, resultados))
            fila = []
            for k, q in verdades.items():
                # La evaluación hereda el MISMO handicap: la pregunta es qué
                # conviene jugar EN ese mundo, no en el nuestro.
                ev = EvaluadorPortfolio(grids, fecha_de, pref, q, prize, sim)
                c = ev.comparar(base, con, n_seeds=a.eval_seeds)
                acum[k].append(c.delta)
                fila.append(f"{k} {c.delta:+,.0f} ± {c.se:,.0f}")
            print(f"Δ=−{deficit:<3} rep {rep + 1} ({time.time() - t0:.0f}s)   "
                  + "   ·   ".join(fila), flush=True)

        resultados_exp[deficit] = {
            "delta_cobertura": {
                k: {"media": float(np.mean(xs)),
                    "se": float(np.std(xs, ddof=1) / np.sqrt(len(xs)))
                          if len(xs) > 1 else 0.0,
                    "reps": xs}
                for k, xs in acum.items()},
            "perfil_base": perfiles,
        }

    print(f"\n{'=' * 72}")
    print("Δ E[premio] de PRENDER cobertura, según cuántos puntos vamos abajo")
    print(f"{'=' * 72}")
    print(f"{'déficit':>8} {'verdad control':>22} {'verdad calibrada':>22}")
    for deficit in a.deficits:
        r = resultados_exp[deficit]["delta_cobertura"]
        celdas = [f"{r[k]['media']:+10,.0f} ± {r[k]['se']:8,.0f}"
                  for k in ("control", "calibrada")]
        print(f"{'−' + str(deficit):>8} {celdas[0]:>22} {celdas[1]:>22}")

    print(f"\nperfil del brazo BASE (5,0) — ¿el optimizador se diversifica solo al ir perdiendo?")
    print(f"{'déficit':>8} {'picks visitante':>16} {'picks empate':>14} "
          f"{'partidos sin V':>15}")
    for deficit in a.deficits:
        ps = resultados_exp[deficit]["perfil_base"]
        v = np.mean([p["picks_visitante"] for p in ps])
        e = np.mean([p["picks_empate"] for p in ps])
        sv = np.mean([p["partidos_sin_visitante"] for p in ps])
        tot = ps[0]["celdas"]
        print(f"{'−' + str(deficit):>8} {v:>10.1f}/{tot:<5} {e:>10.1f}/{tot:<3} "
              f"{sv:>8.1f}/{ps[0]['partidos_abiertos']}")

    if not a.dry_run:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps({
            "medido_utc": datetime.now(timezone.utc).isoformat(),
            "sims": a.sims, "reps": a.reps, "eval_seeds": a.eval_seeds,
            "participaciones": a.participaciones,
            "por_deficit": {str(k): v for k, v in resultados_exp.items()},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nguardado: {OUT_PATH}")


if __name__ == "__main__":
    main()

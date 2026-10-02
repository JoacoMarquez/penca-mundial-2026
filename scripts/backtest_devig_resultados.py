"""Backtest en PUNTOS del de-vig del 1X2 contra los partidos jugados del Clausura 2026.

`scripts/calibracion_supermatch.py` (2/10) mostró que el de-vig proporcional de
producción achata al favorito ~1,8 pp contra Pinnacle y que Shin lo corrige; contra
57 resultados reales el signo coincide pero el log-loss no separa. Esto responde la
pregunta de la penca: con cada método, ¿cuántos puntos habría hecho el pick?

Por partido (último snapshot de cuotas antes del kickoff, el mismo insumo que tuvo
producción) y por método (proportional / shin / power):

  * la grilla igual que producción: λ = 0,7·mercado + 0,3·ratings, λ12 del mercado;
    y también solo mercado (sin ratings), porque los ratings de hoy ya vieron los
    resultados del Clausura (fuga igual en los dos brazos, pero diluye el contraste);
  * el pick EV (argmax E[pts], ×2 si es preferencial) y sus puntos REALES;
  * el menú de los 5 mejores picks por E[pts] (K_EV=5, lo que el optimizador ofrece
    a las 12 participaciones) y sus puntos reales sumados;
  * P(resultado real) bajo la grilla (log-loss del marcador).

Diferencias pareadas contra proportional, con SE entre partidos. NO mide E[premio]:
eso es el A/B del 13/8 (scripts/backtest_devig_1x2.py), que depende de qué se tome
como verdad. Esto usa la única verdad sin supuestos: lo que pasó.

    ssh root@159.203.66.24 'cd /opt/penca && PYTHONPATH=. .venv/bin/python scripts/backtest_devig_resultados.py'
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibracion_supermatch import cargar_clausura  # noqa: E402

from src.clausura.economics import MAX_GOALS  # noqa: E402
from src.clausura.picks import MARKET_WEIGHT, ensure_ratings, market_lambdas  # noqa: E402
from src.clausura.scoring import rank_picks, supermatch_points  # noqa: E402
from src.model.poisson import score_grid  # noqa: E402

METODOS = ("proportional", "shin", "power")
K_MENU = 5


def grilla(o, lam_rt, metodo: str, w: float) -> np.ndarray:
    os.environ["CLAUSURA_DEVIG_1X2"] = metodo
    if metodo == "power":
        # market_lambdas solo despacha proportional/shin: power se inyecta acá
        from calibracion_supermatch import devig_power
        from src.model.poisson import MarketConstraints, fit_params
        p = devig_power(o.x1x2)
        o25 = devig_power(o.totals["2.5"]).get("over") if "2.5" in o.totals else None
        lm = fit_params(MarketConstraints(p_home_win=p["home"], p_draw=p["draw"],
                                          p_away_win=p["away"], p_over_2_5=o25))
    else:
        lm = market_lambdas(o)
    lam_l = w * lm[0] + (1 - w) * lam_rt[0]
    lam_v = w * lm[1] + (1 - w) * lam_rt[1]
    lam12 = min(max(w * lm[2], 0.0), lam_l, lam_v)
    return score_grid(lam_l, lam_v, lam12, max_goals=MAX_GOALS)


def se(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    pares = cargar_clausura()
    ratings = ensure_ratings()

    for nombre, w in (("producción (70% mercado + 30% ratings)", MARKET_WEIGHT),
                      ("solo mercado", 1.0)):
        res = {m: {"pick": [], "menu": [], "ll": [], "ev": [], "elegido": []} for m in METODOS}
        for p, o in pares:
            lam_rt = ratings.lambdas(p.local, p.visitante)
            real = (p.goles_local, p.goles_visitante)
            for m in METODOS:
                g = grilla(o, lam_rt, m, w)
                top = rank_picks(g, p.preferencial, top_k=K_MENU)
                res[m]["elegido"].append(top[0][0])
                res[m]["ev"].append(top[0][1])
                res[m]["pick"].append(supermatch_points(top[0][0], real, p.preferencial))
                res[m]["menu"].append(sum(supermatch_points(s, real, p.preferencial) for s, _ in top))
                gl, gv = min(real[0], MAX_GOALS), min(real[1], MAX_GOALS)
                res[m]["ll"].append(-math.log(max(g[gl, gv], 1e-12)))

        n = len(pares)
        print(f"\n=== {nombre} — {n} partidos ===")
        print(f"{'método':13}{'pts pick EV':>13}{'pts menú top-5':>16}{'log-loss marcador':>19}")
        for m in METODOS:
            r = res[m]
            print(f"{m:13}{sum(r['pick']):9d} pts{sum(r['menu']):12d} pts{np.mean(r['ll']):15.4f}")
        base = res["proportional"]
        for m in ("shin", "power"):
            r = res[m]
            cambia = sum(a != b for a, b in zip(r["elegido"], base["elegido"]))
            dp = np.array(r["pick"]) - np.array(base["pick"])
            dm = np.array(r["menu"]) - np.array(base["menu"])
            dl = np.array(r["ll"]) - np.array(base["ll"])
            print(f"  {m} − proportional: pick EV cambia en {cambia}/{n} partidos · "
                  f"Δ pts pick {dp.sum():+d} (±{se(dp)*n:.1f}) · "
                  f"Δ pts menú {dm.sum():+d} (±{se(dm)*n:.1f}) · "
                  f"Δ log-loss {dl.mean():+.4f} ± {se(dl):.4f}")
            ejemplos = [(pp.local, pp.visitante, b, a, pp.goles_local, pp.goles_visitante)
                        for (pp, _), a, b in zip(pares, r["elegido"], base["elegido"]) if a != b]
            for loc, vis, b, a, gl, gv in ejemplos[:12]:
                print(f"     {loc} vs {vis}: {b[0]}-{b[1]} → {a[0]}-{a[1]}  (salió {gl}-{gv})")


if __name__ == "__main__":
    main()

"""¿Cambiar el de-vig del 1X2 de proportional a Shin mueve PLATA?

CONTEXTO. El 13/8, midiendo contra Pinnacle desde el droplet
(`scripts/odds_sharp_vs_supermatch.py`, 8 partidos de la Fecha 2), quedó claro que
con el de-vig de producción nuestras probabilidades del 1X2 subvalúan al favorito:

    proportional:  favorito −0.0149 ± 0.0042  (t=−3.53, 7/8 partidos)
    shin:          favorito −0.0042 ± 0.0037  (t=−1.15)

Es el sesgo favorito-longshot, y con 14,7% de overround (vs 6,3% de Pinnacle) lo
produce el reparto a prorrata del margen, no la casa. Shin —pensado para vig alto—
recupera casi exactamente al sharp.

Pero eso mide el INSUMO. Este proyecto decide con Δ E[premio] pareado, y ya mató
tres veces a un "número grande" que no sobrevivía esa métrica. Un contraste previo
mostró que el método cambia el marcador modal en apenas 1 de 8 partidos, así que el
efecto en plata puede ser perfectamente cero.

## El diseño, y por qué NO es el obvio

Lo obvio sería optimizar con cada método y comparar cada portfolio bajo su propio
modelo. Eso compara MUNDOS, no decisiones: el brazo Shin gana bajo grillas Shin por
construcción, y el número no dice nada sobre qué conviene jugar.

Acá hay algo que en los experimentos anteriores no había: un **instrumento externo
mejor**. Pinnacle es el sharp de referencia —es literalmente contra quien
`src/valuebet/` capturaba +EV— así que sus probabilidades son la mejor estimación
disponible de la verdad. Entonces:

    verdad     = grillas armadas con Pinnacle (1X2 + over 2.5, de-vig Shin)
    brazo A    = portfolio optimizado con Supermatch + proportional  (producción)
    brazo B    = portfolio optimizado con Supermatch + shin
    Δ          = E[premio](B) − E[premio](A), los dos liquidados bajo la MISMA
                 verdad y con los MISMOS sorteos (common random numbers)

Δ > 0 y significativo ⇒ jugar con Shin nos hace ganar más plata contra la realidad.
Δ ≈ 0 ⇒ el sesgo del insumo es real pero no cambia decisiones, y el default se queda
donde está (que es el resultado más probable a priori, dado el 1/8).

Todo lo demás —snapshot del pool, ratings, rivales, premios, especiales, semillas—
es idéntico entre brazos: la ÚNICA diferencia es el de-vig del 1X2.

## Limitaciones que hay que leer junto con el resultado

  * Pinnacle solo cubre los 8 partidos de la fecha próxima. Los otros 112 usan las
    mismas grillas en los dos brazos, así que no sesgan el Δ, pero sí lo DILUYEN:
    la diferencia entre brazos vive en 8 de 120 partidos.
  * Es UNA fecha. El sesgo del insumo necesita 3-4 fechas para confirmarse
    (`odds_sharp_vs_supermatch` se niega a concluir con menos) y esto hereda esa
    limitación.
  * Asume que Pinnacle es la verdad. Es la mejor referencia que tenemos, no la
    verdad — pero el error de esa suposición afecta a los dos brazos por igual.

## Dónde corre

En el droplet de NYC: necesita Pinnacle. Dos optimizaciones completas a 19.200
sorteos (~11 min cada una) más la evaluación.

    ssh root@VPS 'cd /opt/penca && .venv/bin/python -m scripts.backtest_devig_1x2'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "experimentos"

EVAL_SEEDS = 8            # más que los 5 del gate: acá el Δ esperado es chico


def grids_pinnacle(contexto: dict, pin_eventos: list[dict],
                   metodo_verdad: str = "shin") -> tuple[list, list[int]]:
    """(grillas de verdad, evento_ids reemplazados).

    Parte de las grillas de la corrida —que ya tienen delta en lo jugado y ratings
    en los 112 sin mercado— y reemplaza SOLO los partidos que matchean con Pinnacle
    y todavía no se jugaron. Así la verdad difiere del modelo únicamente donde
    Pinnacle tiene algo que decir.
    """
    from src.model.market_probs import devig
    from src.model.poisson import MarketConstraints, fit_params, score_grid
    from src.clausura.economics import MAX_GOALS
    from scripts.odds_sharp_vs_supermatch import emparejar, supermatch_eventos

    eventos = contexto["eventos"]
    idx_of = contexto["idx_of"]
    grids = [np.array(g, copy=True) for g in contexto["grids"]]

    # Se matchea Supermatch↔Pinnacle (no config↔Pinnacle) para reusar el matcher ya
    # blindado, y de ahí se llega al evento_id por el nombre del par.
    sm = supermatch_eventos()
    por_nombre = {f'{e["home"]} vs {e["away"]}': e for e in sm}
    ev_por_nombre = {}
    for ev in eventos:
        ev_por_nombre[f'{ev["local"]} vs {ev["visitante"]}'] = ev

    reemplazados = []
    for sm_ev, pin_ev in emparejar(sm, pin_eventos):
        clave = f'{sm_ev["home"]} vs {sm_ev["away"]}'
        ev = ev_por_nombre.get(clave)
        if ev is None:
            # el nombre de Supermatch no siempre es el del config del penca-api
            ev = next((e for e in eventos
                       if _mismo(e["local"], sm_ev["home"]) and
                          _mismo(e["visitante"], sm_ev["away"])), None)
        if ev is None:
            log.warning("no ubiqué en el fixture: %s", clave)
            continue
        col = idx_of.get(ev["evento_id"])
        if col is None:
            continue

        p = devig(pin_ev["x1x2"], metodo_verdad)
        o25 = None
        if pin_ev.get("totals", {}).get("2.5"):
            o25 = devig(pin_ev["totals"]["2.5"], metodo_verdad).get("over")
        lam = fit_params(MarketConstraints(
            p_home_win=p["home"], p_draw=p["draw"], p_away_win=p["away"],
            p_over_2_5=o25))
        grids[col] = score_grid(lam[0], lam[1], lam[2], max_goals=MAX_GOALS)
        reemplazados.append(ev["evento_id"])

    return grids, reemplazados


def _mismo(a: str, b: str) -> bool:
    from scripts.odds_sharp_vs_supermatch import _similar
    return _similar(a, b) >= 1.0


def correr_brazo(metodo: str, fecha: int, n_part: int, n_sims: int) -> dict:
    """Corrida COMPLETA del pipeline con un método de de-vig. Devuelve el contexto.

    `guardar=False`: es un experimento, no puede versionar una planilla ni volverse
    el warm start de la próxima corrida.
    """
    from src.clausura.picks import run as picks_run

    os.environ["CLAUSURA_DEVIG_1X2"] = metodo
    contexto: dict = {}
    log.info("=== brazo %s: optimizando (fecha %d, %d sorteos) ===", metodo, fecha, n_sims)
    picks_run(fecha, n_part, telegram=False, n_sims=n_sims, contexto=contexto,
              usar_warm_start=False, guardar=False)
    return contexto


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", type=int, default=None)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--participaciones", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="no persistir el resultado")
    args = ap.parse_args()

    from src.clausura.picks import DEFAULT_SIMS, resolve_fecha
    from src.clausura.rivals import mis_numeros_env
    from scripts.odds_sharp_vs_supermatch import pinnacle_eventos

    fecha = args.fecha or resolve_fecha("auto")
    n_sims = args.sims or DEFAULT_SIMS
    n_part = args.participaciones or (len(mis_numeros_env()) or 5)

    # Pinnacle PRIMERO: si no responde, no tiene sentido gastar 25 min de CPU.
    log.info("bajando Pinnacle (la verdad de la evaluación)…")
    pin = pinnacle_eventos()
    if not pin:
        print("Sin datos de Pinnacle — el experimento necesita la referencia sharp.")
        raise SystemExit(1)
    log.info("Pinnacle: %d partidos", len(pin))

    ctx_a = correr_brazo("proportional", fecha, n_part, n_sims)
    ctx_b = correr_brazo("shin", fecha, n_part, n_sims)

    picks_a = np.asarray(ctx_a["portfolio"].picks, dtype=np.int64)
    picks_b = np.asarray(ctx_b["portfolio"].picks, dtype=np.int64)
    distintos = int((picks_a != picks_b).sum())

    # Se evalúa bajo DOS verdades: Pinnacle de-vigueado con shin y con proportional.
    #
    # Es la sensibilidad que decide si el resultado vale. La verdad "shin" comparte
    # método con el brazo B, y si Shin corre las probabilidades hacia el favorito en
    # CUALQUIER libro, el brazo B se parecería a la verdad por construcción y no por
    # estar mejor calibrado. Medido el 13/8: sobre Pinnacle (6,3% de vig) los dos
    # métodos difieren +0,51 pp en el favorito — un tercio del sesgo que se está
    # midiendo, así que la duda es real y hay que resolverla, no razonarla.
    #
    # Si Shin gana bajo LAS DOS verdades, el resultado es robusto. Si solo gana bajo
    # la suya, lo que se midió es el método, no la calibración.
    resultados = {}
    for metodo_verdad in ("shin", "proportional"):
        verdad, reemplazados = grids_pinnacle(ctx_a, pin, metodo_verdad)
        if not reemplazados:
            print("Pinnacle no matcheó ningún partido del fixture — sin verdad que usar.")
            raise SystemExit(1)
        # El evaluador del brazo A, pero liquidando contra la verdad de Pinnacle.
        # Pool, rivales, premios y especiales son idénticos entre brazos, así que la
        # única diferencia entre las dos liquidaciones son los picks.
        ev = ctx_a["evaluador"].con_grids(verdad)
        resultados[metodo_verdad] = ev.comparar(picks_a, picks_b, n_seeds=EVAL_SEEDS)

    comp = resultados["shin"]
    robusto = all(r.delta > 2 * r.se for r in resultados.values())
    signif = comp.delta > 2 * comp.se or comp.delta < -2 * comp.se
    lineas = [f"\n=== de-vig del 1X2: proportional (A) vs shin (B) ===",
              f"fecha {fecha} · {n_sims} sorteos · {n_part} participaciones",
              f"picks distintos entre brazos: {distintos}/{picks_a.size}", ""]
    for metodo_verdad, r in resultados.items():
        marca = "✓" if r.delta > 2 * r.se else ("✗" if r.delta < -2 * r.se else "~")
        lineas.append(
            f"  verdad = Pinnacle/{metodo_verdad:<12} "
            f"A ${r.valor_a:,.0f} · B ${r.valor_b:,.0f} · "
            f"Δ ${r.delta:+,.0f} ± {r.se:,.0f}  [{marca}]")
    lineas.append("")
    if robusto:
        lineas.append("⚠️  Shin gana bajo LAS DOS verdades ⇒ el resultado es ROBUSTO: "
                      "vale cambiar el default y registrarlo.")
    elif signif and comp.delta > 0:
        lineas.append("⚠️  Shin gana SOLO bajo la verdad que comparte método con él. "
                      "Eso es el método, no la calibración:\n"
                      "    NO alcanza para cambiar el default. Lo que decide es "
                      "calibrar contra RESULTADOS reales\n"
                      "    (odds versionadas + varias fechas), no contra otro de-vig.")
    else:
        lineas.append("Sin efecto distinguible del ruido: el sesgo del insumo es real "
                      "pero NO cambia decisiones.\n  El default (proportional) se "
                      "queda; registrar el rechazo.")
    print("\n".join(lineas))

    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / "devig_1x2.json"
        path.write_text(json.dumps({
            "medido_utc": datetime.now(timezone.utc).isoformat(),
            "fecha": fecha, "n_sims": n_sims, "n_participaciones": n_part,
            "eventos_con_verdad": reemplazados,
            "picks_distintos": distintos, "celdas": int(picks_a.size),
            "por_verdad": {k: {"valor_proportional": float(r.valor_a),
                               "valor_shin": float(r.valor_b),
                               "delta": float(r.delta), "se": float(r.se),
                               "n_seeds": int(r.n_seeds)}
                           for k, r in resultados.items()},
            "robusto": bool(robusto), "significativo": bool(signif),
            # las matrices, para poder re-evaluar bajo otra verdad sin re-optimizar
            # (cada brazo cuesta ~45 min en frío)
            "picks_a": picks_a.tolist(), "picks_b": picks_b.tolist(),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"guardado: {path}")


if __name__ == "__main__":
    main()

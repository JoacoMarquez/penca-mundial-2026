"""¿Cobramos premios de fecha al ritmo que el modelo espera?

La 859 ganó la F8 (44 vs 43) y la tentación es sacar reglas de ese caso. La vara
correcta es otra: si la cola de nuestras filas está bien calibrada, lo cobrado en
premios de fecha tiene que caer dentro de lo que el modelo esperaba cobrar. Si
cobramos sistemáticamente MENOS, el optimizador cree que nuestras filas llegan a
la cola más seguido de lo que llegan — y ahí sí hay algo para tocar.

Por fecha, con todo lo que se sabía ANTES de cada partido:

  * **Nuestros picks**: la última planilla de la fecha (refleja lo cargado: el
    drift audit adopta lo que dice la web).
  * **Los rivales**: sus picks REALES, del último snapshot del pool (post-cierre
    son públicos). Mejor que el modelo del pool: el único azar que queda es el
    resultado de los partidos.
  * **Resultados**: se sortean de la grilla de MERCADO con las cuotas del último
    snapshot versionado ANTES del cierre de cada partido (1X2 + over → Poisson
    bivariado, como market_lambdas). Solo mercado, sin el 30% de ratings de
    producción: los ratings de esa fecha no se pueden reconstruir, y las grillas de
    mercado calibran bien (z=−0,6 en 32 partidos, memoria goleada_corta).

Premio de fecha: $10.000 al máximo, repartido entre empatados (Art. 7a/8). Se
reporta E[premio] y P(alguna nuestra gana) por fecha, lo cobrado real con los
mismos datos (control: tiene que dar la F8 a la 859) y el PIT del margen real
(nuestro máx − máx rival) dentro de la distribución simulada.

Las cuotas versionadas arrancan el 13/8: la F1 queda afuera (sin mercado previo).
Solo cuentan los partidos JUGADOS de cada fecha, en lo esperado y en lo real.

Uso (en el VPS, con /etc/penca/env cargado):
    PYTHONPATH=/opt/penca python -m scripts.premios_fecha_calibracion [--sims 20000]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np

from src.clausura.economics import MAX_GOALS, N_SCORES, SIDE, points_matrix, score_index
from src.clausura.odds import ODDS_DIR, EventOdds
from src.clausura.picks import flat_eventos, load_config, market_lambdas, match_odds
from src.clausura.pool_snapshot import load_latest_snapshot
from src.clausura.postmortem import picks_de_planilla, resultados_de_fecha
from src.clausura.rivals import mis_numeros_env
from src.model.poisson import score_grid

log = logging.getLogger(__name__)

PREMIO_FECHA = 10_000.0


def _idx(gl: int, gv: int) -> int:
    return score_index(min(gl, MAX_GOALS), min(gv, MAX_GOALS))


def odds_snapshots() -> list[tuple[datetime, list[EventOdds]]]:
    out = []
    for p in sorted(ODDS_DIR.glob("odds_*.json")):
        ts = datetime.strptime(p.stem.split("_")[1], "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc)
        out.append((ts, [EventOdds(**d) for d in json.loads(p.read_text(encoding="utf-8"))]))
    return out


def grilla_pre_cierre(ev: dict, snaps) -> np.ndarray | None:
    """Grilla de mercado con el último snapshot de cuotas anterior al cierre."""
    cierre = datetime.fromisoformat(ev["cierre_pronostico_utc"])
    for ts, odds in reversed(snaps):
        if ts >= cierre:
            continue
        o = match_odds([ev], odds).get(ev["evento_id"])
        lam = market_lambdas(o) if o else None
        if lam:
            g = score_grid(lam[0], lam[1], lam[2], max_goals=MAX_GOALS)
            return (g / g.sum()).ravel()
    return None


def premio(mios: np.ndarray, rivales: np.ndarray) -> np.ndarray:
    """(n_mios, S), (R, S) → premio de fecha que cobramos en cada sorteo."""
    top_mio = mios.max(axis=0)
    top_riv = rivales.max(axis=0)
    top = np.maximum(top_mio, top_riv)
    n_mios = (mios == top).sum(axis=0)
    n_riv = (rivales == top).sum(axis=0)
    return np.where(top_mio == top, PREMIO_FECHA * n_mios / (n_mios + n_riv), 0.0)


def analizar_fecha(n: int, cfg, snaps, pool, mis: list[int], S: int, rng) -> dict | None:
    evs = [e for e in flat_eventos(cfg) if e["fecha_n"] == n]
    res, _faltan = resultados_de_fecha(cfg, n)
    picks, _ = picks_de_planilla(n, mis)
    jugados = [e for e in evs if e["evento_id"] in res]

    grids = {}
    for e in jugados:
        g = grilla_pre_cierre(e, snaps)
        if g is None:
            log.warning("fecha %d: sin cuotas pre-cierre para %s vs %s — fecha afuera",
                        n, e["local"], e["visitante"])
            return None
        grids[e["evento_id"]] = g

    R = len(pool)
    mios_sim = np.zeros((len(mis), S), dtype=np.int32)
    riv_sim = np.zeros((R, S), dtype=np.int32)
    mios_real = np.zeros(len(mis), dtype=np.int32)
    riv_real = np.zeros(R, dtype=np.int32)
    for e in jugados:
        eid = e["evento_id"]
        P = points_matrix(bool(e["preferencial"]))
        sorteo = rng.choice(N_SCORES, size=S, p=grids[eid])
        real = _idx(*res[eid])
        for k, m in enumerate(mis):
            pk = picks[m].get(eid)
            if pk is None:
                continue
            pi = _idx(*pk)
            mios_sim[k] += P[pi, sorteo]
            mios_real[k] += P[pi, real]
        for r, part in enumerate(pool):
            pk = part["picks"].get(str(eid))
            if not pk:
                continue          # no cargó: 0 pts, igual en lo real que en lo simulado
            pi = _idx(*pk)
            riv_sim[r] += P[pi, sorteo]
            riv_real[r] += P[pi, real]

    pr_sim = premio(mios_sim, riv_sim)
    pr_real = float(premio(mios_real[:, None], riv_real[:, None])[0])
    margen_sim = mios_sim.max(axis=0) - riv_sim.max(axis=0)
    margen_real = int(mios_real.max() - riv_real.max())
    # PIT aleatorizado (el margen es entero): P(sim < real) + U·P(sim == real)
    pit = float((margen_sim < margen_real).mean()
                + rng.uniform() * (margen_sim == margen_real).mean())
    return {
        "fecha": n, "jugados": len(jugados), "de": len(evs),
        "e_premio": float(pr_sim.mean()), "p_ganar": float((pr_sim > 0).mean()),
        "cobrado": pr_real,
        "nuestro_max": int(mios_real.max()), "max_rival": int(riv_real.max()),
        "mejor_fila": mis[int(mios_real.argmax())] % 1000,
        "e_nuestro_max": float(mios_sim.max(axis=0).mean()),
        "e_max_rival": float(riv_sim.max(axis=0).mean()),
        "pit_margen": pit,
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--hasta", type=int, default=8, help="última fecha a evaluar")
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()

    cfg = load_config()
    mis = sorted(mis_numeros_env())
    snap = load_latest_snapshot()
    pool = [p for p in snap["participaciones"] if p["numero"] not in set(mis)]
    snaps = odds_snapshots()
    rng = np.random.default_rng(args.seed)

    filas = []
    for n in range(1, args.hasta + 1):
        f = analizar_fecha(n, cfg, snaps, pool, mis, args.sims, rng)
        if f is not None:
            filas.append(f)

    print(f"{'F':>2} {'jug':>5} {'E[premio]':>10} {'P(gana)':>8} {'cobrado':>8} "
          f"{'máx nuestro':>11} {'E':>5} {'máx rival':>9} {'E':>5} {'PIT':>5}")
    for f in filas:
        print(f"{f['fecha']:>2} {f['jugados']}/{f['de']:<3} {f['e_premio']:>10,.0f} "
              f"{f['p_ganar']:>8.1%} {f['cobrado']:>8,.0f} "
              f"{f['nuestro_max']:>6} ({f['mejor_fila']}) {f['e_nuestro_max']:>5.1f} "
              f"{f['max_rival']:>9} {f['e_max_rival']:>5.1f} {f['pit_margen']:>5.2f}")
    e_tot = sum(f["e_premio"] for f in filas)
    cobrado = sum(f["cobrado"] for f in filas)
    # SD de lo cobrado bajo el modelo: fechas independientes, premio ≈ Bernoulli·$10k
    sd = PREMIO_FECHA * np.sqrt(sum(f["p_ganar"] * (1 - f["p_ganar"]) for f in filas))
    p_cero = float(np.prod([1 - f["p_ganar"] for f in filas]))
    print(f"\nTotal F{filas[0]['fecha']}-F{filas[-1]['fecha']}: esperado ${e_tot:,.0f} "
          f"± {sd:,.0f} · cobrado ${cobrado:,.0f} · z = {(cobrado - e_tot) / sd:+.2f}")
    print(f"P(no ganar ninguna) según el modelo: {p_cero:.1%}")
    print(f"PIT medio del margen: {np.mean([f['pit_margen'] for f in filas]):.2f} "
          f"(0.5 = calibrado; bajo = nuestro máx queda corto vs lo que el modelo cree)")


if __name__ == "__main__":
    main()

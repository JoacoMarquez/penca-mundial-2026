"""¿Conviene meter el goleador en el Monte Carlo, con la tabla REAL del Clausura?

Diseño: docs/superpowers/specs/2026-09-08-goleador-en-monte-carlo-design.md, con un
cambio decidido el 2026-09-24: el prior central NO sale del consenso del pool
(`goleador_prior_desde_pool`) sino de la tabla de goleadores del Clausura (Art. 3 del
reglamento: el goleador es "del mismo" torneo, no el anual). El 24/9 el líder de la
penca (Peñarol+Arezo) y 20 del top-60 tienen a Arezo, nuestras filas se reparten
Arezo/Gómez/Abel, y el gap al líder va de 22 a 46 según quién salga goleador — la
premisa de la decisión del 10/8 ("los 25 pts son un desplazamiento casi común")
ya no describe el estado del torneo. Pero en la tabla del Clausura ninguno de los
tres lidera: por eso el prior tiene que salir de la tabla, no del pool.

Brazos (todos con los mismos insumos congelados: cuotas, snapshot y ranking de la
primera corrida, y los picks jugados/cerrados tal como se cargaron):
  * control  — producción: GOLEADOR_EN_MC apagado.
  * trat[s]  — goleador dentro del MC con el prior del escenario s:
               central (tabla) e inclinaciones ×1.5 a Arezo / Gómez / Abel.
  * pool     — prender el flag TAL CUAL está hoy (prior = consenso del pool), para
               saber qué pasaría si alguien lo prende sin este prior.

Evaluación: una sola verdad por escenario (la del brazo tratado, que es la que
tiene al goleador) y los dos portfolios liquidados ahí, con sorteos comunes y
semillas frescas (offset propio, disjunto del de los gates de producción). El brazo
`pool` se evalúa bajo la verdad central.

Métrica primaria: P(alguna de nuestras filas cobra parte del premio general).
Regla de decisión (la del diseño): adoptar solo si el central da Δ > 2·SE, ningún
escenario inclinado da Δ < −2·SE, y E[premio general] no cae más de $2.000.

Solo imprime y persiste el JSON de evidencia: no versiona planillas, no manda
Telegram, no toca estado.

Uso (VPS, idealmente vía systemd-run con MemoryMax para no pisar a los timers):
    /opt/penca/.venv/bin/python -m scripts.backtest_goleador_actual
    ... --sims 2400 --seeds 2 --escenarios central     # smoke test (NO concluye nada)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "experimentos"

UMBRAL_SE = 2.0
UMBRAL_CAIDA_PENCA = 2_000.0
# Disjunto de EVAL_SEED_OFFSET + 0..k de rerun/cold_check y del CONFIRM_OFFSET de
# exp_clasico_visitante.
EXP_SEED_OFFSET = 7_000
FACTOR_INCLINACION = 1.5
OTROS = "Otros"          # opción 650 del menú de Supermatch: "cualquier otro jugador"
TEMPORADA_FECHAS = 15
PARTIDOS_AÑO_PREVIOS = 29   # Apertura 15 + Intermedio 7 + Clausura 7 al 24/9


@dataclass(frozen=True)
class Goleador:
    nombre: str            # tal como figura en el menú de Supermatch, o el de la tabla
    goles: int             # goles en el CLAUSURA
    jugados: int           # partidos del equipo en el Clausura
    anual: int | None      # goles en el año (None = sin dato → tasa de liga)
    opcion: str | None = None   # opción del menú que cobra; None = la del mismo nombre


# Tabla del Clausura tras la F7 (futbolargentino.com, consultada 2026-09-24) y goles
# anuales (Wikipedia "2026 Liga AUF Uruguaya", al 21/9). Las fuentes no son del todo
# consistentes entre sí (la lesión de Gómez del 6/9 no cierra con +2 anuales al 21/9):
# por eso el veredicto exige que el signo aguante las inclinaciones.
# Torque y Peñarol tienen un partido menos (Torque-Peñarol, 30/9).
TABLA_CLAUSURA: tuple[Goleador, ...] = (
    Goleador("Nahuel Da Silva", 5, 6, 10, OTROS),
    Goleador("Esteban Obregón", 4, 6, 8, OTROS),
    Goleador("Álvaro López", 4, 7, 16),
    Goleador("A. Cambon", 4, 7, None, OTROS),
    Goleador("Matías Arezo", 3, 6, 12),
    Goleador("Fernando Mimbacas", 3, 7, 9),
    Goleador("Luciano Cosentino", 3, 7, 9),
    Goleador("N. López", 3, 7, None, OTROS),
    Goleador("B. Larregui", 3, 7, None, OTROS),
    Goleador("F. Calvo", 3, 7, None, OTROS),
    Goleador("F. Barcelo", 3, 7, None, OTROS),
    Goleador("C. Jaime", 3, 6, None, OTROS),
    Goleador("Maximiliano Gómez", 2, 7, 15),
    Goleador("Christian Tabó", 2, 7, 9, OTROS),
    Goleador("Rubén Bentancur", 2, 7, None),
    Goleador("Tomás Habib", 2, 7, None),
    Goleador("Salomón Rodriguez", 1, 6, 11),
    Goleador("Abel Hernández", 1, 6, None),
    Goleador("Federico Martínez", 1, 7, 9),
) + tuple(Goleador(f"campo {i}", 2, 7, None, OTROS) for i in range(8))
# "campo": la masa de jugadores con 2 goles que no figuran arriba. Cobran "Otros".

TASA_LIGA = 0.25    # goles/partido de un delantero titular sin dato anual


def tasa(g: Goleador) -> float:
    """Goles por partido: mitad ritmo anual, mitad ritmo del Clausura."""
    anual = g.anual / PARTIDOS_AÑO_PREVIOS if g.anual is not None else TASA_LIGA
    return 0.5 * anual + 0.5 * g.goles / max(g.jugados, 1)


def prior_desde_tabla(
    tabla: tuple[Goleador, ...],
    opciones: list[str],
    n_sims: int = 200_000,
    seed: int = 20260924,
) -> np.ndarray:
    """P(cada opción del menú cobra el goleador), por simulación de lo que falta.

    Goles finales = actuales + Poisson(tasa × partidos restantes). Empate en el
    máximo: se reparte la probabilidad entre los empatados (el reglamento no dice
    cómo se resuelve). Jugadores sin opción propia en el menú cobran "Otros"; si el
    menú tampoco trae "Otros", su masa se descarta y el resto se renormaliza.
    """
    if not opciones:
        raise ValueError("menú de goleador vacío")
    idx = {n: i for i, n in enumerate(opciones)}
    rng = np.random.default_rng(seed)
    goles = np.array([g.goles for g in tabla], dtype=float)
    restante = np.array([(TEMPORADA_FECHAS - g.jugados) * tasa(g) for g in tabla])
    finales = goles[:, None] + rng.poisson(restante[:, None], (len(tabla), n_sims))
    maximos = finales == finales.max(axis=0)
    peso = (maximos / maximos.sum(axis=0)).mean(axis=1)
    p = np.zeros(len(opciones))
    for g, w in zip(tabla, peso):
        destino = g.opcion or g.nombre
        if destino not in idx:
            destino = OTROS
        if destino in idx:
            p[idx[destino]] += w
    if p.sum() <= 0:
        raise ValueError("ninguna opción del menú recibe probabilidad")
    return p / p.sum()


def inclinar(p: np.ndarray, i: int, factor: float = FACTOR_INCLINACION) -> np.ndarray:
    """Multiplica P(opción i) por `factor` y renormaliza el vector completo."""
    q = np.asarray(p, dtype=float).copy()
    q[i] *= factor
    return q / q.sum()


@dataclass
class Pareado:
    delta: float
    se: float
    a: float
    b: float

    @classmethod
    def de(cls, va: list[float], vb: list[float]) -> "Pareado":
        d = np.asarray(vb) - np.asarray(va)
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        return cls(float(d.mean()), se, float(np.mean(va)), float(np.mean(vb)))

    def claro_positivo(self) -> bool:
        return self.delta > UMBRAL_SE * self.se

    def claro_negativo(self) -> bool:
        return self.delta < -UMBRAL_SE * self.se


def clasificar(central_p: Pareado, central_penca: Pareado,
               inclinados_p: dict[str, Pareado]) -> str:
    """adoptar / rechazado / rechazado por fragilidad / inconcluso (regla del diseño)."""
    if central_p.claro_negativo():
        return "rechazado"
    if any(v.claro_negativo() for v in inclinados_p.values()):
        return "rechazado por fragilidad"
    if central_p.claro_positivo() and central_penca.delta >= -UMBRAL_CAIDA_PENCA:
        return "adoptar"
    return "inconcluso"


# ---------------------------------------------------------------------------
# Corrida (solo en el VPS: necesita API, snapshot y planillas vivas)
# ---------------------------------------------------------------------------

def _congelar_insumos() -> dict:
    """Fija cuotas y snapshot a los de la primera lectura, y no escribe cache.

    Sin esto cada brazo re-lee el ES y el snapshot más nuevo: si se mueve una
    cuota entre corridas, el Δ mezcla goleador con mercado.
    """
    import src.clausura.odds as odds_mod
    import src.clausura.picks as picks
    import src.clausura.pool_snapshot as ps

    odds = picks.fetch_primera_odds()
    snap = ps.load_latest_snapshot(max_age_hours=48)
    if snap is None:
        raise SystemExit("sin snapshot del pool de <48h — el experimento no se puede correr")
    picks.fetch_primera_odds = lambda: odds
    odds_mod.save_odds_snapshot = lambda *_a, **_k: None
    ps.load_latest_snapshot = lambda *a, **k: snap
    return {"n_odds": len(odds), "snapshot_utc": snap.get("generado_utc"),
            "n_participaciones_snapshot": len(snap.get("participaciones", []))}


_ORIGINALES: dict = {}


def _con_prior(p_por_nombre: dict[str, float] | None):
    """Parchea el prior del goleador. None = el de producción (consenso del pool)."""
    import src.clausura.especiales as esp

    if not _ORIGINALES:     # una sola vez: re-parchear no debe anidar wrappers
        _ORIGINALES.update(ops=esp.opciones_goleador_desde_snapshot,
                           prior=esp.goleador_prior_desde_pool)
    original_ops, original_prior = _ORIGINALES["ops"], _ORIGINALES["prior"]
    menu: list = []

    def ops(snapshot):
        menu[:] = original_ops(snapshot)
        return list(menu)

    def prior(counts, shrink=0.5):
        if p_por_nombre is None:
            return original_prior(counts, shrink=shrink)
        p = np.array([p_por_nombre.get(o.nombre, 0.0) for o in menu])
        return p / p.sum()

    esp.opciones_goleador_desde_snapshot = ops
    esp.goleador_prior_desde_pool = prior
    return menu


def _optimizar(fecha: int, n_part: int, n_sims: int, goleador: bool,
               p_por_nombre: dict[str, float] | None) -> dict:
    import src.clausura.picks as picks

    picks.GOLEADOR_EN_MC = goleador
    menu = _con_prior(p_por_nombre)
    ctx: dict = {}
    picks.run(fecha, n_part, telegram=False, n_sims=n_sims, contexto=ctx, guardar=False)
    if ctx.get("evaluador") is None:
        raise SystemExit("la corrida no dejó evaluador en el contexto")
    ctx["menu"] = [o.nombre for o in menu]
    return ctx


def _evaluar(ev, pa: np.ndarray, pb: np.ndarray, n_seeds: int) -> dict:
    """Métricas pareadas de A vs B bajo la verdad de `ev`, semillas frescas."""
    from src.clausura.strategy import EVAL_SEED_OFFSET

    cols = {k: ([], []) for k in ("p_gana_penca", "e_premio_penca", "e_premio_total",
                                  "e_puntos_mejor")}
    for k in range(n_seeds):
        s = ev._simulador(ev._cfg.seed + EVAL_SEED_OFFSET + EXP_SEED_OFFSET + k)
        for lado, picks in ((0, pa), (1, pb)):
            ev._cargar(s, picks)
            r = s.result()
            for campo, (va, vb) in cols.items():
                (va, vb)[lado].append(float(getattr(r, campo)))
        del s
        gc.collect()
    return {campo: Pareado.de(va, vb) for campo, (va, vb) in cols.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from src.clausura.picks import DEFAULT_SIMS, resolve_fecha
    from src.clausura.rivals import mis_numeros_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", default="auto")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--escenarios", default="central,arezo,gomez,abel,pool")
    args = ap.parse_args()

    fecha = resolve_fecha(args.fecha)
    n_part = len(mis_numeros_env())
    if n_part == 0:
        raise SystemExit("CLAUSURA_MIS_PARTICIPACIONES vacío")
    escenarios = [e.strip() for e in args.escenarios.split(",") if e.strip()]
    entradas = _congelar_insumos()
    log.info("fecha %d · %d participaciones · %d sorteos · %d semillas · %s",
             fecha, n_part, args.sims, args.seeds, escenarios)

    # ---- control (producción) ----
    ctrl = _optimizar(fecha, n_part, args.sims, goleador=False, p_por_nombre=None)
    pa = np.asarray(ctrl["portfolio"].picks, dtype=np.int64)

    # el menú sale de la primera corrida con goleador (el control no lo arma)
    import src.clausura.especiales as esp
    import src.clausura.pool_snapshot as ps
    menu = [o.nombre for o in esp.opciones_goleador_desde_snapshot(
        ps.load_latest_snapshot())]
    if not menu:
        raise SystemExit("no se pudo reconstruir el menú de goleador del snapshot")
    p_central = prior_desde_tabla(TABLA_CLAUSURA, menu)
    priors = {"central": p_central}
    for nombre, jugador in (("arezo", "Matías Arezo"), ("gomez", "Maximiliano Gómez"),
                            ("abel", "Abel Hernández")):
        if jugador in menu:
            priors[nombre] = inclinar(p_central, menu.index(jugador))
    log.info("prior central: %s", {m: round(float(p), 3) for m, p in zip(menu, p_central)
                                   if p > 0.001})

    resultados, portfolios = {}, {}
    for esc in escenarios:
        if esc == "pool":
            continue
        if esc not in priors:
            log.warning("escenario %s sin opción en el menú — salteado", esc)
            continue
        pmap = dict(zip(menu, priors[esc].tolist()))
        ctx = _optimizar(fecha, n_part, args.sims, goleador=True, p_por_nombre=pmap)
        pb = np.asarray(ctx["portfolio"].picks, dtype=np.int64)
        portfolios[esc] = (ctx["evaluador"], pb)
        m = _evaluar(ctx["evaluador"], pa, pb, args.seeds)
        resultados[esc] = {"metricas": m, "celdas_distintas": int((pa != pb).sum())}
        log.info("%s: ΔP(gana) %+.4f ± %.4f · ΔE[penca] %+.0f ± %.0f · %d celdas",
                 esc, m["p_gana_penca"].delta, m["p_gana_penca"].se,
                 m["e_premio_penca"].delta, m["e_premio_penca"].se,
                 resultados[esc]["celdas_distintas"])
        del ctx
        gc.collect()

    if "pool" in escenarios and "central" in portfolios:
        ctx = _optimizar(fecha, n_part, args.sims, goleador=True, p_por_nombre=None)
        pb = np.asarray(ctx["portfolio"].picks, dtype=np.int64)
        m = _evaluar(portfolios["central"][0], pa, pb, args.seeds)
        resultados["pool"] = {"metricas": m, "celdas_distintas": int((pa != pb).sum()),
                              "nota": "prior de producción, evaluado bajo la verdad central"}
        log.info("pool: ΔP(gana) %+.4f ± %.4f · ΔE[penca] %+.0f ± %.0f",
                 m["p_gana_penca"].delta, m["p_gana_penca"].se,
                 m["e_premio_penca"].delta, m["e_premio_penca"].se)

    veredicto = None
    if "central" in resultados:
        incl = {k: v["metricas"]["p_gana_penca"] for k, v in resultados.items()
                if k in ("arezo", "gomez", "abel")}
        veredicto = clasificar(resultados["central"]["metricas"]["p_gana_penca"],
                               resultados["central"]["metricas"]["e_premio_penca"], incl)

    out = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "fecha": fecha, "n_participaciones": n_part, "n_sims": args.sims,
        "n_seeds": args.seeds, "entradas": entradas, "menu": menu,
        "priors": {k: dict(zip(menu, np.round(v, 4).tolist())) for k, v in priors.items()},
        "resultados": {k: {**v, "metricas": {c: vars(p) for c, p in v["metricas"].items()}}
                       for k, v in resultados.items()},
        "veredicto": veredicto,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"goleador_actual_{datetime.now(timezone.utc):%Y%m%d}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nveredicto: {veredicto}\nguardado: {path}")


if __name__ == "__main__":
    main()

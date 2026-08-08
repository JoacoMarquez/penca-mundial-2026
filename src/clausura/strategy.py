"""Asignación de N participaciones al portfolio, optimizando E[premio].

Dos etapas, igual que en la JMLM pero con el objetivo económico correcto:

  1. **Menú de candidatos por partido** — de los 36 marcadores solo unos pocos son
     jugables. Nos quedamos con los mejores por E[pts] y con los de mayor "hueco"
     (E[pts] por unidad de popularidad del pool). El segundo conjunto es el que mete
     al 0-0 y compañía: marcadores frecuentes en la liga que el pool subjuega, donde
     acertar no obliga a repartir el premio.

  2. **Ascenso por coordenadas sobre E[premio] simulado.** La participación 1 ancla en
     EV puro; el resto arranca desde ahí y se perturba partido a partido eligiendo el
     candidato que más sube E[premio] del portfolio COMPLETO. La diversificación
     emerge del objetivo (empatar contigo mismo divide el premio) en vez de imponerse
     por regla, que es la diferencia con las 5 objective functions fijas del Mundial.

Todas las evaluaciones comparten los mismos sorteos (common random numbers, semilla
fija en SimConfig): la diferencia entre dos portfolios es señal, no ruido de MC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.clausura.economics import (
    N_SCORES,
    PrizeConfig,
    SeasonSimulator,
    SimConfig,
    SimResult,
    flatten_grid,
    index_score,
    score_index,
)
from src.clausura.pool import PoolConfig, pool_distribution
from src.clausura.scoring import expected_points_grid

log = logging.getLogger(__name__)


# Cómo se ordena la rama de "hueco" del menú de candidatos. `legacy_hueco` es la
# VIGENTE y se queda: el A/B del 2026-08-08 midió `mispricing` peor por Δ E[premio]
# −$9.486 ± 2.154 (12 reps pareadas, negativo en 10/12, t≈−4.4). Ver Candidato.hueco.
HUECO_METRIC = "legacy_hueco"


@dataclass(frozen=True)
class Candidato:
    pick: tuple[int, int]
    e_points: float
    pool_q: float
    p_scoreline: float

    @property
    def hueco(self) -> float:
        """E[pts] por unidad de popularidad del pool: valor NO DISPUTADO. La vigente.

        Sí, ordena casi por rareza —de 1-0 a 1-4 el E[pts] cae a la mitad (2.19 →
        1.00) mientras el pool_q cae 50 veces (18.9% → 0.38%), así que el cociente es
        casi 1/pool_q— y por eso el menú incluye marcadores con P<0.7% (4-2, 1-4, 3-3
        en Liverpool–Albion) y deja afuera al 1-3 (2.28%). Parece un defecto y NO lo
        es: en esta penca el premio se REPARTE entre empatados, así que lo que vale no
        es que el pool se equivoque, es que nadie más lo tenga. Un 0-0 subjugado 2.2×
        lo juega igual el 4.4% del pool (30 rivales con quienes repartir); un 1-4, el
        0.38% (2.6 rivales). Dividir por pool_q compra exclusividad, y el E[pts] en el
        numerador impide que sea rareza pura.

        Se intentó reemplazarla por `mispricing` (P real / P del pool) el 2026-08-08 y
        el backtest la rechazó: Δ E[premio] **−$9.486 ± 2.154** en 12 reps pareadas
        (4 temporadas × 3 semillas, 600 sims), negativo en 10/12, t≈−4.4. Reproducir
        con `python -m src.clausura.backtest --experimento-menu --reps 3 --sims 600`.
        No volver a "arreglar" esto sin correr ese A/B."""
        return self.e_points / (self.pool_q + 1e-4)

    @property
    def mispricing(self) -> float:
        """Cuánto se equivoca el pool: P real / P del pool. RECHAZADA por backtest.

        Intuitivamente mejor que `hueco` —mide desajuste en vez de rareza, y numerador
        y denominador tienden a cero juntos— pero mide PEOR (ver `hueco`), porque
        ignora con cuántos habría que repartir el premio. Se conserva para el brazo B
        del A/B y como registro del resultado negativo."""
        return self.p_scoreline / (self.pool_q + 1e-4)


def build_candidates(
    grid: np.ndarray,
    pool_q: np.ndarray,
    preferencial: bool = False,
    k_ev: int = 5,
    k_hueco: int = 3,
    min_prob: float = 0.005,
    metrica: str | None = None,
) -> list[Candidato]:
    """Menú de marcadores jugables: top por E[pts] ∪ top por hueco de pool.

    La segunda rama busca marcadores donde acertar NO obliga a repartir el premio, y
    se ordena por `hueco` (E[pts]/pool_q). `metrica="mispricing"` es la alternativa
    rechazada por el A/B del 8/8 y existe para reproducirlo.
    """
    metrica = metrica or HUECO_METRIC
    p = flatten_grid(grid)
    cands = [
        Candidato(
            pick=index_score(idx),
            e_points=expected_points_grid(index_score(idx), grid, preferencial),
            pool_q=float(pool_q[idx]),
            p_scoreline=float(p[idx]),
        )
        for idx in range(N_SCORES)
        if p[idx] >= min_prob
    ]
    if not cands:   # grilla degenerada: caemos al modal
        idx = int(np.argmax(p))
        return [Candidato(index_score(idx), expected_points_grid(index_score(idx), grid,
                                                                 preferencial),
                          float(pool_q[idx]), float(p[idx]))]

    by_ev = sorted(cands, key=lambda c: -c.e_points)[:k_ev]
    if metrica == "legacy_hueco":
        by_hueco = sorted(cands, key=lambda c: -c.hueco)[:k_hueco]
    else:
        # Desempate por E[pts]: los marcadores "impopulares pero sin sesgo" quedan
        # todos en un mismo escalón de mispricing (~1.5 en el caso medido), y dentro
        # de ese escalón conviene el que más puntos rinde, no el más raro.
        by_hueco = sorted(cands, key=lambda c: (-c.mispricing, -c.e_points))[:k_hueco]

    out, seen = [], set()
    for c in by_ev + by_hueco:
        if c.pick not in seen:
            seen.add(c.pick)
            out.append(c)
    return out


@dataclass
class EspecialesInput:
    """Insumos para optimizar Campeón y Goleador dentro del portfolio."""
    local_de: np.ndarray               # (n_matches,) índice de equipo local
    visita_de: np.ndarray              # (n_matches,) índice de equipo visitante
    n_teams: int
    pool_q_campeon: np.ndarray         # (n_teams,) qué campeón pica el pool
    p_goleador: np.ndarray | None = None       # prior sobre opciones de goleador
    pool_q_goleador: np.ndarray | None = None
    frozen_campeon: np.ndarray | None = None   # (n_part,) equipo ya cargado, -1 = libre
    frozen_goleador: np.ndarray | None = None


@dataclass
class PortfolioClausura:
    picks: np.ndarray                  # (n_participaciones, n_partidos) índices de score
    candidatos: list[list[Candidato]]
    resultado: SimResult
    campeon: np.ndarray | None = None   # (n_participaciones,) índice de equipo
    goleador: np.ndarray | None = None  # (n_participaciones,) índice de opción
    p_campeon: np.ndarray | None = None  # (n_teams,) P(campeón) del modelo

    def as_scores(self) -> list[list[tuple[int, int]]]:
        return [[index_score(int(i)) for i in fila] for fila in self.picks]

    def diversidad(self) -> float:
        """Fracción de partidos donde no todas las participaciones juegan lo mismo."""
        return float(np.mean([len(set(self.picks[:, m])) > 1 for m in range(self.picks.shape[1])]))


def build_portfolio(
    grids: list[np.ndarray],
    fecha_de_partido: list[int],
    preferencial: list[bool],
    n_participaciones: int = 5,
    pool_cfg: PoolConfig | None = None,
    prize: PrizeConfig | None = None,
    sim: SimConfig | None = None,
    max_passes: int = 3,
    frozen_picks: np.ndarray | None = None,
    frozen_mask: np.ndarray | None = None,
    especiales: EspecialesInput | None = None,
    pool_qs: list[np.ndarray] | None = None,
    rivals=None,
) -> PortfolioClausura:
    """Construye el portfolio de N participaciones maximizando E[premio] simulado.

    `frozen_mask[m]=True` marca partidos cuyo pick YA fue cargado en la web (o ya se
    jugó): en esas columnas se usa `frozen_picks` tal cual y el optimizador no las toca.
    Es el mecanismo de re-optimización fecha a fecha: lo pasado queda fijo, lo futuro
    se replanifica con la información nueva.

    Con `especiales`, Campeón y Goleador entran al mismo ascenso por coordenadas como
    dos columnas más de cada participación (25 pts c/u sobre el total general).

    `pool_qs` permite pasar la distribución del pool por partido ya construida (por
    ejemplo la EMPÍRICA del snapshot post-inicio); sin ella se genera del prior.

    `rivals` (RivalModel de src.clausura.rivals) reemplaza el pool i.i.d. por el
    empírico por participación: picks conocidos, estilo γ, ausentismo y standings
    reales. Es el insumo correcto post-inicio del campeonato.
    """
    pool_cfg = pool_cfg or PoolConfig()
    n_matches = len(grids)

    if frozen_mask is None:
        frozen_mask = np.zeros(n_matches, dtype=bool)
    if frozen_mask.any() and frozen_picks is None:
        raise ValueError("frozen_mask sin frozen_picks")

    if pool_qs is None:
        pool_qs = [pool_distribution(g, pool_cfg) for g in grids]
    elif len(pool_qs) != n_matches:
        raise ValueError(f"pool_qs tiene {len(pool_qs)} entradas, se esperaban {n_matches}")
    candidatos = [
        build_candidates(g, q, pref)
        for g, q, pref in zip(grids, pool_qs, preferencial)
    ]

    simulator = SeasonSimulator(grids, fecha_de_partido, preferencial, pool_qs, prize, sim,
                                rivals)

    # ancla de EV puro, replicada en todas las participaciones
    picks = np.zeros((n_participaciones, n_matches), dtype=np.int64)
    for m in range(n_matches):
        if frozen_mask[m]:
            picks[:, m] = frozen_picks[:, m]
            continue
        best = max(candidatos[m], key=lambda c: c.e_points)
        picks[:, m] = score_index(*best.pick)
    simulator.load_picks(picks)

    # especiales: activar y anclar en el argmax de probabilidad
    p_champ = None
    if especiales is not None:
        simulator.enable_campeon(
            especiales.local_de, especiales.visita_de,
            especiales.n_teams, especiales.pool_q_campeon,
        )
        from src.clausura.especiales import p_campeon as _p_campeon
        p_champ = _p_campeon(simulator.champ_sim, especiales.n_teams)
        ancla_campeon = int(np.argmax(p_champ))
        for i in range(n_participaciones):
            fijo = especiales.frozen_campeon[i] if especiales.frozen_campeon is not None else -1
            simulator.set_campeon_pick(i, int(fijo) if fijo >= 0 else ancla_campeon)
        if especiales.p_goleador is not None:
            simulator.enable_goleador(especiales.p_goleador, especiales.pool_q_goleador)
            ancla_gol = int(np.argmax(especiales.p_goleador))
            for i in range(n_participaciones):
                fijo = (especiales.frozen_goleador[i]
                        if especiales.frozen_goleador is not None else -1)
                simulator.set_goleador_pick(i, int(fijo) if fijo >= 0 else ancla_gol)

    # ascenso por coordenadas: la participación 0 queda fija como ancla de EV
    actual = simulator.e_premio_total()
    log.info("ancla EV: E[premio]=%.0f", actual)

    for p in range(max_passes):
        mejoras = 0
        for i in range(1, n_participaciones):
            for m in range(n_matches):
                if frozen_mask[m]:
                    continue
                orig = int(simulator.picks[i, m])
                mejor_idx, mejor_val = orig, actual
                for c in candidatos[m]:
                    cand = score_index(*c.pick)
                    if cand == orig:
                        continue
                    simulator.set_pick(i, m, cand)
                    val = simulator.e_premio_total()
                    if val > mejor_val:
                        mejor_idx, mejor_val = cand, val
                simulator.set_pick(i, m, mejor_idx)
                if mejor_idx != orig:
                    mejoras += 1
                    actual = mejor_val

        # especiales como columnas extra (todas las participaciones, incluida la 0:
        # diversificar el campeón es barato y no compromete el ancla de marcadores)
        if especiales is not None:
            for i in range(n_participaciones):
                if (especiales.frozen_campeon is None
                        or especiales.frozen_campeon[i] < 0):
                    actual, cambio = _optimize_especial(
                        simulator, simulator.set_campeon_pick, simulator.campeon_picks,
                        i, especiales.n_teams, actual)
                    mejoras += cambio
                if (simulator.gol_sim is not None
                        and (especiales.frozen_goleador is None
                             or especiales.frozen_goleador[i] < 0)):
                    actual, cambio = _optimize_especial(
                        simulator, simulator.set_goleador_pick, simulator.goleador_picks,
                        i, len(especiales.p_goleador), actual)
                    mejoras += cambio

        log.info("pasada %d: %d cambios, E[premio]=%.0f", p + 1, mejoras, actual)
        if mejoras == 0:
            break

    # El E[premio] que se reporta se evalúa con sorteos FRESCOS (semilla distinta):
    # el valor in-sample del optimizador está sesgado hacia arriba por construcción
    # (el ascenso por coordenadas maximizó exactamente esos sorteos — winner's curse).
    resultado = _evaluate_fresh(
        simulator, grids, fecha_de_partido, preferencial, pool_qs, prize, especiales,
        rivals,
    )
    log.info("E[premio] out-of-sample: $%.0f (in-sample del optimizador: $%.0f)",
             resultado.e_premio_total, actual)

    return PortfolioClausura(
        picks=simulator.picks.copy(),
        candidatos=candidatos,
        resultado=resultado,
        campeon=simulator.campeon_picks.copy() if simulator.campeon_picks is not None else None,
        goleador=simulator.goleador_picks.copy() if simulator.goleador_picks is not None else None,
        p_campeon=p_champ,
    )


# offset de la semilla de evaluación respecto de la de optimización (fix winner's curse)
EVAL_SEED_OFFSET = 900_001


def _evaluate_fresh(
    simulator: SeasonSimulator,
    grids: list[np.ndarray],
    fecha_de_partido: list[int],
    preferencial: list[bool],
    pool_qs: list[np.ndarray],
    prize: PrizeConfig | None,
    especiales: EspecialesInput | None,
    rivals,
) -> SimResult:
    """Re-liquida el portfolio final en un simulador con semilla independiente."""
    cfg = simulator.cfg
    eval_cfg = SimConfig(n_sims=cfg.n_sims, n_rivales=cfg.n_rivales,
                         seed=cfg.seed + EVAL_SEED_OFFSET)
    ev = SeasonSimulator(grids, fecha_de_partido, preferencial, pool_qs, prize,
                         eval_cfg, rivals)
    ev.load_picks(simulator.picks)
    if especiales is not None and simulator.champ_sim is not None:
        ev.enable_campeon(especiales.local_de, especiales.visita_de,
                          especiales.n_teams, especiales.pool_q_campeon)
        for i in range(simulator.campeon_picks.shape[0]):
            ev.set_campeon_pick(i, int(simulator.campeon_picks[i]))
        if simulator.gol_sim is not None:
            ev.enable_goleador(especiales.p_goleador, especiales.pool_q_goleador)
            for i in range(simulator.goleador_picks.shape[0]):
                ev.set_goleador_pick(i, int(simulator.goleador_picks[i]))
    return ev.result()


def _optimize_especial(simulator, setter, current, i, n_opciones, actual) -> tuple[float, int]:
    """Prueba todas las opciones del especial para la participación i. (nuevo_valor, cambió)."""
    orig = int(current[i])
    mejor_op, mejor_val = orig, actual
    for op in range(n_opciones):
        if op == orig:
            continue
        setter(i, op)
        val = simulator.e_premio_total()
        if val > mejor_val:
            mejor_op, mejor_val = op, val
    setter(i, mejor_op)
    return mejor_val, int(mejor_op != orig)


# -------------------- baselines de comparación --------------------

def baseline_chalk(grids: list[np.ndarray], n_participaciones: int = 5) -> np.ndarray:
    """Todas las participaciones al marcador modal del mercado (chalk puro)."""
    picks = np.zeros((n_participaciones, len(grids)), dtype=np.int64)
    for m, g in enumerate(grids):
        picks[:, m] = int(np.argmax(flatten_grid(g)))
    return picks


def baseline_ev(
    grids: list[np.ndarray],
    preferencial: list[bool],
    n_participaciones: int = 5,
) -> np.ndarray:
    """Todas al argmax E[pts] — lo que haría quien optimiza puntaje esperado."""
    picks = np.zeros((n_participaciones, len(grids)), dtype=np.int64)
    for m, g in enumerate(grids):
        evs = [expected_points_grid(index_score(i), g, preferencial[m]) for i in range(N_SCORES)]
        picks[:, m] = int(np.argmax(evs))
    return picks


def baseline_random_diverse(
    grids: list[np.ndarray],
    preferencial: list[bool],
    n_participaciones: int = 5,
    seed: int = 7,
) -> np.ndarray:
    """Diversidad ingenua: cada participación toma un candidato al azar del top-EV.

    Sirve para separar "diversificar ayuda" de "diversificar BIEN ayuda".
    """
    rng = np.random.default_rng(seed)
    picks = np.zeros((n_participaciones, len(grids)), dtype=np.int64)
    for m, g in enumerate(grids):
        evs = np.array([expected_points_grid(index_score(i), g, preferencial[m])
                        for i in range(N_SCORES)])
        top = np.argsort(-evs)[:n_participaciones]
        picks[:, m] = rng.permutation(top)[:n_participaciones]
    return picks

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


@dataclass(frozen=True)
class Candidato:
    pick: tuple[int, int]
    e_points: float
    pool_q: float
    p_scoreline: float

    @property
    def hueco(self) -> float:
        """E[pts] por unidad de popularidad del pool: 'valor no disputado'."""
        return self.e_points / (self.pool_q + 1e-4)


def build_candidates(
    grid: np.ndarray,
    pool_q: np.ndarray,
    preferencial: bool = False,
    k_ev: int = 5,
    k_hueco: int = 3,
    min_prob: float = 0.005,
) -> list[Candidato]:
    """Menú de marcadores jugables: top por E[pts] ∪ top por hueco de pool."""
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
    by_hueco = sorted(cands, key=lambda c: -c.hueco)[:k_hueco]

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
) -> PortfolioClausura:
    """Construye el portfolio de N participaciones maximizando E[premio] simulado.

    `frozen_mask[m]=True` marca partidos cuyo pick YA fue cargado en la web (o ya se
    jugó): en esas columnas se usa `frozen_picks` tal cual y el optimizador no las toca.
    Es el mecanismo de re-optimización fecha a fecha: lo pasado queda fijo, lo futuro
    se replanifica con la información nueva.

    Con `especiales`, Campeón y Goleador entran al mismo ascenso por coordenadas como
    dos columnas más de cada participación (25 pts c/u sobre el total general).
    """
    pool_cfg = pool_cfg or PoolConfig()
    n_matches = len(grids)

    if frozen_mask is None:
        frozen_mask = np.zeros(n_matches, dtype=bool)
    if frozen_mask.any() and frozen_picks is None:
        raise ValueError("frozen_mask sin frozen_picks")

    pool_qs = [pool_distribution(g, pool_cfg) for g in grids]
    candidatos = [
        build_candidates(g, q, pref)
        for g, q, pref in zip(grids, pool_qs, preferencial)
    ]

    simulator = SeasonSimulator(grids, fecha_de_partido, preferencial, pool_qs, prize, sim)

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

    return PortfolioClausura(
        picks=simulator.picks.copy(),
        candidatos=candidatos,
        resultado=simulator.result(),
        campeon=simulator.campeon_picks.copy() if simulator.campeon_picks is not None else None,
        goleador=simulator.goleador_picks.copy() if simulator.goleador_picks is not None else None,
        p_campeon=p_champ,
    )


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

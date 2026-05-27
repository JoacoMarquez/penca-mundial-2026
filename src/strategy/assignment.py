"""Asignación adaptativa de picks a pencas según ranking actual.

Principio: la penca que va punteando recibe la estrategia más conservadora (lock-in del lead);
la penca que va última recibe la estrategia más arriesgada (Hail Mary para remontar).

Las 5 estrategias están ordenadas naturalmente de menor a mayor varianza:
    1. ev             (Favorito)    — más conservadora
    2. differentiated (Diferencial)
    3. tail           (Goleada)
    4. upset          (Sorpresa)
    5. variance       (Varianza)    — más arriesgada
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


def fetch_my_pencas_standings(
    api_base_url: str,
    api_key: str,
    my_penca_ids: list[int],
) -> dict[int, dict[str, Any]]:
    """GET /leaderboard y devuelve solo las entries de mis pencas.

    Retorna: {penca_id: {"rank": N, "points_total": X, ...}, ...}.
    Si la API falla, devuelve {} (caller decide qué hacer).
    """
    if not api_base_url or not api_key or not my_penca_ids:
        return {}
    try:
        with httpx.Client(
            timeout=10.0,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as c:
            r = c.get(f"{api_base_url.rstrip('/')}/leaderboard")
        if r.status_code != 200:
            log.warning("Leaderboard returned %d", r.status_code)
            return {}
        data = r.json()
        return {
            int(e["penca_id"]): e
            for e in data.get("entries", [])
            if int(e["penca_id"]) in my_penca_ids
        }
    except Exception as e:
        log.warning("fetch standings falló: %s", e)
        return {}


def optimal_assignment(
    picks_in_strategy_order: list[dict],
    penca_ids: list[int],
    grid: Any,  # numpy 2D array P(g_L, g_V)
    pencas_standings: dict[int, dict[str, Any]] | None = None,
    points_rule=None,
) -> list[tuple[int, dict, int | None]]:
    """Asignación ÓPTIMA exacta vía enumeración de las 120 permutaciones.

    Para cada permutación π de los 5 picks a las 5 pencas, calcula:
        E[max(score_final_i para i en 1..5)] = Σ_ω P(ω) · max_i (current_score_i + points(π(i), ω))

    Donde ω = (g_local, g_visit) recorre la grilla de marcadores ponderados por probabilidad.

    Elige la permutación con mayor E[max]. Maximiza directamente la probabilidad de que
    al menos una penca termine bien rankeada en el pool.

    Cost: 120 perms × ~64 outcomes × 5 pencas ≈ 40k operaciones simples — <100ms.
    """
    from itertools import permutations
    import numpy as np
    from src.model.poisson import jmlm_points
    if points_rule is None:
        points_rule = jmlm_points

    if len(penca_ids) != 5 or len(picks_in_strategy_order) != 5:
        raise ValueError(f"5 picks/pencas requeridos, got {len(picks_in_strategy_order)}/{len(penca_ids)}")

    current_pts: dict[int, int] = {
        pid: int(pencas_standings.get(pid, {}).get("points_total", 0)) if pencas_standings else 0
        for pid in penca_ids
    }

    n = grid.shape[0]

    # Pre-computar tabla de puntos: para cada pick × cada outcome → puntos
    pts_table = np.zeros((5, n, n), dtype=np.int8)
    for pi, pick in enumerate(picks_in_strategy_order):
        pick_score = (int(pick["score"][0]), int(pick["score"][1]))
        for gL in range(n):
            for gV in range(n):
                pts_table[pi, gL, gV] = points_rule(pick_score, (gL, gV))

    best_perm: tuple[int, ...] | None = None
    best_objective = -float("inf")
    current_pts_arr = np.array([current_pts[pid] for pid in penca_ids])

    for perm in permutations(range(5)):
        # perm[i] = índice del pick asignado a la penca penca_ids[i]
        # tabla[i, gL, gV] = puntos de penca i en outcome (gL, gV)
        per_penca_pts = pts_table[list(perm)]  # shape (5, n, n)
        # Sumar score actual a cada penca
        finals = per_penca_pts + current_pts_arr.reshape(5, 1, 1)  # broadcasting
        # max sobre las 5 pencas para cada outcome
        max_per_outcome = finals.max(axis=0)  # shape (n, n)
        # E[max] = sum (P(ω) * max)
        exp_max = float((grid * max_per_outcome).sum())
        if exp_max > best_objective:
            best_objective = exp_max
            best_perm = perm

    assert best_perm is not None

    # Ranks de pencas por current score (descendente)
    pencas_by_score = sorted(penca_ids, key=lambda pid: -current_pts[pid])
    rank_by_penca = {pid: i + 1 for i, pid in enumerate(pencas_by_score)}

    return [
        (penca_ids[i], picks_in_strategy_order[best_perm[i]], rank_by_penca[penca_ids[i]])
        for i in range(5)
    ]


def assign_picks_to_pencas(
    picks_in_strategy_order: list[dict],
    penca_ids: list[int],
    pencas_standings: dict[int, dict[str, Any]] | None = None,
) -> list[tuple[int, dict, int | None]]:
    """Asigna 5 picks a 5 pencas según ranking actual.

    Args:
        picks_in_strategy_order: lista de 5 dicts en orden natural [ev, diff, tail, upset, variance].
                                  (idx 0 = más conservador, idx 4 = más arriesgado)
        penca_ids: lista de 5 IDs de pencas.
        pencas_standings: dict {penca_id: {"points_total": N, ...}}. Si None o vacío, fallback a mapeo fijo.

    Returns:
        Lista de 5 tuplas (penca_id, pick, rank_dentro_de_mis_pencas). rank=None si no hay standings.

        El mapeo es:
            penca con MÁS puntos → picks_in_strategy_order[0]  (Favorito)
            penca con MENOS puntos → picks_in_strategy_order[4]  (Varianza)

    Edge cases:
        - Empate de puntos → desempate estable por penca_id ascendente.
        - Sin standings → mapeo fijo P1→ev, P2→diff, ... P5→var.
    """
    if len(picks_in_strategy_order) != 5 or len(penca_ids) != 5:
        raise ValueError(
            f"Se esperaban exactamente 5 picks y 5 penca_ids, recibí {len(picks_in_strategy_order)} y {len(penca_ids)}"
        )

    # Sin standings: mapeo fijo (modo "primer partido del torneo" o fallback)
    if not pencas_standings:
        return [(pid, pick, None) for pid, pick in zip(penca_ids, picks_in_strategy_order)]

    # Ordenar pencas por points_total descendente; desempate por penca_id ascendente
    def sort_key(pid: int) -> tuple[int, int]:
        pts = pencas_standings.get(pid, {}).get("points_total", 0)
        return (-int(pts), pid)

    pencas_sorted = sorted(penca_ids, key=sort_key)

    # Pencas ordenadas: [mejor, ..., peor]
    # Asignamos picks[0] (Favorito) a la mejor, picks[4] (Varianza) a la peor
    return [
        (pid, pick, rank + 1)
        for rank, (pid, pick) in enumerate(zip(pencas_sorted, picks_in_strategy_order))
    ]

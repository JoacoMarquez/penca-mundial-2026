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


def fetch_pool_top_k_threshold(
    api_base_url: str,
    api_key: str,
    k: int = 3,
) -> int | None:
    """Devuelve el score de la K-ésima mejor penca del pool (cutoff de top-K).

    Si hay menos de K pencas o falla la API, devuelve None.
    """
    if not api_base_url or not api_key:
        return None
    try:
        with httpx.Client(
            timeout=10.0,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as c:
            r = c.get(f"{api_base_url.rstrip('/')}/leaderboard")
        if r.status_code != 200:
            return None
        entries = r.json().get("entries", [])
        if len(entries) < k:
            return None
        # entries vienen ordenadas por rank (1° primero)
        return int(entries[k - 1].get("points_total", 0))
    except Exception as e:
        log.warning("fetch_pool_top_k_threshold falló: %s", e)
        return None


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


def optimal_assignment_p_top_k(
    picks_in_strategy_order: list[dict],
    penca_ids: list[int],
    grid: Any,
    pencas_standings: dict[int, dict[str, Any]] | None,
    pool_top_k_threshold: int | None,
    pool_q: Any,
    points_rule=None,
) -> tuple[list[tuple[int, dict, int | None]], dict[str, Any]]:
    """Asignación óptima MAXIMIZANDO P(max_de_mis_5_finales ≥ cutoff_top-K_del_pool).

    Args:
        pool_top_k_threshold: el score que tiene la K-ésima mejor penca del pool ahora.
            Si None → fallback a E[max].
        pool_q: distribución del pick del jugador típico del pool (numpy 2D).

    Returns:
        (assignment_list, metadata):
            - assignment_list = lista de (penca_id, pick, rank) — la asignación óptima.
            - metadata = {"objective": "p_top_k" | "e_max", "p_top_k_value": float, "threshold": int}
              para mostrar en notifs.
    """
    from itertools import permutations
    import numpy as np
    from src.model.poisson import jmlm_points
    if points_rule is None:
        points_rule = jmlm_points

    if len(penca_ids) != 5 or len(picks_in_strategy_order) != 5:
        raise ValueError("5 picks/pencas requeridos")

    # Sin pool data → fallback a E[max]
    if pool_top_k_threshold is None:
        assignment = optimal_assignment(
            picks_in_strategy_order, penca_ids, grid, pencas_standings, points_rule,
        )
        return assignment, {"objective": "e_max", "p_top_k_value": None, "threshold": None}

    current_pts = {
        pid: int(pencas_standings.get(pid, {}).get("points_total", 0)) if pencas_standings else 0
        for pid in penca_ids
    }
    n = grid.shape[0]

    # Tabla de puntos por pick y outcome
    pts_table = np.zeros((5, n, n), dtype=np.int8)
    for pi, pick in enumerate(picks_in_strategy_order):
        ps = (int(pick["score"][0]), int(pick["score"][1]))
        for gL in range(n):
            for gV in range(n):
                pts_table[pi, gL, gV] = points_rule(ps, (gL, gV))

    # Modal del pool: lo que pica el jugador típico
    flat_q = pool_q.flatten()
    modal_idx = int(np.argmax(flat_q))
    modal_pick = (modal_idx // n, modal_idx % n)

    # Gain del jugador top-K asumiendo que pica modal (aproximación)
    modal_gain = np.zeros((n, n), dtype=np.int8)
    for gL in range(n):
        for gV in range(n):
            modal_gain[gL, gV] = points_rule(modal_pick, (gL, gV))

    # Threshold por outcome
    pool_threshold_per_outcome = pool_top_k_threshold + modal_gain.astype(np.int16)

    best_perm = None
    best_objective = -1.0
    current_pts_arr = np.array([current_pts[pid] for pid in penca_ids], dtype=np.int16)

    for perm in permutations(range(5)):
        per_penca_pts = pts_table[list(perm)].astype(np.int16)
        finals = per_penca_pts + current_pts_arr.reshape(5, 1, 1)
        max_per_outcome = finals.max(axis=0)  # (n, n)
        in_top_k = (max_per_outcome >= pool_threshold_per_outcome).astype(float)
        obj = float((grid * in_top_k).sum())  # P(in top-K)
        if obj > best_objective:
            best_objective = obj
            best_perm = perm

    if best_perm is None or best_objective == 0.0:
        # P(top-K) = 0 para toda permutación → fallback a E[max] (sigue siendo informativo)
        log.info("P(top-K)=0 con cualquier asignación, fallback a E[max]")
        assignment = optimal_assignment(
            picks_in_strategy_order, penca_ids, grid, pencas_standings, points_rule,
        )
        return assignment, {
            "objective": "e_max (P(top-K)=0)",
            "p_top_k_value": 0.0,
            "threshold": pool_top_k_threshold,
        }

    pencas_by_score = sorted(penca_ids, key=lambda pid: -current_pts[pid])
    rank_by_penca = {pid: i + 1 for i, pid in enumerate(pencas_by_score)}

    return [
        (penca_ids[i], picks_in_strategy_order[best_perm[i]], rank_by_penca[penca_ids[i]])
        for i in range(5)
    ], {
        "objective": "p_top_k",
        "p_top_k_value": best_objective,
        "threshold": pool_top_k_threshold,
    }


def greedy_assignment(
    candidate_picks: list[dict],
    penca_ids: list[int],
    grid: Any,
    pencas_standings: dict[int, dict[str, Any]] | None = None,
    pool_top_k_threshold: int | None = None,
    pool_q: Any = None,
    points_rule=None,
) -> tuple[list[tuple[int, dict, int | None]], dict[str, Any]]:
    """Asignación VORAZ de N pencas sobre un menú de candidatos, CON repetición.

    Generaliza `optimal_assignment_p_top_k` a cualquier N (la enumeración de N!
    permutaciones es intratable para N=15). En vez de eso, procesa las pencas en orden
    de ranking (líder primero) y a cada una le asigna el candidato que MÁS aumenta el
    objetivo marginal:

        objetivo = P(max_i (puntos_actuales_i + puntos(pick_i, ω)) ≥ cutoff_top-K(ω))

    con E[max] como desempate y fallback. Como el objetivo es la probabilidad de la UNIÓN
    (que al menos una penca supere el cutoff), el voraz produce solo exposición +
    decorrelación: la primera penca toma el ancla EV, las siguientes cubren los outcomes
    donde las anteriores todavía no ganan — repitiendo el ancla cuando repetir es óptimo.

    Args:
        candidate_picks: menú de picks distintas (dicts con "score"). Típicamente de
            `portfolio.generate_candidates(...)` vía `picks_to_dicts(...)`.
        pool_top_k_threshold: score de la K-ésima penca del pool. Si None → objetivo E[max].
        pool_q: distribución del pick del jugador típico (para estimar el cutoff por outcome).

    Returns:
        (assignment_list, meta) — assignment_list = [(penca_id, pick, rank), ...] de largo N.
        meta incluye "objective", "p_top_k_value", "threshold" y "exposure" (conteo por marcador).

    Cost: O(N · K · n²) — trivial incluso con N=15, K=8, n=8.
    """
    import numpy as np
    from collections import Counter
    from src.model.poisson import jmlm_points
    if points_rule is None:
        points_rule = jmlm_points

    if not candidate_picks or not penca_ids:
        return [], {"objective": "none", "p_top_k_value": None, "threshold": pool_top_k_threshold}

    n = grid.shape[0]
    K = len(candidate_picks)

    # Tabla de puntos por candidato y outcome
    pts_tables = []
    for pick in candidate_picks:
        ps = (int(pick["score"][0]), int(pick["score"][1]))
        t = np.zeros((n, n))
        for gL in range(n):
            for gV in range(n):
                t[gL, gV] = points_rule(ps, (gL, gV))
        pts_tables.append(t)

    current_pts = {
        pid: int(pencas_standings.get(pid, {}).get("points_total", 0)) if pencas_standings else 0
        for pid in penca_ids
    }

    use_top_k = pool_top_k_threshold is not None and pool_q is not None
    if use_top_k:
        flat_q = np.asarray(pool_q).flatten()
        modal_idx = int(np.argmax(flat_q))
        modal_pick = (modal_idx // n, modal_idx % n)
        modal_gain = np.zeros((n, n))
        for gL in range(n):
            for gV in range(n):
                modal_gain[gL, gV] = points_rule(modal_pick, (gL, gV))
        threshold_per_outcome = pool_top_k_threshold + modal_gain

    def objective(best_final: np.ndarray) -> tuple[float, float]:
        e_max = float((grid * best_final).sum())
        if use_top_k:
            p = float((grid * (best_final >= threshold_per_outcome)).sum())
            return (p, e_max)
        return (e_max, 0.0)

    # Procesar pencas: líder (más puntos) primero; desempate por id ascendente
    order = sorted(penca_ids, key=lambda pid: (-current_pts[pid], pid))
    rank_by_penca = {pid: i + 1 for i, pid in enumerate(order)}

    NEG = -1e9
    best_final = np.full((n, n), NEG)  # max sobre pencas ya asignadas de (pts_actuales + pts(pick, ω))
    assigned: dict[int, int] = {}

    for pid in order:
        base = current_pts[pid]
        best_c = None
        best_key = None
        best_bf = None
        for c in range(K):
            cand_final = base + pts_tables[c]
            bf = np.maximum(best_final, cand_final)
            # tiebreak: menor índice de candidato (prioridad canónica: ev primero)
            key = objective(bf) + (-c,)
            if best_key is None or key > best_key:
                best_key, best_c, best_bf = key, c, bf
        assigned[pid] = best_c
        best_final = best_bf

    p_val = objective(best_final)[0] if use_top_k else None

    # Fallback: si P(top-K)=0 con cualquier asignación, reasignar por E[max] (más informativo)
    if use_top_k and (p_val is None or p_val == 0.0):
        res2, meta2 = greedy_assignment(
            candidate_picks, penca_ids, grid, pencas_standings,
            pool_top_k_threshold=None, pool_q=None, points_rule=points_rule,
        )
        meta2["objective"] = "e_max (P(top-K)=0)"
        meta2["threshold"] = pool_top_k_threshold
        return res2, meta2

    result = [(pid, candidate_picks[assigned[pid]], rank_by_penca[pid]) for pid in penca_ids]

    exposure = Counter(f'{pick["score"][0]}-{pick["score"][1]}' for _, pick, _ in result)
    meta = {
        "objective": "p_top_k" if use_top_k else "e_max",
        "p_top_k_value": p_val,
        "threshold": pool_top_k_threshold,
        "exposure": dict(exposure),
    }
    return result, meta


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

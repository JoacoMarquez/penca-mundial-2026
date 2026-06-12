"""Asignación adaptativa de N pencas a un menú de candidatos según el ranking actual.

`greedy_assignment` reparte las N pencas sobre las picks candidatas (con repetición),
maximizando P(al menos una penca supere el cutoff top-K del pool). La penca líder recibe
el ancla conservadora; las rezagadas reciben desvíos que cubren outcomes aún no ganados.

También expone helpers para leer el leaderboard real de la penca (standings + cutoff).
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


def greedy_assignment(
    candidate_picks: list[dict],
    penca_ids: list[int],
    grid: Any,
    pencas_standings: dict[int, dict[str, Any]] | None = None,
    pool_top_k_threshold: int | None = None,
    pool_q: Any = None,
    points_rule=None,
    horizon_premium: float = 0.0,
) -> tuple[list[tuple[int, dict, int | None]], dict[str, Any]]:
    """Asignación VORAZ de N pencas sobre un menú de candidatos, CON repetición.

    Evita la enumeración de N! permutaciones (intratable para N=15). Procesa las pencas en orden
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
        horizon_premium: puntos extra sobre el cutoff actual para proyectar dónde va a
            estar el corte al FINAL del torneo (β·√(partidos restantes), backtest: β=2.0
            → +6pp de P(ganar) vs cutoff miope). Temprano sube el listón y fuerza
            agresividad; tiende a 0 en las últimas fechas. 0 = comportamiento miope.

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
        threshold_per_outcome = pool_top_k_threshold + horizon_premium + modal_gain

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

    # Fallback: si P(top-K)=0 con cualquier asignación, reasignar por E[max].
    # Con horizon_premium alto (principio del torneo, listón proyectado inalcanzable en
    # un partido) este fallback es el comportamiento ESPERADO: E[max] de la unión
    # diversifica, y P(top-K) toma el control a medida que el premium se achica.
    if use_top_k and (p_val is None or p_val == 0.0):
        res2, meta2 = greedy_assignment(
            candidate_picks, penca_ids, grid, pencas_standings,
            pool_top_k_threshold=None, pool_q=None, points_rule=points_rule,
        )
        meta2["objective"] = "e_max (P(top-K)=0)"
        meta2["threshold"] = pool_top_k_threshold
        meta2["horizon_premium"] = horizon_premium
        return res2, meta2

    result = [(pid, candidate_picks[assigned[pid]], rank_by_penca[pid]) for pid in penca_ids]

    exposure = Counter(f'{pick["score"][0]}-{pick["score"][1]}' for _, pick, _ in result)
    meta = {
        "objective": "p_top_k" if use_top_k else "e_max",
        "p_top_k_value": p_val,
        "threshold": pool_top_k_threshold,
        "horizon_premium": horizon_premium,
        "exposure": dict(exposure),
    }
    return result, meta



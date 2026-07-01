"""Asignación adaptativa de N pencas a un menú de candidatos según el ranking actual.

`greedy_assignment` reparte las N pencas sobre las picks candidatas (con repetición),
maximizando P(al menos una penca supere el cutoff top-K del pool). Las rezagadas reciben
desvíos que cubren outcomes aún no ganados. Con `protect_theta`, las pencas que YA están
sobre el corte (la líder) se eximen del horizon_premium → vuelven al ancla EV / mirror-chalk
para blindar la ventaja en vez de perseguir un corte proyectado que su propio avance ya
alcanza (sin él, el premium empujaba hasta a la líder del pool a anti-favoritos).

También expone helpers para leer el leaderboard real de la penca (standings + cutoff).
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _max_stale_seconds() -> float | None:
    """Techo de antigüedad para servir un leaderboard cacheado a la asignación en vivo.

    Un corte de hace un partido sigue siendo mucho mejor que None (que apaga el objetivo
    tail-max y cae a E[max] miope). Pero un leaderboard de días atrás sí desvirtúa el corte,
    así que lo acotamos. Default 24h: cubre el hueco normal entre partidos de eliminatorias
    (de una noche para otra los standings no cambian) pero descarta datos de varios días si
    la caída se prolonga. PENCA_LEADERBOARD_MAX_STALE_H (≤0 o inválido = sin techo).
    """
    import os
    raw = os.environ.get("PENCA_LEADERBOARD_MAX_STALE_H", "24").strip()
    try:
        h = float(raw)
    except ValueError:
        return None
    return h * 3600 if h > 0 else None


def fetch_pool_top_k_threshold(
    api_base_url: str,
    api_key: str,
    k: int = 3,
) -> int | None:
    """Devuelve el score de la K-ésima mejor penca del pool (cutoff de top-K).

    Si hay menos de K pencas o no hay leaderboard (ni fresco ni cacheado), devuelve None.
    Ante caída de la API usa el último leaderboard bueno de disco (stale) acotado por
    PENCA_LEADERBOARD_MAX_STALE_H, para no perder el objetivo tail-max por un blip.
    """
    if not api_base_url or not api_key:
        return None
    from src.utils.leaderboard import fetch_leaderboard
    res = fetch_leaderboard(api_base_url, api_key, timeout=8.0, max_stale_seconds=_max_stale_seconds())
    entries = res["entries"]
    if len(entries) < k:
        return None
    # El cutoff es un umbral de puntos: la K-ésima penca por puntaje. Ordenamos por si el
    # fallback (cache/snapshot) no viene rankeado como sí lo trae la API.
    entries = sorted(entries, key=lambda e: -int(e.get("points_total", 0)))
    if res["stale"]:
        log.warning(
            "pool cutoff desde leaderboard STALE (%.0f min, %s)",
            (res.get("age_seconds") or 0) / 60, res.get("error"),
        )
    return int(entries[k - 1].get("points_total", 0))


def fetch_my_pencas_standings(
    api_base_url: str,
    api_key: str,
    my_penca_ids: list[int],
) -> dict[int, dict[str, Any]]:
    """GET /leaderboard y devuelve solo las entries de mis pencas.

    Retorna: {penca_id: {"rank": N, "points_total": X, ...}, ...}.
    Ante caída de la API usa el último leaderboard bueno de disco (stale) acotado por
    PENCA_LEADERBOARD_MAX_STALE_H; si no hay nada usable devuelve {} (caller decide).
    """
    if not api_base_url or not api_key or not my_penca_ids:
        return {}
    from src.utils.leaderboard import fetch_leaderboard
    res = fetch_leaderboard(api_base_url, api_key, timeout=8.0, max_stale_seconds=_max_stale_seconds())
    entries = res["entries"]
    if entries and res["stale"]:
        log.warning("standings desde leaderboard STALE (%.0f min)", (res.get("age_seconds") or 0) / 60)
    want = set(my_penca_ids)
    return {
        int(e["penca_id"]): e
        for e in entries
        if int(e.get("penca_id", 0)) in want
    }


def greedy_assignment(
    candidate_picks: list[dict],
    penca_ids: list[int],
    grid: Any,
    pencas_standings: dict[int, dict[str, Any]] | None = None,
    pool_top_k_threshold: int | None = None,
    pool_q: Any = None,
    points_rule=None,
    horizon_premium: float = 0.0,
    protect_theta: float | None = None,
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

    def _is_protected(base: float) -> bool:
        # Una penca está "a salvo" si su puntaje ya supera el corte de HOY por un colchón
        # θ·premium. protect_theta es la perilla: θ=0 protege apenas pasás el corte (sobre-
        # protege → resigna el #1), θ=1 sólo contra el corte proyectado (≈ no protege).
        # Backtest sembrado (pool 436, E[premio] con pago 20k/10k/5k): θ≈0.4 maximiza la
        # plata en las 3 calibraciones de campo (+22%). Ver scripts/backtest_protect_leader.py.
        return (
            protect_theta is not None
            and base >= pool_top_k_threshold + protect_theta * horizon_premium
        )

    def _threshold_for(base: float):
        # Premium EFECTIVO por penca. Protegida → 0 (mirror-chalk/ancla EV, minimiza varianza
        # vs el campo, blinda la ventaja). Perseguidora → premium completo: IDÉNTICA al status
        # quo, así el β=2.0 validado no se reabre. Sin protect_theta → todas usan el premium
        # completo (comportamiento previo bit-a-bit).
        premium_eff = 0.0 if _is_protected(base) else horizon_premium
        return pool_top_k_threshold + premium_eff + modal_gain

    def objective(best_final: np.ndarray, thr_outcome) -> tuple[float, float]:
        e_max = float((grid * best_final).sum())
        if use_top_k:
            p = float((grid * (best_final >= thr_outcome)).sum())
            return (p, e_max)
        return (e_max, 0.0)

    # Procesar pencas: líder (más puntos) primero; desempate por id ascendente
    order = sorted(penca_ids, key=lambda pid: (-current_pts[pid], pid))
    rank_by_penca = {pid: i + 1 for i, pid in enumerate(order)}

    NEG = -1e9
    best_final = np.full((n, n), NEG)  # max sobre pencas ya asignadas de (pts_actuales + pts(pick, ω))
    assigned: dict[int, int] = {}
    usage = [0] * K  # cuántas pencas ya tomaron cada candidato (para desempatar hacia diversidad)

    for pid in order:
        base = current_pts[pid]
        thr_outcome = _threshold_for(base) if use_top_k else None
        best_c = None
        best_key = None
        best_bf = None
        for c in range(K):
            cand_final = base + pts_tables[c]
            bf = np.maximum(best_final, cand_final)
            obj = objective(bf, thr_outcome)
            # Desempate: cuando dos candidatos rinden igual en el objetivo de la UNIÓN
            # (típico de pencas rezagadas ya dominadas — antes amontonaba todo en el
            # candidato 0 = chalk), preferir el candidato MENOS usado para repartir
            # exposición. La igualdad se mide redondeada (ruido de coma flotante).
            # Último desempate: menor índice (prioridad canónica: ev primero).
            key = (round(obj[0], 9), round(obj[1], 9), -usage[c], -c)
            if best_key is None or key > best_key:
                best_key, best_c, best_bf = key, c, bf
        assigned[pid] = best_c
        best_final = best_bf
        usage[best_c] += 1

    # p_val de referencia: umbral de la líder (mayor base) para un número consistente.
    p_val = (
        objective(best_final, _threshold_for(max(current_pts.values())))[0]
        if use_top_k else None
    )

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
    protected = (
        [pid for pid in penca_ids if use_top_k and _is_protected(current_pts[pid])]
        if protect_theta is not None else []
    )
    meta = {
        "objective": "p_top_k" if use_top_k else "e_max",
        "p_top_k_value": p_val,
        "threshold": pool_top_k_threshold,
        "horizon_premium": horizon_premium,
        "protect_theta": protect_theta,
        "protected_pencas": protected,
        "exposure": dict(exposure),
    }
    return result, meta



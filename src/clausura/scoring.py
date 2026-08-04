"""Kernel de puntos de la Penca Supermatch — Torneo Clausura 2026 (Art. 6 del reglamento).

Puntuación ADITIVA por partido (a diferencia de la JMLM, que es por tiers excluyentes):

    Ganador del partido:   2 puntos
    Diferencia de goles:   1 punto   (solo si el ganador fue acertado; en empates
                                      acertados la diferencia 0 se acierta por definición)
    Goles del local:       1 punto   (independiente del ganador)
    Goles del visitante:   1 punto   (independiente del ganador)
    Resultado exacto:      3 puntos
    Máximo por partido:    8 puntos (2+1+1+1+3)

Un partido "preferencial" por fecha (estrella) vale doble → máximo 16.

Consecuencia estratégica clave: acertar los goles de UN equipo paga aunque erres el
ganador (ej.: pick 2-0, resultado 2-3 → 1 punto). El argmax de E[pts] acá se corre
hacia marcadores cuyos goles por lado son modales en las MARGINALES de la grilla,
no solo hacia la celda modal conjunta como en la regla JMLM.
"""

from __future__ import annotations

import numpy as np


def supermatch_points(
    pick: tuple[int, int],
    actual: tuple[int, int],
    preferencial: bool = False,
) -> int:
    """Puntos de un pronóstico según Art. 6 del reglamento de la Penca Supermatch."""
    pgL, pgV = pick
    agL, agV = actual

    pts = 0

    pick_winner = "H" if pgL > pgV else ("A" if pgL < pgV else "D")
    actual_winner = "H" if agL > agV else ("A" if agL < agV else "D")
    winner_ok = pick_winner == actual_winner

    if winner_ok:
        pts += 2
        if (pgL - pgV) == (agL - agV):
            pts += 1  # diferencia de goles, condicionada a ganador acertado

    if pgL == agL:
        pts += 1  # goles del local (incondicional)
    if pgV == agV:
        pts += 1  # goles del visitante (incondicional)

    if (pgL, pgV) == (agL, agV):
        pts += 3  # resultado exacto

    return pts * 2 if preferencial else pts


def expected_points_grid(
    pick: tuple[int, int],
    grid: np.ndarray,
    preferencial: bool = False,
) -> float:
    """E[pts | pick] bajo la grilla P[gL, gV], en forma cerrada por componentes.

    E = 2·P(ganador del pick) + 1·P(misma diferencia ∧ ganador ok)
        + 1·P(gL = pick_L) + 1·P(gV = pick_V) + 3·P(exacto)
    """
    n = grid.shape[0]
    pgL, pgV = pick
    if not (0 <= pgL < n and 0 <= pgV < n):
        raise ValueError(f"pick {pick} fuera de la grilla {n}x{n}")

    gL_idx, gV_idx = np.indices((n, n))
    diff = gL_idx - gV_idx
    pick_diff = pgL - pgV

    if pick_diff > 0:
        winner_mask = diff > 0
    elif pick_diff < 0:
        winner_mask = diff < 0
    else:
        winner_mask = diff == 0

    e = 0.0
    e += 2.0 * grid[winner_mask].sum()
    e += 1.0 * grid[winner_mask & (diff == pick_diff)].sum()
    e += 1.0 * grid[pgL, :].sum()   # goles local acertados
    e += 1.0 * grid[:, pgV].sum()   # goles visitante acertados
    e += 3.0 * grid[pgL, pgV]

    return e * 2.0 if preferencial else e


def rank_picks(
    grid: np.ndarray,
    preferencial: bool = False,
    top_k: int | None = None,
) -> list[tuple[tuple[int, int], float]]:
    """Todos los picks de la grilla ordenados por E[pts] descendente.

    Devuelve [((gL, gV), ev), ...]. Con top_k recorta la lista.
    """
    n = grid.shape[0]
    scored = [
        ((gL, gV), expected_points_grid((gL, gV), grid, preferencial))
        for gL in range(n)
        for gV in range(n)
    ]
    scored.sort(key=lambda t: -t[1])
    return scored[:top_k] if top_k else scored


def best_pick(grid: np.ndarray, preferencial: bool = False) -> tuple[tuple[int, int], float]:
    """argmax E[pts] sobre la grilla."""
    return rank_picks(grid, preferencial, top_k=1)[0]

"""Kernel de puntos de la penca LUB (Art. 6 del reglamento, igual en 25/26 y 26/27).

    Ganador del partido:  2 puntos
    Margen de victoria:   3 puntos, solo si el ganador fue acertado, por banda:
                          1-5, 6-10, 11-15, 16-20, +20
    Resultado exacto:     10 puntos
    Campeón / goleador uruguayo: 25 cada uno (fuera de este módulo)

Todo pronóstico es un marcador (local, visitante), pero el puntaje depende solo de
(ganador, banda) salvo el exacto, que en básquet tiene probabilidad ~1e-3 y aporta
~0,01 pts esperados. Por eso cada pick se reduce a una de 10 CLASES:

    clase = lado·5 + banda     lado 0 = gana el local, 1 = gana el visitante
                               banda 0..4 = 1-5, 6-10, 11-15, 16-20, 21+

y el kernel es una matriz 10×10 de puntos (pick × resultado). En básquet no hay
empate: el pronóstico con empate no es válido y la prórroga cuenta (Art. 11).
"""

from __future__ import annotations

import numpy as np

N_BANDAS = 5
N_CLASES = 2 * N_BANDAS
BANDAS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 200))
PTS_GANADOR, PTS_MARGEN, PTS_EXACTO = 2, 3, 10


def banda(margen_abs: int) -> int:
    if margen_abs <= 0:
        raise ValueError("en básquet no hay empate: margen debe ser ≥ 1")
    return min((margen_abs - 1) // 5, N_BANDAS - 1)


def clase(pts_local: int, pts_visitante: int) -> int:
    m = pts_local - pts_visitante
    return (0 if m > 0 else N_BANDAS) + banda(abs(m))


def lado(c: int) -> int:
    return c // N_BANDAS


def puntos(pick: tuple[int, int], real: tuple[int, int], preferencial: bool = False) -> int:
    """Puntos de un pronóstico (marcador) contra el resultado real."""
    if pick[0] == pick[1]:
        return 0
    cp, cr = clase(*pick), clase(*real)
    pts = 0
    if lado(cp) == lado(cr):
        pts += PTS_GANADOR
        if cp == cr:
            pts += PTS_MARGEN
    if tuple(pick) == tuple(real):
        pts += PTS_EXACTO
    return pts * 2 if preferencial else pts


def kernel() -> np.ndarray:
    """K[pick, resultado] en puntos (sin exacto). 5 en la diagonal, 2 mismo lado."""
    K = np.zeros((N_CLASES, N_CLASES), dtype=np.int8)
    for p in range(N_CLASES):
        for r in range(N_CLASES):
            if lado(p) == lado(r):
                K[p, r] = PTS_GANADOR + (PTS_MARGEN if p == r else 0)
    return K


KERNEL = kernel()


def expected_points(probs: np.ndarray) -> np.ndarray:
    """E[pts | pick=c] para cada clase, dada la distribución de resultados (10,)."""
    return KERNEL.astype(float) @ probs


def marcador_de_clase(c: int, total_esperado: float, margen_modal: int | None = None) -> tuple[int, int]:
    """Marcador a cargar para una clase: margen representativo de la banda y total ≈ esperado.

    El exacto casi no pesa, pero conviene que el marcador sea plausible (margen modal
    dentro de la banda, total cerca de la media) para no regalar ese 0,1%.
    """
    lo, hi = BANDAS[c % N_BANDAS]
    m = margen_modal if margen_modal is not None and lo <= margen_modal <= hi else min(lo + 2, hi)
    t = int(round(total_esperado))
    if (t + m) % 2:       # local + visitante = t, local − visitante = m ⇒ misma paridad
        t += 1
    ganador, perdedor = (t + m) // 2, (t - m) // 2
    return (ganador, perdedor) if lado(c) == 0 else (perdedor, ganador)

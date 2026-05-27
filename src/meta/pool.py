"""Modelo del pool de competidores.

La penca tiene 150+ jugadores. La web sólo expone el ranking de puntos, no los picks individuales.
Tenemos que inferir Q(pick) — la distribución de qué predicen los otros — de manera indirecta.

Modelo (prior):
    El jugador típico es "chalk-with-popularity-bias":
    - Tiende a elegir el marcador modal del mercado (chalk).
    - Pero sesga hacia marcadores "populares" (1-0, 2-1, 2-0) porque esos son los que la
      experiencia/folclore dice "es lo más probable".

Hiperparámetros:
    chalk_strength: peso del prior basado en P del mercado. Si 1.0, el pool sigue exactamente al mercado.
    popular_score_bias: multiplicadores para scorelines populares. Mapeo (gL, gV) → factor.
    temperature: softmax temperature. Baja → todos pican el mismo top. Alta → distribución más plana.

Calibración online (después de cada jornada):
    Ranking-inversion: dado el leaderboard, inferir qué pick fue el mediano para cada partido
    y reajustar (chalk_strength, popular_score_bias) por mínimos cuadrados.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Sesgo de popularidad: factor multiplicativo sobre P(scoreline) para reflejar
# qué scorelines pica un humano típico de penca latinoamericana.
# Calibrado a ojo desde experiencia; calibrar con ranking-inversion después de jornada 1.
DEFAULT_POPULAR_SCORE_BIAS: dict[tuple[int, int], float] = {
    (1, 0): 1.8,
    (2, 0): 1.4,
    (2, 1): 1.6,
    (1, 1): 1.5,
    (0, 1): 1.3,
    (1, 2): 1.3,
    (0, 0): 0.8,   # subponderado, la gente no le gusta jugar 0-0
    (3, 0): 0.9,
    (0, 3): 0.7,
    (3, 1): 1.0,
    (1, 3): 0.9,
    (2, 2): 1.0,
    (3, 2): 0.7,
    (2, 3): 0.6,
}


@dataclass
class PoolModelConfig:
    chalk_strength: float = 0.7
    popular_score_bias: dict[tuple[int, int], float] = field(default_factory=lambda: dict(DEFAULT_POPULAR_SCORE_BIAS))
    temperature: float = 1.0
    default_bias: float = 1.0  # factor para scorelines no listados en popular_score_bias


def pool_pick_distribution(
    market_grid: np.ndarray,
    config: PoolModelConfig | None = None,
) -> np.ndarray:
    """Distribución Q[gL, gV] de qué predice un jugador típico del pool.

    Combina P del mercado con el bias de popularidad, aplica softmax con temperatura, normaliza.
    """
    if config is None:
        config = PoolModelConfig()

    n = market_grid.shape[0]
    # Bias matrix
    bias = np.full_like(market_grid, config.default_bias)
    for (gL, gV), factor in config.popular_score_bias.items():
        if gL < n and gV < n:
            bias[gL, gV] = factor

    # Score logarítmico: log P(market) * chalk_strength + log(bias)
    # Evitar log(0) con ε
    eps = 1e-12
    score = config.chalk_strength * np.log(market_grid + eps) + np.log(bias + eps)
    score = score / max(config.temperature, 1e-6)

    # softmax para obtener distribución
    score = score - score.max()  # estabilidad numérica
    exp_score = np.exp(score)
    q = exp_score / exp_score.sum()
    return q


def expected_pool_points(
    market_grid: np.ndarray,
    pool_pick_distribution_q: np.ndarray,
    points_rule,
) -> np.ndarray:
    """E[points | scoreline_actual] del jugador típico — pesado por su pick distribution Q.

    Resultado: matrix shape (n, n) donde entry [agL, agV] = E_{p~Q}[points(p, (agL, agV))].
    """
    n = market_grid.shape[0]
    result = np.zeros((n, n))

    for agL in range(n):
        for agV in range(n):
            # Para este outcome real, calcular cuántos puntos saca un pick promedio del pool
            pts = 0.0
            for pgL in range(n):
                for pgV in range(n):
                    pts += pool_pick_distribution_q[pgL, pgV] * points_rule((pgL, pgV), (agL, agV))
            result[agL, agV] = pts

    return result

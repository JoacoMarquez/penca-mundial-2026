"""Modelo del pool de la Penca Supermatch (Capa 5 adaptada).

Igual que en la JMLM, no vemos los picks individuales — pero acá el ranking público
expone `cantResultadosExactos` por participación, que es un canal de calibración
mucho más directo que la ranking-inversion a ciegas del Mundial: la tasa de exactos
del pool identifica casi sola qué tan concentrada está Q en los marcadores modales.

Prior: chalk con sesgo a marcadores "de penca". La novedad respecto del Mundial es
que en la Primera uruguaya el prior humano y la realidad divergen fuerte en un punto
concreto — el 0-0.

    Distribución REAL (598 partidos, 5 temporadas, 2024-2026):
        1-1 11.9% · 0-1 11.7% · 1-0 11.0% · 0-0 9.7% · 1-2 8.9% · 2-1 8.0% · 2-0 7.4%
        E[goles] = 1.28 local + 1.11 visitante = 2.39 (liga de pocos goles)
        empates 27.3% · corr(gL,gV) = 0.00 → Poisson independiente alcanza

    El 0-0 es el CUARTO marcador más frecuente (casi 1 de cada 10 partidos) y es
    justamente el que el folclore de penca descarta. Ese hueco es la asimetría
    explotable más grande del torneo, y por eso el bias de 0-0 acá es un parámetro
    de primera clase y no un detalle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.clausura.economics import N_SCORES, flatten_grid, index_score

# Sesgo de popularidad del pencista uruguayo. Parte del prior del Mundial y ajusta
# a lo que se ve en esta liga: marcadores bajos dominan, pero el 0-0 sigue siendo
# socialmente impopular (nadie quiere "jugar al empate sin goles") pese a ser 9.7% real.
DEFAULT_POPULAR_BIAS: dict[tuple[int, int], float] = {
    (1, 0): 1.8,
    (2, 1): 1.6,
    (1, 1): 1.6,
    (2, 0): 1.4,
    (0, 1): 1.3,
    (1, 2): 1.3,
    (0, 0): 0.55,   # el hueco: real 9.7%, pero el pencista lo evita
    (3, 0): 0.9,
    (3, 1): 0.9,
    (2, 2): 0.9,
    (0, 2): 0.9,
    (0, 3): 0.6,
    (3, 2): 0.6,
    (2, 3): 0.5,
}


@dataclass
class PoolConfig:
    chalk_strength: float = 1.0
    temperature: float = 1.0
    default_bias: float = 0.8
    popular_bias: dict[tuple[int, int], float] = field(
        default_factory=lambda: dict(DEFAULT_POPULAR_BIAS)
    )


def pool_distribution(grid: np.ndarray, cfg: PoolConfig | None = None) -> np.ndarray:
    """Q[pick_idx]: qué juega el pencista típico en este partido. Vector de N_SCORES."""
    cfg = cfg or PoolConfig()
    p_market = flatten_grid(grid)

    bias = np.full(N_SCORES, cfg.default_bias)
    for (gL, gV), f in cfg.popular_bias.items():
        idx = gL * 6 + gV
        if idx < N_SCORES:
            bias[idx] = f

    eps = 1e-12
    score = cfg.chalk_strength * np.log(p_market + eps) + np.log(bias + eps)
    score /= max(cfg.temperature, 1e-6)
    score -= score.max()
    q = np.exp(score)
    return q / q.sum()


def expected_exact_rate(grids: list[np.ndarray], cfg: PoolConfig | None = None) -> float:
    """Tasa esperada de resultados exactos del pencista típico bajo Q.

    Es el observable que publica el ranking (`cantResultadosExactos` / partidos jugados),
    y por lo tanto la función que invertimos para calibrar.
    """
    cfg = cfg or PoolConfig()
    rates = []
    for g in grids:
        q = pool_distribution(g, cfg)
        p = flatten_grid(g)
        rates.append(float((q * p).sum()))
    return float(np.mean(rates)) if rates else 0.0


def calibrate_from_exact_rate(
    grids: list[np.ndarray],
    observed_exact_rate: float,
    base: PoolConfig | None = None,
    grid_temps: tuple[float, ...] = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0),
) -> PoolConfig:
    """Ajusta `temperature` para que la tasa de exactos predicha matchee la observada.

    Temperatura baja = pool concentrado en el marcador modal (más exactos); alta = pool
    disperso (menos exactos). Es la calibración online del Art. 6: después de cada fecha
    leemos el ranking, calculamos exactos/partidos y re-ajustamos.
    """
    base = base or PoolConfig()
    best, best_err = base, float("inf")
    for t in grid_temps:
        cfg = PoolConfig(
            chalk_strength=base.chalk_strength,
            temperature=t,
            default_bias=base.default_bias,
            popular_bias=dict(base.popular_bias),
        )
        err = abs(expected_exact_rate(grids, cfg) - observed_exact_rate)
        if err < best_err:
            best, best_err = cfg, err
    return best


def observed_exact_rate_from_ranking(rows, partidos_jugados: int) -> float | None:
    """Tasa media de exactos del pool desde el ranking público.

    `rows` son RankingRow de src.clausura.api. Devuelve None si todavía no se jugó nada.
    """
    if partidos_jugados <= 0 or not rows:
        return None
    exactos = [r.cant_resultados_exactos for r in rows]
    return float(np.mean(exactos)) / partidos_jugados


def top_pool_picks(grid: np.ndarray, cfg: PoolConfig | None = None, k: int = 5):
    """[(marcador, Q), ...] — los k marcadores más jugados por el pool. Para auditar."""
    q = pool_distribution(grid, cfg)
    order = np.argsort(-q)[:k]
    return [(index_score(int(i)), float(q[i])) for i in order]

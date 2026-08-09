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

import logging
from dataclasses import dataclass, field

import numpy as np

from src.clausura.economics import N_SCORES, flatten_grid, index_score

log = logging.getLogger(__name__)

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
    # Refleja `popular_bias` cuando el favorito es el visitante. Ver pool_distribution.
    # Se puede apagar para reproducir el comportamiento previo al 2026-08-09.
    orientar_al_favorito: bool = True


def _favorito_es_visitante(p_market: np.ndarray, margen: float = 0.0) -> bool:
    """¿El mercado da más chances al visitante que al local?

    `margen` exige una ventaja mínima para reflejar; con 0 alcanza con empatar. En
    partidos casi parejos da igual hacia qué lado se refleje —el sesgo es casi
    simétrico ahí— así que no hace falta una zona muerta.
    """
    idx = np.arange(N_SCORES)
    gl, gv = idx // 6, idx % 6
    p_local = float(p_market[gl > gv].sum())
    p_visita = float(p_market[gl < gv].sum())
    return p_visita > p_local + margen


def pool_distribution(grid: np.ndarray, cfg: PoolConfig | None = None) -> np.ndarray:
    """Q[pick_idx]: qué juega el pencista típico en este partido. Vector de N_SCORES.

    El sesgo de popularidad se aplica ORIENTADO AL FAVORITO, no al local. La tabla
    está escrita en clave local-favorito (1-0 alto, 0-1 bajo) porque el favorito
    suele ser local; cuando NO lo es, se refleja. Sin reflejar, en esos partidos el
    modelo empuja la Q hacia marcadores de local justo donde el pool hace lo
    contrario, y se equivoca del lado caro: cree que hay compañía donde no la hay.

    Medido el 2026-08-09 sobre los 4 partidos cerrados de la Fecha 1 (2.739 picks
    reales de 718 participaciones). En Torque–Peñarol, con Peñarol favorito de
    visita, el sesgo empírico del 1-0 es 0,07 y el modelo le asignaba 1,58 — un
    factor 20 en la dirección equivocada. Reorientar estabiliza el sesgo entre
    partidos: el 1-0 pasa de {0,07 · 1,38 · 2,02 · 2,05} a
    {0,92 · 1,38 · 2,02 · 2,05}, con coeficiente de variación 0,30.

    Es una corrección ESTRUCTURAL, no un parámetro ajustado a esos 4 partidos: no
    agrega grados de libertad, solo aplica la tabla en el eje correcto.
    """
    cfg = cfg or PoolConfig()
    p_market = flatten_grid(grid)

    bias = np.full(N_SCORES, cfg.default_bias)
    for (gL, gV), f in cfg.popular_bias.items():
        idx = gL * 6 + gV
        if idx < N_SCORES:
            bias[idx] = f

    if cfg.orientar_al_favorito and _favorito_es_visitante(p_market):
        bias = bias.reshape(6, 6).T.reshape(-1).copy()

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


def observed_exact_rate_from_ranking(
    rows, partidos_jugados: int, mis_numeros: set[int] | None = None,
) -> float | None:
    """Tasa media de exactos del pool desde el ranking público.

    `rows` son RankingRow de src.clausura.api. Devuelve None si todavía no se jugó nada.
    `mis_numeros` excluye nuestras propias participaciones: el observable calibra a
    los RIVALES, y nuestras 12 planillas diversificadas sesgarían la tasa.

    OJO — `cantResultadosExactos` NO se actualiza en vivo: lo liquida el mismo job
    que llena `puntosPorFecha`, al cierre de la fecha. Medido el 2026-08-08: con
    Cerro Largo 1-1 Juventud ya FINALIZADO y sus puntos sumados al ranking, 146 de
    las 692 participaciones tenían 8 puntos —y 8 en un partido normal SOLO se sacan
    con el marcador exacto (2+1+1+1+3, Art. 6)— pero el contador de exactos venía en
    0 para las 692. Tomarlo literal da tasa 0 → calibra al pool MÁS disperso (T=3.0)
    cuando el pool real estaba en el más concentrado (T=0.5): exactamente al revés,
    y del lado que nos hace jugar chalk y repartir el premio con 146 personas.
    Por eso, si nadie registra exactos pero el pool ya sumó puntos, se devuelve None
    (sin calibrar, queda el prior) en vez de un 0 que miente.
    """
    if partidos_jugados <= 0 or not rows:
        return None
    mis = mis_numeros or set()
    rivales = [r for r in rows if getattr(r, "numero_participacion", None) not in mis]
    exactos = [r.cant_resultados_exactos for r in rivales]
    if not exactos:
        return None
    if max(exactos) == 0 and any(getattr(r, "puntos_totales", 0) > 0 for r in rivales):
        log.warning("el ranking da 0 exactos para las %d participaciones pero ya hay "
                    "puntos cargados: el contador todavía no liquidó — no calibro",
                    len(rivales))
        return None
    return float(np.mean(exactos)) / partidos_jugados


def top_pool_picks(grid: np.ndarray, cfg: PoolConfig | None = None, k: int = 5):
    """[(marcador, Q), ...] — los k marcadores más jugados por el pool. Para auditar."""
    q = pool_distribution(grid, cfg)
    order = np.argsort(-q)[:k]
    return [(index_score(int(i)), float(q[i])) for i in order]

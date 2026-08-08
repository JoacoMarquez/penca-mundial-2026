"""El rerun avisa por VALOR, no por diferencia de picks.

Contexto (2026-08-08): el óptimo es plano — dos corridas con insumos idénticos
reasignan ~43 de 96 picks. Ese día el rerun pidió recargar 56 picks hacia una
planilla PEOR ($238k → $221k). El disparador correcto es el Δ E[premio] medido con
sorteos comunes, no la existencia de un diff.
"""

import numpy as np
import pytest

from src.clausura.economics import PrizeConfig, SimConfig, score_index
from src.clausura.rerun_cierre import (
    UMBRAL_ABS,
    picks_previos,
    valor_del_cambio,
    vale_avisar,
)
from src.clausura.strategy import ComparacionPortfolios, EvaluadorPortfolio
from src.model.poisson import score_grid


def comp(delta, se):
    return ComparacionPortfolios(delta=delta, se=se, valor_a=200_000.0,
                                 valor_b=200_000.0 + delta, n_seeds=5)


# -------------------- umbral --------------------

def test_no_avisa_por_churn():
    """Δ chico o negativo = el ruido del optimizador. Silencio."""
    assert not vale_avisar(comp(-16_489, 3_000))     # el caso real del 8/8
    assert not vale_avisar(comp(0, 3_000))
    assert not vale_avisar(comp(500, 100))           # positivo y significativo, pero migaja


def test_no_avisa_si_el_delta_no_supera_su_ruido():
    """Mejora grande pero indistinguible de cero: no se pide recargar 12 planillas."""
    assert not vale_avisar(comp(10_000, 8_000))      # 1.25 se
    assert vale_avisar(comp(10_000, 2_000))          # 5 se


def test_avisa_cuando_la_mejora_es_real():
    assert vale_avisar(comp(25_000, 4_000))
    assert comp(25_000, 4_000).significativa


def test_el_piso_absoluto_es_el_costo_de_recargar():
    assert not vale_avisar(comp(UMBRAL_ABS - 1, 1.0))
    assert vale_avisar(comp(UMBRAL_ABS + 1, 1.0))


# -------------------- comparación pareada --------------------

def _evaluador(n_matches=6, n_part=3):
    grids = [score_grid(1.3, 1.1, 0.0, max_goals=5) for _ in range(n_matches)]
    from src.clausura.pool import PoolConfig, pool_distribution
    pool_qs = [pool_distribution(g, PoolConfig()) for g in grids]
    fechas = [0] * n_matches
    pref = [False] * n_matches
    cfg = SimConfig(n_sims=200, n_rivales=40)
    return EvaluadorPortfolio(grids, fechas, pref, pool_qs, PrizeConfig(), cfg), n_matches, n_part


def test_comparar_una_planilla_contra_si_misma_da_cero_exacto():
    """Sorteos comunes: sin diferencia de picks no puede haber diferencia de valor.

    Es la propiedad que hace usable el Δ — si cada brazo se evaluara con sorteos
    propios, esto daría ruido en vez de cero.
    """
    ev, n_matches, n_part = _evaluador()
    picks = np.full((n_part, n_matches), score_index(1, 1), dtype=np.int64)
    c = ev.comparar(picks, picks, n_seeds=3)
    assert c.delta == 0.0
    assert c.se == 0.0
    assert not c.significativa


def test_comparar_detecta_una_planilla_peor():
    """Chalk razonable vs todas en 5-5: el Δ tiene que ser claramente negativo."""
    ev, n_matches, n_part = _evaluador()
    buena = np.full((n_part, n_matches), score_index(1, 1), dtype=np.int64)
    mala = np.full((n_part, n_matches), score_index(5, 5), dtype=np.int64)
    c = ev.comparar(buena, mala, n_seeds=3)
    assert c.delta < 0
    assert not vale_avisar(c)


# -------------------- reconstrucción de la matriz vieja --------------------

def test_picks_previos_solo_pisa_la_fecha_objetivo():
    class Port:
        picks = np.full((2, 4), score_index(1, 1), dtype=np.int64)

    contexto = {"portfolio": Port(), "idx_of": {100: 1, 101: 2}}
    prev = {"picks": [
        {"evento_id": 100, "scores": [[0, 0], [2, 1]]},
        {"evento_id": 101, "scores": [[3, 0], [0, 3]]},
        {"evento_id": 999, "scores": [[4, 4], [4, 4]]},   # evento ajeno: se ignora
    ]}
    m = picks_previos(prev, contexto, 2)
    assert m[0, 1] == score_index(0, 0) and m[1, 1] == score_index(2, 1)
    assert m[0, 2] == score_index(3, 0) and m[1, 2] == score_index(0, 3)
    # las columnas que la planilla vieja no menciona quedan como en la nueva
    assert m[0, 0] == score_index(1, 1) and m[0, 3] == score_index(1, 1)


def test_valor_del_cambio_sin_evaluador_no_explota():
    """Contexto incompleto → None, y el rerun avisa igual (no se queda mudo)."""
    assert valor_del_cambio({"picks": []}, {}, 12) is None


def test_build_portfolio_deja_el_evaluador_listo():
    """El seam que usa el rerun: el portfolio trae con qué re-liquidar otras picks."""
    from src.clausura.strategy import build_portfolio

    grids = [score_grid(1.3, 1.1, 0.0, max_goals=5) for _ in range(4)]
    port = build_portfolio(
        grids=grids, fecha_de_partido=[0] * 4, preferencial=[False] * 4,
        n_participaciones=3, sim=SimConfig(n_sims=120, n_rivales=30), max_passes=1,
    )
    assert port.evaluador is not None
    otra = np.full_like(port.picks, score_index(0, 0))
    c = port.evaluador.comparar(port.picks, otra, n_seeds=2)
    assert c.n_seeds == 2
    # el portfolio optimizado no puede ser peor que poner 0-0 en todo
    assert c.delta <= 0

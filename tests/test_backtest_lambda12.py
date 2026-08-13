"""Tests del backtest de dependencia de goles (scripts/backtest_lambda12).

Lo que hay que blindar son las VEROSIMILITUDES: si la versión vectorizada no coincide
con la del repo, el backtest mide un modelo distinto del que corre en producción y
todo el veredicto es basura — sin que nada se ponga rojo.
"""

from __future__ import annotations

from math import log

import numpy as np
import pytest

from scripts.backtest_lambda12 import (
    _tau_dixon_coles,
    fit_lam12,
    fit_rho,
    loglik_bivar,
    loglik_dixon,
    loglik_indep,
)
from src.model.poisson import bivariate_poisson_pmf

GL = np.array([0, 1, 2, 3, 0, 2, 1, 4])
GV = np.array([0, 1, 0, 2, 3, 1, 0, 1])
LL = np.array([1.3, 1.5, 2.0, 1.1, 0.9, 1.8, 1.2, 2.4])
LV = np.array([1.1, 1.2, 0.8, 1.4, 1.6, 1.0, 1.5, 0.7])


# -------------------- equivalencia con la implementación de producción --------------------

def test_indep_coincide_con_la_bivariada_de_produccion_en_lam12_cero():
    mia = loglik_indep(GL, GV, LL, LV)
    ref = np.array([log(bivariate_poisson_pmf(int(a), int(b), x, y, 0.0))
                    for a, b, x, y in zip(GL, GV, LL, LV)])
    assert np.abs(mia - ref).max() < 1e-12


@pytest.mark.parametrize("lam12", [0.0, 0.05, 0.2, 0.5])
def test_bivar_coincide_con_produccion(lam12):
    """La versión vectorizada tiene que ser la MISMA función que score_grid usa.

    Con otra parametrización (λ marginal vs λ del componente) el backtest mediría
    un modelo que producción no ejecuta — el error que ya se cometió con el decay.
    """
    mia = loglik_bivar(GL, GV, LL, LV, lam12)
    ref = np.array([log(bivariate_poisson_pmf(int(a), int(b), x, y,
                                              min(lam12, min(x, y) * 0.999)))
                    for a, b, x, y in zip(GL, GV, LL, LV)])
    assert np.abs(mia - ref).max() < 1e-12


def test_lam12_se_clipea_por_partido():
    """λ12 no puede pasar min(λ_L, λ_V) — es la misma restricción que el blend."""
    ll = np.array([0.3]); lv = np.array([2.0])
    gl = np.array([0]); gv = np.array([1])
    # pedir λ12=0.9 con min(λ)=0.3 no puede explotar: se clipea
    assert np.isfinite(loglik_bivar(gl, gv, ll, lv, 0.9)).all()


# -------------------- Dixon-Coles --------------------

def test_tau_solo_toca_los_cuatro_marcadores_bajos():
    gl = np.array([0, 0, 1, 1, 2, 3])
    gv = np.array([0, 1, 0, 1, 2, 0])
    ll = np.full(6, 1.4); lv = np.full(6, 1.1)
    tau = _tau_dixon_coles(gl, gv, ll, lv, rho=0.1)
    assert not np.isclose(tau[:4], 1.0).any()      # 0-0, 0-1, 1-0, 1-1 se ajustan
    assert np.allclose(tau[4:], 1.0)               # 2-2 y 3-0 quedan intactos


def test_rho_negativo_sube_la_diagonal_baja():
    """ρ<0 es la dirección "más empates chicos de los que da el Poisson".

    Es la forma de dependencia que la correlación de Pearson NO ve, y por eso se
    testea aparte de la bivariada.
    """
    gl = np.array([0, 1]); gv = np.array([0, 1])
    ll = np.array([1.2, 1.2]); lv = np.array([1.1, 1.1])
    base = loglik_indep(gl, gv, ll, lv)
    con_rho = loglik_dixon(gl, gv, ll, lv, rho=-0.1)
    assert (con_rho > base).all()


def test_dixon_con_rho_cero_es_el_poisson_independiente():
    assert np.allclose(loglik_dixon(GL, GV, LL, LV, 0.0), loglik_indep(GL, GV, LL, LV))


def test_rho_que_vuelve_tau_negativo_se_penaliza():
    """Sin esta guarda el optimizador se iría a un ρ que produce probabilidades
    negativas y "gana" con una verosimilitud sin sentido."""
    ll = np.array([2.5]); lv = np.array([2.5])
    gl = np.array([0]); gv = np.array([0])
    # τ(0,0) = 1 − λ1λ2ρ = 1 − 6.25·0.25 < 0
    assert loglik_dixon(gl, gv, ll, lv, 0.25)[0] <= -1e5


# -------------------- el fit recupera lo que se le siembra --------------------

def test_el_fit_recupera_un_lam12_sembrado():
    """Si los datos se generan CON dependencia, el MLE tiene que encontrarla.

    Es el control que hace informativo al resultado nulo: sin esto, "no encontré
    dependencia" podría ser simplemente un fit que no funciona.
    """
    rng = np.random.default_rng(7)
    n = 4000
    lam12_real = 0.25
    l1, l2 = 1.1, 0.9
    w3 = rng.poisson(lam12_real, n)
    gl = rng.poisson(l1, n) + w3
    gv = rng.poisson(l2, n) + w3
    ll = np.full(n, l1 + lam12_real)      # marginales, como las devuelve ratings
    lv = np.full(n, l2 + lam12_real)

    assert fit_lam12(gl, gv, ll, lv) == pytest.approx(lam12_real, abs=0.06)


def test_el_fit_devuelve_cero_si_no_hay_dependencia():
    rng = np.random.default_rng(11)
    n = 4000
    ll = np.full(n, 1.35); lv = np.full(n, 1.05)
    gl = rng.poisson(1.35, n); gv = rng.poisson(1.05, n)
    assert fit_lam12(gl, gv, ll, lv) < 0.06


def test_el_fit_de_rho_recupera_el_signo_sembrado():
    """Datos con exceso de 0-0 y 1-1 ⇒ ρ negativo (que es como DC sube la diagonal)."""
    rng = np.random.default_rng(13)
    n = 3000
    ll = np.full(n, 1.3); lv = np.full(n, 1.1)
    gl = rng.poisson(1.3, n); gv = rng.poisson(1.1, n)
    # se siembra exceso de empates bajos pisando una fracción de los partidos
    idx = rng.choice(n, size=250, replace=False)
    gl[idx[:125]], gv[idx[:125]] = 0, 0
    gl[idx[125:]], gv[idx[125:]] = 1, 1
    assert fit_rho(gl, gv, ll, lv) < 0

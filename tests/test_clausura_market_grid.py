"""Tests del refinamiento de grilla con los mercados ricos de Supermatch."""

import numpy as np
import pytest

from src.clausura.economics import MAX_GOALS
from src.clausura.market_grid import (
    W_MERCADO,
    exact_score_overlay,
    goals_marginal_target,
    margin_discrepancy,
    rake_to_goal_marginals,
    refine_grid,
)
from src.clausura.odds import EventOdds
from src.model.poisson import score_grid


def _base():
    return score_grid(1.3, 1.1, 0.0, max_goals=MAX_GOALS)


def _event(**kw):
    return EventOdds(event_id="sm:1", home="Peñarol", away="Wanderers",
                     start_utc="2026-08-08T20:00:00+00:00", fetched_utc="x",
                     x1x2={"home": 1.53, "draw": 3.8, "away": 5.2}, **kw)


# fixture real recortada (Peñarol vs Wanderers, 2026-08-04)
HOME_GOALS = {"0": 4.25, "1": 2.65, "2": 3.2, "3+": 3.6}
AWAY_GOALS = {"0": 1.75, "1": 2.38, "2": 6.2, "3+": 21.0}
CORRECT_SCORE = {"1:0": 4.85, "2:0": 5.2, "2:1": 7.0, "0:0": 8.0, "otro": 23.0}
MARGIN = {"D": 3.8, "H+1": 3.05, "H+2": 4.05, "H+3": 5.1,
          "A+1": 6.2, "A+2": 19.0, "A+3": 67.0}


# -------------------- marginales --------------------

def test_goals_marginal_target_devig_y_suma_1():
    t = goals_marginal_target(HOME_GOALS)
    assert t.shape == (4,)
    assert t.sum() == pytest.approx(1.0)
    assert t[1] == max(t)   # el "1 gol" es el favorito del mercado (2.65)


def test_goals_marginal_target_incompleto_es_none():
    assert goals_marginal_target({}) is None
    assert goals_marginal_target({"0": 4.0, "1": 2.5}) is None


def test_rake_matchea_los_targets_blendeados():
    g = _base()
    th, ta = goals_marginal_target(HOME_GOALS), goals_marginal_target(AWAY_GOALS)
    out = rake_to_goal_marginals(g, th, ta)

    assert out.sum() == pytest.approx(1.0)
    assert (out >= 0).all()
    # marginal resultante == blend 70/30 entre mercado y base (bucketizada)
    def bucket(m):
        return np.array([m[0], m[1], m[2], m[3:].sum()])
    base_h = bucket(g.sum(axis=1))
    esperado_h = W_MERCADO * th + (1 - W_MERCADO) * base_h
    esperado_h /= esperado_h.sum()
    assert np.allclose(bucket(out.sum(axis=1)), esperado_h, atol=1e-8)


def test_rake_preserva_estructura_dentro_del_bucket():
    """IPF escala buckets enteros: dentro del bucket 3+ las razones no cambian."""
    g = _base()
    th, ta = goals_marginal_target(HOME_GOALS), goals_marginal_target(AWAY_GOALS)
    out = rake_to_goal_marginals(g, th, ta)
    razon_base = g[4, 0] / g[3, 0]
    razon_out = out[4, 0] / out[3, 0]
    assert razon_out == pytest.approx(razon_base, rel=1e-9)


# -------------------- overlay del marcador exacto --------------------

def test_overlay_mueve_las_celdas_hacia_el_mercado():
    g = _base()
    out = exact_score_overlay(g, CORRECT_SCORE)
    assert out.sum() == pytest.approx(1.0)
    # el mercado cotiza el 0-0 a 8.0 (~11% pre-devig), muy arriba del Poisson
    assert out[0, 0] > g[0, 0]


def test_overlay_reparte_otro_proporcional_a_la_grilla():
    g = _base()
    out = exact_score_overlay(g, CORRECT_SCORE)
    # celdas no cotizadas: la razón entre dos de ellas se preserva
    assert out[3, 2] / out[2, 3] == pytest.approx(g[3, 2] / g[2, 3], rel=1e-9)


def test_overlay_pocas_celdas_devuelve_none():
    assert exact_score_overlay(_base(), {"1:0": 4.0, "otro": 1.4}) is None


def test_overlay_ignora_goleadas_fuera_de_la_grilla():
    cs = dict(CORRECT_SCORE)
    cs["7:0"] = 100.0   # fuera de la grilla de trabajo → va a la masa de 'otro'
    out = exact_score_overlay(_base(), cs)
    assert out.sum() == pytest.approx(1.0)


# -------------------- margen (diagnóstico) --------------------

def test_margin_discrepancy_consistente_es_chica():
    """Margen generado DESDE la grilla (sin vig) → discrepancia ~0."""
    g = _base()
    diff = np.arange(MAX_GOALS + 1)[:, None] - np.arange(MAX_GOALS + 1)[None, :]
    margin = {"D": 1 / g[diff == 0].sum()}
    for n in (1, 2):
        margin[f"H+{n}"] = 1 / g[diff == n].sum()
        margin[f"A+{n}"] = 1 / g[diff == -n].sum()
    margin["H+3"] = 1 / g[diff >= 3].sum()
    margin["A+3"] = 1 / g[diff <= -3].sum()
    assert margin_discrepancy(g, margin) == pytest.approx(0.0, abs=1e-9)


def test_margin_discrepancy_sin_mercado_es_none():
    assert margin_discrepancy(_base(), {}) is None


# -------------------- integración --------------------

def test_refine_grid_sin_mercados_ricos_devuelve_la_base_intacta():
    g = _base()
    out, usados = refine_grid(g, _event())
    assert out is g and usados == []
    out, usados = refine_grid(g, None)
    assert out is g and usados == []


def test_refine_grid_con_todo_aplica_ambos_refinamientos():
    g = _base()
    out, usados = refine_grid(g, _event(
        home_goals=HOME_GOALS, away_goals=AWAY_GOALS,
        correct_score=CORRECT_SCORE, margin=MARGIN,
    ))
    assert usados == ["goles", "exacto"]
    assert out.sum() == pytest.approx(1.0)
    assert (out >= 0).all()
    assert out[0, 0] > g[0, 0]   # el 0-0 del mercado sube sobre el Poisson


def test_refine_grid_solo_goles():
    out, usados = refine_grid(_base(), _event(
        home_goals=HOME_GOALS, away_goals=AWAY_GOALS,
    ))
    assert usados == ["goles"]
    assert out.sum() == pytest.approx(1.0)


class _RatingsStub:
    def lambdas(self, local, visitante):
        return 1.3, 1.1


_EV = {"evento_id": 1, "local": "Peñarol", "visitante": "Wanderers",
       "fecha_n": 1, "fecha_id": 280, "preferencial": False,
       "inicio_utc": "2026-08-08T20:00:00+00:00",
       "cierre_pronostico_utc": "2026-08-08T19:45:00+00:00"}


def _odds_ricos():
    return _event(home_goals=HOME_GOALS, away_goals=AWAY_GOALS,
                  correct_score=CORRECT_SCORE)


def test_build_season_grids_flag_apagado_usa_base_y_guarda_sombra(monkeypatch):
    """Default: la grilla activa es la Poisson base, pero la sombra versiona ambas."""
    from src.clausura.picks import build_season_grids

    monkeypatch.delenv("CLAUSURA_MERCADOS_RICOS", raising=False)
    grids, fuentes, pred, shadow = build_season_grids(
        [_EV], _RatingsStub(), odds_by_evento={1: _odds_ricos()}, resultados={})
    assert fuentes[0] == "mercado+ratings"          # sin sufijo: refinado no activo
    assert shadow[0]["mercados"] == ["goles", "exacto"]
    assert len(shadow[0]["base"]) == 36 and len(shadow[0]["rica"]) == 36
    assert sum(shadow[0]["rica"]) == pytest.approx(1.0, abs=1e-4)
    assert shadow[0]["base"] != shadow[0]["rica"]
    # la grilla activa es la base (sin refinar)
    assert np.allclose(pred[0].ravel(), shadow[0]["base"], atol=1e-6)


def test_build_season_grids_flag_prendido_usa_la_refinada(monkeypatch):
    from src.clausura.picks import build_season_grids

    monkeypatch.setenv("CLAUSURA_MERCADOS_RICOS", "1")
    grids, fuentes, pred, shadow = build_season_grids(
        [_EV], _RatingsStub(), odds_by_evento={1: _odds_ricos()}, resultados={})
    assert fuentes[0] == "mercado+ratings+goles+exacto"
    assert pred[0].sum() == pytest.approx(1.0)
    assert np.allclose(pred[0].ravel(), shadow[0]["rica"], atol=1e-6)


def test_build_season_grids_sin_mercados_ricos_no_hay_sombra(monkeypatch):
    from src.clausura.picks import build_season_grids

    monkeypatch.delenv("CLAUSURA_MERCADOS_RICOS", raising=False)
    grids, fuentes, pred, shadow = build_season_grids(
        [_EV], _RatingsStub(), odds_by_evento={1: _event()}, resultados={})
    assert shadow[0] is None
    assert fuentes[0] == "mercado+ratings"

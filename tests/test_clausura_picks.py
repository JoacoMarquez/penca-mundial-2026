"""Tests del pipeline manual de picks (sin red: helpers puros y estado en disco)."""

import json

import numpy as np
import pytest

from src.clausura.economics import score_index
from src.clausura.odds import EventOdds
from src.clausura.picks import (
    build_season_grids,
    delta_grid,
    fecha_dir,
    load_frozen,
    match_odds,
    save_version,
)
from src.clausura.strategy import build_portfolio
from src.model.poisson import score_grid


def _evento(eid, local, visitante, fecha_n=1, pref=False):
    return {
        "evento_id": eid, "local": local, "visitante": visitante,
        "fecha_n": fecha_n, "fecha_id": 279 + fecha_n, "preferencial": pref,
        "inicio_utc": "2026-08-07T22:00:00+00:00",
        "cierre_pronostico_utc": "2026-08-07T21:45:00+00:00",
    }


def _odds(home, away):
    return EventOdds(event_id="sm:1", home=home, away=away,
                     start_utc="2026-08-07T22:00:00+00:00", fetched_utc="x",
                     x1x2={"home": 2.0, "draw": 3.2, "away": 3.8})


# -------------------- matching de odds --------------------

def test_match_odds_nombre_exacto():
    evs = [_evento(1, "Nacional", "Boston River")]
    m = match_odds(evs, [_odds("Nacional", "Boston River")])
    assert 1 in m


def test_match_odds_nombre_parcial_y_acentos():
    """'Montevideo City Torque' (penca-api) vs 'Montevideo City' (ES), y acentos."""
    evs = [_evento(1, "Montevideo City Torque", "Peñarol")]
    m = match_odds(evs, [_odds("Montevideo City", "Penarol")])
    assert 1 in m


def test_match_odds_no_cruza_partidos():
    evs = [_evento(1, "Nacional", "Cerro"), _evento(2, "Danubio", "Cerro Largo")]
    m = match_odds(evs, [_odds("Danubio", "Cerro Largo")])
    assert m.keys() == {2}


# -------------------- grillas --------------------

def test_delta_grid_concentra_la_masa():
    g = delta_grid(2, 1)
    assert g[2, 1] == 1.0 and g.sum() == 1.0


def test_delta_grid_trunca_goleadas():
    g = delta_grid(7, 0)
    assert g[5, 0] == 1.0  # 6+ se truncan a la grilla de trabajo


class _RatingsStub:
    def lambdas(self, local, visitante):
        return 1.3, 1.1


def test_build_season_grids_separa_liquidacion_de_predictivas():
    """Jugados: grilla de liquidación = delta, pero la predictiva sigue siendo la
    del modelo (calibración/Q/γ no pueden ver el resultado)."""
    eventos = [_evento(10, "A", "B"), _evento(20, "C", "D")]
    grids, fuentes, pred, _ = build_season_grids(
        eventos, _RatingsStub(), odds_by_evento={}, resultados={10: (2, 1)})

    assert grids[0][2, 1] == 1.0 and fuentes[0] == "final 2-1"   # delta
    assert pred[0].max() < 0.5                                    # predictiva, no delta
    assert pred[0].sum() == pytest.approx(1.0)
    # futuros: liquidación y predictiva son la misma grilla
    assert grids[1] is pred[1]
    assert fuentes[1] == "ratings"


# -------------------- versionado en disco --------------------

def test_save_version_no_sobreescribe(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    p1 = save_version(3, {"a": 1})
    p2 = save_version(3, {"a": 2})
    assert p1 != p2
    assert p1.name.startswith("v1_") and p2.name.startswith("v2_")
    assert json.loads(p1.read_text())["a"] == 1
    assert json.loads(p2.read_text())["a"] == 2


def test_load_frozen_desde_archivo(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    eventos = [_evento(10, "A", "B", fecha_n=1), _evento(20, "C", "D", fecha_n=2)]
    save_version(1, {"picks": [
        {"evento_id": 10, "scores": [[1, 0], [2, 1], [0, 0]]},
    ]})

    frozen, mask = load_frozen(eventos, target_fecha=2, n_participaciones=3)
    assert mask.tolist() == [True, False]
    assert frozen[0, 0] == score_index(1, 0)
    assert frozen[1, 0] == score_index(2, 1)
    assert frozen[2, 0] == score_index(0, 0)


def test_load_frozen_congela_cerrados_de_la_fecha_objetivo(tmp_path, monkeypatch):
    """Regeneración intra-fecha: el partido del viernes ya jugado conserva SU pick
    guardado; el del sábado (abierto) queda libre. Antes quedaba en 0-0 fantasma."""
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    eventos = [_evento(10, "A", "B", fecha_n=1), _evento(20, "C", "D", fecha_n=1)]
    save_version(1, {"picks": [
        {"evento_id": 10, "scores": [[1, 0], [0, 0]]},
        {"evento_id": 20, "scores": [[2, 1], [1, 1]]},
    ]})

    frozen, mask = load_frozen(eventos, target_fecha=1, n_participaciones=2,
                               cerrados={10})
    assert mask.tolist() == [True, False]
    assert frozen[0, 0] == score_index(1, 0)
    assert frozen[1, 0] == score_index(0, 0)


def test_load_frozen_sin_cerrados_no_congela_la_fecha_objetivo(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)
    eventos = [_evento(10, "A", "B", fecha_n=1)]
    save_version(1, {"picks": [{"evento_id": 10, "scores": [[1, 0]]}]})
    _, mask = load_frozen(eventos, target_fecha=1, n_participaciones=1)
    assert not mask.any()


def test_load_frozen_avisa_si_faltan_participaciones(tmp_path, monkeypatch, caplog):
    """Archivo con menos columnas que las pedidas: las filas extra quedan en 0-0,
    pero ahora con warning (antes era silencioso)."""
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)
    eventos = [_evento(10, "A", "B", fecha_n=1), _evento(20, "C", "D", fecha_n=2)]
    save_version(1, {"picks": [{"evento_id": 10, "scores": [[1, 0], [2, 1]]}]})
    with caplog.at_level("WARNING"):
        frozen, mask = load_frozen(eventos, target_fecha=2, n_participaciones=4)
    assert "participaciones" in caplog.text
    assert frozen[3, 0] == score_index(0, 0)


def test_load_frozen_fecha_sin_archivo_avisa_pero_no_rompe(tmp_path, monkeypatch, caplog):
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)
    eventos = [_evento(10, "A", "B", fecha_n=1), _evento(20, "C", "D", fecha_n=2)]
    with caplog.at_level("WARNING"):
        frozen, mask = load_frozen(eventos, target_fecha=2, n_participaciones=2)
    assert not mask.any()
    assert "sin picks guardados" in caplog.text


# -------------------- frozen en el optimizador --------------------

def test_build_portfolio_respeta_frozen():
    g = score_grid(1.4, 1.0, 0.0, max_goals=5)
    grids = [g] * 8
    fechas = [1] * 4 + [2] * 4
    pref = [False] * 8

    frozen = np.full((3, 8), score_index(3, 3), dtype=np.int64)  # pick deliberadamente malo
    mask = np.array([True] * 4 + [False] * 4)

    from src.clausura.economics import SimConfig
    port = build_portfolio(grids, fechas, pref, n_participaciones=3,
                           sim=SimConfig(n_sims=200, n_rivales=30, seed=9),
                           frozen_picks=frozen, frozen_mask=mask, max_passes=1)

    # las columnas congeladas quedan intactas aunque el pick sea malo
    assert (port.picks[:, :4] == score_index(3, 3)).all()
    # las libres NO quedaron en el pick malo
    assert not (port.picks[:, 4:] == score_index(3, 3)).any()


def test_build_portfolio_frozen_sin_picks_falla():
    g = score_grid(1.4, 1.0, 0.0, max_goals=5)
    from src.clausura.economics import SimConfig
    with pytest.raises(ValueError, match="frozen"):
        build_portfolio([g] * 2, [1, 1], [False] * 2, n_participaciones=2,
                        sim=SimConfig(n_sims=50, n_rivales=10, seed=1),
                        frozen_mask=np.array([True, False]))


# -------------------- sección de la penca gratuita --------------------

def test_format_gratuita_usa_la_participacion_1_y_el_campeon_mas_probable():
    """La gratuita (premio indivisible, 1 planilla) va con EV puro = participación 1,
    pero con el campeón MÁS PROBABLE (los especiales sí se diversifican en la fila 1)."""
    from src.clausura.picks import format_gratuita
    from src.clausura.strategy import PortfolioClausura

    eventos = [_evento(10, "Nacional", "Cerro"), _evento(20, "Danubio", "Racing", pref=True)]
    idx_of = {10: 0, 20: 1}
    picks = np.array([
        [score_index(1, 0), score_index(2, 1)],   # participación 1 (ancla EV)
        [score_index(0, 0), score_index(1, 1)],   # participación 2 (perturbada)
    ], dtype=np.int64)
    port = PortfolioClausura(
        picks=picks, candidatos=[], resultado=None,
        campeon=np.array([2, 0]),                 # la fila 1 tiene el campeón 2…
        p_campeon=np.array([0.5, 0.2, 0.3]),      # …pero el más probable es el 0
    )
    txt = format_gratuita(eventos, port, idx_of, ["Peñarol", "Nacional", "Defensor"], None)

    assert "1-0" in txt and "2-1" in txt        # marcadores de la participación 1
    assert "0-0" not in txt and "1-1" not in txt  # NO los de la participación 2
    assert "Peñarol" in txt                     # argmax P(campeón), no el de la fila 1
    assert "Defensor" not in txt
    assert "⭐x2" in txt                         # marca el partido estrella

"""Tests del pipeline manual de picks (sin red: helpers puros y estado en disco)."""

import json

import numpy as np
import pytest

from src.clausura.economics import score_index
from src.clausura.odds import EventOdds
from src.clausura.picks import (
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

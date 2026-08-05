"""Tests del snapshot del pool: Q empírica, blending y persistencia (sin red)."""

import json

import numpy as np
import pytest

from src.clausura.economics import N_SCORES, score_index
from src.clausura.pool_snapshot import (
    RivalPicks,
    blended_q,
    empirical_campeon_counts,
    empirical_counts,
    load_latest_snapshot,
    save_snapshot,
    snapshot_summary,
)


def _snapshot_dict():
    """Snapshot sintético: 3 rivales, 2 eventos, campeones cargados."""
    return {
        "generado_utc": "2026-08-08T12:00:00+00:00",
        "penca_id": 46,
        "n_participaciones": 3,
        "participaciones": [
            {"participacion_id": 1, "numero": 100, "puntos": 5, "exactos": 1,
             "picks": {"2080": [1, 0], "2085": [1, 1]}, "campeon": "Peñarol",
             "campeon_id": 80, "goleador": "X", "goleador_id": 9},
            {"participacion_id": 2, "numero": 101, "puntos": 3, "exactos": 0,
             "picks": {"2080": [1, 0]}, "campeon": "Nacional",
             "campeon_id": 79, "goleador": None, "goleador_id": None},
            {"participacion_id": 3, "numero": 102, "puntos": 0, "exactos": 0,
             "picks": {"2080": [2, 1]}, "campeon": "Peñarol",
             "campeon_id": 80, "goleador": None, "goleador_id": None},
        ],
    }


# -------------------- conteos empíricos --------------------

def test_empirical_counts_por_evento():
    counts = empirical_counts(_snapshot_dict())
    assert set(counts) == {2080, 2085}
    assert counts[2080][score_index(1, 0)] == 2
    assert counts[2080][score_index(2, 1)] == 1
    assert counts[2080].sum() == 3
    assert counts[2085][score_index(1, 1)] == 1


def test_empirical_counts_trunca_goleadas():
    snap = _snapshot_dict()
    snap["participaciones"][0]["picks"] = {"2080": [9, 0]}
    counts = empirical_counts(snap)
    assert counts[2080][score_index(5, 0)] == 1  # 6+ va al borde de la grilla


def test_empirical_counts_excluye_mis_numeros():
    """La Q empírica modela a los RIVALES: nuestras participaciones no cuentan."""
    counts = empirical_counts(_snapshot_dict(), mis_numeros={100})
    assert set(counts) == {2080}          # el 2085 solo lo tenía el 100 (nuestro)
    assert counts[2080].sum() == 2
    assert counts[2080][score_index(1, 0)] == 1


def test_empirical_campeon_counts():
    idx = {"Peñarol": 0, "Nacional": 1, "Cerro": 2}
    c = empirical_campeon_counts(_snapshot_dict(), idx, 3)
    assert c.tolist() == [2, 1, 0]


def test_empirical_campeon_counts_excluye_mis_numeros():
    idx = {"Peñarol": 0, "Nacional": 1, "Cerro": 2}
    c = empirical_campeon_counts(_snapshot_dict(), idx, 3, mis_numeros={102})
    assert c.tolist() == [1, 1, 0]


# -------------------- blending Dirichlet --------------------

def test_blended_q_sin_observaciones_devuelve_prior():
    prior = np.full(N_SCORES, 1 / N_SCORES)
    assert np.array_equal(blended_q(prior, None), prior)
    assert np.array_equal(blended_q(prior, np.zeros(N_SCORES)), prior)


def test_blended_q_pocas_obs_cerca_del_prior_muchas_cerca_de_lo_observado():
    prior = np.full(N_SCORES, 1 / N_SCORES)
    obs = np.zeros(N_SCORES)
    idx = score_index(1, 0)

    obs[idx] = 2   # 2 observaciones contra strength=25
    q_pocas = blended_q(prior, obs, strength=25.0)

    obs_muchas = np.zeros(N_SCORES)
    obs_muchas[idx] = 500
    q_muchas = blended_q(prior, obs_muchas, strength=25.0)

    assert q_pocas[idx] < 0.15          # domina el prior
    assert q_muchas[idx] > 0.90         # domina lo observado
    assert q_pocas.sum() == pytest.approx(1.0)
    assert q_muchas.sum() == pytest.approx(1.0)


def test_blended_q_es_monotona_en_observaciones():
    prior = np.full(N_SCORES, 1 / N_SCORES)
    idx = score_index(0, 0)
    prev = prior[idx]
    for n in [1, 5, 20, 100]:
        obs = np.zeros(N_SCORES)
        obs[idx] = n
        q = blended_q(prior, obs)
        assert q[idx] > prev
        prev = q[idx]


# -------------------- persistencia --------------------

def test_save_y_load_snapshot(tmp_path, monkeypatch):
    import src.clausura.pool_snapshot as ps
    monkeypatch.setattr(ps, "SNAP_DIR", tmp_path)

    rivales = [
        RivalPicks(participacion_id=1, numero=100, puntos=0, exactos=0,
                   picks={2080: (1, 0)}, campeon="Peñarol", campeon_id=80),
    ]
    p1 = save_snapshot(rivales, penca_id=46)
    p2 = save_snapshot(rivales, penca_id=46)
    assert p1.name.startswith("v1_") and p2.name.startswith("v2_")

    snap = load_latest_snapshot()
    assert snap["penca_id"] == 46
    assert snap["participaciones"][0]["picks"] == {"2080": [1, 0]}


def test_load_snapshot_respeta_max_age(tmp_path, monkeypatch):
    import src.clausura.pool_snapshot as ps
    monkeypatch.setattr(ps, "SNAP_DIR", tmp_path)
    (tmp_path / "v1_20200101T000000Z.json").write_text(json.dumps({
        "generado_utc": "2020-01-01T00:00:00+00:00", "penca_id": 46,
        "n_participaciones": 0, "participaciones": [],
    }), encoding="utf-8")
    assert load_latest_snapshot(max_age_hours=48) is None
    assert load_latest_snapshot() is not None   # sin límite sí lo devuelve


def test_snapshot_summary():
    s = snapshot_summary(_snapshot_dict())
    assert "3 participaciones" in s
    assert "2 eventos" in s
    assert "3 con marcadores" in s
    assert "3 con campeón" in s


# -------------------- integración con el optimizador --------------------

def test_build_portfolio_acepta_pool_qs_externo():
    from src.clausura.economics import SimConfig
    from src.clausura.strategy import build_portfolio
    from src.model.poisson import score_grid

    g = score_grid(1.4, 1.0, 0.0, max_goals=5)
    grids = [g] * 4
    q = np.full(N_SCORES, 1 / N_SCORES)
    port = build_portfolio(grids, [1, 1, 2, 2], [False] * 4, n_participaciones=2,
                           sim=SimConfig(n_sims=150, n_rivales=20, seed=3),
                           pool_qs=[q] * 4, max_passes=1)
    assert port.picks.shape == (2, 4)

    with pytest.raises(ValueError, match="pool_qs"):
        build_portfolio(grids, [1, 1, 2, 2], [False] * 4, n_participaciones=2,
                        sim=SimConfig(n_sims=150, n_rivales=20, seed=3),
                        pool_qs=[q] * 3)

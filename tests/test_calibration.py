"""Tests de la calibración del pool por ranking-inversion (Capa 5).

Incluye el backtest sintético exigido por la regla del proyecto: un pool generado con
parámetros conocidos debe ser recuperado por la calibración, y el fit calibrado debe
predecir los shares observados mejor que el prior.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.meta.calibration import (
    Observation,
    POINT_CLASSES,
    PRIOR_BIAS_SCALE,
    PRIOR_CHALK,
    PRIOR_NO_SHOW,
    _config_for,
    _points_delta_shares,
    build_observations,
    calibrate,
    get_pool_config,
    point_class_masks,
    predicted_shares,
    save_calibration,
    snapshot_leaderboard,
)
from src.meta.pool import PoolModelConfig, pool_pick_distribution
from src.model.poisson import jmlm_points, score_grid


# ---------- helpers ----------

def _grid(lam_L=1.8, lam_V=0.7, lam12=0.1):
    return score_grid(lam_L, lam_V, lam12, max_goals=7)


# ---------- point_class_masks ----------

def test_point_class_masks_match_105():
    """Caso real: México 2-0 bajo la regla 6/4/3/0. Verifica clases a mano."""
    masks = point_class_masks((2, 0), 8)
    assert masks[6][2, 0]            # exacto
    assert masks[4][3, 1]            # ganador + diferencia de gol (2)
    assert masks[4][4, 2]
    assert masks[3][1, 0]            # ganador, diferencia errada (1 vs 2)
    assert masks[3][2, 1]            # ganador, diferencia errada
    assert masks[0][0, 0]            # empate → ganador errado → 0
    assert masks[0][2, 3]            # visita gana → 0
    assert masks[0][0, 1]            # nada
    # Las máscaras particionan la grilla completa
    total = sum(int(m.sum()) for m in masks.values())
    assert total == 64


def test_predicted_shares_sum_to_one():
    shares = predicted_shares(_grid(), (2, 0), PoolModelConfig(), no_show_frac=0.1)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)


# ---------- shares observados desde snapshots ----------

def test_points_delta_shares_first_match():
    cur = [
        {"penca_id": 1, "points_total": 6, "exact_scores": 1, "correct_winners": 1, "predictions_made": 1},
        {"penca_id": 2, "points_total": 4, "exact_scores": 0, "correct_winners": 1, "predictions_made": 1},
        {"penca_id": 3, "points_total": 0, "exact_scores": 0, "correct_winners": 0, "predictions_made": 1},
        {"penca_id": 4, "points_total": 0, "exact_scores": 0, "correct_winners": 0, "predictions_made": 0},  # no-show
    ]
    shares, n, exact_frac, winner_frac = _points_delta_shares(None, cur)
    assert n == 3
    assert shares[6] == pytest.approx(1 / 3)
    assert shares[4] == pytest.approx(1 / 3)
    assert shares[0] == pytest.approx(1 / 3)
    assert exact_frac == pytest.approx(1 / 3)   # solo la penca 1 clavó
    assert winner_frac == pytest.approx(2 / 3)  # pencas 1 y 2


def test_points_delta_shares_with_previous():
    prev = [{"penca_id": 1, "points_total": 6, "predictions_made": 1},
            {"penca_id": 2, "points_total": 4, "predictions_made": 1}]
    cur = [{"penca_id": 1, "points_total": 9, "predictions_made": 2},   # +3
           {"penca_id": 2, "points_total": 10, "predictions_made": 2},  # +6
           {"penca_id": 9, "points_total": 7, "predictions_made": 2}]   # entró tarde: delta 7 inválido
    shares, n, _exact, _winner = _points_delta_shares(prev, cur)
    assert n == 2
    assert shares[3] == pytest.approx(0.5)
    assert shares[6] == pytest.approx(0.5)


# ---------- backtest sintético: recuperación de parámetros ----------

def _synthetic_observations(true_chalk, true_beta, true_no_show, n_matches=6, n_pool=400, seed=42):
    """Simula un pool con parámetros conocidos y devuelve observaciones como las reales."""
    rng = np.random.default_rng(seed)
    cfg = _config_for(true_chalk, true_beta)
    obs = []
    # Partidos variados: favoritos fuertes, parejos, visitantes
    lambdas = [(2.0, 0.6), (1.5, 1.1), (1.2, 1.3), (1.9, 0.8), (0.9, 1.6), (1.6, 0.7)][:n_matches]
    for i, (lL, lV) in enumerate(lambdas):
        grid = _grid(lL, lV)
        n = grid.shape[0]
        q = pool_pick_distribution(grid, cfg).flatten()
        # outcome real sampleado del grid verdadero
        out_idx = rng.choice(n * n, p=grid.flatten() / grid.sum())
        actual = (out_idx // n, out_idx % n)
        # picks del pool sampleados de Q; algunos no-shows
        n_show = int(n_pool * (1 - true_no_show))
        picks = rng.choice(n * n, size=n_show, p=q)
        counts = {c: 0 for c in POINT_CLASSES}
        exact_hits = winner_hits = 0
        aw = "H" if actual[0] > actual[1] else ("A" if actual[0] < actual[1] else "D")
        for p in picks:
            pk = (p // n, p % n)
            pts = jmlm_points(pk, actual)
            counts[pts] += 1
            exact_hits += int(pk == actual)
            pw = "H" if pk[0] > pk[1] else ("A" if pk[0] < pk[1] else "D")
            winner_hits += int(pw == aw)
        counts[0] += n_pool - n_show  # no-shows suman 0
        shares = {c: counts[c] / n_pool for c in counts}
        obs.append(Observation(match_id=f"SYN_{i}", actual=actual, shares=shares,
                               n_entries=n_pool, grid=grid,
                               exact_frac=exact_hits / n_pool,
                               winner_frac=winner_hits / n_pool))
    return obs


def test_calibration_recovers_synthetic_params():
    """Pool sintético chalk-fuerte (1.2, β=0.5, 10% no-show) — el fit debe acercarse
    a los parámetros verdaderos más de lo que estaba el prior, y mejorar el loss."""
    true = dict(chalk=1.2, beta=0.5, ns=0.10)
    obs = _synthetic_observations(true["chalk"], true["beta"], true["ns"])
    fit = calibrate(obs)
    assert fit is not None
    # mejora vs prior
    assert fit["loss"] < fit["prior_loss"]
    # recuperación: más cerca del verdadero que el prior en chalk y β
    assert abs(fit["chalk_strength"] - true["chalk"]) < abs(PRIOR_CHALK - true["chalk"])
    assert abs(fit["bias_scale"] - true["beta"]) < abs(PRIOR_BIAS_SCALE - true["beta"])
    assert abs(fit["no_show_frac"] - true["ns"]) <= 0.10


def test_calibration_neutral_when_pool_matches_prior():
    """Si el pool ES el prior, la calibración no debe alejarse mucho de él."""
    obs = _synthetic_observations(PRIOR_CHALK, PRIOR_BIAS_SCALE, PRIOR_NO_SHOW, seed=7)
    fit = calibrate(obs)
    assert abs(fit["chalk_strength"] - PRIOR_CHALK) <= 0.25
    assert abs(fit["bias_scale"] - PRIOR_BIAS_SCALE) <= 0.4


def test_calibration_single_match_is_conservative():
    """Con UNA observación, la regularización debe evitar saltos extremos."""
    obs = _synthetic_observations(1.5, 0.2, 0.0, n_matches=1)
    fit = calibrate(obs)
    # se mueve hacia el verdadero pero sin irse al borde del grid
    assert PRIOR_CHALK <= fit["chalk_strength"] <= 1.6


# ---------- persistencia + get_pool_config ----------

def test_save_and_get_pool_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # sin calibración → prior
    cfg = get_pool_config()
    assert cfg.chalk_strength == PRIOR_CHALK

    fit = {"chalk_strength": 1.1, "bias_scale": 0.5, "no_show_frac": 0.1,
           "loss": 0.01, "prior_loss": 0.02, "improvement_pct": 50.0,
           "n_observations": 3, "matches": ["a"], "fitted_at": "2026-06-12T00:00:00Z"}
    save_calibration(fit)
    cfg2 = get_pool_config()
    assert cfg2.chalk_strength == 1.1
    # β=0.5 aplicado como exponente al bias
    assert cfg2.popular_score_bias[(1, 0)] == pytest.approx(1.8 ** 0.5)
    # historia acumulada
    data = json.loads((tmp_path / "pool_calibration.json").read_text())
    assert len(data["history"]) == 1


def test_build_observations_end_to_end(tmp_path, monkeypatch):
    """Snapshot + postmortem + predicción en disco → observación bien armada."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # predicción persistida
    pdir = tmp_path / "predictions" / "105"
    pdir.mkdir(parents=True)
    (pdir / "v1_20260611T000000Z.json").write_text(json.dumps({
        "constraints": {"lambda_L": 1.8, "lambda_V": 0.7, "lambda_12": 0.1}}))
    # postmortem con marcador real
    pmdir = tmp_path / "postmortems"
    pmdir.mkdir()
    (pmdir / "105.json").write_text(json.dumps({"actual_home": 2, "actual_away": 0}))
    # snapshot (primer partido → baseline 0)
    entries = [{"penca_id": i, "points_total": 4 if i % 2 else 6,
                "exact_scores": 0 if i % 2 else 1, "correct_winners": 1, "predictions_made": 1}
               for i in range(100)]
    snapshot_leaderboard("105", ["105"], entries=entries)

    obs = build_observations(tmp_path)
    assert len(obs) == 1
    assert obs[0].actual == (2, 0)
    assert obs[0].shares[6] == pytest.approx(0.5)
    assert obs[0].shares[4] == pytest.approx(0.5)
    assert obs[0].n_entries == 100
    assert obs[0].exact_frac == pytest.approx(0.5)
    assert obs[0].winner_frac == pytest.approx(1.0)

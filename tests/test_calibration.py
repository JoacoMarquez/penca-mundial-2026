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
        obs.append(Observation(match_id=f"SYN_{i}", actuals=(actual,), shares=shares,
                               n_entries=n_pool, grids=(grid,),
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
    assert obs[0].actuals == ((2, 0),)
    assert obs[0].n_matches == 1
    assert obs[0].shares[6] == pytest.approx(0.5)
    assert obs[0].shares[4] == pytest.approx(0.5)
    assert obs[0].n_entries == 100
    assert obs[0].exact_frac == pytest.approx(0.5)
    assert obs[0].winner_frac == pytest.approx(1.0)


def test_convolve_point_classes_two_matches():
    """Convolución de dos partidos: las clases conjuntas y sus probabilidades."""
    a = {6: 0.1, 4: 0.2, 3: 0.3, 0: 0.4}
    b = {6: 0.0, 4: 0.0, 3: 0.5, 0: 0.5}
    from src.meta.calibration import _convolve_point_classes
    joint = _convolve_point_classes([a, b])
    assert sum(joint.values()) == pytest.approx(1.0)
    # clase conjunta 0 = solo 0+0
    assert joint[0] == pytest.approx(0.4 * 0.5)
    # clase conjunta 6 = 6+0 (0.1·0.5) + 3+3 (0.3·0.5) — colisión sumada
    assert joint[6] == pytest.approx(0.1 * 0.5 + 0.3 * 0.5)
    # soporte alcanzable para dos partidos
    assert set(joint) <= {0, 3, 4, 6, 7, 8, 9, 10, 12}


def test_joint_point_classes_support():
    from src.meta.calibration import _joint_point_classes
    assert _joint_point_classes(1) == (0, 3, 4, 6)
    assert _joint_point_classes(2) == (0, 3, 4, 6, 7, 8, 9, 10, 12)


def test_predicted_group_shares_reduces_to_single():
    """Grupo de 1 ≡ predicted_shares de siempre."""
    from src.meta.calibration import predicted_group_shares
    cfg = PoolModelConfig()
    g = _grid()
    single = predicted_shares(g, (1, 0), cfg, no_show_frac=0.1)
    group = predicted_group_shares([g], [(1, 0)], cfg, no_show_frac=0.1)
    assert group == pytest.approx(single)


def test_predicted_group_shares_two_matches_sums_to_one():
    from src.meta.calibration import predicted_group_shares
    shares = predicted_group_shares(
        [_grid(1.8, 0.7), _grid(1.2, 1.3)], [(2, 0), (1, 1)],
        PoolModelConfig(), no_show_frac=0.08,
    )
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(shares) <= {0, 3, 4, 6, 7, 8, 9, 10, 12}


def test_simultaneous_matches_become_joint_observation(tmp_path, monkeypatch):
    """Partidos simultáneos (3ª jornada de grupos): el scheduler toma UN snapshot con
    finished=todos. Antes se descartaban; ahora el grupo co-puntuado produce UNA
    observación conjunta cuyas shares viven en las clases SUMA. La cadena sigue limpia:
    105 y 108 (solos) dan observaciones normales."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def _setup_match(mid):
        pdir = tmp_path / "predictions" / mid
        pdir.mkdir(parents=True)
        (pdir / "v1_20260624T000000Z.json").write_text(json.dumps({
            "constraints": {"lambda_L": 1.5, "lambda_V": 0.9, "lambda_12": 0.1}}))
        pmdir = tmp_path / "postmortems"
        pmdir.mkdir(exist_ok=True)
        (pmdir / f"{mid}.json").write_text(json.dumps({"actual_home": 1, "actual_away": 0}))

    def _entries(points, cw):
        # cw = correct_winners acumulados (todas las pencas aciertan ganador cada partido)
        return [{"penca_id": i, "points_total": points, "exact_scores": 0,
                 "correct_winners": cw, "predictions_made": 1} for i in range(100)]

    # Partido 105 solo → observación normal (baseline 0, todos +3, 1 ganador)
    _setup_match("105")
    snapshot_leaderboard("105", ["105"], entries=_entries(3, 1))

    # 106 y 107 en el mismo tick → UN snapshot con finished={105,106,107}. Delta vs 105:
    # +6 puntos (3+3) y +2 ganadores. Clase conjunta válida (6 = 3+3), modelada por convolución.
    _setup_match("106")
    _setup_match("107")
    snapshot_leaderboard("107", ["105", "106", "107"], entries=_entries(9, 3))

    # 108 solo después → observación normal usando el snapshot del grupo como predecesor
    _setup_match("108")
    snapshot_leaderboard("108", ["105", "106", "107", "108"], entries=_entries(15, 4))

    obs = build_observations(tmp_path)
    ids = [o.match_id for o in obs]
    assert ids == ["105", "106+107", "108"]
    by_id = {o.match_id: o for o in obs}
    assert by_id["105"].shares[3] == pytest.approx(1.0)
    # observación conjunta: 2 partidos, delta combinado +6 cae en la clase conjunta 6
    joint = by_id["106+107"]
    assert joint.n_matches == 2
    assert joint.actuals == ((1, 0), (1, 0))
    assert joint.shares[6] == pytest.approx(1.0)
    assert joint.winner_frac == pytest.approx(1.0)  # ambos ganadores acertados
    assert by_id["108"].shares[6] == pytest.approx(1.0)  # delta 15−9, sin contaminación


def _synthetic_group_observations(true_chalk, true_beta, true_no_show, n_pool=400, seed=11):
    """Como _synthetic_observations pero cada observación es un GRUPO de 2 partidos
    co-puntuados: por penca se decide UN no-show (0 en ambos), si no se pica de cada Q y
    se suman los puntos. Fiel a cómo el leaderboard junta deltas de partidos simultáneos."""
    from src.meta.calibration import _joint_point_classes
    rng = np.random.default_rng(seed)
    cfg = _config_for(true_chalk, true_beta)
    pairs = [((2.0, 0.6), (1.5, 1.1)), ((1.2, 1.3), (1.9, 0.8)), ((0.9, 1.6), (1.6, 0.7))]
    valid = _joint_point_classes(2)
    obs = []
    for gi, (lA, lB) in enumerate(pairs):
        gridA, gridB = _grid(*lA), _grid(*lB)
        n = gridA.shape[0]
        qA = pool_pick_distribution(gridA, cfg).flatten()
        qB = pool_pick_distribution(gridB, cfg).flatten()
        oA = rng.choice(n * n, p=gridA.flatten() / gridA.sum())
        oB = rng.choice(n * n, p=gridB.flatten() / gridB.sum())
        actA, actB = (oA // n, oA % n), (oB // n, oB % n)
        awA = "H" if actA[0] > actA[1] else ("A" if actA[0] < actA[1] else "D")
        awB = "H" if actB[0] > actB[1] else ("A" if actB[0] < actB[1] else "D")
        counts = {c: 0 for c in valid}
        exact_hits = winner_hits = 0
        for _ in range(n_pool):
            if rng.random() < true_no_show:
                counts[0] += 1
                continue
            pa, pb = rng.choice(n * n, p=qA), rng.choice(n * n, p=qB)
            pka, pkb = (pa // n, pa % n), (pb // n, pb % n)
            counts[jmlm_points(pka, actA) + jmlm_points(pkb, actB)] += 1
            exact_hits += int(pka == actA) + int(pkb == actB)
            pwa = "H" if pka[0] > pka[1] else ("A" if pka[0] < pka[1] else "D")
            pwb = "H" if pkb[0] > pkb[1] else ("A" if pkb[0] < pkb[1] else "D")
            winner_hits += int(pwa == awA) + int(pwb == awB)
        shares = {c: counts[c] / n_pool for c in valid}
        obs.append(Observation(match_id=f"GRP_{gi}", actuals=(actA, actB), shares=shares,
                               n_entries=n_pool, grids=(gridA, gridB),
                               exact_frac=exact_hits / (n_pool * 2),
                               winner_frac=winner_hits / (n_pool * 2)))
    return obs


def test_grouped_observations_improve_over_prior():
    """Observaciones de grupos de 2 partidos co-puntuados, por sí solas, ya mejoran el
    loss vs el prior y recuperan chalk. (β se diluye en la convolución — su señal vive en
    la concentración por marcador; ver el test mixto para la recuperación completa.)"""
    true = dict(chalk=1.2, beta=0.5, ns=0.10)
    obs = _synthetic_group_observations(true["chalk"], true["beta"], true["ns"])
    fit = calibrate(obs)
    assert fit is not None
    assert fit["loss"] < fit["prior_loss"]
    assert abs(fit["chalk_strength"] - true["chalk"]) < abs(PRIOR_CHALK - true["chalk"])


def test_calibration_recovers_params_with_mixed_singles_and_groups():
    """Escenario real del torneo: jornadas 1-2 individuales + jornada 3 simultánea (grupos).
    Con singles + grupos juntos la calibración recupera chalk y β mejor que el prior — los
    grupos suman datos sin romper la recuperación."""
    true = dict(chalk=1.2, beta=0.5, ns=0.10)
    singles = _synthetic_observations(true["chalk"], true["beta"], true["ns"], n_matches=6)
    groups = _synthetic_group_observations(true["chalk"], true["beta"], true["ns"])
    fit = calibrate(singles + groups)
    assert fit is not None
    assert fit["loss"] < fit["prior_loss"]
    assert abs(fit["chalk_strength"] - true["chalk"]) < abs(PRIOR_CHALK - true["chalk"])
    assert abs(fit["bias_scale"] - true["beta"]) < abs(PRIOR_BIAS_SCALE - true["beta"])


# -------------------- sesgo de popularidad orientado al favorito --------------------

def test_bias_se_refleja_cuando_el_favorito_es_visitante():
    """La tabla está en clave local-favorito; con favorito visitante hay que reflejarla.

    Medido el 2026-08-09 sobre 2.739 picks reales: en Torque–Peñarol (Peñarol
    favorito de visita) el sesgo empírico del 1-0 era 0,07 y el modelo le daba 1,58.
    Sin reflejar, el modelo empuja la Q hacia marcadores de local justo donde el pool
    hace lo contrario.
    """
    import numpy as np
    from src.clausura.economics import score_index
    from src.clausura.pool import PoolConfig, pool_distribution
    from src.model.poisson import score_grid

    local_fav = score_grid(1.9, 0.8, 0.0, max_goals=5)     # local favorito
    visita_fav = score_grid(0.8, 1.9, 0.0, max_goals=5)    # visitante favorito

    q_lf = pool_distribution(local_fav, PoolConfig())
    q_vf = pool_distribution(visita_fav, PoolConfig())

    # con favorito local, el 1-0 (gana el favorito por 1) está premiado
    assert q_lf[score_index(1, 0)] > q_lf[score_index(0, 1)]
    # con favorito visitante, el premiado tiene que ser el 0-1, no el 1-0
    assert q_vf[score_index(0, 1)] > q_vf[score_index(1, 0)]

    # y el efecto es simétrico: la Q reflejada del uno es la del otro
    espejo = q_vf.reshape(6, 6).T.reshape(-1)
    assert np.allclose(q_lf, espejo, atol=1e-12)


def test_orientar_al_favorito_se_puede_apagar():
    """Reproducir el comportamiento previo al 2026-08-09.

    Se compara el sesgo RELATIVO AL MERCADO, no la Q cruda: el sesgo es un
    multiplicador sobre la probabilidad de mercado, y con favorito visitante el
    mercado por sí solo ya hace que 0-1 tenga más masa que 1-0. Lo que distingue a
    los dos modos es hacia dónde empujan RESPECTO del mercado.

    OJO con la fórmula: el sesgo se despeja como Q/P**chalk, no como Q/P. Con
    `q ∝ p**a · bias` el cociente Q/P vale `p**(a-1) · bias`, así que en cuanto el
    chalk dejó de ser 1.0 (pasó a 2.2 el 2026-08-11) arrastra un término de mercado
    que en un partido desparejo domina al sesgo — este test falló por eso, midiendo
    0.49 donde esperaba >1.2, sin que la orientación tuviera nada roto.
    """
    from src.clausura.economics import flatten_grid, score_index
    from src.clausura.pool import PoolConfig, pool_distribution
    from src.model.poisson import score_grid

    visita_fav = score_grid(0.8, 1.9, 0.0, max_goals=5)
    p = flatten_grid(visita_fav)
    i10, i01 = score_index(1, 0), score_index(0, 1)

    def sesgo_rel(cfg):
        q = pool_distribution(visita_fav, cfg)
        pa = p ** cfg.chalk_strength
        return (q[i10] / pa[i10]) / (q[i01] / pa[i01])

    for chalk in (1.0, 2.2):
        # apagado: empuja al 1-0 (tabla en clave local) aunque el favorito sea visita
        apagado = PoolConfig(chalk_strength=chalk, orientar_al_favorito=False)
        assert sesgo_rel(apagado) > 1.2, f"chalk={chalk}"
        # prendido: empuja hacia el 0-1, que es el "gana el favorito por 1"
        prendido = PoolConfig(chalk_strength=chalk, orientar_al_favorito=True)
        assert sesgo_rel(prendido) < 0.9, f"chalk={chalk}"


def test_partido_parejo_no_rompe_la_orientacion():
    """Con λ iguales no hay favorito; la Q tiene que quedar bien formada igual."""
    import numpy as np
    from src.clausura.pool import PoolConfig, pool_distribution
    from src.model.poisson import score_grid

    q = pool_distribution(score_grid(1.2, 1.2, 0.0, max_goals=5), PoolConfig())
    assert np.isfinite(q).all() and abs(q.sum() - 1.0) < 1e-9

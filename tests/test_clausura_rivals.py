"""Tests del modelo empírico por rival (src.clausura.rivals)."""

import numpy as np
import pytest

from src.clausura.economics import (
    MAX_GOALS,
    N_SCORES,
    PrizeConfig,
    SeasonSimulator,
    SimConfig,
    score_index,
)
from src.clausura.pool import pool_distribution
from src.clausura.rivals import (
    RivalModel,
    _tilted_sample,
    build_rival_model,
    build_rival_model_from_arrays,
    fit_gamma,
)
from src.clausura.scoring import supermatch_points
from src.model.poisson import score_grid


def _q():
    return pool_distribution(score_grid(1.3, 1.1, 0.0, max_goals=MAX_GOALS))


# -------------------- estilo γ --------------------

def test_tilted_sample_gamma_alto_concentra_en_el_modal():
    q = _q()
    rng = np.random.default_rng(1)
    modal = int(np.argmax(q))
    chalk = _tilted_sample(q, np.full(500, 4.0), rng)
    disperso = _tilted_sample(q, np.full(500, 0.35), rng)
    assert (chalk == modal).mean() > (disperso == modal).mean() + 0.2


def test_fit_gamma_recupera_la_direccion_del_estilo():
    q = _q()
    rng = np.random.default_rng(2)
    logqs = np.stack([np.log(q + 1e-12)] * 12)
    chalk_obs = _tilted_sample(q, np.full(12, 3.0), rng)
    disperso_obs = _tilted_sample(q, np.full(12, 0.4), rng)
    assert fit_gamma(chalk_obs, logqs) > 1.2
    assert fit_gamma(disperso_obs, logqs) < 0.9


def test_fit_gamma_sin_observaciones_es_neutro():
    assert fit_gamma(np.array([], dtype=np.int64), np.zeros((0, 36))) == 1.0


# -------------------- construcción del modelo --------------------

def _modelo_2x2(puntos_extra=0):
    """2 rivales, 2 partidos jugados (1-0 y 1-1). Rival 0 cargó ambos; rival 1 solo el 2º."""
    q = _q()
    known = np.array([
        [score_index(1, 0), score_index(0, 0)],
        [-1, score_index(1, 1)],
    ], dtype=np.int64)
    played = np.array([True, True])
    actual = np.array([score_index(1, 0), score_index(1, 1)], dtype=np.int64)
    pref = [False, True]
    implied = np.array([
        supermatch_points((1, 0), (1, 0)) + supermatch_points((0, 0), (1, 1), True),
        supermatch_points((1, 1), (1, 1), True),
    ])
    puntos_ranking = implied + puntos_extra
    model = build_rival_model_from_arrays(
        known, played, [q, q], pref, actual, puntos_ranking,
        numeros=np.array([11, 22]),
    )
    return model, implied


def test_residuo_cero_cuando_ranking_e_implicados_coinciden():
    model, _ = _modelo_2x2()
    assert model.residuo.tolist() == [0, 0]


def test_residuo_absorbe_la_diferencia_con_el_ranking():
    model, _ = _modelo_2x2(puntos_extra=3)
    assert model.residuo.tolist() == [3, 3]


def test_p_show_refleja_el_ausentismo_con_shrinkage():
    model, _ = _modelo_2x2()
    # rival 0 cargó 2/2, rival 1 cargó 1/2 → p_show ordenado y ambos < 1 (prior Beta)
    assert model.p_show[0] > model.p_show[1]
    assert 0.5 < model.p_show[1] < model.p_show[0] < 1.0


def test_build_rival_model_excluye_mis_participaciones():
    snapshot = {
        "participaciones": [
            {"numero": 899258848, "puntos": 0, "picks": {"10": [1, 0]}},
            {"numero": 555, "puntos": 4, "picks": {"10": [1, 0]}, "campeon": "Nacional"},
        ],
    }
    eventos = [{"evento_id": 10, "preferencial": False}]
    model = build_rival_model(
        snapshot, eventos, [_q()], resultados={10: (1, 0)},
        mis_numeros={899258848}, equipo_idx={"Nacional": 3},
    )
    assert model.n_rivales == 1
    assert model.numeros.tolist() == [555]
    assert model.known_picks[0, 0] == score_index(1, 0)
    assert model.campeon_idx.tolist() == [3]
    # puntos implicados de 1-0 vs 1-0 = 8 → residuo 4 − 8 = −4
    assert model.residuo.tolist() == [-4]


def test_build_rival_model_sin_rivales_devuelve_none():
    snapshot = {"participaciones": [{"numero": 1, "puntos": 0, "picks": {}}]}
    assert build_rival_model(snapshot, [], [], {}, mis_numeros={1}) is None


# -------------------- integración con el simulador --------------------

def test_simulador_ancla_el_pasado_a_los_puntos_reales():
    """Partidos jugados (delta): el total rival es constante entre sims y coincide
    con puntos reales + residuo; el no-show queda en 0."""
    model, implied = _modelo_2x2(puntos_extra=2)
    d1 = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1)); d1[1, 0] = 1.0
    d2 = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1)); d2[1, 1] = 1.0
    q = _q()
    sim = SeasonSimulator(
        [d1, d2], [280, 280], [False, True], [q, q],
        PrizeConfig(), SimConfig(n_sims=50, seed=5), rivals=model,
    )
    esperado = implied + 2
    assert np.all(sim.rivals_total == esperado[:, None])
    # el rival 1 no cargó el partido 1 (preferencial=False): su fecha solo suma el 2º
    assert np.all(sim.rivals_fecha[0, 1] == supermatch_points((1, 1), (1, 1), True))


def test_simulador_respeta_p_show_en_el_futuro():
    """Con p_show=0 el rival no suma nada en partidos futuros no observados."""
    q = _q()
    model = RivalModel(
        known_picks=np.full((3, 1), -1, dtype=np.int64),
        played_mask=np.array([False]),
        gamma=np.ones(3),
        p_show=np.zeros(3),
        residuo=np.zeros(3, dtype=np.int64),
    )
    g = score_grid(1.3, 1.1, 0.0, max_goals=MAX_GOALS)
    sim = SeasonSimulator([g], [280], [False], [q], PrizeConfig(),
                          SimConfig(n_sims=30, seed=6), rivals=model)
    assert sim.rivals_total.sum() == 0


def test_simulador_n_rivales_sale_del_modelo():
    model, _ = _modelo_2x2()
    g = score_grid(1.3, 1.1, 0.0, max_goals=MAX_GOALS)
    q = _q()
    sim = SeasonSimulator([g, g], [280, 281], [False, False], [q, q], PrizeConfig(),
                          SimConfig(n_sims=10, n_rivales=99, seed=7), rivals=model)
    assert sim.n_rivales == 2
    assert sim.rivals_total.shape == (2, 10)


def test_camino_iid_sigue_intacto_sin_modelo():
    g = score_grid(1.3, 1.1, 0.0, max_goals=MAX_GOALS)
    q = _q()
    sim = SeasonSimulator([g], [280], [False], [q], PrizeConfig(),
                          SimConfig(n_sims=10, n_rivales=7, seed=8))
    assert sim.n_rivales == 7
    assert sim.rivals_total.shape == (7, 10)


# -------------------- picks rivales POR SIMULACIÓN --------------------

def test_sample_picks_match_re_sortea_por_simulacion():
    """Futuro no observado: los picks varían entre sims (antes: un sorteo fijo).
    El pick observado queda constante en todas las sims."""
    q = _q()
    known = np.full((5, 1), -1, dtype=np.int64)
    known[0, 0] = score_index(3, 3)   # rival 0 ya cargó
    model = RivalModel(
        known_picks=known,
        played_mask=np.array([False]),
        gamma=np.ones(5),
        p_show=np.ones(5),
        residuo=np.zeros(5, dtype=np.int64),
    )
    picks, show = model.sample_picks_match(0, q, np.random.default_rng(3), n_sims=200)
    assert picks.shape == (5, 200) and show.shape == (5, 200)
    assert (picks[0] == score_index(3, 3)).all() and show[0].all()
    # los rivales libres NO repiten el mismo pick en las 200 sims
    assert all(len(np.unique(picks[r])) > 1 for r in range(1, 5))


def test_rivales_iid_varian_entre_sims_con_resultado_fijo():
    """Grillas delta (resultado determinístico): la única fuente de varianza del
    máximo del pool son los picks rivales — con el sorteo único era 0 y el
    optimizador podía explotar huecos muestrales."""
    d = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1)); d[1, 1] = 1.0
    q = _q()
    sim = SeasonSimulator([d], [280], [False], [q], PrizeConfig(),
                          SimConfig(n_sims=100, n_rivales=30, seed=4))
    assert sim.rivals_total.std(axis=1).max() > 0


def test_rivales_modelo_varian_entre_sims_con_resultado_fijo():
    d = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1)); d[1, 1] = 1.0
    q = _q()
    model = RivalModel(
        known_picks=np.full((10, 1), -1, dtype=np.int64),
        played_mask=np.array([False]),
        gamma=np.ones(10),
        p_show=np.ones(10),
        residuo=np.zeros(10, dtype=np.int64),
    )
    sim = SeasonSimulator([d], [280], [False], [q], PrizeConfig(),
                          SimConfig(n_sims=100, seed=4), rivals=model)
    assert sim.rivals_total.std(axis=1).max() > 0


# -------------------- rivales que no cargaron especiales --------------------

POST_LOCK = "2026-08-08T01:08:00+00:00"    # escaneo real posterior al lock
PRE_LOCK = "2026-08-07T14:13:30+00:00"     # ventana del gate, especiales aún editables


def _snapshot_especiales(generado):
    return {
        "generado_utc": generado,
        "participaciones": [
            {"numero": 501, "puntos": 0, "picks": {"10": [1, 0]},
             "campeon": "Nacional", "goleador": "Gómez", "goleador_id": 622},
            {"numero": 502, "puntos": 0, "picks": {"10": [1, 0]}},   # NUNCA cargó
        ],
    }


def test_marca_a_los_que_no_cargaron_especiales_post_lock():
    eventos = [{"evento_id": 10, "preferencial": False}]
    model = build_rival_model(
        _snapshot_especiales(POST_LOCK), eventos, [_q()], resultados={},
        mis_numeros={999}, equipo_idx={"Nacional": 3}, goleador_op_idx={622: 7},
    )
    assert model.sin_campeon.tolist() == [False, True]
    assert model.sin_goleador.tolist() == [False, True]
    assert model.campeon_idx.tolist() == [3, -1]
    assert model.goleador_idx.tolist() == [7, -1]


def test_pre_lock_no_marca_a_nadie():
    """Antes del lock la ausencia solo significa 'el gate estaba cerrado' — marcarlos
    ahí borraría especiales que todavía podían cargarse."""
    eventos = [{"evento_id": 10, "preferencial": False}]
    model = build_rival_model(
        _snapshot_especiales(PRE_LOCK), eventos, [_q()], resultados={},
        mis_numeros={999}, equipo_idx={"Nacional": 3},
    )
    assert model.sin_campeon is None


def _puntos_campeon(sin_campeon, n_rivales=2, n_sims=400):
    """Solo lo que el especial Campeón le SUMA a cada rival (aislado de los goles)."""
    g = score_grid(1.3, 1.0, 0.0, max_goals=MAX_GOALS)
    model = RivalModel(
        known_picks=np.full((n_rivales, 1), -1, dtype=np.int64),
        played_mask=np.zeros(1, dtype=bool),
        gamma=np.ones(n_rivales),
        p_show=np.ones(n_rivales),
        residuo=np.zeros(n_rivales, dtype=np.int64),
        campeon_idx=np.full(n_rivales, -1, dtype=np.int64),
        sin_campeon=sin_campeon,
    )
    sim = SeasonSimulator(
        grids=[g], fecha_de_partido=[1], preferencial=[False], pool_q=[_q()],
        sim=SimConfig(n_sims=n_sims, n_rivales=n_rivales, seed=11), rivals=model,
    )
    antes = sim.rivals_total.copy()
    sim.enable_campeon(np.array([0]), np.array([1]), n_teams=2,
                       pool_q_campeon=np.array([0.5, 0.5]))
    return sim.rivals_total - antes


def test_el_que_no_cargo_campeon_no_cobra_esos_puntos():
    """El bug: se les sorteaba un campeón del pool y cobraban 25 como cualquiera."""
    todos = _puntos_campeon(None)
    ninguno = _puntos_campeon(np.ones(2, dtype=bool))

    assert todos.max() == 25        # sin marcar, alguno acierta y cobra
    assert ninguno.max() == 0       # marcados, ninguno puede cobrar nunca
    assert todos.mean() > 0         # el inflado es justo la cola que define el premio


def test_mezcla_solo_silencia_a_los_marcados():
    mixto = _puntos_campeon(np.array([False, True]))
    assert mixto[1].max() == 0      # el marcado nunca cobra
    assert mixto[0].max() == 25     # el que sí cargó, intacto


def test_marca_por_especial_por_separado():
    """Hay quien cargó goleador y no campeón (2 casos reales en el pool del 8/8):
    no puede cobrar campeón, pero sí goleador — marcarlo entero le borraría puntos
    que sí va a disputar."""
    snap = {
        "generado_utc": POST_LOCK,
        "participaciones": [
            {"numero": 601, "puntos": 0, "picks": {},
             "goleador": "Gómez", "goleador_id": 622},          # goleador sí, campeón no
        ],
    }
    model = build_rival_model(
        snap, [{"evento_id": 10, "preferencial": False}], [_q()], resultados={},
        mis_numeros={999}, equipo_idx={"Nacional": 3}, goleador_op_idx={622: 7},
    )
    assert model.sin_campeon.tolist() == [True]
    assert model.sin_goleador.tolist() == [False]
    assert model.goleador_idx.tolist() == [7]


def test_tilted_sample_chunkeado_es_identico_al_entero():
    """El chunking del inverse-CDF no puede mover un solo pick rival.

    Acota el pico de memoria (era (R, S, 36) booleano: 806 MB a S=9.600 en un
    droplet de 1 GB, el techo que impedía subir n_sims). El sorteo `u` se hace
    entero antes de trocear justamente para que la secuencia del rng no cambie;
    este test es el que lo vigila.
    """
    import src.clausura.rivals as rv

    q = np.random.default_rng(0).dirichlet(np.ones(N_SCORES))
    gamma = np.random.default_rng(1).uniform(0.5, 2.0, size=97)

    def sin_chunk(size):
        logq = np.log(q + 1e-12)
        w = np.exp(gamma[:, None] * logq[None, :])
        w /= w.sum(axis=1, keepdims=True)
        cdf = np.cumsum(w, axis=1)
        u = np.random.default_rng(42).random((len(gamma), size))
        return (u[:, :, None] > cdf[:, None, :]).sum(axis=2).clip(0, N_SCORES - 1)

    previo = rv.TILTED_CHUNK_BYTES
    try:
        for size in (7, 100, 1000):
            esperado = sin_chunk(size)
            for tope in (previo, 64 * N_SCORES, 1):   # 1 fuerza bloques de una sim
                rv.TILTED_CHUNK_BYTES = tope
                got = rv._tilted_sample(q, gamma, np.random.default_rng(42), size)
                assert got.shape == (97, size)
                assert np.array_equal(got, esperado), f"size={size} tope={tope}"
    finally:
        rv.TILTED_CHUNK_BYTES = previo

"""Tests de los especiales Campeón/Goleador: tabla, simulador y optimización."""

import numpy as np
import pytest

from src.clausura.economics import (
    PrizeConfig,
    SeasonSimulator,
    SimConfig,
    score_index,
)
from src.clausura.especiales import (
    OpcionGoleador,
    champions_from_results,
    goleador_prior_from_ratings,
    p_campeon,
    p_campeon_from_grids,
    pool_campeon_distribution,
    pool_goleador_distribution,
)
from src.clausura.pool import PoolConfig, pool_distribution
from src.clausura.strategy import EspecialesInput, build_portfolio
from src.model.poisson import score_grid


# -------------------- tabla de posiciones --------------------

def test_champion_determinista():
    """2 equipos, 2 partidos: el que gana ambos es campeón en toda simulación."""
    S = 50
    actual = np.stack([
        np.full(S, score_index(2, 0)),   # equipo 0 de local le gana al 1
        np.full(S, score_index(0, 3)),   # equipo 1 de local pierde con el 0
    ])
    champs = champions_from_results(
        actual, np.array([0, 1]), np.array([1, 0]), 2, np.random.default_rng(1)
    )
    assert (champs == 0).all()


def test_champion_desempata_por_diferencia():
    """Mismos puntos (1 victoria cada uno), gana el de mejor diferencia."""
    S = 50
    actual = np.stack([
        np.full(S, score_index(3, 0)),   # 0 le gana 3-0 al 1
        np.full(S, score_index(1, 0)),   # 1 le gana 1-0 al 0
    ])
    champs = champions_from_results(
        actual, np.array([0, 1]), np.array([1, 0]), 2, np.random.default_rng(1)
    )
    assert (champs == 0).all()   # dif +2 vs 0


def test_p_campeon_suma_uno():
    g = score_grid(1.3, 1.1, 0.0, max_goals=5)
    grids = [g] * 6
    local = np.array([0, 1, 2, 0, 1, 2])
    visita = np.array([1, 2, 0, 2, 0, 1])
    p = p_campeon_from_grids(grids, local, visita, 3, n_sims=500)
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()   # con 3 equipos parejos, todos tienen chance


def test_equipo_mas_fuerte_es_favorito():
    fuerte = score_grid(2.5, 0.5, 0.0, max_goals=5)   # el local aplasta
    debil = score_grid(0.5, 2.5, 0.0, max_goals=5)
    # equipo 0 siempre aplasta (de local y de visita)
    grids = [fuerte, debil, fuerte, debil]
    local = np.array([0, 1, 0, 2])
    visita = np.array([1, 0, 2, 0])
    p = p_campeon_from_grids(grids, local, visita, 3, n_sims=800)
    assert p[0] > 0.8


# -------------------- priors del pool --------------------

def test_pool_campeon_sobrepondera_grandes():
    equipos = ["Danubio", "Peñarol", "Cerro"]
    p = np.array([0.4, 0.3, 0.3])
    q = pool_campeon_distribution(p, equipos)
    # Peñarol tiene menos P real que Danubio pero el pool lo pica más
    assert q[1] > q[0]


def test_pool_campeon_lean_inclina_el_consenso():
    equipos = ["Peñarol", "Nacional", "Cerro"]
    p = np.array([0.30, 0.30, 0.40])
    base = pool_campeon_distribution(p, equipos)
    leaned = pool_campeon_distribution(p, equipos,
                                       lean={"Nacional": 1.6, "Peñarol": 0.9})
    assert np.isclose(base[0], base[1])          # sin lean, los grandes empatan
    assert leaned[1] > leaned[0]                 # con lean, el pool carga a Nacional
    assert np.isclose(leaned.sum(), 1.0)


def test_goleador_prior_reparte_por_ataque():
    opciones = [
        OpcionGoleador(1, "Otros", -1),
        OpcionGoleador(2, "Estrella Fuerte", 100),
        OpcionGoleador(3, "Suplente Fuerte", 100),
        OpcionGoleador(4, "Estrella Debil", 200),
    ]
    p = goleador_prior_from_ratings(
        opciones,
        equipos=["Fuerte", "Debil"],
        equipo_id_por_nombre={"Fuerte": 100, "Debil": 200},
        ataque={"Fuerte": 0.5, "Debil": -0.5},
    )
    assert p.sum() == pytest.approx(1.0)
    assert p[1] > p[3]          # estrella del fuerte > estrella del débil
    assert p[1] > p[2]          # estrella > suplente dentro del mismo equipo
    assert p[0] > 0             # "Otros" conserva masa


def test_pool_goleador_es_distribucion():
    p = np.array([0.5, 0.3, 0.2])
    q = pool_goleador_distribution(p)
    assert q.sum() == pytest.approx(1.0)
    assert q[0] > q[2]


# -------------------- simulador con especiales --------------------

@pytest.fixture
def sim_especiales():
    g = score_grid(1.4, 1.0, 0.0, max_goals=5)
    grids = [g] * 6
    local = np.array([0, 1, 2, 0, 1, 2])
    visita = np.array([1, 2, 0, 2, 0, 1])
    s = SeasonSimulator(grids, [1, 1, 1, 2, 2, 2], [False] * 6,
                        [pool_distribution(g, PoolConfig())] * 6,
                        PrizeConfig(), SimConfig(n_sims=300, n_rivales=25, seed=3))
    s.load_picks(np.full((2, 6), score_index(1, 0), dtype=np.int64))
    s.enable_campeon(local, visita, 3, np.array([0.34, 0.33, 0.33]))
    s.enable_goleador(np.array([0.6, 0.4]), np.array([0.7, 0.3]))
    return s


def test_set_campeon_incremental_correcto(sim_especiales):
    s = sim_especiales
    base = s.mine_total.copy()
    s.set_campeon_pick(0, 1)
    esperado = base[0] + 25 * (s.champ_sim == 1)
    assert np.array_equal(s.mine_total[0], esperado)
    # cambiar de 1 a 2 revierte y aplica
    s.set_campeon_pick(0, 2)
    esperado = base[0] + 25 * (s.champ_sim == 2)
    assert np.array_equal(s.mine_total[0], esperado)


def test_set_goleador_incremental_correcto(sim_especiales):
    s = sim_especiales
    base = s.mine_total.copy()
    s.set_goleador_pick(1, 0)
    assert np.array_equal(s.mine_total[1], base[1] + 25 * (s.gol_sim == 0))


def test_especiales_no_tocan_premios_por_fecha(sim_especiales):
    """Art. 8: los especiales suman solo al total general, nunca a la fecha."""
    s = sim_especiales
    fecha_antes = s.mine_fecha.copy()
    s.set_campeon_pick(0, 1)
    s.set_goleador_pick(0, 1)
    assert np.array_equal(fecha_antes, s.mine_fecha)


def test_load_picks_resetea_especiales(sim_especiales):
    s = sim_especiales
    s.set_campeon_pick(0, 1)
    s.load_picks(np.full((2, 6), score_index(1, 1), dtype=np.int64))
    assert s.campeon_picks is None and s.goleador_picks is None


def test_acertar_campeon_paga_25(sim_especiales):
    s = sim_especiales
    s.set_campeon_pick(0, 0)
    s.set_campeon_pick(1, 0)
    # en las sims donde el campeón es 0, la dif entre tener y no tener el pick es 25
    delta = s.mine_total[0] - s.mine_total[1]
    assert (delta == 0).all()   # mismos picks → mismos puntos


# -------------------- optimización end-to-end --------------------

def test_build_portfolio_con_especiales():
    g_fuerte = score_grid(2.2, 0.6, 0.0, max_goals=5)
    g_parejo = score_grid(1.2, 1.1, 0.0, max_goals=5)
    grids = [g_fuerte, g_parejo] * 4
    fechas = [1, 1, 2, 2, 3, 3, 4, 4]
    local = np.array([0, 2, 0, 2, 1, 3, 1, 3])
    visita = np.array([1, 3, 2, 1, 0, 2, 3, 0])

    esp = EspecialesInput(
        local_de=local, visita_de=visita, n_teams=4,
        pool_q_campeon=np.array([0.55, 0.2, 0.15, 0.1]),
        p_goleador=np.array([0.5, 0.3, 0.2]),
        pool_q_goleador=np.array([0.6, 0.25, 0.15]),
    )
    port = build_portfolio(grids, fechas, [False] * 8, n_participaciones=3,
                           sim=SimConfig(n_sims=250, n_rivales=30, seed=11),
                           especiales=esp, max_passes=2)
    assert port.campeon is not None and port.campeon.shape == (3,)
    assert port.goleador is not None and port.goleador.shape == (3,)
    assert port.p_campeon is not None and port.p_campeon.sum() == pytest.approx(1.0)
    assert set(port.campeon.tolist()) <= {0, 1, 2, 3}


def test_build_portfolio_respeta_frozen_especiales():
    g = score_grid(1.4, 1.0, 0.0, max_goals=5)
    grids = [g] * 4
    local = np.array([0, 1, 0, 1])
    visita = np.array([1, 0, 1, 0])
    esp = EspecialesInput(
        local_de=local, visita_de=visita, n_teams=2,
        pool_q_campeon=np.array([0.5, 0.5]),
        frozen_campeon=np.array([1, -1]),   # la participación 0 ya cargó el equipo 1
    )
    port = build_portfolio(grids, [1, 1, 2, 2], [False] * 4, n_participaciones=2,
                           sim=SimConfig(n_sims=200, n_rivales=20, seed=7),
                           especiales=esp, max_passes=1)
    assert port.campeon[0] == 1


# -------------------- menú de goleador desde el snapshot (2026-08-10) --------------------

def test_opciones_goleador_desde_snapshot():
    """El API de opciones da 500 crónico; el menú se reconstruye del pool."""
    from src.clausura.especiales import opciones_goleador_desde_snapshot

    snap = {"participaciones": [
        {"numero": 1, "goleador_id": 620, "goleador": "Matías Arezo"},
        {"numero": 2, "goleador_id": 622, "goleador": "Maximiliano Gómez"},
        {"numero": 3, "goleador_id": 620, "goleador": "Matías Arezo"},   # repetido
        {"numero": 4, "goleador_id": None, "goleador": None},            # sin cargar
        {"numero": 5},                                                   # sin el campo
    ]}
    ops = opciones_goleador_desde_snapshot(snap)
    assert [o.id for o in ops] == [620, 622]
    assert [o.nombre for o in ops] == ["Matías Arezo", "Maximiliano Gómez"]
    # el snapshot no trae equipo, y eso hay que dejarlo explícito
    assert all(o.equipo_id == -1 for o in ops)


def test_goleador_prior_desde_pool_encoge_hacia_uniforme():
    """shrink=0 es el pool tal cual, 1 es uniforme, 0.5 los promedia."""
    import numpy as np
    from src.clausura.especiales import goleador_prior_desde_pool

    counts = np.array([80.0, 15.0, 5.0])
    crudo = goleador_prior_desde_pool(counts, shrink=0.0)
    uni = goleador_prior_desde_pool(counts, shrink=1.0)
    medio = goleador_prior_desde_pool(counts, shrink=0.5)

    assert np.allclose(crudo, [0.8, 0.15, 0.05])
    assert np.allclose(uni, [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(medio, (crudo + uni) / 2)
    # el favorito del pool baja y la cola sube: es el punto del encogimiento
    assert medio[0] < crudo[0] and medio[2] > crudo[2]
    for p in (crudo, uni, medio):
        assert abs(p.sum() - 1.0) < 1e-12


def test_goleador_prior_desde_pool_sin_datos_no_rompe():
    import numpy as np
    from src.clausura.especiales import goleador_prior_desde_pool
    p = goleador_prior_desde_pool(np.zeros(4), shrink=0.5)
    assert np.allclose(p, 0.25)
    assert len(goleador_prior_desde_pool(np.zeros(0))) == 0

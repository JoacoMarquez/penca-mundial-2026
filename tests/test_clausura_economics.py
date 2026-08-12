"""Tests del simulador económico: reparto de premios y estado incremental."""

import numpy as np
import pytest

from src.clausura.economics import (
    MAX_GOALS,
    N_SCORES,
    PrizeConfig,
    SeasonSimulator,
    SimConfig,
    flatten_grid,
    index_score,
    picks_to_index_matrix,
    points_matrix,
    score_index,
)
from src.clausura.pool import PoolConfig, pool_distribution
from src.clausura.scoring import supermatch_points
from src.model.poisson import score_grid


def test_indices_ida_y_vuelta():
    for gL in range(MAX_GOALS + 1):
        for gV in range(MAX_GOALS + 1):
            assert index_score(score_index(gL, gV)) == (gL, gV)


def test_points_matrix_coincide_con_el_kernel():
    m = points_matrix(False)
    mp = points_matrix(True)
    for pi in range(N_SCORES):
        for ai in range(N_SCORES):
            esperado = supermatch_points(index_score(pi), index_score(ai))
            assert m[pi, ai] == esperado
            assert mp[pi, ai] == esperado * 2


def test_flatten_grid_normaliza_y_ubica_bien():
    g = score_grid(1.3, 1.1, 0.0, max_goals=7)
    f = flatten_grid(g)
    assert f.sum() == pytest.approx(1.0)
    # la celda (1,0) de la grilla tiene que caer en el índice correspondiente
    assert f[score_index(1, 0)] == pytest.approx(g[1, 0] / g[:6, :6].sum(), rel=1e-9)


# -------------------- reparto de premios (Art. 7a) --------------------

def _sim_minimo(n_mine=2, n_rivales=3, n_sims=1):
    g = score_grid(1.3, 1.1, 0.0, max_goals=MAX_GOALS)
    q = pool_distribution(g, PoolConfig())
    return SeasonSimulator(
        grids=[g], fecha_de_partido=[1], preferencial=[False], pool_q=[q],
        prize=PrizeConfig(), sim=SimConfig(n_sims=n_sims, n_rivales=n_rivales, seed=1),
    )


def _liquidar(s, mine, rivals, pozo):
    """Adaptador: `_liquidar` toma el (máximo, empatados) del pool ya cacheado.

    Pasar por `_stats` acá no es un rodeo — es lo que hace que estos cuatro tests
    del Art. 7a sigan cubriendo la cadena entera después del caché del 2026-08-08.
    """
    top, empatados = SeasonSimulator._stats(rivals)
    return s._liquidar(mine, top, empatados, pozo)


def test_liquidar_gana_solo_cobra_todo():
    s = _sim_minimo()
    mine = np.array([[10], [5]])
    rivals = np.array([[3], [4], [2]])
    assert _liquidar(s, mine, rivals, 350_000.0)[0] == pytest.approx(350_000.0)


def test_liquidar_empate_con_rival_divide():
    s = _sim_minimo()
    mine = np.array([[10], [5]])
    rivals = np.array([[10], [4], [2]])
    # 1 nuestra y 1 ajena en el máximo → mitad
    assert _liquidar(s, mine, rivals, 350_000.0)[0] == pytest.approx(175_000.0)


def test_liquidar_empate_entre_nuestras_no_es_perdida():
    """Dos participaciones propias empatadas arriba cobran las DOS partes."""
    s = _sim_minimo()
    mine = np.array([[10], [10]])
    rivals = np.array([[10], [4], [2]])
    # 2 nuestras + 1 ajena = 3 en el máximo, cobramos 2/3
    assert _liquidar(s, mine, rivals, 300_000.0)[0] == pytest.approx(200_000.0)


def test_liquidar_perder_no_cobra():
    s = _sim_minimo()
    mine = np.array([[8], [5]])
    rivals = np.array([[10]])
    assert _liquidar(s, mine, rivals, 350_000.0)[0] == 0.0


def _sim_16(compactar: bool) -> SeasonSimulator:
    g = score_grid(1.4, 1.0, 0.0, max_goals=MAX_GOALS)
    n = 16
    q = pool_distribution(g, PoolConfig())
    return SeasonSimulator(
        grids=[g] * n, fecha_de_partido=[i // 8 for i in range(n)],
        preferencial=[i % 8 == 0 for i in range(n)], pool_q=[q] * n,
        prize=PrizeConfig(), sim=SimConfig(n_sims=200, n_rivales=20, seed=42),
        compactar_fechas=compactar,
    )


def test_compactar_fechas_da_identico_a_guardar_la_matriz_entera():
    """El acumulado por fecha se liquida a (máximo, empatados) y se tira la matriz.

    Era (n_fechas, R, S) — 403 MB con S=9.600 y R=700, en un droplet de 1 GB — y es
    lo que impedía subir n_sims, que es de donde sale la plata. La compactación no
    puede mover un peso, ni al liquidar ni después de cambiar picks propios: si se
    rompe, el ascenso optimiza contra un pool equivocado y nadie se entera.
    """
    compacto, entero = _sim_16(True), _sim_16(False)
    assert compacto.rivals_fecha is None and entero.rivals_fecha is not None
    assert np.array_equal(np.asarray(compacto.rivals_total),
                          np.asarray(entero.rivals_total))

    rng = np.random.default_rng(3)
    picks = rng.integers(0, N_SCORES, size=(4, compacto.n_matches))
    compacto.load_picks(picks)
    entero.load_picks(picks)
    assert compacto.e_premio_total() == entero.e_premio_total()
    assert compacto.result().__dict__ == entero.result().__dict__

    for _ in range(20):   # el ascenso cambia picks PROPIOS; el pool no se mueve
        i, m, v = (int(rng.integers(0, 4)), int(rng.integers(0, compacto.n_matches)),
                   int(rng.integers(0, N_SCORES)))
        compacto.set_pick(i, m, v)
        entero.set_pick(i, m, v)
        assert compacto.e_premio_total() == entero.e_premio_total()


def test_cache_de_rivales_da_identico_a_recalcular():
    """El caché de (máximo, empatados) contra la fórmula cruda del Art. 7a."""
    s = _sim_16(False)
    rng = np.random.default_rng(3)
    s.load_picks(rng.integers(0, N_SCORES, size=(4, s.n_matches)))

    def referencia() -> float:
        def liq(mine, riv, pozo):
            top = np.maximum(mine.max(axis=0), riv.max(axis=0))
            k = (mine == top[None, :]).sum(axis=0)
            j = (riv == top[None, :]).sum(axis=0)
            t = k + j
            return np.where(t > 0, pozo * k / np.maximum(t, 1), 0.0)
        p = liq(s.mine_total, np.asarray(s.rivals_total), s.prize.premio_penca)
        for fi in range(s.n_fechas):
            p = p + liq(s.mine_fecha[fi], np.asarray(s.rivals_fecha[fi]),
                        s.prize.premio_fecha)
        return float(p.mean())

    assert s.e_premio_total() == referencia()
    for _ in range(20):
        s.set_pick(int(rng.integers(0, 4)), int(rng.integers(0, s.n_matches)),
                   int(rng.integers(0, N_SCORES)))
        assert s.e_premio_total() == referencia()


def test_cache_del_premio_por_fecha_se_invalida_por_load_y_por_rivales():
    """El premio por fecha se cachea (cada movida toca UNA fecha, y re-liquidar
    las 15 era el 100% del costo del objetivo): load_picks y reescribir el lado
    rival por fecha tienen que invalidarlo, o el ascenso optimiza contra premios
    viejos en silencio."""
    rng = np.random.default_rng(7)
    s = _sim_16(False)
    picks_a = rng.integers(0, N_SCORES, size=(4, s.n_matches))
    picks_b = rng.integers(0, N_SCORES, size=(4, s.n_matches))

    s.load_picks(picks_a)
    s.e_premio_total()                           # materializa el caché por fecha
    s.load_picks(picks_b)
    fresco = _sim_16(False)
    fresco.load_picks(picks_b)
    assert s.e_premio_total() == fresco.e_premio_total()

    # reescritura del lado rival por fecha (camino de backtest.realized_prizes)
    s.e_premio_total()
    riv = np.zeros((s.n_rivales, s.cfg.n_sims), dtype=np.int32)
    s.rivals_fecha = np.stack([riv + 100] * s.n_fechas)   # pool imbatible por fecha
    assert s.result().e_premio_fechas == 0.0


def test_reescribir_el_lado_rival_invalida_el_cache():
    """`backtest.realized_prizes` reescribe rivals_total DESPUÉS del constructor.

    Con un caché eager eso devolvía premios calculados contra el pool viejo, en
    silencio. El setter tiene que invalidar.
    """
    s = _sim_minimo()
    s.load_picks(np.zeros((2, s.n_matches), dtype=np.int64))
    s.e_premio_total()                                    # materializa el caché
    assert s.result().e_premio_penca > 0.0                # antes cobrábamos algo
    s.rivals_total = np.full_like(np.asarray(s.rivals_total), 10_000)
    top, empatados = s._stats_total()
    assert top.max() == 10_000 and empatados.min() == s.n_rivales
    # el pool nos pasa por arriba en la general (el premio POR FECHA se liquida
    # aparte, contra rivals_fecha, y no lo tocamos: por eso se mira el componente)
    assert s.result().e_premio_penca == 0.0


def test_mutar_el_lado_rival_por_indice_falla_fuerte():
    """El setter no ve `rivals_fecha[fi] += x`; el read-only sí.

    Preferimos un ValueError a un premio calculado contra un máximo viejo.
    """
    s = _sim_16(False)          # compactado no hay matriz que mutar: el caso es este
    s.load_picks(np.zeros((2, s.n_matches), dtype=np.int64))
    s.e_premio_total()
    with pytest.raises(ValueError):
        s.rivals_fecha[0] += 5


# -------------------- estado incremental --------------------

@pytest.fixture
def sim_multi():
    g = score_grid(1.4, 1.0, 0.0, max_goals=MAX_GOALS)
    n = 16
    grids = [g] * n
    q = pool_distribution(g, PoolConfig())
    return SeasonSimulator(
        grids=grids,
        fecha_de_partido=[i // 8 for i in range(n)],
        preferencial=[i % 8 == 0 for i in range(n)],
        pool_q=[q] * n,
        prize=PrizeConfig(),
        sim=SimConfig(n_sims=200, n_rivales=20, seed=42),
    )


def test_set_pick_equivale_a_recargar(sim_multi):
    """El update incremental tiene que dar exactamente lo mismo que recargar de cero."""
    picks = np.full((3, 16), score_index(1, 0), dtype=np.int64)
    sim_multi.load_picks(picks)

    sim_multi.set_pick(1, 5, score_index(2, 1))
    sim_multi.set_pick(2, 0, score_index(0, 0))
    incremental_total = sim_multi.mine_total.copy()
    incremental_fecha = sim_multi.mine_fecha.copy()
    incremental_premio = sim_multi.e_premio_total()

    esperado = picks.copy()
    esperado[1, 5] = score_index(2, 1)
    esperado[2, 0] = score_index(0, 0)
    sim_multi.load_picks(esperado)

    assert np.array_equal(incremental_total, sim_multi.mine_total)
    assert np.array_equal(incremental_fecha, sim_multi.mine_fecha)
    assert incremental_premio == pytest.approx(sim_multi.e_premio_total())


def test_set_pick_al_mismo_valor_es_noop(sim_multi):
    picks = np.full((2, 16), score_index(1, 1), dtype=np.int64)
    sim_multi.load_picks(picks)
    antes = sim_multi.mine_total.copy()
    sim_multi.set_pick(0, 3, score_index(1, 1))
    assert np.array_equal(antes, sim_multi.mine_total)


def test_preferencial_duplica_los_puntos_del_partido(sim_multi):
    """El partido 0 es preferencial y el 1 no: mismo pick, el doble de aporte."""
    pref = sim_multi.match_points(0, score_index(1, 0))
    normal = sim_multi.match_points(1, score_index(1, 0))
    # mismos resultados sorteados no, pero sí el mismo mapeo de puntos:
    pm_pref = sim_multi.pm[0]
    pm_norm = sim_multi.pm[1]
    assert np.array_equal(pm_pref, pm_norm * 2)
    assert pref.max() <= 16 and normal.max() <= 8


def test_load_picks_valida_dimension(sim_multi):
    with pytest.raises(ValueError, match="partidos"):
        sim_multi.load_picks(np.zeros((2, 5), dtype=np.int64))


def test_picks_to_index_matrix():
    m = picks_to_index_matrix([[(1, 0), (2, 1)], [(0, 0), (3, 2)]])
    assert m.shape == (2, 2)
    assert m[0, 0] == score_index(1, 0)
    assert m[1, 1] == score_index(3, 2)


def test_result_reporta_costo_por_participacion(sim_multi):
    sim_multi.load_picks(np.full((5, 16), score_index(1, 0), dtype=np.int64))
    r = sim_multi.result()
    assert r.costo == 5 * 400.0
    assert 0.0 <= r.p_gana_penca <= 1.0
    assert r.e_premio_total == pytest.approx(r.e_premio_penca + r.e_premio_fechas)

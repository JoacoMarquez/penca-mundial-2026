"""Tests del modelo de pool, ratings y construcción del portfolio."""

import numpy as np
import pytest

from src.clausura.economics import (
    MAX_GOALS,
    PrizeConfig,
    SeasonSimulator,
    SimConfig,
    index_score,
    score_index,
)
from src.clausura.historical import PartidoHistorico
from src.clausura.pool import (
    PoolConfig,
    calibrate_from_exact_rate,
    expected_exact_rate,
    observed_exact_rate_from_ranking,
    pool_distribution,
    top_pool_picks,
)
from src.clausura.ratings import fit_ratings, ranking
from src.clausura.strategy import (
    baseline_chalk,
    baseline_ev,
    build_candidates,
    build_portfolio,
)
from src.model.poisson import score_grid


@pytest.fixture(scope="module")
def grid():
    return score_grid(1.4, 1.0, 0.0, max_goals=MAX_GOALS)


# -------------------- pool --------------------

def test_pool_distribution_es_distribucion(grid):
    q = pool_distribution(grid)
    assert q.sum() == pytest.approx(1.0)
    assert (q >= 0).all()


def test_pool_subjuega_el_0_0(grid):
    """El hallazgo central: el 0-0 es frecuente en la liga pero impopular en el pool."""
    q = pool_distribution(grid)
    i00 = score_index(0, 0)
    from src.clausura.economics import flatten_grid
    p_real = flatten_grid(grid)[i00]
    assert q[i00] < p_real, "el pool debería subjugar el 0-0 respecto de su probabilidad real"


def test_temperatura_alta_aplana_el_pool(grid):
    q_frio = pool_distribution(grid, PoolConfig(temperature=0.5))
    q_tibio = pool_distribution(grid, PoolConfig(temperature=3.0))
    # más temperatura → más entropía
    ent = lambda q: -np.sum(q * np.log(q + 1e-12))
    assert ent(q_tibio) > ent(q_frio)


def test_exact_rate_baja_con_temperatura(grid):
    grids = [grid] * 5
    r_frio = expected_exact_rate(grids, PoolConfig(temperature=0.5))
    r_tibio = expected_exact_rate(grids, PoolConfig(temperature=3.0))
    assert r_frio > r_tibio


def test_calibracion_recupera_la_temperatura(grid):
    """Generamos una tasa de exactos con temperatura conocida y la recuperamos."""
    grids = [grid] * 8
    verdadera = PoolConfig(temperature=0.7)
    objetivo = expected_exact_rate(grids, verdadera)
    ajustada = calibrate_from_exact_rate(grids, objetivo)
    assert ajustada.temperature == pytest.approx(0.7)


def test_calibrar_conserva_los_campos_que_no_toca(grid):
    """La calibración solo mueve `temperature`: cualquier otro campo de PoolConfig
    tiene que sobrevivir. El constructor explícito copiaba cuatro campos a mano y se
    comía `orientar_al_favorito` —y se habría comido cualquier campo nuevo—
    reseteándolo al default en silencio.
    """
    base = PoolConfig(chalk_strength=1.7, default_bias=0.6,
                      orientar_al_favorito=False, popular_bias={(0, 0): 3.3})

    cal = calibrate_from_exact_rate([grid] * 4, 0.09, base=base)

    assert cal.orientar_al_favorito is False        # <- lo que se perdía
    assert cal.chalk_strength == 1.7
    assert cal.default_bias == 0.6
    assert cal.popular_bias == {(0, 0): 3.3}
    # y popular_bias es una COPIA: mutar el calibrado no toca el base
    cal.popular_bias[(1, 1)] = 9.9
    assert (1, 1) not in base.popular_bias


def test_observed_exact_rate_desde_ranking():
    class Row:
        def __init__(self, e): self.cant_resultados_exactos = e
    rows = [Row(2), Row(4), Row(3)]
    assert observed_exact_rate_from_ranking(rows, 24) == pytest.approx(3 / 24)
    assert observed_exact_rate_from_ranking(rows, 0) is None
    assert observed_exact_rate_from_ranking([], 24) is None


def test_observed_exact_rate_ignora_el_contador_sin_liquidar():
    """El API manda 0 exactos hasta que liquida la fecha, aunque ya haya puntos.

    Caso real del 2026-08-08: Cerro Largo 1-1 Juventud finalizado, 146 de 692 con 8
    puntos (=marcador exacto en un partido normal) y el contador de exactos en 0
    para todas. Creerle da tasa 0 → calibra al pool más DISPERSO cuando en realidad
    estaba en el más concentrado. Sin dato es mejor que con dato al revés.
    """
    from src.clausura.api import RankingRow

    def fila(numero, puntos, exactos):
        return RankingRow(participacion_id=numero, numero_participacion=numero,
                          puntos_totales=puntos, puntos_por_fecha=0,
                          posicion_general=1, cant_resultados_exactos=exactos)

    sin_liquidar = [fila(n, 8 if n < 146 else 1, 0) for n in range(692)]
    assert observed_exact_rate_from_ranking(sin_liquidar, 1) is None

    # pool genuinamente sin exactos y sin puntos (fecha recién arrancada): tasa 0 real
    arranque = [fila(n, 0, 0) for n in range(10)]
    assert observed_exact_rate_from_ranking(arranque, 1) == 0.0

    # una vez liquidado, el canal vuelve a servir
    liquidado = [fila(n, 8 if n < 146 else 1, 1 if n < 146 else 0) for n in range(692)]
    assert observed_exact_rate_from_ranking(liquidado, 1) == pytest.approx(146 / 692)


def test_observed_exact_rate_excluye_mis_numeros():
    """El observable calibra a los rivales: nuestras filas del ranking no cuentan."""
    class Row:
        def __init__(self, numero, e):
            self.numero_participacion = numero
            self.cant_resultados_exactos = e
    rows = [Row(100, 10), Row(200, 2), Row(300, 4)]
    assert observed_exact_rate_from_ranking(rows, 24, mis_numeros={100}) \
        == pytest.approx(3 / 24)
    assert observed_exact_rate_from_ranking(rows, 24, mis_numeros={100, 200, 300}) is None


def test_top_pool_picks_ordenado(grid):
    top = top_pool_picks(grid, k=4)
    assert len(top) == 4
    assert [q for _, q in top] == sorted([q for _, q in top], reverse=True)


# -------------------- ratings --------------------

def _partido(local, visitante, gl, gv, fecha=1):
    return PartidoHistorico(
        campeonato_id=1, campeonato="test", fecha_nombre=f"Fecha {fecha}", fecha_id=fecha,
        evento_id=0, local=local, visitante=visitante,
        goles_local=gl, goles_visitante=gv, preferencial=False,
        inicio_utc="2026-01-01T00:00:00+00:00",
    )


def test_ratings_detecta_el_equipo_fuerte():
    """Un equipo que golea siempre debe quedar arriba del ranking de fuerza neta."""
    partidos = []
    for i in range(12):
        partidos.append(_partido("Fuerte", "Debil", 3, 0, i))
        partidos.append(_partido("Debil", "Fuerte", 0, 2, i))
        partidos.append(_partido("Medio", "Debil", 2, 1, i))
        partidos.append(_partido("Fuerte", "Medio", 2, 0, i))
    r = fit_ratings(partidos)
    orden = [e for e, *_ in ranking(r)]
    assert orden[0] == "Fuerte"
    assert orden[-1] == "Debil"


def test_ratings_ventaja_local_positiva():
    partidos = []
    for i in range(20):
        partidos.append(_partido("A", "B", 2, 1, i))
        partidos.append(_partido("B", "A", 2, 1, i))
    r = fit_ratings(partidos)
    assert r.ventaja_local > 0


def test_lambdas_equipo_desconocido_no_explota():
    partidos = [_partido("A", "B", 1, 1, i) for i in range(10)]
    r = fit_ratings(partidos)
    lam_l, lam_v = r.lambdas("Recien Ascendido", "Otro Nuevo")
    assert 0.1 < lam_l < 5.0 and 0.1 < lam_v < 5.0


# -------------------- candidatos y portfolio --------------------

def test_build_candidates_incluye_el_mejor_por_ev(grid):
    q = pool_distribution(grid)
    cands = build_candidates(grid, q)
    from src.clausura.scoring import best_pick
    mejor, _ = best_pick(grid)
    assert mejor in [c.pick for c in cands]


def test_build_candidates_sin_duplicados(grid):
    q = pool_distribution(grid)
    cands = build_candidates(grid, q)
    picks = [c.pick for c in cands]
    assert len(picks) == len(set(picks))


def test_build_candidates_trae_huecos_de_pool(grid):
    """El menú tiene que incluir algún marcador que el pool subjuegue."""
    q = pool_distribution(grid)
    cands = build_candidates(grid, q, k_ev=3, k_hueco=3)
    solo_ev = build_candidates(grid, q, k_ev=3, k_hueco=0)
    assert len(cands) > len(solo_ev)


@pytest.fixture(scope="module")
def escenario():
    """Mini-temporada: 3 fechas x 4 partidos, con fuerzas distintas por partido."""
    grids, fechas, pref = [], [], []
    for f in range(3):
        for m in range(4):
            grids.append(score_grid(1.1 + 0.25 * m, 1.0, 0.0, max_goals=MAX_GOALS))
            fechas.append(f)
            pref.append(m == 0)
    return grids, fechas, pref


def test_portfolio_no_empeora_el_ancla_ev(escenario):
    """El ascenso por coordenadas nunca puede terminar peor que el punto de partida."""
    grids, fechas, pref = escenario
    sim = SimConfig(n_sims=300, n_rivales=40, seed=5)
    port = build_portfolio(grids, fechas, pref, n_participaciones=3,
                           sim=sim, max_passes=1)

    pool_qs = [pool_distribution(g) for g in grids]
    s = SeasonSimulator(grids, fechas, pref, pool_qs, PrizeConfig(), sim)
    s.load_picks(baseline_ev(grids, pref, 3))
    ancla = s.e_premio_total()

    s.load_picks(port.picks)
    assert s.e_premio_total() >= ancla - 1e-6


def test_portfolio_diversifica(escenario):
    """Con premio repartido entre empatados, el óptimo NO es clonar la misma planilla."""
    grids, fechas, pref = escenario
    port = build_portfolio(grids, fechas, pref, n_participaciones=4,
                           sim=SimConfig(n_sims=300, n_rivales=40, seed=5), max_passes=1)
    assert port.diversidad() > 0.0


def test_portfolio_respeta_dimensiones(escenario):
    grids, fechas, pref = escenario
    port = build_portfolio(grids, fechas, pref, n_participaciones=3,
                           sim=SimConfig(n_sims=200, n_rivales=30, seed=5), max_passes=1)
    assert port.picks.shape == (3, len(grids))
    scores = port.as_scores()
    assert len(scores) == 3 and len(scores[0]) == len(grids)
    for fila in scores:
        for gL, gV in fila:
            assert 0 <= gL <= MAX_GOALS and 0 <= gV <= MAX_GOALS


def test_participacion_0_es_el_ancla_ev(escenario):
    """La primera participación queda fija en argmax E[pts] (no se perturba)."""
    grids, fechas, pref = escenario
    port = build_portfolio(grids, fechas, pref, n_participaciones=3,
                           sim=SimConfig(n_sims=200, n_rivales=30, seed=5), max_passes=1)
    esperado = baseline_ev(grids, pref, 1)[0]
    assert np.array_equal(port.picks[0], esperado)


def test_resultado_reportado_es_out_of_sample(escenario):
    """El E[premio] del portfolio se evalúa con semilla fresca: no puede coincidir
    con el valor in-sample del optimizador (winner's curse), y sigue determinístico."""
    grids, fechas, pref = escenario
    sim = SimConfig(n_sims=300, n_rivales=40, seed=11)
    port = build_portfolio(grids, fechas, pref, n_participaciones=3,
                           sim=sim, max_passes=1)

    # valor in-sample: mismo simulador/semilla que usó el optimizador
    pool_qs = [pool_distribution(g) for g in grids]
    s = SeasonSimulator(grids, fechas, pref, pool_qs, PrizeConfig(), sim)
    s.load_picks(port.picks)
    in_sample = s.e_premio_total()
    assert port.resultado.e_premio_total != pytest.approx(in_sample, abs=1e-9)

    # determinismo: misma llamada → mismo reporte
    port2 = build_portfolio(grids, fechas, pref, n_participaciones=3,
                            sim=sim, max_passes=1)
    assert port2.resultado.e_premio_total == port.resultado.e_premio_total


def test_baselines_son_uniformes(escenario):
    grids, fechas, pref = escenario
    for picks in (baseline_chalk(grids, 4), baseline_ev(grids, pref, 4)):
        for m in range(len(grids)):
            assert len(set(picks[:, m])) == 1


# -------------------- el menú y la concentración van juntos con n_sims --------------------

def test_menu_de_cinco_por_ev_sin_rama_de_hueco():
    """K_EV=5 desde el 2026-08-11; la rama de rareza sigue apagada.

    El tamaño óptimo del menú DEPENDE de n_sims (más candidatos = más comparaciones
    Monte Carlo ruidosas, pero también más chances de encontrar el pick que conviene).
    A 2.400 el menú de 5 medía −$4.192; a 19.200 mide +$4.568. Si alguien baja los
    sorteos, este test no lo detecta — pero el comentario de K_EV avisa por qué hay
    que volver a medir.
    """
    from src.clausura import strategy
    assert (strategy.K_EV, strategy.K_HUECO) == (5, 0)


def test_pool_arranca_concentrado():
    """chalk_strength calibrado contra picks reales: el pool se amontona más que 1.0."""
    from src.clausura.pool import PoolConfig
    cfg = PoolConfig()
    assert cfg.chalk_strength == 2.2
    assert cfg.temperature == 1.0, "el barrido midió chalk con T=1; lo que manda es chalk/T"


def test_menu_mas_grande_incluye_al_menu_chico():
    """Subir K_EV solo AGREGA candidatos: nada de lo que ya se jugaba desaparece.

    Importa porque el cambio de menú tiene que ser una ampliación del espacio de
    búsqueda, no un reemplazo — si sacara candidatos, el resultado del barrido no se
    podría atribuir a "más opciones".
    """
    from src.clausura.pool import PoolConfig, pool_distribution
    from src.clausura.strategy import build_candidates
    from src.model.poisson import score_grid

    grid = score_grid(1.4, 1.1, 0.0, max_goals=5)
    q = pool_distribution(grid, PoolConfig())
    chicos = {c.pick for c in build_candidates(grid, q, False, k_ev=3, k_hueco=0)}
    grandes = {c.pick for c in build_candidates(grid, q, False, k_ev=5, k_hueco=0)}
    assert chicos < grandes and len(grandes) == 5

"""Tests del pipeline manual de picks (sin red: helpers puros y estado en disco)."""

import json

import numpy as np
import itertools

import pytest

from src.clausura.economics import SimConfig, score_index
from src.clausura.odds import EventOdds
from src.clausura.picks import (
    build_season_grids,
    delta_grid,
    fecha_dir,
    load_frozen,
    load_warm_start,
    market_lambdas,
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


_odds_seq = itertools.count(1)


def _odds(home, away, eid=None):
    # id único por defecto: cada cuota del ES describe UN partido, y match_odds se
    # apoya en eso para no repartir la misma dos veces. Con un id fijo compartido,
    # dos cuotas del fixture se pisaban entre sí.
    return EventOdds(event_id=eid or f"sm:{next(_odds_seq)}", home=home, away=away,
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


def test_match_odds_nombres_reales_del_es():
    """Nombres reales del Elasticsearch de Supermatch (2026-08-06). El caso
    'M.C. Torque' es el bug de la v9: ningún substring, solo tokens."""
    evs = [
        _evento(1, "Montevideo City Torque", "Peñarol"),
        _evento(2, "Liverpool", "Juventud"),
    ]
    m = match_odds(evs, [
        _odds("M.C. Torque", "Peñarol"),
        _odds("Liverpool (URU)", "Juventud de Las Piedras"),
    ])
    assert m.keys() == {1, 2}


def test_match_odds_abreviatura_por_prefijo():
    evs = [_evento(1, "Defensor Sporting", "Racing")]
    m = match_odds(evs, [_odds("Defensor Sp.", "Racing Club de Montevideo")])
    assert 1 in m


def test_match_odds_tokens_distintos_no_matchean():
    evs = [_evento(1, "Nacional", "Cerro")]
    m = match_odds(evs, [_odds("Central Español", "Cerro")])
    assert not m


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


def test_eventos_liquidados_solo_cuenta_fechas_completas():
    """El denominador del exact_rate son las fechas LIQUIDADAS: contar los partidos
    de la fecha en curso diluye la tasa (numerador de F1 sobre F1+F2) y calibra un
    pool más disperso — el mecanismo del incidente T=3.0."""
    from src.clausura.picks import eventos_liquidados

    cfg = {"fechas": {
        "Fecha 1": {"eventos": [{"evento_id": 10}, {"evento_id": 11}]},
        "Fecha 2": {"eventos": [{"evento_id": 20}, {"evento_id": 21}]},
    }}
    # F1 completa, F2 a medias: solo cuentan los 2 de F1
    res = {10: (1, 0), 11: (2, 2), 20: (0, 0)}
    assert eventos_liquidados(cfg, res) == {10, 11}
    # nada terminado: conjunto vacío (el exact_rate cae a None, no a 0)
    assert eventos_liquidados(cfg, {}) == set()
    # todo terminado: cuentan las dos fechas
    res[21] = (1, 1)
    assert eventos_liquidados(cfg, res) == {10, 11, 20, 21}


def test_market_lambdas_conserva_el_lam12_del_fit():
    """El fit bivariado usa λ12 para clavar el empate del mercado; descartarlo
    dejaba la grilla −4 pp corta de empates en partidos parejos."""
    from src.model.market_probs import devig
    from src.model.poisson import marginals

    o = EventOdds(event_id="sm:lam12", home="A", away="B",
                  start_utc="x", fetched_utc="x",
                  x1x2={"home": 2.45, "draw": 3.1, "away": 3.0},
                  totals={"2.5": {"over": 1.85, "under": 1.95}})
    lam_l, lam_v, lam12 = market_lambdas(o)
    assert lam12 > 0.05  # partido parejo: el mercado exige covarianza

    objetivo = devig(o.x1x2, "proportional")["draw"]
    con = marginals(score_grid(lam_l, lam_v, lam12, max_goals=5)).p_draw
    sin = marginals(score_grid(lam_l, lam_v, 0.0, max_goals=5)).p_draw
    # con λ12 el empate queda cerca del mercado; sin él, sistemáticamente corto
    assert abs(con - objetivo) < 0.015
    assert objetivo - sin > 0.02


def test_build_season_grids_propaga_lam12_al_blend():
    """La grilla con mercado tiene más empate que la que descartaba λ12."""
    ev = _evento(10, "A", "B")
    o = EventOdds(event_id="sm:blend", home="A", away="B",
                  start_utc="x", fetched_utc="x",
                  x1x2={"home": 2.45, "draw": 3.1, "away": 3.0},
                  totals={"2.5": {"over": 1.85, "under": 1.95}})
    grids, fuentes, _, _ = build_season_grids(
        [ev], _RatingsStub(), odds_by_evento={10: o}, resultados={})
    assert fuentes[0] == "mercado+ratings"

    from src.model.poisson import marginals
    lam_mkt = market_lambdas(o)
    rt_l, rt_v = _RatingsStub().lambdas("A", "B")
    lam_l = 0.7 * lam_mkt[0] + 0.3 * rt_l
    lam_v = 0.7 * lam_mkt[1] + 0.3 * rt_v
    sin_lam12 = score_grid(lam_l, lam_v, 0.0, max_goals=5)
    assert marginals(grids[0]).p_draw > marginals(sin_lam12).p_draw + 0.015
    assert grids[0][0, 0] > sin_lam12[0, 0]  # y más 0-0, el hueco del pool


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


def test_warm_start_sin_planilla_previa_es_none(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)
    eventos = [_evento(10, "A", "B", fecha_n=1)]
    assert load_warm_start(eventos, target_fecha=1, n_participaciones=3) is None


def test_warm_start_prefiere_la_matriz_de_temporada(tmp_path, monkeypatch):
    """`picks_temporada` cubre los 120 partidos; `picks`, solo los 8 de su fecha."""
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    eventos = [_evento(10, "A", "B", fecha_n=1), _evento(20, "C", "D", fecha_n=2)]
    save_version(1, {
        "picks": [{"evento_id": 10, "scores": [[1, 0], [2, 1]]}],
        "picks_temporada": [
            {"evento_id": 10, "scores": [[1, 0], [2, 1]]},
            {"evento_id": 20, "scores": [[0, 0], [3, 1]]},
        ],
    })

    warm = load_warm_start(eventos, target_fecha=2, n_participaciones=2)
    assert warm.shape == (2, 2)
    assert warm[0, 1] == score_index(0, 0)     # heredado del partido de la fecha 2
    assert warm[1, 1] == score_index(3, 1)
    assert (warm >= 0).all()


def test_warm_start_la_planilla_mas_nueva_manda(tmp_path, monkeypatch):
    """Dos versiones de la misma fecha: gana la v2, no la v1."""
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    eventos = [_evento(10, "A", "B", fecha_n=1)]
    save_version(1, {"picks": [{"evento_id": 10, "scores": [[1, 0], [1, 0]]}]})
    save_version(1, {"picks": [{"evento_id": 10, "scores": [[2, 2], [2, 2]]}]})

    warm = load_warm_start(eventos, target_fecha=1, n_participaciones=2)
    assert warm[0, 0] == score_index(2, 2)


def test_warm_start_ignora_planillas_de_otro_tamano(tmp_path, monkeypatch):
    """Si ayer eran 5 participaciones y hoy 12, esa columna no se puede heredar.

    Rellenar las filas faltantes inventaría un portfolio que nunca se evaluó; -1
    hace que la columna caiga al ancla de EV, que es el comportamiento correcto.
    """
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    eventos = [_evento(10, "A", "B", fecha_n=1), _evento(20, "C", "D", fecha_n=1)]
    save_version(1, {"picks": [
        {"evento_id": 10, "scores": [[1, 0], [2, 1]]},          # solo 2 filas
        {"evento_id": 20, "scores": [[1, 0], [2, 1], [0, 0]]},  # 3 filas
    ]})

    warm = load_warm_start(eventos, target_fecha=1, n_participaciones=3)
    assert (warm[:, 0] == -1).all()            # columna no heredable
    assert (warm[:, 1] >= 0).all()


def test_warm_start_columna_incompleta_cae_al_ancla():
    """Una columna se hereda entera o nada: no se mezcla warm con ancla.

    La fila 0 es la excepción deliberada: es el ancla de EV puro y se re-ancla
    al modelo de hoy aunque la columna se herede (ver strategy.build_portfolio).
    """
    g = score_grid(1.3, 1.1, 0.0, max_goals=5)
    grids = [g] * 4
    warm = np.full((3, 4), -1, dtype=np.int64)
    warm[:, 0] = score_index(4, 4)             # completa: se hereda (filas 1+)
    warm[:2, 1] = score_index(4, 4)            # incompleta: NO se hereda

    port = build_portfolio(
        grids=grids, fecha_de_partido=[1] * 4, preferencial=[False] * 4,
        n_participaciones=3, sim=SimConfig(n_sims=60, n_rivales=20),
        max_passes=0, warm_start=warm,
    )
    assert (port.picks[1:, 0] == score_index(4, 4)).all()
    assert port.picks[0, 0] != score_index(4, 4)   # fila 0 re-anclada al EV de hoy
    assert not (port.picks[:, 1] == score_index(4, 4)).any()


def test_warm_start_shape_equivocado_falla_fuerte():
    g = score_grid(1.3, 1.1, 0.0, max_goals=5)
    with pytest.raises(ValueError, match="warm_start"):
        build_portfolio(
            grids=[g] * 4, fecha_de_partido=[1] * 4, preferencial=[False] * 4,
            n_participaciones=3, sim=SimConfig(n_sims=60, n_rivales=20),
            max_passes=0, warm_start=np.zeros((2, 4), dtype=np.int64),
        )


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


# -------------------- arrastre de goleadores (menú de la API en 500) --------------------

def _planilla_esp(campeones, goleadores):
    return {"picks": [], "especiales": {"por_participacion": [
        {"campeon_idx": c, "campeon": f"eq{c}",
         "goleador_idx": g[0], "goleador": g[1]}
        for c, g in zip(campeones, goleadores)
    ]}}


def test_frozen_especiales_busca_campeon_y_goleador_por_separado(tmp_path, monkeypatch):
    """Una corrida sin menú escribe goleador_idx=-1: el freeze del goleador tiene
    que venir de la planilla ANTERIOR que sí los tenía (bug de la v8 del 5/8)."""
    import src.clausura.picks as picks_mod
    from src.clausura.picks import load_frozen_especiales
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    save_version(1, _planilla_esp([9, 8], [(3, "Gómez"), (5, "Abel")]))     # v1: completa
    save_version(1, _planilla_esp([9, 8], [(-1, None), (-1, None)]))        # v2: sin menú

    campeon, goleador = load_frozen_especiales(target_fecha=1, n_participaciones=2)
    assert campeon.tolist() == [9, 8]          # de la v2 (la última con campeón)
    assert goleador.tolist() == [3, 5]         # de la v1 (la última con goleador)


def _menu_goleador(nombres):
    from src.clausura.especiales import OpcionGoleador
    return [OpcionGoleador(id=100 + i, nombre=n, equipo_id=-1)
            for i, n in enumerate(nombres)]


def test_frozen_especiales_resuelve_goleador_por_nombre_contra_menu(tmp_path, monkeypatch):
    """Planilla con goleador por NOMBRE y goleador_idx=-1 (menú en 500 al generarla,
    caso v14 del 7/8): con el menú disponible, el nombre se resuelve y congela.
    Los nombres de la planilla más nueva mandan sobre los idx de planillas viejas."""
    import src.clausura.picks as picks_mod
    from src.clausura.picks import load_frozen_especiales
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    save_version(1, _planilla_esp([9, 8], [(0, "Abel Hernández"), (0, "Abel Hernández")]))
    save_version(1, _planilla_esp([9, 8], [(-1, "Matías Arezo"), (-1, "Maximiliano Gómez")]))

    menu = _menu_goleador(["Abel Hernández", "Maximiliano Gómez", "Matías Arezo"])
    campeon, goleador = load_frozen_especiales(
        target_fecha=1, n_participaciones=2, opciones_goleador=menu)
    assert campeon.tolist() == [9, 8]
    assert goleador.tolist() == [2, 1]   # v2 por nombre, NO los idx=0 de la v1


def test_frozen_especiales_matchea_nombre_sin_tildes_ni_mayusculas(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    from src.clausura.picks import load_frozen_especiales
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    save_version(1, _planilla_esp([9], [(-1, "maximiliano gomez")]))

    menu = _menu_goleador(["Abel Hernández", "Maximiliano Gómez"])
    _, goleador = load_frozen_especiales(
        target_fecha=1, n_participaciones=1, opciones_goleador=menu)
    assert goleador.tolist() == [1]


def test_frozen_especiales_nombre_fuera_del_menu_queda_libre_y_avisa(tmp_path, monkeypatch, caplog):
    """Si el nombre cargado no está en el menú no se inventa un idx: la fila queda
    -1 (libre) con warning, y las que sí matchean se congelan igual."""
    import src.clausura.picks as picks_mod
    from src.clausura.picks import load_frozen_especiales
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    save_version(1, _planilla_esp([9, 8], [(-1, "Jugador Fantasma"), (-1, "Matías Arezo")]))

    menu = _menu_goleador(["Abel Hernández", "Matías Arezo"])
    with caplog.at_level("WARNING"):
        _, goleador = load_frozen_especiales(
            target_fecha=1, n_participaciones=2, opciones_goleador=menu)
    assert goleador.tolist() == [-1, 1]
    assert any("Jugador Fantasma" in r.message for r in caplog.records)


def test_frozen_especiales_sin_menu_conserva_comportamiento_viejo(tmp_path, monkeypatch):
    """Sin opciones_goleador (menú aún en 500) los nombres no se pueden resolver:
    el freeze del goleador sigue viniendo de la última planilla con idx>=0."""
    import src.clausura.picks as picks_mod
    from src.clausura.picks import load_frozen_especiales
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    save_version(1, _planilla_esp([9], [(3, "Maximiliano Gómez")]))
    save_version(1, _planilla_esp([9], [(-1, "Matías Arezo")]))

    _, goleador = load_frozen_especiales(target_fecha=1, n_participaciones=1)
    assert goleador.tolist() == [3]


def test_goleadores_previos_saltea_planillas_sin_goleador(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    from src.clausura.picks import goleadores_previos
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    save_version(1, _planilla_esp([9], [(3, "Maximiliano Gómez")]))
    save_version(1, _planilla_esp([9], [(-1, None)]))

    prev = goleadores_previos(target_fecha=1, n_participaciones=1)
    assert prev == [{"goleador_idx": 3, "goleador": "Maximiliano Gómez"}]
    assert goleadores_previos(target_fecha=1, n_participaciones=1) is not None


def test_goleadores_previos_sin_historia_devuelve_none(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    from src.clausura.picks import goleadores_previos
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)
    assert goleadores_previos(target_fecha=1, n_participaciones=2) is None


def test_format_especiales_muestra_goleadores_arrastrados():
    from src.clausura.picks import format_especiales
    from src.clausura.strategy import PortfolioClausura

    port = PortfolioClausura(picks=np.zeros((2, 1), dtype=np.int64), candidatos=[],
                             resultado=None, campeon=np.array([0, 1]),
                             p_campeon=np.array([0.6, 0.4]))
    txt = format_especiales(port, ["Peñarol", "Nacional"], None,
                            gol_previos=[{"goleador_idx": 3, "goleador": "Gómez"},
                                         {"goleador_idx": 5, "goleador": "Abel"}])
    assert "Gómez" in txt and "Abel" in txt
    assert "menú aún no publicado" not in txt


# -------------------- cruce de equipos con nombres anidados --------------------

def test_match_odds_no_cruza_cerro_con_cerro_largo():
    """El caso real del 2026-08-11: 8 cuotas producían 9 eventos matcheados.

    "Cerro" es substring de "Cerro Largo", y el guardia viejo —exigir que matcheen
    los DOS equipos— no servía porque el rival era el mismo en los dos partidos:

        Fecha  2: Cerro       vs Albion   ← las cuotas son de acá
        Fecha 10: Cerro Largo vs Albion   ← y también matcheaban acá

    La Fecha 10 quedaba con la grilla del equipo equivocado alimentando P(campeón)
    durante semanas.
    """
    evs = [_evento(1, "Cerro", "Albion", fecha_n=2),
           _evento(2, "Cerro Largo", "Albion", fecha_n=10)]
    m = match_odds(evs, [_odds("Cerro", "Albion")])

    assert m.get(1) is not None, "el partido de Cerro tiene que quedarse su cuota"
    assert 2 not in m, "Cerro Largo NO puede heredar la cuota de Cerro"


def test_match_odds_no_reparte_la_misma_cuota_dos_veces():
    """Una cuota describe UN partido real: compartirla es siempre un error.

    Que sobre un evento sin cuota es correcto y esperable (el ES solo publica la
    fecha próxima); que dos eventos compartan una nunca lo es.
    """
    evs = [_evento(1, "Cerro", "Albion"), _evento(2, "Cerro Largo", "Albion"),
           _evento(3, "Cerro Largo", "Danubio")]
    m = match_odds(evs, [_odds("Cerro", "Albion", eid="sm:A"),
                         _odds("Cerro Largo", "Danubio", eid="sm:B")])
    ids = [o.event_id for o in m.values()]
    assert len(ids) == len(set(ids)), f"cuota repartida dos veces: {ids}"
    assert m[1].event_id == "sm:A" and m[3].event_id == "sm:B"


def test_match_odds_sigue_aceptando_los_alias_reales():
    """Apretar el matcheo no puede romper los alias que sí son el mismo equipo.

    Los tres salen del Elasticsearch real de Supermatch.
    """
    evs = [_evento(1, "Defensor Sporting", "Liverpool"),
           _evento(2, "Juventud", "Montevideo City Torque")]
    m = match_odds(evs, [_odds("Defensor Sporting", "Liverpool (URU)", eid="sm:A"),
                         _odds("Juventud de Las Piedras", "M.C. Torque", eid="sm:B")])
    assert m[1].event_id == "sm:A"
    assert m[2].event_id == "sm:B"


def test_match_odds_prefiere_la_igualdad_exacta_sin_importar_el_orden():
    """La asignación no puede depender de en qué orden vengan los eventos.

    Con matcheo laxo y 'primero que pinta gana', el resultado dependía del orden del
    fixture — un bug que aparece y desaparece según la fecha.
    """
    a, b = _evento(1, "Cerro", "Albion"), _evento(2, "Cerro Largo", "Albion")
    cuota = [_odds("Cerro", "Albion")]
    assert match_odds([a, b], cuota).keys() == {1}
    assert match_odds([b, a], cuota).keys() == {1}


# -------------------- decay temporal de los ratings (medido, apagado) --------------------

def test_ratings_sin_decay_por_default():
    """Se midió el 2026-08-12 y NO mejora: +0.0021 ± 0.0026 nats/partido (t=0.81).

    El parámetro queda para re-medirlo con más temporadas, pero apagado. Si alguien lo
    prende, que sea con evidencia nueva y no por la intuición de que "los datos viejos
    ensucian" — que es exactamente lo que este test documenta como refutado.
    """
    import inspect
    from src.clausura.ratings import fit_ratings
    assert inspect.signature(fit_ratings).parameters["half_life_dias"].default is None


def test_pesos_por_antiguedad():
    """El peso cae a la mitad exactamente en una vida media."""
    from src.clausura.historical import PartidoHistorico
    from src.clausura.ratings import _pesos_por_antiguedad

    def _p(fecha):
        return PartidoHistorico(campeonato_id=1, campeonato="x", fecha_nombre="f",
                                fecha_id=1, evento_id=1, local="A", visitante="B",
                                goles_local=1, goles_visitante=0, preferencial=False,
                                inicio_utc=fecha)
    partidos = [_p("2025-01-01T00:00:00+00:00"), _p("2026-01-01T00:00:00+00:00")]
    w = _pesos_por_antiguedad(partidos, half_life_dias=365.0)
    assert abs(w[1] - 1.0) < 1e-9, "el más reciente pesa 1"
    assert abs(w[0] - 0.5) < 0.01, "a una vida media de distancia pesa 0.5"

    sin = _pesos_por_antiguedad(partidos, half_life_dias=None)
    assert (sin == 1.0).all(), "sin vida media, pesos uniformes"


# -------------------- versionado concurrente --------------------

def test_save_version_no_pierde_escrituras_concurrentes(tmp_path, monkeypatch):
    """Dos escritores en la misma fecha (el rerun termina ~13:05-13:20 y el
    drift-audit corre 13:20) no pueden calcular el mismo N: el segundo archivo
    quedaría invisible para latest_version y su contenido se perdería sin ruido.

    El sleep dentro de latest_version ensancha la ventana leer→escribir para que
    la carrera sea determinística: sin el lock este test falla siempre.
    """
    import threading
    import time

    import src.clausura.picks as picks_mod
    from src.utils.versions import version_num
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    real_latest = picks_mod.latest_version

    def latest_lento(paths):
        r = real_latest(paths)
        time.sleep(0.05)
        return r

    monkeypatch.setattr(picks_mod, "latest_version", latest_lento)

    escritos, lock = [], threading.Lock()

    def escribir(i):
        p = picks_mod.save_version(7, {"quien": i, "picks": []})
        with lock:
            escritos.append(p)

    hilos = [threading.Thread(target=escribir, args=(i,)) for i in range(4)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()

    assert sorted(version_num(p) for p in escritos) == [1, 2, 3, 4]
    # y las 4 sobreviven en disco: ninguna quedó pisada ni huérfana
    assert len(list((tmp_path / "fecha_07").glob("v*_*.json"))) == 4


def test_el_lock_no_se_cuela_en_el_glob_de_versiones(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)
    picks_mod.save_version(3, {"picks": []})
    d = tmp_path / "fecha_03"
    assert (d / ".version.lock").exists()
    assert [p.name for p in d.glob("v*_*.json")] == [
        p.name for p in d.iterdir() if p.suffix == ".json"]

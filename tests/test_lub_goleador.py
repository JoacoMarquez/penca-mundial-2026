import numpy as np

from src.lub.goleador import (PPG_A, PPG_B, SD_CAMBIO, SD_MISMO, Candidato, _buscar, priors,
                              simular_goleador)

GENIUS = {
    "24/25": [{"pid": 1, "nombre": "J. PEREZ", "equipo": "AGUADA", "pj": 30, "pts": 300}],
    "25/26": [
        {"pid": 1, "nombre": "J. PEREZ", "equipo": "PEÑAROL", "pj": 40, "pts": 600},
        {"pid": 2, "nombre": "J. PEREZ", "equipo": "PEÑAROL", "pj": 2, "pts": 10},   # homónimo
        {"pid": 1, "nombre": "J. PEREZ", "equipo": "", "pj": 40, "pts": 600},         # fila sin club
        {"pid": 3, "nombre": "K. WACHSMAN", "equipo": "MALVIN", "pj": 36, "pts": 146},
    ],
}


def test_buscar_desambigua_homonimos_y_tolera_ortografia():
    filas = _buscar(GENIUS, "Juan Pérez")
    assert {f["pid"] for f in filas} == {1}
    assert all(f["equipo"] for f in filas)
    assert {f["pid"] for f in _buscar(GENIUS, "Kiril Wachsmann")} == {3}


def test_priors_regresion_y_sd_por_cambio_de_club():
    c = [{"nombre": "A", "equipo": "Peñarol", "25/26": {"pj": 40, "pts": 600, "ppg": 15.0, "club": "Peñarol"}},
         {"nombre": "B", "equipo": "Nacional", "25/26": {"pj": 40, "pts": 400, "ppg": 10.0, "club": "Aguada"}},
         {"nombre": "C", "equipo": "Goes"}]
    a, b, nuevo = priors(c)
    assert np.isclose(a.mu, PPG_A + PPG_B * 15.0) and a.sd == SD_MISMO
    assert b.sd == SD_CAMBIO
    assert nuevo.origen == "sin datos"


def test_mas_partidos_del_equipo_mas_chance_de_goleador():
    """Mismo jugador en dos equipos: gana el que juega más (los puntos son totales)."""
    cands = [Candidato("A", "X", 10.0, 0.5, 0.95, ""), Candidato("B", "Y", 10.0, 0.5, 0.95, "")]
    S = 4000
    pe = np.tile(np.array([44, 30]), (S, 1))
    gol, total = simular_goleador(pe, ["X", "Y"], cands)
    assert (gol == 0).mean() > 0.9
    assert total[:, 0].mean() > total[:, 1].mean()


def test_goleador_correlacionado_con_partidos_por_sorteo():
    """Si en un sorteo su equipo juega la final, el goleador sale de ahí más seguido."""
    cands = [Candidato("A", "X", 12.0, 1.5, 0.95, ""), Candidato("B", "Y", 12.0, 1.5, 0.95, "")]
    S = 6000
    rng = np.random.default_rng(0)
    largo = rng.random(S) < 0.5
    pe = np.stack([np.where(largo, 44, 32), np.full(S, 38)], 1)
    gol, _ = simular_goleador(pe, ["X", "Y"], cands)
    assert (gol[largo] == 0).mean() > (gol[~largo] == 0).mean() + 0.3

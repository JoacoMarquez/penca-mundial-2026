"""El gate por carga: la corrida diaria no pisa lo YA CARGADO sin medirlo.

Contexto (2026-08-31, Fecha 4): la corrida de las 11:00 reescribía la planilla
sin gate por valor. El domingo movió dos filas al 0-1 del clásico ⭐x2 después de
que el usuario cargara 1-1 — sin aviso, sin medición, −28 puntos realizados. El
rerun no podía verlo: su "planilla previa" es la salida de la misma corrida.

La referencia de "lo cargado" son las marcas del modo carga (carga_state), que
guardan el marcador confirmado por celda.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.clausura.carga_gate import (
    aplicar_gate,
    diffs_vs_cargado,
    marcas_de_fecha,
)
from src.clausura.economics import index_score, score_index
from src.clausura.strategy import ComparacionPortfolios

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
ABIERTO = (NOW + timedelta(hours=6)).isoformat()
CERRADO = (NOW - timedelta(hours=6)).isoformat()
NUMEROS = [111, 222, 333]


def comp(delta, se=100.0):
    return ComparacionPortfolios(delta=delta, se=se, valor_a=200_000.0,
                                 valor_b=200_000.0 + delta, n_seeds=5)


class EvaluadorStub:
    """Devuelve una comparación fija y registra qué matrices le pasaron."""

    def __init__(self, resultado):
        self.resultado = resultado
        self.llamadas = []

    def comparar(self, a, b, n_seeds=5):
        if isinstance(self.resultado, Exception):
            raise self.resultado
        self.llamadas.append((np.array(a), np.array(b)))
        return self.resultado


class PortStub:
    def __init__(self, picks, evaluador):
        self.picks = picks
        self.evaluador = evaluador


def armar(scores_2001=None, scores_2002=None, evaluador=None):
    """payload + port de 2 eventos (2001 abierto, 2002 cerrado) × 3 filas."""
    scores_2001 = scores_2001 or [[1, 0], [1, 1], [2, 1]]
    scores_2002 = scores_2002 or [[1, 0], [1, 0], [1, 0]]
    grid = np.full((6, 6), 1 / 36.0)
    payload = {
        "picks": [
            {"evento_id": 2001, "partido": "A vs B", "preferencial": True,
             "cierre_pronostico_utc": ABIERTO,
             "scores": [list(s) for s in scores_2001],
             "e_pts": [0.0, 0.0, 0.0]},
            {"evento_id": 2002, "partido": "C vs D", "preferencial": False,
             "cierre_pronostico_utc": CERRADO,
             "scores": [list(s) for s in scores_2002],
             "e_pts": [0.0, 0.0, 0.0]},
        ],
        "picks_temporada": [
            {"evento_id": 2001, "scores": [list(s) for s in scores_2001]},
            {"evento_id": 2002, "scores": [list(s) for s in scores_2002]},
        ],
    }
    picks = np.array([[score_index(*scores_2001[k]), score_index(*scores_2002[k])]
                      for k in range(3)], dtype=np.int64)
    port = PortStub(picks, evaluador)
    idx_of = {2001: 0, 2002: 1}
    grids = [grid, grid]
    return payload, port, idx_of, grids


# -------------------- parsing de marcas --------------------

def test_marcas_filtra_fecha_especiales_y_basura():
    marcas = {
        "carga:v2:4:0:2001": "1-1",       # la que importa
        "carga:v2:3:0:2001": "2-0",       # otra fecha
        "carga:v2:esp:5": "Peñarol",      # especial, no es un marcador
        "carga:v2:4:1:2001": "banana",    # valor malformado
        "carga:v2:4:2:2002": " 0-2 ",     # con espacios: se acepta
    }
    out = marcas_de_fecha(4, marcas)
    assert out == {(0, 2001): (1, 1), (2, 2002): (0, 2)}


# -------------------- diff --------------------

def test_diff_solo_partidos_abiertos_y_celdas_marcadas():
    payload, *_ = armar()
    cargado = {
        (0, 2001): (1, 0),   # igual → no cuenta
        (1, 2001): (0, 0),   # distinto y abierto → cuenta
        (0, 2002): (3, 3),   # distinto pero CERRADO → no accionable
    }
    diffs = diffs_vs_cargado(payload, cargado, NOW, 3)
    assert len(diffs) == 1
    row, cambios = diffs[0]
    assert row["evento_id"] == 2001
    assert cambios == [(1, (0, 0), (1, 1))]


def test_sin_marcas_no_hay_diff():
    payload, *_ = armar()
    assert diffs_vs_cargado(payload, {}, NOW, 3) == []


# -------------------- gate: adopta cuando el cambio no paga --------------------

def test_delta_chico_adopta_lo_cargado():
    ev = EvaluadorStub(comp(300.0))       # muy por debajo del piso de $2.000
    payload, port, idx_of, grids = armar(evaluador=ev)
    marcas = {"carga:v2:7:1:2001": "0-0"}

    aviso = aplicar_gate(payload, port, idx_of, grids, 7, 3, NUMEROS,
                         now=NOW, marcas=marcas)

    # la celda volvió a lo cargado en TODAS las copias
    assert payload["picks"][0]["scores"][1] == [0, 0]
    assert payload["picks_temporada"][0]["scores"][1] == [0, 0]
    assert index_score(int(port.picks[1, 0])) == (0, 0)
    # e_pts se recalculó para el pick adoptado (grid uniforme, preferencial x2)
    assert payload["picks"][0]["e_pts"][1] == pytest.approx(1.83, abs=0.01)
    # el resto no se tocó
    assert payload["picks"][0]["scores"][0] == [1, 0]
    v = payload["veredicto_carga"]
    assert v["avisar"] is False and v["medido"] is True and v["n_picks"] == 1
    assert aviso and "respetan" in aviso


def test_delta_grande_mantiene_el_pick_nuevo_y_avisa():
    ev = EvaluadorStub(comp(8_000.0, se=500.0))
    payload, port, idx_of, grids = armar(evaluador=ev)
    marcas = {"carga:v2:7:1:2001": "0-0"}

    aviso = aplicar_gate(payload, port, idx_of, grids, 7, 3, NUMEROS,
                         now=NOW, marcas=marcas)

    assert payload["picks"][0]["scores"][1] == [1, 1]        # el nuevo queda
    assert payload["veredicto_carga"]["avisar"] is True
    assert "YA CARGASTE" in aviso and "222" in aviso
    assert "cargaste 0-0" in aviso and "1-1" in aviso
    # la comparación fue (cargado, nuevo): el A lleva la marca
    a, b = ev.llamadas[0]
    assert index_score(int(a[1, 0])) == (0, 0)
    assert index_score(int(b[1, 0])) == (1, 1)


def test_evaluador_roto_avisa_por_las_dudas():
    ev = EvaluadorStub(RuntimeError("boom"))
    payload, port, idx_of, grids = armar(evaluador=ev)
    marcas = {"carga:v2:7:1:2001": "0-0"}

    aviso = aplicar_gate(payload, port, idx_of, grids, 7, 3, NUMEROS,
                         now=NOW, marcas=marcas)

    assert payload["picks"][0]["scores"][1] == [1, 1]        # no adopta a ciegas
    v = payload["veredicto_carga"]
    assert v["avisar"] is True and v["medido"] is False
    assert "No pude medir" in aviso


def test_sin_marcas_de_la_fecha_no_hace_nada():
    payload, port, idx_of, grids = armar(evaluador=EvaluadorStub(comp(0.0)))
    aviso = aplicar_gate(payload, port, idx_of, grids, 7, 3, NUMEROS,
                         now=NOW, marcas={"carga:v2:3:0:2001": "9-9"})
    assert aviso is None and "veredicto_carga" not in payload


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("CLAUSURA_GATE_CARGA", "0")
    payload, port, idx_of, grids = armar(evaluador=EvaluadorStub(comp(300.0)))
    aviso = aplicar_gate(payload, port, idx_of, grids, 7, 3, NUMEROS,
                         now=NOW, marcas={"carga:v2:7:1:2001": "0-0"})
    assert aviso is None
    assert payload["picks"][0]["scores"][1] == [1, 1]

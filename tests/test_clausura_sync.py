"""Tests de la guarda anti-mutilación del sync del Clausura.

El piso viejo ("n_eventos > 0") dejaba pasar un API en mantenimiento que
devolviera una sola fecha con 8 eventos: el YAML mutilado apagaba
carga_alert/rerun/gate_watch para todos los cierres futuros sin distinguirse de
"no hay partidos". check_contra_anterior exige no perder eventos ya conocidos.
"""

from __future__ import annotations

import pytest

from src.clausura.sync import check_contra_anterior


def _cfg(*ids: int) -> dict:
    return {"fechas": {"Fecha 1": {"fecha_id": 280,
                                   "eventos": [{"evento_id": i} for i in ids]}}}


def test_config_igual_o_creciente_pasa():
    check_contra_anterior(_cfg(1, 2, 3), _cfg(1, 2, 3))
    check_contra_anterior(_cfg(1, 2, 3, 4), _cfg(1, 2, 3))       # crece: OK
    check_contra_anterior(_cfg(1, 2, 3), None)                   # primer sync: OK


def test_perder_muchos_eventos_es_sospechoso():
    previo = _cfg(*range(1, 121))                                # temporada completa
    mutilado = _cfg(*range(1, 9))                                # el API devolvió 1 fecha
    with pytest.raises(RuntimeError, match="NO se pisa"):
        check_contra_anterior(mutilado, previo)


def test_reprogramacion_puntual_pasa():
    # un evento anulado y re-creado con otro id (Art. 14) es legítimo
    previo = _cfg(*range(1, 121))
    reprogramado = _cfg(*(list(range(1, 120)) + [999]))          # pierde 1, gana otro
    check_contra_anterior(reprogramado, previo)


# -------------------- aviso de cambio de estrella (auditoría 13/8) --------------------

def _cfg_pref(pref_id: int) -> dict:
    return {"fechas": {"Fecha 1": {"fecha_id": 280, "eventos": [
        {"evento_id": 1, "local": "Peñarol", "visitante": "Nacional",
         "preferencial": pref_id == 1},
        {"evento_id": 2, "local": "Cerro", "visitante": "Albion",
         "preferencial": pref_id == 2},
    ]}}}


def test_preferenciales_cambiados_detecta_recategorizacion():
    from src.clausura.sync import preferenciales_cambiados

    cambios = preferenciales_cambiados(_cfg_pref(2), _cfg_pref(1))
    assert len(cambios) == 2
    assert any("Peñarol" in c and "PERDIÓ" in c for c in cambios)
    assert any("Cerro" in c and "x2" in c for c in cambios)


def test_preferenciales_sin_cambios_ni_previo_callan():
    from src.clausura.sync import preferenciales_cambiados

    assert preferenciales_cambiados(_cfg_pref(1), _cfg_pref(1)) == []
    assert preferenciales_cambiados(_cfg_pref(1), None) == []
    # un evento nuevo (reprogramado con otro id) no es una recategorización
    nuevo = _cfg_pref(1)
    nuevo["fechas"]["Fecha 1"]["eventos"].append(
        {"evento_id": 999, "local": "X", "visitante": "Y", "preferencial": False})
    assert preferenciales_cambiados(nuevo, _cfg_pref(1)) == []

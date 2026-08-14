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


# -------------------- aviso de cierres movidos en masa (14/8) --------------------

def _cfg_cierres(horas: dict[int, str]) -> dict:
    return {"fechas": {"Fecha 2": {"fecha_id": 281, "eventos": [
        {"evento_id": eid, "local": f"L{eid}", "visitante": f"V{eid}",
         "preferencial": False, "cierre_pronostico_utc": ts}
        for eid, ts in horas.items()]}}}


def test_cierres_movidos_detecta_el_glitch_del_14_8():
    """5 partidos apilados a la misma hora del domingo: el flip-flop real."""
    from src.clausura.sync import cierres_movidos

    previo = _cfg_cierres({1: "2026-08-14T21:45:00+00:00",
                           2: "2026-08-15T12:45:00+00:00",
                           3: "2026-08-15T15:45:00+00:00",
                           4: "2026-08-16T13:45:00+00:00",
                           5: "2026-08-16T21:15:00+00:00"})
    glitch = _cfg_cierres({i: "2026-08-16T20:45:00+00:00" for i in range(1, 6)})
    avisos = cierres_movidos(glitch, previo)
    assert len(avisos) >= 4
    # y el flip de VUELTA (config envenenado → bueno) también avisa: es simétrico
    assert len(cierres_movidos(previo, glitch)) >= 4


def test_un_solo_reprogramado_no_hace_ruido():
    """Un makeup que se mueve (Art. 14) es legítimo y frecuente: silencio."""
    from src.clausura.sync import cierres_movidos

    previo = _cfg_cierres({1: "2026-08-14T21:45:00+00:00",
                           2: "2026-08-15T12:45:00+00:00",
                           3: "2026-08-15T15:45:00+00:00"})
    uno = _cfg_cierres({1: "2026-09-02T22:00:00+00:00",       # re-datado a semanas
                        2: "2026-08-15T12:45:00+00:00",
                        3: "2026-08-15T15:45:00+00:00"})
    assert cierres_movidos(uno, previo) == []


def test_corrimientos_chicos_no_cuentan():
    """±1h de ajuste fino de horario no es un glitch."""
    from src.clausura.sync import cierres_movidos

    previo = _cfg_cierres({i: "2026-08-15T12:45:00+00:00" for i in range(1, 5)})
    fino = _cfg_cierres({i: "2026-08-15T13:30:00+00:00" for i in range(1, 5)})
    assert cierres_movidos(fino, previo) == []

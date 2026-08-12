"""Tests de la corrida T-2h (src.clausura.rerun_cierre)."""

from datetime import datetime, timedelta, timezone

from src.clausura.rerun_cierre import (
    TRIGGER_H,
    cierres_en_ventana,
    debe_correr,
    diff_planillas,
    formatear_diff,
)

NOW = datetime(2026, 8, 8, 15, 40, tzinfo=timezone.utc)
NUMS = [899258848, 899258849, 899258850]


def _ev(cierre: datetime) -> dict:
    return {"evento_id": 1, "cierre_pronostico_utc": cierre.isoformat()}


def _row(eid: int, scores: list, cierre: datetime, pref: bool = False) -> dict:
    return {
        "evento_id": eid,
        "partido": "Liverpool vs Albion",
        "preferencial": pref,
        "cierre_pronostico_utc": cierre.isoformat(),
        "scores": scores,
    }


def test_cierres_en_ventana_ignora_pasados_y_lejanos():
    cerca = NOW + timedelta(hours=2)
    evs = [
        _ev(NOW - timedelta(hours=1)),                    # ya cerró
        _ev(cerca),                                       # en ventana
        _ev(cerca),                                       # mismo cierre → dedup
        _ev(NOW + timedelta(hours=TRIGGER_H + 1)),        # fuera del horizonte
    ]
    assert cierres_en_ventana(evs, NOW) == [cerca]
    assert cierres_en_ventana([], NOW) == []


def test_debe_correr_solo_en_ventana_y_una_vez_por_cierre():
    cierre = NOW + timedelta(hours=2)
    evs = [_ev(cierre)]
    assert debe_correr(evs, NOW, set())
    lejos = [_ev(NOW + timedelta(hours=TRIGGER_H + 1))]
    assert not debe_correr(lejos, NOW, set())                       # muy temprano
    assert not debe_correr(evs, NOW, {cierre.isoformat()})          # tanda ya cubierta
    assert not debe_correr([], NOW, set())                          # sin cierres


def test_cierre_pasada_la_medianoche_utc_entra_en_ventana():
    # Partido 22:00 UY viernes = cierre 00:45 UTC del sábado. Con el filtro viejo
    # por fecha calendario UTC no había rerun NUNCA para este partido.
    noche = datetime(2026, 8, 14, 23, 35, tzinfo=timezone.utc)     # tick vie 23:35
    cierre = datetime(2026, 8, 15, 0, 45, tzinfo=timezone.utc)     # sáb 00:45 UTC
    assert debe_correr([_ev(cierre)], noche, set())


def test_dos_tandas_del_mismo_dia_reciben_dos_reruns():
    temprano = NOW + timedelta(hours=2)                            # tanda 1
    tarde = NOW + timedelta(hours=8)                               # tanda 2 (fuera aún)
    evs = [_ev(temprano), _ev(tarde)]
    assert cierres_en_ventana(evs, NOW) == [temprano]              # solo la tanda 1
    corridos = {temprano.isoformat()}
    assert not debe_correr(evs, NOW, corridos)                     # tanda 1 cubierta
    despues = NOW + timedelta(hours=6)                             # tarde a 2h
    assert debe_correr(evs, despues, corridos)                     # tanda 2 dispara


def test_diff_solo_partidos_abiertos_y_columnas_cambiadas():
    abierto, cerrado = NOW + timedelta(hours=2), NOW - timedelta(hours=1)
    prev = {"picks": [_row(1, [[1, 0], [2, 1], [0, 0]], abierto),
                      _row(2, [[1, 1], [1, 1], [1, 1]], cerrado)]}
    nuevo = {"picks": [_row(1, [[1, 0], [2, 0], [0, 1]], abierto),
                       _row(2, [[9, 9], [9, 9], [9, 9]], cerrado)]}   # cerrado: se ignora
    cambios = diff_planillas(prev, nuevo, NOW)
    assert len(cambios) == 1
    row, cs = cambios[0]
    assert row["evento_id"] == 1
    assert cs == [(1, (2, 1), (2, 0)), (2, (0, 0), (0, 1))]


def test_diff_sin_cambios_es_vacio():
    abierto = NOW + timedelta(hours=2)
    p = {"picks": [_row(1, [[1, 0], [2, 1]], abierto)]}
    assert diff_planillas(p, p, NOW) == []


def test_formato_mapea_columnas_a_numeros():
    abierto = NOW + timedelta(hours=2)
    cambios = [(_row(1, [], abierto, pref=True), [(0, (1, 0), (2, 0)), (2, (0, 0), (1, 1))])]
    txt = formatear_diff(cambios, NUMS, NOW)
    assert "⭐x2" in txt and "HOY" in txt
    assert f"{NUMS[0]}: 1-0 → " in txt and "2-0" in txt
    assert f"{NUMS[2]}: 0-0 → " in txt
    assert "SOLO estos picks" in txt

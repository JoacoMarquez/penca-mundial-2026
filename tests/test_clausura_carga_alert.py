"""Tests de las alertas de carga (src.clausura.carga_alert)."""

from datetime import datetime, timedelta, timezone

from src.clausura.carga_alert import (
    TIERS_H,
    eventos_por_cerrar,
    formatear_alerta,
    pendientes_de_alerta,
    tier_activo,
)

NOW = datetime(2026, 8, 8, 15, 0, tzinfo=timezone.utc)


def _ev(eid: int, cierre: datetime, pref: bool = False) -> dict:
    return {
        "evento_id": eid,
        "local": "Liverpool",
        "visitante": "Albion",
        "preferencial": pref,
        "cierre_pronostico_utc": cierre.isoformat(),
    }


def test_eventos_por_cerrar_filtra_pasados_y_lejanos():
    evs = [
        _ev(1, NOW - timedelta(hours=1)),                 # ya cerró
        _ev(2, NOW + timedelta(hours=3)),                 # dentro del horizonte
        _ev(3, NOW + timedelta(hours=max(TIERS_H) + 1)),  # demasiado lejos
    ]
    assert [ev["evento_id"] for ev, _ in eventos_por_cerrar(evs, NOW)] == [2]


def test_tier_activo_elige_el_nivel_mas_urgente_alcanzado():
    assert tier_activo(NOW + timedelta(hours=5), NOW) == 6.0
    assert tier_activo(NOW + timedelta(hours=1.5), NOW) == 2.0
    assert tier_activo(NOW + timedelta(hours=7), NOW) is None


def test_pendientes_deduplica_por_evento_y_tier():
    evs = [_ev(2, NOW + timedelta(hours=3))]
    pend = pendientes_de_alerta(evs, NOW, ya_avisados=set())
    assert len(pend) == 1
    _, _, tier, clave = pend[0]
    assert tier == 6.0 and clave == "2:6"
    assert pendientes_de_alerta(evs, NOW, ya_avisados={clave}) == []
    # al entrar en la ventana de 2h aparece el nivel siguiente, con otra clave
    pend2 = pendientes_de_alerta(evs, NOW + timedelta(hours=2), ya_avisados={clave})
    assert pend2[0][3] == "2:2"


def test_formato_con_faltantes_y_sin_faltantes():
    ev = _ev(2, NOW + timedelta(hours=1), pref=True)
    cierre = NOW + timedelta(hours=1)
    msg = formatear_alerta(ev, cierre, 2.0, faltantes=[899258848, 899258850], n_participaciones=12)
    assert "🚨" in msg and "2/12" in msg and "899258848" in msg and "⭐x2" in msg
    assert formatear_alerta(ev, cierre, 2.0, faltantes=[], n_participaciones=12) is None


def test_formato_pre_inicio_es_recordatorio_ciego():
    ev = _ev(2, NOW + timedelta(hours=5))
    msg = formatear_alerta(ev, NOW + timedelta(hours=5), 6.0, faltantes=None, n_participaciones=12)
    assert "No puedo verificar" in msg and "⏰" in msg

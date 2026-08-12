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


def test_pendientes_deduplica_por_evento_cierre_y_tier():
    cierre = NOW + timedelta(hours=3)
    evs = [_ev(2, cierre)]
    pend = pendientes_de_alerta(evs, NOW, ya_avisados=set())
    assert len(pend) == 1
    _, _, tier, clave = pend[0]
    assert tier == 6.0 and clave == f"2:{cierre:%Y%m%dT%H%M}:6"
    assert pendientes_de_alerta(evs, NOW, ya_avisados={clave}) == []
    # al entrar en la ventana de 2h aparece el nivel siguiente, con otra clave
    pend2 = pendientes_de_alerta(evs, NOW + timedelta(hours=2), ya_avisados={clave})
    assert pend2[0][3] == f"2:{cierre:%Y%m%dT%H%M}:2"


def test_partido_reprogramado_vuelve_a_avisar():
    """La clave lleva el cierre: si el admin re-data el partido (Torque-Peñarol, F1),
    las claves quemadas del cierre original no silencian el makeup."""
    original = NOW - timedelta(days=5)
    avisados = {f"2:{original:%Y%m%dT%H%M}:6", f"2:{original:%Y%m%dT%H%M}:2"}
    makeup = NOW + timedelta(hours=3)
    pend = pendientes_de_alerta([_ev(2, makeup)], NOW, ya_avisados=avisados)
    assert len(pend) == 1 and pend[0][3] == f"2:{makeup:%Y%m%dT%H%M}:6"


def test_formato_con_faltantes_y_sin_faltantes():
    ev = _ev(2, NOW + timedelta(hours=1), pref=True)
    cierre = NOW + timedelta(hours=1)
    msg = formatear_alerta(ev, cierre, 2.0, faltantes=[899258848, 899258850], n_participaciones=12)
    assert "🚨" in msg and "2/12" in msg and "899258848" in msg and "⭐x2" in msg
    assert formatear_alerta(ev, cierre, 2.0, faltantes=[], n_participaciones=12) is None


def test_formato_sin_verificacion_es_recordatorio():
    """Con faltantes=None el aviso recuerda, no acusa.

    El texto cambió el 2026-08-09: decía "no puedo verificar hasta que inicie el
    campeonato", que sugiere que después sí podría. No puede nunca antes del cierre —
    el gate del API es por partido.
    """
    ev = _ev(2, NOW + timedelta(hours=5))
    msg = formatear_alerta(ev, NOW + timedelta(hours=5), 6.0, faltantes=None, n_participaciones=12)
    assert "Verificá a mano" in msg and "⏰" in msg
    assert "faltan" not in msg.lower()


# -------------------- el gate es por partido (regresión 2026-08-09) --------------------

def test_partido_abierto_no_cuenta_faltantes():
    """Antes del cierre el API no publica NUESTROS picks: no se puede verificar.

    Contarlos como faltantes daba siempre "faltan 12/12" con las 12 cargadas. Salieron
    14 avisos falsos al Telegram el 8-9/8, en el mismo canal donde drift_audit manda lo
    que sí cuesta puntos.
    """
    from datetime import datetime, timedelta, timezone
    from src.clausura.carga_alert import formatear_alerta

    cierre = datetime.now(timezone.utc) + timedelta(hours=2)
    msg = formatear_alerta({"local": "Nacional", "visitante": "Boston River"},
                           cierre, 2.0, None, 12)
    assert msg is not None
    assert "faltan" not in msg.lower()
    assert "verificá a mano" in msg.lower()
    # y no puede prometer que después va a poder: no es cuestión de esperar
    assert "hasta que inicie el campeonato" not in msg.lower()


def test_post_cierre_si_puede_contar_faltantes():
    """Cerrado el partido el gate abre y el conteo vuelve a ser real."""
    from datetime import datetime, timedelta, timezone
    from src.clausura.carga_alert import formatear_alerta

    cierre = datetime.now(timezone.utc) - timedelta(hours=1)
    msg = formatear_alerta({"local": "Nacional", "visitante": "Boston River"},
                           cierre, 2.0, [899258848, 899258854], 12)
    assert "2/12" in msg and "899258848" in msg


def test_todo_cargado_no_avisa():
    from datetime import datetime, timedelta, timezone
    from src.clausura.carga_alert import formatear_alerta
    cierre = datetime.now(timezone.utc) - timedelta(hours=1)
    assert formatear_alerta({"local": "A", "visitante": "B"}, cierre, 2.0, [], 12) is None

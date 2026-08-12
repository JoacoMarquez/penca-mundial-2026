"""Tests del vigía del gate: clasificación, transiciones y formato (sin red)."""

from datetime import datetime, timezone

from src.clausura.gate_watch import clasificar, formatear_alerta, necesita_alerta

NOW = datetime(2026, 8, 7, 15, 0, tzinfo=timezone.utc)

EVENTOS = [
    {"evento_id": 2086, "local": "Cerro Largo", "visitante": "Juventud",
     "inicio_utc": "2026-08-07T22:00:00+00:00",
     "cierre_pronostico_utc": "2026-08-07T21:45:00+00:00"},
    {"evento_id": 2083, "local": "Central Español", "visitante": "Progreso",
     "inicio_utc": "2026-08-08T14:00:00+00:00",
     "cierre_pronostico_utc": "2026-08-08T13:45:00+00:00"},
    {"evento_id": 2099, "local": "Nacional", "visitante": "Progreso",
     "inicio_utc": "2026-08-21T21:00:00+00:00",
     "cierre_pronostico_utc": "2026-08-21T20:45:00+00:00"},
]


# -------------------- clasificar --------------------

def test_clasificar_evento_abierto_es_anomalia():
    cur = clasificar({2086, 2099}, False, EVENTOS, NOW)
    assert cur["abiertos"] == [2083, 2099] or cur["abiertos"] == [2086, 2099]
    # ambos cierres son futuros a las 15:00 del 7/8
    assert set(cur["abiertos"]) == {2086, 2099}


def test_clasificar_evento_cerrado_no_cuenta():
    despues = datetime(2026, 8, 7, 22, 30, tzinfo=timezone.utc)  # 2086 ya cerró
    cur = clasificar({2086}, False, EVENTOS, despues)
    assert cur["abiertos"] == []


def test_clasificar_especiales_pre_y_post_inicio():
    pre = clasificar(set(), True, EVENTOS, NOW)
    assert pre["especiales_pre_inicio"] is True
    post = clasificar(set(), True, EVENTOS,
                      datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc))
    assert post["especiales_pre_inicio"] is False  # post-kickoff es legítimo


def test_clasificar_ignora_eventos_desconocidos():
    cur = clasificar({99999}, False, EVENTOS, NOW)
    assert cur["abiertos"] == []


# -------------------- transiciones --------------------

def test_alerta_solo_en_transicion_cerrado_a_abierto():
    cerrado = {"abiertos": [], "especiales_pre_inicio": False}
    abierto = {"abiertos": [2086], "especiales_pre_inicio": False}
    assert necesita_alerta({}, abierto) is True                      # primer tick abierto
    assert necesita_alerta({"abiertos": False}, abierto) is True
    assert necesita_alerta({"abiertos": True}, abierto) is False     # persiste: silencio
    assert necesita_alerta({"abiertos": True}, cerrado) is False     # cerró: silencio
    assert necesita_alerta({}, cerrado) is False


def test_alerta_por_especiales_pre_inicio():
    cur = {"abiertos": [], "especiales_pre_inicio": True}
    assert necesita_alerta({}, cur) is True
    assert necesita_alerta({"especiales": True}, cur) is False
    # reapertura después de un cierre → alerta de nuevo
    assert necesita_alerta({"especiales": False}, cur) is True


# -------------------- formato --------------------

def test_formatear_alerta_lista_partidos_y_snapshot():
    cur = {"abiertos": [2086], "especiales_pre_inicio": True}
    msg = formatear_alerta(cur, EVENTOS, "507 participaciones · 20 con marcadores")
    assert "Cerro Largo vs Juventud" in msg
    assert "especiales" in msg
    assert "507 participaciones" in msg


def test_formatear_alerta_sin_snapshot_avisa():
    cur = {"abiertos": [2086], "especiales_pre_inicio": False}
    msg = formatear_alerta(cur, EVENTOS, None)
    assert "No pude capturar" in msg


# -------------------- escalación de fallos persistentes --------------------

def test_fallas_consecutivas_escalan_una_vez(tmp_path, monkeypatch):
    """Un tick fallado se traga (exit 0); FALLAS_PARA_ESCALAR consecutivos escalan
    UNA vez y el contador vuelve a cero — ni silencio eterno ni 144 avisos/día."""
    import src.clausura.gate_watch as gw
    monkeypatch.setattr(gw, "FALLAS_PATH", tmp_path / "fallas.json")

    for i in range(1, gw.FALLAS_PARA_ESCALAR):
        assert gw._contar_falla() == i
    assert gw._contar_falla() == gw.FALLAS_PARA_ESCALAR
    gw._reset_fallas()                      # lo que hace main() al escalar
    assert gw._contar_falla() == 1          # y se vuelve a contar desde cero
    gw._reset_fallas()
    assert not (tmp_path / "fallas.json").exists()

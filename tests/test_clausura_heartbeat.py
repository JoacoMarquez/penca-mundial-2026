"""Tests del heartbeat diario del stack Clausura."""

from datetime import datetime, timezone

import src.clausura.heartbeat as hb


def test_mensaje_reporta_planilla_y_proximo_cierre(monkeypatch, tmp_path):
    monkeypatch.setattr(hb, "_timers_estado", lambda: (list(hb.TIMERS), []))
    # intermedio presente
    intermedio = tmp_path / "intermedio_2026.json"
    intermedio.write_text("[]")
    import src.clausura.intermedio as im
    monkeypatch.setattr(im, "OUT_PATH", intermedio)

    msg = hb.construir_mensaje(now=datetime(2026, 8, 6, 11, 0, tzinfo=timezone.utc))
    assert "Clausura vivo" in msg
    assert "timers: 6/6" in msg
    # con el fixture real: hay un próximo cierre por delante
    assert "próximo cierre" in msg


def test_mensaje_marca_problemas(monkeypatch, tmp_path):
    monkeypatch.setattr(hb, "_timers_estado",
                        lambda: ([], ["clausura-picks: NO LISTADO (¿deshabilitado?)"]))
    import src.clausura.intermedio as im
    monkeypatch.setattr(im, "OUT_PATH", tmp_path / "no_existe.json")

    msg = hb.construir_mensaje(now=datetime(2026, 8, 6, 11, 0, tzinfo=timezone.utc))
    assert "PROBLEMAS" in msg
    assert "NO LISTADO" in msg
    assert "intermedio_2026.json AUSENTE" in msg

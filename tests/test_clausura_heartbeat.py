"""Tests del heartbeat diario del stack Clausura."""

from datetime import datetime, timezone

import pytest

import src.clausura.heartbeat as hb
import src.clausura.offline as off


@pytest.fixture(autouse=True)
def _online(monkeypatch):
    """Hermético: sin esto, en el VPS con el archivo offline activo el test le
    pegaría al API de verdad."""
    monkeypatch.setattr(off, "offline_activo", lambda *a, **k: False)
    monkeypatch.setattr(off, "probe_api", lambda *a, **k: 403)


def test_mensaje_reporta_planilla_y_proximo_cierre(monkeypatch, tmp_path):
    monkeypatch.setattr(hb, "_timers_estado", lambda: (list(hb.TIMERS), []))
    # intermedio presente
    intermedio = tmp_path / "intermedio_2026.json"
    intermedio.write_text("[]")
    import src.clausura.intermedio as im
    monkeypatch.setattr(im, "OUT_PATH", intermedio)

    msg = hb.construir_mensaje(now=datetime(2026, 8, 6, 11, 0, tzinfo=timezone.utc))
    assert "Clausura vivo" in msg
    n = len(hb.TIMERS)
    assert f"timers: {n}/{n}" in msg
    # gate-watch tiene que estar vigilado: su timer caído era silencio total
    assert "clausura-gate-watch" in hb.TIMERS
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


def test_timer_con_servicio_corriendo_no_es_falsa_alarma(monkeypatch):
    """El heartbeat corre a las 12:30:15 y SIEMPRE pisa el tick de las 12:30:00 del
    gate-watch (y se ve a sí mismo): un timer cuyo servicio está en ejecución
    muestra '-' como próxima corrida. Sin el chequeo de is-active, la falsa alarma
    salía TODOS los días (pasó el 14/8, primer heartbeat con los 9 timers)."""
    import subprocess

    salida = "\n".join(
        [f"Sat 2026-08-15 12:00:00 UTC 1h left Fri - - {t}.timer {t}.service"
         for t in hb.TIMERS if t not in ("clausura-gate-watch", "clausura-heartbeat")]
        + ["- - - - clausura-gate-watch.timer clausura-gate-watch.service",
           "- - - - clausura-heartbeat.timer clausura-heartbeat.service"])

    corriendo = {"clausura-gate-watch.service", "clausura-heartbeat.service"}

    def fake_run(cmd, **kw):
        class R:
            pass
        r = R()
        if "list-timers" in cmd:
            r.stdout = salida
            r.returncode = 0
        else:                                   # systemctl is-active <unit>
            r.returncode = 0 if cmd[-1] in corriendo else 3
        return r

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, mal = hb._timers_estado()
    assert mal == []                            # los dos "sin próxima" estaban corriendo
    assert set(ok) == set(hb.TIMERS)

    # y si el servicio NO está corriendo, la alarma sigue siendo real
    corriendo.clear()
    ok2, mal2 = hb._timers_estado()
    assert any("gate-watch" in m for m in mal2)


def test_offline_no_cuenta_los_pausados_y_avisa_si_el_api_vuelve(monkeypatch, tmp_path):
    pausados = [f"{t}: NO LISTADO (¿deshabilitado?)" for t in off.TIMERS_PAUSADOS
                if t in hb.TIMERS]
    vivos = [t for t in hb.TIMERS if t not in off.TIMERS_PAUSADOS]
    monkeypatch.setattr(hb, "_timers_estado", lambda: (vivos, pausados))
    monkeypatch.setattr(off, "offline_activo", lambda *a, **k: True)
    intermedio = tmp_path / "intermedio_2026.json"
    intermedio.write_text("[]")
    import src.clausura.intermedio as im
    monkeypatch.setattr(im, "OUT_PATH", intermedio)
    now = datetime(2026, 8, 6, 11, 0, tzinfo=timezone.utc)

    msg = hb.construir_mensaje(now=now)
    assert "MODO OFFLINE" in msg and "403" in msg
    assert "NO LISTADO" not in msg
    assert f"timers: {len(vivos)}/{len(vivos)}" in msg

    monkeypatch.setattr(off, "probe_api", lambda *a, **k: 200)
    assert "RESPONDE 200" in hb.construir_mensaje(now=now)

"""Tests del loader de la página Clausura del dashboard (sin red)."""

import json

import pytest

from src.clausura.dashboard_loader import _to_uy, fecha_actual, load_planilla


def test_to_uy_dia_en_espanol():
    # 2026-08-07 21:45 UTC = viernes 18:45 en UY (UTC-3)
    assert _to_uy("2026-08-07T21:45:00+00:00") == "vie 07/08 18:45"


def test_to_uy_string_invalido_no_rompe():
    assert _to_uy("no-es-fecha") == "no-es-fecha"


def _cfg(f1_inicio: str, f2_inicio: str) -> dict:
    return {"fechas": {
        "Fecha 1": {"fecha_id": 280, "eventos": [{"inicio_utc": f1_inicio}]},
        "Fecha 2": {"fecha_id": 281, "eventos": [{"inicio_utc": f2_inicio}]},
    }}


def test_fecha_actual_primera_pendiente():
    cfg = _cfg("2099-01-01T00:00:00+00:00", "2099-02-01T00:00:00+00:00")
    assert fecha_actual(cfg) == 1


def test_fecha_actual_saltea_jugadas():
    cfg = _cfg("2020-01-01T00:00:00+00:00", "2099-02-01T00:00:00+00:00")
    assert fecha_actual(cfg) == 2


def test_fecha_actual_todo_jugado_devuelve_15():
    cfg = _cfg("2020-01-01T00:00:00+00:00", "2020-02-01T00:00:00+00:00")
    assert fecha_actual(cfg) == 15


def test_load_planilla_enriquece(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)
    # dashboard_loader importa fecha_dir/PRED_DIR desde picks — parchear ambos usos
    import src.clausura.dashboard_loader as dl
    monkeypatch.setattr(dl, "PRED_DIR", tmp_path)

    d = tmp_path / "fecha_03"
    d.mkdir(parents=True)
    (d / "v1_20260804T120000Z.json").write_text(json.dumps({
        "generado_utc": "2026-08-04T12:00:00+00:00",
        "n_participaciones": 2,
        "picks": [{
            "evento_id": 1, "partido": "A vs B", "preferencial": True,
            "cierre_pronostico_utc": "2020-01-01T00:00:00+00:00",   # ya cerró
            "fuente_modelo": "ratings",
            "scores": [[1, 0], [0, 0]],
        }],
    }), encoding="utf-8")

    p = load_planilla(3)
    assert p is not None
    assert p["version_file"].startswith("v1_")
    row = p["picks"][0]
    assert row["cerrado"] is True
    assert row["scores_fmt"] == ["1-0", "0-0"]


def test_load_planilla_fecha_inexistente(tmp_path, monkeypatch):
    import src.clausura.dashboard_loader as dl
    monkeypatch.setattr(dl, "PRED_DIR", tmp_path)
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)
    assert load_planilla(9) is None


def test_webapp_token(monkeypatch):
    """El app standalone respeta DASHBOARD_TOKEN (404 con token malo, 503 sin config)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from src.clausura.webapp import app

    client = TestClient(app)
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)
    assert client.get("/dash/x/").status_code == 503
    monkeypatch.setenv("DASHBOARD_TOKEN", "bueno")
    assert client.get("/dash/malo/").status_code == 404
    assert client.get("/").json()["status"] == "ok"

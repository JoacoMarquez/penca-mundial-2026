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


# -------------------- tarjeta de la penca gratuita --------------------

def _cfg_pencas():
    return {"pencas": {"paga": {"id": 46, "precio": 400.0}, "gratuita": {"id": 47}}}


def _planilla_stub():
    return {
        "picks": [
            {"partido": "Liverpool vs Albion", "preferencial": False,
             "cierre_uy": "vie 07/08 18:45", "cerrado": False,
             "scores": [[1, 0], [0, 0], [2, 1]]},
            {"partido": "Defensor vs Wanderers", "preferencial": True,
             "cierre_uy": "sáb 08/08 19:45", "cerrado": False,
             "scores": [[2, 1], [1, 1], [1, 0]]},
        ],
        "especiales": {
            "p_campeon": {"Peñarol": 0.42, "Nacional": 0.31, "Liverpool": 0.09},
            "por_participacion": [
                {"campeon": "Nacional", "goleador": "Cabrera"},   # fila 1 diversificada
                {"campeon": "Peñarol", "goleador": "Cabrera"},
            ],
        },
    }


def test_gratuita_usa_columna_1_y_campeon_mas_probable(monkeypatch):
    """Marcadores = columna 1 (ancla EV). Campeón = argmax P(campeón), que NO es
    el de la fila 1 (los especiales sí se diversifican en todas las columnas)."""
    import src.clausura.dashboard_loader as dl
    monkeypatch.setattr(dl, "load_ranking", lambda pid, mios=None: {"ok": False, "rows": []})
    monkeypatch.delenv("CLAUSURA_MI_PARTICIPACION_GRATUITA", raising=False)

    g = dl.build_gratuita(_planilla_stub(), _cfg_pencas())
    assert g["ok"] and g["penca_id"] == 47
    assert [p["score"] for p in g["picks"]] == ["1-0", "2-1"]
    assert g["picks"][1]["preferencial"] is True
    assert g["campeon"]["equipo"] == "Peñarol"      # argmax, no "Nacional" de la fila 1
    assert g["goleador"] == "Cabrera"
    assert g["mi_numero"] is None


def test_gratuita_resalta_mi_numero_en_su_ranking(monkeypatch):
    import src.clausura.dashboard_loader as dl
    visto = {}

    def fake_ranking(pid, mios=None):
        visto["pid"], visto["mios"] = pid, mios
        return {"ok": True, "rows": [], "total": 245}

    monkeypatch.setattr(dl, "load_ranking", fake_ranking)
    monkeypatch.setenv("CLAUSURA_MI_PARTICIPACION_GRATUITA", "899299999")

    g = dl.build_gratuita(_planilla_stub(), _cfg_pencas())
    assert g["mi_numero"] == 899299999
    assert visto["pid"] == 47                       # ranking de la GRATUITA, no la paga
    assert visto["mios"] == frozenset({899299999})


def test_gratuita_sin_planilla_no_rompe(monkeypatch):
    import src.clausura.dashboard_loader as dl
    monkeypatch.setattr(dl, "load_ranking", lambda pid, mios=None: {"ok": False, "rows": []})
    g = dl.build_gratuita(None, _cfg_pencas())
    assert g["ok"] and g["picks"] == [] and g["campeon"] is None


def test_gratuita_sin_config_de_penca_queda_apagada(monkeypatch):
    import src.clausura.dashboard_loader as dl
    g = dl.build_gratuita(_planilla_stub(), {"pencas": {"paga": {"id": 46}}})
    assert g["ok"] is False


# -------------------- diff entre corridas (las "pasadas") --------------------

def _version(tmp_path, n, ts, picks, especiales=None):
    import json as _json
    d = tmp_path / "fecha_01"
    d.mkdir(exist_ok=True)
    payload = {"generado_utc": f"2026-08-0{ts}T12:00:00+00:00", "picks": picks}
    if especiales:
        payload["especiales"] = {"por_participacion": especiales}
    (d / f"v{n}_2026080{ts}T120000Z.json").write_text(_json.dumps(payload), encoding="utf-8")


def _row(eid, scores, fuente="mercado+ratings", cierre="2026-12-31T23:45:00+00:00"):
    return {"evento_id": eid, "partido": f"Partido {eid}", "fuente_modelo": fuente,
            "cierre_pronostico_utc": cierre, "scores": scores}


def test_diff_versiones_detecta_cambios_de_pick_fuente_y_especial(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    from src.clausura.dashboard_loader import diff_versiones
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    _version(tmp_path, 1, 5, [_row(10, [[1, 0], [2, 1]], fuente="ratings")],
             especiales=[{"campeon": "Nacional"}, {"campeon": "Nacional"}])
    _version(tmp_path, 2, 6, [_row(10, [[1, 0], [0, 0]], fuente="mercado+ratings")],
             especiales=[{"campeon": "Nacional"}, {"campeon": "Peñarol"}])

    d = diff_versiones(1, mis_numeros=[899258848, 899258854])
    assert d["n_versiones"] == 2
    assert d["cambios"] == [{"partido": "Partido 10", "numero": 899258854,
                             "antes": "2-1", "despues": "0-0", "cerrado": False}]
    assert d["fuentes"][0]["despues"] == "mercado+ratings"
    assert d["especiales"] == [{"numero": 899258854, "campo": "campeon",
                                "antes": "Nacional", "despues": "Peñarol"}]


def test_diff_versiones_sin_cambios_ni_versiones(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    from src.clausura.dashboard_loader import diff_versiones
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    assert diff_versiones(1, [1])["n_versiones"] == 0     # sin archivos no rompe

    _version(tmp_path, 1, 5, [_row(10, [[1, 0]])])
    assert diff_versiones(1, [1])["n_versiones"] == 1     # una sola: sin diff

    _version(tmp_path, 2, 6, [_row(10, [[1, 0]])])
    d = diff_versiones(1, [1])
    assert d["n_versiones"] == 2
    assert d["cambios"] == [] and d["especiales"] == [] and d["fuentes"] == []


def test_diff_versiones_marca_cerrados(tmp_path, monkeypatch):
    """Un cambio en un partido ya cerrado se muestra pero marcado (no accionable)."""
    import src.clausura.picks as picks_mod
    from src.clausura.dashboard_loader import diff_versiones
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    viejo = "2020-01-01T00:00:00+00:00"
    _version(tmp_path, 1, 5, [_row(10, [[1, 0]], cierre=viejo)])
    _version(tmp_path, 2, 6, [_row(10, [[2, 2]], cierre=viejo)])
    d = diff_versiones(1, [1])
    assert d["cambios"][0]["cerrado"] is True

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


def test_fecha_actual_makeup_reprogramado_no_clava_la_fecha_vieja():
    """Un suspendido de la F1 re-datado DESPUÉS de la F2 (caso Torque-Peñarol) no
    debe dejar la F1 como "actual" mientras la F2 es lo próximo que se juega."""
    cfg = {"fechas": {
        "Fecha 1": {"fecha_id": 280, "eventos": [
            {"inicio_utc": "2020-01-01T00:00:00+00:00"},     # jugado
            {"inicio_utc": "2099-03-01T00:00:00+00:00"},     # makeup, lejos
        ]},
        "Fecha 2": {"fecha_id": 281, "eventos": [
            {"inicio_utc": "2099-02-01T00:00:00+00:00"},     # el próximo real
        ]},
    }}
    assert fecha_actual(cfg) == 2


def test_fecha_actual_el_dia_del_makeup_vuelve_a_la_fecha_vieja():
    """Cuando el makeup es lo próximo por jugarse, SU fecha es la actual: hay que
    generarle planilla y cargarlo."""
    cfg = {"fechas": {
        "Fecha 1": {"fecha_id": 280, "eventos": [
            {"inicio_utc": "2099-01-15T00:00:00+00:00"},     # makeup, lo próximo
        ]},
        "Fecha 2": {"fecha_id": 281, "eventos": [
            {"inicio_utc": "2099-02-01T00:00:00+00:00"},
        ]},
    }}
    assert fecha_actual(cfg) == 1


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


def test_load_planilla_saltea_las_versiones_descartadas_por_el_gate(tmp_path, monkeypatch):
    """El rerun versiona su planilla aunque el gate la descarte: al que todavía no
    cargó, el modo carga le servía la descartada (a veces PEOR que la de la
    mañana). Se sirve la última APROBADA y se cuenta cuántas quedaron después."""
    import src.clausura.picks as picks_mod
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)
    import src.clausura.dashboard_loader as dl
    monkeypatch.setattr(dl, "PRED_DIR", tmp_path)

    d = tmp_path / "fecha_03"
    d.mkdir(parents=True)

    def _v(n, scores, veredicto=None):
        payload = {
            "generado_utc": f"2026-08-04T1{n}:00:00+00:00",
            "picks": [{"evento_id": 1, "partido": "A vs B", "preferencial": False,
                       "cierre_pronostico_utc": "2099-01-01T00:00:00+00:00",
                       "scores": scores}],
        }
        if veredicto is not None:
            payload["veredicto_cambio"] = veredicto
        (d / f"v{n}_20260804T1{n}0000Z.json").write_text(
            json.dumps(payload), encoding="utf-8")

    _v(1, [[1, 0]])                                              # mañana: aprobada
    _v(2, [[2, 2]], {"avisar": False, "medido": True})           # rerun descartado
    _v(3, [[3, 3]], {"avisar": False, "medido": True})           # otro descartado

    p = load_planilla(3)
    assert p["version_file"].startswith("v1_")
    assert p["picks"][0]["scores_fmt"] == ["1-0"]
    assert p["descartadas_despues"] == 2

    # un rerun cuyo cambio SÍ vale (avisar=True) es la planilla a cargar
    _v(4, [[4, 0]], {"avisar": True, "medido": True})
    p = load_planilla(3)
    assert p["version_file"].startswith("v4_")
    assert p["descartadas_despues"] == 0


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


def test_modo_carga_incluye_el_vigia_de_version(monkeypatch):
    """La pestaña abierta es el único camino a cargar picks viejos: el modo carga
    tiene que traer el watcher que avisa si apareció una versión más nueva."""
    import os
    os.environ["DASHBOARD_TOKEN"] = "t"
    from fastapi.testclient import TestClient
    import src.clausura.dashboard_loader as dl
    import src.clausura.webapp as webapp

    monkeypatch.setattr(dl, "load_planilla", lambda n: {
        "picks": [{"partido": "A vs B", "preferencial": False, "cierre_uy": "x",
                   "cerrado": False, "scores": [[1, 0]], "scores_fmt": ["1-0"],
                   "evento_id": 1, "cierre_pronostico_utc": "2026-12-01T00:00:00+00:00"}],
        "resultado_sim": {"e_premio_total": 1.0, "e_premio_penca": 1.0,
                          "e_premio_fechas": 0.0, "p_gana_penca": 0.1},
        "pool": {"n_rivales": 1, "temperatura": 1.0},
        "n_participaciones": 1, "version_file": "v8_x.json",
        "generado_uy": "x", "n_fechas_guardadas": 1,
    })
    monkeypatch.setattr(dl, "load_ranking", lambda pid, mios=None: {"ok": False, "rows": []})

    html = TestClient(webapp.app).get("/dash/t/carga/").text
    assert "version-watch" in html
    assert '"v8_x.json"' in html          # la versión actual queda embebida para comparar
    assert "api/data?fecha=" in html      # y el poll apunta al endpoint de datos


# -------------------- modo carga: marcas por valor --------------------

def _render_carga(picks: list[dict], n_part: int = 2, version: str = "v3_x.json") -> str:
    """Renderiza carga.html sin fastapi (el template sólo usa `data` y `token`)."""
    from pathlib import Path

    from jinja2 import Environment, FileSystemLoader

    tpl_dir = Path(__file__).resolve().parents[1] / "src" / "clausura" / "templates"
    env = Environment(loader=FileSystemLoader(str(tpl_dir)))
    data = {
        "ok": True, "fecha_n": 4, "supermatch_url": "http://x",
        "mis_numeros": [899258848, 899258849][:n_part],
        "planilla": {
            "n_participaciones": n_part, "version_file": version,
            "generado_uy": "vie 08/08 10:00", "picks": picks,
        },
    }
    return env.get_template("carga.html").render(data=data, token="t")


def _pick(evento_id: int, scores_fmt: list[str], cerrado: bool = False) -> dict:
    return {"partido": f"L{evento_id} vs V{evento_id}", "preferencial": False,
            "cerrado": cerrado, "evento_id": evento_id, "scores_fmt": scores_fmt}


def test_modo_carga_cada_fila_lleva_evento_y_valor():
    """La marca de 'cargada' guarda EL VALOR, así que cada fila tiene que exponer su
    evento_id (clave estable) y el marcador de ESA participación."""
    html = _render_carga([_pick(111, ["2-1", "1-1"]), _pick(112, ["1-0", "0-0"])])

    assert 'data-part="0" data-ev="111" data-val="2-1"' in html
    assert 'data-part="1" data-ev="111" data-val="1-1"' in html
    assert 'data-part="1" data-ev="112" data-val="0-0"' in html


def test_modo_carga_la_clave_no_depende_de_la_version():
    """Regresión: con la versión en la clave, una planilla nueva borraba todo el
    progreso y el cambio de pick pasaba invisible. La clave es (fecha, part, evento)."""
    html = _render_carga([_pick(111, ["2-1", "1-1"])], version="v9_zzz.json")

    assert "`carga:v2:${FECHA}:${part}:${ev}`" in html
    # el version_file sigue embebido (vigía + migración) pero no como clave de fila
    assert 'carga:v2:${FECHA}:${part}:${ev}:${VER}' not in html
    assert "migrado" in html          # las marcas del esquema viejo se rescatan una vez


def test_modo_carga_trae_el_panel_de_cambios():
    """El desfasaje se muestra: panel con la lista antes → después y aviso por fila."""
    html = _render_carga([_pick(111, ["2-1", "1-1"])])

    assert 'id="drift-panel"' in html and 'id="drift-list"' in html
    assert "todavía se pueden corregir" in html
    assert 'class="aviso' in html          # el "cargaste X → corregí a Y" de cada fila


def test_modo_carga_marca_los_cerrados():
    """Un partido ya cerrado no se puede corregir: la fila lo declara para que el
    aviso no pida ir a arreglar algo que la web ya no deja tocar."""
    html = _render_carga([_pick(111, ["2-1", "1-1"], cerrado=True)])

    assert 'data-cerrado="1"' in html
    assert "ya cerrado" in html


# -------------------- modo carga: lo que acelera el tipeo --------------------

def _render_carga_fixture(n_part=3, n_partidos=4, cerrados=0):
    """Renderiza carga.html sin levantar la app (no necesita fastapi)."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("src/clausura/templates"))
    picks = [{
        "evento_id": 2000 + j, "partido": f"Local {j} vs Visita {j}",
        "preferencial": j == 0, "cerrado": j < cerrados,
        "scores_fmt": [f"{j}-{k}" for k in range(n_part)],
    } for j in range(n_partidos)]
    data = {
        "ok": True, "fecha_n": 2, "supermatch_url": "https://ejemplo",
        "mis_numeros": [899258848 + i for i in range(n_part)],
        "planilla": {"n_participaciones": n_part, "picks": picks,
                     "version_file": "v3_x.json", "generado_uy": "mar 11/08 08:00",
                     "especiales": None},
    }
    return env.get_template("carga.html").render(data=data, token="t")


def test_modo_carga_tiene_filtro_y_progreso_por_participacion():
    """Con 96 picks, un contador global 0/12 no se mueve en ocho toques seguidos.

    El filtro y el contador por tarjeta son lo que hace navegable la carga manual:
    sin ellos hay que buscar a ojo dónde quedaste cada vez que se vuelve de la web.
    """
    html = _render_carga_fixture()
    assert 'id="btn-filtro"' in html, "falta el toggle de 'solo lo que falta'"
    # sólo las del marcado; la cuarta aparición es el querySelector del JS
    assert html.count('class="chip-progreso') == 3, "cada participación necesita su contador"
    # las clases que sólo aparecen por JS tienen que estar sembradas para el CDN de
    # Tailwind, si no el aro de la fila siguiente no se ve
    for clase in ("ring-indigo-400/70", "bg-emerald-600"):
        assert clase in html, f"{clase} no está sembrada para el JIT"


def test_modo_carga_no_esconde_los_picks_que_cambiaron():
    """El filtro esconde lo hecho, pero un pick en estado 'cambio' es lo accionable.

    Si el filtro lo ocultara, el aviso de drift apuntaría a una fila invisible — que
    es peor que no filtrar.
    """
    html = _render_carga_fixture()
    i = html.index("const oculta")
    assert "estado(tr) !== 'cambio'" in html[i:i + 300]


# -------------------- template del modo carga (sin servidor) --------------------

def test_todo_id_que_usa_el_js_existe_en_el_markup():
    """El modo carga es un template con JS inline: el modo real de romperlo es
    referenciar un id que no existe (typo, o markup movido sin tocar el script).
    Un getElementById que devuelve null tira TypeError y mata refrescar() entero,
    que es lo único que pinta el progreso y los avisos.
    """
    import re
    s = _render_carga([_pick(111, ["2-1", "1-1"])])
    usados = set(re.findall(r"getElementById\(['\"]([\w-]+)['\"]\)", s))
    definidos = set(re.findall(r'id="([\w-]+)"', s))
    assert usados, "no encontré ningún getElementById — ¿cambió la forma del script?"
    assert not (usados - definidos), f"ids usados por JS y ausentes del markup: {sorted(usados - definidos)}"


def test_el_panel_de_cambios_es_plegable_y_acotado():
    """Con una fecha entera jugada el panel llega a 30 ítems: sin plegado ni tope de
    alto empujaba las tarjetas fuera de la pantalla y no había forma de cerrarlo.
    """
    s = _render_carga([_pick(111, ["2-1", "1-1"])])
    assert 'id="drift-toggle"' in s                  # se puede cerrar
    assert "max-h-[45vh]" in s and "overflow-y-auto" in s   # nunca tapa la página
    # y los cambios se separan por accionabilidad: lo cerrado no se puede corregir
    assert 'id="drift-list-cerrados"' in s
    assert "cerrado: tr.dataset.cerrado === '1'" in s

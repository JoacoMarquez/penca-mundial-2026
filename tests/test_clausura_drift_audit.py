"""Tests de la auditoría anti-drift (src.clausura.drift_audit)."""

import json
from datetime import datetime, timedelta, timezone

from src.clausura.drift_audit import (
    Cargado,
    diff_especiales,
    diff_picks,
    formatear_reporte,
    payload_con_goleadores_reales,
)

NOW = datetime(2026, 8, 8, 15, 0, tzinfo=timezone.utc)
NUMS = [899258848, 899258849, 899258850]


def _ev(eid: int, cierre: datetime) -> dict:
    return {
        "evento_id": eid,
        "local": "Liverpool",
        "visitante": "Albion",
        "fecha_n": 1,
        "cierre_pronostico_utc": cierre.isoformat(),
    }


def _esperado(eid: int, scores: list[tuple[int, int]]) -> dict:
    return {eid: {NUMS[k]: s for k, s in enumerate(scores)}}


def test_sin_drift_no_reporta_nada():
    evs = [_ev(1, NOW - timedelta(hours=1))]
    esp = _esperado(1, [(1, 0), (2, 1), (0, 0)])
    cargados = [Cargado(n, {1: esp[1][n]}) for n in NUMS]
    assert diff_picks(evs, esp, cargados, NOW) == []


def test_pick_distinto_se_detecta_con_estado_del_cierre():
    evs = [_ev(1, NOW + timedelta(hours=2)), _ev(2, NOW - timedelta(hours=2))]
    esp = {**_esperado(1, [(1, 0)]), **_esperado(2, [(2, 1)])}
    cargados = [Cargado(NUMS[0], {1: (0, 1), 2: (2, 0)})]
    dis = diff_picks(evs, esp, cargados, NOW)
    assert len(dis) == 2 and all(d.tipo == "distinto" for d in dis)
    abierto = next(d for d in dis if "1-0" in d.detalle)
    cerrado = next(d for d in dis if "2-1" in d.detalle)
    assert "AÚN CORREGIBLE" in abierto.detalle
    assert "cerrado" in cerrado.detalle


def test_faltante_solo_es_drift_despues_del_cierre():
    evs = [_ev(1, NOW + timedelta(hours=2)), _ev(2, NOW - timedelta(hours=2))]
    esp = {**_esperado(1, [(1, 0)]), **_esperado(2, [(2, 1)])}
    cargados = [Cargado(NUMS[0], {})]      # nada cargado
    dis = diff_picks(evs, esp, cargados, NOW)
    # el evento 1 (abierto) es territorio de carga_alert; solo el 2 (cerrado) es drift
    assert [d.tipo for d in dis] == ["sin_cargar_cerrado"]
    assert "planilla decía 2-1" in dis[0].detalle


def test_pick_cargado_sin_planilla_se_reporta():
    evs = [_ev(9, NOW + timedelta(hours=5))]
    cargados = [Cargado(NUMS[0], {9: (3, 3)})]
    dis = diff_picks(evs, {}, cargados, NOW)
    assert [d.tipo for d in dis] == ["sin_planilla"]


def test_claves_distinguen_valores_para_no_reavisar():
    evs = [_ev(1, NOW + timedelta(hours=2))]
    esp = _esperado(1, [(1, 0)])
    d1 = diff_picks(evs, esp, [Cargado(NUMS[0], {1: (0, 1)})], NOW)[0]
    d2 = diff_picks(evs, esp, [Cargado(NUMS[0], {1: (2, 2)})], NOW)[0]
    assert d1.clave != d2.clave      # corrigió a OTRO valor equivocado → re-avisa


def test_diff_especiales_ignora_lo_no_definido():
    esperados = {NUMS[0]: ("Peñarol", None), NUMS[1]: ("Nacional", "J. Pérez")}
    cargados = [
        Cargado(NUMS[0], {}, campeon="Peñarol", goleador="X. Otro"),   # goleador sin planilla: OK
        Cargado(NUMS[1], {}, campeon="Liverpool", goleador=None),      # campeón mal, goleador sin cargar
    ]
    dis = diff_especiales(esperados, cargados)
    assert len(dis) == 2 and all(d.tipo == "especial" for d in dis)
    assert any("«Liverpool»" in d.detalle and "«Nacional»" in d.detalle for d in dis)
    assert any("goleador" in d.detalle and "sin cargar" in d.detalle for d in dis)


def test_diff_especiales_gate_cerrado_no_es_drift():
    esperados = {NUMS[0]: ("Peñarol", "J. Pérez")}
    cargados = [Cargado(NUMS[0], {}, campeon=None, goleador=None,
                        especiales_visibles=False)]
    assert diff_especiales(esperados, cargados) == []


def test_adopcion_toma_goleadores_web_solo_donde_planilla_no_tiene():
    payload = {"especiales": {"por_participacion": [
        {"campeon": "Peñarol", "campeon_idx": 0, "goleador": None, "goleador_idx": -1},
        {"campeon": "Nacional", "campeon_idx": 1, "goleador": "M. Terans", "goleador_idx": 2},
        {"campeon": "Nacional", "campeon_idx": 1, "goleador": None, "goleador_idx": -1},
    ]}}
    cargados = [
        Cargado(NUMS[0], {}, goleador="L. Suárez"),
        Cargado(NUMS[1], {}, goleador="Otro"),          # planilla ya tiene → no pisa
        Cargado(NUMS[2], {}, goleador="C. Romero", especiales_visibles=True),
    ]
    nuevo, adoptados = payload_con_goleadores_reales(payload, cargados, NUMS)
    rows = nuevo["especiales"]["por_participacion"]
    assert rows[0]["goleador"] == "L. Suárez" and rows[0]["goleador_idx"] == -1
    assert rows[1]["goleador"] == "M. Terans"           # intacto
    assert rows[2]["goleador"] == "C. Romero"
    assert adoptados == [(NUMS[0], "L. Suárez"), (NUMS[2], "C. Romero")]
    assert "especiales_adoptados_utc" in nuevo
    # segunda pasada: ya no hay nada que adoptar → idempotente
    assert payload_con_goleadores_reales(nuevo, cargados, NUMS) is None


def test_adopcion_ignora_gate_cerrado_y_sin_goleador():
    payload = {"especiales": {"por_participacion": [
        {"campeon": None, "campeon_idx": -1, "goleador": None, "goleador_idx": -1}]}}
    cargados = [Cargado(NUMS[0], {}, goleador="X", especiales_visibles=False),
                Cargado(NUMS[1], {}, goleador=None)]
    assert payload_con_goleadores_reales(payload, cargados, NUMS) is None


def test_reporte_agrupa_por_participacion():
    evs = [_ev(1, NOW - timedelta(hours=1))]
    esp = _esperado(1, [(1, 0), (2, 1)])
    cargados = [Cargado(NUMS[0], {1: (0, 0)}), Cargado(NUMS[1], {})]
    txt = formatear_reporte(diff_picks(evs, esp, cargados, NOW))
    assert f"Participación {NUMS[0]}" in txt and f"Participación {NUMS[1]}" in txt
    assert "❌" in txt and "🕳️" in txt


# -------------------- penca gratuita (columna 1 + detección por picks) --------------------

def _eventos_gratuita():
    return [
        {"evento_id": 10, "local": "Liverpool", "visitante": "Albion", "fecha_n": 1,
         "cierre_pronostico_utc": "2026-08-07T21:45:00+00:00"},
        {"evento_id": 20, "local": "Defensor", "visitante": "Wanderers", "fecha_n": 1,
         "cierre_pronostico_utc": "2026-08-08T22:45:00+00:00"},
    ]


def _cfg_gratuita():
    return {"pencas": {"paga": {"id": 46}, "gratuita": {"id": 47}}}


AHORA = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)   # ambos cierres pasados


def test_gratuita_esperada_toma_la_columna_1(tmp_path, monkeypatch):
    import src.clausura.picks as picks_mod
    from src.clausura.drift_audit import gratuita_esperada
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)
    d = tmp_path / "fecha_01"
    d.mkdir()
    (d / "v1_20260806T120000Z.json").write_text(json.dumps({"picks": [
        {"evento_id": 10, "scores": [[1, 0], [0, 0], [2, 1]]},
        {"evento_id": 20, "scores": [[2, 1], [1, 1], [0, 0]]},
    ]}), encoding="utf-8")

    esp = gratuita_esperada(_eventos_gratuita())
    assert esp == {10: (1, 0), 20: (2, 1)}     # columna 1, no las otras


def test_gratuita_sin_env_no_audita(monkeypatch):
    from src.clausura.drift_audit import auditar_gratuita
    monkeypatch.delenv("CLAUSURA_MI_PARTICIPACION_GRATUITA", raising=False)
    assert auditar_gratuita(_eventos_gratuita(), _cfg_gratuita(), AHORA) == []


def test_gratuita_detecta_pick_distinto(monkeypatch):
    import src.clausura.drift_audit as da
    monkeypatch.setenv("CLAUSURA_MI_PARTICIPACION_GRATUITA", "899258816")
    monkeypatch.setattr(da, "gratuita_esperada", lambda ev: {10: (1, 0), 20: (2, 1)})
    monkeypatch.setattr(da, "fetch_cargados", lambda pid, nums: [
        da.Cargado(numero=899258816, picks={10: (1, 0), 20: (3, 3)})])

    ds = da.auditar_gratuita(_eventos_gratuita(), _cfg_gratuita(), AHORA)
    assert len(ds) == 1
    assert ds[0].tipo == "gratuita" and "3-3" in ds[0].detalle and "2-1" in ds[0].detalle


def test_gratuita_todo_ok_no_reporta(monkeypatch):
    import src.clausura.drift_audit as da
    monkeypatch.setenv("CLAUSURA_MI_PARTICIPACION_GRATUITA", "899258816")
    monkeypatch.setattr(da, "gratuita_esperada", lambda ev: {10: (1, 0), 20: (2, 1)})
    monkeypatch.setattr(da, "fetch_cargados", lambda pid, nums: [
        da.Cargado(numero=899258816, picks={10: (1, 0), 20: (2, 1)})])
    assert da.auditar_gratuita(_eventos_gratuita(), _cfg_gratuita(), AHORA) == []


def test_gratuita_cero_coincidencias_dispara_deteccion_por_picks(monkeypatch):
    """Si ningún pick coincide, escanea el ranking y sugiere el número correcto."""
    import src.clausura.drift_audit as da
    eventos = _eventos_gratuita() + [
        {"evento_id": 30, "local": "Nacional", "visitante": "Cerro", "fecha_n": 1,
         "cierre_pronostico_utc": "2026-08-09T18:45:00+00:00"},
        {"evento_id": 40, "local": "Danubio", "visitante": "Racing", "fecha_n": 1,
         "cierre_pronostico_utc": "2026-08-09T18:45:00+00:00"},
    ]
    esperado = {10: (1, 0), 20: (2, 1), 30: (1, 1), 40: (0, 0)}
    monkeypatch.setenv("CLAUSURA_MI_PARTICIPACION_GRATUITA", "899258816")
    monkeypatch.setattr(da, "gratuita_esperada", lambda ev: esperado)
    monkeypatch.setattr(da, "fetch_cargados", lambda pid, nums: [
        da.Cargado(numero=899258816,
                   picks={10: (3, 3), 20: (3, 3), 30: (3, 3), 40: (3, 3)})])
    monkeypatch.setattr(da, "buscar_por_picks",
                        lambda pid, esp, **kw: [(899259999, 4, 4)])

    ds = da.auditar_gratuita(eventos, _cfg_gratuita(), AHORA)
    sospecha = [d for d in ds if "numero_sospechoso" in d.clave]
    assert len(sospecha) == 1
    assert "899259999" in sospecha[0].detalle


def test_formatear_reporte_soporta_el_tipo_gratuita():
    from src.clausura.drift_audit import Discrepancia, formatear_reporte
    txt = formatear_reporte([Discrepancia("gratuita", 899258816, "algo pasó", "k")])
    assert "🎁" in txt and "899258816" in txt


# -------------------- adopción de picks reales (partidos cerrados) --------------------

def test_adopcion_picks_reescribe_solo_cerrados_distintos():
    from src.clausura.drift_audit import payload_con_picks_reales
    payload = {"picks": [
        {"evento_id": 1, "scores": [[1, 0], [2, 1], [0, 0]]},   # cerrado
        {"evento_id": 2, "scores": [[1, 1], [1, 1], [1, 1]]},   # abierto: no se toca
    ]}
    cargados = [
        Cargado(NUMS[0], {1: (1, 0)}),                          # coincide
        Cargado(NUMS[1], {1: (3, 0)}),                          # difiere → se adopta
        Cargado(NUMS[2], {}),                                   # sin cargar → no se toca
    ]
    res = payload_con_picks_reales(payload, cargados, NUMS, ev_cerrados={1})
    assert res is not None
    nuevo, cambios = res
    assert cambios == [(1, NUMS[1], (2, 1), (3, 0))]
    assert nuevo["picks"][0]["scores"] == [[1, 0], [3, 0], [0, 0]]
    assert nuevo["picks"][1]["scores"] == [[1, 1], [1, 1], [1, 1]]
    assert "picks_adoptados_utc" in nuevo


def test_adopcion_picks_sin_desvios_devuelve_none():
    from src.clausura.drift_audit import payload_con_picks_reales
    payload = {"picks": [{"evento_id": 1, "scores": [[1, 0], [2, 1], [0, 0]]}]}
    cargados = [Cargado(n, {1: (int(s[0]), int(s[1]))})
                for n, s in zip(NUMS, [[1, 0], [2, 1], [0, 0]])]
    assert payload_con_picks_reales(payload, cargados, NUMS, {1}) is None
    # y el payload quedó intacto (no hay versión fantasma)
    assert payload["picks"][0]["scores"] == [[1, 0], [2, 1], [0, 0]]


def test_adoptar_picks_cerrados_versiona_y_reporta_causa_gate(tmp_path, monkeypatch):
    """El caso real: el rerun versionó picks que el gate descartó; el partido cerró
    con los picks VIEJOS en la web. La adopción realinea y dice la causa."""
    import src.clausura.picks as picks_mod
    from src.clausura.drift_audit import adoptar_picks_cerrados
    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)

    d = tmp_path / "fecha_01"
    d.mkdir(parents=True)
    (d / "v2_20260808T120000Z.json").write_text(json.dumps({
        "generado_utc": "2026-08-08T12:00:00+00:00",
        "picks": [{"evento_id": 1, "scores": [[9, 9], [2, 1], [0, 0]],
                   "cierre_pronostico_utc": (NOW - timedelta(hours=2)).isoformat()}],
        "veredicto_cambio": {"avisar": False, "delta": -456.0},
    }), encoding="utf-8")

    eventos = [_ev(1, NOW - timedelta(hours=2))]
    cargados = [Cargado(NUMS[0], {1: (1, 0)}),                  # la web tiene lo viejo
                Cargado(NUMS[1], {1: (2, 1)}),
                Cargado(NUMS[2], {1: (0, 0)})]
    aviso = adoptar_picks_cerrados(cargados, NUMS, eventos, NOW)
    assert aviso is not None and "gate del rerun descartó" in aviso

    from src.utils.versions import latest_version
    nuevo = json.loads(latest_version(d.glob("v*_*.json")).read_text(encoding="utf-8"))
    assert nuevo["picks"][0]["scores"][0] == [1, 0]             # la web mandó
    assert "picks_adoptados_utc" in nuevo

    # idempotente: la segunda corrida no versiona de nuevo
    assert adoptar_picks_cerrados(cargados, NUMS, eventos, NOW) is None


# -------------------- poda del estado --------------------

def test_estado_conserva_la_fecha_del_primer_aviso_y_poda_lo_viejo(tmp_path, monkeypatch):
    import src.clausura.drift_audit as da
    monkeypatch.setattr(da, "STATE_PATH", tmp_path / "drift.json")

    da.save_state({"a", "b"}, now=NOW)
    # 'a' persiste con su fecha original aunque se re-guarde más tarde
    tarde = NOW + timedelta(days=10)
    da.save_state({"a", "c"}, now=tarde)
    guardado = json.loads((tmp_path / "drift.json").read_text())
    assert guardado["a"] == NOW.isoformat()          # no se pisó
    assert guardado["c"] == tarde.isoformat()
    assert "b" not in guardado                        # ya no estaba en el set

    # dentro del TTL sigue viva; pasado el TTL se poda
    assert da.load_state(now=tarde) == {"a", "c"}
    muy_tarde = NOW + timedelta(days=da.STATE_TTL_DIAS + 5)
    assert da.load_state(now=muy_tarde) == {"c"}


def test_estado_en_formato_viejo_lista_se_lee_igual(tmp_path, monkeypatch):
    """Migración: el archivo del VPS es una lista sin fechas — no se puede perder
    ninguna clave o el audit re-avisaría todo lo ya avisado."""
    import src.clausura.drift_audit as da
    monkeypatch.setattr(da, "STATE_PATH", tmp_path / "drift.json")
    (tmp_path / "drift.json").write_text(json.dumps(["vieja:1", "vieja:2"]))

    assert da.load_state(now=NOW) == {"vieja:1", "vieja:2"}
    da.save_state({"vieja:1", "vieja:2"}, now=NOW)    # se migra a dict con fecha
    assert json.loads((tmp_path / "drift.json").read_text()) == {
        "vieja:1": NOW.isoformat(), "vieja:2": NOW.isoformat()}

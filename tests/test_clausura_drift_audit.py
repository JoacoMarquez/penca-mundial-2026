"""Tests de la auditoría anti-drift (src.clausura.drift_audit)."""

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

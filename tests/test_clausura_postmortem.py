"""Tests del postmortem por fecha (src.clausura.postmortem)."""

import json
from src.clausura.postmortem import _percentil, compute_stats, formatear_postmortem

NUMS = [899258848, 899258849, 899258850]


def _eventos():
    return [
        {"evento_id": 1, "local": "Liverpool", "visitante": "Albion",
         "fecha_n": 1, "preferencial": False},
        {"evento_id": 2, "local": "Peñarol", "visitante": "Nacional",
         "fecha_n": 1, "preferencial": True},
    ]


RESULTADOS = {1: (2, 1), 2: (0, 0)}


def _picks():
    return {
        NUMS[0]: {1: (2, 1), 2: (0, 0)},   # todo exacto: 8 + 16 = 24
        NUMS[1]: {1: (1, 0), 2: (1, 1)},   # ganador+dif (3) ; empate+dif (3)x2=6 → 9
        NUMS[2]: {1: (0, 2), 2: (2, 0)},   # nada ; goles visita 0 (1pt x2) → 2
    }


def _pool():
    # dos rivales con picks, uno nuestro (debe excluirse), uno sin picks
    return [
        {"numero": 111, "picks": {"1": [2, 1], "2": [1, 0]}},   # 8 + visita-0 x2 = 10
        {"numero": 222, "picks": {"1": [0, 0], "2": [0, 0]}},   # 0 + exacto x2 = 16
        {"numero": NUMS[0], "picks": {"1": [9, 9], "2": [9, 9]}},  # mío: fuera
        {"numero": 333, "picks": {}},
    ]


def _stats():
    e_pts = {1: {NUMS[0]: 3.2, NUMS[1]: 3.0, NUMS[2]: 2.1},
             2: {NUMS[0]: 6.0, NUMS[1]: 5.5, NUMS[2]: 4.0}}
    return compute_stats(1, _eventos(), RESULTADOS, _picks(), e_pts,
                         _pool(), set(NUMS))


def test_puntos_con_kernel_real_y_estrella_doble():
    st = _stats()
    assert st.puntos[NUMS[0]] == 24 and st.exactos[NUMS[0]] == 2
    assert st.puntos[NUMS[1]] == 9 and st.exactos[NUMS[1]] == 0
    assert st.puntos[NUMS[2]] == 2      # goles del visitante pagan aunque erres ganador


def test_esperados_suman_e_pts_de_la_planilla():
    st = _stats()
    assert st.esperados[NUMS[0]] == 9.2
    assert st.esperados[NUMS[1]] == 8.5


def test_pool_excluye_mis_numeros_y_computa_exactos():
    st = _stats()
    assert sorted(st.pool_puntos) == [10, 16]       # el mío y el vacío quedan fuera
    assert st.pool_exactos_por_evento[1] == 0.5     # rival 111 clavó el 2-1
    assert st.pool_exactos_por_evento[2] == 0.5     # rival 222 clavó el 0-0


def test_percentil_estricto():
    assert _percentil(10, [1, 5, 10, 20]) == 0.5
    assert _percentil(0, []) == 0.0


def test_reporte_menciona_partidos_pool_y_premio():
    st = _stats()
    txt = formatear_postmortem(st, premio_fecha=10000)
    assert "Fecha 1" in txt and "⭐" in txt
    assert "2-1" in txt and "0-0" in txt
    assert "2 rivales" in txt and "mediana" in txt
    # 24 pts ≥ máx del pool (16) y nadie arriba → ganó la fecha
    assert "ganó la fecha" in txt and "10,000" in txt
    assert "(E 9)" in txt      # esperado junto al real


def test_reporte_sin_snapshot_no_rompe():
    st = compute_stats(1, _eventos(), RESULTADOS, _picks(), {}, [], set(NUMS))
    txt = formatear_postmortem(st)
    assert "Sin snapshot" in txt


# -------------------- fecha incompleta y e_pts pre-partido (2026-08-10) --------------------

def test_resultados_de_fecha_no_se_traba_con_un_suspendido(monkeypatch):
    """Un partido suspendido bloqueaba el postmortem para siempre.

    La Fecha 1 tuvo Torque–Peñarol suspendido: exigir la fecha completa dejó el
    análisis de los otros siete sin correr nunca.
    """
    import src.clausura.postmortem as pm

    class _Api:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def _get(self, _):
            return [
                {"id": 1, "resultado": {"golesEquipoLocal": 1, "golesEquipoVisitante": 0}},
                {"id": 2, "resultado": {"golesEquipoLocal": 0, "golesEquipoVisitante": 1}},
                {"id": 3, "resultado": {}},                      # suspendido
            ]

    monkeypatch.setattr(pm, "PencaApiClient", _Api)
    cfg = {"fechas": {"Fecha 1": {"fecha_id": 280}}}

    res, faltan = pm.resultados_de_fecha(cfg, 1, min_jugados=2)
    assert res == {1: (1, 0), 2: (0, 1)} and faltan == 1
    # y con el umbral por encima de lo jugado, no alcanza
    assert pm.resultados_de_fecha(cfg, 1, min_jugados=3) is None


def test_e_pts_sale_de_la_planilla_PRE_partido(tmp_path, monkeypatch):
    """Post-partido la grilla es delta y el E[pts] guardado ES el puntaje real.

    Si el postmortem lo tomara de la última versión, "esperado vs real" daría
    identidad perfecta y no mediría nada.
    """
    import src.clausura.picks as picks_mod
    import src.clausura.postmortem as pm

    monkeypatch.setattr(picks_mod, "PRED_DIR", tmp_path)
    d = tmp_path / "fecha_01"; d.mkdir()
    cierre = "2026-08-09T13:45:00+00:00"

    # v1: generada ANTES del cierre → su e_pts es la expectativa real
    (d / "v1_20260809T100000Z.json").write_text(json.dumps({
        "generado_utc": "2026-08-09T10:00:00+00:00",
        "picks": [{"evento_id": 10, "cierre_pronostico_utc": cierre,
                   "scores": [[1, 0], [1, 1]], "e_pts": [2.36, 1.98]}],
    }))
    # v2: generada DESPUÉS → e_pts contra la delta = el puntaje obtenido
    (d / "v2_20260809T160000Z.json").write_text(json.dumps({
        "generado_utc": "2026-08-09T16:00:00+00:00",
        "picks": [{"evento_id": 10, "cierre_pronostico_utc": cierre,
                   "scores": [[1, 0], [1, 1]], "e_pts": [0.0, 1.0]}],
    }))

    picks, e_pts = pm.picks_de_planilla(1, [111, 222])
    assert picks[111][10] == (1, 0)          # los PICKS sí salen de la última
    assert e_pts[10] == {111: 2.36, 222: 1.98}   # el E[pts], de la PRE-partido

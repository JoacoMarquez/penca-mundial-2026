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


# -------------------- no congelar postmortems parciales --------------------

def test_rehace_el_postmortem_si_aparecieron_mas_resultados(tmp_path, monkeypatch):
    """El de la Fecha 1 se escribió con 6 de 8 y no se iba a rehacer nunca.

    MIN_JUGADOS=6 destrabó el módulo (un partido suspendido ya no lo bloquea) pero
    hizo que dispare apenas hay seis y congele datos parciales. La herramienta que
    existe para aprender de lo que pasó quedaba equivocada para siempre.
    """
    import json
    from src.clausura import postmortem as pm

    cfg = {"fechas": {"Fecha 1": {"fecha_id": 280}}}
    monkeypatch.setattr(pm, "pm_path", lambda n: tmp_path / f"fecha_{n:02d}.json")

    def con(k):
        return lambda cfg, n, min_jugados: ({i: (1, 0) for i in range(k)}, 8 - k)

    # sin archivo previo → hay que analizarla
    monkeypatch.setattr(pm, "resultados_de_fecha", con(6))
    assert pm.fecha_a_analizar(cfg) == 1

    # escrita con 6 y siguen siendo 6 → no se toca
    (tmp_path / "fecha_01.json").write_text(
        json.dumps({"resultados": {str(i): [1, 0] for i in range(6)}}), encoding="utf-8")
    assert pm.fecha_a_analizar(cfg) is None

    # aparecieron los otros dos → se rehace
    monkeypatch.setattr(pm, "resultados_de_fecha", con(8))
    assert pm.fecha_a_analizar(cfg) == 1


def test_no_rehace_si_el_archivo_esta_al_dia(tmp_path, monkeypatch):
    """Rehacer de más también es malo: la corrida cuesta y pisaría el análisis bueno."""
    import json
    from src.clausura import postmortem as pm

    cfg = {"fechas": {"Fecha 1": {"fecha_id": 280}}}
    monkeypatch.setattr(pm, "pm_path", lambda n: tmp_path / f"fecha_{n:02d}.json")
    monkeypatch.setattr(pm, "resultados_de_fecha",
                        lambda cfg, n, min_jugados: ({i: (1, 0) for i in range(8)}, 0))
    (tmp_path / "fecha_01.json").write_text(
        json.dumps({"resultados": {str(i): [1, 0] for i in range(8)}}), encoding="utf-8")
    assert pm.fecha_a_analizar(cfg) is None


# -------------------- auditoría 13/8: trofeo, correcciones, tripwire --------------------

def test_trofeo_solo_con_fecha_liquidada():
    """Con un suspendido pendiente el premio no está definido (Arts. 9 y 14).

    La Fecha 1 tuvo exactamente este estado (Torque–Peñarol suspendido) y el
    reporte cantaba "ganaste la fecha" igual.
    """
    st = _stats()
    con_faltante = formatear_postmortem(st, premio_fecha=10000, faltan=1)
    assert "ganó la fecha" not in con_faltante
    assert "falta 1 partido" in con_faltante
    liquidada = formatear_postmortem(st, premio_fecha=10000, faltan=0)
    assert "ganó la fecha" in liquidada


def test_rehace_si_un_resultado_fue_CORREGIDO(tmp_path, monkeypatch):
    """Misma cantidad de resultados pero un marcador distinto → se rehace.

    Comparar solo conteos dejaba pasar una corrección del proveedor: el
    postmortem quedaba escrito con el resultado viejo para siempre.
    """
    from src.clausura import postmortem as pm

    cfg = {"fechas": {"Fecha 1": {"fecha_id": 280}}}
    monkeypatch.setattr(pm, "pm_path", lambda n: tmp_path / f"fecha_{n:02d}.json")
    monkeypatch.setattr(pm, "resultados_de_fecha",
                        lambda cfg, n, min_jugados: ({1: (1, 0), 2: (2, 2)}, 0))
    (tmp_path / "fecha_01.json").write_text(
        json.dumps({"resultados": {"1": [1, 0], "2": [2, 1]}}), encoding="utf-8")
    assert pm.fecha_a_analizar(cfg) == 1


def test_no_rehace_si_un_resultado_DESAPARECE(tmp_path, monkeypatch):
    """Un glitch transitorio del API (resultado que se esfuma) no pisa un
    postmortem bueno con uno más pobre."""
    from src.clausura import postmortem as pm

    cfg = {"fechas": {"Fecha 1": {"fecha_id": 280}}}
    monkeypatch.setattr(pm, "pm_path", lambda n: tmp_path / f"fecha_{n:02d}.json")
    monkeypatch.setattr(pm, "resultados_de_fecha",
                        lambda cfg, n, min_jugados: ({1: (1, 0)}, 1))
    (tmp_path / "fecha_01.json").write_text(
        json.dumps({"resultados": {"1": [1, 0], "2": [2, 1]}}), encoding="utf-8")
    assert pm.fecha_a_analizar(cfg) is None


def test_resultados_de_fecha_descarta_marcador_parcial(monkeypatch):
    """Un partido EN JUEGO con goles cargados cuenta como faltante, no como
    resultado — la guardia de `estado` (resultado_finalizado)."""
    import src.clausura.postmortem as pm

    class _Api:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def _get(self, _):
            return [
                {"id": 1, "estado": "FINALIZADO",
                 "resultado": {"golesEquipoLocal": 1, "golesEquipoVisitante": 0}},
                {"id": 2, "estado": "EN_JUEGO",
                 "resultado": {"golesEquipoLocal": 2, "golesEquipoVisitante": 0}},
            ]

    monkeypatch.setattr(pm, "PencaApiClient", _Api)
    cfg = {"fechas": {"Fecha 1": {"fecha_id": 280}}}
    res, faltan = pm.resultados_de_fecha(cfg, 1, min_jugados=1)
    assert res == {1: (1, 0)} and faltan == 1


def test_tripwire_puntos_vs_web():
    """Los rivales están anclados al ranking por el residuo; nuestros puntos
    calculados nunca se contrastaban. Cualquier divergencia de liquidación
    sesgaba solo nuestro lado del simulador, sin síntoma."""
    from src.clausura.postmortem import comparar_puntos_publicados

    difs = comparar_puntos_publicados(
        {899258848: 24, 899258849: 9}, {899258848: 24, 899258849: 12})
    assert len(difs) == 1 and "calculado 9 vs web 12" in difs[0]
    # sin dato publicado no hay falsa alarma
    assert comparar_puntos_publicados({899258848: 24}, {}) == []


def test_totales_calculados_suman_postmortems_previos(tmp_path, monkeypatch):
    from src.clausura import postmortem as pm

    monkeypatch.setattr(pm, "pm_path", lambda n: tmp_path / f"fecha_{n:02d}.json")
    (tmp_path / "fecha_01.json").write_text(
        json.dumps({"puntos": {"899258848": 10, "899258849": 7}}), encoding="utf-8")
    tot = pm._totales_calculados(2, {899258848: 5, 899258849: 3})
    assert tot == {899258848: 15, 899258849: 10}
    # sin el postmortem de una fecha previa, no se puede verificar → None
    assert pm._totales_calculados(3, {899258848: 5}) is None

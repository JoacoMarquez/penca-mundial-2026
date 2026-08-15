"""Tests de la vista del pool (resumen del ranking + histórico en disco)."""

import json
from datetime import datetime, timedelta, timezone

from src.clausura.api import RankingRow, parse_ranking_page
from src.clausura.pool_view import (
    entrada_historica,
    historia_por_fecha,
    leer_historia,
    registrar_historia,
    resumen_pool,
)


def row(numero, puntos, exactos=0, pts_fecha=0, escalon=1, pid=None):
    return RankingRow(
        participacion_id=pid if pid is not None else numero,
        numero_participacion=numero,
        puntos_totales=puntos,
        puntos_por_fecha=pts_fecha,
        posicion_general=escalon,
        cant_resultados_exactos=exactos,
    )


# 3 nuestras (100,101,102) entre 7 participaciones
ROWS = [
    row(1, 20, exactos=3, pts_fecha=8, escalon=1),
    row(100, 20, exactos=2, pts_fecha=8, escalon=1),      # nuestra, en el tope
    row(2, 18, exactos=2, pts_fecha=6, escalon=2),
    row(101, 15, exactos=1, pts_fecha=4, escalon=3),      # nuestra
    row(3, 15, exactos=0, pts_fecha=4, escalon=3),
    row(102, 9, exactos=0, pts_fecha=2, escalon=4),       # nuestra
    row(4, 3, exactos=0, pts_fecha=0, escalon=5),
]
MIOS = {100, 101, 102}


def test_puesto_real_no_es_el_escalon():
    """posicionGeneral es el escalón de puntaje, no el puesto: el resumen calcula
    el puesto de competencia (1 + cuántas tienen más puntos)."""
    r = resumen_pool(ROWS, MIOS)
    por_numero = {m["numero"]: m for m in r["mias"]}
    assert por_numero[100]["puesto"] == 1          # empatada arriba
    assert por_numero[101]["puesto"] == 4          # 20, 20, 18 por delante
    assert por_numero[101]["escalon"] == 3         # lo que muestra la web
    assert por_numero[102]["puesto"] == 6


def test_premio_se_reparte_entre_empatados():
    r = resumen_pool(ROWS, MIOS, premio_penca=350_000.0)
    assert r["lider"]["puntos"] == 20
    assert r["lider"]["empatados"] == 2
    assert r["lider"]["mias_en_tope"] == 1
    assert r["lider"]["premio_por_cabeza"] == 175_000.0
    assert r["lider"]["cobro_hoy"] == 175_000.0     # 350k * 1/2


def test_cobro_hoy_barre_el_tope():
    """Dos nuestras solas en la cima cobran el premio entero."""
    rows = [row(100, 20), row(101, 20), row(1, 10)]
    r = resumen_pool(rows, {100, 101}, premio_penca=350_000.0)
    assert r["lider"]["cobro_hoy"] == 350_000.0


def test_rivales_adelante_ignora_las_nuestras():
    r = resumen_pool(ROWS, MIOS)
    mejor = r["mejor"]
    assert mejor["numero"] == 100
    assert mejor["rivales_adelante"] == 0
    assert mejor["empatados_rivales"] == 1
    peor = r["peor"]
    assert peor["numero"] == 102
    assert peor["rivales_adelante"] == 3            # 1, 2 y 3
    assert peor["gap_lider"] == 11


def test_amenazas_y_distribucion():
    r = resumen_pool(ROWS, MIOS)
    amen = {a["delta"]: a["rivales"] for a in r["amenazas"]}
    assert amen[0] == 1                             # solo el rival 1 está en 20
    assert amen[2] == 2                             # + el de 18
    assert amen[5] == 3                             # + el rival de 15 (el otro es nuestro)
    dist = {d["puntos"]: d for d in r["distribucion"]}
    assert dist[20]["count"] == 2 and dist[20]["mias"] == 1
    assert dist[15]["mias"] == 1
    assert r["distribucion"][0]["puntos"] == 20     # ordenado desc


def test_premio_de_fecha():
    r = resumen_pool(ROWS, MIOS, premio_fecha=10_000.0)
    f = r["fecha"]
    assert f["activo"] and f["lider"] == 8 and f["empatados"] == 2
    assert f["cobro_hoy"] == 5_000.0

    # fecha sin liquidar: el API manda 0 para todos → no hay tope ni cobro
    sin_liquidar = [row(n, 5, pts_fecha=0) for n in (1, 2, 100)]
    f0 = resumen_pool(sin_liquidar, MIOS, premio_fecha=10_000.0)["fecha"]
    assert f0["activo"] is False
    assert f0["cobro_hoy"] == 0.0 and f0["empatados"] == 0
    assert f0["nuestra_mejor"] is None


def test_agregados_y_pool_vacio():
    r = resumen_pool(ROWS, MIOS)
    assert r["total"] == 7 and r["mias_encontradas"] == 3
    assert r["puntos"]["mediana"] == 15
    assert r["puntos"]["nuestra_mejor"] == 20 and r["puntos"]["nuestra_peor"] == 9
    assert r["exactos"]["pool_max"] == 3 and r["exactos"]["nuestra_mejor"] == 2
    assert resumen_pool([], MIOS)["ok"] is False


def test_exactos_sin_liquidar_se_marca():
    """0 exactos con puntos ya cargados = el contador del API no liquidó todavía."""
    r = resumen_pool([row(1, 8), row(2, 1), row(100, 8)], MIOS)
    assert r["exactos"]["sin_liquidar"] is True
    # con exactos reportados, o con el pool en cero, no hay nada raro que marcar
    assert resumen_pool([row(1, 8, exactos=1)], MIOS)["exactos"]["sin_liquidar"] is False
    assert resumen_pool([row(1, 0), row(2, 0)], MIOS)["exactos"]["sin_liquidar"] is False


def test_sin_participaciones_propias():
    r = resumen_pool(ROWS, set())
    assert r["ok"] and r["mejor"] is None and r["mias_encontradas"] == 0
    assert r["lider"]["cobro_hoy"] == 0.0


# -------------------- histórico --------------------

def test_registrar_historia_throttlea_pero_deja_pasar_los_cambios(tmp_path):
    p = tmp_path / "ranking.jsonl"
    r = resumen_pool(ROWS, MIOS)
    assert registrar_historia(r, path=p) is True
    assert registrar_historia(r, path=p) is False        # sin cambios, dentro del gap

    movido = resumen_pool([row(1, 25)] + ROWS[1:], MIOS)  # se movió el líder
    assert registrar_historia(movido, path=p) is True

    lineas = leer_historia(p)
    assert [l["lider"] for l in lineas] == [20, 25]
    assert lineas[0]["mejor_puesto"] == 1 and lineas[1]["mejor_puesto"] == 2


def test_registrar_historia_por_tiempo(tmp_path):
    p = tmp_path / "ranking.jsonl"
    r = resumen_pool(ROWS, MIOS)
    vieja = entrada_historica(
        r, ts=datetime.now(timezone.utc) - timedelta(hours=5))
    p.write_text(json.dumps(vieja) + "\n", encoding="utf-8")
    assert registrar_historia(r, path=p, min_gap_h=3.0) is True
    assert len(leer_historia(p)) == 2


def test_leer_historia_aguanta_linea_rota(tmp_path):
    p = tmp_path / "ranking.jsonl"
    p.write_text('{"lider": 5}\n{"lider": 6\n{"lider": 7}\n', encoding="utf-8")
    assert [l["lider"] for l in leer_historia(p)] == [5, 7]


def test_registrar_historia_no_explota_sin_permisos(tmp_path):
    ruta_imposible = tmp_path / "archivo.txt" / "ranking.jsonl"
    (tmp_path / "archivo.txt").write_text("no soy un directorio", encoding="utf-8")
    assert registrar_historia(resumen_pool(ROWS, MIOS), path=ruta_imposible) is False


def test_historia_por_fecha(tmp_path):
    (tmp_path / "fecha_01.json").write_text(json.dumps({
        "fecha": 1,
        "puntos": {"100": 12, "101": 9},
        "exactos": {"100": 1, "101": 0},
        "pool_puntos": [3, 6, 9, 12, 15],
    }), encoding="utf-8")
    (tmp_path / "fecha_02.json").write_text(json.dumps({
        "fecha": 2,
        "puntos": {"100": 20},
        "exactos": {"100": 2},
        "pool_puntos": [5, 10, 15],
    }), encoding="utf-8")
    h = historia_por_fecha(tmp_path)
    assert [f["fecha"] for f in h] == [1, 2]
    assert h[0]["nuestra_mejor"] == 12 and h[0]["pool_max"] == 15
    assert h[0]["ganamos_fecha"] is False and h[0]["percentil"] == 60.0
    assert h[1]["ganamos_fecha"] is True and h[1]["nuestros_exactos"] == 2
    assert historia_por_fecha(tmp_path / "no-existe") == []


def test_ranking_page_size_llega_al_request():
    """El parser no cambió; lo que importa es que ranking() pida size (1 request)."""
    import httpx

    from src.clausura.api import RANKING_PAGE_SIZE, PencaApiClient

    vistos = []

    def handler(request: httpx.Request) -> httpx.Response:
        vistos.append(dict(request.url.params))
        return httpx.Response(200, json={
            "content": [{"participacionId": 1, "numeroParticipacion": 900,
                         "puntosTotales": 8, "puntosPorFecha": 0,
                         "posicionGeneral": 1, "cantResultadosExactos": 0}],
            "totalElements": 1, "totalPages": 1,
        })

    api = PencaApiClient()
    api._client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    rows = api.ranking(46)
    assert len(rows) == 1 and len(vistos) == 1
    assert vistos[0] == {"page": "1", "size": str(RANKING_PAGE_SIZE)}


def test_parse_ranking_page_sigue_intacto():
    rows, total, paginas = parse_ranking_page({
        "content": [{"participacionId": 68223, "numeroParticipacion": 899258497,
                     "puntosTotales": 8, "puntosPorFecha": 0,
                     "posicionGeneral": 1, "cantResultadosExactos": 0}],
        "totalElements": 692, "totalPages": 1,
    })
    assert total == 692 and paginas == 1
    assert rows[0].numero_participacion == 899258497


# -------------------- movimiento por partido --------------------

from src.clausura.pool_view import liquidables_en, movimientos_por_partido  # noqa: E402

EVENTOS_MOV = [
    {"evento_id": 1, "local": "Cerro", "visitante": "Albion",
     "inicio_utc": "2026-08-15T13:00:00+00:00",
     "cierre_pronostico_utc": "2026-08-15T12:45:00+00:00"},
    {"evento_id": 2, "local": "Juventud", "visitante": "Torque",
     "inicio_utc": "2026-08-15T16:00:00+00:00",
     "cierre_pronostico_utc": "2026-08-15T15:45:00+00:00"},
]


def _ts(h, m=0):
    return datetime(2026, 8, 15, h, m, tzinfo=timezone.utc)


def test_liquidables_exige_cierre_y_105_min_de_juego():
    # 15:00: Cerro cerró y arrancó hace 2h → liquidable. Juventud ni cerró.
    assert liquidables_en(EVENTOS_MOV, _ts(15)) == [1]
    # 16:00: Juventud cerró pero ARRANCA recién ahora — no pudo liquidar.
    # Sin el filtro de 105', el sábado en cascada se atribuiría doble.
    assert liquidables_en(EVENTOS_MOV, _ts(16)) == [1]
    assert liquidables_en(EVENTOS_MOV, _ts(18)) == [1, 2]


def _foto(ts, mias, liq):
    return {"ts": ts.isoformat(), "mias": mias, "liq": liq}


def test_movimientos_atribuye_al_partido_nuevo():
    historia = [
        _foto(_ts(12), [{"n": 100, "p": 20, "pu": 50}, {"n": 101, "p": 18, "pu": 80}], []),
        _foto(_ts(15), [{"n": 100, "p": 28, "pu": 30}, {"n": 101, "p": 19, "pu": 95}], [1]),
    ]
    movs = movimientos_por_partido(historia, EVENTOS_MOV)
    assert len(movs) == 1
    assert movs[0]["etiqueta"] == "Cerro vs Albion"
    assert movs[0]["subieron"] == 1 and movs[0]["bajaron"] == 1
    m100 = next(m for m in movs[0]["movimientos"] if m["numero"] == 100)
    assert (m100["puesto_antes"], m100["puesto_despues"]) == (50, 30)
    assert m100["delta_puesto"] == 20 and m100["puntos_ganados"] == 8
    m101 = next(m for m in movs[0]["movimientos"] if m["numero"] == 101)
    assert m101["delta_puesto"] == -15


def test_movimientos_reacomodo_sin_puntos_nuestros():
    """Puestos que se mueven sin puntos nuestros = puntos ajenos, no un partido."""
    historia = [
        _foto(_ts(12), [{"n": 100, "p": 20, "pu": 50}], [1]),
        _foto(_ts(15), [{"n": 100, "p": 20, "pu": 55}], [1]),
    ]
    movs = movimientos_por_partido(historia, EVENTOS_MOV)
    assert len(movs) == 1
    assert movs[0]["partidos"] == [] and "reacomodo" in movs[0]["etiqueta"]


def test_movimientos_dos_partidos_en_la_misma_ventana_salen_juntos():
    historia = [
        _foto(_ts(12), [{"n": 100, "p": 20, "pu": 50}], []),
        _foto(_ts(18), [{"n": 100, "p": 25, "pu": 40}], [1, 2]),
    ]
    movs = movimientos_por_partido(historia, EVENTOS_MOV)
    assert movs[0]["etiqueta"] == "Cerro vs Albion + Juventud vs Torque"


def test_movimientos_ignora_fotos_viejas_sin_mias():
    historia = [
        {"ts": _ts(10).isoformat(), "lider": 20},            # línea vieja, pre-feature
        _foto(_ts(12), [{"n": 100, "p": 20, "pu": 50}], []),
        _foto(_ts(15), [{"n": 100, "p": 28, "pu": 30}], [1]),
    ]
    assert len(movimientos_por_partido(historia, EVENTOS_MOV)) == 1


def test_foto_guarda_mias_y_liq_y_dispara_por_movimiento_del_medio(tmp_path):
    p = tmp_path / "ranking.jsonl"
    r = resumen_pool(ROWS, MIOS)
    assert registrar_historia(r, path=p, liquidables=[2, 1]) is True
    linea = leer_historia(p)[-1]
    assert linea["liq"] == [1, 2]
    assert {m["n"] for m in linea["mias"]} == set(MIOS)

    # Se mueve solo una nuestra del MEDIO de la tabla (ni líder ni la mejor):
    # sin el chequeo de `mias` esta liquidación no dejaría foto.
    rows2 = [ROWS[0], ROWS[1], ROWS[2], row(101, 16), ROWS[4], ROWS[5], ROWS[6]]
    r2 = resumen_pool(rows2, MIOS)
    assert registrar_historia(r2, path=p, liquidables=[1, 2]) is True


def test_firma_ranking_cambia_si_el_tope_se_mueve():
    from src.clausura.gate_watch import firma_ranking
    a = firma_ranking(ROWS)
    assert a == firma_ranking(list(ROWS))
    assert a != firma_ranking([row(1, 25)] + ROWS[1:])

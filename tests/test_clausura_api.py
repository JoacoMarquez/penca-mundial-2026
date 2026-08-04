"""Tests de parseo del penca-api (fixtures = JSON real capturado el 2026-08-04)."""

from datetime import datetime, timezone

from src.clausura.api import parse_eventos, parse_pencas, parse_ranking_page

PENCAS_JSON = [
    {
        "id": 46, "nombre": "Torneo Clausura 2026", "precio": 400,
        "campeonatoId": 44, "estado": "ACTIVO",
        "premios": [
            {"id": 102, "nombre": "Ganador Penca Clausura 2026", "descripcion": "x",
             "orden": 1, "tipo": "PENCA", "monto": 350000.00},
            {"id": 103, "nombre": "Ganador Fecha Clausura 2026", "descripcion": "x",
             "orden": 2, "tipo": "FECHA", "monto": 10000.00},
        ],
    },
    {
        "id": 47, "nombre": "Torneo Clausura 2026 Gratuita", "precio": 0,
        "campeonatoId": 44, "estado": "ACTIVO", "premios": [],
    },
]

EVENTO_JSON = {
    "id": 2080,
    "locacion": "Alfredo Victor Viera",
    "fechaInicio": "07-08-2026 19:00:00",
    "estado": "PENDIENTE",
    "preferencial": False,
    "equipoLocal": {"id": 76, "nombre": "Liverpool", "abreviatura": "LIV", "deporteId": 1},
    "equipoVisitante": {"id": 221, "nombre": "Albion", "abreviatura": "ALB", "deporteId": 1},
    "grupo": {"id": 109, "nombre": "Torneo Clausura 2026"},
    "fechaCierrePronostico": "07-08-2026 18:45:00",
}

RANKING_PAGE_JSON = {
    "content": [
        {"participacionId": 68220, "numeroParticipacion": 899258494, "puntosTotales": 0,
         "puntosPorFecha": 0, "posicionGeneral": 1, "posicionFecha": -1,
         "cantResultadosExactos": 0},
    ],
    "totalElements": 151,
    "totalPages": 8,
}


def test_parse_pencas():
    pencas = parse_pencas(PENCAS_JSON)
    paga = next(p for p in pencas if p.id == 46)
    assert paga.precio == 400
    assert paga.campeonato_id == 44
    assert {pr.tipo for pr in paga.premios} == {"PENCA", "FECHA"}
    assert next(pr.monto for pr in paga.premios if pr.tipo == "PENCA") == 350000.0

    gratuita = next(p for p in pencas if p.id == 47)
    assert gratuita.precio == 0


def test_parse_eventos_convierte_hora_uy_a_utc():
    (ev,) = parse_eventos([EVENTO_JSON], fecha_id=280)
    assert ev.local == "Liverpool" and ev.visitante == "Albion"
    assert ev.preferencial is False
    # 19:00 UY (UTC-3) == 22:00 UTC
    assert ev.inicio_utc == datetime(2026, 8, 7, 22, 0, tzinfo=timezone.utc)
    assert ev.cierre_pronostico_utc == datetime(2026, 8, 7, 21, 45, tzinfo=timezone.utc)
    # el cierre es exactamente T-15min (Art. 4)
    assert (ev.inicio_utc - ev.cierre_pronostico_utc).total_seconds() == 15 * 60


def test_parse_eventos_ordena_por_inicio():
    e2 = dict(EVENTO_JSON, id=2085, fechaInicio="08-08-2026 15:00:00",
              fechaCierrePronostico="08-08-2026 14:45:00")
    eventos = parse_eventos([e2, EVENTO_JSON], fecha_id=280)
    assert [e.id for e in eventos] == [2080, 2085]


def test_parse_ranking_page():
    rows, total, pages = parse_ranking_page(RANKING_PAGE_JSON)
    assert total == 151 and pages == 8
    assert rows[0].participacion_id == 68220
    assert rows[0].cant_resultados_exactos == 0

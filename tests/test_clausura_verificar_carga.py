"""Tests de la verificación de carga (planilla vs lo cargado en la web, sin red)."""

from src.clausura.verificar_carga import DISTINTO, FALTA, OK, SIN_DATOS, comparar


def _planilla(n_part: int = 2) -> dict:
    return {
        "n_participaciones": n_part,
        "picks": [
            {"partido": "Peñarol vs Nacional", "evento_id": 111, "cerrado": False,
             "scores": [[2, 1], [1, 1]], "scores_fmt": ["2-1", "1-1"]},
            {"partido": "Danubio vs Progreso", "evento_id": 112, "cerrado": False,
             "scores": [[1, 0], [0, 0]], "scores_fmt": ["1-0", "0-0"]},
        ],
        "especiales": {"por_participacion": [
            {"campeon": "Peñarol", "goleador": "Arezo"},
            {"campeon": "Nacional", "goleador": None},
        ]},
    }


NUMS = [899258848, 899258849]


def test_todo_coincide():
    web = {899258848: {111: (2, 1), 112: (1, 0)},
           899258849: {111: (1, 1), 112: (0, 0)}}
    esp = {899258848: {"campeon": "Peñarol", "goleador": "Arezo"},
           899258849: {"campeon": "Nacional", "goleador": "Gómez"}}

    r = comparar(_planilla(), NUMS, web, esp)

    # 4 marcadores + campeón x2 + goleador de la part. 1 (el de la 2 no está asignado)
    assert r["resumen"][OK] == 7
    assert r["resumen"][DISTINTO] == 0
    assert r["resumen"][FALTA] == 0
    assert not r["sin_datos_fecha"] and not r["sin_datos_api"]


def test_marcador_tipeado_distinto_se_detecta():
    """El caso que motiva todo: cargaste 2-0 donde la planilla dice 2-1."""
    web = {899258848: {111: (2, 0), 112: (1, 0)},
           899258849: {111: (1, 1), 112: (0, 0)}}

    r = comparar(_planilla(), NUMS, web)

    malas = [f for f in r["filas"] if f["estado"] == DISTINTO]
    assert len(malas) == 1
    assert malas[0]["numero"] == 899258848
    assert malas[0]["planilla"] == "2-1" and malas[0]["web"] == "2-0"


def test_pick_sin_cargar():
    web = {899258848: {111: (2, 1)}, 899258849: {111: (1, 1), 112: (0, 0)}}

    r = comparar(_planilla(), NUMS, web)

    faltan = [f for f in r["filas"] if f["estado"] == FALTA]
    assert [f["evento_id"] for f in faltan] == [112]
    assert faltan[0]["planilla"] == "1-0"


def test_participacion_ausente_del_ranking_no_cuenta_como_falta():
    """Si no está la participación, no sabemos nada: sin_datos, no 'sin cargar'."""
    r = comparar(_planilla(), NUMS, {899258848: {111: (2, 1), 112: (1, 0)}})

    ausentes = [f for f in r["filas"] if f["estado"] == SIN_DATOS]
    assert [f["part"] for f in ausentes] == [1, 1]     # las dos filas de la part. 2
    assert r["resumen"][FALTA] == 0


def test_fecha_que_el_api_todavia_no_expone():
    """El API devuelve pronósticos de OTRA fecha: no es que falten, es que no se
    pueden verificar. Sin esta marca el panel diría '4 sin cargar' y mandaría a
    recargar picks que ya están puestos."""
    web = {899258848: {999: (1, 1)}, 899258849: {999: (0, 0)}}

    r = comparar(_planilla(), NUMS, web)

    assert r["sin_datos_fecha"] is True
    assert r["resumen"][FALTA] == 4


def test_api_mudo_no_es_sin_datos_de_la_fecha():
    r = comparar(_planilla(), NUMS, {899258848: {}, 899258849: {}})

    assert r["sin_datos_api"] is True
    assert r["sin_datos_fecha"] is False


def test_especial_distinto_y_no_asignado():
    esp = {899258848: {"campeon": "Nacional", "goleador": "Arezo"},
           899258849: {"campeon": "Nacional", "goleador": "Gómez"}}

    r = comparar(_planilla(), NUMS, {899258848: {}, 899258849: {}}, esp)

    campeones = [e for e in r["especiales"] if e["campo"] == "campeon"]
    assert campeones[0]["estado"] == DISTINTO and campeones[0]["web"] == "Nacional"
    assert campeones[1]["estado"] == OK
    # el goleador de la participación 2 no está en la planilla: no se verifica
    assert [e["part"] for e in r["especiales"] if e["campo"] == "goleador"] == [0]


def test_comparacion_de_nombres_tolera_espacios_y_mayusculas():
    esp = {899258848: {"campeon": " peñarol ", "goleador": "AREZO"}}

    r = comparar(_planilla(1), [NUMS[0]], {899258848: {}}, esp)

    assert all(e["estado"] == OK for e in r["especiales"])

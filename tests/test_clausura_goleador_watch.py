"""Tests del watcher de menús de especiales (src.clausura.goleador_watch)."""

from src.clausura.goleador_watch import formatear_campeon, formatear_goleador

NUMS = [899258848, 899258849, 899258850]


def test_campeon_agrupa_numeros_por_equipo():
    por_numero = {NUMS[0]: ("Peñarol", None), NUMS[1]: ("Nacional", None),
                  NUMS[2]: ("Nacional", None)}
    txt = formatear_campeon(16, por_numero)
    assert "16 opciones" in txt
    assert f"Peñarol</b> → {NUMS[0]}" in txt
    assert f"Nacional</b> → {NUMS[1]}, {NUMS[2]}" in txt


def test_campeon_sin_planilla_avisa_igual():
    txt = formatear_campeon(16, {})
    assert "sin planilla" in txt


def test_goleador_con_asignaciones_y_prior():
    por_numero = {NUMS[0]: (None, "L. Suárez"), NUMS[1]: (None, "M. Terans"),
                  NUMS[2]: (None, "L. Suárez")}
    top = [("L. Suárez", 0.31), ("M. Terans", 0.12)]
    txt = formatear_goleador(por_numero, top, regenerada=True)
    assert "L. Suárez 31%" in txt
    assert f"L. Suárez</b> → {NUMS[0]}, {NUMS[2]}" in txt
    assert f"M. Terans</b> → {NUMS[1]}" in txt
    assert "hasta el inicio" in txt


def test_goleador_sin_asignaciones_distingue_regeneracion():
    ok = formatear_goleador({}, [], regenerada=True)
    fallo = formatear_goleador({}, [], regenerada=False)
    assert "revisar logs" in ok
    assert "corré el pipeline a mano" in fallo

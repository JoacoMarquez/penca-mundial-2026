import numpy as np
import pytest

from scripts.backtest_goleador_actual import (
    OTROS, Goleador, Pareado, clasificar, inclinar, prior_desde_tabla,
)


def test_inclinar_multiplica_y_renormaliza():
    p = np.array([0.5, 0.3, 0.2])
    q = inclinar(p, 1, 1.5)
    assert q.sum() == pytest.approx(1.0)
    assert q[1] / q[0] == pytest.approx(1.5 * 0.3 / 0.5)
    assert q[2] / q[0] == pytest.approx(0.2 / 0.5)


def test_prior_manda_a_otros_a_quien_no_esta_en_el_menu():
    tabla = (Goleador("Fuera", 9, 7, None), Goleador("Dentro", 0, 7, None))
    p = prior_desde_tabla(tabla, ["Dentro", OTROS], n_sims=2_000)
    assert p.sum() == pytest.approx(1.0)
    assert p[1] > 0.99          # con 9 goles de ventaja y 8 fechas, cobra "Otros"


def test_prior_sin_otros_renormaliza_lo_que_queda():
    tabla = (Goleador("A", 3, 7, None), Goleador("B", 3, 7, None),
             Goleador("Fuera", 3, 7, None))
    p = prior_desde_tabla(tabla, ["A", "B"], n_sims=20_000)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx(0.5, abs=0.03)


def test_prior_reparte_empates():
    tabla = (Goleador("A", 5, 15, None), Goleador("B", 5, 15, None))  # nada por jugar
    p = prior_desde_tabla(tabla, ["A", "B"], n_sims=100)
    assert p == pytest.approx([0.5, 0.5])


def test_prior_rechaza_menu_vacio():
    with pytest.raises(ValueError):
        prior_desde_tabla((Goleador("A", 1, 7, None),), [])


def test_pareado_calcula_delta_y_se():
    r = Pareado.de([1.0, 2.0, 3.0], [2.0, 2.5, 4.5])
    assert r.delta == pytest.approx(1.0)
    assert r.se == pytest.approx(np.std([1.0, 0.5, 1.5], ddof=1) / np.sqrt(3))


def _p(delta, se=1.0):
    return Pareado(delta, se, 0.0, delta)


def test_clasificar_adopta_solo_con_senal_clara_y_sin_fragilidad():
    assert clasificar(_p(3), _p(0), {"arezo": _p(0)}) == "adoptar"
    assert clasificar(_p(1), _p(0), {"arezo": _p(0)}) == "inconcluso"
    assert clasificar(_p(3), _p(0), {"arezo": _p(-3)}) == "rechazado por fragilidad"
    assert clasificar(_p(-3), _p(0), {}) == "rechazado"
    # señal clara pero cae el premio general más de $2.000
    assert clasificar(_p(3), _p(-2_500, 100), {}) == "inconcluso"

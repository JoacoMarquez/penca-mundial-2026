import numpy as np
import pytest

from src.lub.scoring import (KERNEL, N_CLASES, banda, clase, expected_points,
                             marcador_de_clase, puntos)


def test_bandas_bordes():
    assert [banda(m) for m in (1, 5, 6, 10, 11, 15, 16, 20, 21, 60)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    with pytest.raises(ValueError):
        banda(0)


def test_clases():
    assert clase(80, 77) == 0           # local por 3
    assert clase(70, 81) == 7           # visitante por 11
    assert clase(100, 70) == 4          # local por 30


def test_puntos_reglamento():
    assert puntos((80, 75), (90, 86)) == 5       # ganador + banda 1-5
    assert puntos((80, 75), (90, 80)) == 2       # ganador, banda distinta
    assert puntos((80, 75), (70, 75)) == 0       # ganador errado: margen no cuenta
    assert puntos((80, 75), (80, 75)) == 15      # exacto suma sobre ganador+margen
    assert puntos((80, 75), (90, 86), preferencial=True) == 10
    assert puntos((80, 80), (80, 75)) == 0       # empate no es pronóstico válido


def test_kernel_consistente_con_puntos():
    assert KERNEL.shape == (N_CLASES, N_CLASES)
    assert KERNEL[0, 0] == 5 and KERNEL[0, 3] == 2 and KERNEL[0, 5] == 0
    probs = np.full(N_CLASES, 0.1)
    assert np.allclose(expected_points(probs), 0.5 * 2 + 0.1 * 3)


def test_marcador_de_clase_cae_en_la_clase():
    for c in range(N_CLASES):
        pl, pv = marcador_de_clase(c, 152.3)
        assert clase(pl, pv) == c
        assert abs(pl + pv - 152) <= 2

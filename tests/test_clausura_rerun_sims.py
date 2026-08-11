"""Rerun de cierre: sims heredadas, y guardia del menú de candidatos.

El rerun debe optimizar con las MISMAS sims que la planilla que diffea.

Con distinto n_sims el ascenso por coordenadas cae en otro óptimo local aunque la
semilla sea fija, y el diff deja de significar "se movieron los insumos" (medido el
2026-08-08: 56 de 96 picks reasignados con insumos idénticos, E[premio] out-of-sample
$238k → $221k).
"""

import pytest

from src.clausura.rerun_cierre import DEFAULT_SIMS, sims_de


def test_hereda_las_sims_de_la_planilla_previa():
    assert sims_de({"n_sims": 2400}) == 2400
    assert sims_de({"n_sims": 1600}) == 1600


def test_sin_registro_cae_al_default():
    """Planillas anteriores a este cambio no traen n_sims."""
    assert sims_de({}) == DEFAULT_SIMS
    assert sims_de({"n_sims": None}) == DEFAULT_SIMS
    assert sims_de({"n_sims": 0}) == DEFAULT_SIMS


def test_el_pedido_explicito_manda():
    """--sims N sigue funcionando para probar a mano; 0/None = heredar."""
    assert sims_de({"n_sims": 2400}, 800) == 800
    assert sims_de({"n_sims": 2400}, 0) == 2400
    assert sims_de({"n_sims": 2400}, None) == 2400


# -------------------- menú de candidatos: resultado negativo documentado --------------------

def test_el_menu_vigente_es_el_de_hueco():
    """Guardia del resultado negativo del 2026-08-08.

    `mispricing` (P real / P del pool) parece la métrica correcta y mide PEOR:
    Δ E[premio] −$9.486 ± 2.154 en 12 reps pareadas, negativo en 10/12. Si alguien
    cambia el default sin correr `--experimento-menu`, este test lo frena.
    """
    from src.clausura import strategy
    assert strategy.HUECO_METRIC == "legacy_hueco"


def test_las_dos_metricas_ordenan_distinto():
    """Con el kernel aditivo, el hueco premia exclusividad y el mispricing desajuste.

    Números reales de Liverpool–Albion (Fecha 1): el pool se equivoca IGUAL con 1-3 y
    1-4 (1.52× los dos), pero el 1-4 es cuatro veces más exclusivo.
    """
    from src.clausura.strategy import Candidato

    c13 = Candidato(pick=(1, 3), e_points=1.15, pool_q=0.01501, p_scoreline=0.02278)
    c14 = Candidato(pick=(1, 4), e_points=1.00, pool_q=0.00384, p_scoreline=0.00583)

    assert c14.hueco > c13.hueco                      # el hueco prefiere el más exclusivo
    assert c13.mispricing == pytest.approx(c14.mispricing, abs=0.05)   # desajuste igual


def test_default_sims_sigue_a_produccion():
    """El fallback tiene que estar en el régimen de producción, no en el viejo.

    Era 2400 y quedó peligrosamente viejo cuando producción pasó a 19.200: con
    K_EV=5, correr a 2.400 pone al optimizador donde NUESTRO menú mide negativo.
    """
    import re
    from pathlib import Path
    from src.clausura.rerun_cierre import DEFAULT_SIMS, SIMS_MIN_SEGURO

    unit = Path(__file__).resolve().parents[1] / "deploy" / "clausura-picks.service"
    m = re.search(r"--sims (\d+)", unit.read_text())
    assert m, "el unit de picks ya no pasa --sims"
    assert DEFAULT_SIMS == int(m.group(1)), (
        f"DEFAULT_SIMS={DEFAULT_SIMS} pero producción corre con {m.group(1)}")
    assert SIMS_MIN_SEGURO <= DEFAULT_SIMS


def test_avisa_si_corre_bajo_el_piso_seguro(caplog):
    """Correr con pocos sorteos no se clampea, pero tiene que hacer ruido."""
    import logging
    from src.clausura.rerun_cierre import sims_de

    with caplog.at_level(logging.WARNING):
        assert sims_de({"n_sims": 9600}, pedido=2400) == 2400
    assert any("piso seguro" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert sims_de({"n_sims": 19200}) == 19200
    assert not caplog.records, "no tiene que avisar cuando corre en régimen normal"

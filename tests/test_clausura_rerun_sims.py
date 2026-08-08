"""El rerun de cierre debe optimizar con las MISMAS sims que la planilla que diffea.

Con distinto n_sims el ascenso por coordenadas cae en otro óptimo local aunque la
semilla sea fija, y el diff deja de significar "se movieron los insumos" (medido el
2026-08-08: 56 de 96 picks reasignados con insumos idénticos, E[premio] out-of-sample
$238k → $221k).
"""

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

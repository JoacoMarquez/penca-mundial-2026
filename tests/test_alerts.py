"""Tests de las banderas rojas — reproducen los errores reales (m108 flood, Enciso)."""

from src.agent.alerts import (
    check_concentration, check_lambda_vs_xi, check_pool_slippage, check_llm_backfired,
)


def test_concentration_caza_el_flood():
    # m108 / hoy: 13 de 15 en el mismo marcador
    scores = [(1, 0)] * 13 + [(1, 1), (0, 0)]
    f = check_concentration(scores, 15)
    assert f is not None and f.code == "concentration"
    assert "13/15" in f.detail


def test_concentration_no_falsea_con_buen_reparto():
    # reparto sano: 10 marcadores distintos, máximo 2 cada uno
    scores = [(0, 2), (0, 2), (0, 1), (0, 1), (1, 1), (1, 1), (0, 3), (0, 3),
              (1, 2), (1, 4), (0, 4), (2, 3), (2, 4), (1, 3), (1, 3)]
    assert check_concentration(scores, 15) is None


def test_lambda_vs_xi_caza_enciso():
    qa = {
        "delta_lambda_L": 0.05,
        "delta_lambda_V": -0.18,
        "reasoning": "Multiples fuentes confirman que Julio Enciso salio lesionado de un amistoso.",
    }
    f = check_lambda_vs_xi(qa, home_xi=["Matt Turner"], away_xi=["Julio Enciso", "Miguel Almiron"],
                           home_team="EEUU", away_team="Paraguay")
    assert f is not None and f.code == "lambda_xi"
    assert "Enciso" in f.detail


def test_lambda_vs_xi_no_falsea_si_no_esta_en_xi():
    qa = {"delta_lambda_L": 0.0, "delta_lambda_V": -0.18,
          "reasoning": "Enciso lesionado, no juega."}
    # Enciso NO está en el XI → no hay contradicción → sin bandera
    assert check_lambda_vs_xi(qa, home_xi=[], away_xi=["Miguel Almiron"]) is None


def test_lambda_vs_xi_no_falsea_sin_recorte():
    qa = {"delta_lambda_L": 0.0, "delta_lambda_V": 0.0, "reasoning": "Enciso juega normal."}
    assert check_lambda_vs_xi(qa, home_xi=[], away_xi=["Julio Enciso"]) is None


def test_pool_slippage_caza_caida():
    prev = [{"penca_id": 1, "points_total": 18}] + [{"penca_id": 100 + i, "points_total": 20} for i in range(10)]
    # nuestra penca 1 sigue con 18 pero el resto subio mucho → cae muchos puestos
    curr = [{"penca_id": 1, "points_total": 18}] + [{"penca_id": 100 + i, "points_total": 60} for i in range(40)]
    f = check_pool_slippage(prev, curr, my_ids={1})
    assert f is not None and f.code == "pool_slip"


def test_pool_slippage_no_falsea_si_vamos_bien():
    curr = [{"penca_id": 1, "points_total": 50}] + [{"penca_id": 100 + i, "points_total": 10} for i in range(20)]
    assert check_pool_slippage(None, curr, my_ids={1}) is None


def test_llm_backfired():
    report = {"home_team": "EEUU", "away_team": "Paraguay",
              "llm_adjustment_was_helpful": "no",
              "llm_adjustment_applied": {"delta_lambda_L": 0.05, "delta_lambda_V": -0.18}}
    f = check_llm_backfired(report)
    assert f is not None and f.code == "llm_backfire"


def test_llm_backfired_no_falsea_si_fue_neutral():
    report = {"llm_adjustment_was_helpful": "neutral",
              "llm_adjustment_applied": {"delta_lambda_L": 0.0, "delta_lambda_V": -0.18}}
    assert check_llm_backfired(report) is None

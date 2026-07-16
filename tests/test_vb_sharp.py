"""Tests de src/valuebet/sharp.py."""

from src.valuebet.sharp import fair_probs
from src.valuebet.types import segment_key

SHARP_CFG = {"devig": "shin", "exact_score_devig": "proportional"}


def test_fair_probs_3way_suma_1():
    probs = fair_probs({"home": 2.10, "draw": 3.40, "away": 3.60}, "1x2", SHARP_CFG)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["home"] > probs["draw"]


def test_fair_probs_2way_suma_1():
    probs = fair_probs({"home": 1.60, "away": 2.45}, "moneyline", SHARP_CFG)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    # de-vig: la prob del favorito debe ser MENOR que la implícita cruda (1/1.60=0.625)
    assert probs["home"] < 1 / 1.60


def test_fair_probs_mercado_incompleto_falla():
    try:
        fair_probs({"home": 2.10}, "1x2", SHARP_CFG)
        raise AssertionError("debió tirar ValueError")
    except ValueError:
        pass


def test_fair_probs_ignora_odds_invalidas():
    probs = fair_probs({"home": 2.10, "draw": 0.0, "away": 3.60}, "1x2", SHARP_CFG)
    assert set(probs) == {"home", "away"}


def test_segment_key_bandas():
    assert segment_key("soccer", "1x2", 1.8) == "soccer|1x2|1.4-2.0"
    assert segment_key("soccer", "total_2.5", 2.5) == "soccer|total|2.0-3.0"
    assert segment_key("tennis", "moneyline", 4.0) == "tennis|moneyline|3.0-6.0"

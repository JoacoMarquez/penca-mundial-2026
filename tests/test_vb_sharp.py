"""Tests de src/valuebet/sharp.py y matching.py."""

from src.valuebet.matching import match_events, norm_name
from src.valuebet.sharp import fair_probs
from src.valuebet.types import OddsQuote, segment_key

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


# -------------------- matching --------------------

def _q(book, event_name, start="2026-07-17T12:00:00+00:00", sport="soccer"):
    return OddsQuote(book=book, sport=sport, league="l", event_id=f"{book}:1",
                     event_name=event_name, start_utc=start, market="1x2",
                     outcome="home", decimal_odds=2.0, fetched_utc=start)


def test_norm_name():
    assert norm_name("  Peñarol ") == "penarol"
    assert norm_name("SÃO PAULO") != ""  # no explota con unicode raro


def test_match_directo():
    pairs = match_events([_q("supermatch", "Nacional vs Penarol")],
                         [_q("pinnacle", "Nacional vs Peñarol")], aliases={})
    assert len(pairs) == 1


def test_match_por_alias():
    aliases = {"soccer": {"Manchester United": ["Man Utd", "Man United"]}}
    pairs = match_events([_q("supermatch", "Man Utd vs Chelsea")],
                         [_q("pinnacle", "Manchester United vs Chelsea")], aliases)
    assert len(pairs) == 1


def test_unmatched_fuera_de_ventana_horaria():
    pairs = match_events([_q("supermatch", "A vs B", start="2026-07-17T12:00:00+00:00")],
                         [_q("pinnacle", "A vs B", start="2026-07-17T18:00:00+00:00")],
                         aliases={})
    assert pairs == []  # 6h de diferencia > ventana de 2h


def test_unmatched_no_explota():
    pairs = match_events([_q("supermatch", "Equipo Fantasma vs Nadie")],
                         [_q("pinnacle", "A vs B")], aliases={})
    assert pairs == []

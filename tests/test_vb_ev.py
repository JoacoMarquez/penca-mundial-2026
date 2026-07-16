"""Tests de src/valuebet/ev.py — detección de singles y parlays."""

from datetime import datetime, timedelta, timezone

from src.valuebet import config as vbconfig
from src.valuebet.ev import combined_edge, find_parlays, find_singles
from src.valuebet.learning import LearningState
from src.valuebet.types import OddsQuote, Opportunity

CFG = vbconfig.load()
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def _quote(odds=2.2, outcome="home", market="1x2", event_id="e1", hours_ahead=24.0,
           sport="soccer", book="supermatch"):
    start = (NOW + timedelta(hours=hours_ahead)).isoformat()
    return OddsQuote(
        book=book, sport=sport, league="liga", event_id=event_id,
        event_name="A vs B", start_utc=start, market=market, outcome=outcome,
        decimal_odds=odds, fetched_utc=NOW.isoformat(),
    )


def _sharp(soft_q, odds):
    """La cuota sharp del mismo outcome que la soft (lo que arma el runner)."""
    return OddsQuote(
        book="pinnacle", sport=soft_q.sport, league="liga", event_id="pinn:9",
        event_name=soft_q.event_name, start_utc=soft_q.start_utc,
        market=soft_q.market, outcome=soft_q.outcome, decimal_odds=odds,
        fetched_utc=soft_q.fetched_utc,
    )


def test_single_edge_exacto():
    # fair 0.50 a cuota 2.20 → edge = 0.50*2.20 - 1 = 0.10 ≥ min_edge soccer 0.03
    q = _quote(odds=2.2)
    fair = {"home": 0.50, "draw": 0.28, "away": 0.22}
    out = find_singles([(q, fair, _sharp(q, 2.05))], CFG, LearningState(), now_utc=NOW)
    assert len(out) == 1
    assert abs(out[0].edge - 0.10) < 1e-9


def test_single_persiste_sharp_event_id():
    # el close necesita el event_id sharp para reencontrar la línea de cierre
    q = _quote(odds=2.2)
    fair = {"home": 0.50, "draw": 0.28, "away": 0.22}
    out = find_singles([(q, fair, _sharp(q, 2.05))], CFG, LearningState(), now_utc=NOW)
    assert out[0].sharp_event_id == "pinn:9"
    assert out[0].sharp_odds == 2.05


def test_single_bajo_umbral_no_sugiere():
    q = _quote(odds=2.02)  # edge = 0.50*2.02-1 = 0.01 < 0.03
    fair = {"home": 0.50, "draw": 0.28, "away": 0.22}
    assert find_singles([(q, fair, _sharp(q, 2.0))], CFG, LearningState(), now_utc=NOW) == []


def test_filtro_odds_range():
    q = _quote(odds=8.0)  # fuera de [1.4, 6.0] aunque el edge sea enorme
    fair = {"home": 0.30, "draw": 0.40, "away": 0.30}
    assert find_singles([(q, fair, _sharp(q, 7.0))], CFG, LearningState(), now_utc=NOW) == []


def test_filtro_horizonte():
    q = _quote(odds=2.2, hours_ahead=0.5)  # a 30 min, min_hours_ahead=1
    fair = {"home": 0.50, "draw": 0.28, "away": 0.22}
    assert find_singles([(q, fair, _sharp(q, 2.05))], CFG, LearningState(), now_utc=NOW) == []


def test_edge_absurdo_se_descarta():
    # caso real del bug de tarjetas: fair 0.46 a cuota 5.2 → edge 139% > max_edge 0.25
    q = _quote(odds=5.2, outcome="away")
    fair = {"home": 0.44, "draw": 0.10, "away": 0.46}
    assert find_singles([(q, fair, _sharp(q, 2.04))], CFG, LearningState(), now_utc=NOW) == []


def test_edge_justo_bajo_el_tope_si_pasa():
    # edge 20% (< max_edge 25%) con cuota en rango → sí se sugiere
    q = _quote(odds=3.0, outcome="away")
    fair = {"home": 0.30, "draw": 0.30, "away": 0.40}  # edge = 0.40*3.0-1 = 0.20
    out = find_singles([(q, fair, _sharp(q, 2.6))], CFG, LearningState(), now_utc=NOW)
    assert len(out) == 1


def test_segmento_desactivado_no_sugiere():
    q = _quote(odds=2.2)
    fair = {"home": 0.50, "draw": 0.28, "away": 0.22}
    state = LearningState()
    seg = "soccer|1x2|2.0-3.0"
    state.segments[seg] = {"n": 30, "sum_clv": -3.0, "sum_edge_pred": 1.5, "multiplier": 0.0}
    assert find_singles([(q, fair, _sharp(q, 2.05))], CFG, state, now_utc=NOW) == []


def test_multiplicador_sube_umbral():
    # multiplier 0.5 → umbral efectivo soccer = 0.03/0.5 = 0.06 > edge 0.05 → no sugiere
    q = _quote(odds=2.10)
    fair = {"home": 0.50, "draw": 0.28, "away": 0.22}  # edge = 0.05
    state = LearningState()
    state.segments["soccer|1x2|2.0-3.0"] = {"n": 10, "sum_clv": -0.5,
                                            "sum_edge_pred": 0.5, "multiplier": 0.5}
    assert find_singles([(q, fair, _sharp(q, 2.0))], CFG, state, now_utc=NOW) == []


def _opp(event_id, edge, odds=2.2):
    q = _quote(odds=odds, event_id=event_id)
    p = (1.0 + edge) / odds
    return Opportunity(quote=q, fair_prob=p, sharp_odds=odds * 0.95, edge=edge)


def _parlay_cfg():
    cfg = vbconfig.load()
    cfg.parlay = dict(cfg.parlay)
    cfg.parlay["enabled"] = True
    return cfg


def test_parlay_nunca_mismo_evento():
    cfg = _parlay_cfg()
    legs = find_parlays([_opp("e1", 0.06), _opp("e1", 0.05)], cfg)
    assert legs == []  # mismo event_id → no combinable


def test_parlay_edge_multiplicativo():
    cfg = _parlay_cfg()
    out = find_parlays([_opp("e1", 0.06), _opp("e2", 0.05)], cfg)
    assert len(out) == 1
    expected = 1.06 * 1.05 - 1
    assert abs(combined_edge(out[0]) - expected) < 1e-9
    assert expected >= cfg.parlay["min_edge"]


def test_parlay_bajo_min_edge_no_sale():
    cfg = _parlay_cfg()
    # 1.03*1.03-1 ≈ 6.1% < 8%
    assert find_parlays([_opp("e1", 0.03), _opp("e2", 0.03)], cfg) == []


def test_parlay_max_legs():
    cfg = _parlay_cfg()
    opps = [_opp(f"e{i}", 0.06) for i in range(5)]
    for legs in find_parlays(opps, cfg):
        assert len(legs) <= cfg.parlay["max_legs"]


def test_parlay_deshabilitado():
    assert find_parlays([_opp("e1", 0.10), _opp("e2", 0.10)], CFG) == []

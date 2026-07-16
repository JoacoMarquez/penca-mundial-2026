"""Tests del parser Pinnacle a nivel deporte (normalize_sport) — sin red."""

from src.valuebet.books.pinnacle_vb import normalize_sport

# payload sintético estilo /sports/{id}/matchups y /markets/straight, dos ligas
MATCHUPS = [
    {"id": 1, "type": "matchup", "startTime": "2026-07-18T14:00:00Z",
     "league": {"id": 100, "name": "Liga A"},
     "participants": [{"name": "Alpha FC", "alignment": "home"},
                      {"name": "Beta", "alignment": "away"}]},
    {"id": 2, "type": "matchup", "startTime": "2026-07-18T16:00:00Z",
     "league": {"id": 200, "name": "Liga B"},
     "participants": [{"name": "Gamma", "alignment": "home"},
                      {"name": "Delta", "alignment": "away"}]},
    {"id": 9, "type": "special", "startTime": "2026-07-18T14:00:00Z",
     "league": {"id": 100, "name": "Liga A"}, "participants": []},  # no matchup
]
MARKETS = [
    {"matchupId": 1, "type": "moneyline", "period": 0,
     "prices": [{"designation": "home", "price": 150}, {"designation": "draw", "price": 240},
                {"designation": "away", "price": 180}]},
    {"matchupId": 1, "type": "total", "period": 0, "points": 2.5,
     "prices": [{"designation": "over", "price": -110}, {"designation": "under", "price": -110}]},
    {"matchupId": 1, "type": "moneyline", "period": 1,  # 1er tiempo → se ignora
     "prices": [{"designation": "home", "price": 150}, {"designation": "away", "price": 180}]},
    {"matchupId": 2, "type": "moneyline", "period": 0,
     "prices": [{"designation": "home", "price": -200}, {"designation": "draw", "price": 300},
                {"designation": "away", "price": 500}]},
]


def test_normaliza_todas_las_ligas():
    q = normalize_sport("soccer", MATCHUPS, MARKETS)
    eids = {x.event_id for x in q}
    assert eids == {"pinn:1", "pinn:2"}          # ambas ligas
    m1 = {x.market for x in q if x.event_id == "pinn:1"}
    assert "1x2" in m1 and "total_2.5" in m1
    # solo full-game (period 0)
    assert all(x.decimal_odds > 1.0 for x in q)


def test_whitelist_filtra_liga():
    q = normalize_sport("soccer", MATCHUPS, MARKETS, league_ids={100})
    assert {x.event_id for x in q} == {"pinn:1"}  # solo Liga A


def test_1x2_tiene_tres_outcomes():
    q = normalize_sport("soccer", MATCHUPS, MARKETS, league_ids={100})
    x = {o.outcome: o.decimal_odds for o in q if o.market == "1x2"}
    assert set(x) == {"home", "draw", "away"}
    assert abs(x["home"] - 2.5) < 1e-9  # +150 → 2.50


def test_ignora_pseudo_eventos_bookings_corners():
    # Pinnacle publica sub-mercados como pseudo-partidos con "(Bookings)" en el nombre
    matchups = MATCHUPS + [
        {"id": 5, "type": "matchup", "startTime": "2026-07-18T14:00:00Z",
         "league": {"id": 100, "name": "Liga A Bookings"},
         "participants": [{"name": "Alpha FC (Bookings)", "alignment": "home"},
                          {"name": "Beta (Bookings)", "alignment": "away"}]},
    ]
    markets = MARKETS + [
        {"matchupId": 5, "type": "moneyline", "period": 0,
         "prices": [{"designation": "home", "price": -110}, {"designation": "draw", "price": 300},
                    {"designation": "away", "price": 120}]},
    ]
    q = normalize_sport("soccer", matchups, markets)
    assert "pinn:5" not in {x.event_id for x in q}  # el pseudo-evento se descarta
    assert "pinn:1" in {x.event_id for x in q}      # el partido real queda


def test_basket_es_moneyline_2way():
    mk = [{"matchupId": 2, "type": "moneyline", "period": 0,
           "prices": [{"designation": "home", "price": -200}, {"designation": "away", "price": 170}]}]
    q = normalize_sport("basketball", MATCHUPS, mk, league_ids={200})
    ml = [o for o in q if o.market == "moneyline"]
    assert {o.outcome for o in ml} == {"home", "away"}  # sin empate


def test_total_linea_entera_canonica():
    # points=3.0 (float) debe producir "total_3", igual que "Total ( 3 )" de Supermatch
    mk = [{"matchupId": 1, "type": "total", "period": 0, "points": 3.0,
           "prices": [{"designation": "over", "price": -105}, {"designation": "under", "price": -115}]}]
    q = normalize_sport("soccer", MATCHUPS, mk, league_ids={100})
    assert {x.market for x in q} == {"total_3"}

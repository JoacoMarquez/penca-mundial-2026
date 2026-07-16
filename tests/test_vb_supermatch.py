"""Tests del parser de Supermatch contra un fixture real capturado en Fase 0.

Sin red: usa tests/fixtures/valuebet/supermatch_events.json.
"""

import json
from pathlib import Path

from src.valuebet.books.supermatch import parse_events

FIXTURE = Path(__file__).parent / "fixtures" / "valuebet" / "supermatch_events.json"


def _load(sport_name: str):
    return json.loads(FIXTURE.read_text())[sport_name]


def test_futbol_1x2_y_total():
    quotes = parse_events("soccer", _load("Fútbol"))
    assert quotes, "debería parsear al menos una cuota de fútbol"
    markets = {q.market for q in quotes}
    assert "1x2" in markets
    x = [q for q in quotes if q.market == "1x2"]
    assert {q.outcome for q in x} == {"home", "draw", "away"}
    assert all(q.decimal_odds > 1.0 for q in quotes)
    assert all(q.book == "supermatch" and q.sport == "soccer" for q in quotes)
    # totales: mercado con la línea embebida
    assert any(q.market.startswith("total_") for q in quotes)


def test_basquet_moneyline():
    quotes = parse_events("basketball", _load("Baloncesto"))
    ml = [q for q in quotes if q.market == "moneyline"]
    assert ml, "básquet debería tener moneyline (ft2w)"
    assert {q.outcome for q in ml} == {"home", "away"}
    # básquet no tiene empate en moneyline
    assert all(q.outcome != "draw" for q in ml)


def test_tenis_moneyline():
    quotes = parse_events("tennis", _load("Tenis"))
    ml = [q for q in quotes if q.market == "moneyline"]
    assert ml, "tenis debería tener moneyline (ft2w Ganador)"
    assert {q.outcome for q in ml} == {"home", "away"}


def test_event_name_y_start_utc_presentes():
    quotes = parse_events("soccer", _load("Fútbol"))
    q = quotes[0]
    assert " vs " in q.event_name
    assert q.start_utc.endswith("+00:00")
    assert q.event_id.startswith("sm:")

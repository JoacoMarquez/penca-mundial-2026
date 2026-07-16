"""Tests de src/valuebet/matching.py — fuzzy español↔inglés, sufijos, tenis, seguridad.

Los pares reales usados como casos salen del recon en vivo (Supermatch vs Pinnacle).
"""

from src.valuebet.matching import (
    match_events, norm_name, team_similarity, _alias_to_canonical,
)
from src.valuebet.types import OddsQuote


def _ev(book, event_name, sport="soccer", start="2026-07-17T12:00:00+00:00",
        market="1x2", outcomes=("home", "draw", "away")):
    """Genera las cuotas de un evento (una por outcome) para armar inputs de match."""
    eid = f"{book}:{abs(hash(event_name)) % 10000}"
    return [
        OddsQuote(book=book, sport=sport, league="l", event_id=eid,
                  event_name=event_name, start_utc=start, market=market,
                  outcome=o, decimal_odds=2.0, fetched_utc=start)
        for o in outcomes
    ]


def _sim(a, b, aliases=None, sport="soccer"):
    sport_aliases = (aliases or {}).get(sport, {})
    return team_similarity(a, b, _alias_to_canonical(sport_aliases))


# -------------------- norm + similitud --------------------

def test_norm_name():
    assert norm_name("  Peñarol ") == "penarol"
    assert norm_name("São Paulo") == "sao paulo"


def test_sufijos_de_club_no_rompen_match():
    # casos reales: SM trae sufijos, Pinnacle no
    assert _sim("IK Start", "Start") == 1.0
    assert _sim("Chapecoense SC", "Chapecoense") == 1.0
    assert _sim("Tromso IL", "Tromso") == 1.0
    assert _sim("Red Bull Bragantino", "Bragantino") >= 0.5  # token compartido


def test_tenis_apellido_nombre_invertido():
    # "Apellido, Nombre" ↔ "Nombre Apellido" → mismo conjunto de tokens
    assert _sim("Molcan, Alex", "Alex Molcan") == 1.0
    assert _sim("Davidovich Fokina, Alejandro",
                "Alejandro Davidovich Fokina") == 1.0


def test_alias_espanol_ingles():
    aliases = {"soccer": {"Marseille": ["Olympique Marsella"]}}
    assert _sim("Olympique Marsella", "Marseille", aliases) == 1.0


def test_equipos_distintos_no_matchean():
    assert _sim("Nacional", "Peñarol") < 0.80
    assert _sim("Boca Juniors", "River Plate") < 0.80


# -------------------- match_events --------------------

def test_match_directo_con_tildes():
    pairs = match_events(_ev("supermatch", "Nacional vs Penarol"),
                         _ev("pinnacle", "Nacional vs Peñarol"), aliases={})
    assert len(pairs) == 3  # una por outcome del evento soft
    # cada par apunta a las cuotas del evento sharp matcheado
    assert all(sh[0].book == "pinnacle" for _, sh in pairs)


def test_match_sufijo_real():
    pairs = match_events(_ev("supermatch", "Molde vs SK Brann"),
                         _ev("pinnacle", "Molde vs Brann"), aliases={})
    assert len(pairs) == 3


def test_match_por_alias():
    aliases = {"soccer": {"Manchester United": ["Man Utd"]}}
    pairs = match_events(_ev("supermatch", "Man Utd vs Chelsea"),
                         _ev("pinnacle", "Manchester United vs Chelsea"), aliases)
    assert len(pairs) == 3


def test_no_match_fuera_de_ventana():
    pairs = match_events(
        _ev("supermatch", "A Equipo vs B Equipo", start="2026-07-17T12:00:00+00:00"),
        _ev("pinnacle", "A Equipo vs B Equipo", start="2026-07-17T18:00:00+00:00"),
        aliases={})
    assert pairs == []


def test_no_match_equipos_distintos():
    pairs = match_events(_ev("supermatch", "Fantasma vs Nadie"),
                         _ev("pinnacle", "Barcelona vs Madrid"), aliases={})
    assert pairs == []


def test_seguridad_home_away_invertido_no_matchea():
    # mismo fixture pero local/visitante cruzados → se descarta (no invertir un 1x2)
    pairs = match_events(_ev("supermatch", "Fluminense vs Gremio"),
                         _ev("pinnacle", "Gremio vs Fluminense"), aliases={})
    assert pairs == []


def test_elige_el_mejor_entre_varios_sharp():
    # dos partidos sharp simultáneos; debe elegir el del nombre correcto
    soft = _ev("supermatch", "Bahia vs Chapecoense SC")
    sharp = (_ev("pinnacle", "Bahia vs Chapecoense")
             + _ev("pinnacle", "Flamengo vs Palmeiras"))
    pairs = match_events(soft, sharp, aliases={})
    assert len(pairs) == 3
    names = {sh[0].event_name for _, sh in pairs}
    assert names == {"Bahia vs Chapecoense"}

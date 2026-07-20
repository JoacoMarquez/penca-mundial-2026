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


def test_norm_name_apostrofe_no_parte_token():
    # el apóstrofe se elimina, no se vuelve espacio (no debe partir K'un en k + un)
    assert norm_name("Dalian K'un City") == "dalian kun city"
    assert norm_name("O'Higgins") == "ohiggins"


def test_apostrofe_matchea_sin_apostrofe():
    assert _sim("Dalian Kun City", "Dalian K'un City") == 1.0


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


def test_aliases_del_unmatched_puentean_pares_reales():
    # cada par (SM, Pinnacle) salió de data/valuebet/unmatched/ (scan 2026-07-16):
    # con los aliases de config/valuebet.yaml deben puentear a ≥ umbral.
    from src.valuebet import config as vbconfig
    aliases = vbconfig.load().aliases
    pares = [
        ("soccer", "Saint Louis City SC", "St Louis City SC"),
        ("soccer", "Operario PR", "Operario Ferroviario"),
        ("soccer", "Sporting Jax", "Sporting Club Jacksonville"),
        ("soccer", "OFK Belgrado", "OFK Beograd"),
        ("soccer", "Tianjin Teda", "Tianjin Jinmen Tiger"),
        ("soccer", "Mitre Santiago Del Estero", "Club Atletico Mitre"),
        ("basketball", "Macedonia del Norte", "North Macedonia"),
        ("tennis", "Sherif Ahmed Abdelaziz, Maiar", "Mayar Sherif"),
    ]
    for sport, sm, pinn in pares:
        assert _sim(sm, pinn, aliases, sport=sport) >= 0.80, f"{sm} ↔ {pinn}"


def test_equipos_distintos_no_matchean():
    assert _sim("Nacional", "Peñarol") < 0.80
    assert _sim("Boca Juniors", "River Plate") < 0.80


def test_no_confunde_equipos_de_misma_ciudad():
    # caso real: distinto equipo, misma ciudad → los tokens de ciudad NO deben alcanzar
    assert _sim("Gigantes San Francisco De Macoris",
                "Indios de San Francisco de Macoris") < 0.80


def test_tolera_variante_ortografica_de_token():
    # caso real: mismo equipo, una letra de diferencia
    assert _sim("Ljungskille SK", "Ljungskile") == 1.0
    assert _sim("Busan I Park FC", "Busan IPark") == 1.0


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


# -------------------- calificadores de plantel --------------------

def test_reserva_no_matchea_primer_equipo():
    # "Barcelona B" y "Barcelona" son equipos DISTINTOS que juegan el mismo día
    assert _sim("Barcelona B", "Barcelona") == 0.0
    assert _sim("Castilla II", "Castilla") == 0.0
    assert _sim("Ajax Reserves", "Ajax") == 0.0
    assert _sim("Milan U23", "Milan") == 0.0


def test_femenino_no_matchea_masculino():
    assert _sim("Arsenal W", "Arsenal") == 0.0
    assert _sim("Boca Juniors Femenino", "Boca Juniors") == 0.0


def test_mismo_calificador_en_ambos_lados_si_matchea():
    assert _sim("Barcelona B", "Barcelona B") == 1.0
    assert _sim("Arsenal Women", "Arsenal W") == 0.0  # variantes distintas NO se puentean solas
    assert _sim("Ajax Reserves", "Ajax Reserves") == 1.0


def test_match_events_no_cruza_reserva_con_primer_equipo():
    soft = _ev("supermatch", "Real Madrid vs Barcelona B")
    sharp = _ev("pinnacle", "Real Madrid vs Barcelona")
    assert match_events(soft, sharp, aliases={}) == []


# -------------------- ambigüedad --------------------

def test_match_ambiguo_se_descarta():
    # dos candidatos sharp que solo difieren en una variante ortográfica del nombre:
    # ambos puntúan 1.0 contra la soft → no hay forma segura de elegir → se descarta
    soft = _ev("supermatch", "Ljungskile vs Orgryte")
    sharp = (_ev("pinnacle", "Ljungskile vs Orgryte")
             + _ev("pinnacle", "Ljungskille vs Orgryte"))
    assert match_events(soft, sharp, aliases={}) == []


def test_candidato_claro_no_es_ambiguo():
    # un candidato en 1.0 y otro bien abajo del umbral → match normal
    soft = _ev("supermatch", "Bahia vs Chapecoense SC")
    sharp = (_ev("pinnacle", "Bahia vs Chapecoense")
             + _ev("pinnacle", "Flamengo vs Palmeiras"))
    assert len(match_events(soft, sharp, aliases={})) == 3


def test_dump_unmatched_jsonl(tmp_path):
    import json
    soft = _ev("supermatch", "Fantasma vs Nadie")
    sharp = _ev("pinnacle", "Barcelona vs Madrid")
    match_events(soft, sharp, aliases={}, dump_dir=tmp_path)
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(l) for l in files[0].read_text().splitlines()]
    assert rows[0]["home"] == "Fantasma"
    assert rows[0]["reason"] == "below_threshold"

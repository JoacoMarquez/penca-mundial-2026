"""Tests del parseo de alineaciones de SofaScore y su llegada a la Capa 4.

Cubre el fix B: tener el XI confirmado con nombres para que Capa 4 pueda
desmentir rumores de noticias (caso Enciso: rumor de lesión en amistoso que
no se concretó — el jugador terminó en el XI confirmado).
"""

from src.scrapers.sofascore import parse_lineups, _match_event
from src.model.qualitative import MatchContext, build_user_prompt


# ---- payload de muestra (forma real de /event/{id}/lineups) ----

SAMPLE = {
    "confirmed": True,
    "home": {
        "formation": "4-3-3",
        "players": [
            {"player": {"name": "Matt Turner"}, "substitute": False},
            {"player": {"name": "Chris Richards"}, "substitute": False},
            {"player": {"name": "Christian Pulisic"}, "substitute": False},
            {"player": {"name": "Suplente Local"}, "substitute": True},
        ],
    },
    "away": {
        "formation": "4-4-2",
        "players": [
            {"player": {"name": "Julio Enciso"}, "substitute": False},
            {"player": {"name": "Miguel Almirón"}, "substitute": False},
            {"player": {"name": "Suplente Visit"}, "substitute": True},
        ],
    },
}


def test_parse_lineups_extrae_xi_y_confirmed():
    out = parse_lineups(SAMPLE)
    assert out["lineup_confirmed"] is True
    assert out["lineup_source"] == "sofascore"
    assert out["home_lineup_change"] == "Formación: 4-3-3"
    assert out["away_lineup_change"] == "Formación: 4-4-2"
    # el XI titular excluye suplentes
    assert "Julio Enciso" in out["away_xi"]
    assert "Suplente Visit" not in out["away_xi"]
    assert "Suplente Local" not in out["home_xi"]
    assert len(out["home_xi"]) == 3
    assert len(out["away_xi"]) == 2


def test_parse_lineups_vacio_o_none():
    assert parse_lineups(None) == {}
    assert parse_lineups({}) == {}
    # sin formación ni jugadores → nada útil
    assert parse_lineups({"confirmed": False, "home": {}, "away": {}}) == {}


def test_parse_lineups_probable_no_confirmado():
    payload = {"confirmed": False, "home": {"formation": "4-3-3", "players": []}, "away": {}}
    out = parse_lineups(payload)
    assert out["lineup_confirmed"] is False
    assert out["home_lineup_change"] == "Formación: 4-3-3"


def test_match_event_por_nombres():
    search = {"events": [
        {"homeTeam": {"name": "Brazil"}, "awayTeam": {"name": "Croatia"}, "id": 1},
        {"homeTeam": {"name": "USA"}, "awayTeam": {"name": "Paraguay"}, "id": 42},
    ]}
    ev = _match_event(search, "USA", "Paraguay")
    assert ev is not None and ev["id"] == 42
    assert _match_event(search, "Argentina", "Mexico") is None
    assert _match_event(None, "USA", "Paraguay") is None


def test_prompt_marca_xi_confirmado_y_lo_distingue_de_probable():
    base = dict(
        home_team="Estados Unidos", away_team="Paraguay",
        kickoff_local="Vie 12/06 22:00 UY", stage="Fase de grupos",
        market_p_home=0.46, market_p_draw=0.27, market_p_away=0.27,
        market_e_goals_L=1.4, market_e_goals_V=0.9,
        away_news_summary="Enciso salió lesionado de un amistoso, extrema preocupación.",
    )
    confirmado = MatchContext(**base, away_xi=["Julio Enciso", "Miguel Almirón"], lineup_confirmed=True)
    prompt = build_user_prompt(confirmado)
    assert "XI CONFIRMADO VISIT: Julio Enciso" in prompt

    probable = MatchContext(**base, away_xi=["Julio Enciso"], lineup_confirmed=False)
    assert "XI PROBABLE VISIT: Julio Enciso" in build_user_prompt(probable)

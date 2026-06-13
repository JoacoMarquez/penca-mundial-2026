"""Tests del parseo del XI titular confirmado desde el /summary de ESPN.

ESPN es la fuente primaria de alineación confirmada (SofaScore quedó bloqueado
con 403 desde su API). ESPN marca starter==True en los 11 titulares una vez
anunciada la alineación (~1h antes del kickoff). Antes de eso, rosters viene
vacío y devolvemos {} — sin inventar XI.

Mismo contrato de claves que sofascore.parse_lineups, para que el pipeline las
consuma sin cambios (caso Enciso: rumor de lesión que el XI confirmado desmiente).
"""

from src.scrapers.espn import parse_starters_from_summary
from src.model.qualitative import MatchContext, build_user_prompt


def _roster_entry(name: str, starter: bool):
    return {"starter": starter, "athlete": {"displayName": name}}


def _xi(prefix: str, n: int = 11):
    return [_roster_entry(f"{prefix} Titular {i}", True) for i in range(1, n + 1)]


# ---- payload de muestra (forma real de /summary -> rosters[]) ----

SAMPLE = {
    "rosters": [
        {
            "team": {"displayName": "USA"},
            "formation": "4-3-3",
            "roster": _xi("USA") + [_roster_entry("Suplente Local", False)],
        },
        {
            "team": {"displayName": "Paraguay"},
            "formation": "4-4-2",
            "roster": (
                [_roster_entry("Julio Enciso", True)]
                + _xi("PAR", 10)
                + [_roster_entry("Suplente Visit", False)]
            ),
        },
    ]
}


def test_parse_starters_extrae_xi_y_confirmed():
    out = parse_starters_from_summary(SAMPLE, "USA", "Paraguay")
    assert out["lineup_confirmed"] is True
    assert out["lineup_source"] == "espn"
    assert out["home_lineup_change"] == "Formación: 4-3-3"
    assert out["away_lineup_change"] == "Formación: 4-4-2"
    # el XI titular excluye suplentes
    assert "Julio Enciso" in out["away_xi"]
    assert "Suplente Visit" not in out["away_xi"]
    assert "Suplente Local" not in out["home_xi"]
    assert len(out["home_xi"]) == 11
    assert len(out["away_xi"]) == 11


def test_parse_starters_vacio_o_none():
    assert parse_starters_from_summary(None, "USA", "Paraguay") == {}
    assert parse_starters_from_summary({}, "USA", "Paraguay") == {}
    # rosters presentes pero sin starters (alineación aún no anunciada, T lejos del kickoff)
    pre = {"rosters": [
        {"team": {"displayName": "USA"}, "roster": []},
        {"team": {"displayName": "Paraguay"}, "roster": []},
    ]}
    assert parse_starters_from_summary(pre, "USA", "Paraguay") == {}


def test_parse_starters_un_solo_lado_no_confirma():
    """Si solo un equipo anunció XI, devolvemos lo que hay pero NO marcamos confirmado."""
    payload = {"rosters": [
        {"team": {"displayName": "USA"}, "formation": "4-3-3", "roster": _xi("USA")},
        {"team": {"displayName": "Paraguay"}, "roster": []},
    ]}
    out = parse_starters_from_summary(payload, "USA", "Paraguay")
    assert len(out["home_xi"]) == 11
    assert "away_xi" not in out
    assert out["lineup_confirmed"] is False


def test_parse_starters_asignacion_posicional_si_no_matchea_nombre():
    """Aliases/idioma distinto: si el displayName no matchea, cae a orden [home, away]."""
    payload = {"rosters": [
        {"team": {"displayName": "United States"}, "formation": "4-3-3", "roster": _xi("USA")},
        {"team": {"displayName": "Paraguay"}, "formation": "4-4-2", "roster": _xi("PAR")},
    ]}
    out = parse_starters_from_summary(payload, "USA", "Paraguay")
    assert out["lineup_confirmed"] is True
    assert any("USA" in n for n in out["home_xi"])
    assert any("PAR" in n for n in out["away_xi"])


def test_xi_confirmado_de_espn_llega_a_capa4():
    """El XI confirmado de ESPN debe poder distinguirse de un XI probable en el prompt."""
    out = parse_starters_from_summary(SAMPLE, "USA", "Paraguay")
    ctx = MatchContext(
        home_team="USA", away_team="Paraguay", kickoff_local="2026-06-XX",
        stage="group", market_p_home=0.5, market_p_draw=0.25, market_p_away=0.25,
        market_e_goals_L=1.6, market_e_goals_V=1.0,
        home_xi=out.get("home_xi"), away_xi=out.get("away_xi"),
        lineup_confirmed=out.get("lineup_confirmed", False),
    )
    prompt = build_user_prompt(ctx)
    assert "XI CONFIRMADO" in prompt
    assert "Julio Enciso" in prompt

"""Tests del override manual de disponibilidad de jugadores (Capa 4)."""

import textwrap

from src.model.overrides import (
    load_player_overrides,
    team_overrides,
    filter_available,
)
from src.model.qualitative import MatchContext, build_user_prompt


def _write_cfg(tmp_path, body: str) -> str:
    p = tmp_path / "player_overrides.yaml"
    p.write_text(textwrap.dedent(body))
    load_player_overrides.cache_clear()
    return str(p)


def test_loader_parses_available_and_unavailable(tmp_path):
    path = _write_cfg(tmp_path, """
        available:
          tur:
            - "Arda Güler"
        unavailable:
          BRA:
            - "Neymar"
    """)
    ov = load_player_overrides(path)
    # Las claves se normalizan a mayúsculas.
    assert ov["available"]["TUR"] == ["Arda Güler"]
    assert ov["unavailable"]["BRA"] == ["Neymar"]


def test_team_overrides_resolves_by_code(tmp_path):
    path = _write_cfg(tmp_path, """
        available:
          TUR: ["Arda Güler"]
        unavailable: {}
    """)
    avail, out = team_overrides("tur", path)
    assert avail == ["Arda Güler"]
    assert out == []
    # Equipo sin overrides → listas vacías.
    assert team_overrides("ARG", path) == ([], [])
    assert team_overrides(None, path) == ([], [])


def test_filter_available_drops_accent_insensitive():
    # La baja reportada usa 'Guler' sin diéresis; el override usa 'Güler'.
    kept = filter_available(["Arda Guler (lesión muscular)", "Hakan (duda)"], ["Arda Güler"])
    assert kept == ["Hakan (duda)"]


def test_filter_available_noop_when_empty():
    assert filter_available(None, ["X"]) == []
    assert filter_available(["A", "B"], []) == ["A", "B"]


def test_prompt_injects_authoritative_availability_line():
    ctx = MatchContext(
        home_team="Australia",
        away_team="Türkiye",
        kickoff_local="Sab 14/06 01:00 UY",
        stage="group",
        market_p_home=0.18, market_p_draw=0.26, market_p_away=0.56,
        market_e_goals_L=0.9, market_e_goals_V=1.7,
        away_available=["Arda Güler"],
    )
    prompt = build_user_prompt(ctx)
    assert "DISPONIBILIDAD CONFIRMADA VISIT" in prompt
    assert "Arda Güler" in prompt
    # Sin override del local, no aparece la línea local.
    assert "DISPONIBILIDAD CONFIRMADA LOCAL" not in prompt


def test_real_config_has_guler_for_tur():
    """El config commiteado debe traer el fix de Güler para TUR (regresión)."""
    load_player_overrides.cache_clear()
    avail, _ = team_overrides("TUR")
    assert any("güler" in a.casefold() or "guler" in a.casefold() for a in avail)
    load_player_overrides.cache_clear()

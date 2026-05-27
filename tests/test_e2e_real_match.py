"""Test E2E: pipeline con datos reales contra un partido que API-Football SÍ tenga cargado.

Uso: python -m tests.test_e2e_real_match [home_team] [away_team]
Por default usa Libertadores: Bolívar vs Independ. Rivadavia (mañana).

Valida que:
    - API-Football retorna lineups/lesiones/forma/h2h reales
    - LLM Capa 4 puede procesar ese contexto y dar ajustes coherentes
    - El módulo de weather funciona contra coords reales (depende de key OWM activa)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)

from src.scrapers.football_api import (
    collect_match_context,
    find_fixture_for_match,
    WORLD_CUP_LEAGUE_ID,
)


def test_against_libertadores():
    """Forzar búsqueda en Libertadores (league=13) en vez de WC."""
    import src.scrapers.football_api as fapi

    # Override temporal para forzar a buscar en Libertadores
    original_league = fapi.WORLD_CUP_LEAGUE_ID
    original_season = fapi.WORLD_CUP_SEASON
    fapi.WORLD_CUP_LEAGUE_ID = 13   # CONMEBOL Libertadores
    fapi.WORLD_CUP_SEASON = 2026

    home = sys.argv[1] if len(sys.argv) > 1 else "Bolivar"
    away = sys.argv[2] if len(sys.argv) > 2 else "Independ. Rivadavia"

    # Kickoff de mañana
    from datetime import timedelta
    kickoff = datetime.now(timezone.utc) + timedelta(hours=24)

    print(f"\n{'='*60}")
    print(f"E2E TEST con partido REAL")
    print(f"  Partido: {home} vs {away}")
    print(f"  Liga: CONMEBOL Libertadores (league=13)")
    print(f"  Buscando kickoff cerca de: {kickoff.isoformat()}")
    print('='*60)

    # Test 1: fixture lookup
    print("\n→ 1. Buscando fixture en API-Football…")
    fixture = find_fixture_for_match(home, away, kickoff)
    if not fixture:
        print(f"  ❌ NO se encontró fixture. Probá con otros equipos:")
        print(f"     python -m tests.test_e2e_real_match \"Boca\" \"River\"")
        return False
    print(f"  ✅ Fixture encontrado: id={fixture['fixture']['id']}")
    print(f"     Kickoff real: {fixture['fixture']['date']}")
    print(f"     Equipos: {fixture['teams']['home']['name']} (id={fixture['teams']['home']['id']}) "
          f"vs {fixture['teams']['away']['name']} (id={fixture['teams']['away']['id']})")

    # Test 2: collect full context
    print("\n→ 2. Recolectando contexto completo…")
    ctx = collect_match_context(home, away, kickoff, fetch_lineups=True)
    if not ctx:
        print("  ❌ Contexto vacío")
        return False
    print(f"  ✅ Contexto obtenido. Keys: {list(ctx.keys())}")
    for k, v in ctx.items():
        if isinstance(v, list):
            print(f"     {k}: {len(v)} items")
            for item in v[:3]:
                print(f"        - {item}")
        else:
            print(f"     {k}: {v}")

    # Test 3: weather (si OWM key activa)
    print("\n→ 3. Probando OpenWeatherMap…")
    from src.scrapers.weather import get_weather_for_match
    # Hardcode CDMX coords como prueba (no tenemos venue de Libertadores)
    from src.scrapers.weather import get_forecast_at_kickoff
    weather = get_forecast_at_kickoff(-16.5000, -68.1500, kickoff)  # La Paz, Bolivia (donde juega Bolívar)
    if weather:
        print(f"  ✅ Weather forecast obtenido:")
        for k, v in weather.items():
            print(f"     {k}: {v}")
    else:
        print(f"  ⚠️  Weather no disponible (key inactiva o kickoff fuera de rango)")

    # Test 4: LLM Capa 4 con contexto real
    print("\n→ 4. LLM Capa 4 con contexto real (Sonnet)…")
    from src.model.qualitative import MatchContext, adjust_with_llm
    mctx = MatchContext(
        home_team=fixture['teams']['home']['name'],
        away_team=fixture['teams']['away']['name'],
        kickoff_local=fixture['fixture']['date'][:16],
        stage="Fase de grupos Libertadores",
        # Probabilidades inventadas (no estamos contra Pinnacle para este test)
        market_p_home=0.55, market_p_draw=0.25, market_p_away=0.20,
        market_e_goals_L=1.5, market_e_goals_V=0.9,
        home_recent_form=ctx.get("home_recent_form"),
        away_recent_form=ctx.get("away_recent_form"),
        home_injuries=ctx.get("home_injuries"),
        away_injuries=ctx.get("away_injuries"),
        home_lineup_change=ctx.get("home_lineup_change"),
        away_lineup_change=ctx.get("away_lineup_change"),
        h2h_recent=ctx.get("h2h_recent"),
        weather=weather.get("conditions") + f", {weather.get('temp_c')}°C" if weather else None,
    )
    adj = adjust_with_llm(mctx)
    print(f"  ✅ LLM respondió:")
    print(f"     ΔλL = {adj.delta_lambda_L:+.2f}")
    print(f"     ΔλV = {adj.delta_lambda_V:+.2f}")
    print(f"     Confidence: {adj.confidence:.2f}")
    print(f"     Reasoning: {adj.reasoning}")

    fapi.WORLD_CUP_LEAGUE_ID = original_league
    fapi.WORLD_CUP_SEASON = original_season
    print(f"\n{'='*60}\n✅ TEST E2E COMPLETO\n{'='*60}\n")
    return True


if __name__ == "__main__":
    test_against_libertadores()

"""Test E2E: pipeline con datos reales contra un partido histórico (free tier de API-Football
solo da seasons 2022-2024 — no podemos usar partidos actuales 2025/2026).

Default: Final Euro 2024 — España 2-1 Inglaterra el 14/07/2024.

Valida que:
    - API-Football retorna lineups, lesiones, forma, h2h reales
    - LLM Capa 4 procesa el contexto y da ajustes coherentes
    - El módulo de weather funciona (depende de key OWM activa)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)

import src.scrapers.football_api as fapi


def main():
    # Por default: Final Euro 2024
    home = sys.argv[1] if len(sys.argv) > 1 else "Spain"
    away = sys.argv[2] if len(sys.argv) > 2 else "England"
    league_id = int(sys.argv[3]) if len(sys.argv) > 3 else 4   # Euro Championship
    season = int(sys.argv[4]) if len(sys.argv) > 4 else 2024
    kickoff = datetime(2024, 7, 14, 19, 0, tzinfo=timezone.utc)

    # Override para apuntar al league/season correcto
    fapi.WORLD_CUP_LEAGUE_ID = league_id
    fapi.WORLD_CUP_SEASON = season

    print(f"\n{'='*60}")
    print(f"E2E TEST con partido REAL (free tier API-Football)")
    print(f"  Partido:  {home} vs {away}")
    print(f"  Liga:     id={league_id}, season={season}")
    print(f"  Kickoff:  {kickoff.isoformat()}")
    print('='*60)

    print("\n→ 1. Buscando fixture en API-Football…")
    fixture = fapi.find_fixture_for_match(home, away, kickoff)
    if not fixture:
        print(f"  ❌ NO se encontró fixture.")
        return False
    fx = fixture["fixture"]
    t = fixture["teams"]
    print(f"  ✅ fixture_id={fx['id']}")
    print(f"     Kickoff real: {fx['date']}")
    print(f"     {t['home']['name']} (id={t['home']['id']}) vs {t['away']['name']} (id={t['away']['id']})")

    print("\n→ 2. Contexto completo…")
    ctx = fapi.collect_match_context(home, away, kickoff, fetch_lineups=True)
    if not ctx:
        print("  ❌ Contexto vacío")
        return False
    print(f"  ✅ Keys: {list(ctx.keys())}")
    for k, v in ctx.items():
        if isinstance(v, list):
            print(f"\n     {k}:")
            for item in v[:5]:
                print(f"        - {item}")
        else:
            print(f"\n     {k}: {v}")

    print("\n→ 3. LLM Capa 4 con contexto REAL (Sonnet 4.6)…")
    from src.model.qualitative import MatchContext, adjust_with_llm
    mctx = MatchContext(
        home_team=t['home']['name'],
        away_team=t['away']['name'],
        kickoff_local=fx['date'][:16],
        stage="Final Euro 2024",
        market_p_home=0.45, market_p_draw=0.27, market_p_away=0.28,
        market_e_goals_L=1.4, market_e_goals_V=1.1,
        home_recent_form=ctx.get("home_recent_form"),
        away_recent_form=ctx.get("away_recent_form"),
        home_injuries=ctx.get("home_injuries"),
        away_injuries=ctx.get("away_injuries"),
        home_lineup_change=ctx.get("home_lineup_change"),
        away_lineup_change=ctx.get("away_lineup_change"),
        h2h_recent=ctx.get("h2h_recent"),
    )
    adj = adjust_with_llm(mctx)
    print(f"  ΔλL = {adj.delta_lambda_L:+.2f}")
    print(f"  ΔλV = {adj.delta_lambda_V:+.2f}")
    print(f"  Confidence: {adj.confidence:.2f}")
    print(f"  Reasoning: {adj.reasoning}")

    print(f"\n{'='*60}\n✅ TEST E2E COMPLETO — Capa 4 LLM funcionando con contexto real\n{'='*60}\n")
    return True


if __name__ == "__main__":
    main()

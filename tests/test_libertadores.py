"""Test E2E con partido REAL de Libertadores (válido HOY).

Valida que ESPN devuelve data completa (squads, h2h, news, odds, standings, form)
y que el LLM Capa 4 puede usarlo para razonar con contexto rico.

NO publica nada a la penca (DRY mode hard-coded).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from src.scrapers import espn
from src.scrapers.news import collect_news_context
from src.model.qualitative import MatchContext, adjust_with_llm


def main():
    # Cambiar ESPN al league de Libertadores
    espn.set_league("conmebol.libertadores")
    print(f"\n{'=' * 70}")
    print(f"E2E TEST con Libertadores (ESPN league={espn.get_league()})")
    print('=' * 70)

    # 1. Listar próximos partidos
    print("\n→ 1. Próximos partidos:")
    events = espn.get_scoreboard()
    for e in events[:5]:
        print(f"   {e['id']:>9}  {e['name']:<55} @ {e['date'][:16]}")
    if not events:
        print("  (ninguno)")
        return False

    # 2. Elegir Bolívar vs Independ. Rivadavia
    target = next((e for e in events if "Bolívar" in e["name"] or "Bolivar" in e["name"]), None)
    if not target:
        target = events[0]
    print(f"\n→ Test contra: {target['name']} ({target['date'][:16]})")

    # 3. Detalle completo del partido
    print("\n→ 2. ESPN summary endpoint:")
    summary = espn.get_match_summary(target["id"])
    if summary:
        print(f"   keys: {list(summary.keys())}")
        gi = summary.get("gameInfo", {})
        venue = gi.get("venue", {})
        print(f"   venue: {venue.get('fullName')} ({venue.get('address', {}).get('city')})")
        odds_list = summary.get("pickcenter", [])
        if odds_list:
            p = odds_list[0]
            print(f"   odds DraftKings: spread={p.get('spread')}, OU={p.get('overUnder')}, "
                  f"ML home/away={p.get('homeTeamOdds', {}).get('moneyLine')}/"
                  f"{p.get('awayTeamOdds', {}).get('moneyLine')}")
        rosters = summary.get("rosters", [])
        print(f"   rosters: {len(rosters)} teams")
        for r in rosters:
            print(f"     {r['team']['displayName']}: {len(r.get('roster', []))} jugadores")
        news_data = summary.get("news", {})
        articles = news_data.get("articles", []) if isinstance(news_data, dict) else []
        print(f"   news: {len(articles)} artículos")
        h2h = summary.get("headToHeadGames", [])
        h2h_events = sum(len(g.get("events", [])) for g in h2h if isinstance(g, dict))
        print(f"   h2h: {h2h_events} partidos históricos")

    # 4. Standings
    print("\n→ 3. Standings actuales:")
    std = espn.get_standings()
    if std:
        for group, entries in list(std.items())[:1]:
            print(f"   {group}:")
            for e in entries[:6]:
                print(f"     {e['team']:<25} {e['w']}W-{e['d']}D-{e['l']}L  {e['points']}pts")

    # 5. Contexto unificado
    print("\n→ 4. collect_match_context_espn (lo que va al LLM):")
    teams = target["name"].split(" at ")
    away_name = teams[0]
    home_name = teams[1] if len(teams) > 1 else "Home"
    ctx = espn.collect_match_context_espn(home_name, away_name)
    for k, v in ctx.items():
        print(f"   {k}:")
        print(f"     {str(v)[:300]}")

    # 6. Google News
    print("\n→ 5. Google News context:")
    news_ctx = collect_news_context(home_name, away_name, lang="es")
    if news_ctx.get("home_news_summary"):
        print(f"   home: {news_ctx['home_news_summary'][:300]}")
    if news_ctx.get("away_news_summary"):
        print(f"   away: {news_ctx['away_news_summary'][:300]}")

    # 7. LLM Capa 4 con TODO el contexto
    print("\n→ 6. LLM Capa 4 (Sonnet) razonando con contexto real:")
    mctx = MatchContext(
        home_team=home_name,
        away_team=away_name,
        kickoff_local=target["date"][:16],
        stage="Libertadores fase eliminatoria",
        market_p_home=0.55, market_p_draw=0.25, market_p_away=0.20,
        market_e_goals_L=1.4, market_e_goals_V=1.0,
        home_recent_form=ctx.get("home_recent_form_espn"),
        away_recent_form=ctx.get("away_recent_form_espn"),
        h2h_recent=ctx.get("h2h_recent_espn"),
        motivation_notes=ctx.get("standings_context"),
        home_news_summary=(news_ctx.get("home_news_summary", "") + "\n" + (ctx.get("espn_news") or "")).strip(),
        away_news_summary=news_ctx.get("away_news_summary"),
    )
    adj = adjust_with_llm(mctx)
    print(f"   ΔλL = {adj.delta_lambda_L:+.2f}")
    print(f"   ΔλV = {adj.delta_lambda_V:+.2f}")
    print(f"   Confidence: {adj.confidence:.2f}")
    print(f"   Reasoning:\n     {adj.reasoning}")

    print(f"\n{'=' * 70}\n✅ TEST E2E LIBERTADORES COMPLETO\n{'=' * 70}\n")
    return True


if __name__ == "__main__":
    main()

"""Scraper de Euro 2024 — resultados desde Wikipedia + probabilidades pre-partido proxy desde Elo.

Genera data/backtest/euro_2024/matches.csv con:
    match_id,home,away,actual_home_score,actual_away_score,
    market_p_home,market_p_draw,market_p_away,market_p_over_2_5,market_p_btts

Las "probabilidades de mercado" se derivan de un fit Poisson con λ basadas en diferencial Elo,
porque no tenemos closing odds históricas accesibles sin scraping de OddsPortal con Playwright.

Esta aproximación NO es ideal (Elo + Poisson genérico es peor que closing odds reales del mercado),
pero sirve como sanity check de que nuestro portfolio de 5 picks gana al chalk en una muestra histórica.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/UEFA_Euro_2024"

# Elo pre-Euro 2024 (junio 2024) — desde eloratings.net snapshot histórico
# Los 24 equipos clasificados.
EURO_2024_ELO = {
    "Germany": 1949,
    "Scotland": 1727,
    "Hungary": 1745,
    "Switzerland": 1858,
    "Spain": 1995,
    "Croatia": 1893,
    "Italy": 1859,
    "Albania": 1583,
    "Slovenia": 1638,
    "Denmark": 1858,
    "Serbia": 1843,
    "England": 1976,
    "Poland": 1727,
    "Netherlands": 1925,
    "Austria": 1862,
    "France": 2020,
    "Belgium": 1933,
    "Slovakia": 1675,
    "Romania": 1668,
    "Ukraine": 1799,
    "Turkey": 1733,
    "Georgia": 1593,
    "Portugal": 1962,
    "Czech Republic": 1745,
    "Czechia": 1745,  # alias
}


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def fetch_wikipedia_results() -> list[dict]:
    """Devuelve lista de matches con {home, away, home_score, away_score}."""
    resp = httpx.get(WIKIPEDIA_URL, timeout=20.0, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    matches = []
    for box in soup.find_all("table", class_=re.compile(r"footballbox")):
        # estructura típica de Wikipedia:
        # team_home_cell  |  score (a–b)  |  team_away_cell
        rows = box.find_all("tr")
        if not rows:
            continue
        # Buscar la fila con el score
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            # Buscar un texto que parezca "2–1"
            text = " ".join(_strip(c.get_text()) for c in cells)
            score_match = re.search(r"(\d+)\s*[–\-]\s*(\d+)", text)
            if not score_match:
                continue
            home_score = int(score_match.group(1))
            away_score = int(score_match.group(2))

            # Identificar team home y team away por su posición relativa al score
            home_idx = None
            for i, c in enumerate(cells):
                if score_match.group(0).split("–")[0].strip() in _strip(c.get_text()):
                    home_idx = i
                    break
            if home_idx is None:
                continue
            # El home es el cell anterior, away el siguiente
            home_team = None
            away_team = None
            for i in range(home_idx - 1, -1, -1):
                t = _strip(cells[i].get_text())
                if t and not re.fullmatch(r"[\d\s\-–]+", t):
                    home_team = t
                    break
            for i in range(home_idx + 1, len(cells)):
                t = _strip(cells[i].get_text())
                if t and not re.fullmatch(r"[\d\s\-–:]+", t):
                    away_team = t
                    break
            if home_team and away_team:
                matches.append({
                    "home": home_team,
                    "away": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                })
            break   # ya encontramos el score, salir del loop de rows
    return matches


def elo_to_market_probs(home_elo: int, away_elo: int, home_advantage: int = 80) -> dict:
    """Convierte diferencial Elo → P(home/draw/away) usando función logística + ajuste para empate.

    Fórmula estándar: P(home win) = 1 / (1 + 10^(-(home_elo + home_adv - away_elo)/400))
    Luego adjustamos para empate: una fracción del win-prob se traslada al empate.
    """
    diff = home_elo + home_advantage - away_elo
    p_home_no_draw = 1.0 / (1.0 + 10 ** (-diff / 400))
    p_away_no_draw = 1.0 - p_home_no_draw

    # Modelo de empate: P(draw) aumenta cuando los equipos están parejos
    # Empíricamente: ~26% en partidos parejos, ~10% en partidos lopsided
    closeness = 1.0 - abs(p_home_no_draw - 0.5) * 2   # 1.0 si parejos, 0.0 si lopsided
    p_draw = 0.10 + 0.20 * closeness

    # Redistribuir el restante (1 - p_draw) proporcionalmente
    factor = (1.0 - p_draw)
    p_home = p_home_no_draw * factor
    p_away = p_away_no_draw * factor

    # Estimación de E[goles totales] vía Elo:
    # mediana ~2.5 goles, ajustar por strength promedio
    avg_strength = (home_elo + away_elo) / 2
    base_goals = 2.5
    # Equipos más fuertes en promedio marcan más
    e_total = base_goals + (avg_strength - 1800) / 200

    # P(over 2.5): usar Poisson con λ_total = e_total
    from math import exp
    # P(over 2.5) = 1 - P(0) - P(1) - P(2) bajo Poisson(e_total)
    p_o25 = 1.0 - sum(exp(-e_total) * e_total ** k / _factorial(k) for k in range(3))

    # P(BTTS): aproximar como producto de P(home marca al menos 1) * P(away marca al menos 1)
    # Distribuyendo e_total proporcionalmente
    e_home_goals = e_total * (p_home + p_draw / 2)
    e_away_goals = e_total * (p_away + p_draw / 2)
    p_home_scores = 1.0 - exp(-e_home_goals)
    p_away_scores = 1.0 - exp(-e_away_goals)
    p_btts = p_home_scores * p_away_scores

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over_2_5": max(0.0, min(1.0, p_o25)),
        "p_btts": max(0.0, min(1.0, p_btts)),
    }


def _factorial(n: int) -> int:
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


def build_csv(output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Fetching Wikipedia Euro 2024 results…")
    matches_raw = fetch_wikipedia_results()
    log.info("Got %d matches from Wikipedia", len(matches_raw))

    rows = []
    missing_elo = set()
    for i, m in enumerate(matches_raw, 1):
        home_elo = EURO_2024_ELO.get(m["home"])
        away_elo = EURO_2024_ELO.get(m["away"])
        if home_elo is None or away_elo is None:
            missing_elo.add(m["home"] if home_elo is None else m["away"])
            continue
        probs = elo_to_market_probs(home_elo, away_elo)
        rows.append({
            "match_id": f"EURO24_{i:02d}",
            "home": m["home"],
            "away": m["away"],
            "actual_home_score": m["home_score"],
            "actual_away_score": m["away_score"],
            "market_p_home": round(probs["p_home"], 4),
            "market_p_draw": round(probs["p_draw"], 4),
            "market_p_away": round(probs["p_away"], 4),
            "market_p_over_2_5": round(probs["p_over_2_5"], 4),
            "market_p_btts": round(probs["p_btts"], 4),
        })

    if missing_elo:
        log.warning("Equipos sin Elo (skipeados): %s", missing_elo)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d matches to %s", len(rows), output_path)
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = Path("data/backtest/euro_2024/matches.csv")
    n = build_csv(out)
    print(f"\n✅ {n} partidos escritos a {out}")

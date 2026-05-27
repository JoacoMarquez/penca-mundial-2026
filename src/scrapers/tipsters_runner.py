"""Orquestador de tipsters: discover URL → fetch → extract con LLM → agregar consensus.

Cada tipster tiene su propia función `discover_url(home, away, kickoff_utc)` que devuelve
la URL del preview/pronóstico para ese partido específico (o None si no se encuentra).

Después usamos el módulo `tipster_signal.py` para fetchear el artículo, extraer la pick
con LLM, y agregar todas las picks en un consensus.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Callable

import httpx
from bs4 import BeautifulSoup

from src.model.tipster_signal import (
    TipsterArticle,
    aggregate_picks,
    extract_pick,
    fetch_article,
)

log = logging.getLogger(__name__)


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _slug(s: str) -> str:
    """Normaliza nombre de equipo para usar en URLs: lowercase, sin acentos, espacios → guiones."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s-]", "", s.lower())
    return re.sub(r"\s+", "-", s.strip())


# ============ Forebet ============

def forebet_discover_url(home: str, away: str, kickoff_utc: datetime) -> str | None:
    """Busca el URL del preview de Forebet para este partido.

    Estrategia: scrape de la página de tips del Mundial 2026 y buscar el link al partido.
    """
    try:
        # Página de la liga (Mundial 2026)
        index_url = "https://www.forebet.com/en/football-tips-and-predictions-for-leagues/north-and-central-america-fifa-world-cup-2026"
        with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(index_url, follow_redirects=True)
            if r.status_code != 200:
                return None
        soup = BeautifulSoup(r.text, "lxml")

        home_s = _slug(home)
        away_s = _slug(away)
        # Buscar links que contengan ambos slugs
        for a in soup.find_all("a", href=True):
            href = a["href"].lower()
            if home_s in href and away_s in href:
                return _absolutize(href, "https://www.forebet.com")
        return None
    except Exception as e:
        log.warning("forebet discover falló: %s", e)
        return None


# ============ The Analyst / Opta ============

def opta_discover_url(home: str, away: str, kickoff_utc: datetime) -> str | None:
    """The Analyst (Opta) publica previews por partido. Intento via search interno."""
    try:
        q = f"{home} vs {away}"
        search_url = f"https://theanalyst.com/?s={httpx.QueryParams({'s': q})['s']}"
        with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(search_url, follow_redirects=True)
            if r.status_code != 200:
                return None
        soup = BeautifulSoup(r.text, "lxml")

        home_low = home.lower()
        away_low = away.lower()
        # Primer link de la sección de resultados que mencione ambos equipos
        for a in soup.find_all("a", href=True):
            text = a.get_text(" ", strip=True).lower()
            if home_low in text and away_low in text and "/" in a["href"]:
                return _absolutize(a["href"], "https://theanalyst.com")
        return None
    except Exception as e:
        log.warning("opta discover falló: %s", e)
        return None


# ============ Marca (España) ============

def marca_discover_url(home: str, away: str, kickoff_utc: datetime) -> str | None:
    """Marca usa URLs tipo /futbol/mundial/{año}/.../partido. Intento via search."""
    try:
        q = f"{home} {away} mundial 2026"
        search_url = "https://www.marca.com/buscar.html"
        with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(search_url, params={"q": q}, follow_redirects=True)
            if r.status_code != 200:
                return None
        soup = BeautifulSoup(r.text, "lxml")
        home_low = home.lower()
        away_low = away.lower()
        for a in soup.find_all("a", href=True):
            text = a.get_text(" ", strip=True).lower()
            if home_low in text and away_low in text and "mundial" in text:
                return _absolutize(a["href"], "https://www.marca.com")
        return None
    except Exception as e:
        log.warning("marca discover falló: %s", e)
        return None


# ============ Registry ============

TIPSTER_DISCOVERERS: dict[str, tuple[Callable, float, str]] = {
    # name: (discover_fn, weight, lang)
    "forebet":  (forebet_discover_url, 0.8, "en"),
    "opta":     (opta_discover_url, 1.2, "en"),
    "marca":    (marca_discover_url, 0.8, "es"),
}


# ============ Orquestación ============

def collect_tipster_consensus(
    home: str,
    away: str,
    kickoff_utc: datetime,
    market_p_home: float | None = None,
    market_p_away: float | None = None,
) -> dict:
    """Workflow completo: para cada tipster registrado, descubre URL → fetch → LLM extract.

    Retorna un dict serializable con:
        - n_tipsters_picked
        - consensus (p_1, p_X, p_2)
        - info_edge_vs_market
        - by_source: lista de picks individuales
        - top_factors
    """
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.info("No ANTHROPIC_API_KEY — skip tipsters")
        return {"n_tipsters_picked": 0, "consensus": None, "by_source": []}

    picks = []
    for name, (discover_fn, weight, _lang) in TIPSTER_DISCOVERERS.items():
        try:
            url = discover_fn(home, away, kickoff_utc)
            if not url:
                log.info("%s: no se encontró URL para %s vs %s", name, home, away)
                continue
            log.info("%s URL: %s", name, url)
            article = fetch_article(url, source=name, weight=weight)
            if not article or len(article.body) < 200:
                log.warning("%s: artículo muy corto o vacío", name)
                continue
            pick = extract_pick(home, away, article)
            log.info("%s pick: %s conf=%.2f", name, pick.pick, pick.confidence)
            picks.append(pick)
        except Exception as e:
            log.exception("%s falló: %s", name, e)

    if not picks:
        return {"n_tipsters_picked": 0, "consensus": None, "by_source": []}

    consensus = aggregate_picks(picks, market_p_home=market_p_home, market_p_away=market_p_away)
    return {
        "n_tipsters_picked": consensus.n_tipsters,
        "consensus_p_1": round(consensus.p_pick_1, 3),
        "consensus_p_X": round(consensus.p_pick_X, 3),
        "consensus_p_2": round(consensus.p_pick_2, 3),
        "weighted_confidence": round(consensus.weighted_confidence, 3),
        "info_edge_vs_market": consensus.info_edge_vs_market,
        "by_source": [
            {
                "source": p.source,
                "pick": p.pick,
                "confidence": p.confidence,
                "predicted_score": list(p.predicted_score) if p.predicted_score else None,
                "factors": p.factors,
                "reasoning_summary": p.reasoning_summary,
            }
            for p in picks
        ],
        "top_factors": consensus.most_mentioned_factors,
    }


def _absolutize(href: str, base: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return base + href
    return f"{base}/{href}"

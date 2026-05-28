"""Búsqueda de noticias vía Google News RSS — alternativa GRATIS a API-Football
para contexto de partidos.

Google News RSS:
    - Sin auth, sin anti-bot, ilimitado
    - Búsquedas custom: ?q=<query>&hl=<lang>&gl=<country>
    - Retorna 100 items por búsqueda

Estrategia:
    1. Por cada equipo, query "{team} world cup 2026 {keyword}" donde keyword
       rota entre: 'lineup', 'injury', 'team news'.
    2. LLM Capa 4 toma titulares + descripciones y extrae contexto estructurado
       (lesiones, alineación probable, etc).

NOTA: el contenido completo de cada artículo está detrás del paywall de cada medio,
pero los TITULARES + DESCRIPTIONS del RSS suelen ser suficientes para el LLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from xml.etree import ElementTree as ET

import httpx

log = logging.getLogger(__name__)


GNEWS_RSS = "https://news.google.com/rss/search"


@dataclass(frozen=True)
class NewsItem:
    title: str
    description: str
    pub_date: str
    source: str
    link: str


def _decode_html(text: str) -> str:
    """Limpia HTML/entities básicas que aparecen en RSS."""
    import html
    text = html.unescape(text)
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def search_news(query: str, max_items: int = 20, hl: str = "en-US", gl: str = "US") -> list[NewsItem]:
    """Búsqueda en Google News RSS. Retorna lista de NewsItem."""
    params = {"q": query, "hl": hl, "gl": gl, "ceid": f"{gl}:{hl.split('-')[0]}"}
    try:
        with httpx.Client(timeout=10.0, headers={"User-Agent": "PencaMundial/1.0"}) as c:
            r = c.get(GNEWS_RSS, params=params, follow_redirects=True)
        if r.status_code != 200:
            log.warning("Google News %d: %s", r.status_code, r.text[:200])
            return []
        root = ET.fromstring(r.content)
        items = []
        for item in root.iter("item")[:max_items] if hasattr(root.iter("item"), "__getitem__") else list(root.iter("item"))[:max_items]:
            title = item.findtext("title", "")
            desc = _decode_html(item.findtext("description", ""))
            pub = item.findtext("pubDate", "")
            src_el = item.find("source")
            src = src_el.text if src_el is not None else ""
            link = item.findtext("link", "")
            items.append(NewsItem(title=title, description=desc, pub_date=pub, source=src or "Unknown", link=link))
        return items
    except Exception as e:
        log.warning("search_news falló: %s", e)
        return []


def _team_in_title(team_name: str, title: str) -> bool:
    """True si el nombre del equipo (o variantes) aparece en el título."""
    import unicodedata
    def norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        return "".join(c for c in s if not unicodedata.combining(c)).lower()
    n_team = norm(team_name)
    n_title = norm(title)
    # Variantes comunes
    variants = {n_team}
    if " " in n_team:
        # ej "corea del sur" → también "corea"
        variants.add(n_team.split()[0])
    return any(v in n_title for v in variants if len(v) >= 4)


def fetch_team_news(team_name: str, max_items: int = 10, lang: str = "es") -> list[NewsItem]:
    """Búsqueda específica + FILTRO por team_name en el título (descarta artículos genéricos)."""
    # Buscar en ambos idiomas para equipos pequeños donde puede no haber prensa en español
    queries_multi = [
        # Español
        (f"{team_name} mundial 2026 lesionados", "es-419", "AR"),
        (f"{team_name} mundial 2026 alineación", "es-419", "AR"),
        # Inglés
        (f"{team_name} world cup 2026 injury", "en-US", "US"),
        (f"{team_name} world cup 2026 lineup", "en-US", "US"),
    ]
    all_items: list[NewsItem] = []
    seen_titles = set()
    for q, hl, gl in queries_multi:
        items = search_news(q, max_items=max_items, hl=hl, gl=gl)
        for it in items:
            # Filtro: el equipo debe estar en el título
            if not _team_in_title(team_name, it.title):
                continue
            key = it.title[:80].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                all_items.append(it)
            if len(all_items) >= max_items:
                break
        if len(all_items) >= max_items:
            break
    return all_items[:max_items]


def collect_news_context(home_name: str, away_name: str, lang: str = "es") -> dict[str, Any]:
    """Junta noticias para ambos equipos. Retorna dict compatible con MatchContext."""
    home_news = fetch_team_news(home_name, max_items=8, lang=lang)
    away_news = fetch_team_news(away_name, max_items=8, lang=lang)

    def render(items: list[NewsItem]) -> str:
        lines = []
        for it in items[:6]:
            ts = it.pub_date[:25] if it.pub_date else ""
            lines.append(f"[{it.source}] {it.title}")
            if it.description and len(it.description) > 10:
                lines.append(f"  → {it.description[:200]}")
        return "\n".join(lines)

    out: dict[str, Any] = {}
    if home_news:
        out["home_news_summary"] = render(home_news)
    if away_news:
        out["away_news_summary"] = render(away_news)
    return out

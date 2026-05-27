"""Capa 3 del modelo: extracción de picks/predicciones de tipsters con LLM.

Flujo:
    1. Para cada partido, identificar URLs de artículos de tipsters relevantes (config/tipsters.yaml).
    2. Fetchear cada artículo (httpx + bs4).
    3. Para cada uno, pedirle a Claude Sonnet 4.6 que extraiga JSON estructurado:
       {pick: "1"|"X"|"2", confidence: 0..1, predicted_score: [gL,gV]|null, factors: [str], reasoning_summary: str}
    4. Agregar consensus + dispersión.
    5. Si el consensus difiere del mercado en > N pp → flag "info edge candidate".

Costo: ~$0.02 por partido (5-10 artículos × ~1500 tokens × Sonnet). $2-5 total Mundial.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

import httpx

log = logging.getLogger(__name__)


# ============ schemas ============

@dataclass(frozen=True)
class TipsterArticle:
    """Artículo crudo de un tipster antes del LLM."""
    source: str           # slug del tipster: 'forebet' | 'marca' | ...
    weight: float         # de tipsters.yaml
    url: str
    title: str | None
    body: str             # texto plano (post-strip de HTML)


@dataclass(frozen=True)
class TipsterPick:
    """Pick estructurada extraída por el LLM."""
    source: str
    weight: float
    pick: str | None              # "1" | "X" | "2" | None
    confidence: float             # 0..1
    predicted_score: tuple[int, int] | None
    factors: list[str]
    reasoning_summary: str
    raw_response: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TipsterConsensus:
    """Agregación de picks sobre un partido."""
    n_tipsters: int
    p_pick_1: float
    p_pick_X: float
    p_pick_2: float
    weighted_confidence: float
    info_edge_vs_market: dict | None     # { "market_p_home": ..., "tipster_p_home": ..., "delta_pp": ... }
    by_source: list[TipsterPick]
    most_mentioned_factors: list[str]


# ============ prompt LLM ============

SYSTEM_PROMPT = """Extraés predicciones de pronosticadores de fútbol y las devolvés como JSON estructurado.

Recibís un artículo periodístico/blog sobre un partido específico. Tu trabajo es:
1. Identificar si el autor da una pick clara para el partido (ganador o marcador).
2. Estimar su confianza (en base al lenguaje: "favorito claro" = alta, "podría dar el batacazo" = baja).
3. Listar los 2-4 factores principales que cita (lesiones, forma, h2h, motivación, etc.).
4. Devolver un resumen de 1 oración del razonamiento.

Si el artículo NO da pick clara (es un preview neutral sin opinión), devolvé pick=null y confidence=0.

SCHEMA OBLIGATORIO:
{
  "pick": "1" | "X" | "2" | null,
  "confidence": <float 0..1>,
  "predicted_score": [gL, gV] | null,
  "factors": [<string>, ...],
  "reasoning_summary": "<1 oración>"
}

Devolvé SOLO el JSON, sin texto adicional ni code fences."""


def build_extraction_prompt(home_team: str, away_team: str, article: TipsterArticle) -> str:
    body = article.body[:6000]   # cap a 6000 chars (~1500 tokens) para controlar costo
    return f"""PARTIDO: {home_team} (local) vs {away_team} (visitante)
FUENTE: {article.source}
TÍTULO: {article.title or '(sin título)'}

ARTÍCULO:
{body}

---
Extraé la pick del autor del artículo en JSON con el schema indicado."""


# ============ extracción LLM ============

def extract_pick(
    home_team: str,
    away_team: str,
    article: TipsterArticle,
    model: str = "claude-sonnet-4-6",
    api_key: str | None = None,
) -> TipsterPick:
    """Llama a Claude y extrae la pick. Robusto a JSON inválido."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=model,
        max_tokens=400,
        temperature=0.1,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": build_extraction_prompt(home_team, away_team, article)}],
    )
    text = response.content[0].text   # type: ignore[union-attr]
    parsed = _safe_parse_json(text)

    pred_score = parsed.get("predicted_score")
    if pred_score and len(pred_score) == 2:
        try:
            pred_score = (int(pred_score[0]), int(pred_score[1]))
        except (ValueError, TypeError):
            pred_score = None
    else:
        pred_score = None

    return TipsterPick(
        source=article.source,
        weight=article.weight,
        pick=parsed.get("pick") if parsed.get("pick") in ("1", "X", "2", None) else None,
        confidence=float(parsed.get("confidence", 0.0)),
        predicted_score=pred_score,
        factors=list(parsed.get("factors", []))[:5],
        reasoning_summary=str(parsed.get("reasoning_summary", "")),
        raw_response={"text": text, "model": model},
    )


def _safe_parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.startswith("```")]
        text = "\n".join(lines)
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1:
        return {}
    try:
        return json.loads(text[s : e + 1])
    except json.JSONDecodeError:
        return {}


# ============ agregación a consensus ============

def aggregate_picks(
    picks: list[TipsterPick],
    market_p_home: float | None = None,
    market_p_away: float | None = None,
    info_edge_threshold_pp: float = 8.0,
) -> TipsterConsensus:
    """Agrega picks de múltiples tipsters en un consensus ponderado por weight*confidence."""
    valid = [p for p in picks if p.pick in ("1", "X", "2")]
    if not valid:
        return TipsterConsensus(
            n_tipsters=0,
            p_pick_1=0.0, p_pick_X=0.0, p_pick_2=0.0,
            weighted_confidence=0.0,
            info_edge_vs_market=None,
            by_source=picks,
            most_mentioned_factors=[],
        )

    total_w = sum(p.weight * p.confidence for p in valid) or 1.0
    p_1 = sum(p.weight * p.confidence for p in valid if p.pick == "1") / total_w
    p_X = sum(p.weight * p.confidence for p in valid if p.pick == "X") / total_w
    p_2 = sum(p.weight * p.confidence for p in valid if p.pick == "2") / total_w

    # Info edge: tipster consensus difiere del mercado?
    info_edge = None
    if market_p_home is not None:
        delta_pp = (p_1 - market_p_home) * 100
        if abs(delta_pp) >= info_edge_threshold_pp:
            info_edge = {
                "market_p_home": market_p_home,
                "tipster_p_home": p_1,
                "delta_pp": delta_pp,
                "direction": "tipsters más altos en local" if delta_pp > 0 else "tipsters más bajos en local",
            }

    # Top factores mencionados (por frecuencia, no ponderado)
    from collections import Counter
    fac_counter: Counter[str] = Counter()
    for p in valid:
        for f in p.factors:
            fac_counter[f.strip().lower()] += 1
    top_factors = [f for f, _ in fac_counter.most_common(5)]

    weighted_confidence = sum(p.weight * p.confidence for p in valid) / sum(p.weight for p in valid)

    return TipsterConsensus(
        n_tipsters=len(valid),
        p_pick_1=p_1, p_pick_X=p_X, p_pick_2=p_2,
        weighted_confidence=weighted_confidence,
        info_edge_vs_market=info_edge,
        by_source=picks,
        most_mentioned_factors=top_factors,
    )


# ============ fetcher de artículos ============

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def fetch_article(url: str, source: str, weight: float, timeout: float = 15.0) -> TipsterArticle | None:
    """Trae un artículo de un tipster y lo devuelve como texto plano (sin HTML).

    Scraping naive — para sites JS-heavy hay que usar Playwright (TODO).
    """
    from bs4 import BeautifulSoup
    try:
        with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("fetch falló para %s: %s", url, e)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    # Sacar nav/footer/sidebar, quedarse con el <article> o el body
    for sel in ("nav", "footer", "aside", "header", "script", "style"):
        for el in soup.find_all(sel):
            el.decompose()

    article_el = soup.find("article") or soup.find("main") or soup.body
    body_text = article_el.get_text(separator="\n", strip=True) if article_el else ""
    title = soup.title.string if soup.title else None

    return TipsterArticle(
        source=source,
        weight=weight,
        url=url,
        title=title,
        body=body_text,
    )


# ============ CLI ============

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser(description="Smoke test de Capa 3 (LLM tipster extraction)")
    ap.add_argument("--url", required=True, help="URL del artículo de prueba")
    ap.add_argument("--source", default="manual_test")
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--home", default="Uruguay")
    ap.add_argument("--away", default="Spain")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    article = fetch_article(args.url, args.source, args.weight)
    if not article:
        raise SystemExit("No se pudo bajar el artículo")
    print(f"=== ARTÍCULO ({len(article.body)} chars) ===")
    print(article.body[:500])
    print("...")

    if not args.no_llm:
        print("\n=== EXTRACCIÓN LLM ===")
        pick = extract_pick(args.home, args.away, article)
        print(json.dumps({
            "pick": pick.pick,
            "confidence": pick.confidence,
            "predicted_score": pick.predicted_score,
            "factors": pick.factors,
            "reasoning_summary": pick.reasoning_summary,
        }, indent=2, ensure_ascii=False))

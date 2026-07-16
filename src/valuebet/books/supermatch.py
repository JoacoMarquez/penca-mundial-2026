"""Scraper de Supermatch (supermatch.com.uy) — soft book objetivo.

FASE 0 RESUELTA: el front consume un Elasticsearch PÚBLICO en
`elastic-frontend.supermatch.com.uy` que responde a httpx directo SIN cookies ni
Cloudflare (de hecho mandar cookies de sesión da 403 — hay que ir "limpio"). No se
necesita Playwright: un solo POST `_search` trae los eventos con `betLines` embebido.

Taxonomía de mercados (campo betLines[].type / .description):
    3w    "1x2"                       → fútbol H/empate/A   (options idext 1/2/3)
    ft2w  "Ganador (incl. prórroga)"  → básquet/tenis moneyline 2-way (incluye OT,
                                         igual que el moneyline de Pinnacle)
    to    "Total ( 2.5 )" / "Total (incl. prórroga) ( 179.5 )" → over/under
Cada option: {result, dividend (cuota decimal), idext}. dividend es la cuota.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

import httpx

from src.valuebet.config import VBConfig
from src.valuebet.types import OddsQuote

log = logging.getLogger(__name__)

ES_BASE = "https://elastic-frontend.supermatch.com.uy"
SEARCH_PATH = "/elasticsearch_full_prematch_events_index/_search"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Content-Type": "application/json",
    "Origin": "https://www.supermatch.com.uy",
    "Referer": "https://www.supermatch.com.uy/",
    "Accept-Language": "es-UY,es;q=0.9",
}

# nombre del deporte en Supermatch (sportName.keyword) por clave interna
SPORT_NAMES = {"soccer": "Fútbol", "basketball": "Baloncesto", "tennis": "Tenis"}

_TOTAL_RE = re.compile(r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*$")
LOOKAHEAD_MS = 3 * 24 * 3600 * 1000  # 3 días


def _fetch_events(sport_name: str, now_ms: int) -> list[dict]:
    query = {
        "query": {"bool": {"must": [
            {"range": {"dateTime": {"gte": now_ms, "lte": now_ms + LOOKAHEAD_MS}}},
            {"term": {"sportName.keyword": sport_name}},
        ]}},
        "size": 200,
        "_source": ["description", "sportName", "sportId", "leagueName", "leagueId",
                    "dateTime", "betLines"],
    }
    with httpx.Client(timeout=25.0, headers=HEADERS) as c:
        r = c.post(f"{ES_BASE}{SEARCH_PATH}", json=query)
        r.raise_for_status()
        return r.json()["hits"]["hits"]


def parse_events(sport: str, hits: list[dict]) -> list[OddsQuote]:
    """Hits de Elasticsearch → OddsQuote. Función pura (testeable con fixture)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    out: list[OddsQuote] = []

    for hit in hits:
        s = hit.get("_source", {})
        event_id = f"sm:{hit.get('_id')}"
        event_name = _clean_name(s.get("description", ""))
        league = s.get("leagueName", "")
        dt_ms = s.get("dateTime")
        if not dt_ms or " vs " not in event_name:
            continue
        start_utc = datetime.fromtimestamp(dt_ms / 1000, tz=timezone.utc).isoformat()
        home, away = [t.strip() for t in event_name.split(" vs ", 1)]

        for line in s.get("betLines", []):
            market, mapping = _map_line(sport, line, home, away)
            if not market:
                continue
            for outcome, odds in mapping.items():
                if odds and odds > 1.0:
                    out.append(OddsQuote(
                        book="supermatch", sport=sport, league=league,
                        event_id=event_id, event_name=f"{home} vs {away}",
                        start_utc=start_utc, market=market, outcome=outcome,
                        decimal_odds=float(odds), fetched_utc=now_iso,
                    ))
    return out


def _map_line(sport: str, line: dict, home: str, away: str) -> tuple[str | None, dict]:
    ltype = line.get("type")
    desc = line.get("description", "") or ""
    opts = line.get("options") or []

    if ltype == "3w" and desc == "1x2":
        by_idext = {o.get("idext"): o.get("dividend") for o in opts}
        return "1x2", {"home": by_idext.get("1"), "draw": by_idext.get("2"),
                       "away": by_idext.get("3")}

    if ltype == "ft2w":  # moneyline 2-way (incl. prórroga)
        m: dict[str, float] = {}
        for o in opts:
            res = _clean_name(o.get("result", ""))
            if res == _clean_name(home):
                m["home"] = o.get("dividend")
            elif res == _clean_name(away):
                m["away"] = o.get("dividend")
        return ("moneyline", m) if len(m) == 2 else (None, {})

    if ltype == "to" and desc.startswith("Total"):
        pts = _TOTAL_RE.search(desc)
        if not pts:
            return None, {}
        market = f"total_{pts.group(1)}"
        m = {}
        for o in opts:
            res = (o.get("result") or "").lower()
            if res.startswith(("más", "mas", "over")):
                m["over"] = o.get("dividend")
            elif res.startswith(("menos", "under")):
                m["under"] = o.get("dividend")
        return (market, m) if len(m) == 2 else (None, {})

    return None, {}


def _clean_name(s: str) -> str:
    return " ".join(s.split()).strip()


def fetch_quotes(cfg: VBConfig) -> list[OddsQuote]:
    """Cuotas de Supermatch para los deportes configurados. [] con warning ante fallo."""
    now_ms = int(time.time() * 1000)
    out: list[OddsQuote] = []
    for sport in cfg.sports:
        sport_name = SPORT_NAMES.get(sport)
        if not sport_name:
            continue
        try:
            hits = _fetch_events(sport_name, now_ms)
            out.extend(parse_events(sport, hits))
        except Exception as e:
            log.warning("supermatch %s falló: %s", sport, e)
    log.info("supermatch: %d cuotas", len(out))
    return out


if __name__ == "__main__":
    from src.valuebet import config as vbconfig
    logging.basicConfig(level=logging.INFO)
    quotes = fetch_quotes(vbconfig.load())
    for q in quotes[:20]:
        print(f"  {q.sport:10} {q.event_name[:35]:35} {q.market:12} {q.outcome:6} {q.decimal_odds}")

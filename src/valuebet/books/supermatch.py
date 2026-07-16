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
from src.valuebet.types import OddsQuote, total_market

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
PAGE_SIZE = 200
MAX_EVENTS = 2000  # tope de cordura; ES limita from+size a 10k por default


def _fetch_events(sport_name: str, now_ms: int) -> list[dict]:
    """Todos los eventos del deporte en la ventana, paginando con from/size.

    Sin paginación ni sort, ES devuelve 200 hits en orden arbitrario y el resto se
    pierde en silencio (fútbol con 3 días de lookahead supera los 200 eventos).
    """
    hits: list[dict] = []
    with httpx.Client(timeout=25.0, headers=HEADERS) as c:
        while len(hits) < MAX_EVENTS:
            query = {
                "query": {"bool": {"must": [
                    {"range": {"dateTime": {"gte": now_ms, "lte": now_ms + LOOKAHEAD_MS}}},
                    {"term": {"sportName.keyword": sport_name}},
                ]}},
                "sort": [{"dateTime": "asc"}],
                "from": len(hits),
                "size": PAGE_SIZE,
                "_source": ["description", "sportName", "sportId", "leagueName", "leagueId",
                            "dateTime", "betLines"],
            }
            r = c.post(f"{ES_BASE}{SEARCH_PATH}", json=query)
            r.raise_for_status()
            page = r.json()["hits"]["hits"]
            hits.extend(page)
            if len(page) < PAGE_SIZE:
                break
    if len(hits) >= MAX_EVENTS:
        log.warning("supermatch %s: tope de %d eventos alcanzado — puede haber truncamiento",
                    sport_name, MAX_EVENTS)
    return hits


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

    if ltype == "to":
        pts = _TOTAL_RE.search(desc)
        if not pts:
            return None, {}
        # WHITELIST del prefijo: Supermatch publica varios "Total ..." por evento
        # (goles, córners, tarjetas) con el mismo type "to". Solo el total del
        # RESULTADO se compara contra Pinnacle — "Total de córners ( 9.5 )" contra
        # un total de goles daría un edge fantasma (análogo soft del bug bookings).
        prefix = desc[:pts.start()].strip()
        if prefix not in ("Total", "Total (incl. prórroga)"):
            return None, {}
        market = total_market(pts.group(1))
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

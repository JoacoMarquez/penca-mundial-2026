"""Odds de la Primera División uruguaya desde el Elasticsearch público de Supermatch.

Mismo endpoint que src/valuebet/books/supermatch.py, pero con parseo propio porque
acá necesitamos más mercados que 1X2/totales: Supermatch publica para la liga local

    3w   '1x2'
    to   'Total ( N.5 )'
    ftnw 'Marcador exacto'          → distribución de scores directa (¡de-vig y listo!)
    ftnw '{Equipo} Goles exactos'   → marginales de goles por equipo (0/1/2/3+)
    ftnw 'Margen de victoria'       → diferencia de gol

verificado 2026-08-04 con Peñarol vs Wanderers. El 'Marcador exacto' de la propia
casa que organiza la penca es la Capa 1 ideal para este scoring (aditivo sobre
exacto + goles por lado): constriñe la grilla completa, no solo el 1X2.

El de-vig del marcador exacto va con Shin (favoritos fuertes, mercado de colas);
1X2 y totales con proportional — mismos criterios que config/books.yaml.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

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

LEAGUE_NAME = "Uruguay"
CHAMPIONSHIP = "Primera División"
LOOKAHEAD_MS = 8 * 24 * 3600 * 1000  # una fecha entera + margen

_TOTAL_RE = re.compile(r"^Total \(\s*([0-9]+\.5)\s*\)$")
_SCORE_RE = re.compile(r"^(\d+):(\d+)$")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


@dataclass
class EventOdds:
    """Odds crudas (dividendos decimales) de un partido. Sin de-vig — eso es Capa 1."""
    event_id: str
    home: str
    away: str
    start_utc: str
    fetched_utc: str
    x1x2: dict[str, float] = field(default_factory=dict)          # home/draw/away
    totals: dict[str, dict[str, float]] = field(default_factory=dict)   # "2.5" → over/under
    correct_score: dict[str, float] = field(default_factory=dict)  # "2:0" → odds, + "otro"
    home_goals: dict[str, float] = field(default_factory=dict)     # "0"/"1"/"2"/"3+" → odds
    away_goals: dict[str, float] = field(default_factory=dict)
    margin: dict[str, float] = field(default_factory=dict)         # "H+1"/"H+2"/"H+3"/"D"/"A+1"...


def fetch_primera_events(lookahead_ms: int = LOOKAHEAD_MS) -> list[dict]:
    """Hits crudos del ES para la Primera uruguaya en la ventana."""
    now_ms = int(time.time() * 1000)
    query = {
        "query": {"bool": {"must": [
            {"range": {"dateTime": {"gte": now_ms, "lte": now_ms + lookahead_ms}}},
            {"term": {"leagueName.keyword": LEAGUE_NAME}},
            {"term": {"championshipName.keyword": CHAMPIONSHIP}},
        ]}},
        "sort": [{"dateTime": "asc"}],
        "size": 60,
        "_source": ["description", "leagueName", "championshipName", "dateTime", "betLines"],
    }
    with httpx.Client(timeout=25.0, headers=HEADERS) as c:
        r = c.post(f"{ES_BASE}{SEARCH_PATH}", json=query)
        r.raise_for_status()
        return r.json()["hits"]["hits"]


def parse_event(hit: dict) -> EventOdds | None:
    """Hit de ES → EventOdds. Función pura (testeable con fixture)."""
    s = hit.get("_source", {})
    name = s.get("description", "")
    if " vs " not in name:
        return None
    home, away = [t.strip() for t in name.split(" vs ", 1)]
    dt_ms = s.get("dateTime")
    if not dt_ms:
        return None

    ev = EventOdds(
        event_id=f"sm:{hit.get('_id')}",
        home=home,
        away=away,
        start_utc=datetime.fromtimestamp(dt_ms / 1000, tz=timezone.utc).isoformat(),
        fetched_utc=datetime.now(timezone.utc).isoformat(),
    )

    for line in s.get("betLines", []):
        desc = (line.get("description") or "").strip()
        ltype = line.get("type")
        opts = line.get("options") or []
        odds = {(o.get("result") or "").strip(): o.get("dividend") for o in opts}
        odds = {k: float(v) for k, v in odds.items() if v and v > 1.0}

        if ltype == "3w" and desc == "1x2":
            ev.x1x2 = _map_hda(odds, home, away)
        elif ltype == "to" and (m := _TOTAL_RE.match(desc)):
            line_pt = m.group(1)
            ou = {}
            for k, v in odds.items():
                if k.startswith("más de"):
                    ou["over"] = v
                elif k.startswith("menos de"):
                    ou["under"] = v
            if len(ou) == 2:
                ev.totals[line_pt] = ou
        elif ltype == "ftnw" and desc == "Marcador exacto":
            for k, v in odds.items():
                if _SCORE_RE.match(k) or k == "otro":
                    ev.correct_score[k] = v
        elif ltype == "ftnw" and desc == f"{home} Goles exactos":
            ev.home_goals = odds
        elif ltype == "ftnw" and desc == f"{away} Goles exactos":
            ev.away_goals = odds
        elif ltype == "ftnw" and desc == "Margen de victoria":
            ev.margin = _map_margin(odds, home, away)

    return ev


def _map_hda(odds: dict[str, float], home: str, away: str) -> dict[str, float]:
    out = {}
    for k, v in odds.items():
        nk = _norm(k)
        if nk == _norm(home):
            out["home"] = v
        elif nk == _norm(away):
            out["away"] = v
        elif nk == "empate":
            out["draw"] = v
    return out if len(out) == 3 else {}


def _map_margin(odds: dict[str, float], home: str, away: str) -> dict[str, float]:
    """'Peñarol por 2' → 'H+2', 'empate' → 'D', 'Wanderers por 3+' → 'A+3'."""
    out = {}
    for k, v in odds.items():
        nk = _norm(k)
        if nk == "empate":
            out["D"] = v
            continue
        m = re.match(r"^(.*) por (\d+)\+?$", k.strip())
        if not m:
            continue
        team, n = _norm(m.group(1)), m.group(2)
        side = "H" if team == _norm(home) else ("A" if team == _norm(away) else None)
        if side:
            out[f"{side}+{n}"] = v
    return out


def fetch_primera_odds() -> list[EventOdds]:
    """Pipeline completo: ES → EventOdds por partido de la Primera."""
    events = [parse_event(h) for h in fetch_primera_events()]
    out = [e for e in events if e is not None and e.x1x2]
    log.info("supermatch primera: %d eventos con odds", len(out))
    return out

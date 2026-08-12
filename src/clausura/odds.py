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
from pathlib import Path

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


# -------------------- persistencia --------------------
#
# Dos motivos, del análisis del 12/8:
#   1. Un 500/timeout del ES degradaba TODO a ratings puros en silencio — incluso
#      el rerun T-2h, y el gate no avisaba porque compara ambas planillas bajo el
#      mismo modelo degradado. Con cache, un outage corto usa las cuotas de la
#      corrida anterior (mejor cuota vieja que ninguna) y además GRITA.
#   2. Las cuotas crudas versionadas son el dato que faltaba para medir el blend
#      70/30 (criterio de diseño nunca medido): log-loss por fecha contra los
#      resultados reales para una grilla de pesos.

ODDS_DIR = Path(__file__).resolve().parents[2] / "data" / "odds" / "clausura"


def save_odds_snapshot(odds: list[EventOdds]) -> None:
    """Versiona las cuotas crudas de esta corrida (nunca sobreescribe, regla #2)."""
    import json
    from dataclasses import asdict

    if not odds:
        return
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ODDS_DIR / f"odds_{ts}.json"
    path.write_text(json.dumps([asdict(e) for e in odds], ensure_ascii=False),
                    encoding="utf-8")
    log.info("odds versionadas: %s (%d eventos)", path.name, len(odds))


def load_cached_odds(max_age_h: float = 24.0) -> tuple[list[EventOdds], float] | None:
    """(odds del último snapshot, edad en horas), o None si no hay o está viejo.

    24h de tope: la cuota de ayer de un partido de mañana sigue siendo mejor
    insumo que los ratings solos (blend 70/30), pero una de hace tres días puede
    ser de un mercado que ya se movió con noticias.
    """
    import json

    archivos = sorted(ODDS_DIR.glob("odds_*.json")) if ODDS_DIR.exists() else []
    if not archivos:
        return None
    ultimo = archivos[-1]
    ts = datetime.strptime(ultimo.stem.split("_")[1], "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc)
    edad_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    if edad_h > max_age_h:
        log.warning("cache de odds demasiado viejo (%.1fh > %.0fh) — no lo uso",
                    edad_h, max_age_h)
        return None
    data = json.loads(ultimo.read_text(encoding="utf-8"))
    return [EventOdds(**d) for d in data], edad_h


def mercados_perdidos(previos: list[EventOdds], actuales: list[EventOdds]) -> list[str]:
    """Partidos FUTUROS que ayer tenían 1X2 y hoy no — huele a rename del keyword
    o a un ES a medias, no a ausencia normal (el ES solo publica la fecha próxima,
    así que 'nunca tuvo cuota' es lo esperado; 'la tenía y la perdió' no)."""
    now = datetime.now(timezone.utc)
    vivos_ahora = {(_norm(e.home), _norm(e.away)) for e in actuales if e.x1x2}
    out = []
    for e in previos:
        if not e.x1x2:
            continue
        try:
            if datetime.fromisoformat(e.start_utc) <= now:
                continue                      # ya empezó: es normal que no esté
        except ValueError:
            continue
        if (_norm(e.home), _norm(e.away)) not in vivos_ahora:
            out.append(f"{e.home} vs {e.away}")
    return out

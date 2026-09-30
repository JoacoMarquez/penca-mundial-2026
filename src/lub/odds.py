"""Cuotas de la LUB desde el Elasticsearch público de Supermatch → μ de mercado.

Mercados que usa (verificados 2026-09-29 sobre básquet de otras ligas; la LUB todavía
no estaba publicada):

    ft2w "Ganador (incl. prórroga)"                    → P(local) de-vigueada
    to   "Hándicap (incl. prórroga) ( h )"             → P(local cubre h), varias líneas
    to   "Total (incl. prórroga) ( t )"                → total esperado (para el marcador)

Con margen M ~ μ + σ·ε (ε con la forma de los residuos del modelo), cada línea da
μ = −h + σ·F⁻¹(P(cubre)); se promedian las líneas y el moneyline (h = 0). σ viene del
modelo de ratings: las cuotas fijan la media, no la dispersión.

Uso:  python -m src.lub.odds      # lista lo que hay publicado de la LUB
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field

import httpx
from scipy.stats import norm

from src.model.market_probs import devig

log = logging.getLogger(__name__)

ES = ("https://elastic-frontend.supermatch.com.uy"
      "/elasticsearch_full_prematch_events_index/_search")
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Content-Type": "application/json",
    "Origin": "https://www.supermatch.com.uy",
    "Referer": "https://www.supermatch.com.uy/",
}
LOOKAHEAD_DIAS = 10
_NUM = re.compile(r"\(\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\)\s*$")


def norm_team(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    for ruido in (" club", " bbc", " basket", " basquet", " atletico"):
        s = s.replace(ruido, " ")
    return " ".join(s.split())


@dataclass
class LineasPartido:
    local: str
    visitante: str
    inicio_ms: int
    ganador: dict[str, float] = field(default_factory=dict)             # home/away
    handicaps: list[tuple[float, float, float]] = field(default_factory=list)  # (h_local, cuota_local, cuota_visit)
    totales: list[tuple[float, float, float]] = field(default_factory=list)    # (t, over, under)

    def mu_mercado(self, sigma: float) -> float | None:
        ests: list[float] = []
        if len(self.ganador) == 2:
            p = devig(self.ganador, method="shin")["home"]
            ests.append(sigma * norm.ppf(min(max(p, 1e-4), 1 - 1e-4)))
        for h, ol, ov in self.handicaps:
            p = devig({"home": ol, "away": ov}, method="proportional")["home"]
            if 0.2 < p < 0.8:        # líneas lejanas: la cola del de-vig es ruidosa
                ests.append(-h + sigma * norm.ppf(p))
        return sum(ests) / len(ests) if ests else None

    def total_esperado(self) -> float | None:
        mejores = sorted(self.totales, key=lambda x: abs(x[1] - x[2]))
        if not mejores:
            return None
        t, o, u = mejores[0]
        p_over = devig({"o": o, "u": u}, method="proportional")["o"]
        return t + 12.0 * norm.ppf(p_over)   # sd del total ≈ 12 pts (≈ la del margen)


def parse_hit(hit: dict) -> LineasPartido | None:
    s = hit["_source"]
    name = s.get("description", "")
    if " vs " not in name:
        return None
    home, away = [t.strip() for t in name.split(" vs ", 1)]
    lp = LineasPartido(local=home, visitante=away, inicio_ms=s.get("dateTime", 0))
    for bl in s.get("betLines") or []:
        d = bl.get("description") or ""
        opts = bl.get("options") or []
        if bl.get("type") == "ft2w" and d.startswith("Ganador"):
            for o in opts:
                if norm_team(o.get("result")) == norm_team(home):
                    lp.ganador["home"] = o["dividend"]
                elif norm_team(o.get("result")) == norm_team(away):
                    lp.ganador["away"] = o["dividend"]
        elif d.startswith("Hándicap (incl. prórroga)") and len(opts) == 2:
            m = _NUM.search(d)
            if m:
                # la línea está expresada para el LOCAL (primera opción, idext 1714)
                h = float(m.group(1))
                ol = next((o["dividend"] for o in opts if norm_team(o.get("result", "")).startswith(norm_team(home))), None)
                ov = next((o["dividend"] for o in opts if norm_team(o.get("result", "")).startswith(norm_team(away))), None)
                if ol and ov:
                    lp.handicaps.append((h, ol, ov))
        elif d.startswith("Total (incl. prórroga)") and len(opts) == 2:
            m = _NUM.search(d)
            o_ = next((o["dividend"] for o in opts if (o.get("result") or "").lower().startswith("más")), None)
            u_ = next((o["dividend"] for o in opts if (o.get("result") or "").lower().startswith("menos")), None)
            if m and o_ and u_:
                lp.totales.append((float(m.group(1)), o_, u_))
    return lp


def fetch_lub(lookahead_dias: int = LOOKAHEAD_DIAS) -> list[LineasPartido]:
    """Partidos de básquet de la liga 'Uruguay' en la ventana (la LUB y cualquier otra
    liga uruguaya: el matcheo contra el fixture de la penca filtra)."""
    now = int(time.time() * 1000)
    q = {"size": 200, "query": {"bool": {"must": [
            {"term": {"sportName.keyword": "Baloncesto"}},
            {"term": {"leagueName.keyword": "Uruguay"}},
            {"range": {"dateTime": {"gte": now, "lte": now + lookahead_dias * 86400 * 1000}}}]}},
         "_source": ["description", "leagueName", "championshipName", "dateTime", "betLines"]}
    try:
        r = httpx.post(ES, json=q, headers=HEADERS, timeout=30.0)
        r.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("supermatch ES falló: %s", e)
        return []
    return [lp for lp in (parse_hit(h) for h in r.json()["hits"]["hits"]) if lp]


def match_fixture(lineas: list[LineasPartido], local: str, visitante: str, inicio_ms: int,
                  ventana_h: float = 6.0) -> LineasPartido | None:
    """La línea de Supermatch del partido del fixture (mismo local y visitante, ±6 h)."""
    nl, nv = norm_team(local), norm_team(visitante)
    for lp in lineas:
        if abs(lp.inicio_ms - inicio_ms) > ventana_h * 3600 * 1000:
            continue
        a, b = norm_team(lp.local), norm_team(lp.visitante)
        if (nl in a or a in nl) and (nv in b or b in nv):
            return lp
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for lp in fetch_lub():
        print(lp.local, "vs", lp.visitante, lp.ganador, len(lp.handicaps), "hcp", len(lp.totales), "tot",
              "μ(σ=13)=", lp.mu_mercado(13.0), "total≈", lp.total_esperado())

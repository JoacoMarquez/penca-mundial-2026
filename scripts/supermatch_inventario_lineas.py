"""Inventario de TODAS las líneas de apuesta de Supermatch y su vig, + arbitraje intra-casa.

Análisis del 2026-09-29 ("¿se puede no perder plata apostando en Supermatch?").
Recorre el Elasticsearch público (sin auth, anda desde Uruguay) para todos los
deportes con eventos en la ventana, agrega por (deporte, tipo de línea) colapsando
números y nombres de equipo, y calcula el hold teórico de cada mercado:

    vig = Σ 1/cuota − 1        (0 = cuota justa; Supermatch ~15% en 2/3-way)

OJO: en mercados cuyas opciones NO son una partición (doble oportunidad, top-N de
golf) la suma no es un vig — se imprime igual, pero no compararla.

Después chequea arbitraje intra-casa sobre las particiones 1X2 ⊕ doble oportunidad
(1X+2, X2+1, 12+X): si alguna suma implícita da < 1 hay ganancia garantizada.

RESULTADO 29/9 (7 días, 731 eventos, 20 deportes): 1X2/totales/hándicap 15%,
combinadas 25%, margen de victoria 28-40%, marcador exacto 49%, mitad/final 33-70%.
Arbitrajes: 0 en 315 eventos (mejor suma 1,038).

Uso:
    python -m scripts.supermatch_inventario_lineas [--dias 7] [--min-n 5] [--dump eventos.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
import sys
import time
import unicodedata

import httpx

ES = ("https://elastic-frontend.supermatch.com.uy"
      "/elasticsearch_full_prematch_events_index/_search")
HEADERS = {
    "User-Agent": "Mozilla/5.0", "Content-Type": "application/json",
    "Origin": "https://www.supermatch.com.uy", "Referer": "https://www.supermatch.com.uy/",
}
PAGE = 200
MAX_PER_SPORT = 3000


def fetch_events(dias: int) -> list[dict]:
    now = int(time.time() * 1000)
    rng = {"range": {"dateTime": {"gte": now, "lte": now + dias * 86400 * 1000}}}
    hits: list[dict] = []
    with httpx.Client(timeout=30, headers=HEADERS) as c:
        agg = c.post(ES, json={"size": 0, "query": rng,
                               "aggs": {"s": {"terms": {"field": "sportName.keyword", "size": 50}}}}).json()
        sports = [(b["key"], b["doc_count"]) for b in agg["aggregations"]["s"]["buckets"]]
        print("deportes (eventos en la ventana):", sports, file=sys.stderr)
        for sport, n in sports:
            frm = 0
            while frm < min(n, MAX_PER_SPORT):
                r = c.post(ES, json={
                    "query": {"bool": {"must": [rng, {"term": {"sportName.keyword": sport}}]}},
                    "sort": [{"dateTime": "asc"}], "from": frm, "size": PAGE,
                    "_source": ["description", "sportName", "leagueName", "dateTime", "betLines"]})
                page = r.json()["hits"]["hits"]
                hits.extend(page)
                if len(page) < PAGE:
                    break
                frm += PAGE
    print("eventos:", len(hits), file=sys.stderr)
    return hits


def _norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def canon(desc: str | None, home: str, away: str) -> str:
    """Descripción de la línea sin números ni nombres de equipo."""
    d = desc or ""
    for t in sorted([home, away], key=len, reverse=True):
        if t and t in d:
            d = d.replace(t, "{Equipo}")
    d = re.sub(r"\(\s*-?[0-9]+(?:\.[0-9]+)?\s*\)", "( N )", d)
    d = re.sub(r"-?\b[0-9]+(?:\.[0-9]+)?\b", "N", d)
    return d.strip()


def inventario(hits: list[dict], min_n: int) -> None:
    rows: dict = collections.defaultdict(lambda: {"n": 0, "nopt": collections.Counter(), "vig": [], "ev": set()})
    for h in hits:
        s = h["_source"]
        sport = s.get("sportName")
        name = s.get("description", "")
        home, away = [t.strip() for t in name.split(" vs ", 1)] if " vs " in name else (name, "")
        for bl in s.get("betLines") or []:
            divs = [o.get("dividend") for o in bl.get("options") or [] if o.get("dividend") and o["dividend"] > 1]
            if not divs:
                continue
            r = rows[(sport, canon(bl.get("description"), home, away))]
            r["n"] += 1
            r["nopt"][len(divs)] += 1
            r["vig"].append(sum(1 / d for d in divs) - 1)
            r["ev"].add(h["_id"])

    print(f"=== INVENTARIO DE LÍNEAS (líneas con ≥{min_n} apariciones) ===")
    cur = None
    for (sport, d), r in sorted(rows.items(), key=lambda kv: (kv[0][0], -kv[1]["n"])):
        if r["n"] < min_n:
            continue
        if sport != cur:
            cur = sport
            print(f"\n## {sport}")
        med = st.median(r["vig"])
        print(f"  n={r['n']:4} ev={len(r['ev']):3} opts={dict(r['nopt'].most_common(2))} "
              f"vig={med*100:5.1f}%  {d[:70]}")


def arbitraje(hits: list[dict]) -> None:
    """Particiones 1X2 ⊕ doble oportunidad del mismo evento: Σ 1/cuota < 1 ⇒ surebet."""
    print("\n=== ARBITRAJE INTRA-CASA (1X2 ⊕ doble oportunidad) ===")
    found = checked = 0
    sums: list[tuple[float, str, str, float, float]] = []
    for h in hits:
        s = h["_source"]
        if " vs " not in s.get("description", ""):
            continue
        lines: dict[str, dict] = {}
        for bl in s.get("betLines") or []:
            lines.setdefault(_norm(bl.get("description")), bl)

        def opts(desc: str) -> dict[str, float] | None:
            L = lines.get(_norm(desc))
            if not L:
                return None
            return {_norm(o.get("result")): o["dividend"] for o in L.get("options") or [] if o.get("dividend")}

        x12, dc = opts("1x2"), opts("Doble oportunidad")
        if not x12 or not dc or "empate" not in x12:
            continue
        home, away = [_norm(t) for t in s["description"].split(" vs ", 1)]
        teams = [k for k in x12 if k != "empate"]
        if len(teams) != 2:
            continue
        hk = teams[0] if (home[:6] in teams[0] or teams[0][:6] in home) else teams[1]
        ak = teams[1] if hk == teams[0] else teams[0]

        def dcget(*keys: str) -> float | None:
            for k, v in dc.items():
                if all(kk in k for kk in keys):
                    return v
            return None

        combos = {"1X + 2": (dcget(home, "empate") or dcget("1x"), x12[ak]),
                  "X2 + 1": (dcget(away, "empate") or dcget("x2"), x12[hk]),
                  "12 + X": (dcget(home, away) or dcget("12"), x12["empate"])}
        checked += 1
        for name, (a, b) in combos.items():
            if a and b:
                tot = 1 / a + 1 / b
                sums.append((tot, name, s["description"], a, b))
                if tot < 1.0:
                    found += 1
                    print(f"  ARB {s['description']} {name}: {a} + {b} → {tot:.4f}")
    sums.sort()
    print(f"eventos chequeados: {checked}, arbitrajes: {found}")
    if sums:
        print("mejores 5 (suma implícita más baja):")
        for t, n, e, a, b in sums[:5]:
            print(f"  {t:.4f} {n:8} {e[:40]:40} {a} / {b}")
        print("suma implícita mediana:", round(st.median(x[0] for x in sums), 4))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dias", type=int, default=7, help="ventana hacia adelante (default 7)")
    ap.add_argument("--min-n", type=int, default=5, help="mínimo de apariciones para listar una línea")
    ap.add_argument("--dump", help="guardar los eventos crudos (JSON) para re-analizar sin red")
    ap.add_argument("--desde", help="leer eventos de un --dump previo en vez de la red")
    a = ap.parse_args()
    hits = json.load(open(a.desde)) if a.desde else fetch_events(a.dias)
    if a.dump:
        json.dump(hits, open(a.dump, "w"), ensure_ascii=False)
    inventario(hits, a.min_n)
    arbitraje(hits)


if __name__ == "__main__":
    main()

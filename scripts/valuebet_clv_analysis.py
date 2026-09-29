"""¿Hay edge real en Supermatch? CLV de las líneas contra el cierre de Pinnacle.

Corre sobre los snapshots crudos del scan de valuebet (`data/valuebet/raw/{book}/{día}/HHMMSS.json`,
uno por hora). Para cada línea de Supermatch matcheada con Pinnacle:

    edge_t = p_fair_pinnacle(t) · cuota_SM(t) − 1     valor APARENTE al momento t
    CLV    = p_fair_pinnacle(cierre) · cuota_SM(t) − 1 valor contra la línea de CIERRE del
                                                       sharp: el mejor proxy de EV real

Regla evaluada: "apostar la primera vez que edge_t ≥ θ" (una apuesta por línea).
Secciones: (4a) EV de apostar al azar por deporte/mercado/banda, (4b) la regla θ con
CLV, (4c) SM al cierre vs Pinnacle al cierre, (4d) cuánto del edge aparente
sobrevive al cierre, (4e) detalle de los candidatos y ritmo por día, (4f) ligas uruguayas.

RESULTADO 29/9 sobre julio (300 snapshots, 682 eventos, 44k observaciones, SIN totales
porque el parser de Pinnacle los tiraba — fix en PR #220): 20 candidatos en 14 días,
15 en amistosos, CLV +4,6% ± 2,9 (no distinguible de cero). Para decidir hacen falta
≥200 candidatos en temporada.

Uso (en el VPS, sobre lo que acumula el scan reactivado el 29/9):
    PYTHONPATH=/opt/penca .venv/bin/python -m scripts.valuebet_clv_analysis --raw data/valuebet/raw
Local, sobre el backup de julio:
    python -m scripts.valuebet_clv_analysis --raw ~/.../penca-backups/vps-20260729/data/valuebet/raw
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.model.market_probs import devig
from src.valuebet.matching import match_events
from src.valuebet.types import OddsQuote

ODDS_LO, ODDS_HI, MAX_EDGE = 1.4, 6.0, 0.25   # mismos filtros que config/valuebet.yaml


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def band(o: float) -> str:
    return ("1.0-1.4" if o < 1.4 else "1.4-2.0" if o < 2.0 else "2.0-3.0" if o < 3.0
            else "3.0-6.0" if o < 6 else "6+")


def mbase(m: str) -> str:
    return "total" if m.startswith("total") else m


def snapshot_pairs(raw: Path) -> list[tuple[str, Path, Path]]:
    """(día, archivo SM, archivo Pinnacle) emparejados por hora (±2 min)."""
    pairs = []
    for day in sorted(os.listdir(raw / "supermatch")):
        pdir = raw / "pinnacle" / day
        pf = sorted(os.listdir(pdir)) if pdir.exists() else []
        for f in sorted(os.listdir(raw / "supermatch" / day)):
            t = int(f[:6])
            cand = [g for g in pf if abs(int(g[:6]) - t) <= 200]
            if cand:
                pairs.append((day, raw / "supermatch" / day / f, pdir / cand[0]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/valuebet/raw", help="carpeta con raw/supermatch y raw/pinnacle")
    ap.add_argument("--config", default="config/valuebet.yaml")
    ap.add_argument("--theta", default="0,0.02,0.03,0.05,0.08,0.10", help="umbrales de edge a evaluar")
    ap.add_argument("--detalle-min-edge", type=float, default=0.03)
    a = ap.parse_args()
    logging.getLogger("src.valuebet.matching").setLevel(logging.ERROR)

    raw = Path(os.path.expanduser(a.raw))
    aliases = yaml.safe_load(open(a.config))["aliases"]
    pairs = snapshot_pairs(raw)
    print(f"pares de snapshots: {len(pairs)}  ({pairs[0][0]} → {pairs[-1][0]})" if pairs else "sin snapshots")
    if not pairs:
        return

    # ---------- universo de eventos + matcheo UNA vez ----------
    sm_events: dict[str, dict] = {}
    pn_events: dict[str, dict] = {}
    for _, sf, pf in pairs:
        for q in json.load(open(sf)):
            sm_events.setdefault(q["event_id"], q)
        for q in json.load(open(pf)):
            pn_events.setdefault(q["event_id"], q)
    sm2pn = {sq.event_id: pqs[0].event_id for sq, pqs in match_events(
        [OddsQuote(**q) for q in sm_events.values()],
        [OddsQuote(**q) for q in pn_events.values()], aliases)}
    print(f"eventos únicos: SM {len(sm_events)}, Pinnacle {len(pn_events)}, matcheados {len(sm2pn)}")

    # ---------- recorrer snapshots ----------
    obs: list[tuple] = []        # (sm_eid, market, outcome, t, sm_odds, fair_p, pn_odds)
    closing: dict[tuple, dict] = {}   # (pn_eid, market) → fair del último snapshot pre-inicio
    sm_close: dict[tuple, dict] = {}  # (sm_eid, market) → cuotas SM del último snapshot pre-inicio
    start_of: dict[str, datetime] = {}
    sport_of: dict[str, str] = {}
    league_of: dict[str, str] = {}
    for _, sf, pf in pairs:
        sm, pn = json.load(open(sf)), json.load(open(pf))
        if not sm or not pn:
            continue
        t = dt(sm[0]["fetched_utc"])
        pn_mk: dict[tuple, dict] = collections.defaultdict(dict)
        for q in pn:
            pn_mk[(q["event_id"], q["market"])][q["outcome"]] = q["decimal_odds"]
        fair_cache: dict[tuple, dict | None] = {}
        sm_mk: dict[tuple, dict] = collections.defaultdict(dict)
        for q in sm:
            pid = sm2pn.get(q["event_id"])
            if not pid:
                continue
            start = dt(q["start_utc"])
            if t >= start:
                continue
            key = (pid, q["market"])
            mk = pn_mk.get(key)
            if not mk or len(mk) < (3 if q["market"] == "1x2" else 2):
                continue
            if key not in fair_cache:
                try:
                    fair_cache[key] = devig({k: v for k, v in mk.items() if v > 1}, method="shin")
                except Exception:
                    fair_cache[key] = None
            fair = fair_cache[key]
            if not fair or q["outcome"] not in fair:
                continue
            obs.append((q["event_id"], q["market"], q["outcome"], t, q["decimal_odds"],
                        fair[q["outcome"]], mk[q["outcome"]]))
            closing[key] = fair
            sm_mk[(q["event_id"], q["market"])][q["outcome"]] = q["decimal_odds"]
            start_of[q["event_id"]] = start
            sport_of[q["event_id"]] = q["sport"]
            league_of[q["event_id"]] = q["league"]
        sm_close.update(sm_mk)
    print(f"observaciones: {len(obs)}")
    obs.sort(key=lambda r: r[3])

    # ---------- 4a ----------
    print("\n=== 4a. EV de apostar 'al azar' en Supermatch (edge vs Pinnacle de-vigueado, todas las líneas) ===")
    g: dict[tuple, list[float]] = collections.defaultdict(list)
    for e, m, o, t, so, fp, po in obs:
        g[(sport_of[e], mbase(m), band(so))].append(fp * so - 1)
    print(f"{'deporte':11}{'mercado':10}{'banda':9}{'n':>7}{'edge medio':>11}{'% edge>0':>9}{'% edge>3%':>10}")
    for k in sorted(g):
        x = np.array(g[k])
        print(f"{k[0]:11}{k[1]:10}{k[2]:9}{len(x):7d}{x.mean()*100:10.1f}%{(x>0).mean()*100:8.1f}%{(x>0.03).mean()*100:9.1f}%")

    # ---------- 4b ----------
    print(f"\n=== 4b. Regla 'apostar cuando edge_t ≥ θ' (1 apuesta/línea, cuota {ODDS_LO}-{ODDS_HI}, edge ≤{MAX_EDGE:.0%}) — CLV vs cierre Pinnacle ===")
    for theta in [float(x) for x in a.theta.split(",")]:
        taken: dict[tuple, tuple] = {}
        for e, m, o, t, so, fp, po in obs:
            edge = fp * so - 1
            if edge < theta or edge > MAX_EDGE or not (ODDS_LO <= so <= ODDS_HI) or (e, m, o) in taken:
                continue
            cl = closing.get((sm2pn[e], m))
            if cl and o in cl:
                taken[(e, m, o)] = (sport_of[e], mbase(m), edge, cl[o] * so - 1,
                                    (start_of[e] - t).total_seconds() / 3600)
        if not taken:
            print(f"\nθ={theta*100:4.1f}%  n=0")
            continue
        A = np.array([(v[2], v[3]) for v in taken.values()])
        se = A[:, 1].std() / math.sqrt(len(A)) * 100
        print(f"\nθ={theta*100:4.1f}%  n={len(A):5d}  edge_pred={A[:,0].mean()*100:5.1f}%  "
              f"CLV medio={A[:,1].mean()*100:+5.2f}% ± {se:.2f}  %CLV>0={(A[:,1]>0).mean()*100:4.1f}%  "
              f"h antes={np.median([v[4] for v in taken.values()]):.0f}")
        bys: dict[tuple, list[float]] = collections.defaultdict(list)
        for v in taken.values():
            bys[(v[0], v[1])].append(v[3])
        for k in sorted(bys):
            x = np.array(bys[k])
            print(f"    {k[0]:11}{k[1]:8} n={len(x):4d} CLV={x.mean()*100:+5.2f}% ± "
                  f"{x.std()/math.sqrt(len(x))*100:.2f}  %>0={(x>0).mean()*100:4.0f}%")

    # ---------- 4c ----------
    print("\n=== 4c. Supermatch al cierre vs Pinnacle al cierre: EV por banda (todas las líneas) ===")
    gc: dict[tuple, list[float]] = collections.defaultdict(list)
    for (e, m), odds in sm_close.items():
        cl = closing.get((sm2pn[e], m))
        if cl:
            for o, so in odds.items():
                if o in cl:
                    gc[(sport_of[e], mbase(m), band(so))].append(cl[o] * so - 1)
    print(f"{'deporte':11}{'mercado':10}{'banda':9}{'n':>6}{'EV medio':>9}{'% EV>0':>8}")
    for k in sorted(gc):
        x = np.array(gc[k])
        print(f"{k[0]:11}{k[1]:10}{k[2]:9}{len(x):6d}{x.mean()*100:8.1f}%{(x>0).mean()*100:7.1f}%")

    # ---------- 4d ----------
    print("\n=== 4d. ¿Cuánto del edge aparente sobrevive al cierre? (CLV = a + b·edge_pred) ===")
    R = np.array([(fp * so - 1, closing[(sm2pn[e], m)][o] * so - 1, (start_of[e] - t).total_seconds() / 3600)
                  for e, m, o, t, so, fp, po in obs
                  if (sm2pn[e], m) in closing and o in closing[(sm2pn[e], m)] and ODDS_LO <= so <= ODDS_HI])
    for lo, hi in [(0, 3), (3, 12), (12, 48), (48, 1e9)]:
        S = R[(R[:, 2] >= lo) & (R[:, 2] < hi)] if len(R) else R
        if len(S) < 50:
            continue
        b, a0 = np.polyfit(S[:, 0], S[:, 1], 1)
        print(f"  {lo:>3.0f}-{min(hi,999):>3.0f}h antes: n={len(S):6d}  CLV = {a0*100:+.2f}% + {b:.2f}·edge   "
              f"corr={np.corrcoef(S[:,0],S[:,1])[0,1]:.2f}")

    # ---------- 4e ----------
    print(f"\n=== 4e. Candidatos con edge_t ≥ {a.detalle_min_edge:.0%} (cuota 1.01-{ODDS_HI}) y ritmo ===")
    taken = {}
    for e, m, o, t, so, fp, po in obs:
        edge = fp * so - 1
        if edge < a.detalle_min_edge or edge > MAX_EDGE or not (1.01 <= so <= ODDS_HI) or (e, m, o) in taken:
            continue
        cl = closing.get((sm2pn[e], m))
        if cl and o in cl:
            taken[(e, m, o)] = dict(sport=sport_of[e], league=league_of[e], ev=sm_events[e]["event_name"],
                                    mkt=m, out=o, t=t, sm=so, pinn=po, edge=edge, clv=cl[o] * so - 1,
                                    hrs=(start_of[e] - t).total_seconds() / 3600, day=t.date())
    if taken:
        print(f"{'día':11}{'deporte':11}{'liga':26}{'evento':38}{'mkt':10}{'out':6}{'SM':>6}{'Pinn':>6}{'edge':>7}{'CLV':>7}{'h':>5}")
        for v in sorted(taken.values(), key=lambda v: v["t"]):
            print(f"{str(v['day']):11}{v['sport']:11}{v['league'][:25]:26}{v['ev'][:37]:38}{v['mkt'][:9]:10}"
                  f"{v['out']:6}{v['sm']:6.2f}{v['pinn']:6.2f}{v['edge']*100:6.1f}%{v['clv']*100:6.1f}%{v['hrs']:5.0f}")
        ndays = len({d for d, _, _ in pairs})
        A = np.array([[v["edge"], v["clv"]] for v in taken.values()])
        print(f"\nn={len(A)} en {ndays} días → {len(A)/ndays:.2f}/día; edge_pred {A[:,0].mean()*100:.1f}%, "
              f"CLV {A[:,1].mean()*100:+.2f}% ± {A[:,1].std()/math.sqrt(len(A))*100:.2f}; "
              f"CLV>0 {int((A[:,1]>0).sum())}/{len(A)}")
        print("ligas:", collections.Counter(v["league"] for v in taken.values()).most_common(8))
    else:
        print("  sin candidatos")

    # ---------- 4f ----------
    print("\n=== 4f. Ligas uruguayas (cancha propia de Supermatch): edge por banda ===")
    gu: dict[tuple, list[float]] = collections.defaultdict(list)
    for e, m, o, t, so, fp, po in obs:
        if "urug" in league_of[e].lower():
            gu[(sport_of[e], band(so))].append(fp * so - 1)
    for k in sorted(gu):
        x = np.array(gu[k])
        print(f"  {k[0]:11}{k[1]:9} n={len(x):5d} edge medio={x.mean()*100:6.1f}%  %>0={(x>0).mean()*100:4.1f}%")
    print("  eventos uruguayos matcheados:", sum(1 for e in sport_of if "urug" in league_of[e].lower()))


if __name__ == "__main__":
    main()

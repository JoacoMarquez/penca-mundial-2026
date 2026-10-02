"""¿Qué tan calibradas están las probabilidades de Supermatch, y conviene ajustarlas?

Pedido del 2/10. El 70% de cada λ del Clausura sale del Elasticsearch de Supermatch
de-vigueado con `proportional` (picks.market_lambdas); el marcador exacto, con Shin.
Mediciones previas, chicas: 16 partidos contra Pinnacle sin sesgo distinguible
(favorito −0,9pp ± 0,7) y 32 grillas de mercado calibradas contra resultados (z=−0,6).

Dos varas, porque ninguna alcanza sola:

  A. **Contra Pinnacle al cierre** (data/valuebet/raw, fútbol): Pinnacle de-vigueado
     con Shin es el mejor proxy de la probabilidad "verdadera" y da miles de mercados
     sin esperar resultados. Mide sesgo relativo (favorito/empate/no favorito y por
     banda), qué de-vig acerca más a Supermatch al sharp (KL medio), y ajusta una
     recalibración de 2 parámetros para el 1X2

         p_adj_i ∝ q_i^b · exp(c·[i = empate])      (q = de-vig proporcional de SM)

     con validación cruzada 2-fold por evento. b > 1 ⇒ SM es poco confiado
     (achata: hay que estirar hacia el favorito); b < 1 ⇒ sobreconfiado.

  B. **Contra los resultados del Clausura 2026** (data/odds/clausura, último snapshot
     antes de cada kickoff + resultados del penca-api): log-loss del 1X2, del
     over/under 2.5 y del marcador exacto por método de de-vig, la calibración "en
     grande" (esperado vs observado) y el ajuste de A aplicado fuera de muestra.
     Con ~60 partidos el SE del log-loss es grande: B confirma o refuta el signo de
     A, no mide décimas.

Pinnacle bloquea Uruguay: la parte A con datos nuevos corre en el VPS.

    ssh root@159.203.66.24 'cd /opt/penca && PYTHONPATH=. .venv/bin/python scripts/calibracion_supermatch.py'
    python scripts/calibracion_supermatch.py --raw ~/.../penca-backups/vps-20260729/data/valuebet/raw --sin-b
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import brentq, minimize

from src.model.market_probs import devig_proportional, devig_shin

ROOT = Path(__file__).resolve().parents[1]
OUT = ["home", "draw", "away"]


# -------------------- de-vig --------------------

def devig_power(odds: dict[str, float]) -> dict[str, float]:
    """p_i = π_i^k con k tal que Σ = 1 (corrige el favorito-longshot como Shin)."""
    keys = list(odds)
    pi = np.array([1.0 / odds[k] for k in keys])
    if abs(pi.sum() - 1) < 1e-9:
        return dict(zip(keys, pi))
    k = brentq(lambda k: (pi ** k).sum() - 1, 0.5, 5.0)
    return dict(zip(keys, pi ** k))


METODOS = {"proportional": devig_proportional, "shin": devig_shin, "power": devig_power}


def ajustar(q: np.ndarray, b: float, c: float) -> np.ndarray:
    """q: (n, 3) en orden home/draw/away. Recalibración p ∝ q^b · e^{c·[draw]}."""
    w = q ** b * np.exp(np.array([0.0, c, 0.0]))
    return w / w.sum(axis=1, keepdims=True)


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p * (np.log(p) - np.log(q))).sum(axis=1)


def fit_bc(q: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """(b, c) que minimizan el KL medio de target a ajustar(q). target puede ser
    one-hot (resultados) o probabilidades (Pinnacle)."""
    def f(x):
        p = ajustar(q, x[0], x[1])
        return -(target * np.log(p)).sum(axis=1).mean()
    r = minimize(f, x0=[1.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8})
    return float(r.x[0]), float(r.x[1])


def se(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


# -------------------- A: contra Pinnacle --------------------

def cargar_cierres(raw: Path, aliases: dict) -> list[dict]:
    """Por (evento SM, mercado): cuotas SM y Pinnacle del último par de snapshots
    antes del inicio. Solo fútbol, 1x2 y totales."""
    from scripts.valuebet_clv_analysis import dt, snapshot_pairs
    from src.valuebet.matching import match_events
    from src.valuebet.types import OddsQuote

    logging.getLogger("src.valuebet.matching").setLevel(logging.ERROR)
    pairs = snapshot_pairs(raw)
    sm_ev, pn_ev = {}, {}
    for _, sf, pf in pairs:
        for q in json.load(open(sf)):
            if q["sport"] == "soccer":
                sm_ev.setdefault(q["event_id"], q)
        for q in json.load(open(pf)):
            if q["sport"] == "soccer":
                pn_ev.setdefault(q["event_id"], q)
    sm2pn = {sq.event_id: pqs[0].event_id for sq, pqs in match_events(
        [OddsQuote(**q) for q in sm_ev.values()],
        [OddsQuote(**q) for q in pn_ev.values()], aliases)}

    cierre: dict[tuple, dict] = {}
    for _, sf, pf in pairs:
        sm, pn = json.load(open(sf)), json.load(open(pf))
        if not sm or not pn:
            continue
        t = dt(sm[0]["fetched_utc"])
        pn_mk: dict[tuple, dict] = collections.defaultdict(dict)
        for q in pn:
            if q["market"] == "1x2" or q["market"].startswith("total_"):
                pn_mk[(q["event_id"], q["market"])][q["outcome"]] = q["decimal_odds"]
        sm_mk: dict[tuple, dict] = collections.defaultdict(dict)
        meta = {}
        for q in sm:
            pid = sm2pn.get(q["event_id"])
            if not pid or t >= dt(q["start_utc"]):
                continue
            if q["market"] == "1x2" or q["market"].startswith("total_"):
                sm_mk[(q["event_id"], q["market"])][q["outcome"]] = q["decimal_odds"]
                meta[q["event_id"]] = (pid, q["league"], q["event_name"])
        for (e, m), odds in sm_mk.items():
            pid, league, name = meta[e]
            pmk = pn_mk.get((pid, m))
            n = 3 if m == "1x2" else 2
            if pmk and len(pmk) == n and len(odds) == n and set(pmk) == set(odds):
                cierre[(e, m)] = {"sm": odds, "pn": pmk, "league": league, "name": name,
                                  "market": m, "event": e}
    return list(cierre.values())


def seccion_a(cierres: list[dict]) -> tuple[float, float] | None:
    x12 = [c for c in cierres if c["market"] == "1x2"
           and all(v > 1 for v in c["sm"].values()) and all(v > 1 for v in c["pn"].values())]
    print(f"\n=== A. Supermatch vs Pinnacle al cierre (fútbol) — {len(x12)} mercados 1X2 ===")
    if len(x12) < 30:
        print("muy pocos mercados para concluir")
        return None

    pn = np.array([[devig_shin(c["pn"])[o] for o in OUT] for c in x12])
    over = np.array([sum(1 / v for v in c["sm"].values()) for c in x12])
    over_pn = np.array([sum(1 / v for v in c["pn"].values()) for c in x12])
    print(f"overround medio: Supermatch {100*(over.mean()-1):.1f}%  ·  Pinnacle {100*(over_pn.mean()-1):.1f}%")

    q = {m: np.array([[f(c["sm"])[o] for o in OUT] for c in x12]) for m, f in METODOS.items()}
    print("\nKL medio (Pinnacle ‖ SM de-vigueado), más bajo = más cerca del sharp:")
    for m, qm in q.items():
        d = kl(pn, qm)
        print(f"  {m:13} {1000*d.mean():6.2f} ×10⁻³ ± {1000*se(d):.2f}")
    base = kl(pn, q["proportional"])
    for m in ("shin", "power"):
        d = kl(pn, q[m]) - base
        print(f"  Δ {m} − proportional: {1000*d.mean():+.2f} ± {1000*se(d):.2f} ×10⁻³")

    # sesgo por rol (rol según Pinnacle)
    print("\nSesgo SM(proportional) − Pinnacle por rol, en puntos porcentuales:")
    qp = q["proportional"]
    fav = np.where(pn[:, 0] >= pn[:, 2], 0, 2)
    dog = 2 - fav
    idx = np.arange(len(x12))
    for nombre, col in (("favorito", fav), ("empate", np.ones_like(fav)), ("no favorito", dog)):
        d = 100 * (qp[idx, col] - pn[idx, col])
        print(f"  {nombre:12} {d.mean():+5.2f} ± {se(d):.2f}")

    print("\nConfiabilidad: prob SM (proportional) por banda vs Pinnacle en esa banda:")
    p_all, t_all = qp.ravel(), pn.ravel()
    for lo, hi in ((0, .15), (.15, .25), (.25, .35), (.35, .5), (.5, .65), (.65, .8), (.8, 1)):
        mk = (p_all >= lo) & (p_all < hi)
        if mk.sum() >= 5:
            d = 100 * (p_all[mk] - t_all[mk])
            print(f"  {lo:.2f}-{hi:.2f}  n={mk.sum():5d}  SM {100*p_all[mk].mean():5.1f}%  "
                  f"Pinnacle {100*t_all[mk].mean():5.1f}%  Δ {d.mean():+5.2f} ± {se(d):.2f}")

    # recalibración 2-fold por evento
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 2, len(x12))
    ganancia = []
    for k in (0, 1):
        tr, te = fold != k, fold == k
        b, c = fit_bc(qp[tr], pn[tr])
        ganancia.append(kl(pn[te], qp[te]) - kl(pn[te], ajustar(qp[te], b, c)))
    g = np.concatenate(ganancia)
    b, c = fit_bc(qp, pn)
    print(f"\nRecalibración p ∝ q^b·e^(c·empate) sobre proportional: b={b:.3f}  c={c:+.3f}")
    print(f"  mejora de KL fuera de muestra (2-fold): {1000*g.mean():+.2f} ± {1000*se(g):.2f} ×10⁻³ "
          f"(positivo = el ajuste acerca a Pinnacle)")
    adj = ajustar(qp, b, c)
    d_fav = 100 * (adj[idx, fav] - pn[idx, fav])
    print(f"  sesgo del favorito después del ajuste: {d_fav.mean():+.2f} ± {se(d_fav):.2f} pp")

    uy = [i for i, c in enumerate(x12) if "urugu" in c["league"].lower()]
    if uy:
        d = 100 * (qp[uy][np.arange(len(uy)), fav[uy]] - pn[uy][np.arange(len(uy)), fav[uy]])
        print(f"\nSolo ligas uruguayas: n={len(uy)}  sesgo favorito {d.mean():+.2f} ± {se(d):.2f} pp")

    # totales
    tot = [c for c in cierres if c["market"].startswith("total_") and set(c["sm"]) == {"over", "under"}]
    if len(tot) >= 30:
        pn_o = np.array([devig_shin(c["pn"])["over"] for c in tot])
        print(f"\nTotales (over/under, {len(tot)} mercados): SM − Pinnacle en P(over), pp")
        for m, f in METODOS.items():
            d = 100 * (np.array([f(c["sm"])["over"] for c in tot]) - pn_o)
            print(f"  {m:13} {d.mean():+5.2f} ± {se(d):.2f}   |Δ| medio {np.abs(d).mean():.2f}")
    return b, c


# -------------------- B: contra resultados del Clausura --------------------

def cargar_clausura() -> list[tuple]:
    """[(PartidoHistorico, EventOdds del último snapshot pre-kickoff)]."""
    from src.clausura.historical import fetch_temporada
    from src.clausura.odds import EventOdds
    from src.clausura.picks import match_odds

    partidos = fetch_temporada(44, "Torneo Clausura 2026")
    snaps = sorted((ROOT / "data" / "odds" / "clausura").glob("odds_*.json"))
    por_snap = []
    for f in snaps:
        t = datetime.strptime(f.stem[5:], "%Y%m%dT%H%M%SZ").replace(tzinfo=None)
        por_snap.append((t, [EventOdds(**d) for d in json.loads(f.read_text(encoding="utf-8"))]))

    out = []
    for p in partidos:
        ini = datetime.fromisoformat(p.inicio_utc).replace(tzinfo=None)
        elegido = None
        for t, odds in por_snap:
            if t >= ini or ini - t > timedelta(days=6):
                continue
            ev = [{"evento_id": p.evento_id, "local": p.local, "visitante": p.visitante}]
            m = match_odds(ev, odds).get(p.evento_id)
            if m and m.x1x2:
                elegido = m          # snapshots ordenados: queda el último pre-kickoff
        if elegido:
            out.append((p, elegido))
    print(f"\n=== B. Contra resultados del Clausura 2026 — {len(out)} de {len(partidos)} partidos con cuotas pre-kickoff ===")
    return out


def seccion_b(pares: list[tuple], bc: tuple[float, float] | None) -> None:
    if len(pares) < 15:
        print("muy pocos partidos")
        return
    y = np.array([[p.goles_local > p.goles_visitante, p.goles_local == p.goles_visitante,
                   p.goles_local < p.goles_visitante] for p, _ in pares], dtype=float)
    print(f"observado: local {y[:,0].mean():.1%}  empate {y[:,1].mean():.1%}  visitante {y[:,2].mean():.1%}")

    ll = {}
    for m, f in METODOS.items():
        q = np.array([[f(o.x1x2)[k] for k in OUT] for _, o in pares])
        ll[m] = -np.log((q * y).sum(axis=1))
        esp = q.mean(axis=0)
        print(f"  {m:13} log-loss 1X2 {ll[m].mean():.4f} ± {se(ll[m]):.4f}   "
              f"esperado L/E/V {esp[0]:.1%}/{esp[1]:.1%}/{esp[2]:.1%}")
    qp = np.array([[devig_proportional(o.x1x2)[k] for k in OUT] for _, o in pares])
    for m in ("shin", "power"):
        d = ll[m] - ll["proportional"]
        print(f"  Δ {m} − proportional: {d.mean():+.4f} ± {se(d):.4f}")
    if bc:
        d = -np.log((ajustar(qp, *bc) * y).sum(axis=1)) - ll["proportional"]
        print(f"  Δ ajuste de A (b={bc[0]:.3f}, c={bc[1]:+.3f}) − proportional: {d.mean():+.4f} ± {se(d):.4f}")
    b, c = fit_bc(qp, y)
    print(f"  ajuste que mejor calza los resultados (EN muestra, optimista): b={b:.2f} c={c:+.2f}")
    # b por bootstrap
    rng = np.random.default_rng(1)
    bs = [fit_bc(qp[i], y[i])[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(200))]
    print(f"    b bootstrap: mediana {np.median(bs):.2f}, IC 80% [{np.quantile(bs,.1):.2f}, {np.quantile(bs,.9):.2f}]"
          f"  (b=1 ⇒ sin ajuste; >1 ⇒ SM achata al favorito)")

    # over/under 2.5
    tot = [(p, o.totals["2.5"]) for p, o in pares if "2.5" in o.totals]
    if tot:
        yo = np.array([p.goles_local + p.goles_visitante > 2.5 for p, _ in tot], dtype=float)
        print(f"\nOver 2.5 ({len(tot)} partidos): observado {yo.mean():.1%}")
        for m, f in METODOS.items():
            po = np.array([f(t)["over"] for _, t in tot])
            l = -np.log(np.where(yo == 1, po, 1 - po))
            print(f"  {m:13} esperado {po.mean():.1%}  log-loss {l.mean():.4f} ± {se(l):.4f}")

    # marcador exacto
    cs = [(p, o.correct_score) for p, o in pares if o.correct_score]
    if cs:
        print(f"\nMarcador exacto ({len(cs)} partidos):")
        for m in ("proportional", "shin", "power"):
            l, en_cotizado = [], 0
            for p, odds in cs:
                pr = METODOS[m](odds)
                k = f"{p.goles_local}:{p.goles_visitante}"
                if k in pr:
                    en_cotizado += 1
                l.append(-math.log(pr.get(k, pr.get("otro", 1e-6))))
            print(f"  {m:13} log-loss {np.mean(l):.3f} ± {se(l):.3f}   (resultado cotizado en {en_cotizado}/{len(cs)})")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="append", help="carpeta(s) raw de valuebet (default data/valuebet/raw)")
    ap.add_argument("--sin-a", action="store_true")
    ap.add_argument("--sin-b", action="store_true")
    a = ap.parse_args()

    bc = None
    if not a.sin_a:
        aliases = yaml.safe_load(open(ROOT / "config" / "valuebet.yaml"))["aliases"]
        cierres = []
        for r in a.raw or [str(ROOT / "data" / "valuebet" / "raw")]:
            c = cargar_cierres(Path(os.path.expanduser(r)), aliases)
            print(f"{r}: {len(c)} mercados de fútbol matcheados")
            cierres += c
        bc = seccion_a(cierres)
    if not a.sin_b:
        seccion_b(cargar_clausura(), bc)


if __name__ == "__main__":
    main()

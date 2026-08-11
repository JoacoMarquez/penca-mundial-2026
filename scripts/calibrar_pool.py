"""Calibra popular_bias contra los picks reales del pool.

Modelo (el mismo de pool.py con chalk=1, T=1, que es lo que corre en producción):

    q_s  ∝  p_s^a · exp(β_s)

`a` es la fuerza de chalk y exp(β) ES popular_bias. Se ajusta por máxima
verosimilitud multinomial sobre los picks reales, en el marco ORIENTADO al favorito
(igual que pool_distribution: si el favorito es visitante, se refleja la grilla).

La validación es leave-one-match-out: 8 partidos y 36 celdas es poca data para 36
grados de libertad, así que lo que decide no es el ajuste sino la log-loss FUERA de
muestra contra el partido que no se usó para ajustar.
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_AQUI = pathlib.Path(__file__).resolve().parent
_aqui = lambda n: _AQUI / n
from src.clausura.pool import DEFAULT_POPULAR_BIAS, PoolConfig, pool_distribution  # noqa: E402

N = 36
IDX = np.arange(N)
GL, GV = IDX // 6, IDX % 6
MIOS = set(range(899258855, 899258867))          # nuestras 12 participaciones


def reflejar(v):
    """Espeja el eje local↔visitante de un vector de 36 celdas."""
    return v.reshape(6, 6).T.reshape(-1)


def extraer():
    """Arma el dataset desde el repo: grillas de mercado + picks reales del pool.

    La grilla de cada evento sale de `grilla_shadow.base` de la ÚLTIMA planilla donde
    ese partido seguía abierto — o sea la visión del mercado justo antes del cierre,
    que es contra la que hay que medir al pool. Los picks salen del snapshot más
    reciente, que trae todas las participaciones con sus pronósticos.
    """
    import glob
    raiz = pathlib.Path(__file__).resolve().parents[1]
    grillas, nombres = {}, {}
    versiones = sorted(glob.glob(str(raiz / "data/predictions/clausura/fecha_0*/v*.json")),
                       key=lambda f: json.load(open(f))["generado_utc"])
    for f in versiones:
        for r in json.load(open(f)).get("picks", []):
            if r.get("grilla_shadow"):
                grillas[str(r["evento_id"])] = r["grilla_shadow"]["base"]
                nombres[str(r["evento_id"])] = r["partido"]
    snaps = glob.glob(str(raiz / "data/pool_snapshots/clausura/*.json"))
    if not snaps or not grillas:
        raise SystemExit("faltan snapshots del pool o planillas con grilla_shadow")
    snap = max(snaps, key=lambda f: json.load(open(f))["generado_utc"])
    return {"grillas": grillas, "nombres": nombres,
            "participaciones": json.load(open(snap))["participaciones"]}


def cargar():
    cache = _aqui("calib.json")
    d = json.load(open(cache)) if cache.exists() else extraer()
    eventos = []
    for eid_s, base in d["grillas"].items():
        eid = int(eid_s)
        p = np.array(base, dtype=float)
        p = p / p.sum()
        c = np.zeros(N)
        for part in d["participaciones"]:
            if part["numero"] in MIOS:
                continue
            pick = part["picks"].get(str(eid))
            if pick is None:
                continue
            gl, gv = int(pick[0]), int(pick[1])
            if gl < 6 and gv < 6:
                c[gl * 6 + gv] += 1
        # marco orientado al favorito: si el favorito es visitante, espejamos TODO
        # (grilla y picks), así los 8 partidos quedan en el mismo eje.
        visitante_fav = p[GL < GV].sum() > p[GL > GV].sum()
        if visitante_fav:
            p, c = reflejar(p), reflejar(c)
        eventos.append({"eid": eid, "nombre": d["nombres"][eid_s], "p": p, "c": c,
                        "fav_visita": bool(visitante_fav), "n": int(c.sum())})
    # El gate del API es POR PARTIDO: los que todavía no empezaron vienen sin un solo
    # pick. No son partidos "sin datos", son partidos que no se pueden mirar.
    vacios = [e["nombre"] for e in eventos if e["n"] == 0]
    if vacios:
        print(f"(sin picks visibles, aún no empezaron: {', '.join(vacios)})\n")
    return [e for e in eventos if e["n"] > 0]


def q_de(p, a, beta):
    s = a * np.log(p + 1e-12) + beta
    s -= s.max()
    q = np.exp(s)
    return q / q.sum()


def neg_loglik(theta, evs, lam):
    a, beta = theta[0], theta[1:]
    ll = 0.0
    for e in evs:
        ll += float(e["c"] @ np.log(q_de(e["p"], a, beta) + 1e-300))
    return -ll + lam * float(beta @ beta)      # shrink hacia bias = 1


def ajustar(evs, lam):
    x0 = np.zeros(N + 1)
    x0[0] = 1.0
    r = minimize(neg_loglik, x0, args=(evs, lam), method="L-BFGS-B")
    return r.x[0], r.x[1:] - r.x[1:].mean()    # β identificable salvo constante


def logloss(evs, q_fn):
    """Log-loss por pick, promediada sobre los partidos dados."""
    tot_ll, tot_n = 0.0, 0
    for e in evs:
        q = q_fn(e)
        tot_ll += float(e["c"] @ np.log(q + 1e-300))
        tot_n += int(e["c"].sum())
    return -tot_ll / tot_n


def q_actual(e):
    """Lo que juega producción hoy. Ojo: pool_distribution reorienta sola, y nuestros
    eventos YA vienen orientados, así que hay que pasarle la grilla sin espejar."""
    p = reflejar(e["p"]) if e["fav_visita"] else e["p"]
    q = pool_distribution(p.reshape(6, 6), PoolConfig())
    return reflejar(q) if e["fav_visita"] else q


def main():
    evs = cargar()
    print(f"{len(evs)} partidos · {sum(e['n'] for e in evs):,} picks reales\n")
    for e in evs:
        print(f"  {e['nombre']:<38} {e['n']:>4} picks"
              f"{'  (favorito de visita → espejado)' if e['fav_visita'] else ''}")

    print("\n=== sesgo empírico crudo, marco orientado al favorito ===")
    print("(agregando los 8 partidos: Q_real / p_mercado, normalizado)")
    c_tot = sum(e["c"] for e in evs)
    p_pond = sum(e["p"] * e["c"].sum() for e in evs) / sum(e["c"].sum() for e in evs)
    crudo = (c_tot / c_tot.sum()) / (p_pond + 1e-12)
    crudo /= np.exp(np.log(crudo[crudo > 0]).mean())
    orden = np.argsort(-c_tot)
    print(f"\n  {'marcador':<10}{'picks':>7}{'% pool':>8}{'% mercado':>11}"
          f"{'bias emp.':>11}{'bias hoy':>10}")
    for i in orden[:16]:
        hoy = DEFAULT_POPULAR_BIAS.get((GL[i], GV[i]), PoolConfig().default_bias)
        print(f"  {GL[i]}-{GV[i]:<8}{int(c_tot[i]):>7}{100*c_tot[i]/c_tot.sum():>7.1f}%"
              f"{100*p_pond[i]:>10.1f}%{crudo[i]:>11.2f}{hoy:>10.2f}")

    print("\n=== leave-one-match-out: ¿el ajuste gana FUERA de muestra? ===")
    base = logloss(evs, q_actual)
    print(f"  config de hoy                        {base:.4f} nats/pick")
    for lam in (0.0, 1.0, 5.0, 20.0, 100.0, 500.0):
        perdidas = []
        for i in range(len(evs)):
            train = [e for j, e in enumerate(evs) if j != i]
            a, beta = ajustar(train, lam)
            perdidas.append(logloss([evs[i]], lambda e: q_de(e["p"], a, beta)))
        m = float(np.mean(perdidas))
        print(f"  ajustado, shrink λ={lam:<5.0f}                {m:.4f} nats/pick"
              f"   ({m - base:+.4f})")

    print("\n=== ajuste sobre los 8 partidos (el que iría a producción) ===")
    for lam in (5.0, 20.0, 100.0):
        a, beta = ajustar(evs, lam)
        bias = np.exp(beta)
        bias /= np.exp(np.log(bias).mean())
        print(f"\n  λ={lam:.0f} · chalk a={a:.2f}")
        for i in orden[:12]:
            hoy = DEFAULT_POPULAR_BIAS.get((GL[i], GV[i]), PoolConfig().default_bias)
            print(f"    {GL[i]}-{GV[i]}   {bias[i]:>6.2f}   (hoy {hoy:.2f})")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- descomposición
def beta_de_hoy():
    """log(popular_bias) actual, en el marco orientado (la tabla ya está en ese eje)."""
    b = np.full(N, PoolConfig().default_bias)
    for (gl, gv), f in DEFAULT_POPULAR_BIAS.items():
        b[gl * 6 + gv] = f
    lb = np.log(b)
    return lb - lb.mean()


def ajustar_solo_a(evs, beta_fijo):
    f = lambda t: -sum(float(e["c"] @ np.log(q_de(e["p"], t[0], beta_fijo) + 1e-300))
                       for e in evs)
    return float(minimize(f, np.array([1.0]), method="L-BFGS-B").x[0])


def ajustar_solo_beta(evs, a_fijo, lam):
    f = lambda b: (-sum(float(e["c"] @ np.log(q_de(e["p"], a_fijo, b) + 1e-300))
                        for e in evs) + lam * float(b @ b))
    r = minimize(f, np.zeros(N), method="L-BFGS-B").x
    return r - r.mean()


def lomo(evs, fit):
    """fit(train) -> q_fn(evento). Devuelve log-loss media fuera de muestra."""
    return float(np.mean([logloss([evs[i]], fit([e for j, e in enumerate(evs) if j != i]))
                          for i in range(len(evs))]))


def descomponer():
    evs = cargar()
    b_hoy = beta_de_hoy()
    base = logloss(evs, q_actual)
    print(f"\n{'='*66}\nQUÉ HACE EL TRABAJO: chalk o los 36 sesgos\n{'='*66}")
    print(f"  config de hoy (a=1, bias de la tabla)          {base:.4f}")

    m = lomo(evs, lambda tr: (lambda a: lambda e: q_de(e["p"], a, b_hoy))(
        ajustar_solo_a(tr, b_hoy)))
    print(f"  SOLO chalk (bias de hoy intacto)               {m:.4f}   ({m-base:+.4f})")

    m2 = lomo(evs, lambda tr: (lambda b: lambda e: q_de(e["p"], 1.0, b))(
        ajustar_solo_beta(tr, 1.0, 5.0)))
    print(f"  SOLO los 36 sesgos (a=1)                       {m2:.4f}   ({m2-base:+.4f})")

    m3 = lomo(evs, lambda tr: (lambda ab: lambda e: q_de(e["p"], ab[0], ab[1]))(
        ajustar(tr, 5.0)))
    print(f"  los dos juntos                                 {m3:.4f}   ({m3-base:+.4f})")

    a_solo = ajustar_solo_a(evs, b_hoy)
    print(f"\n  chalk ajustado sobre los 7 partidos: a = {a_solo:.2f}  (hoy 1.00)")

    # ¿el 0-0 sigue teniendo hueco una vez que el chalk explica la concentración?
    a, beta = ajustar(evs, 5.0)
    bias = np.exp(beta - beta.mean())
    print(f"\n  con a={a:.2f}, el 0-0 queda en bias {bias[0]:.2f} "
          f"(hoy 0.55) y el 2-1 en {bias[2*6+1]:.2f} (hoy 1.60)")


if __name__ != "__main__":
    pass

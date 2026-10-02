"""Re-simulación del Clausura con los picks REALES de los ~770 rivales.

Todo lo que se midió hasta ahora usa rivales SIMULADOS (pool i.i.d. ∝ Q o el
RivalModel). Este arnés cambia la verdad del lado rival: los rivales juegan lo que
jugaron de verdad, y lo único que se sortea son los resultados, desde la grilla de
mercado pre-cierre (calibrada). Así una política se evalúa contra el pool que
existió, sin el supuesto del modelo, y con mucho menos ruido que contra el único
resultado real de cada fecha (que también se reporta).

Para que la comparación sea honesta, las POLÍTICAS deciden con la información que
tenía producción antes de cada fecha: la grilla de producción (grilla_shadow.base de
la última planilla previa al cierre) y una Q del pool MODELADA (pool_distribution con
la temperatura de esa planilla), nunca los picks reales de esa fecha.

Unidad: premio de FECHA ($10.000 al máximo de la fecha, repartido entre empatados).
Es el objetivo correcto para optimizar un bloque, y el único que se puede liquidar
antes de que termine la temporada. PENDIENTE para el cierre del Clausura (20/11): el
premio grande ($350.000) necesita re-jugar la temporada en secuencia, con los
standings de cada fecha y los especiales, contra las 15 fechas de picks reales.

Resultados del 2/10 (F1-F8, 765 rivales, 19.200 / 20.000 sorteos):
  calib    lo cargado vale $2.503 con rivales modelados vs $2.363 con los reales (0,94).
           Calibrado — OJO: con un p_show GLOBAL por partido daba 1,60 (artefacto).
  menu     K_EV 5→8 +$559 ± 48 (6/8 fechas) · escalera −$2.344 (0/8) · n 12→16 +$893,
           16→20 +$795: ~$25-28 por participación y fecha, ≈ lo que cuesta ($400/15).

Experimentos (--exp):
  calib    E[premio fecha] de lo CARGADO con rivales modelados vs reales (mismos sorteos):
           ¿el modelo del pool sesga cuánto vale nuestra cola?
  menu     ascenso K_EV=5 (producción) vs 8
  escalera ascenso vs "escalera de objetivos": fila i elige por partido el argmax de
           E[pts]·Q^(−α_i), α de 0 (EV) a 1 (contrarian) — la idea del Mundial de mezclar
           objetivos distintos en vez de perturbar alrededor de UN óptimo
  k        cantidad de participaciones (4/8/12/16/20) con el ascenso

Uso (local, con data/replay/clausura.json bajado del VPS — ver scripts/replay_dataset.py):
    python -m scripts.replay_clausura --exp calib,menu,escalera,k [--sims 19200] [--eval 20000]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.clausura.economics import (
    MAX_GOALS, N_SCORES, SIDE, PrizeConfig, SeasonSimulator, SimConfig, points_matrix,
    score_index,
)
from src.clausura.pool import PoolConfig, pool_distribution
from src.clausura.strategy import EVAL_SEED_OFFSET, build_candidates

DATASET = Path("data/replay/clausura.json")
PREMIO_FECHA = 10_000.0
PM = {False: points_matrix(False), True: points_matrix(True)}


def _idx(gl: int, gv: int) -> int:
    return score_index(min(gl, MAX_GOALS), min(gv, MAX_GOALS))


@dataclass
class Fecha:
    n: int
    eventos: list[dict]
    pref: list[bool]
    verdad: list[np.ndarray]         # grilla para sortear resultados (mercado; prod si no hay)
    prod: list[np.ndarray]           # grilla que veía producción
    q_modelo: list[np.ndarray]       # Q del pool modelada (lo que veía producción)
    real: list[int]                  # resultado real (índice)
    riv: np.ndarray                  # (R, M) picks reales de los rivales, −1 = no cargó
    cargado: np.ndarray              # (12, M) lo que cargamos de verdad
    p_show: np.ndarray | None = None # (R,) por rival, de las fechas ANTERIORES (como rivals.py)


def cargar(path: Path = DATASET) -> tuple[dict[int, Fecha], dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    mis = d["mis_numeros"]
    por_num = {p["numero"]: p for p in d["participaciones"]}
    rivales = [p for p in d["participaciones"] if p["numero"] not in set(mis)]
    fechas: dict[int, Fecha] = {}
    for n in sorted({e["fecha_n"] for e in d["eventos"]}):
        evs = [e for e in d["eventos"] if e["fecha_n"] == n and e["resultado"] is not None
               and e["grid_prod"] is not None]
        if not evs:
            continue
        M = len(evs)
        riv = np.full((len(rivales), M), -1, np.int64)
        car = np.full((len(mis), M), -1, np.int64)
        verdad, prod, qs, real = [], [], [], []
        for j, e in enumerate(evs):
            gp = np.asarray(e["grid_prod"], float).reshape(SIDE, SIDE)
            gm = (np.asarray(e["grid_mercado"], float).reshape(SIDE, SIDE)
                  if e["grid_mercado"] is not None else gp)
            verdad.append(gm / gm.sum())
            prod.append(gp / gp.sum())
            qs.append(pool_distribution(gp, PoolConfig(temperature=e["temperatura_pool"] or 1.0)))
            real.append(_idx(*e["resultado"]))
            k = str(e["evento_id"])
            for r, p in enumerate(rivales):
                if k in p["picks"]:
                    riv[r, j] = _idx(*p["picks"][k])
            for i, num in enumerate(mis):
                pk = por_num[num]["picks"].get(k)
                if pk:
                    car[i, j] = _idx(*pk)
        fechas[n] = Fecha(n, evs, [e["preferencial"] for e in evs], verdad, prod, qs, real,
                          riv, car)
    # p_show por rival como rivals.py (prior Beta(9,1)) con lo observado ANTES de la fecha
    previo = np.zeros((len(rivales), 0), np.int64)
    for n in sorted(fechas):
        obs = (previo >= 0).sum(axis=1)
        fechas[n].p_show = (9 + obs) / (10 + previo.shape[1])
        previo = np.concatenate([previo, fechas[n].riv], axis=1)
    return fechas, d


# ---------------------------------------------------------------- liquidación

def puntos(picks: np.ndarray, res: np.ndarray, pref: list[bool]) -> np.ndarray:
    """picks (N, M) con −1 = sin pick; res (M, S) → puntos (N, S)."""
    N, M = picks.shape
    out = np.zeros((N, res.shape[1]), np.int32)
    for m in range(M):
        ok = picks[:, m] >= 0
        out[ok] += PM[pref[m]][picks[ok, m][:, None], res[m][None, :]]
    return out


def premio(mios: np.ndarray, rivales: np.ndarray) -> np.ndarray:
    """Premio de fecha cobrado por sorteo, con reparto entre empatados (Art. 7a)."""
    rtop = rivales.max(axis=0)
    rcnt = (rivales == rtop[None, :]).sum(axis=0)
    top = np.maximum(mios.max(axis=0), rtop)
    k = (mios == top[None, :]).sum(axis=0)
    j = np.where(rtop == top, rcnt, 0)
    return np.where(k > 0, PREMIO_FECHA * k / np.maximum(k + j, 1), 0.0)


def sortear(f: Fecha, S: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(N_SCORES, size=S, p=g.ravel()) for g in f.verdad])


def rivales_modelados(f: Fecha, res: np.ndarray, seed: int) -> np.ndarray:
    """(R, S) puntos de R rivales i.i.d. ∝ Q modelada; cada uno carga cada partido con su
    p_show (Bernoulli por partido, como SimConfig.show_por_fecha=False de producción).

    Medido el 2/10 en F2-F8 con lo cargado: este mundo da $2.279 de premio de fecha
    esperado contra $2.050 con los rivales reales (+11%); con un p_show GLOBAL por
    partido daba +60% (los muertos bajaban el máximo de todos) — no usar ese atajo."""
    rng = np.random.default_rng(seed)
    R, S = f.riv.shape[0], res.shape[1]
    out = np.zeros((R, S), np.int32)
    for m, q in enumerate(f.q_modelo):
        pk = rng.choice(N_SCORES, size=(R, S), p=q)
        pts = PM[f.pref[m]][pk, res[m][None, :]]
        out += np.where(rng.random((R, S)) < f.p_show[:, None], pts, 0)
    return out


@dataclass
class Verdad:
    """Sorteos de resultados + puntos de los rivales reales y modelados, fijos por fecha."""
    res: np.ndarray
    riv_real: np.ndarray
    riv_modelo: np.ndarray | None = None


def armar_verdad(f: Fecha, S: int, seed: int, modelo: bool = False) -> Verdad:
    res = sortear(f, S, seed)
    v = Verdad(res, puntos(f.riv, res, f.pref))
    if modelo:
        v.riv_modelo = rivales_modelados(f, res, seed + 1)
    return v


def evaluar(f: Fecha, picks: np.ndarray, v: Verdad) -> dict:
    mios = puntos(picks, v.res, f.pref)
    pr = premio(mios, v.riv_real)
    out = {"e_real": float(pr.mean()), "se_real": float(pr.std() / math.sqrt(len(pr))),
           "_pr_real": pr}
    if v.riv_modelo is not None:
        pm = premio(mios, v.riv_modelo)
        out |= {"e_modelo": float(pm.mean()), "_pr_modelo": pm}
    # realizado: el único resultado que pasó
    r = np.array(f.real)[:, None]
    m1, r1 = puntos(picks, r, f.pref)[:, 0], puntos(f.riv, r, f.pref)[:, 0]
    out |= {"realizado": float(premio(m1[:, None], r1[:, None])[0]), "max": int(m1.max()),
            "max_rival": int(r1.max())}
    return out


# ---------------------------------------------------------------- políticas

def candidatos(f: Fecha, k_ev: int) -> list[list[int]]:
    return [[score_index(*c.pick) for c in sorted(build_candidates(g, q, p, k_ev=k_ev),
                                                  key=lambda c: -c.e_points)]
            for g, q, p in zip(f.prod, f.q_modelo, f.pref)]


def ascenso(f: Fecha, n: int, k_ev: int, sims: int, seed: int, max_passes: int = 6) -> np.ndarray:
    """build_portfolio sobre el bloque de la fecha: lo que haría producción para ganar la
    fecha, con su información (grilla de producción, Q modelada, rivales i.i.d.)."""
    cands = candidatos(f, k_ev)
    fechas = [f.n] * len(f.eventos)
    sim = SeasonSimulator(f.prod, fechas, f.pref, f.q_modelo, PrizeConfig(),
                          SimConfig(n_sims=sims, n_rivales=f.riv.shape[0], seed=seed), None)
    picks = np.array([[cs[0] for cs in cands]] * n, dtype=np.int64)
    sim.load_picks(picks)
    actual = sim.e_premio_total()
    for _ in range(max_passes):
        mejoras = 0
        for i in range(1, n):
            for m, cs in enumerate(cands):
                orig = int(sim.picks[i, m])
                mejor, mejor_val = orig, actual
                for c in cs:
                    if c == orig:
                        continue
                    sim.set_pick(i, m, c)
                    val = sim.e_premio_total()
                    if val > mejor_val:
                        mejor, mejor_val = c, val
                sim.set_pick(i, m, mejor)
                if mejor != orig:
                    mejoras += 1
                    actual = mejor_val
        if mejoras == 0:
            break
    return sim.picks.copy()


def escalera(f: Fecha, n: int, alfa_max: float = 1.0, k_ev: int = 8) -> np.ndarray:
    """Fila i: argmax_c E[pts(c)] · Q(c)^(−α_i), α_i lineal de 0 a alfa_max, sobre el menú."""
    alfas = np.linspace(0.0, alfa_max, n)
    picks = np.zeros((n, len(f.eventos)), np.int64)
    for m, (g, q, p) in enumerate(zip(f.prod, f.q_modelo, f.pref)):
        cs = build_candidates(g, q, p, k_ev=k_ev)
        idx = np.array([score_index(*c.pick) for c in cs])
        ev = np.array([c.e_points for c in cs])
        qq = np.maximum(q[idx], 1e-4)
        for i, a in enumerate(alfas):
            picks[i, m] = idx[int(np.argmax(ev * qq ** (-a)))]
    return picks


# ---------------------------------------------------------------- reporte

def resumen(nombre: str, filas: list[dict], base: list[dict] | None = None) -> str:
    e = sum(r["e_real"] for r in filas)
    rea = sum(r["realizado"] for r in filas)
    s = f"{nombre:28s} E[premio fechas] ${e:>8,.0f}   realizado ${rea:>7,.0f}"
    if base is not None:
        d = np.concatenate([r["_pr_real"] - b["_pr_real"] for r, b in zip(filas, base)])
        # el Δ se agrega SUMANDO fechas: media por fecha × n fechas, error por sorteos
        dd = np.array([(r["_pr_real"] - b["_pr_real"]).mean() for r, b in zip(filas, base)])
        se = math.sqrt(sum(((r["_pr_real"] - b["_pr_real"]).std() ** 2) / len(r["_pr_real"])
                           for r, b in zip(filas, base)))
        mejor = int((dd > 0).sum())
        s += f"   Δ ${dd.sum():+,.0f} ± {se:,.0f} (sorteos) · mejor en {mejor}/{len(dd)} fechas"
        del d
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="calib,menu,escalera,k")
    ap.add_argument("--sims", type=int, default=19_200, help="sorteos del ascenso")
    ap.add_argument("--eval", type=int, default=20_000, help="sorteos de la evaluación")
    ap.add_argument("--fechas", default=None, help="ej. 2-8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fechas, _ = cargar()
    if a.fechas:
        lo, hi = (int(x) for x in a.fechas.split("-"))
        fechas = {n: f for n, f in fechas.items() if lo <= n <= hi}
    exps = set(a.exp.split(","))
    print(f"fechas {sorted(fechas)} · {next(iter(fechas.values())).riv.shape[0]} rivales · "
          f"ascenso {a.sims} sorteos · evaluación {a.eval}", flush=True)
    verdades = {n: armar_verdad(f, a.eval, 9000 + n, modelo="calib" in exps)
                for n, f in fechas.items()}
    res: dict[str, list[dict]] = {}

    def correr(nombre, fn):
        res[nombre] = []
        for n, f in fechas.items():
            picks = fn(f)
            res[nombre].append(evaluar(f, picks, verdades[n]) | {"fecha": n})
        return res[nombre]

    cargado = correr("cargado", lambda f: f.cargado)
    if "calib" in exps:
        print("\n== calib: lo CARGADO, rivales modelados vs reales (mismos resultados) ==")
        for r in cargado:
            print(f"F{r['fecha']}: modelo ${r['e_modelo']:>6,.0f} · real ${r['e_real']:>6,.0f} "
                  f"· realizado ${r['realizado']:>6,.0f} (máx {r['max']} vs {r['max_rival']})")
        em, er = sum(r["e_modelo"] for r in cargado), sum(r["e_real"] for r in cargado)
        print(f"TOTAL modelo ${em:,.0f} vs real ${er:,.0f} (ratio {er / em:.2f}) · "
              f"realizado ${sum(r['realizado'] for r in cargado):,.0f}")

    base = None
    if exps & {"menu", "escalera", "k"}:
        base = correr("ascenso K_EV=5 n=12", lambda f: ascenso(f, 12, 5, a.sims, 100 + f.n))
    print("\n== políticas (verdad: rivales REALES, resultados ~ grilla de mercado) ==")
    print(resumen("cargado", cargado))
    if base is not None:
        print(resumen("ascenso K_EV=5 n=12", base, cargado))
    if "menu" in exps:
        print(resumen("ascenso K_EV=8 n=12",
                      correr("k8", lambda f: ascenso(f, 12, 8, a.sims, 100 + f.n)), base))
    if "escalera" in exps:
        for am in (0.5, 1.0, 1.5):
            print(resumen(f"escalera α≤{am} n=12",
                          correr(f"esc{am}", lambda f, am=am: escalera(f, 12, am)), base))
    if "k" in exps:
        for n in (4, 8, 16, 20):
            filas = correr(f"n{n}", lambda f, n=n: ascenso(f, n, 5, a.sims, 100 + f.n))
            print(resumen(f"ascenso K_EV=5 n={n}", filas, base))

    if a.out:
        Path(a.out).write_text(json.dumps(
            {k: [{kk: vv for kk, vv in r.items() if not kk.startswith("_")} for r in v]
             for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()

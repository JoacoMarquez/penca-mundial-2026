"""Barrido de N participaciones: ¿cuántas extra conviene comprar?

Reconstruye el entorno EXACTO de la v13 (fecha 1 con grillas del mercado guardadas
en el shadow, futuras con ratings, pool empírico de 489 rivales del snapshot,
gap de especiales real: Peñarol 71% / Arezo 35%) y mide el marginal de E[premio]
y P(cobrar premio) al agregar participaciones 13..28 con las 12 actuales congeladas.

Escenarios:
  A: compradas HOY antes del lock de especiales (21:30 UTC) → cargan los 8 partidos
     de la fecha 1 + campeón + goleador.
  B: compradas mañana (sáb 8/8 a la mañana) → sin especiales, pierden el partido
     de esta noche (Cerro Largo vs Juventud), cargan los otros 7.

Mejoras de realismo sobre la corrida v13 de producción:
  * GOLEADOR incluido: la v13 lo ignoró (menú API en 500), pero 303 rivales YA
    tienen goleador cargado (25 pts). Acá los rivales usan su goleador observado
    y los nuestros se optimizan (asumiendo que lo cargamos por la web antes del lock).
  * Evaluación SIEMPRE con semilla independiente (EVAL_SEED_OFFSET) y pareada
    (mismos sorteos para todos los N → los Δ marginales no tienen ruido MC cruzado).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)
import os
os.chdir(REPO)

# ---------------------------------------------------------------- parámetros
S_OPT = int(os.environ.get("SWEEP_S_OPT", 2400))
S_EVAL = int(os.environ.get("SWEEP_S_EVAL", 6000))
SEED = 20260807
N_MAX = int(os.environ.get("SWEEP_N_MAX", 28))
GOL_TEMPER = 0.7      # p_goleador "verdad" = empírica^0.7 (capa débil, marcada)
V13 = "data/predictions/clausura/fecha_01/v13_20260807T142019Z.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sweep_results.json")

from src.clausura.economics import (  # noqa: E402
    MAX_GOALS, N_SCORES, PrizeConfig, SeasonSimulator, SimConfig, score_index,
)
from src.clausura.especiales import (  # noqa: E402
    p_campeon as p_campeon_fn, pool_campeon_distribution, p_campeon_from_grids,
)
from src.clausura.picks import (  # noqa: E402
    ensure_ratings, flat_eventos, load_config,
)
from src.clausura.pool import PoolConfig, pool_distribution  # noqa: E402
from src.clausura.pool_snapshot import (  # noqa: E402
    blended_q, empirical_campeon_counts, load_latest_snapshot,
)
from src.clausura.rivals import build_rival_model  # noqa: E402
from src.clausura.strategy import EVAL_SEED_OFFSET, build_candidates  # noqa: E402
from src.model.poisson import score_grid  # noqa: E402

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ---------------------------------------------------------------- insumos
cfg = load_config()
eventos = flat_eventos(cfg)
n_matches = len(eventos)
v13 = json.loads(Path(V13).read_text())
idx_of = {ev["evento_id"]: i for i, ev in enumerate(eventos)}

ratings = ensure_ratings()
log(f"ratings ok · {n_matches} eventos")

# grillas: fecha 1 desde el shadow de la v13 (mismas creencias que la planilla
# publicada; CLAUSURA_MERCADOS_RICOS off → la usada fue la 'base'), resto ratings
grids = []
v13_grid = {p["evento_id"]: np.array(p["grilla_shadow"]["base"]).reshape(6, 6)
            for p in v13["picks"]}
for ev in eventos:
    if ev["evento_id"] in v13_grid:
        grids.append(v13_grid[ev["evento_id"]])
    else:
        lam_l, lam_v = ratings.lambdas(ev["local"], ev["visitante"])
        grids.append(score_grid(lam_l, lam_v, 0.0, max_goals=MAX_GOALS))
fecha_de = [ev["fecha_id"] for ev in eventos]
pref = [ev["preferencial"] for ev in eventos]

# pool de marcadores: prior (snapshot sin marcadores → no hay counts)
pool_cfg = PoolConfig()
pool_qs = [pool_distribution(g, pool_cfg) for g in grids]

snapshot = load_latest_snapshot(max_age_hours=48)
assert snapshot, "sin snapshot del pool"
equipos_cfg = cfg["equipos"]
equipo_nombres = [equipos_cfg[k] for k in sorted(equipos_cfg)]
equipo_idx = {n: i for i, n in enumerate(equipo_nombres)}
rival_model = build_rival_model(snapshot, eventos, pool_qs, {}, mis_numeros=set(),
                                equipo_idx=equipo_idx)
R = rival_model.n_rivales
log(f"rival model: {R} rivales")

# especiales: campeón
n_teams = len(equipo_nombres)
local_de = np.array([equipo_idx[ev["local"]] for ev in eventos])
visita_de = np.array([equipo_idx[ev["visitante"]] for ev in eventos])

p_champ_prior = p_campeon_from_grids(grids, local_de, visita_de, n_teams)
pool_q_campeon = pool_campeon_distribution(p_champ_prior, equipo_nombres)
camp_counts = empirical_campeon_counts(snapshot, equipo_idx, n_teams, set())
pool_q_campeon = blended_q(pool_q_campeon, camp_counts)
log("pool campeón empírico: " + ", ".join(
    f"{equipo_nombres[i]} {pool_q_campeon[i]:.0%}"
    for i in np.argsort(-pool_q_campeon)[:4]))

# especiales: goleador sintetizado del snapshot (menú API sigue en 500, pero los
# rivales SÍ cargaron: los nombres del snapshot son el menú de facto)
from collections import Counter
gol_counts = Counter(r["goleador"] for r in snapshot["participaciones"] if r["goleador"])
gol_names = [n for n, _ in gol_counts.most_common()]
gol_idx = {n: i for i, n in enumerate(gol_names)}
n_gol = len(gol_names)
counts_vec = np.array([gol_counts[n] for n in gol_names], dtype=float)
pool_q_gol = counts_vec / counts_vec.sum()
p_gol = counts_vec ** GOL_TEMPER
p_gol /= p_gol.sum()
log(f"goleador: {n_gol} opciones · pool top: "
    + ", ".join(f"{n} {pool_q_gol[gol_idx[n]]:.0%}" for n in gol_names[:3])
    + " · verdad top: " + ", ".join(f"{n} {p_gol[gol_idx[n]]:.0%}" for n in gol_names[:3]))

# goleador observado por rival (alineado al orden de build_rival_model: snapshot sin filtrar)
rival_gol_known = np.array(
    [gol_idx.get(r.get("goleador"), -1) for r in snapshot["participaciones"]],
    dtype=np.int64)

prize = PrizeConfig()
PTS_ESP = 25


# ---------------------------------------------------------------- simuladores
def build_sim(seed: int, n_sims: int) -> SeasonSimulator:
    sim = SeasonSimulator(grids, fecha_de, pref, pool_qs, prize,
                          SimConfig(n_sims=n_sims, n_rivales=R, seed=seed),
                          rivals=rival_model)
    # campeón (rival side): usa el campeón OBSERVADO de cada rival
    sim.enable_campeon(local_de, visita_de, n_teams, pool_q_campeon, puntos=PTS_ESP)
    # goleador (rival side, manual): observado si lo cargó, sampleado si no
    rng = np.random.default_rng(seed + 2)
    sim.gol_sim = rng.choice(n_gol, size=n_sims, p=p_gol)
    sim._puntos_especial = PTS_ESP
    rp = rng.choice(n_gol, size=(R, n_sims), p=pool_q_gol)
    rp = np.where(rival_gol_known[:, None] >= 0, rival_gol_known[:, None], rp)
    sim.rivals_total += PTS_ESP * (rp == sim.gol_sim[None, :])
    return sim


class FastPort:
    """Evaluador rápido: rivales congelados como (max, count) por sim."""

    def __init__(self, sim: SeasonSimulator):
        self.sim = sim
        self.S = sim.cfg.n_sims
        self.F = sim.n_fechas
        # Desde el 2026-08-08 el simulador ya cachea el (máximo, empatados) del lado
        # rival — es exactamente lo que este evaluador calculaba a mano — y además
        # NO guarda el acumulado por fecha entero (era (n_fechas, R, S), el techo de
        # memoria del droplet). Se reusan sus stats en vez de re-derivarlas.
        self.rmax, self.rcnt = sim._stats_total()
        stats = [sim._stats_fecha(f) for f in range(self.F)]
        self.rfmax = np.stack([m for m, _ in stats])
        self.rfcnt = np.stack([c for _, c in stats])
        self.reset(0)

    def reset(self, n_rows: int):
        self.n = 0
        self.picks = np.zeros((0, n_matches), dtype=np.int64)
        self.show = np.zeros((0, n_matches), dtype=bool)
        self.camp = np.zeros(0, dtype=np.int64)
        self.gol = np.zeros(0, dtype=np.int64)
        self.mt = np.zeros((0, self.S), dtype=np.int64)
        self.mf = np.zeros((self.F, 0, self.S), dtype=np.int64)

    def add_row(self, picks_row, show_row=None, camp=-1, gol=-1):
        sim = self.sim
        show_row = np.ones(n_matches, dtype=bool) if show_row is None else show_row
        tot = np.zeros(self.S, dtype=np.int64)
        fec = np.zeros((self.F, self.S), dtype=np.int64)
        for m in range(n_matches):
            if not show_row[m]:
                continue
            pts = sim.pm[m][picks_row[m], sim.actual[m]]
            tot += pts
            fec[sim.match_fecha[m]] += pts
        if camp >= 0:
            tot = tot + PTS_ESP * (sim.champ_sim == camp)
        if gol >= 0:
            tot = tot + PTS_ESP * (sim.gol_sim == gol)
        self.picks = np.vstack([self.picks, picks_row[None, :]])
        self.show = np.vstack([self.show, show_row[None, :]])
        self.camp = np.append(self.camp, camp)
        self.gol = np.append(self.gol, gol)
        self.mt = np.vstack([self.mt, tot[None, :]])
        self.mf = np.concatenate([self.mf, fec[:, None, :]], axis=1)
        self.n += 1

    def set_pick(self, i, m, idx):
        old = int(self.picks[i, m])
        if old == idx or not self.show[i, m]:
            self.picks[i, m] = idx
            return
        sim = self.sim
        delta = sim.pm[m][idx, sim.actual[m]] - sim.pm[m][old, sim.actual[m]]
        self.mt[i] += delta
        self.mf[sim.match_fecha[m], i] += delta
        self.picks[i, m] = idx

    def set_camp(self, i, team):
        old = int(self.camp[i])
        if old == team:
            return
        if old >= 0:
            self.mt[i] -= PTS_ESP * (self.sim.champ_sim == old)
        if team >= 0:
            self.mt[i] += PTS_ESP * (self.sim.champ_sim == team)
        self.camp[i] = team

    def set_gol(self, i, op):
        old = int(self.gol[i])
        if old == op:
            return
        if old >= 0:
            self.mt[i] -= PTS_ESP * (self.sim.gol_sim == old)
        if op >= 0:
            self.mt[i] += PTS_ESP * (self.sim.gol_sim == op)
        self.gol[i] = op

    def _liq(self, mine, rmax, rcnt, pozo):
        mmax = mine.max(axis=0)
        top = np.maximum(mmax, rmax)
        k = (mine == top[None, :]).sum(axis=0)
        j = np.where(rmax == top, rcnt, 0)
        return np.where(k + j > 0, pozo * k / np.maximum(k + j, 1), 0.0)

    def e_premio(self) -> float:
        p = self._liq(self.mt, self.rmax, self.rcnt, prize.premio_penca)
        for f in range(self.F):
            p = p + self._liq(self.mf[f], self.rfmax[f], self.rfcnt[f], prize.premio_fecha)
        return float(p.mean())

    def result(self) -> dict:
        penca = self._liq(self.mt, self.rmax, self.rcnt, prize.premio_penca)
        fechas = np.zeros(self.S)
        for f in range(self.F):
            fechas = fechas + self._liq(self.mf[f], self.rfmax[f], self.rfcnt[f],
                                        prize.premio_fecha)
        total = penca + fechas
        return {
            "e_total": float(total.mean()),
            "e_penca": float(penca.mean()),
            "e_fechas": float(fechas.mean()),
            "p_penca": float((penca > 0).mean()),
            "p_fecha": float((fechas > 0).mean()),
            "p_algo": float((total > 0).mean()),
        }

    def clone_state(self):
        return (self.picks.copy(), self.show.copy(), self.camp.copy(),
                self.gol.copy(), self.mt.copy(), self.mf.copy(), self.n)

    def restore(self, st):
        self.picks, self.show, self.camp, self.gol, self.mt, self.mf, self.n = \
            st[0].copy(), st[1].copy(), st[2].copy(), st[3].copy(), st[4].copy(), \
            st[5].copy(), st[6]


log(f"construyendo simulador de optimización (S={S_OPT})…")
sim_opt = build_sim(SEED, S_OPT)
log(f"construyendo simulador de evaluación (S={S_EVAL}, seed+{EVAL_SEED_OFFSET})…")
sim_ev = build_sim(SEED + EVAL_SEED_OFFSET, S_EVAL)
fp = FastPort(sim_opt)
fe = FastPort(sim_ev)

# verificación: FastPort == SeasonSimulator en el mismo estado
_test = np.zeros((2, n_matches), dtype=np.int64)
for m in range(n_matches):
    _test[:, m] = int(np.argmax([grids[m].ravel()[i] for i in range(N_SCORES)]))
sim_opt.load_picks(_test)
sim_opt.set_campeon_pick(0, equipo_idx["Peñarol"])
sim_opt.set_campeon_pick(1, equipo_idx["Nacional"])
sim_opt.set_goleador_pick(0, 0)
sim_opt.set_goleador_pick(1, 1)
ref = sim_opt.e_premio_total()
fp.reset(0)
fp.add_row(_test[0], camp=equipo_idx["Peñarol"], gol=0)
fp.add_row(_test[1], camp=equipo_idx["Nacional"], gol=1)
got = fp.e_premio()
assert abs(ref - got) < 1e-6, f"verificación falló: sim={ref} fast={got}"
log(f"verificación FastPort ok (E[premio]={got:,.0f})")

# ---------------------------------------------------------------- candidatos y ancla
candidatos = [build_candidates(g, q, p) for g, q, p in zip(grids, pool_qs, pref)]
cand_idx = [[score_index(*c.pick) for c in cs] for cs in candidatos]
ancla = np.array([max(cs, key=lambda c: c.e_points).pick for cs in candidatos])
ancla_idx = np.array([score_index(a, b) for a, b in ancla], dtype=np.int64)

# fecha 1: columnas congeladas con la v13 para las 12 existentes
f1_cols = [idx_of[p["evento_id"]] for p in v13["picks"]]
f1_scores = {idx_of[p["evento_id"]]: p["scores"] for p in v13["picks"]}
frozen_cols = set(f1_cols)
camp_v13 = [row["campeon_idx"] for row in v13["especiales"]["por_participacion"]]

base_rows = []
for k in range(12):
    row = ancla_idx.copy()
    for m in f1_cols:
        gl, gv = f1_scores[m][k]
        row[m] = score_index(gl, gv)
    base_rows.append(row)


def ascend_row(port: FastPort, i: int, open_cols, do_camp: bool, do_gol: bool,
               max_passes: int = 3) -> float:
    """Ascenso por coordenadas de la fila i (partidos + especiales). In-place."""
    actual = port.e_premio()
    for _ in range(max_passes):
        cambios = 0
        for m in open_cols:
            orig = int(port.picks[i, m])
            mejor, mejor_v = orig, actual
            for cand in cand_idx[m]:
                if cand == orig:
                    continue
                port.set_pick(i, m, cand)
                v = port.e_premio()
                if v > mejor_v:
                    mejor, mejor_v = cand, v
            port.set_pick(i, m, mejor)
            if mejor != orig:
                cambios += 1
                actual = mejor_v
        if do_camp:
            orig = int(port.camp[i])
            mejor, mejor_v = orig, actual
            for t in range(n_teams):
                if t == orig:
                    continue
                port.set_camp(i, t)
                v = port.e_premio()
                if v > mejor_v:
                    mejor, mejor_v = t, v
            port.set_camp(i, mejor)
            if mejor != orig:
                cambios += 1
                actual = mejor_v
        if do_gol:
            orig = int(port.gol[i])
            mejor, mejor_v = orig, actual
            for o in range(n_gol):
                if o == orig:
                    continue
                port.set_gol(i, o)
                v = port.e_premio()
                if v > mejor_v:
                    mejor, mejor_v = o, v
            port.set_gol(i, mejor)
            if mejor != orig:
                cambios += 1
                actual = mejor_v
        if cambios == 0:
            break
    return actual


def eval_port(port: FastPort) -> dict:
    """Reconstruye el estado del portfolio en el evaluador fresco y liquida."""
    fe.reset(0)
    for i in range(port.n):
        fe.add_row(port.picks[i], port.show[i], int(port.camp[i]), int(port.gol[i]))
    return fe.result()


# ---------------------------------------------------------------- base de 12
log("optimizando base de 12 (columnas futuras + goleador; fecha 1 y campeón congelados)…")
open_cols = [m for m in range(n_matches) if m not in frozen_cols]
fp.reset(0)
for k in range(12):
    fp.add_row(base_rows[k], camp=camp_v13[k], gol=-1)

# pasadas estilo strategy.py: filas 1..11 (la 0 es ancla EV), goleador para todas
for pase in range(3):
    total_cambios = 0
    for i in range(1, 12):
        before = fp.picks[i].copy()
        ascend_row(fp, i, open_cols, do_camp=False, do_gol=False, max_passes=1)
        total_cambios += int((fp.picks[i] != before).sum())
    for i in range(12):
        og = int(fp.gol[i])
        ascend_row(fp, i, [], do_camp=False, do_gol=True, max_passes=1)
        total_cambios += int(int(fp.gol[i]) != og)
    log(f"  pasada {pase+1}: {total_cambios} cambios · E[premio] opt={fp.e_premio():,.0f}")
    if total_cambios == 0:
        break

base_state = fp.clone_state()
goles_12 = [gol_names[int(g)] if g >= 0 else None for g in fp.gol]
log(f"goleadores asignados a las 12: {goles_12}")

# curva de prefijos 1..12 (evaluación fresca, pareada)
prefix_curve = []
for k in range(1, 13):
    fe.reset(0)
    for i in range(k):
        fe.add_row(fp.picks[i], fp.show[i], int(fp.camp[i]), int(fp.gol[i]))
    r = fe.result()
    prefix_curve.append({"n": k, **r})
    log(f"  prefijo {k:2d}: E=${r['e_total']:,.0f} · P(algo)={r['p_algo']:.1%}")

# ---------------------------------------------------------------- escenario A
log("escenario A: extras compradas HOY (todo abierto + especiales)…")
sweepA = []
fp.restore(base_state)
r0 = eval_port(fp)
sweepA.append({"n": 12, **r0})
log(f"  n=12: E=${r0['e_total']:,.0f} · P(algo)={r0['p_algo']:.1%} · "
    f"P(penca)={r0['p_penca']:.1%}")
for n in range(13, N_MAX + 1):
    fp.add_row(ancla_idx.copy(), camp=int(np.argmax(p_champ_prior)), gol=int(np.argmax(p_gol)))
    ascend_row(fp, n - 1, list(range(n_matches)), do_camp=True, do_gol=True)
    r = eval_port(fp)
    extra = {"camp": equipo_nombres[int(fp.camp[n-1])],
             "gol": gol_names[int(fp.gol[n-1])] if fp.gol[n-1] >= 0 else None}
    sweepA.append({"n": n, **r, **extra})
    d = r["e_total"] - sweepA[-2]["e_total"]
    log(f"  n={n}: E=${r['e_total']:,.0f} (Δ${d:+,.0f}) · P(algo)={r['p_algo']:.1%} · "
        f"campeón={extra['camp']} · gol={extra['gol']}")
stateA = fp.clone_state()

# ---------------------------------------------------------------- escenario B
log("escenario B: extras compradas MAÑANA (sin especiales, sin el partido de hoy)…")
show_B = np.ones(n_matches, dtype=bool)
show_B[idx_of[2086]] = False   # Cerro Largo vs Juventud, cierra hoy 21:45Z
sweepB = []
fp.restore(base_state)
sweepB.append({"n": 12, **eval_port(fp)})
for n in range(13, N_MAX + 1):
    fp.add_row(ancla_idx.copy(), show_row=show_B.copy(), camp=-1, gol=-1)
    open_b = [m for m in range(n_matches) if show_B[m]]
    ascend_row(fp, n - 1, open_b, do_camp=False, do_gol=False)
    r = eval_port(fp)
    sweepB.append({"n": n, **r})
    d = r["e_total"] - sweepB[-2]["e_total"]
    log(f"  n={n}: E=${r['e_total']:,.0f} (Δ${d:+,.0f}) · P(algo)={r['p_algo']:.1%}")

# ---------------------------------------------------------------- salida
out = {
    "meta": {
        "generado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "s_opt": S_OPT, "s_eval": S_EVAL, "seed": SEED, "n_rivales": R,
        "gol_temper": GOL_TEMPER,
        "pool_q_campeon_top": {equipo_nombres[i]: round(float(pool_q_campeon[i]), 4)
                               for i in np.argsort(-pool_q_campeon)[:5]},
        "pool_q_gol": {n: round(float(pool_q_gol[gol_idx[n]]), 4) for n in gol_names},
        "p_gol_verdad": {n: round(float(p_gol[gol_idx[n]]), 4) for n in gol_names},
        "goleadores_base12": goles_12,
        "campeones_base12": [equipo_nombres[c] for c in camp_v13],
    },
    "prefix": prefix_curve,
    "escenario_A": sweepA,
    "escenario_B": sweepB,
    "extras_A": [{"n": 13 + i,
                  "camp": sweepA[i + 1].get("camp"),
                  "gol": sweepA[i + 1].get("gol")} for i in range(N_MAX - 12)],
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
log(f"guardado {OUT}")

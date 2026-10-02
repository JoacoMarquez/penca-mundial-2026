"""¿Está calibrada P(campeón) del Clausura? Walk-forward contra torneos cortos reales.

El especial "campeón" vale 25 puntos y su probabilidad es 100% modelo propio (no hay
outright del campeonato uruguayo en ninguna casa). Nunca se había validado contra
resultados reales. Esto la valida con el MISMO camino de código de producción:

    ratings   src.clausura.ratings.fit_ratings      (ridge 0.05, SIN decay: así corre prod)
    grillas   src.clausura.picks.build_season_grids (delta en lo jugado, Poisson en el resto)
    campeón   src.clausura.especiales.p_campeon_from_grids → champions_from_results
              (tabla 3/1/0, empate en la cima → sorteo uniforme)

Walk-forward por torneo y checkpoint k (antes de la fecha 1 = k 0, después de la 3, 6,
9, 12): el corte es el inicio del primer partido de una fecha > k. Los ratings se
ajustan SOLO con partidos anteriores al corte (penca-api 2023-2026 + Intermedios desde
Wikipedia con el parser de producción + extras a mano), lo jugado del propio torneo
queda como grilla delta y el resto se sortea.

LIMITACIÓN: sin cuotas. Producción blendea 70/30 mercado/ratings en los partidos que
ya tienen cuota (en la práctica, la fecha próxima: ~8 de los partidos que faltan). No
hay cuotas históricas, así que acá TODO es ratings. Para el Clausura en curso se corre
también la versión con cuotas vivas, para ver cuánto mueve.

Además de la P(campeón) de producción (escala 1.0) se corre una familia "estructural":
ataque y defensa multiplicados por s (s < 1 aplana, s > 1 separa), que propaga la
sobre/sub-confianza a nivel partido a través de la tabla. Y la temperatura post-hoc
P^(1/T) normalizada. Ambas se ajustan por verosimilitud del campeón real.

El n efectivo es el NÚMERO DE TORNEOS (6), no de partidos ni de checkpoints: los 5
checkpoints de un torneo comparten campeón y ratings. El bootstrap remuestrea torneos.

Uso:
    python scripts/calibracion_campeon.py                    # todo, 30.000 sorteos
    python scripts/calibracion_campeon.py --sims 5000 --boot 500   # smoke test
    python scripts/calibracion_campeon.py --refrescar        # re-baja penca-api + Wikipedia
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import pathlib
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clausura.especiales import p_campeon_from_grids  # noqa: E402
from src.clausura.historical import TEMPORADAS, PartidoHistorico, fetch_temporada  # noqa: E402
from src.clausura.intermedio import load_partidos_extra, parse_partidos  # noqa: E402
from src.clausura.picks import build_season_grids  # noqa: E402
from src.clausura.ratings import TeamRatings, fit_ratings  # noqa: E402

log = logging.getLogger("calibracion_campeon")

CACHE = ROOT / "data" / "processed" / "calibracion_campeon"
OUT_DIR = ROOT / "data" / "experimentos"

# Todos los torneos cortos de Primera que lista el penca-api (front/campeonatos).
# historical.TEMPORADAS (lo que usa producción) arranca en 2024; 2 y 4 (2023) existen
# en el API con 120/120 partidos y se suman acá como historia para los ratings.
CAMPEONATOS = {
    2: "Torneo Apertura 2023",
    4: "Torneo Clausura 2023",
    21: "Torneo Apertura 2024",
    25: "Torneo Clausura 2024",
    27: "Torneo Apertura 2025",
    30: "Torneo Clausura 2025",
    41: "Torneo Apertura 2026",
    44: "Torneo Clausura 2026",
}
ACTUAL = 44
# Apertura 2023 se descarta como objetivo: no hay NADA antes en el API (ni el
# Clausura/Intermedio 2022), así que en k=0 los ratings serían todos 0 — un régimen que
# producción nunca ejecuta. Entra solo como historia.
EVALUADOS = [4, 21, 25, 27, 30, 41]

# Campeón según Wikipedia (lead del artículo / tabla final, verificado 2026-10-02).
# El script exige que coincida con el líder ÚNICO de la tabla 3/1/0 armada con los
# resultados del penca-api: si no, la definición del penca no es la de la AUF.
CAMPEON_WIKIPEDIA = {
    2: "Peñarol",      # tabla final: Peñarol 34, Nacional 29
    4: "Liverpool",    # Liverpool 35, Peñarol 25
    21: "Peñarol",     # "El Club Atlético Peñarol fue el campeón de este torneo" (41 vs 34)
    25: "Peñarol",     # ídem (38 vs Nacional 36)
    27: "Liverpool",   # "El Liverpool Fútbol Club fue el campeón" (32 vs Nacional 31)
    30: "Peñarol",     # ídem (35; Nacional 30−3 por sanción del Intermedio 2025)
    41: "Racing",      # "Racing Club de Montevideo fue el campeón" (31 vs D. Maldonado 29)
}

# Dos partidos que el API marca FINALIZADO pero sin goles (resultado_finalizado → None,
# así que fetch_temporada los descarta). Resultados de Wikipedia:
CORRECCIONES = [
    # Apertura 2024 F12, evento 514. Wikipedia lo lista como Liverpool 2:1 Cerro Largo
    # (Belvedere, 12/5/2024); el API tiene la localía al revés. Se respeta la del API.
    PartidoHistorico(21, "Torneo Apertura 2024", "Fecha 12", -514, 514, "Cerro Largo",
                     "Liverpool", 1, 2, False, "2024-05-12T13:00:00+00:00"),
    # Apertura 2025 F13, evento 1112: Peñarol 3-1 Cerro, suspendido al 62' por incidentes
    # de la hinchada de Cerro; la tabla final de Wikipedia se lo cuenta a Peñarol (27 pts)
    # y le resta 1 punto a Cerro. Sin esto Peñarol queda 4º con 24 y no 4º con 27.
    PartidoHistorico(27, "Torneo Apertura 2025", "Fecha 13", -1112, 1112, "Peñarol",
                     "Cerro", 3, 1, False, "2025-04-27T21:15:00+00:00"),
]

# Wikipedia → nombre del penca-api que el ALIASES de producción no cubre (Rampla no
# está en el Clausura 2026, así que producción nunca lo necesitó).
ALIAS_EXTRA = {"Rampla Juniors": "Rampla Jrs"}

P_PLANILLA_VIGENTE = {"Peñarol": 0.38, "Montevideo City Torque": 0.33,
                           "Liverpool": 0.14, "Nacional": 0.10}


# ------------------------------------------------------------------ datos

def _ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _fecha_n(p) -> int:
    return int(str(p.fecha_nombre).split()[-1])


def cargar_pencaapi(refrescar: bool) -> list[PartidoHistorico]:
    path = CACHE / "pencaapi.json"
    if refrescar or not path.exists():
        todos = []
        for cid, nombre in CAMPEONATOS.items():
            todos.extend(fetch_temporada(cid, nombre))
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(p) for p in todos], ensure_ascii=False))
    return [PartidoHistorico(**d) for d in json.loads(path.read_text())]


def _wikitext(page: str, refrescar: bool) -> str:
    """Wikitext cacheado. curl y no httpx: Wikimedia le devuelve 403 (robot policy)
    al httpx desde esta máquina con cualquier UA; a curl no."""
    path = CACHE / ("wiki_" + page.replace(" ", "_") + ".txt")
    if refrescar or not path.exists():
        out = subprocess.run(
            ["curl", "-s", "-m", "30", "-A", "penca-clausura/1.0 (research script)", "-G",
             "https://es.wikipedia.org/w/api.php", "--data-urlencode", "action=parse",
             "--data-urlencode", f"page={page}", "--data-urlencode", "prop=wikitext",
             "--data-urlencode", "format=json", "--data-urlencode", "formatversion=2"],
            capture_output=True, text=True, check=True).stdout
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.loads(out)["parse"]["wikitext"])
        time.sleep(2)
    return path.read_text()


def cargar_intermedios(equipos: set[str], refrescar: bool) -> list[PartidoHistorico]:
    """Intermedios 2023-2026 con el parser de producción (src.clausura.intermedio)."""
    out = []
    validos = equipos | set(ALIAS_EXTRA)
    for y in (2023, 2024, 2025, 2026):
        ps = parse_partidos(_wikitext(f"Torneo Intermedio {y}", refrescar), validos, year=y)
        ps = [replace(p, campeonato=f"Torneo Intermedio {y}", campeonato_id=-y,
                      evento_id=-(y * 1000 + i), fecha_id=-(y * 1000 + i),
                      local=ALIAS_EXTRA.get(p.local, p.local),
                      visitante=ALIAS_EXTRA.get(p.visitante, p.visitante))
              for i, p in enumerate(ps)]
        log.info("Intermedio %d: %d partidos", y, len(ps))
        out.extend(ps)
    return out


# ------------------------------------------------------------------ modelo

def escalar(r: TeamRatings, s: float) -> TeamRatings:
    """Ataque/defensa × s (s=1 es producción). μ y localía intactos."""
    return TeamRatings(equipos=r.equipos,
                       ataque={k: v * s for k, v in r.ataque.items()},
                       defensa={k: v * s for k, v in r.defensa.items()},
                       mu=r.mu, ventaja_local=r.ventaja_local)


def p_campeon(eventos: list[dict], ratings: TeamRatings, jugados: dict[int, tuple[int, int]],
              equipos: list[str], n_sims: int, odds_by_evento: dict | None = None) -> np.ndarray:
    """P(campeón) por el camino de producción: build_season_grids → p_campeon_from_grids."""
    grids, _, _, _ = build_season_grids(eventos, ratings, odds_by_evento or {}, jugados)
    idx = {e: i for i, e in enumerate(equipos)}
    local_de = np.array([idx[ev["local"]] for ev in eventos])
    visita_de = np.array([idx[ev["visitante"]] for ev in eventos])
    return p_campeon_from_grids(grids, local_de, visita_de, len(equipos), n_sims=n_sims)


def tabla(eventos: list[dict], jugados: dict[int, tuple[int, int]], equipos: list[str],
          ajustes: dict[str, int] | None = None) -> np.ndarray:
    idx = {e: i for i, e in enumerate(equipos)}
    pts = np.zeros(len(equipos))
    for ev in eventos:
        r = jugados.get(ev["evento_id"])
        if r is None:
            continue
        gl, gv = r
        pts[idx[ev["local"]]] += 3 if gl > gv else (1 if gl == gv else 0)
        pts[idx[ev["visitante"]]] += 3 if gv > gl else (1 if gl == gv else 0)
    for e, a in (ajustes or {}).items():
        pts[idx[e]] += a
    return pts


def temperar(p: np.ndarray, T: float, piso: float) -> np.ndarray:
    q = np.maximum(p, piso) ** (1.0 / T)
    return q / q.sum()


# ------------------------------------------------------------------ métricas

def metricas(p: np.ndarray, c: int, piso: float) -> dict:
    y = np.zeros(len(p)); y[c] = 1.0
    pc = max(float(p[c]), piso)
    return {
        "logloss": -math.log(pc),
        "brier": float(((p - y) ** 2).sum()),
        "p_campeon_real": float(p[c]),
        "esperado_si_calibrado": float((p ** 2).sum()),   # E[p_campeón] bajo el propio modelo
        "rank": int((p > p[c]).sum() + 1),
        "clipeado": bool(p[c] < piso),
    }


BINS = [0.0, 0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.90, 1.0001]


def confiabilidad(pares: list[tuple[float, int]]) -> list[dict]:
    p = np.array([a for a, _ in pares]); y = np.array([b for _, b in pares])
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if n == 0:
            continue
        out.append({"bin": f"[{lo:.2f},{min(hi, 1):.2f})", "n": n, "p_media": float(p[m].mean()),
                    "freq_obs": float(y[m].mean()), "campeones": int(y[m].sum()),
                    "esperados": float(p[m].sum())})
    return out


# ------------------------------------------------------------------ main

def eventos_de(partidos: list) -> list[dict]:
    return sorted(({"evento_id": p.evento_id, "local": p.local, "visitante": p.visitante,
                    "fecha_n": _fecha_n(p), "inicio_utc": p.inicio_utc} for p in partidos),
                  key=lambda e: e["inicio_utc"])


def correr_historico(args, api, inter, extra) -> list[dict]:
    corr_ids = {c.evento_id for c in CORRECCIONES}
    base = [p for p in api if p.evento_id not in corr_ids] + CORRECCIONES
    historia = base + inter + extra
    filas = []
    for cid in EVALUADOS:
        del_t = [p for p in base if p.campeonato_id == cid]
        assert len(del_t) == 120, (cid, len(del_t))
        evs = eventos_de(del_t)
        equipos = sorted({e["local"] for e in evs} | {e["visitante"] for e in evs})
        final = {p.evento_id: (p.goles_local, p.goles_visitante) for p in del_t}
        pts_fin = tabla(evs, final, equipos)
        orden = np.argsort(-pts_fin)
        assert pts_fin[orden[0]] > pts_fin[orden[1]], f"{cid}: empate en la cima"
        campeon = equipos[orden[0]]
        assert campeon == CAMPEON_WIKIPEDIA[cid], (cid, campeon, CAMPEON_WIKIPEDIA[cid])
        c = equipos.index(campeon)
        for k in args.checkpoints:
            corte = min(_ts(e["inicio_utc"]) for e in evs if e["fecha_n"] > k)
            jugados = {e["evento_id"]: final[e["evento_id"]] for e in evs
                       if _ts(e["inicio_utc"]) < corte}
            train = [p for p in historia if _ts(p.inicio_utc) < corte]
            t0 = time.time()
            rt = fit_ratings(train)        # defaults de producción: ridge 0.05, sin decay
            sin_datos = [e for e in equipos if not any(e in (p.local, p.visitante) for p in train)]
            p_s = {s: p_campeon(evs, escalar(rt, s), jugados, equipos, args.sims)
                   for s in args.escalas}
            # baselines
            pts = tabla(evs, jugados, equipos)
            p_unif = np.full(len(equipos), 1 / len(equipos))
            p_tabla = (pts + 1) / (pts + 1).sum()
            p_iguales = p_s.get(0.0)
            if p_iguales is None:
                p_iguales = p_campeon(evs, escalar(rt, 0.0), jugados, equipos, args.sims)
            filas.append({
                "campeonato_id": cid, "torneo": CAMPEONATOS[cid], "k": k,
                "corte_utc": datetime.fromtimestamp(corte, timezone.utc).isoformat(),
                "jugados": len(jugados), "n_train": len(train), "sin_datos": sin_datos,
                "equipos": equipos, "campeon": campeon, "c": c,
                "pts": pts.tolist(), "lider": equipos[int(np.argmax(pts))] if pts.any() else None,
                "p": {f"{s:g}": v.tolist() for s, v in p_s.items()},
                "p_uniforme": p_unif.tolist(), "p_tabla": p_tabla.tolist(),
                "p_iguales": p_iguales.tolist(),
            })
            top = np.argsort(-p_s[1.0])[:3]
            log.info("%s k=%2d: campeón %s P=%.3f (rank %d) | top %s | %.1fs",
                     CAMPEONATOS[cid], k, campeon, p_s[1.0][c],
                     int((p_s[1.0] > p_s[1.0][c]).sum() + 1),
                     ", ".join(f"{equipos[t]} {p_s[1.0][t]:.2f}" for t in top), time.time() - t0)
    return filas


def tests_h0(sub: list[dict], Ts: np.ndarray, piso: float, rng, B: int) -> dict:
    """Tests bajo H0 = "el modelo está calibrado" (el campeón sale ∝ p).

    Con 6 torneos el bootstrap no paramétrico es degenerado: si los 6 favoritos
    tardíos ganaron, ninguna remuestra tiene un batacazo y el IC de T se pega al
    borde. Estos tests preguntan lo correcto — ¿qué tan raro es lo observado si p
    fuera la verdad? — y su varianza sale del propio modelo:
      * z_logloss: Σ(−log p_c − H(p)) / √ΣVar. >0 ⇒ más sorpresa que la prometida
        (sobreconfiado); <0 ⇒ menos (subconfiado).
      * z_pcamp:   Σ(p_c − Σp²) / √Σ(Σp³ − (Σp²)²). >0 ⇒ subconfiado.
      * T* nulo:   distribución de T* sorteando campeones ∝ p (bootstrap paramétrico);
        p-valores de que T* nulo quede del lado del observado.
    Solo son válidos por checkpoint (en "todos" los checkpoints de un torneo comparten
    campeón y no son independientes: se reportan igual, marcados)."""
    ll_d, ll_v, pc_d, pc_v = 0.0, 0.0, 0.0, 0.0
    P = []
    for f in sub:
        p = np.maximum(np.array(f["p"]["1"]), 0.0)
        lp = np.log(np.maximum(p, piso))
        H = float(-(p * lp).sum())
        ll_d += -lp[f["c"]] - H
        ll_v += float((p * lp ** 2).sum()) - H ** 2
        s2 = float((p ** 2).sum())
        pc_d += p[f["c"]] - s2
        pc_v += float((p ** 3).sum()) - s2 ** 2
        P.append(p / p.sum())
    # T* nulo
    logT = np.array([[np.log(temperar(p, T, piso)) for T in Ts] for p in P])   # (n, nT, eq)
    obs = np.array([f["c"] for f in sub])
    T_obs = Ts[int(np.argmin(-logT[np.arange(len(sub)), :, obs].sum(0)))]
    nulos = np.empty(B)
    for b in range(B):
        cs = np.array([rng.choice(len(p), p=p) for p in P])
        nulos[b] = Ts[int(np.argmin(-logT[np.arange(len(sub)), :, cs].sum(0)))]
    return {
        "z_logloss": float(ll_d / math.sqrt(ll_v)) if ll_v > 0 else None,
        "z_pcamp": float(pc_d / math.sqrt(pc_v)) if pc_v > 0 else None,
        "T_nulo_pct": [float(np.percentile(nulos, q)) for q in (2.5, 50, 97.5)],
        "p_T_nulo_menor_igual": float((nulos <= T_obs + 1e-12).mean()),
        "p_T_nulo_mayor_igual": float((nulos >= T_obs - 1e-12).mean()),
    }


def resumir(filas: list[dict], args) -> dict:
    piso = 0.5 / args.sims
    Ts = np.exp(np.linspace(math.log(0.25), math.log(4.0), 161))
    torneos = sorted({f["campeonato_id"] for f in filas})
    ks = sorted({f["k"] for f in filas})

    # matriz (torneo, k) × T de log-loss para el bootstrap
    LL = {(f["campeonato_id"], f["k"]): np.array([
        -math.log(temperar(np.array(f["p"]["1"]), T, piso)[f["c"]]) for T in Ts]) for f in filas}
    LS = {(f["campeonato_id"], f["k"]): np.array([
        -math.log(max(f["p"][f"{s:g}"][f["c"]], piso)) for s in args.escalas]) for f in filas}

    def mejor(sub_keys, M, grid):
        tot = sum(M[key] for key in sub_keys)
        return float(grid[int(np.argmin(tot))]), tot

    rng = np.random.default_rng(args.seed_boot)

    def bootstrap(k_filtro, M, grid):
        res = []
        for _ in range(args.boot):
            samp = rng.choice(torneos, size=len(torneos), replace=True)
            keys = [(t, k) for t in samp for k in ks if k in k_filtro]
            res.append(mejor(keys, M, grid)[0])
        return [float(np.percentile(res, 2.5)), float(np.percentile(res, 50)),
                float(np.percentile(res, 97.5))]

    def perfil_ic(tot, grid):
        """IC por verosimilitud de perfil (Δ = 1.92). Supone obs. independientes: NO lo
        son (5 checkpoints por torneo), así que es optimista; el bootstrap manda."""
        ok = tot <= tot.min() + 1.92
        return [float(grid[ok].min()), float(grid[ok].max())]

    escalas = np.array(args.escalas)
    por_k = []
    for k in ks + ["todos"]:
        kf = ks if k == "todos" else [k]
        sub = [f for f in filas if f["k"] in kf]
        keys = [(f["campeonato_id"], f["k"]) for f in sub]
        fila = {"k": k, "n_obs": len(sub), "n_torneos": len({f["campeonato_id"] for f in sub})}
        for nombre, getp in [("modelo", lambda f: np.array(f["p"]["1"])),
                             ("uniforme", lambda f: np.array(f["p_uniforme"])),
                             ("tabla_pts+1", lambda f: np.array(f["p_tabla"])),
                             ("mc_equipos_iguales", lambda f: np.array(f["p_iguales"]))]:
            ms = [metricas(getp(f), f["c"], piso) for f in sub]
            fila[nombre] = {
                "logloss": float(np.mean([m["logloss"] for m in ms])),
                "brier": float(np.mean([m["brier"] for m in ms])),
                "p_campeon_real": float(np.mean([m["p_campeon_real"] for m in ms])),
                "esperado_si_calibrado": float(np.mean([m["esperado_si_calibrado"] for m in ms])),
                "rank_medio": float(np.mean([m["rank"] for m in ms])),
                "ranks": [m["rank"] for m in ms],
                "clipeados": int(sum(m["clipeado"] for m in ms)),
            }
        # diferencia observado − esperado de P(campeón real), con SE entre torneos
        d = collections.defaultdict(list)
        for f in sub:
            p = np.array(f["p"]["1"])
            d[f["campeonato_id"]].append(p[f["c"]] - (p ** 2).sum())
        por_t = np.array([np.mean(v) for v in d.values()])
        fila["obs_menos_esp"] = {"media": float(por_t.mean()),
                                 "se_entre_torneos": float(por_t.std(ddof=1) / math.sqrt(len(por_t)))
                                 if len(por_t) > 1 else None}
        T_hat, totT = mejor(keys, LL, Ts)
        s_hat, totS = mejor(keys, LS, escalas)
        fila["T_opt"] = T_hat
        fila["T_ic_perfil"] = perfil_ic(totT, Ts)
        fila["T_ic_boot"] = bootstrap(kf, LL, Ts)
        fila["logloss_T_opt"] = float(totT.min() / len(sub))
        fila["s_opt"] = s_hat
        fila["s_ic_boot"] = bootstrap(kf, LS, escalas)
        fila["logloss_por_s"] = {f"{s:g}": float(v / len(sub)) for s, v in zip(escalas, totS)}
        fila["confiabilidad"] = confiabilidad(
            [(float(pi), int(i == f["c"])) for f in sub for i, pi in enumerate(f["p"]["1"])])
        fila.update(tests_h0(sub, Ts, piso, rng, args.boot))
        por_k.append(fila)
    return {"por_checkpoint": por_k, "T_grid": [float(Ts[0]), float(Ts[-1])], "piso": piso}


def correr_actual(args, api, inter, extra, resumen) -> dict:
    """Clausura 2026 en curso: (a) datos exactos de producción, (b) + cuotas vivas,
    (c) datos del walk-forward, y su versión temperada con la T estimada."""
    from src.clausura.picks import flat_eventos, match_odds
    from src.clausura.sync import build_config
    # fixture vivo (sync.build_config: solo lectura, NO pisa config/clausura2026.yaml,
    # que en local está viejo). Se cachea: el API corta a las ~200 requests (429).
    cfg_path = CACHE / "clausura2026_cfg.json"
    if args.refrescar or not cfg_path.exists():
        try:
            cfg_path.write_text(json.dumps(build_config(), ensure_ascii=False, default=str))
        except Exception as e:
            # Fallback: el config versionado (puede tener fechas viejas, pero los cruces
            # y evento_id no cambian; para la TABLA solo importan los cruces). Se exige
            # que cada partido jugado exista con los mismos equipos.
            from src.clausura.picks import load_config
            log.warning("build_config falló (%s) — uso config/clausura2026.yaml", e)
            cfg = load_config()
            por_id = {ev["evento_id"]: ev for f in cfg["fechas"].values() for ev in f["eventos"]}
            for p in api:
                if p.campeonato_id == ACTUAL:
                    ev = por_id[p.evento_id]
                    assert (ev["local"], ev["visitante"]) == (p.local, p.visitante), p
            assert len(por_id) == 120
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, default=str))
    cfg = json.loads(cfg_path.read_text())
    eventos = flat_eventos(cfg)
    equipos = [cfg["equipos"][k] for k in sorted(cfg["equipos"], key=int)]
    resultados = {p.evento_id: (p.goles_local, p.goles_visitante)
                  for p in api if p.campeonato_id == ACTUAL}
    jugados_pl = [p for p in api if p.campeonato_id == ACTUAL]
    # (a) lo que ve ensure_ratings: TEMPORADAS (2024-2026, sin las correcciones) +
    #     Intermedio 2026 + extras + lo jugado del Clausura
    prod_base = [p for p in api if p.campeonato_id in TEMPORADAS] + \
                [p for p in inter if p.campeonato == "Torneo Intermedio 2026"] + extra + jugados_pl
    rt_prod = fit_ratings(prod_base)
    out = {"jugados": len(resultados), "equipos": equipos, "pts": tabla(
        eventos, resultados, equipos).tolist(), "n_train_prod": len(prod_base)}
    out["p_prod_ratings"] = p_campeon(eventos, rt_prod, resultados, equipos, args.sims).tolist()
    try:
        from src.clausura.odds import fetch_primera_odds
        odds_by = match_odds(eventos, fetch_primera_odds())
        odds_by = {k: v for k, v in odds_by.items() if k not in resultados}
        out["eventos_con_cuota"] = len(odds_by)
        out["p_prod_mercado"] = p_campeon(eventos, rt_prod, resultados, equipos, args.sims,
                                          odds_by).tolist()
    except Exception as e:                     # el ES puede estar caído: se reporta
        log.warning("sin cuotas vivas: %s", e)
        out["p_prod_mercado"] = None
    corr_ids = {c.evento_id for c in CORRECCIONES}
    wf_base = [p for p in api if p.evento_id not in corr_ids] + CORRECCIONES + inter + extra
    rt_wf = fit_ratings(wf_base)
    out["p_walkforward"] = p_campeon(eventos, rt_wf, resultados, equipos, args.sims).tolist()
    out["p_por_escala"] = {f"{s:g}": p_campeon(eventos, escalar(rt_prod, s), resultados,
                                                equipos, args.sims).tolist()
                           for s in args.escalas}
    piso = 0.5 / args.sims
    todos = next(f for f in resumen["por_checkpoint"] if f["k"] == "todos")
    base = np.array(out["p_prod_mercado"] or out["p_prod_ratings"])
    out["temperado"] = {f"T={T:.2f}": temperar(base, T, piso).tolist()
                        for T in sorted({todos["T_opt"], *todos["T_ic_boot"][::2], 1.0})}
    out["planilla_vigente_reportada"] = P_PLANILLA_VIGENTE
    return out


def imprimir(resumen: dict, actual: dict | None) -> None:
    if not resumen["por_checkpoint"]:
        return _imprimir_actual(actual)
    print("\n=== Métricas por checkpoint (media sobre torneos; n = torneos) ===")
    print(f"{'k':>5} {'n':>3} | {'modelo LL':>9} {'unif':>6} {'tabla':>6} {'iguales':>7} | "
          f"{'Brier mod':>9} {'unif':>6} {'tabla':>6} {'igual':>6} | {'P(camp)':>7} {'esp':>6} "
          f"{'rank':>5} | {'T*':>5} {'IC boot T':>13} | {'s*':>4} {'IC boot s':>11}")
    for f in resumen["por_checkpoint"]:
        m, u, t, q = f["modelo"], f["uniforme"], f["tabla_pts+1"], f["mc_equipos_iguales"]
        print(f"{str(f['k']):>5} {f['n_obs']:>3} | {m['logloss']:9.3f} {u['logloss']:6.3f} "
              f"{t['logloss']:6.3f} {q['logloss']:7.3f} | {m['brier']:9.3f} {u['brier']:6.3f} "
              f"{t['brier']:6.3f} {q['brier']:6.3f} | {m['p_campeon_real']:7.3f} "
              f"{m['esperado_si_calibrado']:6.3f} {m['rank_medio']:5.2f} | {f['T_opt']:5.2f} "
              f"[{f['T_ic_boot'][0]:.2f},{f['T_ic_boot'][2]:.2f}] | {f['s_opt']:4.2f} "
              f"[{f['s_ic_boot'][0]:.2f},{f['s_ic_boot'][2]:.2f}]")
    print("\n=== Tests bajo H0 'calibrado' (z>0 en logloss = sobreconfiado; z>0 en P = subconfiado) ===")
    for f in resumen["por_checkpoint"]:
        o = f["obs_menos_esp"]
        se = f"{o['se_entre_torneos']:.3f}" if o["se_entre_torneos"] is not None else "—"
        print(f"  k={str(f['k']):>5}: P(camp) obs−esp {o['media']:+.3f} (SE entre torneos {se}) | "
              f"z_logloss {f['z_logloss']:+.2f} | z_P {f['z_pcamp']:+.2f} | T* {f['T_opt']:.2f} vs "
              f"nulo [{f['T_nulo_pct'][0]:.2f}, {f['T_nulo_pct'][2]:.2f}] "
              f"(P[T*nulo ≤ obs] = {f['p_T_nulo_menor_igual']:.3f}, "
              f"P[≥] = {f['p_T_nulo_mayor_igual']:.3f})")
    todos = next(f for f in resumen["por_checkpoint"] if f["k"] == "todos")
    print("\n=== Confiabilidad (todos los checkpoints, todos los equipos) ===")
    for b in todos["confiabilidad"]:
        print(f"  {b['bin']:>13} n={b['n']:4d}  p media {b['p_media']:.3f}  obs {b['freq_obs']:.3f}"
              f"  (campeones {b['campeones']} vs esperados {b['esperados']:.2f})")
    print("\nlog-loss por escala s (todos):", {k: round(v, 3) for k, v in todos["logloss_por_s"].items()})
    _imprimir_actual(actual)


def _imprimir_actual(actual: dict | None) -> None:
    if actual:
        print("\n=== Clausura 2026 en curso ===")
        eq = actual["equipos"]
        base = np.array(actual["p_prod_mercado"] or actual["p_prod_ratings"])
        orden = np.argsort(-base)[:6]
        cols = [("ratings(prod)", actual["p_prod_ratings"]), ("+mercado", actual["p_prod_mercado"]),
                ("walk-fwd data", actual["p_walkforward"])] + \
               [(k, v) for k, v in actual["temperado"].items()]
        print(f"{'equipo':>24} {'pts':>4} " + " ".join(f"{c:>13}" for c, _ in cols))
        for i in orden:
            print(f"{eq[i]:>24} {actual['pts'][i]:4.0f} " + " ".join(
                f"{(v[i] if v is not None else float('nan')):13.3f}" for _, v in cols))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for ruidoso in ("src.clausura.intermedio", "httpx", "src.clausura.odds"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sims", type=int, default=30_000,
                    help="sorteos por P(campeón) (producción: 30.000 en p_campeon_from_grids)")
    ap.add_argument("--checkpoints", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0, 3, 6, 9, 12])
    ap.add_argument("--escalas", type=lambda s: [float(x) for x in s.split(",")],
                    default=[0.0, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0],
                    help="multiplicadores de ataque/defensa (1 = producción, 0 = equipos iguales)")
    ap.add_argument("--boot", type=int, default=5000, help="réplicas del bootstrap por torneo")
    ap.add_argument("--seed-boot", type=int, default=20261002)
    ap.add_argument("--refrescar", action="store_true", help="re-baja penca-api y Wikipedia")
    ap.add_argument("--sin-actual", action="store_true", help="no corre el Clausura 2026")
    ap.add_argument("--out", type=pathlib.Path,
                    default=OUT_DIR / f"calibracion_campeon_{datetime.now():%Y%m%d}.json")
    args = ap.parse_args()
    if 1.0 not in args.escalas:
        args.escalas.append(1.0)

    api = cargar_pencaapi(args.refrescar)
    equipos_api = {p.local for p in api} | {p.visitante for p in api}
    inter = cargar_intermedios(equipos_api, args.refrescar)
    extra = load_partidos_extra(set())   # final del Intermedio 2026 (config/partidos_extra.yaml)
    log.info("dataset: %d partidos penca-api, %d Intermedio, %d extra", len(api), len(inter), len(extra))

    filas = correr_historico(args, api, inter, extra)
    resumen = resumir(filas, args)
    imprimir(resumen, None)
    actual = None
    if not args.sin_actual:
        try:
            actual = correr_actual(args, api, inter, extra, resumen)
            imprimir({"por_checkpoint": []}, actual)
        except Exception as e:                 # el histórico no se pierde por un 429
            log.error("no se pudo correr el Clausura en curso: %s", e)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
        "evaluados": {cid: CAMPEONATOS[cid] for cid in EVALUADOS},
        "descartados": {2: "Apertura 2023: sin historia previa en el API (k=0 con ratings nulos)"},
        "correcciones": [asdict(c) for c in CORRECCIONES],
        "resumen": resumen, "filas": filas, "actual": actual,
    }, ensure_ascii=False, indent=1))
    print(f"\nescrito {args.out}")


if __name__ == "__main__":
    main()

"""Monte Carlo de la temporada LUB y liquidación de premios de la penca.

Formato simulado (26/27 = 25/26):
    Regular      22 fechas × 6 (doble round-robin de 12)
    Liguilla     top-6 de la regular, doble round-robin (10 fechas × 3 partidos)
    Reclasif.    bottom-6, doble round-robin (mismas 10 fechas × 3 partidos)
    Playoffs     liguilla 1-6 + reclasificatorio 1-2 → cuartos (bo5), semis (bo5),
                 final (bo7) en el Antel Arena (neutral); localía para el mejor
                 sembrado (2-2-1). Liguilla y reclasificatorio arrastran el puntaje
                 de la regular. Formato 26/27 confirmado igual al 25/26 (sin play-in).
Fechas de la penca: 22 regulares + 10 de liguilla + 5 de playoffs (cuartos 1-2,
cuartos 3-5, semis 1-2, semis 3-5, finales) = 37 — igual que 25/26.

Cada partido es un SLOT (fecha, orden). En liguilla y playoffs los equipos del slot
cambian de sorteo en sorteo, así que todo se calcula por sorteo: μ (S,), resultado
(S,), picks de los rivales (S, R). Un slot que no se juega (serie cerrada) da 0.

Incertidumbre de los ratings: cada sorteo le suma a cada equipo un δ ~ N(0, τ) fijo
para toda la temporada (τ se achica con los partidos jugados). Sin eso, P(campeón) y
la correlación entre partidos del mismo equipo quedan subestimadas.

Premios (Art. 7 y 9): fecha → $1.000 al máximo de la fecha; penca → $50.000 al
máximo total (partidos + especiales). Empates reparten por partes iguales.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.lub.model import Ratings, class_probs, margin_pmf
from src.lub.scoring import KERNEL, N_BANDAS, N_CLASES

PREMIO_PENCA = 50_000.0
PREMIO_FECHA = 1_000.0
PTS_ESPECIAL = 25
PTS_EXACTO = 10
# P(exacto | acertó la clase), medido en la penca 37: 51 exactos / 5.809 aciertos de clase.
# Parece poco pero son ~1 exacto por fecha en el pool, y +10 decide la fecha.
RHO_EXACTO = 0.0088
# Habilidad del pool (ver simular_rivales). Calibrado con src.lub.pit sobre la 25/26:
# κ=0,28 → media de los plenos 466 (real 466), PIT temporada 0,57, PIT fechas 0,50.
KAPPA_POOL = 0.28
MARGEN_GRID = np.arange(-80, 81)


@dataclass
class Slot:
    fecha: str
    fase: str                       # regular | liguilla | playoff
    local: str | None = None        # fijo si se conoce (regular); None → lo define el sorteo
    visitante: str | None = None
    preferencial: bool = False
    evento_id: int | None = None
    resultado: tuple[int, int] | None = None   # ya jugado


@dataclass
class Config:
    n_sims: int = 4000
    n_rivales: int = 320
    seed: int = 20261012
    tau0: float = 4.0               # sd preseason del error de rating (pts de margen)
    tau_k: float = 10.0             # partidos que "valen" como la prior
    ruido_politica: float = 0.35    # dispersión de los picks futuros propios (ver politica_futura)


# ----------------------------------------------------------------------------------

class ProbCache:
    """Probabilidades de clase por μ redondeado a 0,25 (la PMF es cara)."""

    def __init__(self, rt: Ratings, paso: float = 0.25):
        self.rt, self.paso = rt, paso
        self.mus = np.arange(-60, 60 + paso, paso)
        self.tabla = np.array([class_probs(margin_pmf(m, rt.sigma_pred, rt.resid, rt.params.min_resid))
                               for m in self.mus])          # (M, 10)

    def idx(self, mu: np.ndarray) -> np.ndarray:
        return np.clip(np.round((mu + 60) / self.paso).astype(int), 0, len(self.mus) - 1)

    def probs(self, mu: np.ndarray) -> np.ndarray:
        return self.tabla[self.idx(mu)]


def sample_classes(probs: np.ndarray, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    """Clases muestreadas fila a fila. probs (S, 10) → (S,) ó (S, n) si n."""
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    if n is None:
        u = rng.random(probs.shape[0])
        return (u[:, None] > cdf).sum(1).astype(np.int8)
    u = rng.random((probs.shape[0], n), dtype=np.float32)
    out = np.zeros(u.shape, dtype=np.int8)
    for c in range(N_CLASES - 1):            # 9 comparaciones (S, n): sin bucle por fila
        out += (u > cdf[:, c:c + 1]).astype(np.int8)
    return out


def margen_de_clase(c: np.ndarray, probs_pmf_mu: np.ndarray | None = None) -> np.ndarray:
    """Margen representativo (para standings/desempates): centro de la banda."""
    centros = np.array([3, 8, 13, 18, 25])
    m = centros[c % N_BANDAS]
    return np.where(c < N_BANDAS, m, -m)


# ----------------------------------------------------------------------------------

@dataclass
class Sorteo:
    """Resultado de simular la temporada: todo lo que necesita la liquidación."""
    slots: list[Slot]
    fechas: list[str]
    fecha_de_slot: np.ndarray                 # (J,)
    jugado: np.ndarray                        # (S, J) bool — el slot se disputa
    clase: np.ndarray                         # (S, J) int8 resultado
    probs: np.ndarray                         # (S, J, 10) float32 prob. del modelo (para picks)
    pref: np.ndarray                          # (J,) bool
    campeon: np.ndarray                       # (S,) índice de equipo
    equipos: list[str]
    riv_fecha_max: np.ndarray = field(default=None)   # (S, F)
    riv_fecha_cnt: np.ndarray = field(default=None)   # (S, F)
    riv_total_max: np.ndarray = field(default=None)   # (S,)
    riv_total_cnt: np.ndarray = field(default=None)   # (S,)


def simular(rt: Ratings, slots_regulares: list[Slot], cfg: Config,
            partidos_jugados_por_equipo: dict[str, int] | None = None,
            mu_override: dict[int, float] | None = None) -> Sorteo:
    """Simula resultados de toda la temporada (regular fija + liguilla/playoffs por sorteo)."""
    rng = np.random.default_rng(cfg.seed)
    S = cfg.n_sims
    cache = ProbCache(rt)
    equipos = sorted({s.local for s in slots_regulares} | {s.visitante for s in slots_regulares})
    ix = {e: i for i, e in enumerate(equipos)}
    T = len(equipos)
    jug = partidos_jugados_por_equipo or {}
    tau = np.array([cfg.tau0 * math.sqrt(cfg.tau_k / (cfg.tau_k + jug.get(e, 0))) for e in equipos])
    base = np.array([rt.r.get(e, rt.params.r_nuevo) for e in equipos])
    rating = base[None, :] + rng.normal(0, 1, (S, T)) * tau[None, :]      # (S, T)

    slots: list[Slot] = []
    clase_cols, probs_cols, jug_cols = [], [], []
    wins = np.zeros((S, T))
    dif = np.zeros((S, T))

    k_mu = rt.params.escala_mu

    def jugar(h: np.ndarray, a: np.ndarray, slot: Slot, activo: np.ndarray | None = None,
              mu_fijo: float | None = None, neutral: bool = False) -> np.ndarray:
        """Juega un slot con local h (S,) y visitante a (S,). Devuelve la clase (S,).

        La media pasa por la MISMA escala que Ratings.mu (escala_mu): antes se armaba a
        mano y la corrección de sub-confianza no llegaba a la simulación. `neutral`:
        sin localía (las finales se juegan en el Antel Arena)."""
        hca = 0.0 if neutral else rt.hca
        mu_modelo = k_mu * (hca + base[h] - base[a])     # lo que "ve" el modelo al pickear
        delta = (rating[np.arange(S), h] - base[h]) - (rating[np.arange(S), a] - base[a])
        mu_real = mu_modelo + delta
        if mu_fijo is not None:                          # cuota de mercado disponible
            mu_real = mu_real - mu_modelo + mu_fijo
            mu_modelo = np.full(S, mu_fijo)
        p_ver = cache.probs(mu_modelo).astype(np.float32)
        if slot.resultado is not None:
            from src.lub.scoring import clase as clase_de
            c = np.full(S, clase_de(*slot.resultado), dtype=np.int8)
        else:
            c = sample_classes(cache.probs(mu_real), rng)
        act = np.ones(S, bool) if activo is None else activo
        m = margen_de_clase(c)
        np.add.at(wins, (np.arange(S)[act], h[act]), (c[act] < N_BANDAS))
        np.add.at(wins, (np.arange(S)[act], a[act]), (c[act] >= N_BANDAS))
        np.add.at(dif, (np.arange(S)[act], h[act]), m[act])
        np.add.at(dif, (np.arange(S)[act], a[act]), -m[act])
        slots.append(slot)
        clase_cols.append(c)
        probs_cols.append(p_ver)
        jug_cols.append(act)
        return c

    # ---- regular: equipos fijos
    for s in slots_regulares:
        h = np.full(S, ix[s.local]); a = np.full(S, ix[s.visitante])
        mf = (mu_override or {}).get(s.evento_id) if s.evento_id else None
        jugar(h, a, s, mu_fijo=mf)

    def ranking(w: np.ndarray, d: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
        """(S, T) orden de equipos por victorias, desempate por diferencia (+ ruido)."""
        score = w * 1000 + d + rng.random(w.shape) * 0.1
        if subset is not None:
            score = np.where(subset, score, -1e9)
        return np.argsort(-score, axis=1)

    orden = ranking(wins, dif)
    top6, bot6 = orden[:, :6], orden[:, 6:12]

    # ---- liguilla y reclasificatorio: doble round-robin de 6 (10 fechas × 3 + 3)
    rr = _round_robin(6)
    w_l, d_l = wins.copy(), dif.copy()     # la liguilla arrastra lo de la regular
    wins[:], dif[:] = 0, 0
    for f, ronda in enumerate(rr):
        nombre = f"Fecha {f + 1} (Liguilla/ Reclasificatorio)"
        for grupo in (top6, bot6):
            for (i, j) in ronda:
                jugar(grupo[:, i], grupo[:, j], Slot(nombre, "liguilla"))
    w_lig = w_l + wins
    d_lig = d_l + dif
    in_top = np.zeros((S, T), bool); in_top[np.arange(S)[:, None], top6] = True
    lig = ranking(w_lig, d_lig, in_top)[:, :6]
    # el reclasificatorio TAMBIÉN arrastra el puntaje de la regular (formato 26/27:
    # "acarrearán el puntaje", montevideo.com.uy, lanzamiento de la LUB)
    rec = ranking(w_lig, d_lig, ~in_top)[:, :2]
    semb = np.concatenate([lig, rec], axis=1)      # (S, 8) sembrados 1..8

    # ---- playoffs
    siembra = np.full((S, T), 99)
    siembra[np.arange(S)[:, None], semb] = np.arange(8)[None, :]

    def por_siembra(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mejor sembrado, peor sembrado) — el mejor tiene la localía de la serie."""
        x_mejor = siembra[np.arange(S), x] <= siembra[np.arange(S), y]
        return np.where(x_mejor, x, y), np.where(x_mejor, y, x)

    def serie(alto: np.ndarray, bajo: np.ndarray, mejor_de: int, fechas: list[str],
              neutral: bool = False) -> np.ndarray:
        necesita = mejor_de // 2 + 1
        va, vb = np.zeros(S, int), np.zeros(S, int)
        patron = [1, 1, 0, 0, 1] if mejor_de == 5 else [1, 1, 0, 0, 1, 0, 1]
        for g in range(mejor_de):
            activo = (va < necesita) & (vb < necesita)
            local_alto = patron[g] == 1
            h = alto if local_alto else bajo
            a = bajo if local_alto else alto
            nombre = fechas[0] if g < 2 else fechas[-1]
            c = jugar(h, a, Slot(nombre, "playoff"), activo=activo, neutral=neutral)
            gana_local = c < N_BANDAS
            gana_alto = gana_local if local_alto else ~gana_local
            va += (activo & gana_alto)
            vb += (activo & ~gana_alto)
        return np.where(va >= necesita, alto, bajo)

    wins[:], dif[:] = 0, 0
    qf = ["Cuartos de Final Playoffs (1 y 2)", "Cuartos de Final Playoffs (3, 4 y 5)"]
    g1 = serie(semb[:, 0], semb[:, 7], 5, qf)
    g4 = serie(semb[:, 3], semb[:, 4], 5, qf)
    g2 = serie(semb[:, 1], semb[:, 6], 5, qf)
    g3 = serie(semb[:, 2], semb[:, 5], 5, qf)
    sf = ["Semifinales (1 y 2)", "Semifinales (3, 4 y 5)"]
    # semis: A(4-5) vs D(1-8) y B(3-6) vs C(2-7); localía para el mejor sembrado que
    # quedó vivo (antes se le daba al ganador de la llave 1-8 aunque fuera el 8)
    f1 = serie(*por_siembra(g1, g4), 5, sf)
    f2 = serie(*por_siembra(g2, g3), 5, sf)
    # final al mejor de 7 en cancha NEUTRAL (Antel Arena)
    campeon = serie(*por_siembra(f1, f2), 7, ["Finales", "Finales"], neutral=True)

    return ordenar_por_fecha(Sorteo(
        slots=slots, fechas=[],
        fecha_de_slot=np.zeros(len(slots), int),
        jugado=np.stack(jug_cols, 1), clase=np.stack(clase_cols, 1),
        probs=np.stack(probs_cols, 1), pref=np.array([s.preferencial for s in slots]),
        campeon=campeon, equipos=equipos,
    ))


def ordenar_por_fecha(so: Sorteo) -> Sorteo:
    """Reordena los slots para que cada fecha quede CONTIGUA (orden de primera aparición).

    La acumulación por fecha de simular_rivales asume contigüidad; en playoffs las
    series se simulan una tras otra (cuartos 1-2 y 3-5 intercalados) y en temporadas
    reales hay partidos postergados. Sin esto el máximo de fecha se pisa y da basura."""
    fechas = list(dict.fromkeys(s.fecha for s in so.slots))
    fidx = {f: i for i, f in enumerate(fechas)}
    fs = np.array([fidx[s.fecha] for s in so.slots])
    o = np.argsort(fs, kind="stable")
    so.slots = [so.slots[i] for i in o]
    so.fechas = fechas
    so.fecha_de_slot = fs[o]
    so.jugado, so.clase, so.probs, so.pref = so.jugado[:, o], so.clase[:, o], so.probs[:, o], so.pref[o]
    return so


def _round_robin(n: int) -> list[list[tuple[int, int]]]:
    """Doble round-robin (método del círculo), vuelta con localía invertida."""
    idx = list(range(n))
    ida = []
    for _ in range(n - 1):
        ida.append([(idx[i], idx[n - 1 - i]) for i in range(n // 2)])
        idx = [idx[0]] + [idx[-1]] + idx[1:-1]
    return ida + [[(b, a) for a, b in r] for r in ida]


# ----------------------------------------------------------------------------------

def puntos_slot(picks: np.ndarray, clase: np.ndarray, jugado: np.ndarray, pref: bool,
                rng: np.random.Generator | None = None, rho: float = RHO_EXACTO) -> np.ndarray:
    """Puntos de picks (S, n) contra el resultado (S,) de un slot.

    Con rng, cada acierto de clase es además exacto con prob. ρ (+10)."""
    pts = KERNEL[picks, clase[:, None]].astype(np.int16)
    if rng is not None and rho > 0:
        acierto = picks == clase[:, None]
        pts += (acierto & (rng.random(picks.shape, dtype=np.float32) < rho)).astype(np.int16) * PTS_EXACTO
    pts *= jugado[:, None]
    return pts * 2 if pref else pts


@dataclass
class Perfiles:
    """Perfiles de actividad de rivales, bootstrapeados de una temporada real.

    part (P, 3): fracción de partidos con pick en regular/liguilla/playoff.
    esp (P,) bool: cargó especiales. Un rival simulado toma un perfil al azar y carga
    cada partido con prob. part[fase] — así la competencia efectiva son los
    constantes, no los 320 nominales."""
    part: np.ndarray
    esp: np.ndarray
    gamma: np.ndarray | None = None     # (P,) estilo de cada rival: γ propio (chalk vs disperso)

    @classmethod
    def plenos(cls) -> "Perfiles":
        return cls(part=np.ones((1, 3)), esp=np.ones(1, bool))


FASE_IDX = {"regular": 0, "liguilla": 1, "playoff": 2}


def simular_rivales(so: Sorteo, cfg: Config, q_de: Callable[..., np.ndarray],
                    campeon_shares: np.ndarray,
                    goleador_real: np.ndarray | None = None, goleador_shares: np.ndarray | None = None,
                    perfiles: Perfiles | None = None,
                    n_rivales: int | None = None, seed: int | None = None,
                    kappa: float = 0.0, kappa_spread: float = 0.0) -> np.ndarray:
    """Puntajes de R rivales por fecha y totales → max y cantidad en el max (in place).

    q_de(probs (S,10)) → (S,10): distribución de picks del pool dado lo que "ve" el modelo.
    campeon_shares (T,): prob. de que un rival (que carga especiales) elija cada equipo.
    goleador_real (S,) índice del goleador por sorteo; goleador_shares (G,) picks del pool.
    kappa: HABILIDAD del pool. Los rivales constantes de la 25/26 sacaron 466 pts de
    partidos contra 435 de rivales simulados con el mismo estilo: saben algo que el
    modelo no (lesiones, extranjeros, cuotas). Se modela como una señal sobre el
    resultado: Q(c) × e^κ_i en la clase que efectivamente sale, con κ_i ~ U(κ ± spread)
    por rival (la habilidad varía: sd real de los plenos 40 vs 34 con κ común).
    κ y spread se calibran con el PIT.
    Devuelve los totales (S, R) — útil para diagnósticos.
    """
    rng = np.random.default_rng(seed or cfg.seed + 1)
    R = n_rivales or cfg.n_rivales
    S, J = so.clase.shape
    F = len(so.fechas)
    perf = perfiles or Perfiles.plenos()
    # composición del pool: un perfil por (sorteo, rival). Antes se sorteaba UNA vez por
    # corrida → todos los sorteos compartían la misma composición y el SE subestimaba
    # (OOS vs in-sample difería 5 SE). Los γ se agrupan en 6 bins sobre los perfiles.
    k = rng.integers(0, len(perf.part), (S, R))
    part = perf.part.astype(np.float32)[k]  # (S, R, 3)
    carga_esp = perf.esp[k]                 # (S, R)
    if perf.gamma is not None:
        gbin = _bins(perf.gamma, 6)
        grupos_g = [float(np.median(perf.gamma[gbin == b])) for b in range(gbin.max() + 1)]
        grp = gbin[k].astype(np.int8)       # (S, R)
    else:
        grupos_g, grp = [None], np.zeros((S, R), np.int8)
    k_r = (kappa + (rng.random((S, R)) * 2 - 1) * kappa_spread).astype(np.float32)
    total = np.zeros((S, R), np.int16)
    fmax = np.zeros((S, F), np.int16)
    fcnt = np.zeros((S, F), np.int16)
    fecha_pts = np.zeros((S, R), np.int16)
    fecha_actual = so.fecha_de_slot[0] if J else 0
    for j in range(J):
        f = so.fecha_de_slot[j]
        if f != fecha_actual:
            fmax[:, fecha_actual], fcnt[:, fecha_actual] = _max_cnt(fecha_pts)
            fecha_pts[:] = 0
            fecha_actual = f
        qg = np.stack([q_de(so.probs[:, j]) if g is None else q_de(so.probs[:, j], gamma=g)
                       for g in grupos_g]).astype(np.float32)          # (G, S, 10)
        q = qg[grp, np.arange(S)[:, None]]                              # (S, R, 10)
        if kappa or kappa_spread:
            hit = np.zeros((S, 1, N_CLASES), np.float32)
            hit[np.arange(S), 0, so.clase[:, j]] = 1.0
            q = q * np.exp(k_r[:, :, None] * hit)
            q /= q.sum(-1, keepdims=True)
        cdf = np.cumsum(q, -1)
        u = rng.random((S, R, 1), dtype=np.float32)
        picks = (u > cdf[:, :, :-1]).sum(-1).astype(np.int8)
        p = puntos_slot(picks, so.clase[:, j], so.jugado[:, j], bool(so.pref[j]), rng)
        carga = rng.random((S, R), dtype=np.float32) < part[:, :, FASE_IDX[so.slots[j].fase]]
        p *= carga
        fecha_pts += p
        total += p
    if J:
        fmax[:, fecha_actual], fcnt[:, fecha_actual] = _max_cnt(fecha_pts)
    # especiales (solo los que los cargan)
    camp_pick = rng.choice(len(campeon_shares), size=(S, R), p=campeon_shares)
    total += ((camp_pick == so.campeon[:, None]) & carga_esp).astype(np.int16) * PTS_ESPECIAL
    if goleador_real is not None and goleador_shares is not None:
        gol_pick = rng.choice(len(goleador_shares), size=(S, R), p=goleador_shares)
        total += ((gol_pick == goleador_real[:, None]) & carga_esp).astype(np.int16) * PTS_ESPECIAL
    so.riv_fecha_max, so.riv_fecha_cnt = fmax, fcnt
    so.riv_total_max, so.riv_total_cnt = _max_cnt(total)
    return total


def _bins(x: np.ndarray, n: int) -> np.ndarray:
    cortes = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)))
    return np.clip(np.searchsorted(cortes, x, side="right") - 1, 0, max(len(cortes) - 2, 0))


def _max_cnt(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = x.max(1)
    return m.astype(np.int16), (x == m[:, None]).sum(1).astype(np.int16)


def liquidar(nuestros_fecha: np.ndarray, nuestros_total: np.ndarray, so: Sorteo,
             premio_penca: float = PREMIO_PENCA, premio_fecha: float = PREMIO_FECHA) -> dict:
    """E[premio] del portfolio.

    nuestros_fecha (S, F, K): puntos de nuestras K participaciones por fecha.
    nuestros_total (S, K): totales con especiales.
    """
    def premio(ours: np.ndarray, rmax: np.ndarray, rcnt: np.ndarray, monto: float) -> np.ndarray:
        best = ours.max(-1)
        top = np.maximum(best, rmax)
        n_ours = (ours == top[..., None]).sum(-1)
        n_riv = np.where(rmax == top, rcnt, 0)
        return monto * n_ours / np.maximum(n_ours + n_riv, 1)

    pf = premio(nuestros_fecha, so.riv_fecha_max, so.riv_fecha_cnt, premio_fecha)   # (S, F)
    pp = premio(nuestros_total, so.riv_total_max, so.riv_total_cnt, premio_penca)   # (S,)
    tot = pf.sum(1) + pp
    return {"e_premio": float(tot.mean()), "se": float(tot.std() / math.sqrt(len(tot))),
            "e_penca": float(pp.mean()), "e_fechas": float(pf.sum(1).mean()),
            "p_penca": float((pp > 0).mean()), "e_por_fecha": pf.mean(0), "_tot": tot}

"""Modelo de margen de la LUB: ratings por equipo → distribución de las 10 clases.

    margen (local − visitante, con prórroga) = HCA + r_local − r_visitante + ε

Ratings por mínimos cuadrados ponderados (ridge hacia 0, decaimiento exponencial por
antigüedad). Entre temporadas los planteles cambian mucho (extranjeros), así que el
rating que cruza de temporada se achica por `carry`. Equipo sin historia (ascendido)
arranca en `r_nuevo`.

ε se modela con los RESIDUOS EMPÍRICOS (no una normal): el margen en básquet tiene
colas más pesadas que la normal y las bandas extremas (+20) pagan lo mismo que las
centrales. Un residuo r con media μ produce margen μ + r; lo que cae en |m| < 0,5 es
el empate de los 40' que resuelve la prórroga — se reparte en la banda 1-5 de cada
lado según quién es favorito (las prórrogas se ganan por poco).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.lub.data import Partido
from src.lub.scoring import BANDAS, N_BANDAS, N_CLASES

MARGEN_MAX = 80


@dataclass
class Params:
    half_life_dias: float = 120.0
    ridge: float = 3.0
    carry: float = 0.6          # fracción del rating que sobrevive al cambio de temporada
    r_nuevo: float = -4.0       # equipo sin historia (ascendido)
    min_resid: int = 60         # residuos mínimos para usar la empírica; si no, normal
    escala_mu: float = 1.0      # >1 corrige el encogimiento del ridge/decay (el modelo sub-confía)
    escala_sigma: float = 1.0   # σ de predicción / σ de residuos in-sample


@dataclass
class Ratings:
    hca: float
    r: dict[str, float]
    sigma: float
    resid: np.ndarray = field(repr=False)
    temporada: str = ""
    params: Params = field(default_factory=Params)

    def mu(self, local: str, visitante: str) -> float:
        p = self.params
        return p.escala_mu * (self.hca + self.r.get(local, p.r_nuevo) - self.r.get(visitante, p.r_nuevo))

    @property
    def sigma_pred(self) -> float:
        return self.sigma * self.params.escala_sigma


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s)


def fit(partidos: list[Partido], hasta: datetime, temporada_objetivo: str,
        params: Params | None = None) -> Ratings:
    """Ratings con los partidos terminados ANTES de `hasta` (sin mirar el futuro)."""
    p = params or Params()
    jugados = [x for x in partidos if x.pts_local is not None and _t(x.inicio_utc) < hasta]
    if not jugados:
        return Ratings(hca=3.0, r={}, sigma=15.0, resid=np.array([]), temporada=temporada_objetivo, params=p)
    equipos = sorted({x.local for x in jugados} | {x.visitante for x in jugados})
    idx = {e: i for i, e in enumerate(equipos)}
    n, k = len(jugados), len(equipos)
    X = np.zeros((n, k + 1))
    y = np.zeros(n)
    w = np.zeros(n)
    for i, x in enumerate(jugados):
        X[i, 0] = 1.0
        X[i, 1 + idx[x.local]] += 1.0
        X[i, 1 + idx[x.visitante]] -= 1.0
        y[i] = x.pts_local - x.pts_visitante
        edad = (hasta - _t(x.inicio_utc)).total_seconds() / 86400
        w[i] = 0.5 ** (edad / p.half_life_dias)
        if x.temporada != temporada_objetivo:
            w[i] *= p.carry       # la temporada anterior pesa menos además de por edad
    R = np.eye(k + 1) * p.ridge
    R[0, 0] = 0.0
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X + R, X.T @ W @ y)
    resid = y - X @ beta
    sigma = float(np.sqrt(np.sum(w * resid ** 2) / np.sum(w)))
    r = {e: float(beta[1 + i]) for e, i in idx.items()}
    # equipos que NO jugaron esta temporada todavía: el rating viejo se achica
    en_temporada = {x.local for x in jugados if x.temporada == temporada_objetivo} | \
                   {x.visitante for x in jugados if x.temporada == temporada_objetivo}
    for e in r:
        if e not in en_temporada:
            r[e] *= p.carry
    return Ratings(hca=float(beta[0]), r=r, sigma=sigma, resid=resid, temporada=temporada_objetivo, params=p)


def margin_pmf(mu: float, sigma: float, resid: np.ndarray | None = None,
               min_resid: int = 60) -> np.ndarray:
    """P(margen = m) para m en [−MARGEN_MAX, MARGEN_MAX], sin masa en 0 (prórroga)."""
    ms = np.arange(-MARGEN_MAX, MARGEN_MAX + 1)
    if resid is not None and len(resid) >= min_resid:
        # empírica suavizada: mezcla de normales angostas centradas en μ + residuo,
        # re-escalada para que su sd coincida con σ (los residuos in-sample subestiman)
        rs = resid - resid.mean()
        s_in = rs.std() or 1.0
        centros = mu + rs * (sigma / s_in)
        h = 1.06 * sigma * len(rs) ** -0.2
        cdf = lambda x: 0.5 * (1 + _erf((x[:, None] - centros[None, :]) / (h * math.sqrt(2)))).mean(1)
    else:
        cdf = lambda x: 0.5 * (1 + _erf((x - mu) / (sigma * math.sqrt(2))))
    edges = np.arange(-MARGEN_MAX - 0.5, MARGEN_MAX + 1.5)
    c = cdf(edges)
    c[0], c[-1] = 0.0, 1.0
    pmf = np.diff(c)
    empate = pmf[MARGEN_MAX]
    pmf[MARGEN_MAX] = 0.0
    # la prórroga (5') se decide por poco y la gana el favorito un poco más seguido
    p_local_ot = 0.5 + 0.5 * math.tanh(mu / 20.0)
    ot = np.array([0.30, 0.25, 0.20, 0.15, 0.10])       # márgenes 1..5 en prórroga
    pmf[MARGEN_MAX + 1:MARGEN_MAX + 6] += empate * p_local_ot * ot
    pmf[MARGEN_MAX - 5:MARGEN_MAX][::-1] += empate * (1 - p_local_ot) * ot
    return pmf / pmf.sum()


def _erf(x):
    from scipy.special import erf
    return erf(x)


def class_probs(pmf: np.ndarray) -> np.ndarray:
    """PMF de margen → probabilidades de las 10 clases."""
    out = np.zeros(N_CLASES)
    for b, (lo, hi) in enumerate(BANDAS):
        hi = min(hi, MARGEN_MAX)
        out[b] = pmf[MARGEN_MAX + lo:MARGEN_MAX + hi + 1].sum()
        out[N_BANDAS + b] = pmf[MARGEN_MAX - hi:MARGEN_MAX - lo + 1].sum()
    return out / out.sum()


def probs_partido(rt: Ratings, local: str, visitante: str, empirica: bool = True) -> np.ndarray:
    mu = rt.mu(local, visitante)
    return class_probs(margin_pmf(mu, rt.sigma_pred, rt.resid if empirica else None, rt.params.min_resid))


# -------------------- validación walk-forward --------------------

def walk_forward(partidos: list[Partido], temporadas: tuple[str, ...] = ("LUB 25/26",),
                 params: Params | None = None, empirica: bool = True,
                 refit_cada: int = 6) -> dict:
    """Predice cada partido de `temporadas` con ratings ajustados solo con el pasado.

    Devuelve log-loss por clase (10), log-loss ganador, Brier del ganador y la
    calibración por clase (predicho vs observado)."""
    obj = sorted([x for x in partidos if x.temporada in temporadas and x.pts_local is not None],
                 key=lambda x: x.inicio_utc)
    ll_c, ll_w, brier, pred, obs = [], [], [], [], []
    rt, desde = None, -1
    for i, x in enumerate(obj):
        if rt is None or i - desde >= refit_cada:
            rt, desde = fit(partidos, _t(x.inicio_utc), x.temporada, params), i
        pc = probs_partido(rt, x.local, x.visitante, empirica)
        from src.lub.scoring import clase
        c = clase(x.pts_local, x.pts_visitante)
        ph = pc[:N_BANDAS].sum()
        ll_c.append(-math.log(max(pc[c], 1e-9)))
        won = x.pts_local > x.pts_visitante
        ll_w.append(-math.log(max(ph if won else 1 - ph, 1e-9)))
        brier.append((ph - won) ** 2)
        pred.append(pc)
        onehot = np.zeros(N_CLASES); onehot[c] = 1; obs.append(onehot)
    pred, obs = np.array(pred), np.array(obs)
    return {"n": len(obj), "ll_clase": float(np.mean(ll_c)), "ll_ganador": float(np.mean(ll_w)),
            "brier": float(np.mean(brier)), "pred_medio": pred.mean(0), "obs_medio": obs.mean(0),
            "ll_uniforme_clase": math.log(N_CLASES)}


# Elegidos walk-forward sobre la LUB 25/26 (203 partidos, 2026-09-29): la superficie es
# plana (Δ log-loss de clase ≤ 0,006 en todo el rango razonable), así que se toman
# valores centrales; escala_mu 1,2 corrige la sub-confianza (favorito 72% → ganó 76%).
PARAMS_PROD = Params(half_life_dias=240, ridge=3.0, carry=0.6, r_nuevo=-6.0, escala_mu=1.2)

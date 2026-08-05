"""Modelo EMPÍRICO por rival: picks conocidos + estilo + standings reales.

Cierra el hueco principal del simulador (analizado 2026-08-04): el snapshot del pool
trae los picks y puntos REALES de cada participación, pero `SeasonSimulator` los
usaba solo agregados en una Q por partido y re-sorteaba rivales i.i.d. Eso pierde:

  1. **El estado real de la carrera.** A mitad de temporada el premio grande depende
     de la tabla real; re-sortear el pasado hace creer que la general está más (o
     menos) abierta de lo que está y asigna mal las fechas de la cola.
  2. **La correlación por rival.** El que es chalk, es chalk en todos los partidos:
     muestrear i.i.d. de la Q agregada adelgaza las colas del máximo del pool.
  3. **El ausentismo.** No todos cargan todos los partidos; para el premio por fecha
     los que no cargan no compiten.

El modelo por rival r:

    pick futuro  ~  Categórica ∝ Q_m^γ_r        (γ_r = estilo: >1 más chalk que el
                                                 pool agregado, <1 más disperso)
    carga futura ~  Bernoulli(p_show_r)
    pasado       =  su pick observado (puntos exactos vía delta-grid) o 0 si no cargó

γ_r se ajusta por MAP sobre los picks observados con shrinkage a 1 (con 8-16 picks
por rival la MLE cruda es ruidosa). El residuo `puntos_ranking − puntos_implicados`
se suma al total del rival: garantiza que los standings del simulador coincidan con
la tabla real aunque el kernel o el snapshot tengan alguna diferencia menor.

`SeasonSimulator` recibe este objeto por duck-typing (usa .n_rivales, .residuo,
.campeon_idx y .sample_picks_match()) — economics.py no importa este módulo.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from src.clausura.economics import (
    MAX_GOALS,
    N_SCORES,
    points_matrix,
    score_index,
)

log = logging.getLogger(__name__)

# Shrinkage del estilo hacia γ=1 (el pool agregado). Penaliza log-lik con τ·(ln γ)²:
# con ~10 picks observados hace falta evidencia consistente para alejarse de 1.
GAMMA_TAU = 3.0
GAMMA_GRID = np.geomspace(0.35, 4.0, 25)

# Prior Beta(a,b) de p_show: el pencista típico carga casi siempre.
SHOW_PRIOR_A, SHOW_PRIOR_B = 9.0, 1.0

_PM_NORMAL = points_matrix(False)
_PM_PREF = points_matrix(True)


@dataclass
class RivalModel:
    """Estado empírico del pool rival, listo para enchufar al SeasonSimulator.

    known_picks[r, m] = índice de score observado, -1 = no observado.
    played_mask[m]    = True si el partido ya se jugó (no observado ⇒ no cargó ⇒ 0 pts).
    residuo[r]        = puntos_ranking − puntos_implicados por los picks del pasado.
    campeon_idx[r]    = índice de equipo del especial Campeón, -1 = no observado.
    """
    known_picks: np.ndarray
    played_mask: np.ndarray
    gamma: np.ndarray
    p_show: np.ndarray
    residuo: np.ndarray
    numeros: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    campeon_idx: np.ndarray | None = None

    @property
    def n_rivales(self) -> int:
        return self.known_picks.shape[0]

    def sample_picks_match(self, m: int, pool_q: np.ndarray, rng: np.random.Generator,
                           n_sims: int) -> tuple[np.ndarray, np.ndarray]:
        """(picks, show) del partido m, ambos (n_rivales, n_sims).

        Observado → su pick tal cual (constante entre sims). Pasado no observado →
        no cargó (show=False). Futuro no observado → Bernoulli(p_show) + categórica
        ∝ Q^γ, re-sorteados POR SIMULACIÓN: E[premio] integra la incertidumbre de
        los picks del pool en vez de condicionar a una única realización (que el
        ascenso por coordenadas podía explotar como huecos muestrales inexistentes).
        """
        R = self.n_rivales
        picks = np.zeros((R, n_sims), dtype=np.int64)
        show = np.zeros((R, n_sims), dtype=bool)
        known = self.known_picks[:, m]
        has = known >= 0
        picks[has] = known[has, None]
        show[has] = True
        if self.played_mask[m] or bool(has.all()):
            return picks, show
        libre = ~has
        picks[libre] = _tilted_sample(pool_q, self.gamma[libre], rng, size=n_sims)
        show[libre] = rng.random((int(libre.sum()), n_sims)) < self.p_show[libre, None]
        return picks, show

    def resumen(self) -> str:
        q25, q50, q75 = np.percentile(self.gamma, [25, 50, 75])
        obs = float((self.known_picks >= 0).sum(axis=1).mean())
        return (f"{self.n_rivales} rivales · γ mediana {q50:.2f} [{q25:.2f}–{q75:.2f}] · "
                f"p_show media {self.p_show.mean():.2f} · {obs:.1f} picks observados c/u · "
                f"residuo medio {self.residuo.mean():+.2f} pts "
                f"(máx |{np.abs(self.residuo).max()}|)")


def _tilted_sample(q: np.ndarray, gamma: np.ndarray, rng: np.random.Generator,
                   size: int | None = None) -> np.ndarray:
    """Picks de la categórica ∝ q^γ_r, vectorizado por inverse-CDF.

    Sin `size`: un pick por rival, shape (R,). Con `size`: sorteos independientes
    por simulación, shape (R, size).
    """
    logq = np.log(q + 1e-12)
    w = np.exp(gamma[:, None] * logq[None, :])
    w /= w.sum(axis=1, keepdims=True)
    cdf = np.cumsum(w, axis=1)
    if size is None:
        u = rng.random(len(gamma))
        return (u[:, None] > cdf).sum(axis=1).clip(0, N_SCORES - 1)
    u = rng.random((len(gamma), size))
    return (u[:, :, None] > cdf[:, None, :]).sum(axis=2).clip(0, N_SCORES - 1)


def fit_gamma(obs_idx: np.ndarray, logqs: np.ndarray) -> float:
    """MAP de γ para un rival: obs_idx (n_obs,) picks, logqs (n_obs, N_SCORES).

    score(γ) = Σ_m [γ·log Q_m(x_m) − logsumexp(γ·log Q_m)] − τ·(ln γ)²
    """
    if len(obs_idx) == 0:
        return 1.0
    lq_x = logqs[np.arange(len(obs_idx)), obs_idx]          # (n_obs,)
    best_g, best_s = 1.0, -np.inf
    for g in GAMMA_GRID:
        z = g * logqs
        z_max = z.max(axis=1, keepdims=True)
        lse = (z_max[:, 0] + np.log(np.exp(z - z_max).sum(axis=1)))
        s = float(g * lq_x.sum() - lse.sum()) - GAMMA_TAU * np.log(g) ** 2
        if s > best_s:
            best_g, best_s = float(g), s
    return best_g


def build_rival_model_from_arrays(
    known_picks: np.ndarray,
    played_mask: np.ndarray,
    pool_qs: list[np.ndarray],
    preferencial: list[bool],
    actual_idx: np.ndarray,
    puntos_ranking: np.ndarray,
    numeros: np.ndarray | None = None,
    campeon_idx: np.ndarray | None = None,
) -> RivalModel:
    """Ajusta γ, p_show y residuo desde las matrices ya mapeadas a índices de partido.

    `actual_idx[m]` = índice de score del resultado real (solo se usa donde
    played_mask[m]; el resto puede ser -1).
    """
    R, n_matches = known_picks.shape
    if len(pool_qs) != n_matches or len(preferencial) != n_matches:
        raise ValueError("pool_qs/preferencial no matchean known_picks")

    logqs = np.stack([np.log(q + 1e-12) for q in pool_qs])   # (n_matches, N_SCORES)

    gamma = np.ones(R)
    for r in range(R):
        has = known_picks[r] >= 0
        gamma[r] = fit_gamma(known_picks[r, has], logqs[has])

    # p_show con prior Beta: solo los partidos jugados informan (los futuros aún
    # pueden cargarse hasta el cierre).
    jugados = int(played_mask.sum())
    cargados = (known_picks[:, played_mask] >= 0).sum(axis=1) if jugados else np.zeros(R)
    p_show = (cargados + SHOW_PRIOR_A) / (jugados + SHOW_PRIOR_A + SHOW_PRIOR_B)

    # puntos implicados por los picks del pasado → residuo vs el ranking real
    implied = np.zeros(R, dtype=np.int64)
    for m in np.flatnonzero(played_mask):
        pm = _PM_PREF if preferencial[m] else _PM_NORMAL
        has = known_picks[:, m] >= 0
        implied[has] += pm[known_picks[has, m], actual_idx[m]]
    residuo = puntos_ranking.astype(np.int64) - implied
    if jugados and np.abs(residuo).max() > 0:
        peor = int(np.abs(residuo).argmax())
        log.warning("residuo puntos ranking≠implicados: medio %+.2f, peor %+d "
                    "(participación %s) — se corrige sumándolo al total",
                    float(residuo.mean()), int(residuo[peor]),
                    numeros[peor] if numeros is not None else peor)

    return RivalModel(
        known_picks=known_picks.copy(),
        played_mask=played_mask.copy(),
        gamma=gamma,
        p_show=p_show,
        residuo=residuo,
        numeros=numeros if numeros is not None else np.zeros(R, dtype=np.int64),
        campeon_idx=campeon_idx,
    )


def mis_numeros_env() -> set[int]:
    """Números de participación propios (CLAUSURA_MIS_PARTICIPACIONES del env)."""
    raw = os.environ.get("CLAUSURA_MIS_PARTICIPACIONES", "")
    return {int(x) for x in raw.split(",") if x.strip().isdigit()}


def build_rival_model(
    snapshot: dict,
    eventos: list[dict],
    pool_qs: list[np.ndarray],
    resultados: dict[int, tuple[int, int]],
    mis_numeros: set[int] | None = None,
    equipo_idx: dict[str, int] | None = None,
) -> RivalModel | None:
    """RivalModel desde un snapshot del pool (src.clausura.pool_snapshot).

    Excluye NUESTRAS participaciones (si quedaran adentro, el simulador las contaría
    como rivales y nos dividiría el premio contra nosotros mismos). Devuelve None si
    tras filtrar no queda ningún rival.
    """
    if mis_numeros is None:
        mis_numeros = mis_numeros_env()
    if not mis_numeros:
        log.warning("CLAUSURA_MIS_PARTICIPACIONES vacío — no puedo excluir mis "
                    "participaciones del modelo de rivales")

    idx_by_evento = {ev["evento_id"]: i for i, ev in enumerate(eventos)}
    n_matches = len(eventos)
    filas = [r for r in snapshot.get("participaciones", [])
             if int(r.get("numero", -1)) not in mis_numeros]
    if not filas:
        return None

    R = len(filas)
    known = np.full((R, n_matches), -1, dtype=np.int64)
    numeros = np.zeros(R, dtype=np.int64)
    puntos = np.zeros(R, dtype=np.int64)
    campeon = np.full(R, -1, dtype=np.int64) if equipo_idx is not None else None
    for r, fila in enumerate(filas):
        numeros[r] = int(fila.get("numero", 0))
        puntos[r] = int(fila.get("puntos", 0))
        for eid_str, (gl, gv) in fila.get("picks", {}).items():
            m = idx_by_evento.get(int(eid_str))
            if m is not None:
                known[r, m] = score_index(min(int(gl), MAX_GOALS), min(int(gv), MAX_GOALS))
        if campeon is not None and fila.get("campeon") in equipo_idx:
            campeon[r] = equipo_idx[fila["campeon"]]

    played = np.zeros(n_matches, dtype=bool)
    actual = np.full(n_matches, -1, dtype=np.int64)
    for i, ev in enumerate(eventos):
        res = resultados.get(ev["evento_id"])
        if res is not None:
            played[i] = True
            actual[i] = score_index(min(res[0], MAX_GOALS), min(res[1], MAX_GOALS))

    model = build_rival_model_from_arrays(
        known, played, pool_qs, [ev["preferencial"] for ev in eventos],
        actual, puntos, numeros, campeon,
    )
    log.info("modelo de rivales: %s (excluidas %d participaciones propias)",
             model.resumen(), len(snapshot.get("participaciones", [])) - R)
    return model

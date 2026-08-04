"""Fuerzas de ataque/defensa por equipo para la Primera uruguaya (Capa 2 local).

ClubElo y EloRatings no cubren la Primera uruguaya, así que la Capa 2 del Mundial no
transfiere. Acá ajustamos un Poisson ataque-defensa clásico (Maher 1982) sobre las
temporadas históricas que devuelve el penca-api:

    log λ_local   = μ + ventaja_local + ataque[local]  − defensa[visitante]
    log λ_visita  = μ                 + ataque[visita] − defensa[local]

con la normalización Σ ataque = Σ defensa = 0 para que el modelo sea identificable.

El histórico dice que la covarianza entre goles es ~0 (corr +0.002 en 598 partidos),
así que no hace falta el término bivariado: λ12 = 0 y Poisson independiente alcanza.

Uso principal: darle al backtest grillas que varíen por partido (sin eso todos los
partidos son idénticos y la optimización no tiene nada que optimizar). En producción
es el prior "stat" que se blendea con el mercado, igual que la Capa 2 del Mundial.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from src.clausura.historical import PartidoHistorico


@dataclass
class TeamRatings:
    equipos: list[str]
    ataque: dict[str, float] = field(default_factory=dict)
    defensa: dict[str, float] = field(default_factory=dict)
    mu: float = 0.0
    ventaja_local: float = 0.0

    def lambdas(self, local: str, visitante: str) -> tuple[float, float]:
        """(λ_local, λ_visita) para un cruce. Equipos desconocidos caen a la media."""
        a_l = self.ataque.get(local, 0.0)
        d_l = self.defensa.get(local, 0.0)
        a_v = self.ataque.get(visitante, 0.0)
        d_v = self.defensa.get(visitante, 0.0)
        lam_l = np.exp(self.mu + self.ventaja_local + a_l - d_v)
        lam_v = np.exp(self.mu + a_v - d_l)
        return float(lam_l), float(lam_v)


def fit_ratings(partidos: list[PartidoHistorico], ridge: float = 0.05) -> TeamRatings:
    """MLE Poisson de ataque/defensa con regularización ridge.

    El ridge evita que equipos con pocos partidos (ascendidos, por ejemplo) exploten
    a valores extremos; con λ≈0.05 el encogimiento es suave y no aplana las diferencias
    reales entre Peñarol/Nacional y el resto.
    """
    equipos = sorted({p.local for p in partidos} | {p.visitante for p in partidos})
    idx = {e: i for i, e in enumerate(equipos)}
    n = len(equipos)

    li = np.array([idx[p.local] for p in partidos])
    vi = np.array([idx[p.visitante] for p in partidos])
    gl = np.array([p.goles_local for p in partidos], dtype=float)
    gv = np.array([p.goles_visitante for p in partidos], dtype=float)

    def unpack(theta):
        att = np.concatenate([theta[: n - 1], [-theta[: n - 1].sum()]])
        deff = np.concatenate([theta[n - 1: 2 * n - 2], [-theta[n - 1: 2 * n - 2].sum()]])
        return att, deff, theta[-2], theta[-1]

    def neg_loglik(theta):
        att, deff, mu, home = unpack(theta)
        log_lam_l = mu + home + att[li] - deff[vi]
        log_lam_v = mu + att[vi] - deff[li]
        lam_l, lam_v = np.exp(log_lam_l), np.exp(log_lam_v)
        ll = (gl * log_lam_l - lam_l).sum() + (gv * log_lam_v - lam_v).sum()
        penal = ridge * (np.sum(att ** 2) + np.sum(deff ** 2))
        return -ll / len(partidos) + penal

    theta0 = np.zeros(2 * n - 2 + 2)
    theta0[-2] = np.log(max(gl.mean(), 0.1))   # μ
    theta0[-1] = 0.1                           # ventaja de localía
    res = minimize(neg_loglik, theta0, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-10})

    att, deff, mu, home = unpack(res.x)
    return TeamRatings(
        equipos=equipos,
        ataque={e: float(att[i]) for e, i in idx.items()},
        defensa={e: float(deff[i]) for e, i in idx.items()},
        mu=float(mu),
        ventaja_local=float(home),
    )


def ranking(ratings: TeamRatings) -> list[tuple[str, float, float, float]]:
    """[(equipo, ataque, defensa, neto)] ordenado por fuerza neta descendente."""
    filas = [
        (e, ratings.ataque[e], ratings.defensa[e], ratings.ataque[e] + ratings.defensa[e])
        for e in ratings.equipos
    ]
    return sorted(filas, key=lambda t: -t[3])

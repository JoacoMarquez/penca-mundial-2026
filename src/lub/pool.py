"""Modelo del pool de la penca LUB, ajustado con los picks reales de la penca 37 (25/26).

Cada rival, en cada partido, elige una clase con probabilidad

    Q(c) ∝ p_c^γ · w_banda(c)

donde p es la distribución de clases del MODELO (lo que "ve" cualquiera que mire las
cuotas) y w_banda captura la preferencia del pool por ciertas bandas de margen
independientemente de la probabilidad (ej. todos cargan "gana por 6-10"). γ > 1 ⇒ el
pool concentra más que el modelo (chalk). Se ajusta por máxima verosimilitud.

Además del reparto de picks, el pool ABANDONA: muchos dejan de cargar a mitad de
temporada y ese partido les da 0. `participacion` es la fracción de rivales con pick
por slot, en función del avance de la temporada.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.lub.scoring import N_BANDAS, N_CLASES, clase


@dataclass
class PoolModel:
    gamma: float = 1.0
    w_banda: np.ndarray = field(default_factory=lambda: np.ones(N_BANDAS))
    # fracción de rivales que cargan pick, por tercio de temporada (regular, liguilla, playoff)
    participacion: dict[str, float] = field(default_factory=lambda: {"regular": 1.0, "liguilla": 1.0, "playoff": 1.0})
    campeon: dict[str, float] = field(default_factory=dict)     # equipo → share de picks
    goleador_hit: float = 0.2                                    # P(que el goleador del rival acierte)

    def q(self, p: np.ndarray, gamma: float | None = None) -> np.ndarray:
        """p (..., 10) → Q (..., 10). `gamma` pisa el γ del pool (estilo de un rival)."""
        w = np.concatenate([self.w_banda, self.w_banda])
        x = np.power(np.clip(p, 1e-9, 1), self.gamma if gamma is None else gamma) * w
        return x / x.sum(-1, keepdims=True)

    def to_json(self) -> dict:
        return {"gamma": self.gamma, "w_banda": list(map(float, self.w_banda)),
                "participacion": self.participacion, "campeon": self.campeon,
                "goleador_hit": self.goleador_hit}

    @classmethod
    def from_json(cls, d: dict) -> "PoolModel":
        return cls(gamma=d["gamma"], w_banda=np.array(d["w_banda"]), participacion=d["participacion"],
                   campeon=d.get("campeon", {}), goleador_hit=d.get("goleador_hit", 0.2))


def perfiles_de_json(d: dict):
    from src.lub.season import Perfiles
    pf = d["perfiles"]
    return Perfiles(part=np.array(pf["part"]), esp=np.array(pf["esp"], bool), gamma=np.array(pf["gamma"]))


def fit_q(p_modelo: np.ndarray, picks: np.ndarray) -> tuple[float, np.ndarray, float]:
    """MLE de (γ, w_banda) con p_modelo (N, 10) y picks (N,) clases elegidas.

    Devuelve (γ, w_banda normalizado con w_0 = 1, log-lik medio)."""
    lp = np.log(np.clip(p_modelo, 1e-9, 1))

    def nll(theta):
        g, lw = theta[0], np.concatenate([[0.0], theta[1:]])
        lw10 = np.concatenate([lw, lw])
        s = g * lp + lw10
        s = s - s.max(1, keepdims=True)
        logq = s - np.log(np.exp(s).sum(1, keepdims=True))
        return -logq[np.arange(len(picks)), picks].mean()

    r = minimize(nll, np.array([1.0, 0, 0, 0, 0]), method="L-BFGS-B")
    return float(r.x[0]), np.exp(np.concatenate([[0.0], r.x[1:]])), float(-r.fun)


def load_pool(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def picks_por_evento(pool: dict) -> dict[int, list[int]]:
    """evento_id → clases elegidas por los rivales (sin picks de empate/ inválidos)."""
    out: dict[int, list[int]] = {}
    for fila in pool["participaciones"]:
        for p in fila["picks"]:
            if p["local"] is None or p["visitante"] is None or p["local"] == p["visitante"]:
                continue
            out.setdefault(p["evento_id"], []).append(clase(p["local"], p["visitante"]))
    return out

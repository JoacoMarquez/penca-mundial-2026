"""Capa 1 del modelo: conversión de odds a probabilidades de-vigeadas y agregación entre casas.

Las casas de apuestas cobran un margen ("vig" o "overround") incluido en las cuotas: si vos sumás
1/odds para los 3 outcomes de 1X2, el resultado va a dar > 1.0 (típicamente 1.05-1.10). Ese exceso
es el margen. Hay que removerlo para recuperar las probabilidades "verdaderas" que estima la casa.

Métodos:
- proportional: cada prob implícita se divide por el overround. Simple, baseline.
- shin: modelo de Shin (1993) que asume insider trading. Mejor para mercados con favoritos fuertes
  o para correct_score donde proportional sobrestima al favorito.

Después del de-vig, agregamos múltiples casas con pesos configurables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import brentq


# -------------------- de-vig de un único mercado --------------------

def implied_probs(odds: Mapping[str, float]) -> dict[str, float]:
    """Convierte decimal odds a probabilidades implícitas (sin de-vig)."""
    return {k: 1.0 / v for k, v in odds.items()}


def overround(odds: Mapping[str, float]) -> float:
    """Suma de probs implícitas. > 1.0 indica margen de casa; 1.0 es libre de vig."""
    return sum(1.0 / v for v in odds.values())


def devig_proportional(odds: Mapping[str, float]) -> dict[str, float]:
    """De-vig proporcional: divide cada prob implícita por el overround.

    Sobrestima al favorito. Adecuado para mercados balanceados (1X2 sin favorito claro,
    over/under, BTTS). Para correct_score preferí Shin.
    """
    o = overround(odds)
    return {k: (1.0 / v) / o for k, v in odds.items()}


def devig_shin(odds: Mapping[str, float], max_iter: int = 100) -> dict[str, float]:
    """De-vig por método de Shin (1993): asume que el overround compensa por insider traders.

    El parámetro z ∈ [0, 1] representa la fracción de bettors informados.
    Resuelve numéricamente para z tal que las probs ajustadas suman 1.

    Referencia: Shin, H.S. (1993) "Measuring the Incidence of Insider Trading in a Market for
    State-Contingent Claims". Economic Journal.
    """
    pi_impl = np.array([1.0 / v for v in odds.values()])
    keys = list(odds.keys())

    def f(z: float) -> float:
        # Para z dado, prob ajustada = ((z**2 + 4*(1-z)*pi**2/B) ** 0.5 - z) / (2*(1-z))
        # donde B es el overround
        B = pi_impl.sum()
        adj = (np.sqrt(z**2 + 4 * (1 - z) * pi_impl**2 / B) - z) / (2 * (1 - z))
        return adj.sum() - 1.0

    # Si el overround es muy chico (~1.0), z tiende a 0 → casi proportional
    if abs(overround(odds) - 1.0) < 1e-6:
        return devig_proportional(odds)

    try:
        z = brentq(f, 1e-9, 1 - 1e-9, maxiter=max_iter)
    except (ValueError, RuntimeError):
        # fallback si Shin no converge
        return devig_proportional(odds)

    B = pi_impl.sum()
    adj = (np.sqrt(z**2 + 4 * (1 - z) * pi_impl**2 / B) - z) / (2 * (1 - z))
    # normalizar por seguridad (debería sumar 1 ya)
    adj = adj / adj.sum()
    return dict(zip(keys, adj.tolist()))


def devig(odds: Mapping[str, float], method: str = "proportional") -> dict[str, float]:
    """Dispatcher de de-vig por método."""
    if method == "proportional":
        return devig_proportional(odds)
    if method == "shin":
        return devig_shin(odds)
    raise ValueError(f"método de-vig desconocido: {method}")


# -------------------- agregación entre casas --------------------

@dataclass(frozen=True)
class BookQuote:
    """Quote de una casa para un mercado puntual."""
    book: str          # "pinnacle" | "bet365" | "betfair_exchange" | ...
    market: str        # "1x2" | "correct_score" | "over_under_2.5" | ...
    probs: Mapping[str, float]   # ya de-vigeadas


def aggregate(
    quotes: list[BookQuote],
    weights: Mapping[str, float],
    fallback: str = "redistribute_proportional",
) -> dict[str, float]:
    """Promedio ponderado de probs entre casas.

    Si una casa configurada no aparece en quotes, su peso se redistribuye entre las que sí están
    (política `redistribute_proportional`).
    """
    if not quotes:
        raise ValueError("no hay quotes para agregar")

    present_books = {q.book for q in quotes}
    used_weights = {b: w for b, w in weights.items() if b in present_books}
    total_w = sum(used_weights.values())
    if total_w == 0:
        raise ValueError(f"ninguna casa con peso > 0 está presente. quotes={present_books}, weights={set(weights)}")

    if fallback == "redistribute_proportional":
        used_weights = {b: w / total_w for b, w in used_weights.items()}
    else:
        raise ValueError(f"fallback policy desconocida: {fallback}")

    outcomes = quotes[0].probs.keys()
    if not all(set(q.probs.keys()) == set(outcomes) for q in quotes):
        raise ValueError("todas las quotes deben tener los mismos outcomes")

    agg = {o: 0.0 for o in outcomes}
    for q in quotes:
        w = used_weights.get(q.book, 0.0)
        for o, p in q.probs.items():
            agg[o] += w * p

    # normalizar por seguridad (pequeños errores de redondeo)
    total = sum(agg.values())
    return {o: p / total for o, p in agg.items()}


# -------------------- helpers de alto nivel --------------------

def book_to_probs(
    raw_odds: Mapping[str, float],
    market: str,
    devig_method: str,
) -> dict[str, float]:
    """Pipeline: odds crudas → probs de-vigeadas para una sola casa, un solo mercado."""
    return devig(raw_odds, method=devig_method)


if __name__ == "__main__":
    # Smoke check
    odds_1x2 = {"H": 2.10, "D": 3.40, "A": 3.60}
    print("overround:", overround(odds_1x2))
    print("prop:", devig_proportional(odds_1x2))
    print("shin:", devig_shin(odds_1x2))

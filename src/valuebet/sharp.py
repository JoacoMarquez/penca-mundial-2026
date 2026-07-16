"""Precio justo desde el sharp (Pinnacle): wrapper fino sobre src.model.market_probs.

Shin para mercados 2-3 way (favoritos fuertes), proportional para exact score
(Shin con n grande es inestable).
"""

from __future__ import annotations

from src.model.market_probs import devig


def fair_probs(
    sharp_odds_same_market: dict[str, float],
    market: str,
    cfg_sharp: dict[str, str],
) -> dict[str, float]:
    """Odds sharp crudas de UN mercado completo → probs de-vigeadas.

    `sharp_odds_same_market` debe traer TODOS los outcomes del mercado
    (el de-vig necesita el vector completo), con odds > 1.0.
    """
    clean = {k: v for k, v in sharp_odds_same_market.items() if v and v > 1.0}
    if len(clean) < 2:
        raise ValueError(f"mercado {market!r} incompleto para de-vig: {sharp_odds_same_market}")
    method = cfg_sharp["exact_score_devig"] if market == "exact_score" else cfg_sharp["devig"]
    return devig(clean, method=method)

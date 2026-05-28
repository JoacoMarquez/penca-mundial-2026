"""Detección de anomalías en el movimiento de odds entre pasadas.

Si entre T-24h y T-3h (o T-3h y T-30min) las probabilidades del mercado se mueven
significativamente (> 5pp en P(home/draw/away)), es señal de algo: lesión confirmada,
cambio de DT, suspensión, etc. Disparamos:
    1. Flag en el JSON de predicción
    2. Alerta a Telegram con la dirección del movimiento

El threshold default es 5pp. Si Pinnacle se mueve esto entre pasadas, vale la pena re-procesar
todo el contexto LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


ANOMALY_THRESHOLD_PP = 5.0   # puntos porcentuales


@dataclass(frozen=True)
class OddsAnomaly:
    """Resultado de comparar dos pasadas de odds."""
    detected: bool
    delta_home_pp: float
    delta_draw_pp: float
    delta_away_pp: float
    max_abs_delta_pp: float
    direction: str           # "shift toward home" | "shift toward away" | "shift toward draw" | "none"
    summary: str             # texto legible para alerta


def compare_market_probs(
    previous: dict | None,
    current: dict,
    threshold_pp: float = ANOMALY_THRESHOLD_PP,
) -> OddsAnomaly | None:
    """Compara constraints (p_home, p_draw, p_away) entre dos pasadas.

    Args:
        previous: dict con keys p_home, p_draw, p_away. None si no hay versión previa.
        current: ídem para versión actual.
        threshold_pp: umbral en puntos porcentuales (default 5).

    Returns:
        OddsAnomaly si detectó movimiento significativo. None si no hubo previa o no hay anomalía.
    """
    if not previous:
        return None

    try:
        d_home = (current.get("p_home", 0) - previous.get("p_home", 0)) * 100
        d_draw = (current.get("p_draw", 0) - previous.get("p_draw", 0)) * 100
        d_away = (current.get("p_away", 0) - previous.get("p_away", 0)) * 100
    except Exception:
        return None

    max_abs = max(abs(d_home), abs(d_draw), abs(d_away))
    if max_abs < threshold_pp:
        return None  # sin anomalía

    # Dirección dominante: cuál outcome se movió más positivamente
    sorted_deltas = sorted([("home", d_home), ("draw", d_draw), ("away", d_away)], key=lambda x: -x[1])
    biggest_up = sorted_deltas[0]
    biggest_down = sorted_deltas[-1]
    direction = f"shift toward {biggest_up[0]} (+{biggest_up[1]:.1f}pp), away from {biggest_down[0]} ({biggest_down[1]:+.1f}pp)"

    summary = (
        f"⚠️ Movimiento de odds entre pasadas: "
        f"P(local) {previous.get('p_home',0)*100:.0f}%→{current.get('p_home',0)*100:.0f}% ({d_home:+.1f}pp) · "
        f"P(empate) {previous.get('p_draw',0)*100:.0f}%→{current.get('p_draw',0)*100:.0f}% ({d_draw:+.1f}pp) · "
        f"P(visit) {previous.get('p_away',0)*100:.0f}%→{current.get('p_away',0)*100:.0f}% ({d_away:+.1f}pp). "
        f"{direction}. Razón probable: lesión, suspensión o cambio táctico no anunciado todavía."
    )

    return OddsAnomaly(
        detected=True,
        delta_home_pp=round(d_home, 2),
        delta_draw_pp=round(d_draw, 2),
        delta_away_pp=round(d_away, 2),
        max_abs_delta_pp=round(max_abs, 2),
        direction=direction,
        summary=summary,
    )

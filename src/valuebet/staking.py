"""Kelly fraccionado + caps de exposición.

Kelly pleno es demasiado agresivo cuando la prob estimada tiene ruido (y la nuestra
lo tiene: viene de un de-vig, no de un oráculo). Fracción default 0.25, multiplicada
por el multiplicador de learning del segmento.
"""

from __future__ import annotations

from src.valuebet.config import VBConfig


def kelly_stake(p: float, odds: float, bankroll: float, fraction: float) -> float:
    """Stake por criterio de Kelly fraccionado. 0 si no hay edge.

    f* = (p*odds - 1) / (odds - 1)
    """
    if odds <= 1.0 or not (0.0 < p < 1.0):
        return 0.0
    f_star = (p * odds - 1.0) / (odds - 1.0)
    if f_star <= 0:
        return 0.0
    return bankroll * fraction * f_star


def apply_caps(
    stake: float,
    kind: str,                 # "single" | "parlay"
    bankroll: float,
    open_exposure: float,      # suma de stakes de sugerencias abiertas (mismo modo)
    open_parlay_exposure: float,
    cfg: VBConfig,
) -> float:
    """Aplica caps de riesgo y redondeo. Devuelve 0.0 si la apuesta se descarta."""
    if kind == "single":
        stake = min(stake, bankroll * cfg.caps["max_single_pct"])
    elif kind == "parlay":
        stake = min(stake, bankroll * float(cfg.parlay["max_parlay_pct"]))
        parlay_room = bankroll * float(cfg.parlay["max_open_parlay_pct"]) - open_parlay_exposure
        stake = min(stake, max(0.0, parlay_room))
    else:
        raise ValueError(f"kind desconocido: {kind!r}")

    total_room = bankroll * cfg.caps["max_open_pct"] - open_exposure
    stake = min(stake, max(0.0, total_room))

    # redondeo a múltiplos configurados; mínimo o se descarta
    r = cfg.stake_round_to
    stake = round(stake / r) * r
    if stake < cfg.stake_min:
        return 0.0
    return float(stake)

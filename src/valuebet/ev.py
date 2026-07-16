"""Detección de +EV: singles y parlays.

edge = fair_prob(sharp) * cuota_soft - 1

Filtros de singles: edge >= umbral efectivo del segmento, cuota dentro de odds_range,
partido a más de min_hours_ahead, segmento activo en learning.

Parlays: solo legs que ya son +EV como singles, eventos DISTINTOS (nunca dos mercados
del mismo partido — correlación destruye la multiplicación de probs), máx N legs,
edge combinado = ∏(1+e_i) - 1 >= parlay.min_edge.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from itertools import combinations

from src.valuebet.config import VBConfig
from src.valuebet.learning import LearningState
from src.valuebet.types import OddsQuote, Opportunity

log = logging.getLogger(__name__)


def find_singles(
    matched: list[tuple[OddsQuote, dict[str, float], OddsQuote]],
    cfg: VBConfig,
    learning: LearningState,
    now_utc: datetime | None = None,
) -> list[Opportunity]:
    """Detecta value bets simples.

    `matched`: lista de (cuota_soft, fair_probs_del_mercado_completo, cuota_sharp_del_outcome).
    fair_probs viene de sharp.fair_probs sobre el mercado sharp completo. La cuota
    sharp viene entera (no solo odds) para persistir su event_id en la Opportunity —
    el close lo necesita para reencontrar la línea de cierre sin re-matchear nombres.
    """
    now = now_utc or datetime.now(timezone.utc)
    lo, hi = cfg.odds_range
    out: list[Opportunity] = []

    for quote, fair, sharp_q in matched:
        p = fair.get(quote.outcome)
        if p is None or not (0.0 < p < 1.0):
            continue
        if not (lo <= quote.decimal_odds <= hi):
            continue
        start = datetime.fromisoformat(quote.start_utc.replace("Z", "+00:00"))
        if (start - now).total_seconds() < cfg.min_hours_ahead * 3600:
            continue

        opp = Opportunity(
            quote=quote, fair_prob=p, sharp_odds=sharp_q.decimal_odds,
            edge=p * quote.decimal_odds - 1.0,
            sharp_event_id=sharp_q.event_id,
        )
        # Guardrail: un edge enorme es casi seguro un error de datos o un match a un
        # mercado equivocado (ej. cuota de tarjetas vs cuota de resultado), no una
        # oportunidad real. Se descarta y se loguea para revisar, nunca se sugiere.
        if opp.edge > cfg.max_edge:
            log.warning("edge sospechoso %.0f%% descartado (>%.0f%%): %s %s %s @ %.2f (sharp %.2f)",
                        opp.edge * 100, cfg.max_edge * 100, quote.sport, quote.event_name,
                        quote.outcome, quote.decimal_odds, sharp_q.decimal_odds)
            continue
        if not learning.is_active(opp.segment):
            continue
        threshold = cfg.min_edge[quote.sport] / learning.multiplier(opp.segment)
        if quote.market == "exact_score":
            threshold += cfg.exact_score_extra_edge
        if opp.edge >= threshold:
            out.append(opp)

    return sorted(out, key=lambda o: o.edge, reverse=True)


def find_parlays(singles: list[Opportunity], cfg: VBConfig) -> list[list[Opportunity]]:
    """Combina singles +EV en parlays de eventos independientes."""
    if not cfg.parlay.get("enabled", False) or len(singles) < 2:
        return []
    max_legs = int(cfg.parlay["max_legs"])
    min_edge = float(cfg.parlay["min_edge"])

    # una sola pata por evento: quedarse con la de mayor edge de cada event_id
    best_by_event: dict[str, Opportunity] = {}
    for o in singles:
        cur = best_by_event.get(o.quote.event_id)
        if cur is None or o.edge > cur.edge:
            best_by_event[o.quote.event_id] = o
    legs_pool = sorted(best_by_event.values(), key=lambda o: o.edge, reverse=True)

    out: list[list[Opportunity]] = []
    for k in range(2, max_legs + 1):
        for combo in combinations(legs_pool, k):
            edge = 1.0
            for o in combo:
                edge *= (1.0 + o.edge)
            edge -= 1.0
            if edge >= min_edge:
                out.append(list(combo))
    # ordenar por edge combinado desc y devolver como máximo 3 (no spamear)
    out.sort(key=lambda legs: combined_edge(legs), reverse=True)
    return out[:3]


def combined_edge(legs: list[Opportunity]) -> float:
    e = 1.0
    for o in legs:
        e *= (1.0 + o.edge)
    return e - 1.0


def combined_odds(legs: list[Opportunity]) -> float:
    x = 1.0
    for o in legs:
        x *= o.quote.decimal_odds
    return x


def combined_fair_prob(legs: list[Opportunity]) -> float:
    p = 1.0
    for o in legs:
        p *= o.fair_prob
    return p

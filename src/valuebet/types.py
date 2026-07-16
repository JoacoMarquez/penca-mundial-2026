"""Dataclasses compartidas del sistema valuebet.

Convención de mercados (string `market`):
    "1x2"          — fútbol H/D/A
    "moneyline"    — 2-way (tenis, básquet con OT incluido)
    "total_<pts>"  — over/under con la línea embebida, ej "total_2.5", "total_212.5"
    "exact_score"  — marcador exacto fútbol, outcome "2-1" etc.

`outcome` es la key dentro del mercado: "home"|"draw"|"away"|"over"|"under"|"2-1"...
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OddsQuote:
    """Una cuota puntual de una casa para un outcome de un mercado."""
    book: str            # "supermatch" | "pinnacle" | "oddsapi:<bookie>"
    sport: str           # "soccer" | "basketball" | "tennis"
    league: str
    event_id: str        # id estable dentro de la casa
    event_name: str      # "Nacional vs Peñarol"
    start_utc: str       # ISO 8601
    market: str
    outcome: str
    decimal_odds: float
    fetched_utc: str     # ISO 8601


@dataclass(frozen=True)
class Opportunity:
    """Un value bet detectado: cuota soft vs prob justa del sharp."""
    quote: OddsQuote               # la cuota soft (Supermatch)
    fair_prob: float               # prob de-vigeada del sharp para el mismo outcome
    sharp_odds: float              # cuota sharp cruda (para referencia/CLV)
    edge: float                    # fair_prob * quote.decimal_odds - 1

    @property
    def segment(self) -> str:
        """deporte|mercado_base|banda de cuota — clave del learning."""
        return segment_key(self.quote.sport, self.quote.market, self.quote.decimal_odds)


def segment_key(sport: str, market: str, odds: float) -> str:
    base_market = market.split("_")[0] if market.startswith("total") else market
    if odds < 2.0:
        band = "1.4-2.0"
    elif odds < 3.0:
        band = "2.0-3.0"
    else:
        band = "3.0-6.0"
    return f"{sport}|{base_market}|{band}"


@dataclass
class Suggestion:
    """Una fila del ledger. legs=1 → simple; legs>1 → parlay."""
    id: str
    ts_utc: str
    mode: str                          # "paper" | "real"
    legs: list[dict[str, Any]]         # Opportunity serializadas (asdict)
    combined_odds: float
    combined_fair_prob: float
    edge: float
    stake_suggested: float
    bankroll_at_ts: float
    kelly_fraction_used: float
    status: str = "open"               # open | won | lost | void | push
    taken: bool | None = None          # en modo real, confirmación del usuario
    stake_real: float | None = None
    closing_odds: float | None = None          # producto de cuotas sharp al cierre
    closing_fair_prob: float | None = None     # producto de fair probs al cierre
    clv: float | None = None                   # combined_odds * closing_fair_prob - 1
    pnl: float | None = None
    legs_closing: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_parlay(self) -> bool:
        return len(self.legs) > 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_suggestion_id() -> str:
    return uuid.uuid4().hex[:8]


def opportunity_to_dict(opp: Opportunity) -> dict[str, Any]:
    d = asdict(opp)
    d["segment"] = opp.segment
    return d


def suggestion_from_dict(d: dict[str, Any]) -> Suggestion:
    known = {f for f in Suggestion.__dataclass_fields__}
    return Suggestion(**{k: v for k, v in d.items() if k in known})

"""Carga y validación estricta de config/valuebet.yaml.

Falla ruidoso si falta una clave — mejor un crash al deploy que un default silencioso
que apueste distinto de lo que el operador cree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(os.environ.get("VALUEBET_CONFIG", "config/valuebet.yaml"))

_REQUIRED_KEYS = [
    "mode", "bankroll_file", "kelly_fraction", "min_edge", "odds_range",
    "min_hours_ahead", "max_edge", "markets", "parlay", "caps", "stake_round_to",
    "stake_min", "scan_times_utc", "sharp", "sport_ids", "leagues", "aliases",
    "exact_score_extra_edge",
]


@dataclass
class VBConfig:
    mode: str
    bankroll_file: Path
    kelly_fraction: float
    min_edge: dict[str, float]
    odds_range: tuple[float, float]
    min_hours_ahead: float
    max_edge: float
    markets: dict[str, list[str]]
    exact_score_extra_edge: float
    parlay: dict[str, Any]
    caps: dict[str, float]
    stake_round_to: int
    stake_min: float
    scan_times_utc: list[str]
    sharp: dict[str, str]
    sport_ids: dict[str, int]
    leagues: dict[str, list[int]]
    aliases: dict[str, dict[str, list[str]]]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def sports(self) -> list[str]:
        return list(self.markets.keys())


def load(path: Path | None = None) -> VBConfig:
    p = path or DEFAULT_CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise KeyError(f"config/valuebet.yaml: faltan claves {missing}")

    if raw["mode"] not in ("paper", "real"):
        raise ValueError(f"mode inválido: {raw['mode']!r} (paper|real)")
    lo, hi = raw["odds_range"]
    if not (1.0 < lo < hi):
        raise ValueError(f"odds_range inválido: {raw['odds_range']}")
    if not (0.0 < raw["kelly_fraction"] <= 1.0):
        raise ValueError(f"kelly_fraction fuera de (0, 1]: {raw['kelly_fraction']}")
    for sport in raw["markets"]:
        if sport not in raw["min_edge"]:
            raise ValueError(f"min_edge sin entrada para deporte {sport!r}")
        if sport not in raw["sport_ids"]:
            raise ValueError(f"sport_ids sin entrada para deporte {sport!r}")

    return VBConfig(
        mode=raw["mode"],
        bankroll_file=Path(raw["bankroll_file"]),
        kelly_fraction=float(raw["kelly_fraction"]),
        min_edge={k: float(v) for k, v in raw["min_edge"].items()},
        odds_range=(float(lo), float(hi)),
        min_hours_ahead=float(raw["min_hours_ahead"]),
        max_edge=float(raw["max_edge"]),
        markets=raw["markets"],
        exact_score_extra_edge=float(raw["exact_score_extra_edge"]),
        parlay=raw["parlay"],
        caps={k: float(v) for k, v in raw["caps"].items()},
        stake_round_to=int(raw["stake_round_to"]),
        stake_min=float(raw["stake_min"]),
        scan_times_utc=raw["scan_times_utc"],
        sharp=raw["sharp"],
        sport_ids={k: int(v) for k, v in raw["sport_ids"].items()},
        leagues=raw["leagues"],
        aliases=raw["aliases"],
        raw=raw,
    )

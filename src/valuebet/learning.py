"""Loop de aprendizaje: edge realizado (vía CLV) por segmento, con shrinkage bayesiano.

CLV (closing line value) converge en ~30-50 apuestas; el PnL necesita cientos. Por eso
el learning se alimenta de CLV apenas cierra la línea, sin esperar resultados.

Segmento = "deporte|mercado|banda_de_cuota" (ver types.segment_key).

Shrinkage: edge_shrunk = (k * prior + n * clv_medio) / (k + n), prior = 0 (escéptico),
k = 25 pseudo-observaciones. El multiplicador resultante ajusta DOS perillas:
  1. umbral de sugerencia:   min_edge_efectivo = cfg.min_edge / multiplier
  2. tamaño de apuesta:      fraction_efectiva = kelly_fraction * multiplier
Segmento con n >= 20 y edge_shrunk < -1% se desactiva (multiplier = 0) y se avisa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.versions import latest_version
from src.valuebet.types import Suggestion

PRIOR_EDGE = 0.0
PSEUDO_OBS_K = 25.0
DEACTIVATE_N = 20
DEACTIVATE_EDGE = -0.01
MULT_MIN, MULT_MAX = 0.3, 1.5


@dataclass
class LearningState:
    # segment -> {"n": int, "sum_clv": float, "sum_edge_pred": float, "multiplier": float}
    segments: dict[str, dict[str, float]] = field(default_factory=dict)

    def multiplier(self, segment: str) -> float:
        return float(self.segments.get(segment, {}).get("multiplier", 1.0))

    def is_active(self, segment: str) -> bool:
        return self.multiplier(segment) > 0.0

    # -------------------- update --------------------

    def observe(self, segment: str, clv: float, edge_pred: float) -> None:
        seg = self.segments.setdefault(
            segment, {"n": 0, "sum_clv": 0.0, "sum_edge_pred": 0.0, "multiplier": 1.0}
        )
        seg["n"] += 1
        seg["sum_clv"] += clv
        seg["sum_edge_pred"] += edge_pred
        seg["multiplier"] = self._compute_multiplier(seg)

    @staticmethod
    def _compute_multiplier(seg: dict[str, float]) -> float:
        n = seg["n"]
        if n == 0:
            return 1.0
        clv_mean = seg["sum_clv"] / n
        edge_shrunk = (PSEUDO_OBS_K * PRIOR_EDGE + n * clv_mean) / (PSEUDO_OBS_K + n)
        if n >= DEACTIVATE_N and edge_shrunk < DEACTIVATE_EDGE:
            return 0.0
        edge_pred_mean = seg["sum_edge_pred"] / n
        if edge_pred_mean <= 0:
            return 1.0
        mult = 1.0 + edge_shrunk / edge_pred_mean
        return float(min(MULT_MAX, max(MULT_MIN, mult)))


def update_from_suggestion(state: LearningState, s: Suggestion) -> list[str]:
    """Incorpora el CLV de una sugerencia cerrada. Devuelve segmentos desactivados en este update.

    Para parlays, el CLV combinado se atribuye por leg usando el CLV individual
    (cada leg trae su closing en legs_closing).
    """
    deactivated: list[str] = []
    if s.clv is None:
        return deactivated
    legs_closing = {lc["outcome_key"]: lc for lc in s.legs_closing}
    for leg in s.legs:
        key = _leg_outcome_key(leg)
        lc = legs_closing.get(key)
        clv_leg = lc["clv"] if lc and lc.get("clv") is not None else s.clv
        segment = leg["segment"]
        was_active = state.is_active(segment)
        state.observe(segment, clv=clv_leg, edge_pred=leg["edge"])
        if was_active and not state.is_active(segment):
            deactivated.append(segment)
    return deactivated


def _leg_outcome_key(leg: dict[str, Any]) -> str:
    q = leg["quote"]
    return f"{q['event_id']}|{q['market']}|{q['outcome']}"


# -------------------- persistencia (JSON versionado, patrón penca) --------------------

def _learning_dir() -> Path:
    import os
    return Path(os.environ.get("VALUEBET_DATA_DIR", "data/valuebet")) / "learning"


def load_state(dirpath: Path | None = None) -> LearningState:
    d = dirpath or _learning_dir()
    if not d.exists():
        return LearningState()
    latest = latest_version(d.glob("v*_*.json"))
    if latest is None:
        return LearningState()
    with open(latest, encoding="utf-8") as f:
        raw = json.load(f)
    return LearningState(segments=raw.get("segments", {}))


def save_state(state: LearningState, dirpath: Path | None = None) -> Path:
    d = dirpath or _learning_dir()
    d.mkdir(parents=True, exist_ok=True)
    latest = latest_version(d.glob("v*_*.json"))
    n = 1 if latest is None else int(latest.name.split("_")[0][1:]) + 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = d / f"v{n}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"segments": state.segments}, f, ensure_ascii=False, indent=1)
    return path

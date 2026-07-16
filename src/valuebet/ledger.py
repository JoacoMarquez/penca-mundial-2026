"""Libro de sugerencias: persistencia versionada, settlement, PnL, CLV, bankroll.

Persistencia: data/valuebet/ledger/vN_<ts>.json — snapshot COMPLETO de la lista de
sugerencias en cada escritura (mismo patrón que las predicciones penca: nunca
sobreescribir, siempre versionar). Con <10k filas/año, JSON alcanza de sobra.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.utils.versions import latest_version
from src.valuebet.types import Suggestion, suggestion_from_dict


def _data_dir() -> Path:
    return Path(os.environ.get("VALUEBET_DATA_DIR", "data/valuebet"))


def _ledger_dir() -> Path:
    return _data_dir() / "ledger"


# -------------------- lectura / escritura --------------------

def load_all(dirpath: Path | None = None) -> list[Suggestion]:
    d = dirpath or _ledger_dir()
    if not d.exists():
        return []
    latest = latest_version(d.glob("v*_*.json"))
    if latest is None:
        return []
    with open(latest, encoding="utf-8") as f:
        raw = json.load(f)
    return [suggestion_from_dict(x) for x in raw]


def save_all(suggestions: list[Suggestion], dirpath: Path | None = None) -> Path:
    d = dirpath or _ledger_dir()
    d.mkdir(parents=True, exist_ok=True)
    latest = latest_version(d.glob("v*_*.json"))
    n = 1 if latest is None else int(latest.name.split("_")[0][1:]) + 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = d / f"v{n}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in suggestions], f, ensure_ascii=False, indent=1)
    return path


def append(new: list[Suggestion], dirpath: Path | None = None) -> Path:
    cur = load_all(dirpath)
    return save_all(cur + new, dirpath)


def open_suggestions(all_s: list[Suggestion] | None = None) -> list[Suggestion]:
    return [s for s in (all_s if all_s is not None else load_all()) if s.status == "open"]


def find(suggestion_id: str, all_s: list[Suggestion]) -> Suggestion | None:
    return next((s for s in all_s if s.id == suggestion_id), None)


def is_duplicate(candidate_legs_keys: set[str], all_s: list[Suggestion], min_edge_gain: float, candidate_edge: float) -> bool:
    """Misma event+market+outcome abierta no se re-sugiere salvo que el edge suba > min_edge_gain."""
    for s in open_suggestions(all_s):
        keys = {f"{l['quote']['event_id']}|{l['quote']['market']}|{l['quote']['outcome']}" for l in s.legs}
        if keys == candidate_legs_keys and candidate_edge < s.edge + min_edge_gain:
            return True
    return False


# -------------------- settlement --------------------

def settle(s: Suggestion, result: str) -> Suggestion:
    """result: won | lost | void | push. Calcula pnl sobre el stake efectivo."""
    if result not in ("won", "lost", "void", "push"):
        raise ValueError(f"resultado inválido: {result!r}")
    stake = s.stake_real if (s.taken and s.stake_real) else s.stake_suggested
    s.status = result
    if result == "won":
        s.pnl = stake * (s.combined_odds - 1.0)
    elif result == "lost":
        s.pnl = -stake
    else:  # void / push
        s.pnl = 0.0
    return s


# -------------------- CLV --------------------

def apply_closing(s: Suggestion, legs_closing: list[dict]) -> Suggestion:
    """legs_closing: [{outcome_key, closing_odds, closing_fair_prob, clv}] por leg.

    CLV combinado = cuota_tomada_combinada * ∏(closing_fair_prob) - 1.
    """
    s.legs_closing = legs_closing
    fair = 1.0
    odds = 1.0
    for lc in legs_closing:
        if lc.get("closing_fair_prob") is None:
            return s  # cierre incompleto: no computar CLV parcial
        fair *= lc["closing_fair_prob"]
        odds *= lc.get("closing_odds") or 1.0
    s.closing_fair_prob = fair
    s.closing_odds = odds
    s.clv = s.combined_odds * fair - 1.0
    return s


# -------------------- bankroll --------------------

def load_bankroll(path: Path | None = None) -> float:
    p = path or (_data_dir() / "bankroll.json")
    if not p.exists():
        raise FileNotFoundError(
            f"no existe {p} — inicializá con `python -m src.valuebet.runner bankroll --set N`"
        )
    with open(p, encoding="utf-8") as f:
        return float(json.load(f)["bankroll"])


def save_bankroll(amount: float, path: Path | None = None) -> None:
    p = path or (_data_dir() / "bankroll.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"bankroll": amount, "currency": "UYU"}, f)


def open_exposure(all_s: list[Suggestion], mode: str) -> tuple[float, float]:
    """(exposición abierta total, exposición abierta en parlays) para el modo dado."""
    total = 0.0
    parlay = 0.0
    for s in open_suggestions(all_s):
        if s.mode != mode:
            continue
        stake = s.stake_real if (s.taken and s.stake_real) else s.stake_suggested
        total += stake
        if s.is_parlay:
            parlay += stake
    return total, parlay

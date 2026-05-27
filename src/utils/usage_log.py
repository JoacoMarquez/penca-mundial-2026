"""Tracking de uso y costo de la API de Anthropic.

Cada llamada al LLM loguea una línea JSON en `data/anthropic_usage.jsonl` con:
    timestamp, model, input_tokens, output_tokens, cache_creation_tokens,
    cache_read_tokens, cost_usd, purpose, match_id

El heartbeat agrega esa data para mostrar gasto día/mes/total.

Precios (USD por 1M tokens) — actualizados a 2026-05:
    Sonnet 4.6:  input $3.00 / output $15.00 / cache_write $3.75 / cache_read $0.30
    Opus 4.7:    input $15.00 / output $75.00 / cache_write $18.75 / cache_read $1.50
    Haiku 4.5:   input $1.00 / output $5.00
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# USD por 1M tokens
PRICING_PER_M_TOK: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {
        "input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30,
    },
    "claude-opus-4-7": {
        "input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50,
    },
    "claude-haiku-4-5": {
        "input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10,
    },
}


def _usage_log_path() -> Path:
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    return data_dir / "anthropic_usage.jsonl"


def estimate_cost_usd(model: str, usage: Any) -> float:
    """Calcula USD a partir del objeto usage del response de Anthropic.

    `usage` puede ser un dict o un objeto Pydantic con atributos input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens.
    """
    # Tomamos por defecto Sonnet si el modelo no está en la tabla
    rate = PRICING_PER_M_TOK.get(model, PRICING_PER_M_TOK["claude-sonnet-4-6"])

    def get(name: str) -> int:
        if isinstance(usage, dict):
            return int(usage.get(name) or 0)
        return int(getattr(usage, name, 0) or 0)

    inp = get("input_tokens")
    out = get("output_tokens")
    cw = get("cache_creation_input_tokens")
    cr = get("cache_read_input_tokens")

    cost = (
        inp * rate["input"]
        + out * rate["output"]
        + cw * rate.get("cache_write", rate["input"])
        + cr * rate.get("cache_read", rate["input"] / 10)
    ) / 1_000_000
    return round(cost, 6)


def log_anthropic_call(
    model: str,
    usage: Any,
    purpose: str,
    match_id: str | int | None = None,
) -> None:
    """Loguea una entry de uso. No-throws — si falla, solo warn."""
    try:
        def get(name: str) -> int:
            if isinstance(usage, dict):
                return int(usage.get(name) or 0)
            return int(getattr(usage, name, 0) or 0)

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "input_tokens": get("input_tokens"),
            "output_tokens": get("output_tokens"),
            "cache_write_tokens": get("cache_creation_input_tokens"),
            "cache_read_tokens": get("cache_read_input_tokens"),
            "cost_usd": estimate_cost_usd(model, usage),
            "purpose": purpose,
            "match_id": match_id,
        }
        path = _usage_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning("falló log de uso Anthropic: %s", e)


def read_usage_summary(within_hours: int | None = None) -> dict[str, Any]:
    """Lee el JSONL y suma costos. Si within_hours=None, devuelve total.

    Retorna: {"calls": N, "input_tokens": ..., "output_tokens": ..., "cost_usd": ..., "by_purpose": {...}}
    """
    path = _usage_log_path()
    if not path.exists():
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "by_purpose": {}}

    cutoff_ts = None
    if within_hours is not None:
        cutoff = datetime.now(timezone.utc).timestamp() - within_hours * 3600
        cutoff_ts = cutoff

    calls = 0
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    by_purpose: dict[str, dict[str, float | int]] = {}

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cutoff_ts is not None:
            try:
                ts = datetime.fromisoformat(e["ts"]).timestamp()
                if ts < cutoff_ts:
                    continue
            except Exception:
                continue

        calls += 1
        input_tokens += e.get("input_tokens", 0)
        output_tokens += e.get("output_tokens", 0)
        cost_usd += e.get("cost_usd", 0.0)

        p = e.get("purpose", "unknown")
        if p not in by_purpose:
            by_purpose[p] = {"calls": 0, "cost_usd": 0.0}
        by_purpose[p]["calls"] += 1
        by_purpose[p]["cost_usd"] += e.get("cost_usd", 0.0)

    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 4),
        "by_purpose": by_purpose,
    }

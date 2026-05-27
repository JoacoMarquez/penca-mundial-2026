"""Scheduler que el systemd timer ejecuta cada 5 min.

Lee fixtures.yaml, calcula qué partido entra en alguna ventana (T-24h, T-3h, T-30min) AHORA,
y dispara la pipeline correspondiente. Idempotente: si ya corrimos esa fase para ese partido,
no la repite.

La ventana es ±2.5 min para emparejar con el timer cada 5 min. Si la ventana T-24h cae a las
14:32 y el timer corre 14:30 y 14:35, agarra una de las dos.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from src.agent.pipeline import Phase, load_fixtures, PREDICTIONS_DIR, run_match_pipeline

log = logging.getLogger(__name__)


PHASE_OFFSETS_MIN = {
    Phase.T_24H: 24 * 60,
    Phase.T_3H: 3 * 60,
    Phase.T_30MIN: 30,
}
WINDOW_HALF_MIN = 2.5   # ±2.5 min alrededor del target → emparejar con timer de 5 min


def parse_kickoff(iso: str) -> datetime:
    """Parsea ISO 8601 UTC (con Z) → datetime aware UTC."""
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def phase_already_ran(match_id: str, phase: Phase) -> bool:
    """Chequea en data/predictions/{match_id}/ si ya hay alguna versión con esa fase."""
    match_dir = PREDICTIONS_DIR / match_id
    if not match_dir.exists():
        return False
    for f in match_dir.glob("v*_*.json"):
        try:
            import json
            data = json.loads(f.read_text())
            if data.get("phase") == phase.value or data.get("phase") == phase:
                return True
        except Exception:
            continue
    return False


def matches_in_window(fixtures: dict, now: datetime | None = None) -> list[tuple[str, Phase]]:
    """Devuelve (match_id, phase) que deberían correrse AHORA dentro de la ventana ±2.5 min."""
    now = now or datetime.now(timezone.utc)
    out = []

    all_matches = fixtures.get("fase_grupos", []) + fixtures.get("eliminatorias", [])
    for m in all_matches:
        kickoff = parse_kickoff(m["kickoff_utc"])
        for phase, offset_min in PHASE_OFFSETS_MIN.items():
            target = kickoff - timedelta(minutes=offset_min)
            delta_min = (now - target).total_seconds() / 60.0
            if abs(delta_min) <= WINDOW_HALF_MIN:
                if phase_already_ran(m["id"], phase):
                    continue
                out.append((m["id"], phase))

    return out


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
    )
    fixtures = load_fixtures()
    pending = matches_in_window(fixtures)
    if not pending:
        log.info("scheduler tick — nada que hacer")
        return 0
    log.info("scheduler tick — %d tarea(s): %s", len(pending), pending)
    for match_id, phase in pending:
        try:
            run_match_pipeline(match_id, phase)
        except Exception:
            log.exception("error corriendo pipeline | match=%s phase=%s", match_id, phase.value)
            # TODO: notificar via Telegram.send_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

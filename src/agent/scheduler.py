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
    """Devuelve (match_id, phase) que deberían correrse AHORA, con CATCH-UP.

    Para cada partido aún no jugado (now < kickoff), una fase está "pendiente" si su target
    (kickoff − offset) ya pasó y todavía no corrió. Devolvemos la fase pendiente MÁS TARDÍA
    (la más cercana al kickoff): si nos atrasamos o se saltó un slot del timer, igual disparamos
    la pasada relevante en el próximo despertar — sin ventana fija de ±N min que se pueda perder.

    Garantía clave: T-30min (la que PUBLICA) corre sí o sí mientras el scheduler despierte al
    menos una vez entre su target y el kickoff. `phase_already_ran` evita duplicados.
    """
    now = now or datetime.now(timezone.utc)
    out = []

    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    for m in all_matches:
        kickoff = parse_kickoff(m["kickoff_utc"])
        if now >= kickoff:
            continue  # partido ya arrancó (la web bloquea predicciones) → demasiado tarde

        # Fases cuyo target (kickoff − offset) ya pasó
        due = [
            (phase, offset_min)
            for phase, offset_min in PHASE_OFFSETS_MIN.items()
            if kickoff - timedelta(minutes=offset_min) <= now
        ]
        if not due:
            continue
        # La pasada MÁS RELEVANTE = la de target más tardío = menor offset (más cerca del
        # kickoff). Solo la emitimos si todavía no corrió. Si ya corrió, no hay nada que
        # hacer — NO volvemos a fases anteriores (ya publicamos / es agua pasada).
        latest_phase = min(due, key=lambda p: p[1])[0]
        if not phase_already_ran(m["id"], latest_phase):
            out.append((m["id"], latest_phase))

    return out


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
    )
    fixtures = load_fixtures()
    pending = matches_in_window(fixtures)

    # Pipeline para partidos en ventana
    if pending:
        log.info("scheduler tick — %d tarea(s): %s", len(pending), pending)
        for match_id, phase in pending:
            try:
                run_match_pipeline(match_id, phase)
            except Exception:
                log.exception("error corriendo pipeline | match=%s phase=%s", match_id, phase.value)
                _notify_error("pipeline", f"match={match_id} phase={phase.value}")

    # Postmortems para partidos terminados sin reporte aún
    try:
        from src.agent.postmortem import (
            find_finished_matches_pending_postmortem,
            compute_postmortem,
            save_postmortem,
        )
        from src.notifier.telegram import TelegramNotifier, TelegramConfig
        pm_pending = find_finished_matches_pending_postmortem(fixtures)
        if pm_pending:
            log.info("scheduler tick — %d postmortem(s) pendiente(s): %s", len(pm_pending), pm_pending)
            try:
                notifier = TelegramNotifier(TelegramConfig.from_env())
            except Exception:
                notifier = None
            for mid in pm_pending:
                try:
                    report = compute_postmortem(mid)
                    if not report:
                        continue
                    save_postmortem(report)
                    if notifier:
                        notifier.send_postmortem(report)
                    log.info("Postmortem completado para match %s", mid)
                except Exception:
                    log.exception("error en postmortem | match=%s", mid)
                    _notify_error("postmortem", f"match={mid}")
    except Exception:
        log.exception("error general en postmortem block")

    if not pending and not _has_pending_postmortems(fixtures):
        log.info("scheduler tick — nada que hacer")
    return 0


def _has_pending_postmortems(fixtures: dict) -> bool:
    try:
        from src.agent.postmortem import find_finished_matches_pending_postmortem
        return bool(find_finished_matches_pending_postmortem(fixtures))
    except Exception:
        return False


def _notify_error(context: str, detail: str) -> None:
    """Best-effort Telegram alert on internal errors."""
    try:
        from src.notifier.telegram import TelegramNotifier, TelegramConfig
        TelegramNotifier(TelegramConfig.from_env()).send_error(context, detail)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())

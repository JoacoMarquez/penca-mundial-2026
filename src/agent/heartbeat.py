"""Heartbeat diario: manda un mensaje a Telegram con resumen de estado.

Se ejecuta vía systemd timer una vez al día. Reporta:
    - Próximo partido + cuánto falta
    - Última versión de predicción generada (si hay)
    - Disk free + memoria
    - Estado de la API de la penca (¿auth funciona?)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.agent.pipeline import PREDICTIONS_DIR, load_fixtures
from src.notifier.telegram import TelegramConfig, TelegramNotifier, _escape_md

log = logging.getLogger(__name__)


def next_match_summary() -> str:
    """Próximo partido y cuánto falta."""
    fixtures = load_fixtures()
    now = datetime.now(timezone.utc)
    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    upcoming = []
    for m in all_matches:
        try:
            ko = datetime.fromisoformat(m["kickoff_utc"].replace("Z", "+00:00"))
            if ko > now:
                upcoming.append((ko, m))
        except Exception:
            continue
    if not upcoming:
        return "(no quedan partidos pendientes)"
    upcoming.sort()
    ko, m = upcoming[0]
    delta = ko - now
    days = delta.days
    hours = delta.seconds // 3600
    home = m.get("home_name") or m.get("home") or "?"
    away = m.get("away_name") or m.get("away") or "?"
    return f"{home} vs {away} en {days}d {hours}h"


def total_predictions_generated() -> int:
    if not PREDICTIONS_DIR.exists():
        return 0
    return sum(1 for _ in PREDICTIONS_DIR.rglob("v*_*.json"))


def disk_summary() -> str:
    du = shutil.disk_usage(str(PREDICTIONS_DIR.parent if PREDICTIONS_DIR.parent.exists() else "/"))
    return f"{du.free // (1024**3)}GB libres / {du.total // (1024**3)}GB"


def check_penca_api() -> str:
    base = os.environ.get("PENCA_API_BASE_URL", "").rstrip("/")
    key = os.environ.get("PENCA_API_KEY", "")
    if not base or not key:
        return "no configurada"
    try:
        with httpx.Client(timeout=5.0, headers={"Authorization": f"Bearer {key}"}) as c:
            r = c.get(f"{base}/matches")
        return f"OK (matches→{r.status_code})"
    except Exception as e:
        return f"ERR ({type(e).__name__})"


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        notif = TelegramNotifier(TelegramConfig.from_env())
    except Exception as e:
        log.error("no se pudo inicializar Telegram: %s", e)
        return 1

    next_m = _escape_md(next_match_summary())
    api_status = _escape_md(check_penca_api())
    disk = _escape_md(disk_summary())
    total_preds = total_predictions_generated()
    now_str = _escape_md(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    text = "\n".join([
        "💓 *Heartbeat Penca Mundial*",
        f"⏰ {now_str}",
        "",
        f"📅 Próximo: {next_m}",
        f"📊 Predicciones generadas: `{total_preds}`",
        f"💾 Disk: {disk}",
        f"🌐 API penca: {api_status}",
    ])
    notif.send(text)
    log.info("heartbeat OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

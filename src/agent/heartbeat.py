"""Heartbeat diario: mensaje a Telegram con resumen de TODOS los componentes.

Se ejecuta vía systemd timer a las 11:00 UTC (= 08:00 America/Montevideo).
Diseñado para detectar fallas silenciosas — si algún componente está rojo, se ve al toque.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from src.agent.pipeline import PREDICTIONS_DIR, load_fixtures
from src.notifier.telegram import TelegramConfig, TelegramNotifier

log = logging.getLogger(__name__)


UY_TZ = timezone(timedelta(hours=-3))


def _now_uy_str() -> str:
    return datetime.now(UY_TZ).strftime("%a %d/%m %H:%M UY")


def _next_match_summary() -> str:
    """Próximo partido + cuánto falta."""
    fixtures = load_fixtures()
    now = datetime.now(timezone.utc)
    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    upcoming = []
    for m in all_matches:
        try:
            ko = datetime.fromisoformat(m["kickoff_utc"].replace("Z", "+00:00"))
            if ko > now and (m.get("home") or m.get("home_name")):
                upcoming.append((ko, m))
        except Exception:
            continue
    if not upcoming:
        return "no quedan partidos"
    upcoming.sort(key=lambda x: x[0])
    ko, m = upcoming[0]
    delta = ko - now
    days = delta.days
    hours = delta.seconds // 3600
    home = m.get("home_name") or m.get("home") or "?"
    away = m.get("away_name") or m.get("away") or "?"
    if days > 0:
        return f"{home}-{away} en {days}d {hours}h"
    return f"{home}-{away} en {hours}h"


def _predictions_count(within_hours: int | None = None) -> int:
    if not PREDICTIONS_DIR.exists():
        return 0
    files = list(PREDICTIONS_DIR.rglob("v*_*.json"))
    if within_hours is None:
        return len(files)
    cutoff = datetime.now().timestamp() - within_hours * 3600
    return sum(1 for f in files if f.stat().st_mtime > cutoff)


def _disk_free() -> str:
    du = shutil.disk_usage("/var/lib/penca" if Path("/var/lib/penca").exists() else "/")
    return f"{du.free // (1024**3)}GB libres ({100 - int(du.used * 100 / du.total)}%)"


def _ram_used() -> str:
    try:
        out = subprocess.run(["free", "-m"], capture_output=True, text=True, check=True).stdout
        # Mem: total used free shared buff/cache available
        line = [l for l in out.splitlines() if l.startswith("Mem:")][0]
        parts = line.split()
        total, used = int(parts[1]), int(parts[2])
        return f"{used}MB / {total}MB ({int(used*100/total)}%)"
    except Exception:
        return "?"


def _scheduler_status() -> tuple[str, str]:
    """Devuelve (status_string, last_run_string)."""
    try:
        timer = subprocess.run(
            ["systemctl", "is-active", "penca-scheduler.timer"],
            capture_output=True, text=True,
        ).stdout.strip()
        last_run = subprocess.run(
            ["systemctl", "show", "penca-scheduler.service", "-p", "ActiveEnterTimestamp", "--value"],
            capture_output=True, text=True,
        ).stdout.strip()
        result = subprocess.run(
            ["systemctl", "show", "penca-scheduler.service", "-p", "Result", "--value"],
            capture_output=True, text=True,
        ).stdout.strip()
        status = f"{timer} ({result})"
        return status, last_run or "—"
    except Exception as e:
        return f"err: {e}", "—"


def _check_api_penca() -> str:
    base = os.environ.get("PENCA_API_BASE_URL", "").rstrip("/")
    key = os.environ.get("PENCA_API_KEY", "")
    if not base or not key:
        return "no configurada"
    try:
        with httpx.Client(timeout=5.0, headers={"Authorization": f"Bearer {key}"}) as c:
            r = c.get(f"{base}/matches")
        return f"OK ({r.status_code})" if r.status_code == 200 else f"err ({r.status_code})"
    except Exception as e:
        return f"err ({type(e).__name__})"


def _check_pinnacle() -> str:
    headers = {
        "X-API-Key": "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R",
        "Referer": "https://www.pinnacle.com/",
        "User-Agent": "Mozilla/5.0",
    }
    try:
        with httpx.Client(timeout=5.0, headers=headers) as c:
            r = c.get("https://guest.api.arcadia.pinnacle.com/0.1/sports/29/leagues")
        return f"OK ({r.status_code})" if r.status_code == 200 else f"err ({r.status_code})"
    except Exception as e:
        return f"err ({type(e).__name__})"


def _check_anthropic() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("sk-ant-xxx"):
        return "no configurada"
    # No hacemos llamada real para no gastar tokens — solo verificamos shape de la key
    if key.startswith("sk-ant-"):
        return "OK (key presente)"
    return "key inválida"


def _errors_24h() -> int:
    try:
        out = subprocess.run(
            ["journalctl", "-u", "penca-scheduler", "--since", "24h ago", "-p", "err", "--no-pager"],
            capture_output=True, text=True, check=True,
        ).stdout
        return sum(1 for line in out.splitlines() if "ERROR" in line)
    except Exception:
        return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        notif = TelegramNotifier(TelegramConfig.from_env())
    except Exception as e:
        log.error("Telegram no inicializado: %s", e)
        return 1

    scheduler_status, last_run = _scheduler_status()
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"

    status = {
        "now_uy": _now_uy_str(),
        "next_match": _next_match_summary(),
        "scheduler_status": scheduler_status,
        "last_scheduler_run": last_run,
        "api_penca_status": _check_api_penca(),
        "pinnacle_status": _check_pinnacle(),
        "anthropic_status": _check_anthropic(),
        "disk_free": _disk_free(),
        "ram_used": _ram_used(),
        "predictions_total": _predictions_count(),
        "predictions_24h": _predictions_count(within_hours=24),
        "errors_24h": _errors_24h(),
        "dry_run": dry_run,
    }

    notif.send_heartbeat(status)
    log.info("heartbeat OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

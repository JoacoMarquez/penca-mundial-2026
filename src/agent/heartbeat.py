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
from src.utils.usage_log import read_usage_summary

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


def _expected_pasadas_next_24h() -> int:
    """Cuántas pasadas de pipeline (T-24h, T-3h, T-30min) deberían dispararse en las próximas 24h."""
    fixtures = load_fixtures()
    now = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=24)
    count = 0
    for m in (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or []):
        if not m.get("kickoff_utc"):
            continue
        try:
            ko = datetime.fromisoformat(m["kickoff_utc"].replace("Z", "+00:00"))
        except Exception:
            continue
        for offset_min in (24 * 60, 3 * 60, 30):
            phase_t = ko - timedelta(minutes=offset_min)
            if now <= phase_t <= window_end:
                count += 1
    return count


def _next_scheduled_pasada() -> str:
    """Próximo momento en que el scheduler va a disparar la pipeline."""
    fixtures = load_fixtures()
    now = datetime.now(timezone.utc)
    phase_labels = {24 * 60: "T-24h", 3 * 60: "T-3h", 30: "T-30min"}
    next_events: list[tuple[datetime, str, dict]] = []
    for m in (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or []):
        if not m.get("kickoff_utc") or not (m.get("home") or m.get("home_name")):
            continue
        try:
            ko = datetime.fromisoformat(m["kickoff_utc"].replace("Z", "+00:00"))
        except Exception:
            continue
        for offset_min, label in phase_labels.items():
            phase_t = ko - timedelta(minutes=offset_min)
            if phase_t > now:
                next_events.append((phase_t, label, m))
    if not next_events:
        return "—"
    next_events.sort(key=lambda x: x[0])
    when, label, m = next_events[0]
    when_uy = when.astimezone(UY_TZ).strftime("%d/%m %H:%M")
    home = m.get("home_name") or m.get("home", "?")
    away = m.get("away_name") or m.get("away", "?")
    return f"{label} {home}-{away} · {when_uy} UY"


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


def _do_balance() -> dict[str, str]:
    """Consulta el balance del mes de DigitalOcean."""
    token = os.environ.get("DO_API_TOKEN", "")
    if not token:
        return {"month_to_date": "no token", "balance": "—"}
    try:
        with httpx.Client(timeout=5.0, headers={"Authorization": f"Bearer {token}"}) as c:
            r = c.get("https://api.digitalocean.com/v2/customers/my/balance")
        if r.status_code != 200:
            return {"month_to_date": f"err {r.status_code}", "balance": "—"}
        d = r.json()
        # month_to_date_usage = gasto del mes corriente (positivo)
        # account_balance = saldo (negativo si debés)
        mtd = float(d.get("month_to_date_usage", 0))
        bal = float(d.get("account_balance", 0))
        return {
            "month_to_date": f"${mtd:.2f}",
            "balance": f"${bal:.2f}",
        }
    except Exception as e:
        return {"month_to_date": f"err ({type(e).__name__})", "balance": "—"}


def _anthropic_spending() -> dict[str, str]:
    """Resumen de gasto Anthropic: 24h y total."""
    last_24h = read_usage_summary(within_hours=24)
    total = read_usage_summary()
    return {
        "spending_24h": f"${last_24h['cost_usd']:.4f}",
        "spending_total": f"${total['cost_usd']:.4f}",
        "calls_24h": last_24h["calls"],
        "calls_total": total["calls"],
    }


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
    do = _do_balance()
    anth = _anthropic_spending()

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
        "expected_pasadas_24h": _expected_pasadas_next_24h(),
        "next_pasada": _next_scheduled_pasada(),
        "errors_24h": _errors_24h(),
        "dry_run": dry_run,
        "do_mtd": do["month_to_date"],
        "do_balance": do["balance"],
        "anthropic_24h": anth["spending_24h"],
        "anthropic_total": anth["spending_total"],
        "anthropic_calls_24h": anth["calls_24h"],
        "anthropic_calls_total": anth["calls_total"],
    }

    # Pin del mensaje más reciente: despinear el anterior, pinear el nuevo
    pin_state_path = Path(os.environ.get("DATA_DIR", "data")) / "last_heartbeat_msg.txt"
    prev_msg_id: int | None = None
    if pin_state_path.exists():
        try:
            prev_msg_id = int(pin_state_path.read_text().strip())
        except Exception:
            prev_msg_id = None

    new_msg_id = notif.send_heartbeat(status)

    # Pinear el nuevo
    try:
        notif.pin_message(new_msg_id, disable_notification=True)
    except Exception as e:
        log.warning("no se pudo pinear: %s", e)

    # Despinear el anterior (si había)
    if prev_msg_id and prev_msg_id != new_msg_id:
        notif.unpin_message(prev_msg_id)

    # Guardar el message_id del nuevo
    try:
        pin_state_path.parent.mkdir(parents=True, exist_ok=True)
        pin_state_path.write_text(str(new_msg_id))
    except Exception as e:
        log.warning("no se pudo guardar last_heartbeat_msg: %s", e)

    log.info("heartbeat OK (msg_id=%d, pinned, prev_unpinned=%s)", new_msg_id, prev_msg_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

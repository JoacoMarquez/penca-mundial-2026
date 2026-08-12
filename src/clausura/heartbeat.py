"""Heartbeat diario del stack Clausura: un Telegram que confirma que TODO vive.

Cierra el último hueco de la auditoría del 5/8: OnFailure avisa cuando un unit
corre y FALLA, pero un timer que nunca dispara (deshabilitado tras un deploy,
VPS caído, reloj mal) es silencio total. Este mensaje diario convierte el
silencio en señal: si un día no llega, algo está mal.

Corre a las 11:00 UTC (08:00 UY), después de la pasada de picks de las 09:00:
también confirma que la planilla del día se generó.

Chequea:
  * timers del stack activos y con próxima ejecución programada
  * planilla más reciente: versión y edad
  * config del fixture: edad del último sync
  * dataset del Intermedio presente (sin él P(campeón) se invierte)
  * próximo cierre de pronóstico

Uso:
    python -m src.clausura.heartbeat            # manda por Telegram
    python -m src.clausura.heartbeat --dry-run  # imprime sin mandar
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# clausura-gate-watch ES parte de la lista: es el vigía cuya única razón de ser es
# cazar ventanas del gate — si su timer queda deshabilitado tras un deploy, un
# heartbeat que reporta "todo activo" sin contarlo es exactamente el silencio que
# este módulo existe para cerrar (una sola ventana cazada financió la v13).
TIMERS = ("clausura-picks", "clausura-rerun-cierre", "clausura-drift-audit",
          "clausura-carga-alert", "clausura-postmortem", "clausura-goleador-watch",
          "clausura-gate-watch", "clausura-heartbeat")

# Servicios de larga vida (no-timer) que también tienen que estar corriendo: si
# uvicorn muere con los Restart agotados, el modo carga desaparece justo el sábado.
SERVICIOS = ("clausura-dashboard",)
UY = timezone(timedelta(hours=-3))


def _timers_estado() -> tuple[list[str], list[str]]:
    """(ok, problemas) según systemctl list-timers."""
    try:
        out = subprocess.run(
            ["systemctl", "list-timers", "--all", "--no-pager", "--no-legend"],
            capture_output=True, text=True, timeout=15).stdout
    except Exception as e:  # entorno sin systemd (dev/test): no es un problema
        log.info("systemctl no disponible (%s)", e)
        return [], []
    ok, mal = [], []
    for t in TIMERS:
        linea = next((ln for ln in out.splitlines() if f"{t}.timer" in ln), None)
        if linea is None:
            mal.append(f"{t}: NO LISTADO (¿deshabilitado?)")
        elif linea.strip().startswith("-"):
            mal.append(f"{t}: sin próxima ejecución")
        else:
            ok.append(t)
    return ok, mal


def _servicios_estado() -> list[str]:
    """Problemas en los servicios de larga vida (lista vacía = todo bien)."""
    mal = []
    for s in SERVICIOS:
        try:
            r = subprocess.run(["systemctl", "is-active", "--quiet", f"{s}.service"],
                               timeout=15)
        except Exception as e:  # entorno sin systemd (dev/test): no es un problema
            log.info("systemctl no disponible (%s)", e)
            return []
        if r.returncode != 0:
            mal.append(f"{s}: servicio CAÍDO (systemctl restart {s})")
    return mal


def _edad_h(path: Path) -> float | None:
    if not path.exists():
        return None
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600


def construir_mensaje(now: datetime | None = None) -> str:
    from src.clausura.picks import PRED_DIR, CONFIG_PATH, fecha_dir, flat_eventos, load_config
    from src.clausura.dashboard_loader import fecha_actual
    from src.clausura.intermedio import OUT_PATH as INTERMEDIO_PATH
    from src.utils.versions import latest_version

    now = now or datetime.now(timezone.utc)
    lineas = [f"<b>💓 Clausura vivo</b> · {now.astimezone(UY).strftime('%a %d/%m %H:%M')} UY"]
    problemas: list[str] = []

    ok, mal = _timers_estado()
    if mal:
        problemas += mal
    elif ok:
        lineas.append(f"timers: {len(ok)}/{len(TIMERS)} activos")

    problemas += _servicios_estado()

    try:
        cfg = load_config()
        edad_cfg = _edad_h(CONFIG_PATH)
        if edad_cfg is not None and edad_cfg > 30:
            problemas.append(f"config sin sync hace {edad_cfg:.0f}h")

        f = fecha_actual(cfg)
        d = fecha_dir(f)
        latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
        if latest is None:
            problemas.append(f"Fecha {f}: SIN planilla generada")
        else:
            data = json.loads(latest.read_text(encoding="utf-8"))
            gen = datetime.fromisoformat(data["generado_utc"])
            lineas.append(f"planilla F{f}: {latest.name.split('_')[0]} de las "
                          f"{gen.astimezone(UY).strftime('%H:%M')} UY")

        proximos = [ev for ev in flat_eventos(cfg)
                    if datetime.fromisoformat(ev["cierre_pronostico_utc"]) > now]
        if proximos:
            ev = proximos[0]
            cierre = datetime.fromisoformat(ev["cierre_pronostico_utc"])
            horas = (cierre - now).total_seconds() / 3600
            lineas.append(f"próximo cierre: {ev['local']} vs {ev['visitante']} en "
                          f"{horas:.0f}h ({cierre.astimezone(UY).strftime('%a %H:%M')} UY)")
    except Exception as e:
        problemas.append(f"no pude leer config/planillas: {e}")

    if not INTERMEDIO_PATH.exists():
        problemas.append("intermedio_2026.json AUSENTE (P(campeón) se invierte)")

    if problemas:
        lineas.append("\n<b>⚠️ PROBLEMAS:</b>")
        lineas += [f"  • {p}" for p in problemas]
    else:
        lineas.append("sin problemas detectados")
    return "\n".join(lineas)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    msg = construir_mensaje()
    print(msg.replace("<b>", "").replace("</b>", ""))
    if not args.dry_run:
        from src.notifier.telegram import TelegramConfig, TelegramNotifier
        TelegramNotifier(TelegramConfig.from_env()).send(msg)


if __name__ == "__main__":
    main()

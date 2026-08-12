"""Alertas de carga: aviso por Telegram si falta subir picks cerca del cierre.

La carga en la web es manual (decisión operativa 2026-08-04) y el cierre es 15 min
antes de cada kickoff. Este módulo corre por timer cada hora y avisa en dos niveles
antes del cierre de cada partido (TIERS_H): a 6h como recordatorio y a 2h como
alarma. Cada aviso se manda UNA vez por (evento, nivel) — estado en disco.

Es un RECORDATORIO, no una verificación, y conviene tenerlo claro: el gate del
penca-api publica nuestros propios pronósticos recién cuando cierra CADA partido, o
sea después de que ya no se pueden corregir. Un aviso que llega antes del cierre —
que es el único útil — nunca puede saber si están cargados.

Se creyó lo contrario hasta el 2026-08-09 y salieron 14 avisos falsos de "faltan
cargar 12/12" con las 12 cargadas, en el mismo canal donde drift_audit manda lo que
sí cuesta puntos. Verificar de verdad exigiría autenticarse como el usuario, y eso
se descartó a propósito para no exponer la cuenta. Para chequear de verdad está el
Modo carga del dashboard, que se mira mientras se carga.

Por eso este módulo NO consulta el API (2026-08-12): quedaban 13 requests por tick
horario alimentando una rama que nunca se tomaba. Lo que sí cuenta faltantes es
drift_audit, post-cierre, donde el dato existe.

Uso:
    python -m src.clausura.carga_alert            # chequea y avisa si corresponde
    python -m src.clausura.carga_alert --dry-run  # imprime sin mandar ni marcar estado
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.clausura.api import TZ_UY
from src.clausura.rivals import mis_numeros_env

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "state" / "carga_alerts.json"

# Horas antes del cierre en que se avisa: recordatorio y alarma.
TIERS_H = (6.0, 2.0)


# -------------------- lógica pura (testeable) --------------------

def eventos_por_cerrar(
    eventos: list[dict],
    now: datetime,
    horizon_h: float = max(TIERS_H),
) -> list[tuple[dict, datetime]]:
    """Eventos cuyo cierre está adelante pero dentro del horizonte de aviso."""
    out = []
    for ev in eventos:
        cierre = datetime.fromisoformat(ev["cierre_pronostico_utc"])
        if now < cierre <= now + timedelta(hours=horizon_h):
            out.append((ev, cierre))
    out.sort(key=lambda t: t[1])
    return out


def tier_activo(cierre: datetime, now: datetime) -> float | None:
    """El nivel de aviso que corresponde ahora: el tier MÁS chico que ya se alcanzó."""
    horas = (cierre - now).total_seconds() / 3600
    alcanzados = [t for t in TIERS_H if horas <= t]
    return min(alcanzados) if alcanzados else None


def pendientes_de_alerta(
    eventos: list[dict],
    now: datetime,
    ya_avisados: set[str],
) -> list[tuple[dict, datetime, float, str]]:
    """[(evento, cierre, tier, clave_estado)] que requieren aviso ahora."""
    out = []
    for ev, cierre in eventos_por_cerrar(eventos, now):
        tier = tier_activo(cierre, now)
        if tier is None:
            continue
        # El cierre va EN la clave: si el admin re-programa un partido (pasó con
        # Torque-Peñarol de la F1), la clave vieja ya está quemada y con
        # `evento_id:tier` a secas el makeup no recibiría NINGÚN recordatorio —
        # justo el partido que nadie tiene en la cabeza.
        clave = f"{ev['evento_id']}:{cierre:%Y%m%dT%H%M}:{tier:g}"
        if clave not in ya_avisados:
            out.append((ev, cierre, tier, clave))
    return out


def formatear_alerta(
    ev: dict,
    cierre: datetime,
    tier: float,
    faltantes: list[int] | None,
    n_participaciones: int,
) -> str | None:
    """Texto del aviso. None si está todo cargado (no hay nada que avisar).

    `faltantes` = números de participación sin pick para este evento; None = no se
    pudo verificar. Desde el timer SIEMPRE llega None (el gate hace imposible
    verificar antes del cierre); la rama con faltantes queda para quien tenga el
    dato de verdad — hoy drift_audit, post-cierre.
    """
    cierre_uy = cierre.astimezone(TZ_UY).strftime("%H:%M")
    icono = "🚨" if tier <= min(TIERS_H) else "⏰"
    partido = f"{ev['local']} vs {ev['visitante']}"
    pref = " ⭐x2" if ev.get("preferencial") else ""

    if faltantes is None:
        # No es "todavía no puedo": es que NO SE PUEDE. El gate del API publica
        # nuestros picks recién cuando cierra el partido, o sea después de que ya no
        # sirve. El aviso es un recordatorio, y tiene que decirlo sin adornos.
        return (f"{icono} <b>{partido}</b>{pref} cierra a las {cierre_uy} UY.\n"
                f"Verificá a mano que estén las {n_participaciones} participaciones "
                f"— el API no publica los picks propios hasta el cierre, así que "
                f"nadie puede chequearlo por vos antes.")
    if not faltantes:
        return None
    nums = ", ".join(str(n) for n in sorted(faltantes))
    return (f"{icono} <b>{partido}</b>{pref} cierra a las {cierre_uy} UY y faltan "
            f"cargar <b>{len(faltantes)}/{n_participaciones}</b> participaciones:\n{nums}")


# -------------------- estado --------------------

def load_state(now: datetime | None = None) -> set[str]:
    """Claves ya avisadas. Poda las de cierres viejos (>7 días) y las del formato
    anterior sin cierre (`evento_id:tier`), que quedan obsoletas solas."""
    if not STATE_PATH.exists():
        return set()
    now = now or datetime.now(timezone.utc)
    vivas = set()
    for k in json.loads(STATE_PATH.read_text(encoding="utf-8")):
        partes = k.split(":")
        if len(partes) != 3:
            continue
        try:
            cierre = datetime.strptime(partes[1], "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if cierre > now - timedelta(days=7):
            vivas.add(k)
    return vivas


def save_state(avisados: set[str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(sorted(avisados)), encoding="utf-8")


# -------------------- main --------------------

def run(dry_run: bool = False, now: datetime | None = None) -> list[str]:
    from src.clausura.picks import flat_eventos, load_config

    now = now or datetime.now(timezone.utc)
    cfg = load_config()
    eventos = flat_eventos(cfg)
    mis_numeros = mis_numeros_env()

    avisados = load_state(now)
    pendientes = pendientes_de_alerta(eventos, now, avisados)
    if not pendientes:
        log.info("sin cierres dentro de %.0fh (o ya avisados)", max(TIERS_H))
        return []

    # NO se consulta el API. El gate publica nuestros picks recién al cierre de CADA
    # partido, y `pendientes` solo trae cierres FUTUROS: la verificación era imposible
    # por construcción. El código igual la intentaba —13 requests por tick horario— y
    # el resultado moría en una rama que nunca se tomaba (`cierre > now` siempre era
    # verdadero). Se sacó el 2026-08-12: menos tráfico contra el 429 y una promesa
    # menos en el docstring. Para verificar de verdad está el Modo carga del
    # dashboard (src.clausura.verificar_carga), post-cierre, y drift_audit.
    mensajes = []
    for ev, cierre, tier, clave in pendientes:
        msg = formatear_alerta(ev, cierre, tier, None, max(len(mis_numeros), 1))
        if msg:
            mensajes.append(msg)
        avisados.add(clave)   # todo-cargado también se marca: no re-chequear este tier

    if mensajes and not dry_run:
        from src.notifier.telegram import TelegramConfig, TelegramNotifier
        TelegramNotifier(TelegramConfig.from_env()).send(
            "<b>📋 Carga de picks — Penca Clausura</b>\n\n" + "\n\n".join(mensajes)
        )
    if not dry_run:
        save_state(avisados)
    for m in mensajes:
        print(m.replace("<b>", "").replace("</b>", ""))
    return mensajes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="imprime los avisos sin mandar Telegram ni marcar estado")
    args = ap.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

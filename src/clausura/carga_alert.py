"""Alertas de carga: aviso por Telegram si falta subir picks cerca del cierre.

La carga en la web es manual (decisión operativa 2026-08-04) y el cierre es 15 min
antes de cada kickoff. Este módulo corre por timer cada hora y avisa en dos niveles
antes del cierre de cada partido (TIERS_H): a 6h como recordatorio y a 2h como
alarma. Cada aviso se manda UNA vez por (evento, nivel) — estado en disco.

La verificación es real, no un recordatorio ciego: post-inicio del campeonato los
pronósticos propios son públicos (mismo endpoint que usa pool_snapshot), así que el
aviso dice exactamente cuántas de nuestras participaciones están sin cargar y cuáles.
Pre-inicio (gate del API cerrado, solo afecta la Fecha 1) cae a recordatorio ciego.

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

import httpx

from src.clausura.api import BASE, HEADERS, TZ_UY, PencaApiClient
from src.clausura.rivals import mis_numeros_env

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "state" / "carga_alerts.json"

# Horas antes del cierre en que se avisa: recordatorio y alarma.
TIERS_H = (6.0, 2.0)
GATE_MSG = "campeonato ya inicio"


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
        clave = f"{ev['evento_id']}:{tier:g}"
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
    pudo verificar (pre-inicio del campeonato).
    """
    cierre_uy = cierre.astimezone(TZ_UY).strftime("%H:%M")
    icono = "🚨" if tier <= min(TIERS_H) else "⏰"
    partido = f"{ev['local']} vs {ev['visitante']}"
    pref = " ⭐x2" if ev.get("preferencial") else ""

    if faltantes is None:
        return (f"{icono} <b>{partido}</b>{pref} cierra a las {cierre_uy} UY.\n"
                f"No puedo verificar la carga hasta que inicie el campeonato — "
                f"chequeá que estén las {n_participaciones} participaciones.")
    if not faltantes:
        return None
    nums = ", ".join(str(n) for n in sorted(faltantes))
    return (f"{icono} <b>{partido}</b>{pref} cierra a las {cierre_uy} UY y faltan "
            f"cargar <b>{len(faltantes)}/{n_participaciones}</b> participaciones:\n{nums}")


# -------------------- estado --------------------

def load_state() -> set[str]:
    if STATE_PATH.exists():
        return set(json.loads(STATE_PATH.read_text(encoding="utf-8")))
    return set()


def save_state(avisados: set[str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(sorted(avisados)), encoding="utf-8")


# -------------------- verificación contra el API --------------------

def cargados_por_participacion(
    penca_id: int,
    mis_numeros: set[int],
) -> dict[int, set[int]] | None:
    """numero_participacion → evento_ids con pick cargado. None si el gate del API
    sigue cerrado (pre-inicio) y no se puede verificar."""
    with PencaApiClient() as api:
        ranking = api.ranking(penca_id)
    mios = [r for r in ranking if r.numero_participacion in mis_numeros]
    if not mios:
        log.warning("ninguna de mis participaciones (%s) está en el ranking", mis_numeros)
        return {}
    out: dict[int, set[int]] = {}
    with httpx.Client(base_url=BASE, timeout=20.0, headers=HEADERS) as c:
        for r in mios:
            resp = c.get(f"/front/pencas/{r.participacion_id}/pronosticosEventos")
            if resp.status_code == 400 and GATE_MSG in resp.text:
                return None
            ids: set[int] = set()
            if resp.status_code == 200:
                for p in resp.json().get("data", []):
                    if (p.get("golesEquipoLocal") is not None
                            and p.get("encuentroId") is not None):
                        ids.add(int(p["encuentroId"]))
            else:
                log.warning("pronosticosEventos %d → %d", r.participacion_id, resp.status_code)
            out[r.numero_participacion] = ids
    return out


# -------------------- main --------------------

def run(dry_run: bool = False, now: datetime | None = None) -> list[str]:
    from src.clausura.picks import flat_eventos, load_config

    now = now or datetime.now(timezone.utc)
    cfg = load_config()
    eventos = flat_eventos(cfg)
    mis_numeros = mis_numeros_env()

    avisados = load_state()
    pendientes = pendientes_de_alerta(eventos, now, avisados)
    if not pendientes:
        log.info("sin cierres dentro de %.0fh (o ya avisados)", max(TIERS_H))
        return []

    cargados = None
    try:
        cargados = cargados_por_participacion(cfg["pencas"]["paga"]["id"], mis_numeros)
    except Exception as e:
        log.warning("no pude verificar la carga (%s) — aviso ciego", e)

    mensajes = []
    for ev, cierre, tier, clave in pendientes:
        if cargados is None:
            faltantes = None
        else:
            faltantes = [n for n, ids in cargados.items() if ev["evento_id"] not in ids]
        msg = formatear_alerta(ev, cierre, tier, faltantes, max(len(mis_numeros), 1))
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

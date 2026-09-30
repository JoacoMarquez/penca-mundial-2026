"""Recordatorios de carga de la penca LUB: Telegram a 6h y 2h de cada cierre.

Misma lógica que el del Clausura (src.clausura.carga_alert, de donde salen los niveles
y el estado por (evento, cierre, nivel)) y la misma limitación: es un RECORDATORIO,
no una verificación — el gate del API publica nuestros picks recién al cierre de cada
partido. Diferencia: la LUB cierra de a varios partidos a la misma hora, así que el
aviso agrupa por horario de cierre en vez de mandar uno por partido.

El fixture sale de data/lub/temporadas.json; si el último refresco propio tiene más
de REFRESCO_H se re-baja solo la temporada actual (un partido re-programado no puede
quedar sin recordatorio). El refresco se marca en data/state/ y no con el mtime del
archivo: safe_pull descarta los cambios locales de ese archivo (está trackeado) y lo
deja con mtime nuevo y contenido viejo.

Uso:
    python -m src.lub.carga_alert            # chequea y avisa si corresponde
    python -m src.lub.carga_alert --dry-run  # imprime sin mandar ni marcar estado
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.clausura.carga_alert import TIERS_H, pendientes_de_alerta
from src.lub.data import (CAMPEONATO_ID, DATA_DIR, TZ_UY, fetch_temporadas, load_temporadas,
                          mis_numeros_env, save_temporadas)
from src.lub.picks import TEMPORADA

log = logging.getLogger(__name__)

STATE_PATH = Path("data/state/lub_carga_alerts.json")
REFRESCO_PATH = Path("data/state/lub_fixture_refresco")
REFRESCO_H = 6.0


def eventos_lub(now: datetime) -> list[dict]:
    path = DATA_DIR / "temporadas.json"
    ultimo = REFRESCO_PATH.stat().st_mtime if REFRESCO_PATH.exists() else 0.0
    if not path.exists() or time.time() - ultimo > REFRESCO_H * 3600:
        try:
            otras = [p for p in load_temporadas()
                     if p.campeonato_id != CAMPEONATO_ID] if path.exists() else []
            save_temporadas(otras + fetch_temporadas([CAMPEONATO_ID]))
            REFRESCO_PATH.parent.mkdir(parents=True, exist_ok=True)
            REFRESCO_PATH.touch()
        except Exception as e:           # sin API: avisar con el fixture que haya
            log.warning("no pude refrescar el fixture LUB: %s", e)
    return [{"evento_id": p.evento_id, "local": p.local, "visitante": p.visitante,
             "preferencial": p.preferencial, "fecha": p.fecha_nombre,
             "cierre_pronostico_utc": p.cierre_utc}
            for p in load_temporadas() if p.temporada == TEMPORADA and p.pts_local is None]


def formatear(grupo: list[tuple[dict, datetime, float]], n_part: int) -> str:
    """Un aviso por horario de cierre (todos los partidos que cierran juntos)."""
    cierre, tier = grupo[0][1], min(t for _, _, t in grupo)
    icono = "🚨" if tier <= min(TIERS_H) else "⏰"
    fechas = sorted({ev["fecha"] for ev, _, _ in grupo})
    partidos = "\n".join(f"  • {html.escape(ev['local'])} vs {html.escape(ev['visitante'])}"
                         f"{' ⭐x2' if ev.get('preferencial') else ''}" for ev, _, _ in grupo)
    return (f"{icono} <b>{len(grupo)} partido(s) de la LUB cierran a las "
            f"{cierre.astimezone(TZ_UY):%H:%M} UY</b> ({', '.join(fechas)}):\n{partidos}\n"
            f"Cargá las {n_part} participaciones con la planilla de hoy. El API no publica "
            f"los picks propios hasta el cierre, así que nadie puede verificarlo antes.")


def load_state(now: datetime) -> set[str]:
    if not STATE_PATH.exists():
        return set()
    vivas = set()
    for k in json.loads(STATE_PATH.read_text()):
        try:
            cierre = datetime.strptime(k.split(":")[1], "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
        except (IndexError, ValueError):
            continue
        if cierre > now - timedelta(days=7):
            vivas.add(k)
    return vivas


def save_state(avisados: set[str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(sorted(avisados)))


def run(dry_run: bool = False, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    avisados = load_state(now)
    pendientes = pendientes_de_alerta(eventos_lub(now), now, avisados)
    if not pendientes:
        log.info("sin cierres LUB dentro de %.0fh (o ya avisados)", max(TIERS_H))
        return []
    grupos: dict[datetime, list] = defaultdict(list)
    for ev, cierre, tier, clave in pendientes:
        grupos[cierre].append((ev, cierre, tier))
        avisados.add(clave)
    n_part = len(mis_numeros_env()) or "(K)"
    mensajes = [formatear(g, n_part) for _, g in sorted(grupos.items())]
    if not dry_run:
        from src.notifier.telegram import TelegramConfig, TelegramNotifier
        TelegramNotifier(TelegramConfig.from_env()).send(
            "<b>📋 Carga de picks — Penca LUB</b>\n\n" + "\n\n".join(mensajes))
        save_state(avisados)
    for m in mensajes:
        print(m.replace("<b>", "").replace("</b>", ""))
    return mensajes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    run(dry_run=ap.parse_args().dry_run)


if __name__ == "__main__":
    main()

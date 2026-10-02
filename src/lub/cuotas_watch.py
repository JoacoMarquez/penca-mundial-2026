"""Vigía de cuotas de la LUB en Supermatch: avisa cuando aparecen y cuando NO matchean.

La planilla usa 70% mercado / 30% ratings cuando hay cuota. El parseo de los mercados
de básquet se verificó con otras ligas (29/9), pero la LUB todavía no estaba publicada,
así que dos cosas pueden fallar en silencio el día que aparezca:

  * los nombres de Supermatch no matchean con los del penca-api → el partido sale
    "(sin cuota)" y la planilla corre solo con ratings sin que nadie lo note;
  * un mercado de CAMPEÓN (outright) publicado antes del arranque, que es la mejor
    referencia externa para el especial de 25 pts y el modelo no lo mira.

Avisa UNA vez: primeras cuotas matcheadas, cada evento uruguayo con fecha de la LUB
que no matchea (con los nombres, para agregar el alias) y el outright de campeón.

Uso:
    python -m src.lub.cuotas_watch             # chequea y avisa
    python -m src.lub.cuotas_watch --dry-run   # sin Telegram ni estado
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.lub import odds as lub_odds
from src.lub.data import TZ_UY, load_temporadas
from src.lub.picks import TEMPORADA

log = logging.getLogger(__name__)

STATE_PATH = Path("data/state/lub_cuotas_watch.json")
VENTANA_MATCH_H = 6.0


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def clasificar(lineas: list[lub_odds.LineasPartido], fixture: list) -> tuple[list, list]:
    """(matcheados [(partido, línea)], sueltos [línea]).

    Suelto = evento uruguayo que cae en una ventana de ±6h de algún partido de la LUB
    pero no matchea con ninguno: casi seguro un problema de nombres. Los que no caen
    cerca de ningún partido son de otra liga uruguaya y se ignoran."""
    matcheados, sueltos = [], []
    for lp in lineas:
        cerca = [p for p in fixture if abs(lp.inicio_ms - _ms(p.inicio_utc)) <= VENTANA_MATCH_H * 3.6e6]
        if not cerca:
            continue
        p = next((p for p in cerca if lub_odds.match_fixture([lp], p.local, p.visitante,
                                                             _ms(p.inicio_utc), VENTANA_MATCH_H)), None)
        if p is None:
            sueltos.append(lp)
        else:
            matcheados.append((p, lp))
    return matcheados, sueltos


def run(dry_run: bool = False, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    fixture = [p for p in load_temporadas() if p.temporada == TEMPORADA and p.pts_local is None
               and datetime.fromisoformat(p.inicio_utc) <= now + timedelta(days=lub_odds.LOOKAHEAD_DIAS)]
    lineas = lub_odds.fetch_lub()
    matcheados, sueltos = clasificar(lineas, fixture)
    log.info("cuotas LUB: %d eventos uruguayos, %d matcheados, %d sueltos", len(lineas), len(matcheados), len(sueltos))
    mensajes = []

    if matcheados and not state.get("primeras"):
        sin_mu = [p for p, lp in matcheados if lp.mu_mercado(13.0) is None]
        L = [f"<b>💹 Supermatch publicó cuotas de la LUB</b>: {len(matcheados)} partido(s) matchean "
             "con el fixture. La planilla pasa a 70% mercado / 30% ratings."]
        for p, lp in matcheados[:8]:
            mu = lp.mu_mercado(13.0)
            L.append(f"  • {html.escape(p.local)} vs {html.escape(p.visitante)}: "
                     + (f"μ mercado {mu:+.1f}" if mu is not None else "sin ganador/hándicap parseable"))
        if sin_mu:
            L.append(f"⚠️ {len(sin_mu)} sin mercado parseable: revisar los betLines.")
        mensajes.append("\n".join(L))
        state["primeras"] = True

    avisados = set(state.get("sueltos", []))
    nuevos = [lp for lp in sueltos if f"{lp.local}|{lp.visitante}" not in avisados]
    if nuevos:
        L = ["<b>⚠️ Cuotas de la LUB que NO matchean con el fixture</b> — esos partidos salen "
             "\"sin cuota\" en la planilla. Nombres en Supermatch:"]
        for lp in nuevos:
            hora = datetime.fromtimestamp(lp.inicio_ms / 1000, timezone.utc).astimezone(TZ_UY)
            L.append(f"  • {html.escape(lp.local)} vs {html.escape(lp.visitante)} ({hora:%d/%m %H:%M})")
            avisados.add(f"{lp.local}|{lp.visitante}")
        L.append("Hay que agregar el alias en src/lub/odds.py (mismo_equipo).")
        mensajes.append("\n".join(L))
        state["sueltos"] = sorted(avisados)

    if not state.get("outright"):
        for h in lub_odds.fetch_uruguay_hits(outrights=True):
            probs = lub_odds.parse_outright(h)
            if probs:
                nombre = h["_source"].get("description", "?")
                top = " · ".join(f"{html.escape(k)} {v:.0%}" for k, v in list(probs.items())[:6])
                mensajes.append(f"<b>🏆 Mercado de campeón en Supermatch</b> ({html.escape(nombre)}): {top}\n"
                                "Referencia externa para el especial de campeón — comparalo con el "
                                "P(campeón) de la planilla.")
                state["outright"] = True
                break

    if mensajes and not dry_run:
        from src.notifier.telegram import TelegramConfig, TelegramNotifier
        TelegramNotifier(TelegramConfig.from_env()).send("\n\n".join(mensajes))
    if not dry_run:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state))
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

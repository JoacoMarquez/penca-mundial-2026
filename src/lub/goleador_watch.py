"""Vigía del menú de especiales de la penca LUB (campeón y goleador uruguayo, 25 pts c/u).

Los menús se piden con el id del CAMPEONATO (45), no de la penca (48): con el de la
penca el API da 500 siempre, y hasta el 9/10 eso se leyó como "menú sin publicar"
(estaba publicado). Los especiales se cargan hasta el primer partido de la temporada. Este módulo corre por timer cada hora y avisa
UNA vez por menú cuando aparece; después de arrancada la temporada sale en el acto.

El goleador no se puede optimizar sin P(goleador) por candidato, y el penca-api no
tiene estadísticas de jugadores: el prior se arma a mano sobre el menú real en
data/lub/goleador_prior.json ({nombre: prob}). El aviso guarda el menú en
data/lub/goleador_opciones.json y dice si falta el prior; con el prior puesto, la
próxima planilla asigna goleador por participación.

Uso:
    python -m src.lub.goleador_watch             # chequea y avisa si aparecieron
    python -m src.lub.goleador_watch --dry-run   # sin Telegram ni estado
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.clausura.api import BASE, HEADERS
from src.lub.data import CAMPEONATO_ID, DATA_DIR, TZ_UY, fetch_opciones_goleador, load_temporadas
from src.lub.picks import GOLEADOR_PRIOR, TEMPORADA

log = logging.getLogger(__name__)

STATE_PATH = Path("data/state/lub_especiales_menu.json")
OPCIONES_PATH = DATA_DIR / "goleador_opciones.json"


def fetch_opciones_campeon(campeonato_id: int = CAMPEONATO_ID) -> list[str] | None:
    # id del CAMPEONATO, no de la penca: ver fetch_opciones_goleador.
    with httpx.Client(base_url=BASE, headers=HEADERS, timeout=20.0) as c:
        r = c.get(f"/front/pencas/{campeonato_id}/opcionesEquiposCampeon")
    if r.status_code != 200:
        return None
    data = r.json().get("opcionesEquiposCampeon", {}).get("data", [])
    return [o.get("nombre", "?") for o in data] or None


def primer_partido() -> datetime | None:
    ps = [datetime.fromisoformat(p.inicio_utc) for p in load_temporadas() if p.temporada == TEMPORADA]
    return min(ps) if ps else None


def formatear_goleador(opciones: list[str], hay_prior: bool, cierre: datetime | None) -> str:
    L = [f"<b>🏀 Apareció el menú de GOLEADOR de la LUB</b> ({len(opciones)} candidatos, 25 pts)"]
    L.append(", ".join(html.escape(o) for o in opciones[:40]) + (" …" if len(opciones) > 40 else ""))
    if cierre:
        L.append(f"Se carga hasta el primer partido: {cierre.astimezone(TZ_UY):%d/%m %H:%M} UY.")
    if hay_prior:
        L.append("El prior ya está: la próxima planilla asigna goleador por participación.")
    else:
        L.append("⚠️ Falta P(goleador) por candidato (data/lub/goleador_prior.json): pedíselo a "
                 "Claude con este menú. Sin eso la planilla no asigna goleador. Referencia: en "
                 "la 25/26 ganó Vescovi y lo eligió el 28% de los que cargaron.")
    return "\n".join(L)


def run(dry_run: bool = False, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    inicio = primer_partido()
    if inicio and now >= inicio:
        log.info("la temporada ya arrancó (%s): los especiales están cerrados", inicio)
        return []
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    mensajes = []
    if not state.get("goleador"):
        opciones = fetch_opciones_goleador()
        if opciones:
            if not dry_run:
                OPCIONES_PATH.parent.mkdir(parents=True, exist_ok=True)
                OPCIONES_PATH.write_text(json.dumps(opciones, ensure_ascii=False, indent=1))
            mensajes.append(formatear_goleador(opciones, GOLEADOR_PRIOR.exists(), inicio))
            state["goleador"] = True
    if not state.get("campeon"):
        equipos = fetch_opciones_campeon()
        if equipos:
            mensajes.append(f"<b>🏆 Apareció el menú de CAMPEÓN de la LUB</b> ({len(equipos)} equipos): "
                            "ya se pueden cargar los especiales. El campeón por participación "
                            "sale en la planilla.")
            state["campeon"] = True
    log.info("menús LUB: goleador %s · campeón %s", state.get("goleador", False), state.get("campeon", False))
    if mensajes and not dry_run:
        from src.notifier.telegram import TelegramConfig, TelegramNotifier
        TelegramNotifier(TelegramConfig.from_env()).send("\n\n".join(mensajes))
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

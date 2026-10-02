"""Vigía de pencas nuevas en Supermatch: avisa por Telegram cuando aparece una.

Relevamiento del 2026-10-02: fuera de Supermatch no hay pencas con plata jugables
desde Uruguay (monopolio estatal; la DNLQ cerró las de los medios en el Mundial).
Las de Supermatch tienen overlay estructural (garantía fija contra un pool chico:
Apertura 2026 recaudó ~$344k y garantizaba ≥$500k), pero ese overlay se decide por
el N final, y comprar tarde o enterarse tarde de los especiales cuesta puntos. Este
vigía corre por timer y avisa UNA vez por cada id nuevo de `/front/pencas/visibles`,
con lo necesario para decidir: precio, premios, participaciones al momento, fechas
del campeonato, garantía mínima y N de break-even (garantía / precio).

La primera lectura exitosa solo registra las pencas visibles, sin avisar.
Los ids vistos nunca se borran: una penca que sale y vuelve no se re-anuncia.
Si el API falla FALLOS_PARA_AVISAR corridas seguidas se avisa una vez (un vigía que
muere callado es peor que no tenerlo).

Uso:
    python -m src.clausura.pencas_watch             # chequea y avisa si hay nuevas
    python -m src.clausura.pencas_watch --dry-run   # sin Telegram ni estado
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from pathlib import Path

from src.clausura.api import Penca, PencaApiClient

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "state" / "pencas_vistas.json"

# 2 corridas por día → dos días seguidos sin poder leer el API
FALLOS_PARA_AVISAR = 4


def load_state() -> dict | None:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return None


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


# -------------------- lógica pura (testeable) --------------------

def nuevas(visibles: list[Penca], vistos: set[int]) -> list[Penca]:
    return [p for p in visibles if p.id not in vistos]


def garantia_minima(penca: Penca, n_fechas: int | None) -> float | None:
    """Premios en plata garantizados: campeón + fecha × n_fechas + grupo amigo.

    None si hay premio por fecha y no se sabe cuántas fechas hay (el fixture
    a veces se publica después que la penca)."""
    total = 0.0
    for pr in penca.premios:
        if pr.tipo == "FECHA":
            if pr.monto and n_fechas is None:
                return None
            total += pr.monto * (n_fechas or 0)
        else:
            total += pr.monto
    return total


def formatear_nueva(penca: Penca, n_partic: int | None, n_fechas: int | None) -> str:
    gratis = penca.precio <= 0
    titulo = "🆓 Penca GRATUITA nueva" if gratis else "🆕 Penca PAGA nueva"
    lines = [f"<b>{titulo} en Supermatch: {html.escape(penca.nombre)}</b> (id {penca.id})"]
    if not gratis:
        lines.append(f"Precio: ${penca.precio:,.0f} por participación")
    for pr in penca.premios:
        monto = f"${pr.monto:,.0f}" if pr.monto else html.escape(pr.descripcion)
        lines.append(f"  • {html.escape(pr.nombre)}: {monto}")
    lines.append(f"Fechas publicadas: {n_fechas if n_fechas is not None else '¿?'}"
                 f" · participaciones hoy: {n_partic if n_partic is not None else '¿?'}")

    g = garantia_minima(penca, n_fechas)
    if not gratis and g:
        break_even = g / penca.precio
        lines.append(f"Garantía mínima con esas fechas ≈ ${g:,.0f} → paga más de lo que recauda "
                     f"mientras haya menos de <b>{break_even:,.0f}</b> participaciones")
        if n_partic:
            lines.append(f"Hoy: recaudado ${n_partic * penca.precio:,.0f}, "
                         f"payout {g / (n_partic * penca.precio):.0%} (sube el N hasta el cierre)")
    elif not gratis:
        lines.append("<i>Garantía no calculable todavía (falta el fixture o premios sin monto).</i>")
    else:
        lines.append("<i>Gratis = EV positivo; suele ser una participación por cuenta.</i>")
    lines.append("Reglamento: /front/pencas/{}/reglamento — revisar puntos, especiales "
                 "y cierre antes de decidir.".format(penca.id))
    return "\n".join(lines)


def formatear_falla(n: int, error: str) -> str:
    return (f"<b>⚠️ Vigía de pencas: {n} corridas seguidas sin poder leer el API</b>\n"
            f"Último error: {html.escape(error[:300])}\n"
            "Mientras dure, una penca nueva no se detecta.")


# -------------------- main --------------------

def _detalle(api: PencaApiClient, penca: Penca) -> tuple[int | None, int | None]:
    """(participaciones, fechas). Best effort: el aviso sale igual sin ellos."""
    n_partic = n_fechas = None
    try:
        r = api._get_429(f"/front/pencas/{penca.id}/ranking", {"page": 1, "size": 1})
        r.raise_for_status()
        n_partic = int(r.json().get("totalElements", 0))
    except Exception as e:                                       # noqa: BLE001
        log.warning("ranking de la penca %d no disponible (%s)", penca.id, e)
    try:
        data = api._get(f"/front/campeonatos/{penca.campeonato_id}/fechas")
        n_fechas = len(data) or None
    except Exception as e:                                       # noqa: BLE001
        log.warning("fechas del campeonato %d no disponibles (%s)", penca.campeonato_id, e)
    return n_partic, n_fechas


def _enviar(texto: str) -> None:
    from src.notifier.telegram import TelegramConfig, TelegramNotifier
    TelegramNotifier(TelegramConfig.from_env()).send(texto)


def run(dry_run: bool = False) -> list[str]:
    state = load_state() or {"inicializado": False, "vistos": [],
                             "fallos": 0, "falla_avisada": False}
    primera = not state["inicializado"]

    with PencaApiClient() as api:
        try:
            visibles = api.pencas_visibles()
        except Exception as e:                                   # noqa: BLE001
            state["fallos"] += 1
            log.error("no pude leer /front/pencas/visibles (%d seguidas): %s",
                      state["fallos"], e)
            mensajes = []
            if state["fallos"] >= FALLOS_PARA_AVISAR and not state["falla_avisada"]:
                mensajes.append(formatear_falla(state["fallos"], str(e)))
                state["falla_avisada"] = True
            if not dry_run:
                for m in mensajes:
                    _enviar(m)
                save_state(state)
            return mensajes

        state["fallos"] = 0
        state["falla_avisada"] = False
        vistos = set(state["vistos"])
        log.info("visibles: %s", ", ".join(f"{p.id} {p.nombre}" for p in visibles))

        if primera:
            log.info("primera corrida: registro %d pencas sin avisar", len(visibles))
            mensajes = []
        else:
            mensajes = [formatear_nueva(p, *_detalle(api, p))
                        for p in nuevas(visibles, vistos)]

    state["vistos"] = sorted(vistos | {p.id for p in visibles})
    state["inicializado"] = True
    for m in mensajes:
        print(m.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    if not dry_run:
        for m in mensajes:
            _enviar(m)
        save_state(state)
    return mensajes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="sin Telegram ni estado")
    args = ap.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

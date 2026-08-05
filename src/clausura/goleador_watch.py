"""Watcher de los menús de Especiales (Campeón y Goleador, 25 pts c/u).

Al 2026-08-05 los dos endpoints de opciones devuelven 500 (el admin aún no los
configuró), así que los especiales NO se pueden cargar en la web todavía — y valen
50 puntos por participación. Este watcher corre por timer cada hora y, cuando un
menú aparece (500 → 200 con data), avisa por Telegram UNA vez con instrucciones
accionables por participación:

  - **Campeón**: la planilla ya lo asigna por participación (se modela sin menú);
    el aviso lista qué campeón cargar en cada número.
  - **Goleador**: no se pudo asignar hasta ahora porque no había opciones. El
    watcher REGENERA la planilla (pipeline completo, una vez) para que el
    optimizador asigne goleador por participación con el menú real, y el aviso
    lista qué cargar en cada número + el prior por si querés sanity-checkear.

Urgencia: los especiales se cargan hasta el inicio del campeonato (vie 2026-08-07
19:00 UY) — si el menú aparece el viernes, no llega a la corrida del sábado.

Uso:
    python -m src.clausura.goleador_watch             # chequea y avisa si aparecieron
    python -m src.clausura.goleador_watch --dry-run   # sin Telegram / estado / regeneración
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.clausura.especiales import fetch_opciones, goleador_prior_from_ratings

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "state" / "especiales_menu.json"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"campeon": False, "goleador": False}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


# -------------------- formateo (puro, testeable) --------------------

def formatear_campeon(
    n_opciones: int,
    por_numero: dict[int, tuple[str | None, str | None]],
) -> str:
    lines = [f"<b>🏆 Apareció el menú de CAMPEÓN</b> ({n_opciones} opciones — 25 pts)",
             "Cargá en la pestaña Especial de cada participación:"]
    por_campeon: dict[str, list[int]] = {}
    for numero, (campeon, _) in sorted(por_numero.items()):
        if campeon:
            por_campeon.setdefault(campeon, []).append(numero)
    for campeon, numeros in por_campeon.items():
        nums = ", ".join(str(n) for n in numeros)
        lines.append(f"  <b>{campeon}</b> → {nums}")
    if not por_campeon:
        lines.append("  <i>(sin planilla con campeones asignados — corré el pipeline)</i>")
    return "\n".join(lines)


def formatear_goleador(
    por_numero: dict[int, tuple[str | None, str | None]],
    top_prior: list[tuple[str, float]],
    regenerada: bool,
) -> str:
    lines = ["<b>⚽ Apareció el menú de GOLEADOR</b> (25 pts)"]
    if top_prior:
        probs = " · ".join(f"{n} {p:.0%}" for n, p in top_prior)
        lines.append(f"<i>Prior del modelo: {probs}</i>")
    asignados: dict[str, list[int]] = {}
    for numero, (_, goleador) in sorted(por_numero.items()):
        if goleador:
            asignados.setdefault(goleador, []).append(numero)
    if asignados:
        lines.append("Cargá en la pestaña Especial de cada participación:")
        for goleador, numeros in asignados.items():
            nums = ", ".join(str(n) for n in numeros)
            lines.append(f"  <b>{goleador}</b> → {nums}")
    elif regenerada:
        lines.append("<i>La planilla se regeneró pero no asignó goleadores — revisar logs.</i>")
    else:
        lines.append("<i>No pude regenerar la planilla — corré el pipeline a mano para "
                     "asignar goleador por participación.</i>")
    lines.append("\n⚠️ Los especiales se cargan hasta el inicio del campeonato.")
    return "\n".join(lines)


# -------------------- main --------------------

def run(dry_run: bool = False) -> list[str]:
    from src.clausura.drift_audit import especiales_esperados
    from src.clausura.picks import ensure_ratings, load_config
    from src.clausura.rivals import mis_numeros_env

    cfg = load_config()
    penca_id = cfg["pencas"]["paga"]["id"]
    state = load_state()
    if state["campeon"] and state["goleador"]:
        log.info("ambos menús ya avisados — nada que hacer")
        return []

    opciones_campeon, opciones_goleador = fetch_opciones(penca_id)
    log.info("menús: campeón %s · goleador %s",
             "SÍ" if opciones_campeon else "no",
             "SÍ" if opciones_goleador else "no")

    mis_numeros = sorted(mis_numeros_env())
    mensajes: list[str] = []

    # goleador primero: puede regenerar la planilla, y el aviso de campeón
    # debe leer la versión post-regeneración
    if opciones_goleador and not state["goleador"]:
        # regenerar la planilla para que el optimizador asigne goleador con el menú real
        regenerada = False
        if not dry_run:
            try:
                import src.clausura.picks as picks
                target = picks.resolve_fecha("auto")
                picks.run(target, n_participaciones=max(len(mis_numeros), 1),
                          telegram=False)
                regenerada = True
            except Exception as e:
                log.error("regeneración de planilla falló (%s) — aviso igual", e)

        top_prior: list[tuple[str, float]] = []
        try:
            equipos_cfg: dict[int, str] = cfg["equipos"]
            equipo_nombres = [equipos_cfg[k] for k in sorted(equipos_cfg)]
            equipo_id_por_nombre = {v: k for k, v in equipos_cfg.items()}
            p = goleador_prior_from_ratings(
                opciones_goleador, equipo_nombres, equipo_id_por_nombre,
                ensure_ratings().ataque)
            orden = sorted(range(len(p)), key=lambda i: -p[i])[:4]
            top_prior = [(opciones_goleador[i].nombre, float(p[i])) for i in orden]
        except Exception as e:
            log.warning("prior de goleador no computable (%s)", e)

        por_numero = especiales_esperados(mis_numeros)
        mensajes.append(formatear_goleador(por_numero, top_prior, regenerada))
        state["goleador"] = True

    if opciones_campeon and not state["campeon"]:
        por_numero = especiales_esperados(mis_numeros)
        mensajes.append(formatear_campeon(len(opciones_campeon), por_numero))
        state["campeon"] = True

    if mensajes:
        texto = "\n\n".join(mensajes)
        print(texto.replace("<b>", "").replace("</b>", "")
                   .replace("<i>", "").replace("</i>", ""))
        if not dry_run:
            from src.notifier.telegram import TelegramConfig, TelegramNotifier
            TelegramNotifier(TelegramConfig.from_env()).send(texto)
            save_state(state)
    return mensajes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="sin Telegram, sin estado y sin regenerar planilla")
    args = ap.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

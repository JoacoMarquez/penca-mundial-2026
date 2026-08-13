"""Genera config/clausura2026.yaml desde el penca-api público.

Uso:
    python -m src.clausura.sync            # escribe config/clausura2026.yaml
    python -m src.clausura.sync --stdout   # imprime sin escribir

El YAML es la fuente de verdad local de ids (penca, campeonato, fechas, eventos,
equipos) para el resto del pipeline: el scheduler lee los cierres de pronóstico,
el publisher los ids de evento, el matcher de odds los nombres de equipos.

Re-correr es idempotente y seguro: la API puede reprogramar partidos (Art. 14:
suspensiones se posponen), así que conviene refrescar antes de cada fecha.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.clausura.api import PencaApiClient

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "clausura2026.yaml"

CAMPEONATO_NOMBRE = "Torneo Clausura 2026"


def build_config() -> dict:
    with PencaApiClient() as api:
        pencas = [p for p in api.pencas_visibles() if p.nombre.startswith("Torneo Clausura")]
        if not pencas:
            raise RuntimeError("no hay pencas del Clausura visibles en la API")
        paga = next((p for p in pencas if p.precio > 0), None)
        gratuita = next((p for p in pencas if p.precio == 0), None)
        camp_id = pencas[0].campeonato_id

        fechas = api.fechas(camp_id)

        fechas_cfg = {}
        equipos: dict[int, str] = {}
        for nombre, fid in fechas.items():
            eventos = api.eventos(fid)
            fechas_cfg[nombre] = {
                "fecha_id": fid,
                "eventos": [
                    {
                        "evento_id": ev.id,
                        "local": ev.local,
                        "visitante": ev.visitante,
                        "local_id": ev.local_id,
                        "visitante_id": ev.visitante_id,
                        "inicio_utc": ev.inicio_utc.isoformat(),
                        "cierre_pronostico_utc": ev.cierre_pronostico_utc.isoformat(),
                        "estadio": ev.estadio,
                        "preferencial": ev.preferencial,
                        "estado": ev.estado,
                    }
                    for ev in eventos
                ],
            }
            for ev in eventos:
                equipos[ev.local_id] = ev.local
                equipos[ev.visitante_id] = ev.visitante

    return {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "campeonato": {"id": camp_id, "nombre": CAMPEONATO_NOMBRE},
        "pencas": {
            "paga": {"id": paga.id, "precio": paga.precio} if paga else None,
            "gratuita": {"id": gratuita.id} if gratuita else None,
        },
        "premios": [
            {"nombre": pr.nombre, "tipo": pr.tipo, "monto": pr.monto}
            for pr in (paga.premios if paga else [])
        ],
        "equipos": {tid: nombre for tid, nombre in sorted(equipos.items())},
        "fechas": fechas_cfg,
    }


def preferenciales_cambiados(cfg: dict, previo: dict | None) -> list[str]:
    """Eventos conocidos cuyo flag `preferencial` cambió entre configs.

    El ×2 del partido estrella entra al optimizador vía el config: si el admin
    recategoriza la estrella DESPUÉS de cargada la planilla, los picks de esa
    fecha quedaron optimizados con el ×2 en el partido equivocado — y sin este
    aviso, el pisado del YAML era silencioso.
    """
    if previo is None:
        return []
    prev_flags = {e["evento_id"]: (bool(e.get("preferencial")), e)
                  for f in previo.get("fechas", {}).values() for e in f.get("eventos", [])}
    cambios = []
    for nombre, f in cfg.get("fechas", {}).items():
        for e in f.get("eventos", []):
            conocido = prev_flags.get(e["evento_id"])
            if conocido and conocido[0] != bool(e.get("preferencial")):
                estrella = "ahora ⭐x2" if e.get("preferencial") else "PERDIÓ la ⭐x2"
                cambios.append(f"{nombre}: {e['local']} vs {e['visitante']} {estrella}")
    return cambios


def check_contra_anterior(cfg: dict, previo: dict | None) -> None:
    """Rechaza un config que PIERDE eventos ya conocidos, salvo casos legítimos.

    El piso "n_eventos > 0" dejaba pasar un API en mantenimiento que devolviera
    una sola fecha: el YAML mutilado apagaba carga_alert/rerun/gate_watch para
    todos los cierres futuros sin distinguirse de "no hay partidos". Un evento
    puede desaparecer legítimamente (se anula y re-crea con otro id al
    reprogramar), pero perder MUCHOS a la vez es un glitch, no el fixture.
    """
    if previo is None:
        return
    ids_previos = {e["evento_id"] for f in previo.get("fechas", {}).values()
                   for e in f.get("eventos", [])}
    ids_nuevos = {e["evento_id"] for f in cfg.get("fechas", {}).values()
                  for e in f.get("eventos", [])}
    perdidos = ids_previos - ids_nuevos
    if len(perdidos) > max(2, len(ids_previos) // 10):
        raise RuntimeError(
            f"config sospechoso del API: perdería {len(perdidos)} de "
            f"{len(ids_previos)} eventos conocidos ({sorted(perdidos)[:8]}…) — "
            f"NO se pisa {CONFIG_PATH.name}. Si el cambio es real, --force")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="imprimir sin escribir el archivo")
    ap.add_argument("--force", action="store_true",
                    help="pisar el config aunque pierda eventos conocidos")
    args = ap.parse_args()

    cfg = build_config()
    text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=120)

    if args.stdout:
        sys.stdout.write(text)
        return

    # Piso de sanidad: este YAML es la fuente de verdad de TODO lo que agenda por
    # cierres (carga_alert, rerun, gate_watch). Un glitch del API (mantenimiento,
    # fecha con eventos: []) no puede mutilarlo — mejor quedarse con el config
    # anterior y que el error suene, que apagar los avisos sin distinguirlo de
    # "no hay partidos".
    n_eventos = sum(len(f["eventos"]) for f in cfg["fechas"].values())
    if not cfg.get("fechas") or n_eventos == 0:
        raise RuntimeError(
            f"config sospechoso del API ({len(cfg.get('fechas', {}))} fechas, "
            f"{n_eventos} eventos) — NO se pisa {CONFIG_PATH.name}")
    if CONFIG_PATH.exists():
        previo = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        if not args.force:
            check_contra_anterior(cfg, previo)
        cambios = preferenciales_cambiados(cfg, previo)
        if cambios:
            msg = ("⭐ <b>Cambió el partido preferencial</b> — si la fecha ya está "
                   "cargada, los picks se optimizaron con el ×2 en el partido "
                   "equivocado:\n" + "\n".join(f"  · {c}" for c in cambios))
            print(msg.replace("<b>", "").replace("</b>", ""), file=sys.stderr)
            try:
                from src.notifier.telegram import TelegramConfig, TelegramNotifier
                TelegramNotifier(TelegramConfig.from_env()).send(msg)
            except Exception:                                  # noqa: BLE001
                pass    # sin Telegram (dev local): el stderr ya lo dice

    # Escritura atómica: gate_watch (10 min), carga_alert (horario) y el dashboard
    # leen este archivo concurrentemente; un write_text a medias les sirve YAML
    # truncado. rename() en el mismo directorio es atómico en POSIX.
    tmp = CONFIG_PATH.with_suffix(".yaml.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(CONFIG_PATH)
    print(f"escrito {CONFIG_PATH} — {len(cfg['fechas'])} fechas, {n_eventos} eventos, "
          f"{len(cfg['equipos'])} equipos")


if __name__ == "__main__":
    main()

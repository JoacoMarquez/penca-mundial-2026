"""Vigía del gate del penca-api: caza ventanas donde se ve lo que no debería.

El 2026-08-07 (mañana del kickoff) el gate de especiales quedó abierto ~20 minutos
pre-lock; ese escaneo casual financió la re-optimización de campeones de la v13.
Este módulo institucionaliza esa suerte: sondea barato en cada tick del timer
(1 página de ranking + 3-4 requests) y, cuando detecta una TRANSICIÓN
cerrado→abierto, captura el snapshot completo del pool y avisa por Telegram para
decidir si re-correr picks mientras dure la ventana.

Qué cuenta como "abierto de más":
  - marcadores de eventos ABIERTOS (cierre futuro) visibles en pronosticosEventos —
    last-mover advantage directo, en cualquier momento del campeonato;
  - especiales visibles ANTES del primer kickoff del campeonato (la ventana del 7/8).

Alerta solo en la transición (estado en disco): si la visibilidad resulta ser el
comportamiento permanente post-inicio, avisa UNA vez —esa primera alerta responde
la incógnita del gate del backlog— y después calla mientras siga igual. Cada
alerta captura el snapshot versionado en data/pool_snapshots/clausura/, que el
pipeline y el rerun T-2h levantan solos (load_latest_snapshot, 48h).

Los errores de red salen 0 con warning: un tick perdido cada tanto no amerita
OnFailure, la próxima pasada es en 10 minutos.

Uso:
    python -m src.clausura.gate_watch            # tick del timer
    python -m src.clausura.gate_watch --dry-run  # sin Telegram, estado ni snapshot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.clausura.api import BASE, HEADERS, TZ_UY, parse_ranking_page
from src.clausura.pool_snapshot import fetch_snapshot, save_snapshot, snapshot_summary
from src.clausura.rivals import mis_numeros_env

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "state" / "gate_watch.json"
FALLAS_PATH = ROOT / "data" / "state" / "gate_watch_fallas.json"

# Rivales sondeados por tick: suficiente para no depender de UNO que no cargó nada.
PROBE_N = 3

# Ticks consecutivos fallados antes de escalar UNA vez por OnFailure. Con el timer
# de 10 min, 12 = ~2h roto. Sin este colchón, un cambio de shape del API (o un 200
# con HTML de mantenimiento) crashea 144 veces por día en el mismo canal donde
# viven las señales reales — fatiga de alarma sobre el canal único.
FALLAS_PARA_ESCALAR = 12


# -------------------- lógica pura (testeable) --------------------

def clasificar(
    visibles_ids: set[int],
    especiales_visibles: bool,
    eventos: list[dict],
    now: datetime,
) -> dict:
    """Estado de visibilidad del gate según lo sondeado.

    - `abiertos`: evento_ids con picks rivales visibles cuyo cierre NO pasó.
    - `especiales_pre_inicio`: especiales visibles antes del primer kickoff.
    """
    cierres = {ev["evento_id"]: datetime.fromisoformat(ev["cierre_pronostico_utc"])
               for ev in eventos}
    abiertos = sorted(eid for eid in visibles_ids
                      if eid in cierres and cierres[eid] > now)
    inicio = min(datetime.fromisoformat(ev["inicio_utc"]) for ev in eventos)
    return {
        "abiertos": abiertos,
        "especiales_pre_inicio": bool(especiales_visibles and now < inicio),
    }


def necesita_alerta(prev: dict, cur: dict) -> bool:
    """Solo la transición cerrado→abierto dispara: sin re-alertas mientras persista."""
    abrio_marcadores = bool(cur["abiertos"]) and not prev.get("abiertos", False)
    abrio_especiales = cur["especiales_pre_inicio"] and not prev.get("especiales", False)
    return abrio_marcadores or abrio_especiales


def formatear_alerta(cur: dict, eventos: list[dict], resumen_snapshot: str | None) -> str:
    por_id = {ev["evento_id"]: ev for ev in eventos}
    partes = ["🕳️ <b>Gate del penca-api ABIERTO</b> — ventana de visibilidad detectada"]
    if cur["abiertos"]:
        lineas = []
        for eid in cur["abiertos"][:10]:
            ev = por_id[eid]
            cierre_uy = (datetime.fromisoformat(ev["cierre_pronostico_utc"])
                         .astimezone(TZ_UY).strftime("%a %d %H:%M"))
            lineas.append(f"  · {ev['local']} vs {ev['visitante']} (cierra {cierre_uy} UY)")
        extra = f"\n  … y {len(cur['abiertos']) - 10} más" if len(cur["abiertos"]) > 10 else ""
        partes.append("Se ven picks rivales de <b>partidos ABIERTOS</b>:\n"
                      + "\n".join(lineas) + extra)
    if cur["especiales_pre_inicio"]:
        partes.append("Se ven <b>especiales</b> (campeón/goleador) antes del kickoff.")
    if resumen_snapshot:
        partes.append(f"📸 Snapshot capturado: {resumen_snapshot}")
    else:
        partes.append("⚠️ No pude capturar el snapshot completo (¿la ventana se cerró?).")
    partes.append("Si la ventana sigue abierta: <code>rerun_cierre --force</code> "
                  "re-optimiza con estos datos.")
    return "\n\n".join(partes)


# -------------------- estado --------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(cur: dict, now: datetime) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({
        "abiertos": bool(cur["abiertos"]),
        "especiales": cur["especiales_pre_inicio"],
        "ts": now.isoformat(),
    }), encoding="utf-8")


# -------------------- sondeo --------------------

def sondear(penca_id: int, mis_numeros: set[int]) -> tuple[set[int], bool]:
    """(evento_ids con picks visibles en la muestra, especiales visibles).

    Barato a propósito: 1 página de ranking + PROBE_N pronosticosEventos +
    1 pronosticoCampeonGoleador.
    """
    with httpx.Client(base_url=BASE, timeout=20.0, headers=HEADERS) as c:
        r = c.get(f"/front/pencas/{penca_id}/ranking", params={"page": 1})
        r.raise_for_status()
        rows, _, _ = parse_ranking_page(r.json())
        rivales = [row for row in rows if row.numero_participacion not in mis_numeros]

        visibles: set[int] = set()
        for row in rivales[:PROBE_N]:
            resp = c.get(f"/front/pencas/{row.participacion_id}/pronosticosEventos")
            if resp.status_code == 200:
                for p in resp.json().get("data", []):
                    if (p.get("golesEquipoLocal") is not None
                            and p.get("encuentroId") is not None):
                        visibles.add(int(p["encuentroId"]))

        especiales = False
        if rivales:
            resp = c.get(f"/front/pencas/{rivales[0].participacion_id}"
                         "/pronosticoCampeonGoleador")
            if resp.status_code == 200:
                d = resp.json()
                especiales = bool((d.get("equipoCampeon") or {}).get("nombre")
                                  or (d.get("opcionGoleador") or {}).get("goleador"))
    return visibles, especiales


# -------------------- main --------------------

def run(dry_run: bool = False, now: datetime | None = None) -> str | None:
    from src.clausura.picks import flat_eventos, load_config

    now = now or datetime.now(timezone.utc)
    cfg = load_config()
    eventos = flat_eventos(cfg)
    penca_id = cfg["pencas"]["paga"]["id"]
    mis_numeros = mis_numeros_env()

    visibles, especiales = sondear(penca_id, mis_numeros)
    cur = clasificar(visibles, especiales, eventos, now)
    prev = load_state()
    log.info("gate: %d eventos abiertos visibles · especiales_pre_inicio=%s",
             len(cur["abiertos"]), cur["especiales_pre_inicio"])

    msg = None
    if necesita_alerta(prev, cur):
        resumen = None
        if not dry_run:
            try:
                rivales = fetch_snapshot(penca_id, strict_gate=False)
                path = save_snapshot(rivales, penca_id)
                snap = json.loads(path.read_text(encoding="utf-8"))
                resumen = snapshot_summary(snap)
                log.info("snapshot de la ventana guardado en %s", path)
            except Exception as e:
                log.warning("no pude capturar el snapshot completo (%s)", e)
        msg = formatear_alerta(cur, eventos, resumen)
        if not dry_run:
            from src.notifier.telegram import TelegramConfig, TelegramNotifier
            TelegramNotifier(TelegramConfig.from_env()).send(msg)
        print(msg.replace("<b>", "").replace("</b>", "").replace("<code>", "")
                 .replace("</code>", ""))
    if not dry_run:
        save_state(cur, now)
    return msg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="sondear e imprimir sin Telegram, estado ni snapshot")
    args = ap.parse_args()
    try:
        run(dry_run=args.dry_run)
    except Exception as e:
        # Cualquier fallo de un tick (red, API 5xx, cambio de shape, HTML de
        # mantenimiento con 200) se traga con warning: el próximo tick es en 10 min.
        # Pero la rotura PERSISTENTE sí escala — una vez, no 144 veces por día.
        fallas = _contar_falla()
        log.warning("tick del vigía falló (%s) — %d consecutivas", e, fallas)
        if fallas >= FALLAS_PARA_ESCALAR:
            _reset_fallas()     # escalar una vez y volver a contar desde cero
            print(f"ERROR: el vigía lleva {fallas} ticks fallando: {e}", file=sys.stderr)
            sys.exit(1)         # OnFailure → Telegram
        sys.exit(0)
    else:
        _reset_fallas()


def _contar_falla() -> int:
    try:
        n = json.loads(FALLAS_PATH.read_text(encoding="utf-8")).get("consecutivas", 0)
    except Exception:
        n = 0
    n += 1
    FALLAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FALLAS_PATH.write_text(json.dumps({"consecutivas": n}), encoding="utf-8")
    return n


def _reset_fallas() -> None:
    try:
        FALLAS_PATH.unlink(missing_ok=True)
    except OSError:
        pass


if __name__ == "__main__":
    main()

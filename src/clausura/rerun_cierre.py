"""Corrida extra cerca del cierre — el T-3h del Mundial, versión Clausura.

La planilla del día se genera a las 12:00 UTC (09:00 UY). Entre esa corrida y los
cierres (17:45-22:45 UTC) los inputs se mueven: odds de Supermatch, snapshot del
pool (rivales que cargaron a la tarde), resultados de partidos ya jugados de la
fecha. Este módulo re-corre el pipeline UNA vez por día de partidos, ~2h antes del
primer cierre del día, y notifica por Telegram SOLO si algún pick de un partido
todavía abierto cambió respecto de la planilla vigente.

El optimizador es determinístico (seeds fijos), así que un diff refleja cambios
reales de inputs, no ruido de simulación. La corrida versiona una planilla nueva
(v+1, regla de trabajo #2); si no hay cambios, no manda nada.

Uso:
    python -m src.clausura.rerun_cierre                    # corre si toca (timer)
    python -m src.clausura.rerun_cierre --force            # ignora ventana y estado
    python -m src.clausura.rerun_cierre --dry-run          # sin Telegram ni estado
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.clausura.api import TZ_UY

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "state" / "rerun_cierre.json"

# Disparo: el primer cierre del día está a menos de este umbral.
TRIGGER_H = 2.5


# -------------------- lógica de ventana (pura, testeable) --------------------

def primer_cierre_del_dia(eventos: list[dict], now: datetime) -> datetime | None:
    """El cierre abierto más próximo DE HOY (fecha UTC). None si hoy no hay."""
    cierres = []
    for ev in eventos:
        cierre = datetime.fromisoformat(ev["cierre_pronostico_utc"])
        if cierre > now and cierre.date() == now.date():
            cierres.append(cierre)
    return min(cierres) if cierres else None


def debe_correr(eventos: list[dict], now: datetime, ya_corridos: set[str]) -> bool:
    if now.date().isoformat() in ya_corridos:
        return False
    cierre = primer_cierre_del_dia(eventos, now)
    if cierre is None:
        return False
    return (cierre - now).total_seconds() / 3600 <= TRIGGER_H


# -------------------- diff de planillas (puro, testeable) --------------------

def diff_planillas(
    prev: dict,
    nuevo: dict,
    now: datetime,
) -> list[tuple[dict, list[tuple[int, tuple[int, int], tuple[int, int]]]]]:
    """Cambios en partidos aún abiertos: [(fila_evento_nuevo, [(col, viejo, nuevo)])].

    Solo compara eventos presentes en ambas planillas y con cierre futuro — lo
    cerrado ya no es accionable y lo congelado no puede cambiar.
    """
    prev_by_eid = {int(r["evento_id"]): r for r in prev.get("picks", [])}
    out = []
    for row in nuevo.get("picks", []):
        eid = int(row["evento_id"])
        prow = prev_by_eid.get(eid)
        if prow is None:
            continue
        if datetime.fromisoformat(row["cierre_pronostico_utc"]) <= now:
            continue
        cambios = []
        for k, (old, new) in enumerate(zip(prow.get("scores", []), row.get("scores", []))):
            if list(old) != list(new):
                cambios.append((k, (int(old[0]), int(old[1])), (int(new[0]), int(new[1]))))
        if cambios:
            out.append((row, cambios))
    return out


def formatear_diff(
    cambios: list[tuple[dict, list[tuple[int, tuple[int, int], tuple[int, int]]]]],
    mis_numeros: list[int],
    now: datetime,
) -> str:
    """Aviso compacto: solo lo que cambió, con hora de cierre en UY."""
    n_picks = sum(len(c) for _, c in cambios)
    lines = ["<b>🔄 Corrida T-2h — cambios vs la planilla de la mañana</b>",
             f"{n_picks} pick(s) cambiaron en {len(cambios)} partido(s) aún abiertos:"]
    for row, cs in cambios:
        cierre = datetime.fromisoformat(row["cierre_pronostico_utc"])
        cierre_uy = cierre.astimezone(TZ_UY)
        hoy = " HOY" if cierre.date() == now.date() else cierre_uy.strftime(" %a").lower()
        pref = " ⭐x2" if row.get("preferencial") else ""
        lines.append(f"\n<b>{row['partido']}</b>{pref} · cierra {cierre_uy:%H:%M}{hoy}")
        for k, old, new in cs:
            numero = mis_numeros[k] if k < len(mis_numeros) else f"col{k + 1}"
            lines.append(f"  {numero}: {old[0]}-{old[1]} → <b>{new[0]}-{new[1]}</b>")
    lines.append("\nSi ya cargaste la planilla de la mañana, actualizá SOLO estos picks "
                 "en la web. La planilla nueva quedó versionada.")
    return "\n".join(lines)


# -------------------- estado --------------------

def load_state() -> set[str]:
    if STATE_PATH.exists():
        return set(json.loads(STATE_PATH.read_text(encoding="utf-8")))
    return set()


def save_state(corridos: set[str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(sorted(corridos)), encoding="utf-8")


# -------------------- main --------------------

def run(
    n_participaciones: int = 12,
    n_sims: int = 800,
    force: bool = False,
    dry_run: bool = False,
    now: datetime | None = None,
) -> list | None:
    import src.clausura.picks as picks
    from src.clausura.rivals import mis_numeros_env
    from src.utils.versions import latest_version

    now = now or datetime.now(timezone.utc)
    cfg = picks.load_config()
    eventos = picks.flat_eventos(cfg)
    estado = load_state()

    if not force and not debe_correr(eventos, now, estado):
        log.info("fuera de ventana (primer cierre de hoy a >%sh, sin cierres hoy, "
                 "o ya corrido) — no toca", TRIGGER_H)
        return None

    target_fecha = picks.resolve_fecha("auto")
    d = picks.fecha_dir(target_fecha)
    prev_path = latest_version(d.glob("v*_*.json")) if d.exists() else None
    if prev_path is None:
        log.warning("no hay planilla previa de la fecha %d — corré primero el pipeline "
                    "de la mañana; no hay contra qué diffear", target_fecha)
        return None
    prev = json.loads(prev_path.read_text(encoding="utf-8"))

    log.info("re-corriendo pipeline de la fecha %d (planilla previa: %s)",
             target_fecha, prev_path.name)
    new_path = picks.run(target_fecha, n_participaciones=n_participaciones,
                         telegram=False, n_sims=n_sims)
    nuevo = json.loads(new_path.read_text(encoding="utf-8"))

    cambios = diff_planillas(prev, nuevo, now)
    mis_numeros = sorted(mis_numeros_env())

    if cambios:
        msg = formatear_diff(cambios, mis_numeros, now)
        print(msg.replace("<b>", "").replace("</b>", ""))
        if not dry_run:
            from src.notifier.telegram import TelegramConfig, TelegramNotifier
            TelegramNotifier(TelegramConfig.from_env()).send(msg)
    else:
        log.info("sin cambios en partidos abiertos — no se notifica (planilla %s igual "
                 "quedó versionada)", new_path.name)

    if not dry_run:
        save_state(estado | {now.date().isoformat()})
    return cambios


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--participaciones", type=int, default=12)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--force", action="store_true",
                    help="correr aunque no toque (ignora ventana y estado)")
    ap.add_argument("--dry-run", action="store_true",
                    help="sin Telegram ni marca de estado")
    args = ap.parse_args()
    run(n_participaciones=args.participaciones, n_sims=args.sims,
        force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

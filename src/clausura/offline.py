"""Modo offline: generar la planilla sin el penca-api ni el ES de cuotas.

Desde el 2026-10-05 ~16:50 UTC Cloudflare le devuelve 403 a todo cliente que no
sea un navegador en todo *.supermatch.com.uy (home, penca-api, ES de cuotas), desde
el VPS y desde la Mac. No se evade: el sistema corre con lo que ya tenía guardado
más lo que el usuario copia a mano de la web.

Fuentes, en orden:
  * resultados   ← data/postmortems/clausura/fecha_NN.json + `resultados` del archivo
  * cuotas       ← último cache de data/odds/clausura (sin tope de edad) pisado
                   por `cuotas` del archivo, partido por partido
  * pool         ← último snapshot (sin tope de edad)
  * puntos       ← `ranking` del archivo (numero → puntos, típicamente el top);
                   el resto de los rivales queda con los del snapshot y el
                   simulador les imputa los partidos jugados después
                   (RivalModel.puntos_del_snapshot)

Archivo: data/state/clausura_offline.yaml (fuera de git: lo edita el operador en
el VPS). Plantilla en config/clausura_offline.example.yaml.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.clausura.odds import EventOdds, _norm

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OFFLINE_PATH = ROOT / "data" / "state" / "clausura_offline.yaml"
PM_DIR = ROOT / "data" / "postmortems" / "clausura"
PROBE_URL = "https://penca.supermatch.com.uy/penca-api/v1/front/pencas/visibles"

# Timers que dependen de Supermatch y se pausan en offline (deploy/offline_timers.sh
# lee esta lista; el heartbeat no los reporta como caídos mientras dure).
TIMERS_PAUSADOS = (
    "clausura-picks", "clausura-rerun-cierre", "clausura-drift-audit",
    "clausura-gate-watch", "clausura-postmortem", "clausura-cold-check",
    "clausura-pencas-watch", "clausura-sharp-compare", "lub-cuotas-watch",
    "lub-aviso-compra", "valuebet-scan", "valuebet-close",
)


def load_offline(path: Path = OFFLINE_PATH) -> dict:
    """Contenido del archivo offline ({} si no existe)."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def offline_activo(path: Path = OFFLINE_PATH) -> bool:
    return bool(load_offline(path).get("activo"))


def resultados_offline(off: dict, pm_dir: Path = PM_DIR) -> dict[int, tuple[int, int]]:
    """Resultados de los postmortems guardados + los cargados a mano (ganan)."""
    out: dict[int, tuple[int, int]] = {}
    for f in sorted(pm_dir.glob("fecha_*.json")) if pm_dir.exists() else []:
        data = json.loads(f.read_text(encoding="utf-8"))
        for eid, (gl, gv) in (data.get("resultados") or {}).items():
            out[int(eid)] = (int(gl), int(gv))
    for eid, (gl, gv) in (off.get("resultados") or {}).items():
        out[int(eid)] = (int(gl), int(gv))
    return out


def ranking_manual(off: dict) -> dict[int, int]:
    """numero de participación → puntos, copiado de la web."""
    return {int(k): int(v) for k, v in (off.get("ranking") or {}).items()}


def cuotas_manuales(off: dict) -> list[EventOdds]:
    """Cuotas copiadas de la web: 1X2 obligatorio, total 2.5 opcional."""
    ahora = datetime.now(timezone.utc).isoformat()
    out = []
    for c in off.get("cuotas") or []:
        ev = EventOdds(
            event_id=f"manual:{c['local']}-{c['visitante']}",
            home=c["local"], away=c["visitante"],
            start_utc=str(c.get("inicio_utc", "")), fetched_utc=str(c.get("copiado", ahora)),
            x1x2={"home": float(c["1"]), "draw": float(c["X"]), "away": float(c["2"])},
        )
        if c.get("over_2_5") and c.get("under_2_5"):
            ev.totals["2.5"] = {"over": float(c["over_2_5"]), "under": float(c["under_2_5"])}
        out.append(ev)
    return out


def merge_cuotas(cache: list[EventOdds], manuales: list[EventOdds]) -> list[EventOdds]:
    """El cache, con cada partido cargado a mano reemplazando al cacheado."""
    claves = {(_norm(e.home), _norm(e.away)) for e in manuales}
    return [e for e in cache if (_norm(e.home), _norm(e.away)) not in claves] + manuales


def probe_api(timeout: float = 15.0) -> int | None:
    """Código HTTP del penca-api con el cliente normal (None si no hay red)."""
    import httpx
    try:
        return httpx.get(PROBE_URL, timeout=timeout).status_code
    except Exception as e:                                       # noqa: BLE001
        log.warning("probe del penca-api sin respuesta: %s", e)
        return None

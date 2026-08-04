"""Loaders del dashboard para la página del Clausura.

Fuentes:
    - config/clausura2026.yaml               → fixture, ids, premios
    - data/predictions/clausura/fecha_NN/    → planillas versionadas del pipeline
    - penca-api público                      → ranking en vivo + resultados (cache TTL)

Para resaltar tus participaciones en el ranking: CLAUSURA_MIS_PARTICIPACIONES en el
.env con los números de participación separados por coma (se ven en la web, columna
"numeroParticipacion" del ranking; ej: 899258510,899258511).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.clausura.picks import CONFIG_PATH, PRED_DIR, fecha_dir
from src.utils.versions import latest_version

log = logging.getLogger(__name__)

RANKING_TTL = 120.0     # segundos de cache para no golpear el penca-api en cada request
CONFIG_TTL = 300.0


def _cached(key: str, ttl: float, loader):
    from src.dashboard.data_loader import _cached as base_cached
    return base_cached(f"clausura:{key}", ttl, loader)


def _load_config() -> dict | None:
    import yaml
    if not CONFIG_PATH.exists():
        return None
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_clausura_config() -> dict | None:
    return _cached("config", CONFIG_TTL, _load_config)


_DIAS = {"Mon": "lun", "Tue": "mar", "Wed": "mié", "Thu": "jue",
         "Fri": "vie", "Sat": "sáb", "Sun": "dom"}


def _to_uy(iso: str) -> str:
    from src.clausura.api import TZ_UY
    try:
        dt = datetime.fromisoformat(iso).astimezone(TZ_UY)
        return f"{_DIAS[dt.strftime('%a')]} {dt.strftime('%d/%m %H:%M')}"
    except Exception:
        return iso


def fecha_actual(cfg: dict) -> int:
    """Primera fecha con algún partido todavía por jugarse (por hora de inicio)."""
    now = datetime.now(timezone.utc)
    for nombre in sorted(cfg["fechas"], key=lambda n: int(n.split()[-1])):
        for ev in cfg["fechas"][nombre]["eventos"]:
            if datetime.fromisoformat(ev["inicio_utc"]) > now:
                return int(nombre.split()[-1])
    return 15


def load_planilla(fecha_n: int) -> dict | None:
    """Última versión guardada de la planilla de esa fecha, enriquecida para la UI."""
    d = fecha_dir(fecha_n)
    latest = latest_version(d.glob("v*_*.json")) if d.exists() else None
    if latest is None:
        return None
    data = json.loads(latest.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    for row in data.get("picks", []):
        cierre = datetime.fromisoformat(row["cierre_pronostico_utc"])
        row["cierre_uy"] = _to_uy(row["cierre_pronostico_utc"])
        row["cerrado"] = cierre <= now
        row["scores_fmt"] = [f"{a}-{b}" for a, b in row["scores"]]
    data["version_file"] = latest.name
    data["generado_uy"] = _to_uy(data.get("generado_utc", ""))
    data["n_fechas_guardadas"] = sum(
        1 for f in range(1, 16)
        if (PRED_DIR / f"fecha_{f:02d}").exists()
        and latest_version((PRED_DIR / f"fecha_{f:02d}").glob("v*_*.json"))
    )
    return data


def fechas_guardadas() -> list[int]:
    out = []
    for f in range(1, 16):
        d = PRED_DIR / f"fecha_{f:02d}"
        if d.exists() and latest_version(d.glob("v*_*.json")):
            out.append(f)
    return out


def _fetch_ranking(penca_id: int) -> dict:
    from src.clausura.api import PencaApiClient
    try:
        with PencaApiClient(timeout=10.0) as api:
            rows = api.ranking(penca_id)
    except Exception as e:
        log.warning("ranking no disponible: %s", e)
        return {"ok": False, "error": str(e), "rows": [], "total": None}

    mios_raw = os.environ.get("CLAUSURA_MIS_PARTICIPACIONES", "")
    mios = {int(x) for x in mios_raw.split(",") if x.strip().isdigit()}

    rows.sort(key=lambda r: (-r.puntos_totales, -r.cant_resultados_exactos))
    out_rows = []
    for pos, r in enumerate(rows, start=1):
        es_mia = r.numero_participacion in mios
        if pos <= 15 or es_mia:
            out_rows.append({
                "pos": pos,
                "numero": r.numero_participacion,
                "puntos": r.puntos_totales,
                "puntos_fecha": r.puntos_por_fecha,
                "exactos": r.cant_resultados_exactos,
                "mia": es_mia,
            })
    exactos_pool = [r.cant_resultados_exactos for r in rows]
    return {
        "ok": True,
        "rows": out_rows,
        "total": len(rows),
        "mias_encontradas": sum(1 for r in rows if r.numero_participacion in mios),
        "exactos_max": max(exactos_pool) if exactos_pool else 0,
        "exactos_media": (sum(exactos_pool) / len(exactos_pool)) if exactos_pool else 0.0,
    }


def load_ranking(penca_id: int) -> dict:
    return _cached(f"ranking:{penca_id}", RANKING_TTL, lambda: _fetch_ranking(penca_id))


def load_clausura_page(fecha_q: int | None = None) -> dict:
    """Todo lo que necesita el template en un solo dict."""
    cfg = load_clausura_config()
    if cfg is None:
        return {"ok": False, "error": "config/clausura2026.yaml no existe — "
                                       "correr `python -m src.clausura.sync`"}

    actual = fecha_actual(cfg)
    fecha_n = fecha_q or actual
    planilla = load_planilla(fecha_n)
    penca_id = cfg["pencas"]["paga"]["id"]

    premios = {p["tipo"]: p["monto"] for p in cfg.get("premios", [])}

    return {
        "ok": True,
        "fecha_n": fecha_n,
        "fecha_actual": actual,
        "fechas_guardadas": fechas_guardadas(),
        "planilla": planilla,
        "ranking": load_ranking(penca_id),
        "premios": premios,
        "penca_id": penca_id,
        "precio": cfg["pencas"]["paga"].get("precio", 400),
    }

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


_CACHE: dict[str, tuple[float, object]] = {}


def _cached(key: str, ttl: float, loader):
    """Memoize con TTL (mismo patrón que el dashboard del Mundial, sin depender de él)."""
    import time
    now = time.time()
    entry = _CACHE.get(key)
    if entry and (now - entry[0]) < ttl:
        return entry[1]
    value = loader()
    _CACHE[key] = (now, value)
    return value


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


def _fetch_ranking(penca_id: int, mios: frozenset[int] | None = None) -> dict:
    from src.clausura.api import PencaApiClient
    try:
        with PencaApiClient(timeout=10.0) as api:
            rows = api.ranking(penca_id)
    except Exception as e:
        log.warning("ranking no disponible: %s", e)
        return {"ok": False, "error": str(e), "rows": [], "total": None}

    if mios is None:
        mios_raw = os.environ.get("CLAUSURA_MIS_PARTICIPACIONES", "")
        mios = frozenset(int(x) for x in mios_raw.split(",") if x.strip().isdigit())

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


def load_ranking(penca_id: int, mios: frozenset[int] | None = None) -> dict:
    clave = f"ranking:{penca_id}:{sorted(mios) if mios is not None else '-'}"
    return _cached(clave, RANKING_TTL, lambda: _fetch_ranking(penca_id, mios))


def _mi_numero_gratuita() -> int | None:
    raw = os.environ.get("CLAUSURA_MI_PARTICIPACION_GRATUITA", "").strip()
    return int(raw) if raw.isdigit() else None


def build_gratuita(planilla: dict | None, cfg: dict) -> dict:
    """Tarjeta de la penca GRATUITA (id 47): 1 participación, premio PlayStation 5.

    El premio es INDIVISIBLE con desempate por exactos → empatar no diluye, así que
    el óptimo es EV puro = la columna 1 de la planilla (el ancla que el optimizador
    nunca perturba). Los ESPECIALES, en cambio, sí se diversifican en todas las
    columnas incluida la 1: acá va el argmax de P(campeón), no el de la fila 1.
    """
    penca = (cfg.get("pencas") or {}).get("gratuita") or {}
    penca_id = penca.get("id")
    out: dict = {
        "ok": penca_id is not None,
        "penca_id": penca_id,
        "mi_numero": _mi_numero_gratuita(),
        "supermatch_url": (f"https://www.supermatch.com.uy/pencas/1/{penca_id}/penca"
                           if penca_id else None),
        "picks": [],
        "campeon": None,
        "goleador": None,
    }
    if not out["ok"]:
        return out

    if planilla:
        for row in planilla.get("picks", []):
            scores = row.get("scores") or []
            if not scores:
                continue
            gl, gv = scores[0]                       # columna 1 = ancla EV
            out["picks"].append({
                "partido": row.get("partido", ""),
                "score": f"{gl}-{gv}",
                "preferencial": row.get("preferencial", False),
                "cierre_uy": row.get("cierre_uy", ""),
                "cerrado": row.get("cerrado", False),
            })
        esp = planilla.get("especiales") or {}
        p_campeon = esp.get("p_campeon") or {}
        if p_campeon:
            equipo, p = max(p_campeon.items(), key=lambda kv: kv[1])
            out["campeon"] = {"equipo": equipo, "p": p}
        por_part = esp.get("por_participacion") or []
        if por_part and por_part[0].get("goleador"):
            out["goleador"] = por_part[0]["goleador"]

    mio = out["mi_numero"]
    out["ranking"] = load_ranking(
        penca_id, frozenset({mio}) if mio is not None else frozenset())
    return out


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

    # Números de participación propios (CLAUSURA_MIS_PARTICIPACIONES), ascendente.
    # Convención de carga: columna i de la planilla ↔ i-ésimo número — cualquier
    # mapeo 1:1 consistente sirve; este es el que muestran las tarjetas del modo carga.
    mios_raw = os.environ.get("CLAUSURA_MIS_PARTICIPACIONES", "")
    mis_numeros = sorted(int(x) for x in mios_raw.split(",") if x.strip().isdigit())

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
        "mis_numeros": mis_numeros,
        "gratuita": build_gratuita(planilla, cfg),
        # Deep link a la penca en la web (la ruta lleva un cache-buster numérico,
        # mismo patrón que usa el propio front de Supermatch)
        "supermatch_url": f"https://www.supermatch.com.uy/pencas/1/{penca_id}/penca",
    }

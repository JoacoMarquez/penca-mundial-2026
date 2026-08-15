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
    """Fecha del PRÓXIMO partido por empezar (el de menor inicio futuro).

    No es "la primera fecha con algo pendiente": un suspendido reprogramado
    (Torque-Peñarol de la F1, re-datado semanas después) dejaría esa fecha
    vieja como "actual" todo el intermedio, y picks/rerun/drift-audit/carga
    apuntarían ahí en vez de a la fecha del fin de semana. Con el mínimo por
    hora de inicio, la fecha del makeup solo es "actual" los días en que ese
    partido es realmente lo próximo que se juega — que es cuando hay que
    generarle planilla y cargarlo.
    """
    now = datetime.now(timezone.utc)
    proximo: tuple[datetime, int] | None = None
    for nombre in cfg["fechas"]:
        n = int(nombre.split()[-1])
        for ev in cfg["fechas"][nombre]["eventos"]:
            inicio = datetime.fromisoformat(ev["inicio_utc"])
            if inicio > now and (proximo is None or inicio < proximo[0]):
                proximo = (inicio, n)
    return proximo[1] if proximo else 15


def load_planilla(fecha_n: int) -> dict | None:
    """Última versión APROBADA de la planilla de esa fecha, enriquecida para la UI.

    "Aprobada" = sin veredicto del gate, o con veredicto avisar=True. El rerun
    T-2h versiona su planilla aunque el gate la descarte (regla de trabajo #2), y
    servir esa —a veces PEOR que la de la mañana, Δ −456 ± 171 el 10/8— al que
    todavía no cargó lo hacía copiar exactamente lo que el gate resolvió no tocar.
    Las versiones descartadas quedan en disco y el panel de cambios las explica;
    `descartadas_despues` dice cuántas hay más nuevas que la servida.
    """
    from src.utils.versions import version_num

    d = fecha_dir(fecha_n)
    versiones = sorted(d.glob("v*_*.json"), key=version_num) if d.exists() else []
    if not versiones:
        return None
    elegida, data, descartadas = None, None, 0
    for path in reversed(versiones):
        candidata = json.loads(path.read_text(encoding="utf-8"))
        # `veredicto_heredado` lo deja la adopción del drift-audit: esa versión no
        # se midió, pero arrastra intactos los picks ABIERTOS de su fuente (solo
        # reescribe partidos cerrados), así que hereda su estado de aprobación.
        # Sin esto, una adopción sobre una versión descartada la resucitaría y el
        # modo carga volvería a mostrar justo los picks que el gate resolvió no tocar.
        veredicto = (candidata.get("veredicto_cambio")
                     or candidata.get("veredicto_heredado"))
        if veredicto and not veredicto.get("avisar", True):
            descartadas += 1
            continue
        elegida, data = path, candidata
        break
    if elegida is None:               # todas descartadas (no debería): la última
        elegida = versiones[-1]
        data = json.loads(elegida.read_text(encoding="utf-8"))
    latest = elegida
    data["descartadas_despues"] = descartadas
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


def _fetch_ranking_rows(penca_id: int) -> dict:
    """Las filas crudas del ranking. UNA request (size=1000) cada RANKING_TTL.

    Todas las vistas —la tabla del home y la página del pool— derivan de este mismo
    valor cacheado: pedir el ranking dos veces por render sería duplicar el tráfico
    al pedo, y este API tira 429 con facilidad.
    """
    from src.clausura.api import PencaApiClient
    try:
        with PencaApiClient(timeout=15.0) as api:
            return {"ok": True, "rows": api.ranking(penca_id)}
    except Exception as e:
        log.warning("ranking no disponible: %s", e)
        return {"ok": False, "error": str(e), "rows": []}


def load_ranking_rows(penca_id: int) -> dict:
    return _cached(f"ranking_rows:{penca_id}", RANKING_TTL,
                   lambda: _fetch_ranking_rows(penca_id))


def mis_numeros_env() -> frozenset[int]:
    raw = os.environ.get("CLAUSURA_MIS_PARTICIPACIONES", "")
    return frozenset(int(x) for x in raw.split(",") if x.strip().isdigit())


def _fetch_ranking(penca_id: int, mios: frozenset[int] | None = None) -> dict:
    crudo = load_ranking_rows(penca_id)
    if not crudo["ok"]:
        return {"ok": False, "error": crudo["error"], "rows": [], "total": None}
    rows = list(crudo["rows"])

    if mios is None:
        mios = mis_numeros_env()

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
    exactos_max = max(exactos_pool) if exactos_pool else 0
    return {
        "ok": True,
        "rows": out_rows,
        "total": len(rows),
        "mias_encontradas": sum(1 for r in rows if r.numero_participacion in mios),
        "exactos_max": exactos_max,
        "exactos_media": (sum(exactos_pool) / len(exactos_pool)) if exactos_pool else 0.0,
        # el contador de exactos lo liquida el API al cierre de la fecha, no en vivo
        "exactos_sin_liquidar": exactos_max == 0 and any(r.puntos_totales > 0 for r in rows),
    }


def load_ranking(penca_id: int, mios: frozenset[int] | None = None) -> dict:
    clave = f"ranking:{penca_id}:{sorted(mios) if mios is not None else '-'}"
    return _cached(clave, RANKING_TTL, lambda: _fetch_ranking(penca_id, mios))


def diff_versiones(fecha_n: int, mis_numeros: list[int]) -> dict:
    """Cambios entre las dos últimas corridas de la fecha (las 'pasadas' del Clausura).

    Equivalente a los diffs T-24h/T-3h/T-30min del Mundial: acá cada corrida del
    timer (o rerun-cierre) escribe v<N>.json, así que el cambio entre versiones ES
    el cambio entre pasadas. Muestra pick por pick qué se movió, más especiales y
    fuente del modelo (p.ej. 'ratings' → 'mercado+ratings' cuando aparecen cuotas).
    """
    d = fecha_dir(fecha_n)
    versiones = sorted(d.glob("v*_*.json")) if d.exists() else []
    out = {"n_versiones": len(versiones), "cambios": [], "fuentes": [],
           "especiales": [], "prev": None, "cur": None, "veredicto": None,
           "adopcion": None}
    if len(versiones) < 2:
        return out

    from src.utils.versions import latest_version, version_num
    versiones.sort(key=version_num)
    prev = json.loads(versiones[-2].read_text(encoding="utf-8"))
    cur = json.loads(versiones[-1].read_text(encoding="utf-8"))
    out["prev"] = {"file": versiones[-2].name, "generado_uy": _to_uy(prev.get("generado_utc", ""))}
    out["cur"] = {"file": versiones[-1].name, "generado_uy": _to_uy(cur.get("generado_utc", ""))}
    # Lo escribe rerun_cierre: si el gate por valor descartó estos cambios, el panel
    # tiene que mostrarlos apagados en vez de en amarillo. Las corridas del timer de
    # picks no pasan por el gate y no lo traen — ahí queda None y se muestra como antes.
    out["veredicto"] = cur.get("veredicto_cambio")
    # Una adopción del drift-audit no propone nada: reescribe partidos ya CERRADOS
    # con lo que dice la web. Sin esta bandera el panel la pintaba de amarillo
    # ("conviene corregir") sobre picks que ya no se pueden tocar.
    if cur.get("picks_adoptados_utc"):
        out["adopcion"] = {
            "cuando_uy": _to_uy(cur["picks_adoptados_utc"]),
            "de": (cur.get("veredicto_heredado") or {}).get("de"),
        }

    prev_by_ev = {r["evento_id"]: r for r in prev.get("picks", [])}
    now = datetime.now(timezone.utc)
    for row in cur.get("picks", []):
        p = prev_by_ev.get(row["evento_id"])
        if p is None:
            continue
        cerrado = datetime.fromisoformat(row["cierre_pronostico_utc"]) <= now
        if p.get("fuente_modelo") != row.get("fuente_modelo"):
            out["fuentes"].append({
                "partido": row["partido"],
                "antes": p.get("fuente_modelo"), "despues": row.get("fuente_modelo"),
            })
        for k, (a, b) in enumerate(zip(p.get("scores", []), row.get("scores", []))):
            if list(a) != list(b):
                out["cambios"].append({
                    "partido": row["partido"],
                    "numero": mis_numeros[k] if k < len(mis_numeros) else k + 1,
                    "antes": f"{a[0]}-{a[1]}", "despues": f"{b[0]}-{b[1]}",
                    "cerrado": cerrado,
                })

    pe = (prev.get("especiales") or {}).get("por_participacion") or []
    ce = (cur.get("especiales") or {}).get("por_participacion") or []
    for k, (a, b) in enumerate(zip(pe, ce)):
        for campo in ("campeon", "goleador"):
            if a.get(campo) != b.get(campo):
                out["especiales"].append({
                    "numero": mis_numeros[k] if k < len(mis_numeros) else k + 1,
                    "campo": campo, "antes": a.get(campo), "despues": b.get(campo),
                })
    return out


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


def load_pool_page() -> dict:
    """Página del pool: dónde caen nuestras participaciones entre las ~700.

    Costo en API: la MISMA request cacheada que usa el home (ranking completo en una
    sola llamada). La trayectoria y el fecha-a-fecha salen de disco.
    """
    from src.clausura.picks import flat_eventos
    from src.clausura.pool_view import (
        historia_por_fecha, leer_historia, liquidables_en, movimientos_por_partido,
        registrar_historia, resumen_pool,
    )

    cfg = load_clausura_config()
    if cfg is None:
        return {"ok": False, "error": "config/clausura2026.yaml no existe — "
                                       "correr `python -m src.clausura.sync`"}
    penca_id = cfg["pencas"]["paga"]["id"]
    premios = {p["tipo"]: p["monto"] for p in cfg.get("premios", [])}

    crudo = load_ranking_rows(penca_id)
    if not crudo["ok"]:
        return {"ok": False, "error": f"ranking no disponible: {crudo['error']}",
                "por_fecha": historia_por_fecha(), "trayectoria": leer_historia()}

    mios = mis_numeros_env()
    resumen = resumen_pool(
        crudo["rows"], mios,
        premio_penca=premios.get("PENCA", 350_000.0),
        premio_fecha=premios.get("FECHA", 10_000.0),
    )
    eventos = flat_eventos(cfg)
    registrar_historia(
        resumen,
        liquidables=liquidables_en(eventos, datetime.now(timezone.utc)),
    )

    resumen["fecha_actual"] = fecha_actual(cfg)
    resumen["penca_id"] = penca_id
    resumen["sin_env"] = not mios
    historia = leer_historia()
    resumen["trayectoria"] = historia[-20:]
    resumen["movimientos"] = movimientos_por_partido(historia, eventos)
    resumen["por_fecha"] = historia_por_fecha()
    resumen["supermatch_url"] = f"https://www.supermatch.com.uy/pencas/1/{penca_id}/penca"
    return resumen


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
        "cambios": diff_versiones(fecha_n, mis_numeros),
        # Deep link a la penca en la web (la ruta lleva un cache-buster numérico,
        # mismo patrón que usa el propio front de Supermatch)
        "supermatch_url": f"https://www.supermatch.com.uy/pencas/1/{penca_id}/penca",
    }

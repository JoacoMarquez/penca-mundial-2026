"""Verificación real de la carga: qué dice la WEB contra qué dice la planilla.

El modo carga marca lo que vos decís que cargaste; esto lee lo que efectivamente
quedó cargado. Post-inicio del campeonato los pronósticos propios son públicos
(mismo endpoint que usa pool_snapshot para leer al pool), así que la comparación es
contra la verdad, no contra la memoria: un 2-1 tipeado como 2-0 aparece acá y en
ningún otro lado.

Complementa a carga_alert (que avisa por Telegram si FALTA cargar, sin mirar el
valor) y al panel de cambios del modo carga (que compara contra tu propia marca).

Uso:
    python -m src.clausura.verificar_carga --fecha 3
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import httpx

from src.clausura.api import BASE, HEADERS, PencaApiClient
from src.clausura.pool_snapshot import GATE_MSG, REQUEST_PAUSE_S, _get_pacing

log = logging.getLogger(__name__)

VERIF_TTL = 60.0        # cache del endpoint: un doble-clic no dispara dos escaneos

OK = "ok"               # la web dice lo mismo que la planilla
DISTINTO = "distinto"   # la web dice otra cosa → hay que corregir
FALTA = "falta"         # el pick no está cargado (solo comprobable en partidos cerrados)
NO_VISIBLE = "no_visible"   # el partido no cerró: el API todavía no publica ese pick
SIN_DATOS = "sin_datos"     # la participación no apareció en el ranking


class GateCerrado(RuntimeError):
    """El API no expone pronósticos hasta que arranca el campeonato."""


# -------------------- comparación pura (testeable sin red) --------------------

def _fmt(par) -> str:
    return f"{par[0]}-{par[1]}"


def _norm(s: str | None) -> str:
    return (s or "").strip().casefold()


def comparar(
    planilla: dict,
    mis_numeros: list[int],
    picks_web: dict[int, dict[int, tuple[int, int]]],
    esp_web: dict[int, dict] | None = None,
) -> dict:
    """Planilla vs lo cargado en la web, fila por fila.

    `picks_web` y `esp_web` van indexados por numeroParticipacion; la columna i de
    la planilla corresponde a `mis_numeros[i]` (misma convención de carga que las
    tarjetas del modo carga).

    OJO con el alcance: el API publica el pick de un partido recién cuando ESE
    partido cierra (gate por evento, verificado 2026-08-07). Un partido abierto que
    no aparece NO es un pick faltante — es un pick que todavía no se puede ver, y
    decir "sin cargar" ahí manda a recargar lo que ya está puesto. Por eso `FALTA`
    se reserva a los partidos cerrados y el resto cae en `NO_VISIBLE`.
    """
    n = int(planilla.get("n_participaciones", 0))
    picks = planilla.get("picks", [])
    filas: list[dict] = []
    vistos_de_la_fecha = 0

    for i in range(n):
        numero = mis_numeros[i] if i < len(mis_numeros) else None
        cargado = picks_web.get(numero) if numero is not None else None
        for row in picks:
            ev = int(row["evento_id"])
            if row.get("scores"):
                plan = tuple(row["scores"][i])
            else:
                plan = tuple(int(x) for x in row["scores_fmt"][i].split("-"))
            web = cargado.get(ev) if cargado else None
            cerrado = bool(row.get("cerrado"))
            if cargado is None:
                estado = SIN_DATOS
            elif web is None:
                estado = FALTA if cerrado else NO_VISIBLE
            else:
                vistos_de_la_fecha += 1
                estado = OK if tuple(web) == plan else DISTINTO
            filas.append({
                "part": i,
                "numero": numero,
                "evento_id": ev,
                "partido": row.get("partido", ""),
                "cerrado": cerrado,
                "planilla": _fmt(plan),
                "web": _fmt(web) if web is not None else None,
                "estado": estado,
            })

    especiales: list[dict] = []
    por_part = (planilla.get("especiales") or {}).get("por_participacion") or []
    for i, esp in enumerate(por_part[:n]):
        numero = mis_numeros[i] if i < len(mis_numeros) else None
        web = (esp_web or {}).get(numero) if numero is not None else None
        for campo in ("campeon", "goleador"):
            plan_v = esp.get(campo)
            if not plan_v:
                continue          # la planilla no lo asignó: nada que verificar
            if web is None:
                estado, web_v = SIN_DATOS, None
            else:
                web_v = web.get(campo)
                if not web_v:
                    estado = FALTA
                else:
                    estado = OK if _norm(web_v) == _norm(plan_v) else DISTINTO
            especiales.append({
                "part": i, "numero": numero, "campo": campo,
                "planilla": plan_v, "web": web_v, "estado": estado,
            })

    todo = filas + especiales
    resumen = {e: sum(1 for f in todo if f["estado"] == e)
               for e in (OK, DISTINTO, FALTA, NO_VISIBLE, SIN_DATOS)}
    n_picks_web = sum(len(v) for v in picks_web.values())
    # Qué se pudo verificar de verdad: lo que quedó con veredicto. Un partido abierto
    # (NO_VISIBLE) o una participación que no está en el ranking (SIN_DATOS) no
    # cuentan. Si el total da cero, el botón no puede decir nada todavía — y hay que
    # decirlo así, no dar un ✅ vacío.
    verificables = sum(1 for f in todo if f["estado"] in (OK, DISTINTO, FALTA))
    return {
        "ok": True,
        "verificado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filas": filas,
        "especiales": especiales,
        "resumen": resumen,
        "n_participaciones": n,
        "n_verificables": verificables,
        "n_no_visibles": resumen[NO_VISIBLE],
        "nada_verificable": verificables == 0,
        # El API devolvió picks pero ninguno de esta fecha (ni siquiera de los
        # partidos ya cerrados): algo no cuadra con el gate, no es "falta cargar".
        "sin_datos_fecha": (vistos_de_la_fecha == 0 and n_picks_web > 0
                            and any(f["cerrado"] for f in filas)),
        "sin_datos_api": n_picks_web == 0 and not esp_web,
    }


# -------------------- lectura del API --------------------

def fetch_cargado(
    penca_id: int,
    mis_numeros: list[int],
    pause_s: float = REQUEST_PAUSE_S,
    con_especiales: bool = True,
) -> tuple[dict[int, dict[int, tuple[int, int]]], dict[int, dict]]:
    """(picks, especiales) por numeroParticipacion, leídos de la web.

    Son 1-2 requests por participación (≈24 con las 12 nuestras): pacing igual que
    el escaneo del pool para no comernos el 429 del API.
    """
    with PencaApiClient() as api:
        ranking = api.ranking(penca_id)
    mios = [r for r in ranking if r.numero_participacion in set(mis_numeros)]
    if not mios:
        log.warning("ninguna de mis participaciones (%s) está en el ranking", mis_numeros)
        return {}, {}

    picks: dict[int, dict[int, tuple[int, int]]] = {}
    esp: dict[int, dict] = {}
    with httpx.Client(base_url=BASE, timeout=20.0, headers=HEADERS) as c:
        for r in mios:
            num, pid = r.numero_participacion, r.participacion_id
            resp = _get_pacing(c, f"/front/pencas/{pid}/pronosticosEventos", pause_s)
            if resp.status_code == 400 and GATE_MSG in resp.text:
                raise GateCerrado(resp.text[:120])
            fila: dict[int, tuple[int, int]] = {}
            if resp.status_code == 200:
                for p in resp.json().get("data", []):
                    gl, gv, eid = (p.get("golesEquipoLocal"), p.get("golesEquipoVisitante"),
                                   p.get("encuentroId"))
                    if gl is not None and gv is not None and eid is not None:
                        fila[int(eid)] = (int(gl), int(gv))
            else:
                log.warning("pronosticosEventos %d → %d", pid, resp.status_code)
            picks[num] = fila

            if con_especiales:
                resp = _get_pacing(c, f"/front/pencas/{pid}/pronosticoCampeonGoleador", pause_s)
                if resp.status_code == 200:
                    d = resp.json()
                    esp[num] = {
                        "campeon": (d.get("equipoCampeon") or {}).get("nombre"),
                        "goleador": (d.get("opcionGoleador") or {}).get("goleador"),
                    }
    return picks, esp


def verificar(fecha_n: int | None = None) -> dict:
    """Todo junto, listo para el dashboard. Nunca levanta: devuelve ok=False."""
    from src.clausura.dashboard_loader import (
        fecha_actual, load_clausura_config, load_planilla, mis_numeros_env,
    )

    cfg = load_clausura_config()
    if cfg is None:
        return {"ok": False, "error": "falta config/clausura2026.yaml"}
    fecha_n = fecha_n or fecha_actual(cfg)
    planilla = load_planilla(fecha_n)
    if not planilla:
        return {"ok": False, "error": f"sin planilla para la Fecha {fecha_n}"}
    mis_numeros = sorted(mis_numeros_env())
    if not mis_numeros:
        return {"ok": False, "error": "CLAUSURA_MIS_PARTICIPACIONES no está en el .env"}

    try:
        picks, esp = fetch_cargado(cfg["pencas"]["paga"]["id"], mis_numeros)
    except GateCerrado:
        return {"ok": False, "error": "el API no expone los pronósticos hasta que "
                                      "arranca el campeonato"}
    except Exception as e:                                   # 429, red, formato
        log.warning("verificación fallida: %s", e)
        return {"ok": False, "error": f"no pude leer la web ({type(e).__name__}: {e})"}

    out = comparar(planilla, mis_numeros, picks, esp)
    out["fecha_n"] = fecha_n
    out["version_file"] = planilla.get("version_file")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", type=int, default=None)
    args = ap.parse_args()

    res = verificar(args.fecha)
    if not res["ok"]:
        print(f"❌ {res['error']}")
        raise SystemExit(1)
    r = res["resumen"]
    print(f"Fecha {res['fecha_n']} · {res['version_file']}")
    print(f"✅ {r[OK]} coinciden · ⚠️ {r[DISTINTO]} distintos · "
          f"⬜ {r[FALTA]} sin cargar · 🔒 {r[NO_VISIBLE]} no visibles todavía · "
          f"❔ {r[SIN_DATOS]} sin datos")
    if res["nada_verificable"]:
        print("(ningún partido de la fecha cerró todavía: el API publica cada pick "
              "recién al cierre de ESE partido — no hay nada que verificar aún)")
    if res["sin_datos_fecha"]:
        print("(el API no devolvió pronósticos de esta fecha — no es que falten)")
    for f in res["filas"] + res["especiales"]:
        if f["estado"] in (DISTINTO, FALTA):
            que = f.get("partido") or f.get("campo")
            print(f"  {f['estado']:8s} N°{f['numero']} {que}: "
                  f"planilla {f['planilla']} · web {f['web']}")


if __name__ == "__main__":
    main()

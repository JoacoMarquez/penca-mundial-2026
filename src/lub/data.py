"""Datos de la LUB desde el penca-api (lectura, sin auth).

- Temporadas (fixture + resultados): campeonato 26 = LUB 24/25, 31 = LUB 25/26,
  45 = LUB 26/27 (la actual). Los nombres de fecha de playoffs no son "Fecha N"
  ("Cuartos de Final Playoffs (1 y 2)"), por eso NO se usa api.fechas() — ordena
  por int(nombre.split()[-1]) y revienta.
- Pool histórico: los picks de las 320 participaciones de la penca 37 (LUB 25/26),
  públicos por participación en /front/pencas/{participacion_id}/pronosticosEventos.
  Con eso se modela cómo juega el pool (Q por clase) sin suponer nada.

Uso:
    python -m src.lub.data temporadas      # → data/lub/temporadas.json
    python -m src.lub.data pool --penca 37 # → data/lub/pool_37.json (~320 requests, pacing)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.clausura.api import BASE, HEADERS, TZ_UY, PencaApiClient, resultado_finalizado
from src.clausura.pool_snapshot import _get_pacing

log = logging.getLogger(__name__)

DATA_DIR = Path("data/lub")
PENCA_ID = 48
CAMPEONATO_ID = 45
TEMPORADAS = {26: "LUB 24/25", 31: "LUB 25/26", 45: "LUB 26/27"}
PENCA_HISTORICA = {37: 31}   # penca → campeonato (LUB 25/26)
PAUSE_S = 0.6


@dataclass
class Partido:
    campeonato_id: int
    temporada: str
    fecha_id: int
    fecha_nombre: str
    fase: str               # "regular" | "liguilla" | "playoff"
    evento_id: int
    local: str
    visitante: str
    local_id: int
    visitante_id: int
    inicio_utc: str
    cierre_utc: str
    preferencial: bool
    estado: str
    pts_local: int | None
    pts_visitante: int | None


def fase_de(nombre: str) -> str:
    n = nombre.lower()
    if "liguilla" in n or "reclasif" in n:
        return "liguilla"
    if n.startswith("fecha"):
        return "regular"
    return "playoff"


def _dt(s: str) -> str:
    return (datetime.strptime(s, "%d-%m-%Y %H:%M:%S").replace(tzinfo=TZ_UY)
            .astimezone(timezone.utc).isoformat())


def fetch_temporada(c: httpx.Client, campeonato_id: int) -> list[Partido]:
    fechas = c.get(f"/front/campeonatos/{campeonato_id}/fechas").json()["data"]
    out: list[Partido] = []
    for f in fechas:
        evs = c.get(f"/front/campeonatos/fechas/{f['id']}/eventos").json()["data"]
        time.sleep(0.2)
        for e in evs:
            real = resultado_finalizado(e)
            out.append(Partido(
                campeonato_id=campeonato_id, temporada=TEMPORADAS.get(campeonato_id, str(campeonato_id)),
                fecha_id=f["id"], fecha_nombre=f["nombre"], fase=fase_de(f["nombre"]),
                evento_id=e["id"], local=e["equipoLocal"]["nombre"].strip(),
                visitante=e["equipoVisitante"]["nombre"].strip(),
                local_id=e["equipoLocal"]["id"], visitante_id=e["equipoVisitante"]["id"],
                inicio_utc=_dt(e["fechaInicio"]), cierre_utc=_dt(e["fechaCierrePronostico"]),
                preferencial=bool(e.get("preferencial")), estado=e.get("estado", ""),
                pts_local=real[0] if real else None, pts_visitante=real[1] if real else None,
            ))
    out.sort(key=lambda p: p.inicio_utc)
    return out


def fetch_temporadas(ids: list[int] | None = None) -> list[Partido]:
    todos: list[Partido] = []
    with httpx.Client(base_url=BASE, headers=HEADERS, timeout=30.0) as c:
        for cid in ids or list(TEMPORADAS):
            ps = fetch_temporada(c, cid)
            log.info("campeonato %d: %d partidos (%d con resultado)", cid, len(ps),
                     sum(p.pts_local is not None for p in ps))
            todos.extend(ps)
    return todos


def save_temporadas(ps: list[Partido], path: Path | None = None) -> Path:
    path = path or DATA_DIR / "temporadas.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in ps], ensure_ascii=False, indent=0))
    return path


def load_temporadas(path: Path | None = None) -> list[Partido]:
    path = path or DATA_DIR / "temporadas.json"
    return [Partido(**d) for d in json.loads(path.read_text())]


def fetch_pool(penca_id: int, pause_s: float = PAUSE_S) -> dict:
    """Picks + especiales de TODAS las participaciones de una penca."""
    with PencaApiClient() as api:
        ranking = api.ranking(penca_id)
    filas = []
    with httpx.Client(base_url=BASE, headers=HEADERS, timeout=30.0) as c:
        for i, r in enumerate(ranking):
            pe = _get_pacing(c, f"/front/pencas/{r.participacion_id}/pronosticosEventos", pause_s)
            es = _get_pacing(c, f"/front/pencas/{r.participacion_id}/pronosticoCampeonGoleador", pause_s)
            picks = pe.json() if pe.status_code == 200 else []
            picks = picks.get("data", picks) if isinstance(picks, dict) else picks
            esp = es.json() if es.status_code == 200 else {}
            filas.append({
                "participacion_id": r.participacion_id, "numero": r.numero_participacion,
                "puntos_totales": r.puntos_totales, "posicion": r.posicion_general,
                "picks": [{"evento_id": p["encuentroId"], "local": p["golesEquipoLocal"],
                           "visitante": p["golesEquipoVisitante"], "puntos": p.get("puntos"),
                           "creado": p.get("creationDate"), "modificado": p.get("lastModifiedDate")}
                          for p in picks],
                "campeon": (esp.get("equipoCampeon") or {}).get("nombre"),
                "puntos_campeon": esp.get("puntosCampeon"),
                "goleador": (esp.get("opcionGoleador") or {}).get("goleador"),
                "puntos_goleador": esp.get("puntosGoleador"),
            })
            if (i + 1) % 50 == 0:
                log.info("pool %d: %d/%d", penca_id, i + 1, len(ranking))
    return {"penca_id": penca_id, "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "participaciones": filas}


def mis_numeros_env() -> list[int]:
    """Números de participación propios en la LUB (LUB_MIS_PARTICIPACIONES del env).

    Lista y no set, a diferencia del Clausura: el ORDEN es el contrato con la planilla
    (columna i ↔ i-ésimo número) y lo tiene que ver igual el modo carga."""
    raw = os.environ.get("LUB_MIS_PARTICIPACIONES", "")
    return [int(x) for x in raw.split(",") if x.strip().isdigit()]


def fetch_opciones_goleador(campeonato_id: int = CAMPEONATO_ID) -> list[str] | None:
    """Nombres del menú de goleador; None si el API no lo sirve.

    OJO: la ruta dice `/pencas/` pero el id es el del CAMPEONATO (45), no el de la
    penca (48) — así lo llama la web. Con el id de la penca da 500 siempre, que es lo
    que se leyó durante semanas como "menú no publicado"."""
    with httpx.Client(base_url=BASE, headers=HEADERS, timeout=20.0) as c:
        r = c.get(f"/front/pencas/{campeonato_id}/opcionesGoleador")
    if r.status_code != 200:
        return None
    data = r.json().get("opcionesGoleador", {}).get("data", [])
    return [o["goleador"].strip() for o in data if o.get("goleador")] or None


def fetch_especiales_propios(numeros: list[int], penca_id: int = PENCA_ID,
                             pause_s: float = PAUSE_S) -> list[tuple[str | None, str | None]] | None:
    """(campeón, goleador) cargados en cada número propio, en el orden de `numeros`.

    Son públicos recién desde el primer partido (mismo gate que los picks). None si el
    ranking todavía no trae alguno de los números: mejor no evaluar que evaluar con
    especiales inventados."""
    with PencaApiClient() as api:
        ranking = api.ranking(penca_id)
    pid = {r.numero_participacion: r.participacion_id for r in ranking}
    if any(n not in pid for n in numeros):
        return None
    out = []
    with httpx.Client(base_url=BASE, headers=HEADERS, timeout=30.0) as c:
        for n in numeros:
            r = _get_pacing(c, f"/front/pencas/{pid[n]}/pronosticoCampeonGoleador", pause_s)
            esp = r.json() if r.status_code == 200 else {}
            out.append(((esp.get("equipoCampeon") or {}).get("nombre"),
                        (esp.get("opcionGoleador") or {}).get("goleador")))
    return out


def puntos_parciales(picks_por_part: list[list[dict]], partidos: list[Partido]) -> dict[str, list[int]]:
    """Puntos de cada participación en las fechas ABIERTAS que ya tienen partidos jugados.

    Se calculan con el kernel propio contra el resultado real, no con el campo `puntos`
    del API: en el Clausura algunos campos de liquidación se llenaban recién al cerrar
    la fecha, y acá la fecha está justamente a medio jugar."""
    from src.lub.scoring import puntos

    actual = [p for p in partidos if p.campeonato_id == CAMPEONATO_ID]
    abiertas = {p.fecha_nombre for p in actual if p.pts_local is None}
    jugados = {p.evento_id: p for p in actual if p.pts_local is not None and p.fecha_nombre in abiertas}
    fechas = sorted({p.fecha_nombre for p in jugados.values()})
    out = {f: [0] * len(picks_por_part) for f in fechas}
    for i, picks in enumerate(picks_por_part):
        for pk in picks:
            e = jugados.get(pk.get("encuentroId"))
            if e is None or pk.get("golesEquipoLocal") is None:
                continue
            out[e.fecha_nombre][i] += puntos((pk["golesEquipoLocal"], pk["golesEquipoVisitante"]),
                                             (e.pts_local, e.pts_visitante), e.preferencial)
    return out


def fetch_previos(numeros: list[int], partidos: list[Partido], penca_id: int = PENCA_ID,
                  pause_s: float = PAUSE_S):
    """Previos (season.Previos) desde el ranking vivo, o None si falta algún número propio.

    Los totales salen del ranking (una request). Los parciales de fechas abiertas exigen
    los picks de TODAS las participaciones (una request cada una, ~4 min con pacing) y
    solo se piden si alguna fecha abierta ya tiene partidos jugados."""
    import numpy as np

    from src.lub.season import Previos

    with PencaApiClient() as api:
        ranking = api.ranking(penca_id)
    por_num = {r.numero_participacion: r for r in ranking}
    if any(n not in por_num for n in numeros):
        log.warning("números propios fuera del ranking LUB: %s", [n for n in numeros if n not in por_num])
        return None
    propios = [por_num[n] for n in numeros]
    rivales = [r for r in ranking if r.numero_participacion not in set(numeros)]
    prev = Previos(tot_riv=np.array([r.puntos_totales or 0 for r in rivales], np.int16),
                   tot_prop=np.array([r.puntos_totales or 0 for r in propios], np.int16))
    actual = [p for p in partidos if p.campeonato_id == CAMPEONATO_ID]
    abiertas = {p.fecha_nombre for p in actual if p.pts_local is None}
    if not any(p.pts_local is not None and p.fecha_nombre in abiertas for p in actual):
        return prev
    filas = propios + rivales
    picks = []
    with httpx.Client(base_url=BASE, headers=HEADERS, timeout=30.0) as c:
        for i, r in enumerate(filas):
            resp = _get_pacing(c, f"/front/pencas/{r.participacion_id}/pronosticosEventos", pause_s)
            d = resp.json() if resp.status_code == 200 else []
            picks.append(d.get("data", d) if isinstance(d, dict) else d)
            if (i + 1) % 100 == 0:
                log.info("parciales LUB: %d/%d participaciones", i + 1, len(filas))
    parc = puntos_parciales(picks, partidos)
    k = len(propios)
    prev.parcial_prop = {f: np.array(v[:k], np.int16) for f, v in parc.items()}
    prev.parcial_riv = {f: np.array(v[k:], np.int16) for f, v in parc.items()}
    return prev


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("temporadas")
    t.add_argument("--ids", type=int, nargs="*")
    p = sub.add_parser("pool")
    p.add_argument("--penca", type=int, default=37)
    a = ap.parse_args()
    if a.cmd == "temporadas":
        print(save_temporadas(fetch_temporadas(a.ids)))
    else:
        out = DATA_DIR / f"pool_{a.penca}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fetch_pool(a.penca), ensure_ascii=False))
        print(out)


if __name__ == "__main__":
    main()

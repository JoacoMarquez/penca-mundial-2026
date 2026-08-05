"""Snapshot del pool: los picks de TODAS las participaciones, leídos del penca-api.

Desde el inicio del campeonato (gate del API: "Solo puede ver pronósticos si el
campeonato ya inicio"), los pronósticos de cada participación son públicos:

    GET front/pencas/{participacionId}/pronosticosEventos      → marcadores
    GET front/pencas/{participacionId}/pronosticoCampeonGoleador → especiales

(OJO: el path param es participacionId — sale del ranking — NO el id de la penca.)

Esto convierte la Capa 5 de modelada a EMPÍRICA: la distribución Q de picks por
partido se observa, no se infiere. El prior con bias de popularidad queda solo como
arranque en frío (Fecha 1) y como relleno donde falten observaciones.

Uso:
    python -m src.clausura.pool_snapshot                 # toma snapshot (error si no inició)
    python -m src.clausura.pool_snapshot --if-started    # sale 0 en silencio pre-inicio
                                                          (para el ExecStartPre del timer)

El snapshot se versiona en data/pool_snapshots/clausura/ (nunca sobreescribe).
Riesgo simétrico documentado en memoria: nuestros picks también son públicos.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from src.clausura.api import BASE, HEADERS, PencaApiClient
from src.clausura.economics import MAX_GOALS, N_SCORES, score_index

log = logging.getLogger(__name__)

SNAP_DIR = Path(__file__).resolve().parents[2] / "data" / "pool_snapshots" / "clausura"
REQUEST_PAUSE_S = 0.12   # ~8 req/s con 2 GETs por rival — cortés con el API
GATE_MSG = "campeonato ya inicio"


@dataclass
class RivalPicks:
    participacion_id: int
    numero: int
    puntos: int
    exactos: int
    # evento_id → (goles_local, goles_visitante)
    picks: dict[int, tuple[int, int]] = field(default_factory=dict)
    campeon: str | None = None
    campeon_id: int | None = None
    goleador: str | None = None
    goleador_id: int | None = None


class CampeonatoNoIniciado(RuntimeError):
    pass


# -------------------- fetch --------------------

def fetch_snapshot(penca_id: int, pause_s: float = REQUEST_PAUSE_S) -> list[RivalPicks]:
    """Baja los picks de todas las participaciones del ranking. Lanza
    CampeonatoNoIniciado si el gate del API sigue cerrado."""
    with PencaApiClient() as api:
        ranking = api.ranking(penca_id)
    log.info("snapshot: %d participaciones en el ranking", len(ranking))

    out: list[RivalPicks] = []
    with httpx.Client(base_url=BASE, timeout=20.0, headers=HEADERS) as c:
        for i, row in enumerate(ranking):
            pid = row.participacion_id
            rival = RivalPicks(
                participacion_id=pid,
                numero=row.numero_participacion,
                puntos=row.puntos_totales,
                exactos=row.cant_resultados_exactos,
            )

            r = c.get(f"/front/pencas/{pid}/pronosticosEventos")
            if r.status_code == 400 and GATE_MSG in r.text:
                raise CampeonatoNoIniciado(r.text[:120])
            if r.status_code == 200:
                for p in r.json().get("data", []):
                    gl, gv = p.get("golesEquipoLocal"), p.get("golesEquipoVisitante")
                    eid = p.get("encuentroId")
                    if gl is not None and gv is not None and eid is not None:
                        rival.picks[int(eid)] = (int(gl), int(gv))
            else:
                log.warning("pronosticosEventos %d → %d", pid, r.status_code)

            r = c.get(f"/front/pencas/{pid}/pronosticoCampeonGoleador")
            if r.status_code == 400 and GATE_MSG in r.text:
                raise CampeonatoNoIniciado(r.text[:120])
            if r.status_code == 200:
                d = r.json()
                eq = d.get("equipoCampeon") or {}
                gol = d.get("opcionGoleador") or {}
                rival.campeon, rival.campeon_id = eq.get("nombre"), eq.get("id")
                rival.goleador, rival.goleador_id = gol.get("goleador"), gol.get("id")

            out.append(rival)
            if (i + 1) % 50 == 0:
                log.info("  %d/%d", i + 1, len(ranking))
            time.sleep(pause_s)
    return out


def save_snapshot(rivales: list[RivalPicks], penca_id: int) -> Path:
    from src.utils.versions import latest_version, version_num
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    prev = latest_version(SNAP_DIR.glob("v*_*.json"))
    n = version_num(prev) + 1 if prev else 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = SNAP_DIR / f"v{n}_{ts}.json"
    payload = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "penca_id": penca_id,
        "n_participaciones": len(rivales),
        "participaciones": [
            {**asdict(r), "picks": {str(k): list(v) for k, v in r.picks.items()}}
            for r in rivales
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_latest_snapshot(max_age_hours: float | None = None) -> dict | None:
    from src.utils.versions import latest_version
    latest = latest_version(SNAP_DIR.glob("v*_*.json")) if SNAP_DIR.exists() else None
    if latest is None:
        return None
    data = json.loads(latest.read_text(encoding="utf-8"))
    if max_age_hours is not None:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(data["generado_utc"])).total_seconds() / 3600
        if age > max_age_hours:
            log.info("snapshot más viejo que %.0fh (%.1fh) — se ignora", max_age_hours, age)
            return None
    return data


# -------------------- Q empírica --------------------

def empirical_counts(
    snapshot: dict, mis_numeros: set[int] | None = None,
) -> dict[int, np.ndarray]:
    """evento_id → conteos de picks sobre los N_SCORES índices.

    `mis_numeros` excluye nuestras participaciones: la Q empírica modela a los
    RIVALES; contarnos (12 planillas anti-chalk sobre ~150) infla justo los
    marcadores diferenciados y hace desaparecer el hueco que explotamos.
    """
    mis = mis_numeros or set()
    counts: dict[int, np.ndarray] = {}
    for r in snapshot.get("participaciones", []):
        if int(r.get("numero", -1)) in mis:
            continue
        for eid_str, (gl, gv) in r.get("picks", {}).items():
            eid = int(eid_str)
            if eid not in counts:
                counts[eid] = np.zeros(N_SCORES)
            counts[eid][score_index(min(int(gl), MAX_GOALS), min(int(gv), MAX_GOALS))] += 1
    return counts


def blended_q(prior_q: np.ndarray, counts: np.ndarray | None, strength: float = 25.0) -> np.ndarray:
    """Posterior Dirichlet: Q = (conteos + strength·prior) / (n + strength).

    Con pocas observaciones domina el prior; con muchas, lo observado. strength≈25
    equivale a "el prior vale 25 rivales imaginarios".
    """
    if counts is None or counts.sum() == 0:
        return prior_q
    q = counts + strength * prior_q
    return q / q.sum()


def empirical_campeon_counts(
    snapshot: dict, equipo_idx: dict[str, int], n_teams: int,
    mis_numeros: set[int] | None = None,
) -> np.ndarray:
    """Conteos de picks de campeón por equipo (índices del optimizador).

    `mis_numeros` excluye nuestras participaciones, igual que en empirical_counts.
    """
    mis = mis_numeros or set()
    counts = np.zeros(n_teams)
    for r in snapshot.get("participaciones", []):
        if int(r.get("numero", -1)) in mis:
            continue
        nombre = r.get("campeon")
        if nombre and nombre in equipo_idx:
            counts[equipo_idx[nombre]] += 1
    return counts


def snapshot_summary(snapshot: dict) -> str:
    """Resumen humano para el log/Telegram."""
    n = snapshot.get("n_participaciones", 0)
    counts = empirical_counts(snapshot)
    con_picks = sum(1 for r in snapshot.get("participaciones", []) if r.get("picks"))
    con_campeon = sum(1 for r in snapshot.get("participaciones", []) if r.get("campeon"))
    return (f"{n} participaciones · {con_picks} con marcadores cargados · "
            f"{con_campeon} con campeón · {len(counts)} eventos observados")


# -------------------- CLI --------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-started", action="store_true",
                    help="si el campeonato no inició, salir 0 en silencio (para timers)")
    args = ap.parse_args()

    import yaml
    from src.clausura.picks import CONFIG_PATH
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    penca_id = cfg["pencas"]["paga"]["id"]

    try:
        rivales = fetch_snapshot(penca_id)
    except CampeonatoNoIniciado:
        if args.if_started:
            print("campeonato no iniciado — snapshot omitido")
            sys.exit(0)
        print("ERROR: el campeonato no inició; los picks aún no son públicos", file=sys.stderr)
        sys.exit(1)

    path = save_snapshot(rivales, penca_id)
    snap = json.loads(path.read_text(encoding="utf-8"))
    print(f"guardado {path}")
    print(snapshot_summary(snap))


if __name__ == "__main__":
    main()

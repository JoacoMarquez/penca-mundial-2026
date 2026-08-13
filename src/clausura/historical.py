"""Descarga de temporadas pasadas del penca-api para backtest.

El endpoint `front/campeonatos` lista TODOS los campeonatos históricos, y los eventos
de un campeonato terminado traen `resultado.golesEquipoLocal/Visitante`. Eso nos da el
dataset de validación que en el Mundial tuvimos que reconstruir a mano desde la Euro.

Temporadas de la Primera uruguaya disponibles (verificado 2026-08-04):
    21 Apertura 2024 · 25 Clausura 2024 · 27 Apertura 2025 · 30 Clausura 2025 · 41 Apertura 2026

Uso:
    python -m src.clausura.historical            # baja todas a data/processed/
    python -m src.clausura.historical --camp 41  # solo una
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from src.clausura.api import BASE, HEADERS, _parse_dt, resultado_finalizado

import httpx

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Campeonatos de Primera uruguaya con formato idéntico al Clausura 2026
# (15 fechas x 8 partidos, mismo scoring de penca).
TEMPORADAS = {
    21: "Torneo Apertura 2024",
    25: "Torneo Clausura 2024",
    27: "Torneo Apertura 2025",
    30: "Torneo Clausura 2025",
    41: "Torneo Apertura 2026",
}


@dataclass(frozen=True)
class PartidoHistorico:
    campeonato_id: int
    campeonato: str
    fecha_nombre: str
    fecha_id: int
    evento_id: int
    local: str
    visitante: str
    goles_local: int
    goles_visitante: int
    preferencial: bool
    inicio_utc: str


def fetch_temporada(campeonato_id: int, nombre: str) -> list[PartidoHistorico]:
    """Todos los partidos FINALIZADOS de un campeonato, con resultado."""
    out: list[PartidoHistorico] = []
    with httpx.Client(base_url=BASE, timeout=30.0, headers=HEADERS) as c:
        r = c.get(f"/front/campeonatos/{campeonato_id}/fechas")
        r.raise_for_status()
        fechas = r.json().get("data", [])
        fechas.sort(key=lambda f: int(f["nombre"].split()[-1]))

        for f in fechas:
            r = c.get(f"/front/campeonatos/fechas/{f['id']}/eventos")
            r.raise_for_status()
            for e in r.json().get("data", []):
                real = resultado_finalizado(e)
                if real is None:
                    continue  # no jugado / sin cargar / en juego
                gl, gv = real
                out.append(PartidoHistorico(
                    campeonato_id=campeonato_id,
                    campeonato=nombre,
                    fecha_nombre=f["nombre"],
                    fecha_id=f["id"],
                    evento_id=e["id"],
                    local=e["equipoLocal"]["nombre"],
                    visitante=e["equipoVisitante"]["nombre"],
                    goles_local=int(gl),
                    goles_visitante=int(gv),
                    preferencial=bool(e.get("preferencial", False)),
                    inicio_utc=_parse_dt(e["fechaInicio"]).isoformat(),
                ))
    log.info("%s: %d partidos con resultado", nombre, len(out))
    return out


def load_dataset(path: Path | None = None) -> list[PartidoHistorico]:
    """Lee el dataset ya descargado desde disco."""
    path = path or (DATA_DIR / "primera_uy_historico.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [PartidoHistorico(**d) for d in raw]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--camp", type=int, help="solo este campeonato_id")
    args = ap.parse_args()

    temporadas = {args.camp: TEMPORADAS[args.camp]} if args.camp else TEMPORADAS

    todos: list[PartidoHistorico] = []
    for cid, nombre in temporadas.items():
        todos.extend(fetch_temporada(cid, nombre))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "primera_uy_historico.json"
    out.write_text(
        json.dumps([asdict(p) for p in todos], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"escrito {out} — {len(todos)} partidos de {len(temporadas)} temporadas")


if __name__ == "__main__":
    main()

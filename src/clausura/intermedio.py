"""Ingesta del Torneo Intermedio 2026 desde Wikipedia.

El penca-api NO lista el Intermedio (salta del Apertura 2026 al Mundial), pero es
la señal más fresca de fuerza de los equipos antes del Clausura: 7 fechas + final,
jugadas mayo-agosto 2026. Sin estos partidos, los ratings quedan con datos hasta
el 2026-05-11 y castigan a los equipos que levantaron en el Intermedio.

Fuente: wikitext del artículo "Torneo Intermedio 2026" (API de MediaWiki). El
fixture viene en tablas por fecha con filas `local | X:Y | visitante`. Se parsea
solo lo jugado (filas con marcador). Los nombres se mapean a los del penca-api
(ej. "Montevideo Wanderers" → "Wanderers").

Salida: data/processed/intermedio_2026.json con el mismo shape que el histórico;
`load_dataset_completo()` devuelve histórico + intermedio para el fit de ratings.

Uso:
    python -m src.clausura.intermedio             # baja, parsea y guarda
    python -m src.clausura.intermedio --dry-run   # imprime sin escribir
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.clausura.historical import DATA_DIR, PartidoHistorico, load_dataset

log = logging.getLogger(__name__)

WIKI_API = "https://es.wikipedia.org/w/api.php"
PAGE = "Torneo Intermedio 2026"
OUT_PATH = DATA_DIR / "intermedio_2026.json"

CAMPEONATO = "Torneo Intermedio 2026"
CAMPEONATO_ID = -2026  # sintético: no existe en el penca-api

# Wikipedia → nombre del penca-api
ALIASES = {
    "montevideo wanderers": "Wanderers",
    "wanderers": "Wanderers",
    "defensor": "Defensor Sporting",
}

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
         "julio": 7, "agosto": 8, "setiembre": 9, "septiembre": 9, "octubre": 10,
         "noviembre": 11, "diciembre": 12}

# goles acotados a un dígito: una hora ("21:00" en la fila de un partido no jugado)
# también matchearía \d+:\d+ y entraría como 21-0 a los ratings
_SCORE_RE = re.compile(r"^(\d)\s*[:]\s*(\d)$")
_DIA_RE = re.compile(r"^(\d{1,2}) de ([a-zñ]+)$")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def _clean_cell(raw: str) -> str:
    """Celda wikitext → texto plano: saca estilos, negritas, wikilinks y archivos."""
    s = raw
    s = re.sub(r"rowspan=\d+\s*\|", "", s)
    s = re.sub(r"bgcolor=#?\w+\s*\|", "", s)
    s = re.sub(r"align=\"?\w+\"?\s*\|", "", s)
    s = re.sub(r"\[\[Archivo:[^\]]*\]\]", "", s)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", s)
    s = s.replace("'''", "").replace("''", "")
    return s.strip()


def fetch_wikitext(page: str = PAGE) -> str:
    r = httpx.get(WIKI_API, params={
        "action": "parse", "page": page, "prop": "wikitext",
        "format": "json", "formatversion": 2,
    }, timeout=30.0, headers={"User-Agent": "penca-clausura/1.0"})
    r.raise_for_status()
    return r.json()["parse"]["wikitext"]


def parse_partidos(
    wikitext: str,
    equipos_validos: set[str],
    year: int = 2026,
) -> list[PartidoHistorico]:
    """Partidos jugados del fixture. Pura (testeable con el wikitext como fixture).

    Recorre las tablas del fixture; en cada fila busca la celda con marcador X:Y y
    toma la celda anterior como local y la siguiente como visitante. Mapea nombres
    contra `equipos_validos` (los del penca-api) vía ALIASES; si un nombre no
    mapea, descarta la fila y avisa (mejor menos datos que datos mal atribuidos).
    """
    idx_por_norm = {_norm(e): e for e in equipos_validos}
    for alias, real in ALIASES.items():
        if real in equipos_validos:
            idx_por_norm[alias] = real

    out: list[PartidoHistorico] = []
    fecha_actual = "?"
    dia_actual: str | None = None
    seq = 0
    # el fixture arranca en la sección "== Fixture ==" — las tablas de posiciones
    # de arriba también tienen filas con |, pero no tienen celdas marcador X:Y
    for linea_tabla in re.split(r"\{\|", wikitext):
        m_titulo = re.search(r"\|(Fecha \d+|Final)\s*\n", linea_tabla)
        titulo = m_titulo.group(1) if m_titulo else None
        filas = re.split(r"\|-", linea_tabla)
        for fila in filas:
            celdas = [_clean_cell(c) for c in fila.split("\n|") if _clean_cell(c)]
            # rastrear "N de mes" en cualquier celda de la fila
            for c in celdas:
                if _DIA_RE.match(_norm(c)):
                    dia_actual = _norm(c)
            score_i = next((i for i, c in enumerate(celdas) if _SCORE_RE.match(c)), None)
            if score_i is None or score_i == 0 or score_i + 1 >= len(celdas):
                continue
            gl, gv = map(int, _SCORE_RE.match(celdas[score_i]).groups())
            local_n = _norm(celdas[score_i - 1])
            visita_n = _norm(celdas[score_i + 1])
            local = idx_por_norm.get(local_n)
            visita = idx_por_norm.get(visita_n)
            if local is None or visita is None:
                log.warning("fila descartada, equipo desconocido: %r vs %r",
                            celdas[score_i - 1], celdas[score_i + 1])
                continue
            inicio = f"{year}-07-01T00:00:00+00:00"
            if dia_actual:
                m = _DIA_RE.match(dia_actual)
                if m and m.group(2) in MESES:
                    inicio = datetime(year, MESES[m.group(2)], int(m.group(1)),
                                      tzinfo=timezone.utc).isoformat()
            seq += 1
            out.append(PartidoHistorico(
                campeonato_id=CAMPEONATO_ID,
                campeonato=CAMPEONATO,
                fecha_nombre=titulo or fecha_actual,
                fecha_id=-seq,
                evento_id=-seq,
                local=local,
                visitante=visita,
                goles_local=gl,
                goles_visitante=gv,
                preferencial=False,
                inicio_utc=inicio,
            ))
        if titulo:
            fecha_actual = titulo
    return out


def load_dataset_completo() -> list[PartidoHistorico]:
    """Histórico del penca-api + Intermedio 2026 (si fue ingerido)."""
    base = load_dataset()
    if OUT_PATH.exists():
        extra = [PartidoHistorico(**d)
                 for d in json.loads(OUT_PATH.read_text(encoding="utf-8"))]
        base = base + extra
    return base


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="imprime sin escribir")
    args = ap.parse_args()

    import yaml
    from src.clausura.picks import CONFIG_PATH
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    equipos = set(cfg["equipos"].values())

    partidos = parse_partidos(fetch_wikitext(), equipos)
    print(f"{len(partidos)} partidos del Intermedio parseados")
    for p in partidos[-4:]:
        print(f"  {p.fecha_nombre}: {p.local} {p.goles_local}-{p.goles_visitante} "
              f"{p.visitante} ({p.inicio_utc[:10]})")
    if not args.dry_run:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps([asdict(p) for p in partidos], ensure_ascii=False),
                            encoding="utf-8")
        print(f"guardado {OUT_PATH}")


if __name__ == "__main__":
    main()

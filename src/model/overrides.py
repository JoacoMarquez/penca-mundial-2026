"""Overrides manuales de disponibilidad de jugadores para la Capa 4.

Lee `config/player_overrides.yaml` y resuelve, por código de equipo, qué jugadores
están forzados como disponibles (ignorar reportes de baja) o como ausentes.

Motivación: la Capa 4 (cualitativa) lee titulares de Google News y, sin XI confirmado,
a veces recorta λ por una "baja" que en realidad no existe (caso Arda Güler, 2026-06-13:
varios medios lo dieron de baja vs Australia y jugó igual). Este override deja al usuario
corregir el falso positivo sin tocar código y sobrevive a re-runs / al eco de la prensa.
"""

from __future__ import annotations

import logging
import unicodedata
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "player_overrides.yaml"


def _normalize(s: str) -> str:
    """Casefold + sin acentos, para matchear 'Güler' contra 'Guler' en textos de prensa."""
    nfkd = unicodedata.normalize("NFKD", s)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_accents.casefold().strip()


@lru_cache(maxsize=1)
def load_player_overrides(path: str | None = None) -> dict:
    """Carga el YAML de overrides. Cacheado; tolerante a archivo ausente o malformado.

    Devuelve {"available": {CODE: [names]}, "unavailable": {CODE: [names]}}.
    """
    import yaml

    p = Path(path) if path else _CONFIG_PATH
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except FileNotFoundError:
        return {"available": {}, "unavailable": {}}
    except Exception as e:
        log.warning("player_overrides.yaml malformado (%s) — sin overrides", e)
        return {"available": {}, "unavailable": {}}

    def _clean(section) -> dict:
        out: dict[str, list[str]] = {}
        for code, names in (section or {}).items():
            if not names:
                continue
            out[str(code).upper()] = [str(n).strip() for n in names if str(n).strip()]
        return out

    return {
        "available": _clean(raw.get("available")),
        "unavailable": _clean(raw.get("unavailable")),
    }


def team_overrides(team_code: str | None, path: str | None = None) -> tuple[list[str], list[str]]:
    """(disponibles, ausentes) forzados para un código de equipo. Listas vacías si no hay."""
    if not team_code:
        return [], []
    ov = load_player_overrides(path)
    code = str(team_code).upper()
    return ov["available"].get(code, []), ov["unavailable"].get(code, [])


def filter_available(injuries: list[str] | None, available: list[str]) -> list[str]:
    """Saca de una lista de bajas reportadas a los jugadores forzados como disponibles.

    Matchea por substring normalizado: si una entrada de lesión menciona a un jugador
    'available', se descarta esa entrada entera.
    """
    if not injuries:
        return []
    if not available:
        return list(injuries)
    norm_avail = [_normalize(a) for a in available]
    kept = []
    for entry in injuries:
        ne = _normalize(entry)
        if any(a and a in ne for a in norm_avail):
            log.info("override: descarto baja reportada por jugador disponible → %r", entry)
            continue
        kept.append(entry)
    return kept

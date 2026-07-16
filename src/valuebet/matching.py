"""Matcheo de eventos soft (Supermatch) ↔ sharp (Pinnacle).

Clave compuesta: deporte + hora de inicio (±2h) + nombres de equipos normalizados
(lowercase, sin tildes) con tabla de aliases en config/valuebet.yaml. Los eventos
no matcheados se loguean y NUNCA se sugieren.
"""

from __future__ import annotations

import logging
import unicodedata
from collections import defaultdict
from datetime import datetime

from src.valuebet.types import OddsQuote

log = logging.getLogger(__name__)

MATCH_WINDOW_S = 2 * 3600


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def _canonical(name: str, sport_aliases: dict[str, list[str]]) -> str:
    """Nombre normalizado, resolviendo aliases → nombre canónico (el sharp)."""
    n = norm_name(name)
    for canon, variants in sport_aliases.items():
        if n == norm_name(canon) or n in [norm_name(v) for v in variants]:
            return norm_name(canon)
    return n


def _event_key(q: OddsQuote, aliases: dict[str, dict[str, list[str]]]) -> tuple[str, frozenset[str]]:
    sport_aliases = aliases.get(q.sport, {})
    teams = frozenset(_canonical(t.strip(), sport_aliases) for t in q.event_name.split(" vs "))
    return (q.sport, teams)


def match_events(
    soft: list[OddsQuote],
    sharp: list[OddsQuote],
    aliases: dict[str, dict[str, list[str]]],
) -> list[tuple[OddsQuote, list[OddsQuote]]]:
    """Para cada cuota soft, las cuotas sharp del MISMO evento (todos sus mercados).

    Devuelve [(cuota_soft, cuotas_sharp_del_evento)] solo para matcheados.
    """
    sharp_by_key: dict[tuple, list[OddsQuote]] = defaultdict(list)
    for q in sharp:
        sharp_by_key[_event_key(q, aliases)].append(q)

    out: list[tuple[OddsQuote, list[OddsQuote]]] = []
    unmatched: set[str] = set()
    for q in soft:
        candidates = sharp_by_key.get(_event_key(q, aliases), [])
        near = [
            c for c in candidates
            if abs((_dt(c.start_utc) - _dt(q.start_utc)).total_seconds()) <= MATCH_WINDOW_S
        ]
        if near:
            out.append((q, near))
        else:
            unmatched.add(f"{q.sport}: {q.event_name} ({q.start_utc})")

    for u in sorted(unmatched):
        log.warning("unmatched soft event (agregar alias en valuebet.yaml?): %s", u)
    return out


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))

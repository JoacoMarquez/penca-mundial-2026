"""Matcheo de eventos soft (Supermatch) ↔ sharp (Pinnacle).

El desafío real: Supermatch nombra en español y con sufijos de club ("Olympique
Marsella", "IK Start", "Chapecoense SC") y a los tenistas como "Apellido, Nombre";
Pinnacle usa inglés, sin sufijos y "Nombre Apellido". Emparejar por hora es inútil
(hay decenas de partidos simultáneos), así que el matcheo es por NOMBRE con:

  1. Normalización: minúsculas, sin tildes, sin puntuación.
  2. Alias español↔inglés por deporte (config valuebet.yaml, ampliable desde los
     logs de "unmatched").
  3. Stripping de sufijos de club (FC, SC, BK, IL, ...) antes de comparar tokens.
  4. Similitud por conjunto de tokens (subconjunto → 1.0; si no, Jaccard con
     respaldo de SequenceMatcher). El formato "Apellido, Nombre" ↔ "Nombre Apellido"
     cae solo porque el conjunto de tokens es el mismo.

SEGURIDAD: un match exige que AMBOS equipos superen el umbral y que la alineación
sea home↔home / away↔away (posicional). Si un fixture viene con local/visitante
invertido entre casas, el score baja y se descarta — nunca se invierte un 1x2, así
que jamás se apuesta al outcome equivocado. Se pierde ese evento, no se arriesga.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher

from src.valuebet.types import OddsQuote

log = logging.getLogger(__name__)

MATCH_WINDOW_S = 2 * 3600
TEAM_SIM_THRESHOLD = 0.80

# Sufijos/ruido de nombres de club a descartar antes de comparar tokens.
CLUB_SUFFIXES = {
    "fc", "cf", "sc", "afc", "ac", "as", "ss", "ssc", "sv", "tsv", "vfl", "vfb",
    "bk", "if", "ik", "fk", "cd", "ud", "rc", "ca", "club", "bois", "boi", "ec",
    "sk", "aif", "ff", "il", "gk", "kb", "cfr", "asc", "bbc", "cc", "sad", "aa",
    "de", "do", "the",
}


def norm_name(s: str) -> str:
    """Minúsculas, sin tildes, sin puntuación, espacios colapsados."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _alias_to_canonical(sport_aliases: dict[str, list[str]]) -> dict[str, str]:
    """{ variante_normalizada → canónico_normalizado } para un deporte."""
    out: dict[str, str] = {}
    for canon, variants in sport_aliases.items():
        cn = norm_name(canon)
        out[cn] = cn
        for v in variants:
            out[norm_name(v)] = cn
    return out


def _tokens(name: str, alias_map: dict[str, str]) -> set[str]:
    """Tokens significativos del nombre, con alias aplicado y sufijos removidos."""
    n = norm_name(name)
    n = alias_map.get(n, n)  # alias sobre el nombre completo
    raw = [t for t in n.split() if len(t) > 1]
    core = [t for t in raw if t not in CLUB_SUFFIXES]
    return set(core) if core else set(raw)


def team_similarity(a: str, b: str, alias_map: dict[str, str]) -> float:
    """Similitud [0,1] entre dos nombres de equipo/jugador."""
    ta, tb = _tokens(a, alias_map), _tokens(b, alias_map)
    if not ta or not tb:
        return 0.0
    if ta <= tb or tb <= ta:          # uno es subconjunto del otro
        return 1.0
    jaccard = len(ta & tb) / len(ta | tb)
    return max(jaccard, SequenceMatcher(None, norm_name(a), norm_name(b)).ratio())


def _split_teams(event_name: str) -> tuple[str, str] | None:
    parts = event_name.split(" vs ")
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _group_events(quotes: list[OddsQuote]) -> dict[str, dict]:
    """event_id → {sport, teams, start, quotes}."""
    ev: dict[str, dict] = {}
    for q in quotes:
        e = ev.get(q.event_id)
        if e is None:
            teams = _split_teams(q.event_name)
            if teams is None:
                continue
            ev[q.event_id] = {"sport": q.sport, "home": teams[0], "away": teams[1],
                              "start": q.start_utc, "quotes": [q]}
        else:
            e["quotes"].append(q)
    return ev


def match_events(
    soft: list[OddsQuote],
    sharp: list[OddsQuote],
    aliases: dict[str, dict[str, list[str]]],
) -> list[tuple[OddsQuote, list[OddsQuote]]]:
    """Para cada cuota soft de un evento matcheado, las cuotas sharp de ese evento.

    Devuelve [(cuota_soft, cuotas_sharp_del_evento)]. Alineación home↔home garantizada,
    así que soft.outcome 'home' se compara contra sharp 'home' sin remapear.
    """
    soft_ev = _group_events(soft)
    sharp_ev = _group_events(sharp)

    # indexar sharp por deporte para acotar la búsqueda
    sharp_by_sport: dict[str, list[dict]] = defaultdict(list)
    for e in sharp_ev.values():
        sharp_by_sport[e["sport"]].append(e)

    out: list[tuple[OddsQuote, list[OddsQuote]]] = []
    unmatched: set[str] = set()

    for se in soft_ev.values():
        alias_map = _alias_to_canonical(aliases.get(se["sport"], {}))
        best, best_score = None, 0.0
        for pe in sharp_by_sport.get(se["sport"], []):
            if abs((_dt(se["start"]) - _dt(pe["start"])).total_seconds()) > MATCH_WINDOW_S:
                continue
            score = min(
                team_similarity(se["home"], pe["home"], alias_map),
                team_similarity(se["away"], pe["away"], alias_map),
            )
            if score > best_score:
                best_score, best = score, pe

        if best is not None and best_score >= TEAM_SIM_THRESHOLD:
            for sq in se["quotes"]:
                out.append((sq, best["quotes"]))
        else:
            unmatched.add(f"{se['sport']}: {se['home']} vs {se['away']} ({se['start']})")

    for u in sorted(unmatched):
        log.warning("unmatched soft event (agregar alias en valuebet.yaml?): %s", u)
    log.info("match_events: %d/%d eventos soft matcheados",
             len(soft_ev) - len(unmatched), len(soft_ev))
    return out

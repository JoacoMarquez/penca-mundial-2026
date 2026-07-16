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

import json
import logging
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from src.valuebet.types import OddsQuote

log = logging.getLogger(__name__)

MATCH_WINDOW_S = 2 * 3600
TEAM_SIM_THRESHOLD = 0.80
TOKEN_EQ = 0.85          # dos tokens cuentan como el mismo si se parecen ≥ esto
AMBIGUITY_MARGIN = 0.10  # si el 2º candidato también supera el umbral y está a menos
                         # de esto del 1º, el evento se descarta (mejor perderlo que
                         # apostar al partido equivocado)

# Sufijos/ruido de nombres de club a descartar antes de comparar tokens.
CLUB_SUFFIXES = {
    "fc", "cf", "sc", "afc", "ac", "as", "ss", "ssc", "sv", "tsv", "vfl", "vfb",
    "bk", "if", "ik", "fk", "cd", "ud", "rc", "ca", "club", "bois", "boi", "ec",
    "sk", "aif", "ff", "il", "gk", "kb", "cfr", "asc", "bbc", "cc", "sad", "aa",
    "de", "do", "the",
}

# Calificadores de plantel: distinguen al equipo reserva/juvenil/femenino del primer
# equipo. NUNCA se descartan — "Barcelona B" vs "Barcelona" son equipos DISTINTOS
# aunque compartan todos los demás tokens, y las reservas juegan el mismo día.
SQUAD_QUALIFIERS = {
    "b", "c", "ii", "iii", "u17", "u18", "u19", "u20", "u21", "u23",
    "w", "women", "fem", "femenino", "femenil", "ladies",
    "reserve", "reserves", "res", "amateur", "youth",
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


def _qualifiers(name: str, alias_map: dict[str, str]) -> set[str]:
    """Calificadores de plantel presentes en el nombre (post-alias)."""
    n = norm_name(name)
    n = alias_map.get(n, n)
    return {t for t in n.split() if t in SQUAD_QUALIFIERS}


def _token_eq(t: str, u: str) -> bool:
    """Dos tokens son 'el mismo' si son iguales o muy parecidos (variantes de ortografía)."""
    return t == u or SequenceMatcher(None, t, u).ratio() >= TOKEN_EQ


def team_similarity(a: str, b: str, alias_map: dict[str, str]) -> float:
    """Similitud [0,1] por conjunto de tokens con igualdad DIFUSA.

    Usa Jaccard difuso (no SequenceMatcher sobre el string entero): así los tokens
    distintivos tienen que coincidir. Evita el falso positivo de dos equipos de la
    misma ciudad ('Gigantes San Francisco' vs 'Indios de San Francisco'), pero tolera
    variantes de ortografía a nivel token ('Ljungskille' ↔ 'Ljungskile').

    Los calificadores de plantel deben coincidir EXACTO en ambos lados: 'Barcelona B'
    jamás matchea 'Barcelona' (equipos distintos que juegan el mismo día).
    """
    if _qualifiers(a, alias_map) != _qualifiers(b, alias_map):
        return 0.0
    ta, tb = _tokens(a, alias_map), _tokens(b, alias_map)
    if not ta or not tb:
        return 0.0
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    # subconjunto difuso: cada token del más chico tiene contraparte en el más grande
    if all(any(_token_eq(t, u) for u in big) for t in small):
        return 1.0
    # Jaccard difuso: intersección = tokens con contraparte
    inter = sum(1 for t in small if any(_token_eq(t, u) for u in big))
    union = len(ta) + len(tb) - inter
    return inter / union if union else 0.0


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
    dump_dir: Path | None = None,
) -> list[tuple[OddsQuote, list[OddsQuote]]]:
    """Para cada cuota soft de un evento matcheado, las cuotas sharp de ese evento.

    Devuelve [(cuota_soft, cuotas_sharp_del_evento)]. Alineación home↔home garantizada,
    así que soft.outcome 'home' se compara contra sharp 'home' sin remapear.

    Si dump_dir se pasa, los eventos sin match se appendean a
    {dump_dir}/{fecha}.jsonl para minar aliases en batch (además del warning en logs).
    """
    soft_ev = _group_events(soft)
    sharp_ev = _group_events(sharp)

    # indexar sharp por deporte para acotar la búsqueda
    sharp_by_sport: dict[str, list[dict]] = defaultdict(list)
    for e in sharp_ev.values():
        sharp_by_sport[e["sport"]].append(e)

    out: list[tuple[OddsQuote, list[OddsQuote]]] = []
    unmatched: list[dict] = []

    for se in soft_ev.values():
        alias_map = _alias_to_canonical(aliases.get(se["sport"], {}))
        best, best_score, second_score = None, 0.0, 0.0
        for pe in sharp_by_sport.get(se["sport"], []):
            if abs((_dt(se["start"]) - _dt(pe["start"])).total_seconds()) > MATCH_WINDOW_S:
                continue
            score = min(
                team_similarity(se["home"], pe["home"], alias_map),
                team_similarity(se["away"], pe["away"], alias_map),
            )
            if score > best_score:
                best_score, second_score, best = score, best_score, pe
            elif score > second_score:
                second_score = score

        ambiguous = (second_score >= TEAM_SIM_THRESHOLD
                     and best_score - second_score < AMBIGUITY_MARGIN)
        if best is not None and best_score >= TEAM_SIM_THRESHOLD and not ambiguous:
            for sq in se["quotes"]:
                out.append((sq, best["quotes"]))
        else:
            unmatched.append({
                "sport": se["sport"], "home": se["home"], "away": se["away"],
                "start_utc": se["start"], "best_score": round(best_score, 3),
                "reason": "ambiguous" if ambiguous else "below_threshold",
                "best_sharp": best["home"] + " vs " + best["away"] if best else None,
            })

    for u in sorted(unmatched, key=lambda x: (x["sport"], x["home"])):
        if u["reason"] == "ambiguous":
            log.warning("match ambiguo descartado (dos candidatos sharp ≥ umbral): "
                        "%s: %s vs %s", u["sport"], u["home"], u["away"])
        else:
            log.warning("unmatched soft event (agregar alias en valuebet.yaml?): "
                        "%s: %s vs %s (%s)", u["sport"], u["home"], u["away"], u["start_utc"])
    if dump_dir is not None and unmatched:
        _dump_unmatched(unmatched, dump_dir)
    log.info("match_events: %d/%d eventos soft matcheados",
             len(soft_ev) - len(unmatched), len(soft_ev))
    return out


def _dump_unmatched(unmatched: list[dict], dump_dir: Path) -> None:
    """Appendea los unmatched del scan a {dump_dir}/{fecha}.jsonl. Best-effort."""
    now = datetime.now(timezone.utc)
    path = dump_dir / f"{now.strftime('%Y-%m-%d')}.jsonl"
    try:
        dump_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for u in unmatched:
                f.write(json.dumps({"ts_utc": now.isoformat(), **u}, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("no pude volcar unmatched a %s: %s", path, e)

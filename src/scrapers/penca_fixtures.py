"""Sync de config/fixtures.yaml desde la API de la penca.

Reemplaza el fixtures.yaml manual con los datos canónicos del backend de la penca:
    - match IDs reales (integers, 105..208) en vez de strings (MATCH_A_01)
    - kickoff_utc autoritativo
    - estado actual (pending/live/finished) y resultado real cuando aplica
    - placeholders de eliminatorias hasta que se resuelven

Uso:
    python -m src.scrapers.penca_fixtures              # sync (lee API, escribe fixtures.yaml)
    python -m src.scrapers.penca_fixtures --dry-run    # solo muestra cambios

NOTA: usa la PENCA_API_KEY del .env. El endpoint /matches funciona aunque /me/* esté roto
(el bug del admin no afecta este endpoint).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import httpx
import yaml

log = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_PATH = PROJECT_ROOT / "config" / "fixtures.yaml"
TEAMS_PATH = PROJECT_ROOT / "config" / "teams.yaml"


def fetch_matches() -> list[dict]:
    """GET /api/v1/matches con bearer auth. Retorna lista cruda."""
    base = os.environ["PENCA_API_BASE_URL"].rstrip("/")
    key = os.environ["PENCA_API_KEY"]
    with httpx.Client(timeout=15.0, headers={"Authorization": f"Bearer {key}"}) as client:
        resp = client.get(f"{base}/matches")
        resp.raise_for_status()
    data = resp.json()
    return data.get("matches") if isinstance(data, dict) else data


def build_name_to_code(teams_yaml: dict) -> dict[str, str]:
    """Inverso de aliases: nombre (cualquier idioma) → código FIFA."""
    aliases = teams_yaml.get("aliases", {})
    out: dict[str, str] = {}
    for code, names in aliases.items():
        for n in names:
            out[_norm(n)] = code
        out[_norm(code)] = code
    return out


def _norm(s: str) -> str:
    """Normalización: lower + strip espacios + sin acentos básicos."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


# Aliases extra para nombres en español que la API usa pero no estaban en teams.yaml
EXTRA_ALIASES = {
    "republica checa": "CZE",   # API usa "República Checa" en vez de "Czechia"
    "estados unidos": "USA",
    "turquia": "TUR",
    "iran": "IRN",
    "arabia saudi": "KSA",
    "republica democratica del congo": "COD",
    "rd del congo": "COD",
    "uzbekistan": "UZB",
    "argelia": "ALG",
    "jordania": "JOR",
    "noruega": "NOR",
    "polonia": "POL",   # ojo: Polonia no clasificó al Mundial; por si aparece en otros endpoints
}


def map_team(team_obj: dict, name_to_code: dict[str, str]) -> tuple[str | None, str]:
    """Mapea {"id": X, "name": "México"} → (FIFA_code, display_name)."""
    name = team_obj.get("name", "")
    nname = _norm(name)
    code = name_to_code.get(nname) or EXTRA_ALIASES.get(nname)
    if not code:
        log.warning("No se encontró código para equipo: %r (normalized: %r)", name, nname)
    return code, name


def convert_match(m: dict, name_to_code: dict[str, str]) -> dict:
    home_code, home_name = (map_team(m["home_team"], name_to_code) if m.get("home_team") else (None, None))
    away_code, away_name = (map_team(m["away_team"], name_to_code) if m.get("away_team") else (None, None))

    out: dict = {
        "id": m["id"],   # ID de la API (int)
        "stage": m["stage"],
        "kickoff_utc": m["kickoff_utc"],
        "status": m.get("status", "pending"),
    }
    if m.get("group"):
        out["group"] = m["group"]
    if home_code:
        out["home"] = home_code
    if home_name:
        out["home_name"] = home_name
    if away_code:
        out["away"] = away_code
    if away_name:
        out["away_name"] = away_name
    # Placeholders de eliminatorias
    if m.get("home_placeholder"):
        out["home_placeholder"] = m["home_placeholder"]
    if m.get("away_placeholder"):
        out["away_placeholder"] = m["away_placeholder"]
    if m.get("venue"):
        out["venue"] = m["venue"]
    # Resultados si terminó
    if m.get("home_score") is not None:
        out["home_score"] = m["home_score"]
    if m.get("away_score") is not None:
        out["away_score"] = m["away_score"]
    return out


def sync(dry_run: bool = False) -> None:
    teams_yaml = yaml.safe_load(TEAMS_PATH.read_text())
    name_to_code = build_name_to_code(teams_yaml)

    matches_raw = fetch_matches()
    log.info("API devolvió %d partidos", len(matches_raw))

    fase_grupos = []
    eliminatorias = []
    unmapped: list[str] = []

    for m in matches_raw:
        converted = convert_match(m, name_to_code)
        if m["stage"] == "group":
            if not converted.get("home") or not converted.get("away"):
                unmapped.append(f"  ⚠️ {m['id']} ({m.get('home_team', {}).get('name')} vs {m.get('away_team', {}).get('name')})")
            fase_grupos.append(converted)
        else:
            eliminatorias.append(converted)

    fase_grupos.sort(key=lambda d: d["kickoff_utc"])
    eliminatorias.sort(key=lambda d: d["kickoff_utc"])

    print(f"→ fase_grupos: {len(fase_grupos)}, eliminatorias: {len(eliminatorias)}")
    if unmapped:
        print("Equipos NO mapeados a código FIFA:")
        for u in unmapped:
            print(u)

    # Preservar la sección `config` del fixtures.yaml actual
    existing = yaml.safe_load(FIXTURES_PATH.read_text()) if FIXTURES_PATH.exists() else {}
    config_section = existing.get("config", {
        "timezone_display": "America/Montevideo",
        "kickoff_buffer_minutes": 5,
        "scheduling_anchors": {
            "T_72h_minutes": 4320,
            "T_24h_minutes": 1440,
            "T_3h_minutes": 180,
            "T_30min_minutes": 30,
        },
    })

    output = {
        "fase_grupos": fase_grupos,
        "eliminatorias": eliminatorias,
        "config": config_section,
    }
    yaml_text = "# Sync automático desde /api/v1/matches.\n# Para regenerar: python -m src.scrapers.penca_fixtures\n\n"
    yaml_text += yaml.dump(output, sort_keys=False, allow_unicode=True, default_flow_style=False)

    if dry_run:
        print("\n=== DRY RUN — no escribo. Primeros 1500 chars del output: ===")
        print(yaml_text[:1500])
    else:
        FIXTURES_PATH.write_text(yaml_text)
        print(f"\n✅ Escrito {FIXTURES_PATH} ({len(yaml_text)} chars)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sync(dry_run=args.dry_run)

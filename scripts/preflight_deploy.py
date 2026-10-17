"""Pre-check de deploy: ¿hay algún partido en ventana de publicación AHORA?

El deploy en sí es de bajo riesgo (el scheduler es un timer→oneshot: cada tick es un
proceso fresco, así que el `git pull` toma efecto solo en el próximo tick y reiniciar el
`.timer` no mata un `.service` corriendo). Aun así, como cortesía, no reiniciamos el timer
justo cuando un partido está por publicarse.

Sale 0 (OK, deployá) si ninguna ventana está activa; 1 (esperá) si un partido cae dentro
de [kickoff − PRE_MIN, kickoff + POST_MIN]. Si no puede leer fixtures, permite el deploy
(fail-open): el peor caso es reiniciar el timer durante una publicación, y eso es inocuo.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import yaml

# Ventana de "no tocar": desde 40 min antes del kickoff (cubre la pasada T-30min que
# publica + margen del timer de 5 min) hasta 5 min después del kickoff.
PRE_MIN = 40
POST_MIN = 5

FIXTURES_PATH = os.environ.get("FIXTURES_PATH", "config/fixtures.yaml")


def blocked_match(
    matches: list[dict], now: datetime, pre_min: int = PRE_MIN, post_min: int = POST_MIN,
) -> tuple[str, str] | None:
    """Primer partido (id, kickoff_utc) cuya ventana de publicación contiene `now`, o None."""
    for m in matches:
        ko_raw = m.get("kickoff_utc")
        if not ko_raw:
            continue
        try:
            ko = datetime.fromisoformat(str(ko_raw).replace("Z", "+00:00"))
        except Exception:
            continue
        if ko - timedelta(minutes=pre_min) <= now <= ko + timedelta(minutes=post_min):
            return str(m.get("id", "?")), str(ko_raw)
    return None


def main() -> int:
    try:
        with open(FIXTURES_PATH) as f:
            fx = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"WARN: no pude leer {FIXTURES_PATH} ({e}); permito el deploy (fail-open)")
        return 0
    now = datetime.now(timezone.utc)
    matches = (fx.get("fase_grupos") or []) + (fx.get("eliminatorias") or [])
    hit = blocked_match(matches, now)
    if hit:
        print(f"BLOCK: {hit[0]} kickoff {hit[1]} en ventana [-{PRE_MIN}m,+{POST_MIN}m] (now={now.isoformat()})")
        return 1
    print(f"OK: ninguna ventana de publicación activa (now={now.isoformat()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

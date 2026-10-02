"""Dataset autocontenido para re-simular el Clausura con los picks REALES del pool.

Corre en el VPS (resultados del API, cuotas versionadas, planillas y snapshot viven
ahí) y deja un JSON que alcanza para todos los experimentos de scripts/replay_clausura.py
sin tocar la red. Por partido:

  * resultado real (si se jugó)
  * grilla de MERCADO con las cuotas del último snapshot antes del cierre (la verdad
    para sortear resultados: calibra, z=−0,6 en 32 partidos — memoria goleada_corta)
  * grilla de PRODUCCIÓN (grilla_shadow.base de la última planilla generada antes del
    cierre): lo que el sistema creía, para que las políticas decidan con su información
  * temperatura del pool de esa planilla

Por participación: picks reales, especiales y puntos del ranking.

    PYTHONPATH=/opt/penca python -m scripts.replay_dataset --out data/replay/clausura.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from src.clausura.picks import PRED_DIR, flat_eventos, load_config
from src.clausura.pool_snapshot import load_latest_snapshot
from src.clausura.postmortem import resultados_de_fecha
from src.clausura.rivals import mis_numeros_env
from scripts.premios_fecha_calibracion import grilla_pre_cierre, odds_snapshots

log = logging.getLogger(__name__)


def planillas(fecha_n: int) -> list[dict]:
    d = PRED_DIR / f"fecha_{fecha_n:02d}"
    out = []
    for p in d.glob("v*_*.json") if d.exists() else []:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            log.warning("planilla ilegible: %s", p)
    return sorted(out, key=lambda x: x["generado_utc"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/replay/clausura.json")
    a = ap.parse_args()

    cfg = load_config()
    snaps = odds_snapshots()
    snap = load_latest_snapshot()
    # el ORDEN es el de las filas de la planilla (columna i ↔ i-ésimo número del env)
    import os
    mis = [int(x) for x in os.environ.get("CLAUSURA_MIS_PARTICIPACIONES", "").split(",")
           if x.strip().isdigit()] or sorted(mis_numeros_env())
    eventos = []
    for n in sorted({e["fecha_n"] for e in flat_eventos(cfg)}):
        evs = [e for e in flat_eventos(cfg) if e["fecha_n"] == n]
        res = resultados_de_fecha(cfg, n)
        res = res[0] if res else {}
        pls = planillas(n)
        for e in evs:
            cierre = datetime.fromisoformat(e["cierre_pronostico_utc"])
            g_m = grilla_pre_cierre(e, snaps)
            prev = [p for p in pls if datetime.fromisoformat(p["generado_utc"]) < cierre]
            fila = None
            for p in reversed(prev):
                fila = next((x for x in p["picks"] if x["evento_id"] == e["evento_id"]), None)
                if fila is not None:
                    temp = p.get("pool", {}).get("temperatura")
                    break
            eventos.append({
                "evento_id": e["evento_id"], "fecha_n": n, "fecha_id": e["fecha_id"],
                "local": e["local"], "visitante": e["visitante"],
                "preferencial": bool(e["preferencial"]), "cierre_utc": e["cierre_pronostico_utc"],
                "resultado": list(res[e["evento_id"]]) if e["evento_id"] in res else None,
                "grid_mercado": [float(x) for x in g_m] if g_m is not None else None,
                "grid_prod": ((fila or {}).get("grilla_shadow") or {}).get("base"),
                "temperatura_pool": temp if fila else None,
                "planilla_picks": (fila or {}).get("scores"),
            })
            log.info("F%d %s-%s: res %s · mercado %s · prod %s", n, e["local"], e["visitante"],
                     eventos[-1]["resultado"], g_m is not None, fila is not None)
    parts = [{"numero": p["numero"], "puntos": p.get("puntos"), "picks": p["picks"],
              "campeon": p.get("campeon"), "goleador": p.get("goleador")}
             for p in snap["participaciones"]]
    out = {"generado_snapshot": snap["generado_utc"], "mis_numeros": mis,
           "eventos": eventos, "participaciones": parts}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    jug = sum(e["resultado"] is not None for e in eventos)
    print(f"OK {a.out}: {len(eventos)} eventos ({jug} jugados, "
          f"{sum(e['grid_mercado'] is not None for e in eventos)} con mercado), "
          f"{len(parts)} participaciones")


if __name__ == "__main__":
    main()

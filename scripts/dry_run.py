"""Pre-Mundial dry-run: corre la pipeline T-24h sobre un partido sin publicar.

Uso:
    python -m scripts.dry_run --match-id 1234
    python -m scripts.dry_run --next                # primer partido futuro de fixtures
    python -m scripts.dry_run --next --phase T_3h   # otra fase

Fuerza DRY_RUN=true durante la corrida (restaura el valor previo al salir), por
lo que el publisher nunca toca la API de la penca. Las notificaciones de Telegram
sí se envían (es parte de lo que se valida).

Output: resumen en stdout con los 5 picks, métricas del modelo, capas del dossier,
costo LLM estimado, y la ruta del JSON versionado generado.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _pick_next_match(fixtures: dict) -> dict | None:
    now = datetime.now(timezone.utc)
    upcoming = []
    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or []) + (fixtures.get("matches") or [])
    for m in all_matches:
        ko = m.get("kickoff_utc") or m.get("kickoff")
        if not ko:
            continue
        try:
            dt = datetime.fromisoformat(str(ko).replace("Z", "+00:00"))
        except Exception:
            continue
        if dt > now:
            upcoming.append((dt, m))
    if not upcoming:
        return None
    upcoming.sort(key=lambda x: x[0])
    return upcoming[0][1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run de la pipeline (no publica).")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--match-id", help="ID de partido en fixtures.yaml")
    g.add_argument("--next", action="store_true", help="Primer partido futuro de fixtures")
    parser.add_argument("--phase", default="T_24h", choices=["T_24h", "T_3h", "T_30min"])
    parser.add_argument("--no-telegram", action="store_true",
                        help="También suprime notificaciones (TELEGRAM_BOT_TOKEN vacío)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    # Forzar DRY_RUN durante la corrida
    prev_dry = os.environ.get("DRY_RUN")
    os.environ["DRY_RUN"] = "true"
    prev_tg = os.environ.get("TELEGRAM_BOT_TOKEN")
    if args.no_telegram:
        os.environ["TELEGRAM_BOT_TOKEN"] = ""

    try:
        from src.agent.pipeline import Phase, load_fixtures, run_match_pipeline

        if args.match_id:
            match_id = args.match_id
        else:
            fixtures = load_fixtures()
            m = _pick_next_match(fixtures)
            if not m:
                print("No hay partidos futuros en fixtures.yaml", file=sys.stderr)
                return 2
            match_id = str(m.get("id") or m.get("match_id"))

        phase = Phase(args.phase)

        print(f"\n{'='*70}\nDRY RUN | match_id={match_id} | phase={phase.value}\n{'='*70}\n")
        run = run_match_pipeline(match_id, phase)

        d = asdict(run)
        print("\n--- Constraints (modelo) ---")
        c = d.get("constraints", {})
        print(f"  P(home/draw/away) = {c.get('p_home',0):.3f} / {c.get('p_draw',0):.3f} / {c.get('p_away',0):.3f}")
        print(f"  λ_local={c.get('lambda_L'):.3f}  λ_visit={c.get('lambda_V'):.3f}  λ12={c.get('lambda_12'):.3f}")
        print(f"  P(O2.5)={c.get('p_over_2_5',0):.3f}  P(BTTS)={c.get('p_btts',0):.3f}")

        if d.get("odds_anomaly"):
            print("\n--- ⚠️ Odds anomaly detectada ---")
            print(json.dumps(d["odds_anomaly"], indent=2))

        print("\n--- Portfolio (5 picks) ---")
        for p in d.get("portfolio", {}).get("picks", []):
            print(f"  [{p.get('objective'):<14}] {p.get('score')[0]}-{p.get('score')[1]}  "
                  f"E[pts]={p.get('e_points',0):.3f}  Var={p.get('var_points',0):.3f}")

        print("\n--- Asignación por penca ---")
        for a in d.get("assignment", []):
            print(f"  penca={a['penca_id']}  rank={a.get('rank')}  → {a['objective']} {a['score']}")

        meta = d.get("assignment_meta") or {}
        if meta:
            print(f"\n--- Assignment meta ---")
            print(f"  objective={meta.get('objective')}  P(top_k)={meta.get('p_top_k_value')}  cutoff={meta.get('threshold')}")

        qa = d.get("qualitative_adjustment")
        if qa:
            print(f"\n--- Capa 4 (LLM cualitativo) ---")
            print(f"  ΔλL={qa.get('delta_lambda_L'):+.2f}  ΔλV={qa.get('delta_lambda_V'):+.2f}  conf={qa.get('confidence'):.2f}")
            print(f"  reasoning: {(qa.get('reasoning') or '')[:200]}...")

        tc = d.get("tipster_consensus")
        if tc:
            print(f"\n--- Capa 3 (Tipsters) ---")
            print(f"  n={tc.get('n_tipsters')}  consensus_1X2={tc.get('consensus_1x2')}")

        # Última versión escrita
        latest = sorted((PROJECT_ROOT / "data" / "predictions" / match_id).glob("v*.json"))
        if latest:
            print(f"\n📝 JSON versionado: {latest[-1]}")

        print(f"\n{'='*70}\n✅ Dry-run OK (sin publicar a la penca)\n{'='*70}\n")
        return 0

    except Exception as e:
        logging.exception("Dry-run falló")
        print(f"\n❌ Dry-run FAILED: {e}", file=sys.stderr)
        return 1
    finally:
        if prev_dry is None:
            os.environ.pop("DRY_RUN", None)
        else:
            os.environ["DRY_RUN"] = prev_dry
        if args.no_telegram:
            if prev_tg is None:
                os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            else:
                os.environ["TELEGRAM_BOT_TOKEN"] = prev_tg


if __name__ == "__main__":
    sys.exit(main())

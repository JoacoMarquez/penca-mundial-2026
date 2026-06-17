"""Recalcula points_earned + portfolio_total/max de los postmortems con la regla VIGENTE.

Necesario tras el cambio de regla del 2026-06-12 (exacto 5→6, se eliminó el tier de
1 punto por goles de un equipo, y el tier de 4 pasó a 'ganador + diferencia de gol').
Los postmortems generados ANTES del cambio (jornada 1) quedaron con la regla vieja; el
admin recalculó el sitio retroactivamente pero nuestros archivos no. Este script los
realinea con `jmlm_points`. Es idempotente: los ya correctos no cambian.

Uso:
    python -m scripts.recompute_postmortems            # dry run (no escribe)
    python -m scripts.recompute_postmortems --write    # escribe (con backup .bak por archivo)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.model.poisson import jmlm_points
from src.utils.env import get_str


def _winner(h: int, a: int) -> int:
    return 0 if h > a else (1 if h == a else 2)


def main() -> None:
    write = "--write" in sys.argv
    pmdir = Path(get_str("DATA_DIR", "data")) / "postmortems"
    if not pmdir.exists():
        print(f"No existe {pmdir}")
        return

    changed = []
    for f in sorted(pmdir.glob("*.json")):
        if ".bak" in f.name:
            continue
        try:
            pm = json.loads(f.read_text())
        except Exception:
            continue
        ah, aa = pm.get("actual_home"), pm.get("actual_away")
        if ah is None or aa is None:
            continue
        actual = (int(ah), int(aa))

        picks_fixed = 0
        for r in pm.get("pencas_results", []):
            ps = r.get("predicted_score")
            if not ps or ps[0] is None:
                continue
            new_pts = jmlm_points((int(ps[0]), int(ps[1])), actual)
            new_exact = [int(ps[0]), int(ps[1])] == [actual[0], actual[1]]
            new_cw = _winner(int(ps[0]), int(ps[1])) == _winner(actual[0], actual[1])
            if (r.get("points_earned") != new_pts or bool(r.get("is_exact")) != new_exact
                    or bool(r.get("correct_winner")) != new_cw):
                picks_fixed += 1
                r["points_earned"], r["is_exact"], r["correct_winner"] = new_pts, new_exact, new_cw

        results = pm.get("pencas_results", [])
        new_max = max((r.get("points_earned", 0) for r in results), default=0)
        new_tot = sum(r.get("points_earned", 0) for r in results)
        if picks_fixed or pm.get("portfolio_max_points") != new_max or pm.get("portfolio_total_points") != new_tot:
            old_tot = pm.get("portfolio_total_points")
            old_max = pm.get("portfolio_max_points")
            pm["portfolio_max_points"], pm["portfolio_total_points"] = new_max, new_tot
            changed.append((f.name, old_tot, new_tot, old_max, new_max, picks_fixed))
            if write:
                (f.parent / (f.name + ".bak")).write_text(json.dumps(json.loads(f.read_text()), ensure_ascii=False, indent=2))
                f.write_text(json.dumps(pm, ensure_ascii=False, indent=2))

    print(f"{'ESCRITO' if write else 'DRY RUN (sin escribir)'} — {len(changed)} postmortems con cambios:")
    for name, ot, nt, om, nm, d in changed:
        print(f"  {name}: total {ot}->{nt}  max {om}->{nm}  ({d} picks recalculadas)")
    if not write and changed:
        print("\nCorré con --write para aplicar (hace backup .bak de cada uno).")


if __name__ == "__main__":
    main()

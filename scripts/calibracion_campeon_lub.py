"""P(campeón) de la LUB walk-forward sobre la 25/26 — DESCRIPTIVO (n = 1 temporada).

Mismo camino de producción que src.lub.picks.correr:
    ratings   src.lub.model.fit(partidos, hasta=corte, "LUB 25/26", PARAMS_PROD)
    sorteo    src.lub.season.simular(rt, slots regulares, Config, jugados por equipo)
    campeón   np.bincount(so.campeon) / n_sims

con el fixture regular de la 25/26 (lo jugado antes del corte como Slot.resultado) y la
24/25 como historia. Checkpoints: pretemporada, mitad de la regular (después de la
fecha 11), fin de la regular (antes de la liguilla).

"Inicio de playoffs" NO se corre: `simular` sortea liguilla y playoffs SIEMPRE desde
cero (solo los slots regulares aceptan resultado), así que no hay forma de fijar la
liguilla jugada sin reimplementar el simulador — y medir un objeto distinto al de
producción es justo lo que no hay que hacer.

Una temporada es UN campeón: nada de esto es evidencia de calibración.

Uso:
    python scripts/calibracion_campeon_lub.py [--sims 20000]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lub.data import load_temporadas  # noqa: E402
from src.lub.model import PARAMS_PROD, fit  # noqa: E402
from src.lub.season import Config, Slot, simular  # noqa: E402

TEMP = "LUB 25/26"


def campeon_real(partidos) -> str:
    """Ganador de la serie final (la fecha 'Finales' de la temporada)."""
    fin = [p for p in partidos if p.temporada == TEMP and p.fecha_nombre.strip() == "Finales"
           and p.pts_local is not None]
    g = collections.Counter(p.local if p.pts_local > p.pts_visitante else p.visitante for p in fin)
    (eq, w), = g.most_common(1)
    assert w == 4, g            # final al mejor de 7
    return eq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--path", type=pathlib.Path, default=ROOT / "data" / "lub" / "temporadas.json")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "experimentos" /
                    f"calibracion_campeon_lub_{datetime.now():%Y%m%d}.json")
    args = ap.parse_args()

    partidos = load_temporadas(args.path)
    camp = campeon_real(partidos)
    reg = sorted([p for p in partidos if p.temporada == TEMP and p.fase == "regular"],
                 key=lambda p: (int(p.fecha_nombre.split()[-1]), p.inicio_utc))
    t = lambda s: datetime.fromisoformat(s)  # noqa: E731
    cortes = {
        "pretemporada": min(t(p.inicio_utc) for p in reg),
        "mitad_regular (tras F11)": min(t(p.inicio_utc) for p in reg
                                        if int(p.fecha_nombre.split()[-1]) > 11),
        "fin_regular (antes de liguilla)": min(t(p.inicio_utc) for p in partidos
                                               if p.temporada == TEMP and p.fase == "liguilla"),
    }
    filas = []
    for nombre, corte in cortes.items():
        rt = fit(partidos, corte, TEMP, PARAMS_PROD)
        jugados = collections.Counter()
        slots = []
        for p in reg:
            res = None
            if p.pts_local is not None and t(p.inicio_utc) < corte:
                res = (p.pts_local, p.pts_visitante)
                jugados[p.local] += 1
                jugados[p.visitante] += 1
            slots.append(Slot(p.fecha_nombre, "regular", p.local, p.visitante, p.preferencial,
                              p.evento_id, res))
        so = simular(rt, slots, Config(n_sims=args.sims), dict(jugados))
        pc = np.bincount(so.campeon, minlength=len(so.equipos)) / args.sims
        c = so.equipos.index(camp)
        orden = np.argsort(-pc)
        fila = {
            "checkpoint": nombre, "corte_utc": corte.astimezone(timezone.utc).isoformat(),
            "regulares_jugados": int(sum(s.resultado is not None for s in slots)),
            "p_campeon": {so.equipos[i]: round(float(pc[i]), 4) for i in orden},
            "campeon_real": camp, "p_real": float(pc[c]), "rank": int((pc > pc[c]).sum() + 1),
            "logloss": -math.log(max(pc[c], 0.5 / args.sims)),
            "logloss_uniforme": math.log(len(so.equipos)),
            "esperado_si_calibrado": float((pc ** 2).sum()),
        }
        filas.append(fila)
        print(f"{nombre:>32}: P({camp}) = {pc[c]:.3f} (rank {fila['rank']}, "
              f"logloss {fila['logloss']:.2f} vs uniforme {fila['logloss_uniforme']:.2f}, "
              f"Σp² {fila['esperado_si_calibrado']:.3f}) | top: "
              + ", ".join(f"{so.equipos[i]} {pc[i]:.2f}" for i in orden[:4]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "generado_utc": datetime.now(timezone.utc).isoformat(), "sims": args.sims,
        "temporada": TEMP, "campeon_real": camp, "filas": filas,
        "nota": "n = 1 temporada: descriptivo, no evidencia de calibración",
    }, ensure_ascii=False, indent=1))
    print(f"escrito {args.out}")


if __name__ == "__main__":
    main()

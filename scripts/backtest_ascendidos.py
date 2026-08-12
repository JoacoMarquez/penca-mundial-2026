"""¿Los ascendidos/debutantes rinden peor de lo que el modelo cree? NO (medido 12/8).

La premisa de la auditoría era razonable: un equipo sin histórico encoge hacia la
media de la liga (ridge → 0), y "los ascendidos históricamente rinden bajo la
media" — Albion con P(campeón) 4.4% arriba de Racing parecía el síntoma. Pero el
residuo walk-forward dice lo contrario: en su torneo de debut, los debutantes
hacen +0.089 ± 0.091 goles MÁS de lo predicho y reciben −0.082 ± 0.094 MENOS
(156 partidos-equipo, cohortes 2025 y 2026). Un prior de encogimiento hacia
abajo habría empeorado. Sin señal, no hay perilla.

Reproduce el número del registro: python scripts/backtest_ascendidos.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.clausura.intermedio import load_dataset_completo  # noqa: E402
from src.clausura.ratings import fit_ratings  # noqa: E402


def main():
    ps = sorted(load_dataset_completo(), key=lambda p: p.inicio_utc)
    por_camp = {}
    for p in ps:
        por_camp.setdefault(p.campeonato, []).append(p)
    orden = sorted(por_camp, key=lambda c: min(p.inicio_utc for p in por_camp[c]))
    equipos_de = {c: {q.local for q in por_camp[c]} | {q.visitante for q in por_camp[c]}
                  for c in orden}
    anio = {c: min(p.inicio_utc for p in por_camp[c])[:4] for c in orden}
    vistos = {}
    for c in orden:
        vistos.setdefault(anio[c], set()).update(equipos_de[c])
    debut = {}
    for c in orden:
        prev = vistos.get(str(int(anio[c]) - 1), set())
        if prev:
            debut[c] = equipos_de[c] - prev

    res_f, res_c = [], []
    for i, c in enumerate(orden):
        if not debut.get(c):
            continue
        previas = [p for cc in orden[:i] for p in por_camp[cc]]
        if len(previas) < 100:
            continue
        r = fit_ratings(previas)
        for p in por_camp[c]:
            for eq, local in ((p.local, True), (p.visitante, False)):
                if eq not in debut[c]:
                    continue
                ll, lv = r.lambdas(p.local, p.visitante)
                res_f.append((p.goles_local if local else p.goles_visitante)
                             - (ll if local else lv))
                res_c.append((p.goles_visitante if local else p.goles_local)
                             - (lv if local else ll))

    rf, rc = np.array(res_f), np.array(res_c)
    print(f"n={len(rf)} partidos-equipo de debutantes (walk-forward)")
    print(f"goles A FAVOR:  residuo {rf.mean():+.3f} ± {rf.std(ddof=1) / np.sqrt(len(rf)):.3f}")
    print(f"goles EN CONTRA: residuo {rc.mean():+.3f} ± {rc.std(ddof=1) / np.sqrt(len(rc)):.3f}")
    print("residuo = real − λ del modelo fiteado solo con lo anterior.")
    print("Positivo a favor / negativo en contra = rinden MEJOR que lo predicho.")


if __name__ == "__main__":
    main()

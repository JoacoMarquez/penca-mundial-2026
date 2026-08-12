"""¿Conviene ponderar los partidos viejos al ajustar los ratings?

`fit_ratings` suma todos los partidos con peso idéntico: el Apertura 2024 pesa lo mismo
que el Intermedio 2026, en un fútbol donde los planteles rotan cada seis meses.

Pega más de lo que parece porque el Elasticsearch solo publica cuotas de la fecha
próxima — 8 de 120 eventos el 2026-08-11. Los otros 112 partidos de la temporada, más
todo P(campeón), salen 100% de estos ratings.

DISEÑO: walk-forward estricto. Para cada temporada, se ajusta con TODO lo anterior y se
predice esa temporada sin haberla visto. La métrica es la log-verosimilitud Poisson por
partido — cuán probable era el resultado real bajo el λ predicho. No hace falta Monte
Carlo: esto mide calidad de predicción, que es lo que el decay pretende mejorar.

Se reporta también el acierto de 1X2 como métrica secundaria legible, pero la que manda
es la log-verosimilitud: es propia (no se puede engañar prediciendo siempre lo mismo).
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.clausura.intermedio import load_dataset_completo  # noqa: E402
from src.clausura.ratings import fit_ratings  # noqa: E402

VIDAS_MEDIA = (None, 1095.0, 730.0, 545.0, 365.0, 270.0, 180.0, 120.0)


def _ts(p):
    from datetime import datetime
    return datetime.fromisoformat(p.inicio_utc.replace("Z", "+00:00")).timestamp()


def evaluar_por_fecha(partidos, half_life, min_train=120):
    """Como se usa en PRODUCCION: reajustar con todo lo jugado y predecir la fecha
    proxima. Es el horizonte donde el decay deberia rendir mas — predecir la temporada
    entera diluye cualquier ventaja de tener el plantel reciente bien medido."""
    porc = sorted(partidos, key=_ts)
    fechas = []
    for p in porc:
        if not fechas or fechas[-1][0] != (p.campeonato_id, p.fecha_id):
            fechas.append(((p.campeonato_id, p.fecha_id), []))
        fechas[-1][1].append(p)

    from math import lgamma
    ll_tot, aciertos, n = 0.0, 0, 0
    for i in range(1, len(fechas)):
        previas = [p for _, ps in fechas[:i] for p in ps]
        if len(previas) < min_train:
            continue
        r = fit_ratings(previas, half_life_dias=half_life)
        for p in fechas[i][1]:
            ll, lv = r.lambdas(p.local, p.visitante)
            gl, gv = p.goles_local, p.goles_visitante
            ll_tot += (gl * np.log(ll) - ll - lgamma(gl + 1)
                       + gv * np.log(lv) - lv - lgamma(gv + 1))
            pred = "H" if ll > lv else ("A" if lv > ll else "D")
            real = "H" if gl > gv else ("A" if gv > gl else "D")
            aciertos += pred == real
            n += 1
    return ll_tot / n, aciertos / n, n


def evaluar(partidos, half_life, min_train=120):
    """(loglik por partido, acierto 1X2, n) prediciendo cada temporada con las previas."""
    porc = sorted(partidos, key=_ts)
    temporadas = []
    for p in porc:
        if not temporadas or temporadas[-1][0] != p.campeonato_id:
            temporadas.append((p.campeonato_id, []))
        temporadas[-1][1].append(p)

    ll_tot, aciertos, n = 0.0, 0, 0
    for i in range(1, len(temporadas)):
        previas = [p for _, ps in temporadas[:i] for p in ps]
        if len(previas) < min_train:
            continue
        r = fit_ratings(previas, half_life_dias=half_life)
        for p in temporadas[i][1]:
            ll, lv = r.lambdas(p.local, p.visitante)
            gl, gv = p.goles_local, p.goles_visitante
            # Poisson: log P(g) = g·log λ − λ − log(g!)
            from math import lgamma
            ll_tot += (gl * np.log(ll) - ll - lgamma(gl + 1)
                       + gv * np.log(lv) - lv - lgamma(gv + 1))
            pred = "H" if ll > lv else ("A" if lv > ll else "D")
            real = "H" if gl > gv else ("A" if gv > gl else "D")
            aciertos += pred == real
            n += 1
    return ll_tot / n, aciertos / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-train", type=int, default=120)
    ap.add_argument("--modo", choices=("temporada", "fecha"), default="fecha")
    a = ap.parse_args()

    partidos = load_dataset_completo()
    print(f"{len(partidos)} partidos - modo {a.modo}\n")
    print(f"  {'vida media':>14}{'loglik/partido':>18}{'Δ vs actual':>14}{'1X2':>9}{'n':>7}")

    base = None
    for hl in VIDAS_MEDIA:
        fn = evaluar_por_fecha if a.modo == 'fecha' else evaluar
        ll, acc, n = fn(partidos, hl, a.min_train)
        if base is None:
            base = ll
        etiqueta = "sin decay" if hl is None else f"{hl:,.0f} días"
        marca = "  ← hoy" if hl is None else ""
        print(f"  {etiqueta:>14}{ll:>18.4f}{ll - base:>+14.4f}{100*acc:>8.1f}%{n:>7}{marca}")

    print("\n  loglik más ALTA es mejor (más probable el resultado real).")
    print("  El 1X2 es secundario: se puede subir prediciendo siempre al favorito.")


if __name__ == "__main__":
    main()

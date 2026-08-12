"""¿Cuánto vale que los ratings ingieran los resultados de la PROPIA temporada?

La auditoría del 12/8 encontró que producción NO lo hace: el histórico termina en
el Apertura 2026 y `ensure_ratings` solo descarga si el archivo falta, así que los
ratings quedan congelados pre-torneo toda la temporada — mientras el backtest del
decay (scripts/backtest_ratings_decay.py) refiteaba por fecha CON los partidos
intra-temporada, midiendo un régimen que producción no ejecuta.

Esto separa las dos cosas. Walk-forward por fecha sobre las 4 temporadas con
historia previa, mismas predicciones en ambos brazos (comparación pareada):

  * congelado : fit solo con los campeonatos ANTERIORES (producción hoy)
  * con intra : fit con lo anterior + las fechas ya jugadas de ESTA temporada

Métrica: log-verosimilitud Poisson por partido (la que manda) + acierto 1X2.
Además reporta el Δ por tramo de temporada (fechas 2-8 vs 9-15): la ventaja de
ingerir la propia temporada debería CRECER con las fechas acumuladas.
"""
import argparse
import pathlib
import sys
from math import lgamma

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.clausura.intermedio import load_dataset_completo  # noqa: E402
from src.clausura.ratings import fit_ratings  # noqa: E402


def _ts(p):
    from datetime import datetime
    return datetime.fromisoformat(p.inicio_utc.replace("Z", "+00:00")).timestamp()


def _loglik(r, p):
    ll, lv = r.lambdas(p.local, p.visitante)
    gl, gv = p.goles_local, p.goles_visitante
    lik = (gl * np.log(ll) - ll - lgamma(gl + 1)
           + gv * np.log(lv) - lv - lgamma(gv + 1))
    pred = "H" if ll > lv else ("A" if lv > ll else "D")
    real = "H" if gl > gv else ("A" if gv > gl else "D")
    return lik, pred == real


def run(min_train: int = 120):
    partidos = sorted(load_dataset_completo(), key=_ts)

    fechas = []
    for p in partidos:
        if not fechas or fechas[-1][0] != (p.campeonato_id, p.fecha_id):
            fechas.append(((p.campeonato_id, p.fecha_id), []))
        fechas[-1][1].append(p)

    stats = {"congelado": [0.0, 0, 0], "con_intra": [0.0, 0, 0]}
    por_tramo = {"temprano": [], "tardio": []}   # Δ loglik por partido
    fecha_n_en_temporada: dict[int, int] = {}

    for i in range(1, len(fechas)):
        (camp, _), del_dia = fechas[i][0], fechas[i][1]
        fecha_n_en_temporada[camp] = fecha_n_en_temporada.get(camp, 0) + 1
        n_fecha = fecha_n_en_temporada[camp]

        previas_todas = [p for _, ps in fechas[:i] for p in ps]
        previas_congeladas = [p for p in previas_todas if p.campeonato_id != camp]
        if len(previas_congeladas) < min_train:
            continue
        if len(previas_congeladas) == len(previas_todas):
            continue   # primera fecha de la temporada: los brazos son idénticos

        r_cong = fit_ratings(previas_congeladas)
        r_intra = fit_ratings(previas_todas)
        for p in del_dia:
            lik_c, acc_c = _loglik(r_cong, p)
            lik_i, acc_i = _loglik(r_intra, p)
            stats["congelado"][0] += lik_c
            stats["congelado"][1] += acc_c
            stats["congelado"][2] += 1
            stats["con_intra"][0] += lik_i
            stats["con_intra"][1] += acc_i
            stats["con_intra"][2] += 1
            tramo = "temprano" if n_fecha <= 8 else "tardio"
            por_tramo[tramo].append(lik_i - lik_c)

    n = stats["congelado"][2]
    print(f"{n} predicciones pareadas (fechas 2+ de temporadas con historia previa)\n")
    print(f"  {'brazo':>12}{'loglik/partido':>18}{'1X2':>9}")
    for k, (ll, acc, m) in stats.items():
        print(f"  {k:>12}{ll / m:>18.4f}{100 * acc / m:>8.1f}%")

    deltas = np.array(por_tramo["temprano"] + por_tramo["tardio"])
    se = deltas.std(ddof=1) / np.sqrt(len(deltas))
    print(f"\n  Δ (con_intra − congelado): {deltas.mean():+.4f} ± {se:.4f} "
          f"nats/partido (pareado, n={len(deltas)})")
    for tramo, ds in por_tramo.items():
        d = np.array(ds)
        se_t = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        print(f"    fechas {'2-8 ' if tramo == 'temprano' else '9-15'}: "
              f"{d.mean():+.4f} ± {se_t:.4f} (n={len(d)})")
    print("\n  loglik más ALTA es mejor. Referencia de escala: la calibración del "
          "pool movió 0.106 nats; el decay rechazado medía +0.0021 ± 0.0026.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-train", type=int, default=120)
    run(ap.parse_args().min_train)

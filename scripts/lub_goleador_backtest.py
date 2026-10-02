"""Backtest del goleador uruguayo LUB: predecir la 25/26 con datos de la 24/25.

Candidatos = nacionales que jugaron la 25/26 (se sabe ex ante en qué club estaban):
los de los planteles 26/27 + los que eligió el pool de la penca 37, con su club de la
25/26 en Genius. Partidos por equipo = los REALES de la 25/26 (oráculo: aísla el
modelo de jugador del de equipo). n = 1 temporada: es descriptivo, no evidencia.
"""
import json
import unicodedata

import numpy as np

from src.lub.goleador import (EQUIPO_PENCA, GENIUS_JSON, PLANTELES_JSON, _buscar, priors,
                              simular_goleador)

POOL_37 = {  # picks de goleador del pool 25/26 (penca 37), bajados el 2/10/2026
    "Santiago Vescovi": 46, "Luciano Parodi": 19, "Nicola Pomoli": 12, "Facundo Terra": 11,
    "Gaston Semiglia": 8, "Emiliano Serres": 7, "Santiago Vidal": 6, "Federico Bavosi": 5,
    "Juan Santiso": 5, "Santiago Moglia": 5, "Joaquin Osimani": 4, "Patricio Prieto": 4,
    "Federico Haller": 3, "Marcos Cabot": 2, "Federico Pereiras": 2, "Nicolas Martinez": 2,
}


def main():
    g = json.loads(GENIUS_JSON.read_text())
    nombres = set(POOL_37)
    for eq, pl in json.loads(PLANTELES_JSON.read_text()).items():
        if not eq.startswith("_"):
            nombres |= set(pl["mayores"]) | set(pl["sub23"])
    solo_prev = {"24/25": g["24/25"]}
    cands, real = [], {}
    for n in sorted(nombres):
        filas = _buscar(g, n)
        f26 = [f for f in filas if f["temporada"] == "25/26"]
        if not f26:
            continue
        club = EQUIPO_PENCA.get(max(f26, key=lambda f: f["pj"])["equipo"])
        f25 = [f for f in _buscar(solo_prev, n)]
        c = {"nombre": n, "equipo": club}
        if f25:
            pj, pts = sum(f["pj"] for f in f25), sum(f["pts"] for f in f25)
            c["24/25"] = {"pj": pj, "pts": pts, "ppg": pts / pj,
                          "club": EQUIPO_PENCA.get(max(f25, key=lambda f: f["pj"])["equipo"])}
        cands.append(c)
        real[n] = sum(f["pts"] for f in f26)
    pc = priors(cands)
    for c in pc:      # sin priores manuales de la 26/27 (no existían)
        if c.origen == "manual":
            c.mu, c.sd, c.origen = 5.0, 3.0, "sin datos"
    juegos = {"Aguada": 45, "Peñarol": 43, "Defensor Sporting": 39, "Nacional": 39, "Malvín": 36,
              "Hebraica Macabi": 34, "Urunday Universitario": 30, "Bigua": 30, "Unión Atlética": 28,
              "Welcome": 28, "Goes": 27, "Cordón": 27}
    equipos = sorted(juegos)
    S = 20_000
    pe = np.tile(np.array([juegos[e] for e in equipos]), (S, 1))
    gol, total = simular_goleador(pe, equipos, pc)
    p = np.bincount(gol, minlength=len(pc)) / S
    tot_pool = sum(POOL_37.values())
    orden_real = sorted(real, key=real.get, reverse=True)
    print(f"{len(pc)} candidatos. Real: " + ", ".join(f"{n} {real[n]}" for n in orden_real[:5]))
    print(f"{'jugador':22s} {'P modelo':>8s} {'pool':>6s} {'real':>5s}  origen")
    for i in np.argsort(-p)[:12]:
        n = pc[i].nombre
        print(f"{n:22s} {p[i]:8.1%} {POOL_37.get(n, 0) / tot_pool:6.1%} {real[n]:5d}  {pc[i].origen}")
    i_v = [c.nombre for c in pc].index(orden_real[0])
    print(f"\nganador real {orden_real[0]}: P modelo {p[i_v]:.1%}, pool "
          f"{POOL_37.get(orden_real[0], 0) / tot_pool:.1%}, rank modelo "
          f"{int((p > p[i_v]).sum()) + 1}")


if __name__ == "__main__":
    main()

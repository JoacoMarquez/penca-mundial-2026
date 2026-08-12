"""¿El sesgo del 0-0 depende de cuán parejo sea el partido?

LA HIPÓTESIS (auditoría externa del 2026-08-11): el factor del 0-0 es una constante
0.55 en pool.py, pero el multiplicador correcto variaría ~16× — alto en partidos
parejos, casi cero en desparejos. Si es así, sobreestimamos las colisiones de 0-0 en
los parejos y evitamos un hueco vacío en los desparejos.

MI RESERVA: con chalk=2.2 el cociente Q/p del 0-0 YA varía 2,5× entre parejo y muy
desparejo, porque p^2.2 aplasta lo improbable. Así que parte de esos 16× ya está
cubierta, y lo que falta es menos de lo que dice el titular.

Y hay evidencia en contra de que sirva: la tabla de 36 sesgos calibrada midió EMPATE
contra el chalk solo en E[premio] (+$1.503 ± 439 vs +$1.273 ± 194). Refinar la FORMA
de la Q no se convirtió en plata; lo que pagó fue la concentración.

Por eso esto es la ETAPA 1 y es barata: ¿mejora el ajuste fuera de muestra? Si no
mejora acá, no vale gastar las 2h de medir E[premio].

Se compara contra la Q DE PRODUCCIÓN (chalk 2.2 + tabla actual), no contra un modelo
de juguete, y se valida leave-one-match-out.
"""
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.clausura.economics import flatten_grid, score_index  # noqa: E402
from src.clausura.pool import PoolConfig, pool_distribution  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrar_pool import cargar, reflejar  # noqa: E402

N, I00 = 36, score_index(0, 0)
IDX = np.arange(N)
GL, GV = IDX // 6, IDX % 6


def desbalance(p):
    """0 = parejo, 1 = un lado se lleva toda la probabilidad de ganar."""
    return abs(float(p[GL > GV].sum()) - float(p[GL < GV].sum()))


def q_produccion(e):
    """Lo que juega producción hoy. Los eventos vienen ya orientados al favorito."""
    p = reflejar(e["p"]) if e["fav_visita"] else e["p"]
    q = pool_distribution(p.reshape(6, 6), PoolConfig())
    return reflejar(q) if e["fav_visita"] else q


def q_ajustada(e, b0, b1):
    """Producción con el 0-0 multiplicado por exp(b0 + b1·desbalance)."""
    q = q_produccion(e).copy()
    q[I00] *= np.exp(b0 + b1 * e["desb"])
    return q / q.sum()


def loglik(evs, fn):
    return sum(float(e["c"] @ np.log(fn(e) + 1e-300)) for e in evs)


def ajustar(evs, con_paridad):
    def neg(th):
        b0, b1 = (th[0], th[1]) if con_paridad else (th[0], 0.0)
        return -loglik(evs, lambda e: q_ajustada(e, b0, b1))
    x0 = np.zeros(2 if con_paridad else 1)
    r = minimize(neg, x0, method="L-BFGS-B").x
    return (float(r[0]), float(r[1])) if con_paridad else (float(r[0]), 0.0)


def main():
    evs = cargar()
    for e in evs:
        e["desb"] = desbalance(e["p"])
    n_picks = sum(e["n"] for e in evs)
    print(f"{len(evs)} partidos · {n_picks:,} picks reales\n")

    print("=== el 0-0 empírico por paridad del partido ===")
    print(f"  {'partido':<34}{'desbalance':>12}{'0-0 pool':>11}{'0-0 mercado':>13}{'ratio':>8}")
    for e in sorted(evs, key=lambda x: x["desb"]):
        real = e["c"][I00] / e["c"].sum()
        merc = e["p"][I00]
        print(f"  {e['nombre']:<34}{e['desb']:>12.2f}{100*real:>10.1f}%{100*merc:>12.1f}%"
              f"{real/merc:>8.2f}")
    ratios = [e["c"][I00] / e["c"].sum() / e["p"][I00] for e in evs]
    print(f"\n  variación del ratio crudo entre las puntas: "
          f"{max(ratios)/max(min(ratios), 1e-9):.1f}×")

    print("\n=== ¿mejora el ajuste FUERA de muestra? (leave-one-match-out) ===")
    filas = []
    for nombre, con_par in (("producción (0-0 constante)", None),
                            ("+ 0-0 global ajustado", False),
                            ("+ 0-0 según paridad", True)):
        tot_ll, tot_n = 0.0, 0
        for i in range(len(evs)):
            train = [e for j, e in enumerate(evs) if j != i]
            if con_par is None:
                fn = q_produccion
            else:
                b0, b1 = ajustar(train, con_par)
                fn = (lambda b0, b1: lambda e: q_ajustada(e, b0, b1))(b0, b1)
            tot_ll += loglik([evs[i]], fn)
            tot_n += int(evs[i]["c"].sum())
        filas.append((nombre, -tot_ll / tot_n))

    base = filas[0][1]
    for nombre, ll in filas:
        print(f"  {nombre:<32}{ll:>10.4f} nats/pick{ll - base:>+11.4f}")

    b0, b1 = ajustar(evs, True)
    print(f"\n  ajuste sobre los {len(evs)}: 0-0 ×= exp({b0:+.2f} {b1:+.2f}·desbalance)")
    for d, etiq in ((0.05, "parejo"), (0.55, "desparejo")):
        print(f"    {etiq:<12} desbalance {d:.2f} → factor {np.exp(b0 + b1*d):.2f}")
    print("\n  loglik más BAJA es mejor. Si la mejora no es clara acá, no vale medir E[premio].")


if __name__ == "__main__":
    main()

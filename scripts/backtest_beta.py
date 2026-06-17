"""Barrido de β (PENCA_HORIZON_BETA) y w_exact con el backtest secuencial.

β controla el premium de horizonte (β·√partidos_restantes) que sube el cutoff proyectado:
β alto = se queda en modo "Sumando" (ancla EV/1-0) más tiempo; β bajo = pasa a "A ganar"
(persigue el corte, suelta el ancla) antes. w_exact pondera el desempate por marcador exacto.

Settings de producción: N=15, pool=436, top_k=3 (podio), max_candidates=10.

Uso:
    python -m scripts.backtest_beta <dataset> <mode> [sims]
    mode = beta | wexact | grid
    ej: python -m scripts.backtest_beta euro_2024 beta 5000
"""
import sys, time
from pathlib import Path
from src.backtest.runner import load_backtest_data, run_sequential_mc

dataset = sys.argv[1]
mode = sys.argv[2]
sims = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

matches = load_backtest_data(dataset, Path("data/backtest"))
COMMON = dict(n_pool_players=436, n_pencas=15, max_candidates=10, top_k=3, seed=42)

BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
WEXACTS = [0.0, 1.0, 2.0, 5.0, 10.0]


def run(beta, w):
    t = time.time()
    r = run_sequential_mc(matches, n_sims=sims, horizon_beta=beta, w_exact=w, **COMMON)
    r["_dt"] = time.time() - t
    return r


if mode == "beta":
    print(f"# {dataset} | barrido β | {len(matches)} partidos | pool 436, N=15, top_k=3, sims={sims}")
    print(f"{'β':>5} {'P(gana)':>9} {'P(g+desemp)':>12} {'nuestro_max':>12} {'pool_top':>9} {'gap':>6}")
    for b in BETAS:
        r = run(b, 0.0)
        gap = r["portfolio_max_mean"] - r["pool_top_mean"]
        print(f"{b:>5.1f} {r['p_win']:>8.1%} {r['p_win_resolved']:>11.1%} "
              f"{r['portfolio_max_mean']:>12.1f} {r['pool_top_mean']:>9.1f} {gap:>+6.1f}  ({r['_dt']:.0f}s)",
              flush=True)

elif mode == "wexact":
    print(f"# {dataset} | barrido w_exact (β=2.0 prod) | {len(matches)} partidos | sims={sims}")
    print(f"{'w':>5} {'P(gana)':>9} {'P(g+desemp)':>12} {'nuestro_max':>12} {'pool_top':>9}")
    for w in WEXACTS:
        r = run(2.0, w)
        print(f"{w:>5.1f} {r['p_win']:>8.1%} {r['p_win_resolved']:>11.1%} "
              f"{r['portfolio_max_mean']:>12.1f} {r['pool_top_mean']:>9.1f}  ({r['_dt']:.0f}s)",
              flush=True)

elif mode == "grid":
    print(f"# {dataset} | grilla β × w_exact | sims={sims} | celda = P(gana+desempate)")
    hdr = "  β\\w " + "".join(f"{w:>8.0f}" for w in WEXACTS)
    print(hdr)
    for b in [0.0, 0.5, 1.0, 2.0]:
        cells = []
        for w in WEXACTS:
            r = run(b, w)
            cells.append(f"{r['p_win_resolved']:>8.1%}")
        print(f"{b:>5.1f} " + "".join(cells), flush=True)

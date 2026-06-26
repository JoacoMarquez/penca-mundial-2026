"""Backtest SEMBRADO para la decisión "¿proteger a la penca líder?".

El backtest secuencial normal (runner.run_sequential_mc) arranca todas las standings en
CERO, así que nunca pone a una de nuestras pencas SOBRE el corte del pool — el escenario
real de hoy (1891 = #1 absoluto, +9 sobre el corte top-3) queda fuera de su distribución y
no puede medir ni el bug ni el fix.

Este script:
  1. SIEMBRA las standings reales de hoy (nuestras 15) + un pool de campo de 421 jugadores
     calibrado a los cuantiles reales del pool (top ~157, corte top-3 ~152, mediana 104).
  2. Simula los partidos restantes (proxy: Euro 2024, 51 partidos, escala similar).
  3. Corre el A/B:
        - baseline  : status quo (premium global para todas → la líder persigue → anti-favorito)
        - protect θ : premium_eff = 0 si base >= corte + θ·premium, sino premium completo
     θ barre el criterio de protección: θ=0 protege contra el corte de HOY, θ=1 contra el
     PROYECTADO (corte+premium). Mantiene el premium completo para las perseguidoras
     (preserva el β=2.0 validado); sólo exime a las pencas ya a salvo.
  4. Mide lo que la métrica-unión oculta: P(la líder 1891 conserva podio), descomposición
     de la victoria (líder vs remontada), exactos de la líder, y % de picks anti-favorito
     de la líder (sanity del mecanismo).

Uso:
    python3 -m scripts.backtest_protect_leader [sims] [dataset]
    ej: python3 -m scripts.backtest_protect_leader 3000 euro_2024
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.backtest.runner import _precompute_matches, load_backtest_data
from src.meta.pool import PoolModelConfig

# ---- standings REALES de hoy (2026-06-25). Índice 0 = la líder 1891. ----
OUR_PTS0 = np.array(
    [161, 141, 135, 122, 121, 121, 120, 120, 120, 120, 118, 117, 117, 76, 76],
    dtype=float,
)
OUR_LABELS = ["1891", "1887", "1654", "1652", "1655", "1892", "1890",
              "1651", "1889", "1893", "1886", "1888", "1894", "1653", "1885"]
LEADER = 0  # índice de 1891

N_FIELD = 421          # 436 del pool − 15 nuestras
FIELD_MEDIAN = 104.0
FIELD_SIGMA = 17.5     # calibrado: max de 421 draws ≈ 104 + 17.5·3.0 ≈ 157 (= 2° del pool real)
HORIZON_BETA = 2.0
TOP_K = 3              # podio
CHALK = 0.70


def field_seed() -> np.ndarray:
    """Vector determinístico de 421 puntos calibrado a los cuantiles del pool real.

    Usa la inversa-CDF normal en cuantiles equiespaciados → reproduce top ~157, mediana 104.
    Determinístico a propósito: el estado inicial del campo es un HECHO (el leaderboard de
    hoy), no algo aleatorio; sólo el futuro (los partidos) se randomiza.

    σ overridable por env PENCA_FIELD_SIGMA (chequeo de robustez a la calibración del campo).
    """
    import os
    from scipy.stats import norm
    sigma = float(os.environ.get("PENCA_FIELD_SIGMA", FIELD_SIGMA))
    q = (np.arange(N_FIELD) + 0.5) / N_FIELD
    pts = FIELD_MEDIAN + sigma * norm.ppf(q)
    return np.round(pts)


def _build_M(matches, max_candidates: int) -> list[dict]:
    pre = _precompute_matches(matches, PoolModelConfig(), max_candidates, with_pts_matrix=True)
    M = []
    for p, match in zip(pre, matches):
        n = p["n"]
        pts = p["pts_matrix"]
        cand_idx = np.array([int(c["score"][0]) * n + int(c["score"][1]) for c in p["cand_dicts"]])
        # ganador de cada candidato (1/X/2)
        cand_win = []
        for ci in cand_idx:
            gL, gV = ci // n, ci % n
            cand_win.append("1" if gL > gV else ("2" if gL < gV else "X"))
        fav = "1" if match.market_p_home > match.market_p_away else (
            "2" if match.market_p_away > match.market_p_home else "X")
        M.append({
            "cand_pts_grid": pts[cand_idx, :].astype(float),  # (K,G)
            "grid_probs": p["flat_grid"],
            "modal_gain": pts[p["modal_idx"], :].astype(float),
            "cand_idx": cand_idx,
            "cand_win": cand_win,
            "fav": fav,
            "pts": pts, "modal_idx": p["modal_idx"], "top_idx": p["top_idx"],
            "top_p": p["top_p"], "flat_grid": p["flat_grid"], "G": n * n,
        })
    return M


def _alloc(cand_pts_grid, grid_probs, current_pts, cutoff, premium, modal_gain,
           protect: bool, theta: float) -> np.ndarray:
    """Voraz per-penca. baseline = protect False (premium global). protect = exime a las ya
    a salvo (base >= cutoff + θ·premium) poniéndoles premium_eff 0 → mirror-chalk/ancla EV."""
    K, G = cand_pts_grid.shape
    N = current_pts.shape[0]
    order = np.argsort(-current_pts, kind="stable")  # líder primero
    best_final = np.full(G, -1e9)
    assigned = np.empty(N, dtype=int)
    usage = np.zeros(K)
    protect_cut = cutoff + theta * premium
    for pid in order:
        base = current_pts[pid]
        premium_eff = 0.0 if (protect and base >= protect_cut) else premium
        thr = cutoff + premium_eff + modal_gain  # (G,)
        bf = np.maximum(best_final[None, :], base + cand_pts_grid)  # (K,G)
        emax = bf @ grid_probs
        key = ((bf >= thr[None, :]) @ grid_probs) * 1e6 + emax  # P(top-K) domina; e_max desempata
        tied = np.where(key >= key.max() - 1e-9)[0]
        c = int(tied[np.argmin(usage[tied])])
        assigned[pid] = c
        usage[c] += 1
        best_final = bf[c]
    return assigned


def run_arm(M, n_sims: int, seed: int, protect: bool, theta: float) -> dict:
    rng = np.random.default_rng(seed)
    field0 = field_seed()
    nM = len(M)
    N = len(OUR_PTS0)

    win = win_tie = 0
    any_podium = leader_podium = leader_is1 = 0
    podium_from_leader = podium_from_comeback = 0
    leader_pts_acc = leader_ex_acc = 0.0
    leader_antifav = 0
    leader_total_picks = 0
    # Premio graduado real: #1=20k, #2=10k, #3=5k. Optimizamos E[premio].
    PAYOUT = [20000, 10000, 5000]
    money_acc = 0.0
    hold1 = hold2 = hold3 = 0

    for _ in range(n_sims):
        our = OUR_PTS0.copy()
        field = field0.copy()
        our_ex = np.zeros(N, dtype=int)
        for t, m in enumerate(M):
            combined = np.concatenate([our, field])
            cutoff = float(np.partition(combined, -TOP_K)[-TOP_K])  # corte top-3 del pool (incluye us)
            premium = HORIZON_BETA * float(np.sqrt(nM - t))
            assigned = _alloc(m["cand_pts_grid"], m["grid_probs"], our, cutoff, premium,
                              m["modal_gain"], protect, theta)
            o = int(rng.choice(m["G"], p=m["flat_grid"]))
            our = our + m["cand_pts_grid"][assigned, o]
            our_ex = our_ex + (m["cand_idx"][assigned] == o)
            # sanity del mecanismo: ¿la líder jugó anti-favorito?
            lead_c = assigned[LEADER]
            leader_total_picks += 1
            if m["fav"] != "X" and m["cand_win"][lead_c] != m["fav"]:
                leader_antifav += 1
            # campo
            is_chalk = rng.random(N_FIELD) < CHALK
            sampled = rng.choice(m["top_idx"], size=N_FIELD, p=m["top_p"])
            pool_pick = np.where(is_chalk, m["modal_idx"], sampled)
            field = field + m["pts"][pool_pick, o]

        field_max = field.max()
        our_max = our.max()
        win += our_max > field_max
        win_tie += our_max >= field_max
        # rango por competición (cuántos ESTRICTAMENTE arriba) +1
        ranks = np.array([1 + int((field > our[i]).sum()) + int((our > our[i]).sum()) for i in range(N)])
        podium_mask = ranks <= TOP_K
        ap = bool(podium_mask.any())
        any_podium += ap
        lp = bool(podium_mask[LEADER])
        leader_podium += lp
        leader_is1 += int(ranks[LEADER] == 1)
        if ap:
            if lp:
                podium_from_leader += 1
            else:
                podium_from_comeback += 1
        leader_pts_acc += our[LEADER]
        leader_ex_acc += our_ex[LEADER]
        # E[premio]: campo primero en el combinado → desempate de borde CONSERVADOR (favorece
        # al campo en empates exactos de puntos). order[:3] = los 3 puestos con plata.
        combined_money = np.concatenate([field, our])
        order = np.argsort(-combined_money, kind="stable")
        for slot in range(3):
            if order[slot] >= N_FIELD:           # ese puesto lo ocupa una penca nuestra
                money_acc += PAYOUT[slot]
                if slot == 0: hold1 += 1
                elif slot == 1: hold2 += 1
                else: hold3 += 1

    s = n_sims
    return {
        "label": ("baseline (status quo)" if not protect else f"protect θ={theta:g}"),
        "p_win": win / s,
        "p_win_tie": win_tie / s,
        "p_podium": any_podium / s,
        "p_leader_podium": leader_podium / s,
        "p_leader_is1": leader_is1 / s,
        "podium_from_leader": podium_from_leader / s,
        "podium_from_comeback": podium_from_comeback / s,
        "leader_pts": leader_pts_acc / s,
        "leader_exactos": leader_ex_acc / s,
        "leader_antifav_pct": 100.0 * leader_antifav / max(leader_total_picks, 1),
        "e_money": money_acc / s,
        "p_hold1": hold1 / s, "p_hold2": hold2 / s, "p_hold3": hold3 / s,
    }


def main() -> int:
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    dataset = sys.argv[2] if len(sys.argv) > 2 else "euro_2024"
    seed = 42
    matches = load_backtest_data(dataset, Path("data/backtest"))
    M = _build_M(matches, max_candidates=10)

    print(f"\n{'='*92}")
    print(f"BACKTEST SEMBRADO — proteger a la líder | dataset={dataset} ({len(matches)} partidos) "
          f"| sims={n_sims} | pool 436 (15 nuestras + {N_FIELD} campo)")
    print(f"Siembra: líder 1891=161 (+9 sobre corte), corte top-3≈152, mediana campo={FIELD_MEDIAN:g}")
    print(f"{'='*92}")

    arms = [run_arm(M, n_sims, seed, protect=False, theta=0.0)]            # status quo
    for th in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]:
        arms.append(run_arm(M, n_sims, seed, protect=True, theta=th))

    print(f"\n{'arm':>22} | {'E[premio]':>10} | {'P(#1)':>7} {'P(#2)':>7} {'P(#3)':>7} | "
          f"{'P(gana)':>8} {'P(podio)':>9} {'P(líd pod)':>10} | {'antifav':>7}")
    print(f"{'-'*22}-+-{'-'*10}-+-{'-'*23}-+-{'-'*29}-+-{'-'*7}")
    base = arms[0]
    best = max(arms, key=lambda a: a["e_money"])
    for a in arms:
        star = " ★" if a is best else ("  " if a is base else "  ")
        delta = a["e_money"] - base["e_money"]
        print(f"{a['label']:>22} | {a['e_money']:>9,.0f}{star}| "
              f"{a['p_hold1']:>6.1%} {a['p_hold2']:>6.1%} {a['p_hold3']:>6.1%} | "
              f"{a['p_win']:>7.1%} {a['p_podium']:>8.1%} {a['p_leader_podium']:>9.1%} | "
              f"{a['leader_antifav_pct']:>6.0f}%")

    print(f"\nLectura (premio real: #1=20k, #2=10k, #3=5k):")
    print(f"  · E[premio] es la métrica objetivo dado el pago graduado. ★ = óptimo.")
    print(f"  · baseline (status quo) E[premio]={base['e_money']:,.0f}; mejor protect "
          f"={best['e_money']:,.0f} (Δ {best['e_money']-base['e_money']:+,.0f}, "
          f"{100*(best['e_money']/base['e_money']-1):+.0f}%).")
    print(f"  · antifav líder baseline={base['leader_antifav_pct']:.0f}% confirma el bug; el fix lo baja.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

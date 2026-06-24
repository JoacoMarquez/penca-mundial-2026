"""Fase 2 del backtest del bound de motivación.

¿Bajar el λ del equipo ya-clasificado en la 3ra fecha (lo que el bound de motivación habilita)
mejora P(gana el pool)? Barre el bound B y mide el objetivo real.

Modelo (modo resultado HISTÓRICO, no MC):
  - El resultado de cada partido es el REAL (Fase 1 mostró que el clasificado mete menos goles
    de lo que el mercado esperaba → ahí está el edge).
  - El POOL sintético pica desde la grilla del MERCADO sin ajustar (los humanos no tienen nuestro
    ajuste de motivación). Esa asimetría es el edge.
  - NUESTRAS pencas, en los partidos de 3ra fecha con un equipo clasificado, generan su menú de
    candidatos desde una grilla con el λ de ese equipo reducido en B (recortado, como Capa 4).
  - B = 0 reproduce el baseline exacto (sin motivación). Barremos B ∈ {0, .10, .20, .30}.

Sin leakage: el equipo "clasificado" se determina con la tabla post-fecha-2; el ajuste no usa
el resultado de la 3ra fecha, solo lo hace la puntuación final (igual para todos los B).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.backtest.runner import (
    load_backtest_data, _precompute_matches, _simulate_pool_fast,
    _portfolio_for_n, _build_result,
)
from src.meta.pool import PoolModelConfig
from src.model.poisson import MarketConstraints, fit_params, marginals, score_grid
from src.strategy.portfolio import generate_candidates, picks_to_dicts

from scripts.backtest_motivation_premise import standings_after_2, stake

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "backtest"


def qualified_sides(matches) -> dict[int, list[str]]:
    """{idx_partido_3ra_fecha: ['L'|'V', ...]} con los equipos clasificados (pts>=4 post-fecha-2)."""
    flags: dict[int, list[str]] = {}
    for g in range(6):
        block = matches[g * 6:(g + 1) * 6]
        tbl = standings_after_2([
            {"home": m.home, "away": m.away, "hs": m.actual_home_score, "as": m.actual_away_score}
            for m in block
        ])
        for pos in (4, 5):  # posiciones de 3ra fecha dentro del bloque
            idx = g * 6 + pos
            m = matches[idx]
            sides = []
            if stake(m.home, tbl) in ("CLASIFICADO_1", "CASI_ADENTRO"):
                sides.append("L")
            if stake(m.away, tbl) in ("CLASIFICADO_1", "CASI_ADENTRO"):
                sides.append("V")
            if sides:
                flags[idx] = sides
    return flags


def adjusted_entry(match, sides: list[str], B: float, max_candidates: int) -> dict:
    """Recomputa grid + menú de candidatos con el λ del clasificado reducido en B."""
    c = MarketConstraints(
        p_home_win=match.market_p_home, p_draw=match.market_p_draw, p_away_win=match.market_p_away,
        p_over_2_5=match.market_p_over_2_5, p_btts=match.market_p_btts,
    )
    lamL, lamV, lam12 = fit_params(c)
    if "L" in sides:
        lamL = max(0.15, lamL - B)
    if "V" in sides:
        lamV = max(0.15, lamV - B)
    lam12 = min(lam12, lamL, lamV)
    grid = score_grid(lamL, lamV, lam12)
    m = marginals(grid)
    cands = picks_to_dicts(
        generate_candidates(grid, m.p_home_win, m.p_away_win, max_candidates=max_candidates)
    )
    return {"grid": grid, "cand_dicts": cands}


def run_for_bound(matches, flags, B, n_pencas, n_sims, n_pool, max_candidates, pool_total, base_pre):
    """Portfolio con ajuste de motivación de magnitud B; pool fijo (base_pre/pool_total)."""
    pre_ours = [dict(e) for e in base_pre]   # copia shallow; vamos a pisar grid+cands de algunos
    for idx, sides in flags.items():
        adj = adjusted_entry(matches[idx], sides, B, max_candidates)
        pre_ours[idx]["grid"] = adj["grid"]
        pre_ours[idx]["cand_dicts"] = adj["cand_dicts"]
        # pts_for_pick / n NO cambian (dependen solo del resultado real)
    total_by_penca, portfolio_max, distinct, chalk_total = _portfolio_for_n(pre_ours, n_pencas)
    res = _build_result(
        base_pre, n_sims, n_pool, n_pencas, max_candidates,
        pool_total, total_by_penca, portfolio_max, distinct, chalk_total,
    )
    return res, portfolio_max


def main():
    n_sims, n_pool, n_pencas, max_candidates, seed = 4000, 150, 5, 8, 42
    matches = load_backtest_data("euro_2024", DATA_ROOT)
    flags = qualified_sides(matches)

    print("=" * 84)
    print("FASE 2 — ¿soltar el bound de motivación mejora P(gana el pool)? (Euro 2024)")
    print(f"sims={n_sims}  pool={n_pool}  pencas={n_pencas}  menú={max_candidates}  seed={seed}")
    print("=" * 84)
    print("\nPartidos de 3ra fecha con equipo clasificado (λ a reducir):")
    for idx, sides in sorted(flags.items()):
        m = matches[idx]
        who = [m.home if "L" in sides else None, m.away if "V" in sides else None]
        who = [w for w in who if w]
        print(f"  {m.match_id}: {m.home} vs {m.away}  → reduce λ de: {', '.join(who)}")

    rng = np.random.default_rng(seed)
    base_pre = _precompute_matches(matches, PoolModelConfig(), max_candidates)
    pool_total = _simulate_pool_fast(base_pre, n_pool, n_sims, rng)

    print(f"\n{'bound B':>8} {'P(gana pool)':>14} {'P(top1%)':>10} {'P(top5%)':>10} {'pts portfolio_max':>18}")
    print("-" * 84)
    base_pmax = None
    for B in (0.0, 0.10, 0.20, 0.30):
        res, pmax = run_for_bound(matches, flags, B, n_pencas, n_sims, n_pool,
                                  max_candidates, pool_total, base_pre)
        if base_pmax is None:
            base_pmax = pmax
        dpts = pmax - base_pmax
        print(f"{B:8.2f} {res.p_at_least_one_wins:14.4f} {res.p_top_1_percent:10.4f} "
              f"{res.p_top_5_percent:10.4f} {pmax:12d} ({dpts:+d})")

    print("\nNota: portfolio_max es determinístico (picks + resultados reales fijos); la varianza")
    print("de P(gana pool) viene solo del pool sampleado. Δpts es el cambio en puntos de nuestra")
    print("mejor penca al activar el ajuste. n de partidos tocados es chico → leer como dirección.")


if __name__ == "__main__":
    main()

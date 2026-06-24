"""Fase 1 del backtest del bound de motivación.

Pregunta: en la 3ra fecha de grupos de Euro 2024, ¿los equipos YA CLASIFICADOS (que suelen
rotar) metieron MENOS goles que el λ que implicaba el mercado de cierre?

Solo si el mercado SUB-PRECIA la rotación tiene sentido que Capa 4 ajuste por encima de él,
y la magnitud de ese sub-precio es lo que acota el bound (±0.10 vs ±0.30).

NO hay leakage: la clasificación de cada equipo se computa con los resultados de fechas 1 y 2
(previos al partido), nunca con el resultado de la 3ra fecha.
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.model.poisson import MarketConstraints, fit_params, marginals, score_grid

CSV = Path(__file__).resolve().parents[1] / "data" / "backtest" / "euro_2024" / "matches.csv"

# Euro 2024: 6 grupos de 4, bloques de 6 partidos en orden de fecha.
# Posiciones dentro del bloque (1-based): F1 = 1,2 | F2 = 3,4 | F3 = 5,6
GROUPS = ["A", "B", "C", "D", "E", "F"]


def load() -> list[dict]:
    rows = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            r["hs"] = int(r["actual_home_score"])
            r["as"] = int(r["actual_away_score"])
            for k in ("market_p_home", "market_p_draw", "market_p_away",
                      "market_p_over_2_5", "market_p_btts"):
                r[k] = float(r[k]) if r.get(k) else None
            rows.append(r)
    return rows


def market_lambdas(r: dict) -> tuple[float, float]:
    """λ esperado por equipo según el mercado de cierre (incluye λ12 compartido)."""
    c = MarketConstraints(
        p_home_win=r["market_p_home"], p_draw=r["market_p_draw"], p_away_win=r["market_p_away"],
        p_over_2_5=r["market_p_over_2_5"], p_btts=r["market_p_btts"],
    )
    lamL, lamV, lam12 = fit_params(c)
    m = marginals(score_grid(lamL, lamV, lam12))
    return m.expected_goals_L, m.expected_goals_V


def standings_after_2(block: list[dict]) -> dict[str, dict]:
    """Tabla del grupo tras fechas 1 y 2 (los 4 primeros partidos del bloque)."""
    tbl: dict[str, dict] = {}
    def ensure(t):
        tbl.setdefault(t, {"pts": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0})
    for r in block[:4]:
        h, a, hs, as_ = r["home"], r["away"], r["hs"], r["as"]
        ensure(h); ensure(a)
        tbl[h]["gf"] += hs; tbl[h]["ga"] += as_
        tbl[a]["gf"] += as_; tbl[a]["ga"] += hs
        if hs > as_:
            tbl[h]["pts"] += 3; tbl[h]["w"] += 1; tbl[a]["l"] += 1
        elif hs < as_:
            tbl[a]["pts"] += 3; tbl[a]["w"] += 1; tbl[h]["l"] += 1
        else:
            tbl[h]["pts"] += 1; tbl[a]["pts"] += 1; tbl[h]["d"] += 1; tbl[a]["d"] += 1
    return tbl


def stake(team: str, tbl: dict[str, dict]) -> str:
    """Clasifica lo que se juega el equipo en la 3ra fecha (pre-partido).

    Reglas transparentes sobre la tabla post-fecha-2 (máx 6 pts):
      - 6 pts y líder claro → 'CLASIFICADO_1' (ya adentro, suele rotar) → señal motivación ↓
      - 4 pts → 'CASI_ADENTRO' (muy probable top-2)
      - 0 pts y peor difer. → 'AL_BORDE' (poco que jugar; mejores-3ros lo complica)
      - resto → 'EN_DISPUTA'
    """
    pts = tbl[team]["pts"]
    ordered = sorted(tbl.values(), key=lambda x: (x["pts"], x["gf"] - x["ga"]), reverse=True)
    leader_pts = ordered[0]["pts"]
    gd = tbl[team]["gf"] - tbl[team]["ga"]
    if pts == 6:
        return "CLASIFICADO_1"
    if pts == 4:
        return "CASI_ADENTRO"
    if pts == 0:
        return "AL_BORDE"
    return "EN_DISPUTA"


def main():
    rows = load()
    group_blocks = {g: rows[i * 6:(i + 1) * 6] for i, g in enumerate(GROUPS)}

    print("=" * 92)
    print("FASE 1 — ¿el mercado sub-precia la rotación del ya-clasificado? (Euro 2024, 3ra fecha)")
    print("=" * 92)

    records = []
    for g in GROUPS:
        block = group_blocks[g]
        tbl = standings_after_2(block)
        md3 = block[4:6]  # posiciones 5 y 6
        for r in md3:
            lamL, lamV = market_lambdas(r)
            for side, team, opp, lam, goals in (
                ("L", r["home"], r["away"], lamL, r["hs"]),
                ("V", r["away"], r["home"], lamV, r["as"]),
            ):
                st = stake(team, tbl)
                records.append({
                    "grp": g, "team": team, "opp": opp, "stake": st,
                    "lam_mkt": lam, "goals": goals, "resid": goals - lam,
                })

    # Detalle por instancia, ordenado por stake
    order = {"CLASIFICADO_1": 0, "CASI_ADENTRO": 1, "EN_DISPUTA": 2, "AL_BORDE": 3}
    print(f"\n{'grupo':5} {'equipo':14} {'vs':14} {'stake':14} {'λ_mkt':>6} {'goles':>5} {'resid':>7}")
    print("-" * 92)
    for rec in sorted(records, key=lambda x: (order[x["stake"]], x["grp"])):
        print(f"{rec['grp']:5} {rec['team']:14} {rec['opp']:14} {rec['stake']:14} "
              f"{rec['lam_mkt']:6.2f} {rec['goals']:5d} {rec['resid']:+7.2f}")

    # Resumen por bucket
    print("\n" + "=" * 92)
    print("RESIDUAL PROMEDIO (goles reales − λ del mercado) por categoría")
    print("resid < 0  → el mercado esperaba MÁS goles de los que metió (sub-preció la rotación)")
    print("=" * 92)
    print(f"{'stake':16} {'n':>3} {'λ_mkt μ':>8} {'goles μ':>8} {'resid μ':>9} {'resid σ':>9}")
    print("-" * 60)
    import statistics as stx
    for st in ["CLASIFICADO_1", "CASI_ADENTRO", "EN_DISPUTA", "AL_BORDE"]:
        b = [r for r in records if r["stake"] == st]
        if not b:
            continue
        resids = [r["resid"] for r in b]
        lam_mu = stx.mean(r["lam_mkt"] for r in b)
        g_mu = stx.mean(r["goals"] for r in b)
        r_mu = stx.mean(resids)
        r_sd = stx.pstdev(resids) if len(resids) > 1 else 0.0
        print(f"{st:16} {len(b):3d} {lam_mu:8.2f} {g_mu:8.2f} {r_mu:+9.2f} {r_sd:9.2f}")

    # El bucket que importa para el feature: ya-clasificados (rotan)
    rot = [r for r in records if r["stake"] in ("CLASIFICADO_1",)]
    if rot:
        r_mu = stx.mean(r["resid"] for r in rot)
        print("\n" + "-" * 60)
        print(f"SEÑAL CLAVE (CLASIFICADO_1, n={len(rot)}): residual medio = {r_mu:+.2f} goles")
        print("Interpretación: ese número es el sub-precio del mercado que Capa 4 podría capturar.")
        print(f"  |resid| ≈ {abs(r_mu):.2f} → bound de motivación que el dato soporta.")


if __name__ == "__main__":
    main()

"""Sistemas de apuestas y progresiones bajo EV negativo, nulo y positivo — Monte Carlo.

Pregunta del 2026-09-29: con un presupuesto B y la meta "no perder plata", ¿qué hace
cada sistema? Se simulan N apuestas consecutivas con probabilidad real p y cuota o.

Sistemas: flat, Martingale, Fibonacci, D'Alembert, Labouchère, Paroli, Oscar's Grind,
Kelly (1, ½, ¼). Métricas: E[final], mediana, P5/P95, P(ruina), P(terminar arriba),
P(perder la mitad), drawdown máximo medio.

LA LEY: E[final] = B0 + Σ stake·EV. El staking cambia la FORMA de la distribución
(mediana ≠ media), nunca el signo. Con EV<0 todas pierden y las progresiones agregan
ruina (Martingale 69%, Labouchère 90% en 300 apuestas a EV −7,9%). Con EV>0 solo Kelly
fraccionado convierte el edge en crecimiento sin ruina.

Uso:
    python -m scripts.apuestas_progresiones_sim [--apuestas 300] [--caminos 20000] [--bankroll 10000]
"""

from __future__ import annotations

import argparse

import numpy as np


class Sim:
    def __init__(self, b0: float, unit_pct: float, max_stake_pct: float, seed: int = 7):
        self.B0, self.UNIT, self.MAX_STAKE = b0, b0 * unit_pct, b0 * max_stake_pct
        self.rng = np.random.default_rng(seed)

    def run(self, system, p: float, o: float, n_bets: int, n_paths: int) -> dict:
        wins = self.rng.random((n_paths, n_bets)) < p
        B = np.full(n_paths, self.B0)
        alive = np.ones(n_paths, bool)
        peak, maxdd = B.copy(), np.zeros(n_paths)
        state = system.init(n_paths, self.UNIT)
        for t in range(n_bets):
            stake = np.minimum(system.stake(state, B, self.UNIT), self.MAX_STAKE)
            stake = np.where(alive, np.minimum(stake, B), 0.0)
            w = wins[:, t]
            B = B + np.where(w, stake * (o - 1.0), -stake)
            system.update(state, w, stake, B, self.UNIT)
            alive &= B >= self.UNIT   # no puede ni hacer la apuesta mínima
            peak = np.maximum(peak, B)
            maxdd = np.maximum(maxdd, (peak - B) / peak)
        return dict(mean=B.mean(), median=np.median(B), p_ruin=(~alive).mean(), p_up=(B > self.B0).mean(),
                    p_half=(B < self.B0 / 2).mean(), dd=maxdd.mean(),
                    p5=np.percentile(B, 5), p95=np.percentile(B, 95))


class Flat:
    name = "Flat 1%"
    def init(self, n, u): return {}
    def stake(self, s, B, u): return np.full(len(B), u)
    def update(self, s, w, stake, B, u): pass


class Martingale:
    """Tras perder, apostar lo necesario para recuperar todo + 1 unidad."""
    name = "Martingale"
    def __init__(self, o): self.o = o
    def init(self, n, u): return {"cur": np.full(n, u)}
    def stake(self, s, B, u): return s["cur"]
    def update(self, s, w, stake, B, u): s["cur"] = np.where(w, u, np.ceil(stake * self.o / (self.o - 1)))


class Fibonacci:
    name = "Fibonacci"
    fib = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987])
    def init(self, n, u): return {"i": np.zeros(n, int)}
    def stake(self, s, B, u): return u * self.fib[np.minimum(s["i"], len(self.fib) - 1)]
    def update(self, s, w, stake, B, u): s["i"] = np.where(w, np.maximum(s["i"] - 2, 0), s["i"] + 1)


class DAlembert:
    name = "D'Alembert"
    def init(self, n, u): return {"k": np.ones(n)}
    def stake(self, s, B, u): return u * s["k"]
    def update(self, s, w, stake, B, u): s["k"] = np.where(w, np.maximum(s["k"] - 1, 1), s["k"] + 1)


class Labouchere:
    """Lista 1-2-3-4; apuesta = primero+último; gana → borra ambos; pierde → agrega la apuesta."""
    name = "Labouchère"
    def init(self, n, u): self.lists = [[1, 2, 3, 4] for _ in range(n)]; return {}
    def stake(self, s, B, u):
        out = np.empty(len(B))
        for i, L in enumerate(self.lists):
            if not L:
                L[:] = [1, 2, 3, 4]
            out[i] = u * (L[0] + (L[-1] if len(L) > 1 else 0))
        return out
    def update(self, s, w, stake, B, u):
        for i, L in enumerate(self.lists):
            if w[i]:
                L.pop()
                if L:
                    L.pop(0)
            else:
                L.append(int(round(stake[i] / u)))


class Paroli:
    name = "Paroli (x2, 3 pasos)"
    def init(self, n, u): return {"streak": np.zeros(n, int)}
    def stake(self, s, B, u): return u * 2.0 ** s["streak"]
    def update(self, s, w, stake, B, u): s["streak"] = np.where(w & (s["streak"] < 2), s["streak"] + 1, 0)


class Oscar:
    name = "Oscar's Grind"
    def __init__(self, o): self.o = o
    def init(self, n, u): return {"k": np.ones(n), "cyc": np.zeros(n)}
    def stake(self, s, B, u): return u * s["k"]
    def update(self, s, w, stake, B, u):
        s["cyc"] += np.where(w, stake * (self.o - 1), -stake)
        done = s["cyc"] >= u
        s["k"] = np.where(done, 1, np.where(w, s["k"] + 1, s["k"]))
        s["cyc"] = np.where(done, 0, s["cyc"])


class Kelly:
    def __init__(self, p_est, o, frac):
        self.f = max(0.0, (p_est * o - 1) / (o - 1)) * frac
        self.name = f"Kelly {frac:g} (f={self.f:.3f})"
    def init(self, n, u): return {}
    def stake(self, s, B, u): return np.maximum(B * self.f, 0.0)
    def update(self, s, w, stake, B, u): pass


def table(sim: Sim, p: float, o: float, label: str, n_bets: int, n_paths: int) -> None:
    print(f"\n=== {label}: p={p:.3f} o={o:.2f} EV/apuesta={(p*o-1)*100:+.1f}%  "
          f"({n_bets} apuestas, B0={sim.B0:.0f}, unidad={sim.UNIT:.0f}) ===")
    print(f"{'sistema':26} {'E[final]':>9} {'mediana':>9} {'P5':>8} {'P95':>9} {'P(ruina)':>9} "
          f"{'P(arriba)':>9} {'P(<½)':>7} {'DD medio':>8}")
    for sy in [Flat(), Martingale(o), Fibonacci(), DAlembert(), Labouchere(), Paroli(), Oscar(o),
               Kelly(p, o, 1.0), Kelly(p, o, 0.5), Kelly(p, o, 0.25)]:
        r = sim.run(sy, p, o, n_bets, n_paths)
        print(f"{sy.name:26} {r['mean']:9.0f} {r['median']:9.0f} {r['p5']:8.0f} {r['p95']:9.0f} "
              f"{r['p_ruin']*100:8.1f}% {r['p_up']*100:8.1f}% {r['p_half']*100:6.1f}% {r['dd']*100:7.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apuestas", type=int, default=300)
    ap.add_argument("--caminos", type=int, default=20_000)
    ap.add_argument("--bankroll", type=float, default=10_000)
    ap.add_argument("--unidad", type=float, default=0.01, help="unidad base como fracción del bankroll")
    ap.add_argument("--max-stake", type=float, default=0.5, help="tope por apuesta (premio máximo de la casa)")
    a = ap.parse_args()
    sim = Sim(a.bankroll, a.unidad, a.max_stake)
    # favorito típico con el vig de Supermatch: cuota 1.75 donde el precio justo es 1.90
    table(sim, 1 / 1.90, 1.75, "Apuesta 'normal' en Supermatch, sin edge (paga el vig)", a.apuestas, a.caminos)
    table(sim, 0.5, 2.00, "Moneda justa (EV 0)", a.apuestas, a.caminos)
    table(sim, 0.55, 1.95, "Value bet con edge +7% (optimista)", a.apuestas, a.caminos)
    table(sim, 0.50, 2.05, "Value bet con edge +2.5% (realista)", a.apuestas, a.caminos)


if __name__ == "__main__":
    main()

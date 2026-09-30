"""Tests de la simulación de temporada y el portfolio de la penca LUB (sin red)."""

import numpy as np

from src.lub.model import Params, Ratings, class_probs, margin_pmf
from src.lub.portfolio import Evaluador, evaluar, optimizar
from src.lub.scoring import N_CLASES
from src.lub.season import (Config, Perfiles, Slot, Sorteo, _round_robin, ordenar_por_fecha,
                            sample_classes, simular, simular_rivales)

EQUIPOS = [f"E{i}" for i in range(12)]


def _ratings(spread: float = 4.0) -> Ratings:
    r = {e: spread * (5.5 - i) / 5.5 for i, e in enumerate(EQUIPOS)}
    return Ratings(hca=3.0, r=r, sigma=12.0, resid=np.array([]), params=Params())


def _slots_regulares() -> list[Slot]:
    idx = list(range(12))
    out = []
    for f in range(22):
        for i in range(6):
            a, b = idx[i], idx[11 - i]
            h, v = (a, b) if f % 2 == 0 else (b, a)
            out.append(Slot(f"Fecha {f + 1}", "regular", EQUIPOS[h], EQUIPOS[v], i == 0, evento_id=1000 + f * 6 + i))
        idx = [idx[0]] + [idx[-1]] + idx[1:-1]
    return out


def test_class_probs_suman_uno_y_favorito():
    p = class_probs(margin_pmf(6.0, 12.0))
    assert abs(p.sum() - 1) < 1e-9 and p.shape == (N_CLASES,)
    assert p[:5].sum() > 0.6


def test_round_robin_doble():
    rr = _round_robin(6)
    assert len(rr) == 10 and all(len(r) == 3 for r in rr)
    pares = [p for r in rr for p in r]
    assert len(set(pares)) == 30                   # cada cruce una vez de local


def test_sample_classes_respeta_probs():
    rng = np.random.default_rng(0)
    p = np.tile(np.array([0.5, 0.5] + [0] * 8), (2000, 1))
    c = sample_classes(p, rng, 10)
    assert set(np.unique(c)) <= {0, 1} and abs((c == 0).mean() - 0.5) < 0.02


def test_simular_formato_37_fechas_y_campeon_fuerte():
    so = simular(_ratings(8.0), _slots_regulares(), Config(n_sims=400, seed=1))
    assert len(so.fechas) == 22 + 10 + 5
    # fechas contiguas (lo que asume la acumulación por fecha)
    assert (np.diff(so.fecha_de_slot) >= 0).all()
    camp = np.bincount(so.campeon, minlength=12) / 400
    assert camp[so.equipos.index("E0")] > camp[so.equipos.index("E11")]


def test_ordenar_por_fecha_contiguo():
    S, J = 3, 4
    slots = [Slot("A", "playoff"), Slot("B", "playoff"), Slot("A", "playoff"), Slot("B", "playoff")]
    so = Sorteo(slots=slots, fechas=[], fecha_de_slot=np.zeros(J, int), jugado=np.ones((S, J), bool),
                clase=np.tile(np.arange(J, dtype=np.int8), (S, 1)), probs=np.zeros((S, J, 10), np.float32),
                pref=np.zeros(J, bool), campeon=np.zeros(S, int), equipos=["x"])
    so = ordenar_por_fecha(so)
    assert [s.fecha for s in so.slots] == ["A", "A", "B", "B"]
    assert so.clase[0].tolist() == [0, 2, 1, 3]


def test_portfolio_optimizar_no_empeora_y_oos():
    cfg = Config(n_sims=300, n_rivales=60, seed=3)
    so = simular(_ratings(), _slots_regulares(), cfg)
    shares = np.full(12, 1 / 12)
    simular_rivales(so, cfg, lambda p, gamma=1.5: (p ** gamma) / (p ** gamma).sum(-1, keepdims=True),
                    shares, perfiles=Perfiles.plenos(), kappa=0.2)
    actuales = [j for j, s in enumerate(so.slots) if s.fecha == "Fecha 1"]
    ev = Evaluador(so, 4, actuales)
    port = optimizar(ev, list(range(12)), max_pasadas=2)
    assert port.picks_actual.shape == (4, 6)
    assert port.e_premio >= port.detalle["e_premio_inicial"]
    v = evaluar(ev, port.picks_actual, port.campeon)
    assert abs(v["e_premio"] - port.e_premio) < 1e-6

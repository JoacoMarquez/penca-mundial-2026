"""Tests de src/valuebet/staking.py — Kelly fraccionado y caps."""

from src.valuebet import config as vbconfig
from src.valuebet.staking import apply_caps, kelly_stake

CFG = vbconfig.load()


def test_kelly_cerrado():
    # p=0.55, o=2.0 → f* = (1.10-1)/1 = 0.10; con fraction=1 y bankroll 1000 → 100
    assert abs(kelly_stake(0.55, 2.0, 1000.0, 1.0) - 100.0) < 1e-9


def test_kelly_fraccionado():
    assert abs(kelly_stake(0.55, 2.0, 1000.0, 0.25) - 25.0) < 1e-9


def test_kelly_sin_edge_es_cero():
    assert kelly_stake(0.50, 2.0, 1000.0, 0.25) == 0.0    # edge exactamente 0
    assert kelly_stake(0.40, 2.0, 1000.0, 0.25) == 0.0    # edge negativo


def test_kelly_inputs_invalidos():
    assert kelly_stake(0.55, 1.0, 1000.0, 0.25) == 0.0
    assert kelly_stake(0.0, 2.0, 1000.0, 0.25) == 0.0
    assert kelly_stake(1.0, 2.0, 1000.0, 0.25) == 0.0


def test_cap_single():
    # stake pedido 400 sobre bankroll 1000 → cap 5% = 50
    s = apply_caps(400.0, "single", 1000.0, 0.0, 0.0, CFG)
    assert s == 50.0


def test_cap_parlay():
    # cap parlay 2% de 1000 = 20 = justo el mínimo
    s = apply_caps(400.0, "parlay", 1000.0, 0.0, 0.0, CFG)
    assert s == 20.0


def test_cap_parlay_agregado():
    # exposición parlay abierta 95 con cap agregado 10% de 1000 → queda 5 de room → < mínimo → 0
    s = apply_caps(400.0, "parlay", 1000.0, 95.0, 95.0, CFG)
    assert s == 0.0


def test_cap_exposicion_total():
    # exposición abierta 290 de un máx 300 (30% de 1000) → room 10 < stake_min 20 → 0
    s = apply_caps(50.0, "single", 1000.0, 290.0, 0.0, CFG)
    assert s == 0.0


def test_redondeo_y_minimo():
    # 24 → redondea a 20 (múltiplo de 10) y pasa el mínimo
    assert apply_caps(24.0, "single", 1000.0, 0.0, 0.0, CFG) == 20.0
    # 12 → redondea a 10 < mínimo 20 → descarta
    assert apply_caps(12.0, "single", 1000.0, 0.0, 0.0, CFG) == 0.0

"""Tests de src/valuebet/learning.py — shrinkage bayesiano y desactivación."""

from src.valuebet.learning import (
    DEACTIVATE_N, LearningState, MULT_MAX, MULT_MIN, load_state, save_state,
)


def test_segmento_nuevo_multiplier_neutro():
    state = LearningState()
    assert state.multiplier("soccer|1x2|2.0-3.0") == 1.0
    assert state.is_active("soccer|1x2|2.0-3.0")


def test_clv_positivo_sube_multiplier():
    state = LearningState()
    for _ in range(30):
        state.observe("s", clv=0.04, edge_pred=0.04)
    assert state.multiplier("s") > 1.0


def test_clv_negativo_baja_multiplier():
    state = LearningState()
    for _ in range(10):
        state.observe("s", clv=-0.03, edge_pred=0.04)
    m = state.multiplier("s")
    assert MULT_MIN <= m < 1.0


def test_desactivacion_por_clv_negativo_sostenido():
    state = LearningState()
    for _ in range(DEACTIVATE_N):
        state.observe("s", clv=-0.05, edge_pred=0.04)
    assert state.multiplier("s") == 0.0
    assert not state.is_active("s")


def test_pocas_obs_no_desactivan():
    # mismo CLV horrible pero n < DEACTIVATE_N → shrinkage amortigua, sigue activo
    state = LearningState()
    for _ in range(5):
        state.observe("s", clv=-0.10, edge_pred=0.04)
    assert state.is_active("s")


def test_clips():
    state = LearningState()
    for _ in range(200):
        state.observe("up", clv=0.50, edge_pred=0.02)
    assert state.multiplier("up") == MULT_MAX
    state2 = LearningState()
    for _ in range(19):  # bajo el umbral de desactivación
        state2.observe("down", clv=-0.50, edge_pred=0.02)
    assert state2.multiplier("down") == MULT_MIN


def test_persistencia_roundtrip(tmp_path):
    state = LearningState()
    for _ in range(8):
        state.observe("soccer|1x2|1.4-2.0", clv=0.02, edge_pred=0.03)
    p1 = save_state(state, tmp_path)
    loaded = load_state(tmp_path)
    assert loaded.segments == state.segments
    # segunda escritura versiona, no sobreescribe
    p2 = save_state(loaded, tmp_path)
    assert p1 != p2
    assert p1.exists() and p2.exists()

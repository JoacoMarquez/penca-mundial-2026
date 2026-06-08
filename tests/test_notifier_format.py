"""Tests del formato del notifier (sin red)."""

from src.notifier.telegram import _exposure_block


def test_exposure_block_renders_counts():
    meta = {"objective": "p_top_k", "exposure": {"1-0": 6, "1-1": 4, "2-1": 5}}
    block = _exposure_block(meta)
    assert block is not None
    assert "15 pencas" in block  # 6+4+5
    assert "1-0" in block and "×6" in block
    # ordenado por frecuencia desc: 1-0 (6) antes que 2-1 (5) antes que 1-1 (4)
    assert block.index("1-0") < block.index("2-1") < block.index("1-1")


def test_exposure_block_none_when_missing():
    assert _exposure_block(None) is None
    assert _exposure_block({}) is None
    assert _exposure_block({"objective": "e_max"}) is None

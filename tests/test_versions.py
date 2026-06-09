"""Tests del ordenamiento de versiones (no lexicográfico)."""

from pathlib import Path

from src.utils.versions import version_num, sort_versions, latest_version


def test_version_num_extracts_int():
    assert version_num(Path("v9_20260527T000000Z.json")) == 9
    assert version_num(Path("v20_20260527T000000Z.json")) == 20
    assert version_num(Path("garbage.json")) == -1


def test_latest_is_highest_not_lexicographic():
    # 20 versiones: sorted() lexicográfico daría v9 como "última" — el bug que arreglamos.
    names = [Path(f"v{i}_20260527T000000Z.json") for i in range(1, 21)]
    assert latest_version(names).name.startswith("v20_")
    # confirmamos que el sorted() ingenuo SÍ fallaría
    assert sorted(names)[-1].name.startswith("v9_")


def test_sort_versions_numeric_order():
    names = [Path("v10_x.json"), Path("v2_x.json"), Path("v1_x.json"), Path("v9_x.json")]
    assert [version_num(p) for p in sort_versions(names)] == [1, 2, 9, 10]


def test_latest_version_empty():
    assert latest_version([]) is None

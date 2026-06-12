"""Tests de la guarda anti-vaciado del sync de fixtures.

Caso que motiva estos tests: el timer `penca-fixtures-sync` corre 2x/día y
sobreescribía `config/fixtures.yaml` con lo que devolviera la API — sin guarda.
Si la API tenía un hipo y devolvía [] (o menos partidos), pisaba el fixture
bueno con uno vacío y el scheduler quedaba mudo. _suspicious_reason() corta eso.
"""

from __future__ import annotations

from src.scrapers.penca_fixtures import _suspicious_reason


def _grupos(n: int) -> list[dict]:
    return [{"id": i, "stage": "group"} for i in range(n)]


def test_api_vacia_es_sospechosa():
    """API devolvió 0 partidos → no escribir."""
    reason = _suspicious_reason([], [], {"fase_grupos": _grupos(72)})
    assert reason is not None
    assert "0 partidos" in reason


def test_fase_grupos_se_achica_es_sospechosa():
    """Los grupos son 72 fijos: si decrecen, algo salió mal."""
    reason = _suspicious_reason(_grupos(6), [], {"fase_grupos": _grupos(72)})
    assert reason is not None
    assert "fase_grupos" in reason


def test_fase_grupos_completa_es_ok():
    """Mismo número de grupos (sin eliminatorias todavía) → escribir."""
    assert _suspicious_reason(_grupos(72), [], {"fase_grupos": _grupos(72)}) is None


def test_crece_con_eliminatorias_es_ok():
    """Grupos estables + se agregan eliminatorias resueltas → escribir."""
    new_elims = [{"id": 200, "stage": "round_of_32"}]
    assert _suspicious_reason(_grupos(72), new_elims, {"fase_grupos": _grupos(72)}) is None


def test_primer_sync_sin_fixture_previo_es_ok():
    """Fixture previo vacío (primer arranque) → cualquier cosa con partidos pasa."""
    assert _suspicious_reason(_grupos(6), [], {}) is None


def test_primer_sync_pero_api_vacia_sigue_sospechosa():
    """Sin fixture previo PERO API vacía → igual cortamos."""
    reason = _suspicious_reason([], [], {})
    assert reason is not None
    assert "0 partidos" in reason

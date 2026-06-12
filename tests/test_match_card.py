"""Tests de la tarjeta Telegram por partido (un mensaje editado por fases)."""

from __future__ import annotations

import json

import pytest

from src.notifier.match_card import (
    _portfolio_diff_lines,
    build_match_card,
    upsert_match_card,
)


def _version(phase, exposure, assignment=None, published=False, qa=None):
    v = {
        "phase": phase,
        "run_at": "2026-06-12T16:02:00+00:00",
        "constraints": {"p_home": 0.37, "p_draw": 0.31, "p_away": 0.32,
                        "e_goals_L": 1.13, "e_goals_V": 1.03},
        "assignment_meta": {"exposure": exposure},
        "assignment": assignment or [],
        "published": published,
        "qualitative_adjustment": qa,
    }
    return v


def test_card_single_version_has_core_sections():
    v = _version("T_24h", {"1-0": 6, "1-1": 1}, published=True)
    text = build_match_card("Corea del Sur vs República Checa", "Jue 11/06 23:00 UY", [v])
    assert "Corea del Sur vs República Checa" in text
    assert "37% / 31% / 32%" in text
    assert "1-0" in text and "×6" in text
    assert "T-24h ✓" in text
    assert "Publicado" in text
    assert "RESULTADO" not in text


def test_card_diff_portfolio_level_not_per_penca():
    a1 = [{"penca_id": i, "score": [1, 0]} for i in range(10)] + [
        {"penca_id": 10, "score": [2, 0]}, {"penca_id": 11, "score": [3, 0]}]
    a2 = [{"penca_id": i, "score": [1, 0]} for i in range(8)] + [
        {"penca_id": 8, "score": [0, 2]}, {"penca_id": 9, "score": [1, 1]},
        {"penca_id": 10, "score": [1, 0]}, {"penca_id": 11, "score": [1, 0]}]
    v1 = _version("T_24h", {"1-0": 10, "2-0": 1, "3-0": 1}, a1)
    v2 = _version("T_3h", {"1-0": 10, "0-2": 1, "1-1": 1}, a2)
    lines = _portfolio_diff_lines([v1, v2])
    joined = "\n".join(lines)
    # nivel portfolio: marcadores que entran/salen
    assert "2-0" in joined and "3-0" in joined          # salieron
    assert "0-2" in joined and "1-1" in joined          # entraron
    # los enroques se resumen en una nota, no en 11 líneas
    assert "reasignadas por ranking" in joined
    assert len(lines) <= 3


def test_card_diff_reshuffle_only():
    """Mismos marcadores, otras pencas → una sola línea de 'sin cambios + reasignadas'."""
    a1 = [{"penca_id": 1, "score": [1, 0]}, {"penca_id": 2, "score": [0, 2]}]
    a2 = [{"penca_id": 1, "score": [0, 2]}, {"penca_id": 2, "score": [1, 0]}]
    v1 = _version("T_24h", {"1-0": 1, "0-2": 1}, a1)
    v2 = _version("T_3h", {"1-0": 1, "0-2": 1}, a2)
    lines = _portfolio_diff_lines([v1, v2])
    assert len(lines) == 1
    assert "cobertura sin cambios" in lines[0]
    assert "2 pencas" in lines[0]


def test_card_with_postmortem_shows_result_on_top():
    v = _version("T_30min", {"1-0": 8, "0-2": 1}, published=True)
    pm = {
        "actual_home": 2, "actual_away": 0,
        "pencas_results": [
            {"penca_id": 1654, "predicted_score": [2, 0], "points_earned": 5, "is_exact": True},
            {"penca_id": 1651, "predicted_score": [1, 0], "points_earned": 4, "is_exact": False},
        ],
        "portfolio_total_points": 55,
        "our_best_rank_in_pool": 1,
    }
    text = build_match_card("México vs Sudáfrica", "Jue 11/06 16:00 UY", [v], postmortem=pm)
    assert "RESULTADO: 2-0" in text
    assert "penca 1654" in text and "+5pts" in text and "exacto" in text
    assert "portfolio 55pts" in text
    assert "#1" in text
    # con resultado ya no se muestra el diff
    assert "Cambios" not in text


def test_card_llm_adjustment_shown():
    qa = {"delta_lambda_L": -0.12, "delta_lambda_V": 0.0, "confidence": 0.45,
          "reasoning": "Edson Álvarez afuera según TUDN."}
    v = _version("T_30min", {"1-0": 6}, qa=qa)
    text = build_match_card("México vs Sudáfrica", "Jue 11/06 16:00 UY", [v])
    assert "δL -0.12" in text
    assert "TUDN" in text


def test_card_under_telegram_limit():
    exposure = {f"{i}-{j}": 2 for i in range(4) for j in range(3)}
    versions = [_version(p, exposure) for p in ("T_24h", "T_3h", "T_30min")]
    text = build_match_card("Un Equipo Con Nombre Largo vs Otro Equipo Con Nombre Largo",
                            "Vie 12/06 23:00 UY", versions)
    assert len(text) < 4096


# ---------- upsert: send vs edit ----------

class FakeNotifier:
    def __init__(self, edit_ok=True):
        self.sent, self.edited = [], []
        self.edit_ok = edit_ok

    def send(self, text, parse_mode="HTML"):
        self.sent.append(text)
        return 100 + len(self.sent)

    def edit(self, message_id, text, parse_mode="HTML"):
        if self.edit_ok:
            self.edited.append((message_id, text))
            return True
        return False


def test_upsert_creates_then_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    n = FakeNotifier()
    upsert_match_card(n, "106", "v1 text")
    assert len(n.sent) == 1 and not n.edited
    upsert_match_card(n, "106", "v2 text")
    assert len(n.sent) == 1
    assert n.edited == [(101, "v2 text")]
    # estado persistido
    cards = json.loads((tmp_path / "telegram_cards.json").read_text())
    assert cards == {"106": 101}


def test_upsert_resends_if_edit_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    n = FakeNotifier(edit_ok=False)
    upsert_match_card(n, "106", "v1")
    upsert_match_card(n, "106", "v2")
    assert len(n.sent) == 2  # edit falló → mensaje nuevo
    cards = json.loads((tmp_path / "telegram_cards.json").read_text())
    assert cards["106"] == 102  # re-vinculado al nuevo


def test_upsert_separate_matches_separate_cards(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    n = FakeNotifier()
    upsert_match_card(n, "106", "corea")
    upsert_match_card(n, "107", "canada")
    assert len(n.sent) == 2
    cards = json.loads((tmp_path / "telegram_cards.json").read_text())
    assert set(cards) == {"106", "107"}

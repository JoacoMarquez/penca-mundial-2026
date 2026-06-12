"""Tests de evaluate_llm_adjustment() — ¿el ajuste de Capa 4 ayudó?

Bug que motivó estos tests: la versión anterior usaba constraints.lambda_L +
lambda_V como "base", pero esos λ persistidos YA incluyen el delta del LLM
(el pipeline pisa lam_L/lam_V antes de persistir), y encima les volvía a sumar
δ — aplicaba el ajuste dos veces. El base correcto es λ_post − δ, igual que
llm_counterfactual() en src/dashboard/data_loader.py.
"""

from __future__ import annotations

import pytest

from src.agent import postmortem
from src.agent.postmortem import evaluate_llm_adjustment


# ---------- caso real: match 105 ----------

def test_match_105_real_case():
    """Match 105: λ_post=(1.76, 0.65), δ=(-0.12, 0), resultado real 2-0.

    El LLM bajó λ_local de 1.88 a 1.76, alejándolo de los 2 goles reales del
    local → el ajuste NO ayudó. La fórmula buggy (doble aplicación del delta)
    concluía "yes" para este mismo partido.
    """
    verdict = evaluate_llm_adjustment(
        d_l=-0.12, d_v=0.0, post_l=1.76, post_v=0.65, actual=(2, 0)
    )
    assert verdict == "no"


def test_match_105_old_formula_was_wrong():
    """Documenta el bug: la fórmula vieja decía "yes" en el match 105."""
    d_l, d_v = -0.12, 0.0
    base_total = 1.76 + 0.65          # ya incluía el delta del LLM
    adjusted_total = base_total + d_l + d_v  # delta aplicado DOS veces
    actual_total = 2 + 0
    base_dir = 1 if actual_total > base_total else -1
    adj_dir = 1 if adjusted_total > base_total else -1
    assert ("yes" if base_dir == adj_dir else "no") == "yes"  # veredicto buggy
    # La versión corregida da lo contrario:
    assert evaluate_llm_adjustment(d_l, d_v, 1.76, 0.65, (2, 0)) == "no"


# ---------- direcciones por equipo ----------

def test_helpful_adjustment_up():
    # LLM subió λ_local 1.7→1.9 y el local metió 3 → acercó, "yes"
    assert evaluate_llm_adjustment(0.2, 0.0, 1.9, 0.8, (3, 0)) == "yes"


def test_unhelpful_adjustment_up():
    # LLM subió λ_visit 0.8→1.1 pero el visitante metió 0 → alejó, "no"
    assert evaluate_llm_adjustment(0.0, 0.3, 1.5, 1.1, (1, 0)) == "no"


def test_mixed_adjustment_is_neutral():
    # δ_L acercó (1.8→2.0 con 3 goles) pero δ_V alejó (0.7→1.0 con 0 goles)
    assert evaluate_llm_adjustment(0.2, 0.3, 2.0, 1.0, (3, 0)) == "neutral"


def test_per_team_evaluation_not_just_total():
    """Deltas que se cancelan en el total deben evaluarse por equipo.

    δ=(+0.3, −0.3): el total no cambia, pero δ_L acercó λ_local a los 3 goles
    reales y δ_V acercó λ_visit a los 0 goles reales → ambos ayudaron, "yes".
    El heurístico viejo (solo total de goles) era ciego a esto.
    """
    assert evaluate_llm_adjustment(0.3, -0.3, 1.8, 0.5, (3, 0)) == "yes"


def test_no_adjustment_returns_none():
    assert evaluate_llm_adjustment(0.0, 0.0, 1.5, 1.0, (1, 1)) is None
    assert evaluate_llm_adjustment(0.005, -0.004, 1.5, 1.0, (1, 1)) is None


# ---------- integración: compute_postmortem con datos del match 105 ----------

def test_compute_postmortem_match_105(monkeypatch):
    pred = {
        "constraints": {"lambda_L": 1.76, "lambda_V": 0.65, "lambda_12": 0.1},
        "qualitative_adjustment": {
            "delta_lambda_L": -0.12,
            "delta_lambda_V": 0.0,
            "reasoning": "baja del local por rotación",
            "confidence": 0.6,
        },
        "assignment": [
            {"penca_id": 1, "score": [2, 0], "objective": "ev"},
            {"penca_id": 2, "score": [1, 0], "objective": "differentiated"},
        ],
    }
    status = {
        "home_score": 2,
        "away_score": 0,
        "home_team": {"name": "Local"},
        "away_team": {"name": "Visitante"},
    }
    monkeypatch.setattr(postmortem, "_latest_prediction", lambda mid: pred)
    monkeypatch.setattr(postmortem, "_fetch_match_status", lambda mid: status)
    monkeypatch.setattr(postmortem, "_fetch_full_leaderboard", lambda: None)

    report = postmortem.compute_postmortem(105)
    assert report is not None
    assert report.llm_adjustment_was_helpful == "no"
    # El insight de "dirección correcta" no debe aparecer
    assert not any("dirección correcta" in i for i in report.insights)
    assert any("sentido contrario" in i for i in report.insights)

"""Tests del guardrail de Capa 4 cuando no hay XI confirmado (bound especulativo)."""

from src.model.qualitative import (
    MAX_ABS_ADJUSTMENT,
    MAX_ABS_ADJUSTMENT_SPECULATIVE,
    MatchContext,
    QualitativeAdjustment,
    build_user_prompt,
)


def _ctx(**kw) -> MatchContext:
    base = dict(
        home_team="Australia", away_team="Türkiye", kickoff_local="x", stage="group",
        market_p_home=0.18, market_p_draw=0.26, market_p_away=0.56,
        market_e_goals_L=0.9, market_e_goals_V=1.7,
    )
    base.update(kw)
    return MatchContext(**base)


def test_speculative_bound_is_tighter():
    assert MAX_ABS_ADJUSTMENT_SPECULATIVE < MAX_ABS_ADJUSTMENT
    assert MAX_ABS_ADJUSTMENT_SPECULATIVE == 0.10


def test_clipped_default_uses_full_bound():
    adj = QualitativeAdjustment(-0.50, 0.40, "r", 0.5, {}).clipped()
    assert adj.delta_lambda_L == -MAX_ABS_ADJUSTMENT
    assert adj.delta_lambda_V == MAX_ABS_ADJUSTMENT


def test_clipped_speculative_caps_to_010():
    # El caso Güler: -0.18 sin XI debe quedar acotado a -0.10.
    adj = QualitativeAdjustment(0.0, -0.18, "Güler no juega", 0.55, {}).clipped(
        MAX_ABS_ADJUSTMENT_SPECULATIVE
    )
    assert adj.delta_lambda_V == -MAX_ABS_ADJUSTMENT_SPECULATIVE
    assert adj.delta_lambda_L == 0.0


def test_clipped_speculative_leaves_small_adjustments_intact():
    adj = QualitativeAdjustment(0.05, -0.08, "r", 0.4, {}).clipped(
        MAX_ABS_ADJUSTMENT_SPECULATIVE
    )
    assert adj.delta_lambda_L == 0.05
    assert adj.delta_lambda_V == -0.08


def test_prompt_warns_when_no_confirmed_xi():
    prompt = build_user_prompt(_ctx(no_confirmed_xi=True))
    assert "SIN XI CONFIRMADO" in prompt
    assert "±0.10" in prompt


def test_prompt_no_warning_when_xi_present():
    prompt = build_user_prompt(_ctx(no_confirmed_xi=False, lineup_confirmed=True,
                                    home_xi=["A"], away_xi=["B"]))
    assert "SIN XI CONFIRMADO" not in prompt


# ---- bandera "sin XI confirmado" (alerts) ----

def test_no_xi_flag_fires_without_xi_and_with_adjustment():
    from src.agent.alerts import check_no_confirmed_xi
    qa = {"delta_lambda_L": 0.0, "delta_lambda_V": -0.10, "reasoning": "Güler no juega"}
    flag = check_no_confirmed_xi(qa, home_xi=None, away_xi=None)
    assert flag is not None and flag.code == "no_xi"


def test_no_xi_flag_silent_when_xi_present():
    from src.agent.alerts import check_no_confirmed_xi
    qa = {"delta_lambda_L": 0.0, "delta_lambda_V": -0.10}
    assert check_no_confirmed_xi(qa, home_xi=["A"], away_xi=["B"]) is None


def test_no_xi_flag_silent_when_no_adjustment():
    from src.agent.alerts import check_no_confirmed_xi
    qa = {"delta_lambda_L": 0.0, "delta_lambda_V": 0.0}
    assert check_no_confirmed_xi(qa, home_xi=None, away_xi=None) is None

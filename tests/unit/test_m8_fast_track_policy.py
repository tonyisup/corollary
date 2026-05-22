import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from thalamus.attributor import AttributionResult, SourceVerdict
from pfc.agentive_controller import AgentiveController, AgentState
from pfc.m8_fast_track import M8FastTrackAction, M8FastTrackPolicy


def _world_result(*, metadata=None, verdict=SourceVerdict.WORLD, is_speech=True):
    return AttributionResult(
        verdict=verdict,
        confidence=0.9,
        self_match_score=0.1,
        is_in_motor_window=False,
        causal_score=0.0,
        energy_db=-22.0,
        is_speech=is_speech,
        metadata=metadata,
    )


def test_m8_fast_track_policy_interrupts_high_salience_world_speech_during_speak():
    policy = M8FastTrackPolicy()
    ctrl = AgentiveController(m8_policy=policy)
    assert ctrl.request_output("speaking now") is True

    decision = ctrl.process_attribution(
        _world_result(metadata={"m4_disagreed": True, "m4_degraded_mode": True}),
        current_time=1.0,
    )

    assert decision.state == AgentState.LISTEN
    assert decision.should_interrupt is True
    assert decision.m8_action == M8FastTrackAction.INTERRUPT
    assert decision.m8_score >= policy.interrupt_threshold
    assert "WORLD" in decision.m8_reason
    assert "m4" in decision.m8_reason.lower()


def test_m8_fast_track_policy_keeps_self_low_urgency_when_speaking():
    policy = M8FastTrackPolicy()
    ctrl = AgentiveController(m8_policy=policy)
    assert ctrl.request_output("speaking now") is True

    decision = ctrl.process_attribution(
        AttributionResult(
            verdict=SourceVerdict.SELF,
            confidence=0.95,
            self_match_score=0.9,
            is_in_motor_window=True,
            causal_score=0.9,
            energy_db=-18.0,
            is_speech=False,
        ),
        current_time=1.0,
    )

    assert decision.should_interrupt is False
    assert decision.m8_action == M8FastTrackAction.CONTINUE
    assert decision.m8_score < policy.warn_threshold
    assert decision.m8_valence in {"neutral", "positive"}


def test_m8_fast_track_policy_never_interrupts_non_speech_even_under_low_thresholds():
    policy = M8FastTrackPolicy(warn_threshold=0.05, interrupt_threshold=0.10, escalate_threshold=0.15)
    decision = policy.evaluate(
        AttributionResult(
            verdict=SourceVerdict.WORLD,
            confidence=0.9,
            self_match_score=0.0,
            is_in_motor_window=False,
            causal_score=0.0,
            energy_db=-35.0,
            is_speech=False,
            metadata={
                "m4_disagreed": True,
                "m4_degraded_mode": True,
                "m4_final_is_self": False,
                "m4_bank_is_self": True,
                "m4_scorer_is_self": False,
            },
        ),
        controller_state=AgentState.SPEAK,
        current_time=1.0,
    )

    assert decision.action == M8FastTrackAction.CONTINUE
    assert decision.score <= 0.10
    assert decision.urgency == "low"
    assert "interrupt" not in decision.reason.lower()
    assert "escalation" not in decision.reason.lower()


def test_m8_disagreement_boost_ignored_on_low_confidence():
    """M4 disagreement should not push score up when confidence is low."""
    policy = M8FastTrackPolicy()
    decision = policy.evaluate(
        _world_result(
            metadata={
                "m4_disagreed": True,
                "m4_degraded_mode": True,
                "m4_final_is_self": False,
                "m4_bank_is_self": True,
                "m4_scorer_is_self": False,
            },
            verdict=SourceVerdict.BOTH,
            is_speech=False,
        ),
        controller_state=AgentState.IDLE,
        current_time=1.0,
    )
    # Non-speech + disagreement → all boosts ignored
    assert decision.score <= 0.10
    assert "ignored" in decision.reason.lower()


def test_m8_disagreement_ignored_when_bank_trusts_self_during_speak():
    """When bank says SELF and agent is speaking, disagreement should not fast-track."""
    policy = M8FastTrackPolicy()
    decision = policy.evaluate(
        _world_result(
            metadata={
                "m4_disagreed": True,
                "m4_degraded_mode": True,
                "m4_final_is_self": False,
                "m4_bank_is_self": True,
                "m4_scorer_is_self": False,
            },
            verdict=SourceVerdict.BOTH,
        ),
        controller_state=AgentState.SPEAK,
        current_time=1.0,
    )
    # Bank trusts SELF + agent speaking → disagreement ignored, score stays at base
    assert decision.score < policy.interrupt_threshold
    assert "bank trusts SELF" in decision.reason
    assert decision.action != M8FastTrackAction.INTERRUPT
    assert decision.action != M8FastTrackAction.ESCALATE


def test_m8_disagreement_boost_applies_on_high_confidence_world_speech():
    """M4 disagreement should boost score for real WORLD speech during SPEAK."""
    policy = M8FastTrackPolicy()
    decision = policy.evaluate(
        _world_result(
            metadata={
                "m4_disagreed": True,
                "m4_degraded_mode": True,
                "m4_final_is_self": False,
                "m4_bank_is_self": False,
                "m4_scorer_is_self": False,
            },
            verdict=SourceVerdict.WORLD,
        ),
        controller_state=AgentState.SPEAK,
        current_time=1.0,
    )
    # WORLD speech + high confidence + disagreement → score should be elevated
    assert policy.warn_threshold <= decision.score <= 1.0
    assert "M4 disagreement raises salience" in decision.reason


def test_m8_bank_scorer_disagreement_ignored_on_low_confidence():
    """Bank/scorer disagreement on low-confidence frames should not boost salience."""
    policy = M8FastTrackPolicy()
    low_conf_result = AttributionResult(
        verdict=SourceVerdict.WORLD,
        confidence=0.1,
        self_match_score=0.0,
        is_in_motor_window=False,
        causal_score=0.0,
        energy_db=-50.0,
        is_speech=True,
        metadata={
            "m4_bank_is_self": True,
            "m4_scorer_is_self": False,
        },
    )
    decision = policy.evaluate(
        low_conf_result,
        controller_state=AgentState.IDLE,
        current_time=1.0,
    )
    assert "ignored" in decision.reason.lower()
    assert decision.score < policy.interrupt_threshold

def test_m8_fast_track_policy_warns_on_high_salience_world_when_not_speaking():
    policy = M8FastTrackPolicy()
    decision = policy.evaluate(
        _world_result(),
        controller_state=AgentState.IDLE,
        current_time=1.0,
    )

    assert decision.action == M8FastTrackAction.WARN
    assert decision.score >= policy.warn_threshold
    assert decision.urgency == "medium"
    assert "WORLD" in decision.reason


def test_m8_cooldown_suppresses_repeated_action():
    """Same fast-track action within cooldown window should be suppressed."""
    policy = M8FastTrackPolicy(cooldown_s=0.5)
    result = _world_result()
    state = AgentState.IDLE

    # First call: should WARN
    d1 = policy.evaluate(result, controller_state=state, current_time=1.0)
    assert d1.action == M8FastTrackAction.WARN

    # Second call within cooldown: should be suppressed to CONTINUE
    d2 = policy.evaluate(result, controller_state=state, current_time=1.1)
    assert d2.action == M8FastTrackAction.CONTINUE
    assert "cooldown" in d2.reason.lower()

    # Third call after cooldown: should WARN again
    d3 = policy.evaluate(result, controller_state=state, current_time=2.0)
    assert d3.action == M8FastTrackAction.WARN


def test_m8_cooldown_does_not_affect_different_actions():
    """Different fast-track actions should not trigger cooldown."""
    policy = M8FastTrackPolicy(cooldown_s=0.5)

    # First: WARN on WORLD (score 0.66, below warn_threshold 0.65 → actually CONTINUE)
    # Use a higher-salience input to get a clear WARN
    high_salience = _world_result(
        metadata={"m4_disagreed": True, "m4_degraded_mode": True}
    )
    d1 = policy.evaluate(high_salience, controller_state=AgentState.IDLE, current_time=1.0)
    assert d1.action == M8FastTrackAction.WARN

    # Second: INTERRUPT on BOTH during SPEAK with high disagreement
    # This should be a different action (INTERRUPT vs WARN), so no cooldown
    d2 = policy.evaluate(
        _world_result(
            verdict=SourceVerdict.BOTH,
            metadata={"m4_disagreed": True, "m4_degraded_mode": True, "m4_final_is_self": False,
                      "m4_bank_is_self": False, "m4_scorer_is_self": False},
        ),
        controller_state=AgentState.SPEAK,
        current_time=1.1,
    )
    # BOTH+SPEAK with disagreement → score 0.70+0.12+0.08+0.10 = 1.0 → INTERRUPT
    assert d2.action == M8FastTrackAction.INTERRUPT


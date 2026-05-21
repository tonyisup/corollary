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


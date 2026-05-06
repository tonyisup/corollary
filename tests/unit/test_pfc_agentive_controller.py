import pytest
import sys
import os

# Ensure packages is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from thalamus.attributor import AttributionResult, SourceVerdict
from pfc.agentive_controller import AgentiveController, AgentState


def test_controller_enters_listen_state():
    ctrl = AgentiveController(listen_threshold_ms=100)

    res = AttributionResult(
        verdict=SourceVerdict.WORLD,
        confidence=0.9,
        self_match_score=0.1,
        is_in_motor_window=False,
        causal_score=0.0,
        energy_db=-30.0,
        is_speech=True,
    )

    ctrl.process_attribution(res, current_time=1.0)
    assert ctrl.speech_start_time == 1.0
    assert ctrl.state == AgentState.IDLE

    ctrl.process_attribution(res, current_time=1.15)
    assert ctrl.state == AgentState.LISTEN


def test_controller_defers_output():
    ctrl = AgentiveController(listen_threshold_ms=100)
    res = AttributionResult(
        verdict=SourceVerdict.WORLD,
        confidence=0.9,
        self_match_score=0.1,
        is_in_motor_window=False,
        causal_score=0.0,
        energy_db=-30.0,
        is_speech=True,
    )

    ctrl.process_attribution(res, current_time=1.0)
    ctrl.process_attribution(res, current_time=1.15)
    assert ctrl.state == AgentState.LISTEN

    accepted = ctrl.request_output("Hello")
    assert accepted is False
    assert ctrl.has_deferred_output is True
    assert ctrl.deferred_utterance == "Hello"


def test_controller_recovers_to_idle():
    ctrl = AgentiveController(listen_threshold_ms=100, cooldown_ms=200)
    res_speech = AttributionResult(
        verdict=SourceVerdict.WORLD,
        confidence=0.9,
        self_match_score=0.1,
        is_in_motor_window=False,
        causal_score=0.0,
        energy_db=-30.0,
        is_speech=True,
    )
    res_silence = AttributionResult(
        verdict=SourceVerdict.SILENCE,
        confidence=1.0,
        self_match_score=0.0,
        is_in_motor_window=False,
        causal_score=0.0,
        energy_db=-60.0,
        is_speech=False,
    )

    ctrl.process_attribution(res_speech, current_time=1.0)
    ctrl.process_attribution(res_speech, current_time=1.15)
    assert ctrl.state == AgentState.LISTEN

    ctrl.process_attribution(res_silence, current_time=1.45)
    assert ctrl.state == AgentState.IDLE

    accepted = ctrl.request_output("Hello")
    assert accepted is True
    assert ctrl.has_deferred_output is False


def test_controller_pop_deferred_output_clears_queue():
    ctrl = AgentiveController()
    ctrl.deferred_utterance = "Queued response"

    assert ctrl.has_deferred_output is True
    assert ctrl.pop_deferred_output() == "Queued response"
    assert ctrl.has_deferred_output is False
    assert ctrl.pop_deferred_output() is None


def test_controller_interrupts_active_speech_on_external_overlap():
    ctrl = AgentiveController(listen_threshold_ms=100, cooldown_ms=200)
    self_res = AttributionResult(
        verdict=SourceVerdict.SELF,
        confidence=0.9,
        self_match_score=0.8,
        is_in_motor_window=True,
        causal_score=0.9,
        energy_db=-20.0,
        is_speech=False,
    )
    overlap_res = AttributionResult(
        verdict=SourceVerdict.BOTH,
        confidence=0.9,
        self_match_score=0.6,
        is_in_motor_window=False,  # external overlap, not self
        causal_score=0.5,
        energy_db=-20.0,
        is_speech=True,
    )

    assert ctrl.request_output("speaking now") is True
    assert ctrl.state == AgentState.SPEAK

    ctrl.process_attribution(self_res, current_time=1.0)
    decision = ctrl.process_attribution(overlap_res, current_time=1.15)

    assert decision.state == AgentState.LISTEN
    assert decision.can_speak is False
    assert decision.should_interrupt is True
    assert ctrl.has_deferred_output is True
    assert ctrl.deferred_utterance == "speaking now"

import pytest
import sys
import os

# Ensure packages is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from thalamus.attributor import AttributionResult, SourceVerdict
from pfc.agentive_controller import AgentiveController, AgentState

def test_controller_enters_listen_state():
    ctrl = AgentiveController(listen_threshold_ms=100)
    
    # Send a world-speech result using keyword arguments
    res = AttributionResult(
        verdict=SourceVerdict.WORLD,
        confidence=0.9,
        self_match_score=0.1,
        is_in_motor_window=False,
        causal_score=0.0,
        energy_db=-30.0,
        is_speech=True
    )
    
    # 0ms: start speech
    ctrl.process_attribution(res, current_time=1.0)
    assert ctrl.speech_start_time == 1.0
    assert ctrl.state == AgentState.IDLE
    
    # 150ms: threshold met
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
        is_speech=True
    )
    
    ctrl.process_attribution(res, current_time=1.0)
    ctrl.process_attribution(res, current_time=1.15)
    assert ctrl.state == AgentState.LISTEN
    
    accepted = ctrl.request_output("Hello")
    assert accepted == False
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
        is_speech=True
    )
    res_silence = AttributionResult(
        verdict=SourceVerdict.SILENCE,
        confidence=1.0,
        self_match_score=0.0,
        is_in_motor_window=False,
        causal_score=0.0,
        energy_db=-60.0,
        is_speech=False
    )
    
    # Enter LISTEN
    ctrl.process_attribution(res_speech, current_time=1.0)
    ctrl.process_attribution(res_speech, current_time=1.15)
    assert ctrl.state == AgentState.LISTEN
    
    # Silence for 300ms (threshold 200ms)
    ctrl.process_attribution(res_silence, current_time=1.45)
    assert ctrl.state == AgentState.IDLE
    
    accepted = ctrl.request_output("Hello")
    assert accepted == True

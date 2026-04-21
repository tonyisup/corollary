import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from pfc.agentive_controller import AgentiveController, AgentState
from thalamus.attributor import AttributionResult, SourceVerdict

def test_controller_debug():
    ctrl = AgentiveController(listen_threshold_ms=100)
    res = AttributionResult(SourceVerdict.WORLD, 0.9, 0.1, False, 0.0, -30.0, True)
    
    print(f"\n1.0: Start")
    ctrl.process_attribution(res, current_time=1.0)
    print(f"State: {ctrl.state}, Start: {ctrl.speech_start_time}")
    
    print(f"1.15: 150ms later")
    ctrl.process_attribution(res, current_time=1.15)
    print(f"State: {ctrl.state}, Start: {ctrl.speech_start_time}")
    
    assert ctrl.state == AgentState.LISTEN

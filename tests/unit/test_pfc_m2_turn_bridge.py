from thalamus.attributor import AttributionResult, SourceVerdict
from pfc.agentive_controller import AgentiveController, AgentState
from pfc.m2_turn_bridge import M2Frame, run_turn_taking_bridge


def attr(verdict: SourceVerdict, is_speech: bool, is_in_motor_window: bool | None = None) -> AttributionResult:
    if is_in_motor_window is None:
        is_in_motor_window = verdict in {SourceVerdict.SELF, SourceVerdict.BOTH}
    return AttributionResult(
        verdict=verdict,
        confidence=0.9,
        self_match_score=0.1,
        is_in_motor_window=is_in_motor_window,
        causal_score=0.0,
        energy_db=-30.0 if verdict != SourceVerdict.SILENCE else -70.0,
        is_speech=is_speech,
    )


def test_controller_treats_both_speech_as_agentive_external_speech() -> None:
    ctrl = AgentiveController(listen_threshold_ms=100)
    # BOTH with is_in_motor_window=False = external overlap (not self-caused)
    both_speech = AttributionResult(
        verdict=SourceVerdict.BOTH, confidence=0.9, self_match_score=0.1,
        is_in_motor_window=False, causal_score=0.0, energy_db=-30.0, is_speech=True,
    )

    ctrl.process_attribution(both_speech, current_time=1.0)
    assert ctrl.state == AgentState.IDLE

    decision = ctrl.process_attribution(both_speech, current_time=1.15)
    assert decision.state == AgentState.LISTEN
    assert decision.can_speak is False


def test_controller_can_disable_both_speech_agentive_policy() -> None:
    ctrl = AgentiveController(listen_threshold_ms=100, treat_both_speech_as_agentive=False)
    both_speech = attr(SourceVerdict.BOTH, is_speech=True)

    ctrl.process_attribution(both_speech, current_time=1.0)
    decision = ctrl.process_attribution(both_speech, current_time=1.15)
    assert decision.state == AgentState.IDLE
    assert decision.can_speak is True


def test_m2_turn_bridge_defers_then_resumes_after_cooldown() -> None:
    ctrl = AgentiveController(listen_threshold_ms=100, cooldown_ms=200)
    frames = [
        M2Frame(1.00, attr(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)),
        M2Frame(1.15, attr(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)),
        M2Frame(1.50, attr(SourceVerdict.SILENCE, is_speech=False)),
    ]

    result = run_turn_taking_bridge(frames, "hello after you", controller=ctrl)

    assert result.accepted_initial_output is False
    assert result.deferred_output is None
    assert result.resumed_output == "hello after you"
    assert result.decisions[-1].state == AgentState.IDLE


def test_m2_turn_bridge_accepts_output_when_no_agentive_speech() -> None:
    ctrl = AgentiveController(listen_threshold_ms=100, cooldown_ms=200)
    frames = [
        M2Frame(1.00, attr(SourceVerdict.SELF, is_speech=False)),
        M2Frame(1.15, attr(SourceVerdict.SILENCE, is_speech=False)),
    ]

    result = run_turn_taking_bridge(frames, "speaking now", controller=ctrl, request_at_s=1.0)

    assert result.accepted_initial_output is True
    assert result.deferred_output is None
    assert result.resumed_output is None


def test_m2_turn_bridge_interrupts_speaking_then_resumes_after_overlap() -> None:
    ctrl = AgentiveController(listen_threshold_ms=100, cooldown_ms=200)
    frames = [
        M2Frame(1.00, attr(SourceVerdict.SELF, is_speech=False)),
        M2Frame(1.15, attr(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)),
        M2Frame(1.50, attr(SourceVerdict.SILENCE, is_speech=False)),
    ]

    result = run_turn_taking_bridge(frames, "speaking now", controller=ctrl, request_at_s=1.0)

    assert result.accepted_initial_output is True
    assert result.decisions[1].should_interrupt is True
    assert result.decisions[1].state == AgentState.LISTEN
    assert result.deferred_output is None
    assert result.resumed_output == "speaking now"

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


def attr_with_ratio(
    verdict: SourceVerdict,
    is_speech: bool,
    is_in_motor_window: bool,
    ratio: float | None,
    match_score: float = 0.1,
) -> AttributionResult:
    result = attr(verdict, is_speech=is_speech, is_in_motor_window=is_in_motor_window)
    object.__setattr__(result, "self_match_score_per_frame", ratio)
    object.__setattr__(result, "self_match_score", match_score)
    return result


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
    assert result.decisions[1].collision_repair_text == "Sorry, go ahead."


def test_collision_repair_can_be_disabled_or_customized() -> None:
    ctrl = AgentiveController(
        listen_threshold_ms=100,
        cooldown_ms=200,
        collision_repair_text="Pardon me — you go first.",
    )
    frames = [
        M2Frame(1.00, attr(SourceVerdict.SELF, is_speech=False)),
        M2Frame(1.15, attr(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)),
        M2Frame(1.50, attr(SourceVerdict.SILENCE, is_speech=False)),
    ]

    result = run_turn_taking_bridge(frames, "speaking now", controller=ctrl, request_at_s=1.0)

    repair_texts = [d.collision_repair_text for d in result.decisions if d.collision_repair_text]
    assert repair_texts == ["Pardon me — you go first."]
    assert result.resumed_output == "speaking now"

    no_repair_ctrl = AgentiveController(
        listen_threshold_ms=100,
        cooldown_ms=200,
        collision_repair_text=None,
    )
    result_without_repair = run_turn_taking_bridge(
        frames,
        "speaking now",
        controller=no_repair_ctrl,
        request_at_s=1.0,
    )
    assert any(d.should_interrupt for d in result_without_repair.decisions)
    assert all(d.collision_repair_text is None for d in result_without_repair.decisions)


def test_motor_window_silence_does_not_cancel_pending_speech_before_overlap_resume() -> None:
    ctrl = AgentiveController(
        listen_threshold_ms=100,
        cooldown_ms=200,
        both_external_energy_ratio_threshold=1.2,
        both_external_min_frames=3,
    )
    frames = [
        # request_output happens after this frame; subsequent low-energy motor
        # silence must not kick SPEAK back to IDLE before playback/overlap.
        M2Frame(0.00, attr(SourceVerdict.SILENCE, is_speech=False, is_in_motor_window=False)),
        M2Frame(0.03, attr(SourceVerdict.SILENCE, is_speech=False, is_in_motor_window=True)),
        M2Frame(0.06, attr_with_ratio(SourceVerdict.BOTH, True, True, ratio=0.5, match_score=0.5)),
        M2Frame(1.00, attr_with_ratio(SourceVerdict.BOTH, True, True, ratio=20.0, match_score=20.0)),
        M2Frame(1.03, attr_with_ratio(SourceVerdict.BOTH, True, True, ratio=30.0, match_score=30.0)),
        M2Frame(1.06, attr_with_ratio(SourceVerdict.BOTH, True, True, ratio=40.0, match_score=40.0)),
        # After overlap hysteresis expires, low ratio means human overlap ended.
        # High match_score should not consume deferred output via pre-playback
        # LISTEN→SPEAK recovery because this was a real overlap interruption.
        M2Frame(1.50, attr_with_ratio(SourceVerdict.BOTH, True, True, ratio=0.0, match_score=0.5)),
    ]

    result = run_turn_taking_bridge(frames, "resume this", controller=ctrl, request_at_s=0.0)

    assert result.accepted_initial_output is True
    assert any(d.should_interrupt for d in result.decisions)
    assert result.deferred_output is None
    assert result.resumed_output == "resume this"


def test_m2_turn_bridge_defers_request_at_agent_start_when_user_already_speaking() -> None:
    ctrl = AgentiveController(listen_threshold_ms=100, cooldown_ms=200)
    frames = [
        M2Frame(0.00, attr(SourceVerdict.SILENCE, is_speech=False, is_in_motor_window=False)),
        M2Frame(0.30, attr(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)),
        M2Frame(0.45, attr(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)),
        M2Frame(0.51, attr(SourceVerdict.BOTH, is_speech=True, is_in_motor_window=False)),
        M2Frame(0.90, attr(SourceVerdict.SILENCE, is_speech=False, is_in_motor_window=False)),
    ]

    result = run_turn_taking_bridge(
        frames,
        "wait until user finishes",
        controller=ctrl,
        request_at_s=0.5,
    )

    assert result.accepted_initial_output is False
    assert result.deferred_output is None
    assert result.resumed_output == "wait until user finishes"

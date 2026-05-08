"""M3 bridge from M2 attribution frames to agentive turn-taking.

This module keeps the first M3 slice deliberately small: M2 emits attribution
frames (`SELF`, `WORLD`, `BOTH`, speech flag). The bridge feeds those frames to
`AgentiveController`, whose job is to defer agent output while agentive external
speech is active and resume after cooldown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from thalamus.attributor import AttributionResult

from .agentive_controller import AgentiveController, ControllerDecision


@dataclass(frozen=True)
class M2Frame:
    """A timestamped M2 attribution frame for M3 turn-taking."""

    time_s: float
    attribution: AttributionResult


@dataclass(frozen=True)
class M3BridgeResult:
    """Result of feeding an M2 frame sequence to the controller."""

    decisions: list[ControllerDecision]
    accepted_initial_output: bool
    deferred_output: str | None
    resumed_output: str | None


def run_turn_taking_bridge(
    frames: Iterable[M2Frame],
    requested_output: str,
    controller: AgentiveController | None = None,
    request_at_s: float | None = None,
) -> M3BridgeResult:
    """Feed M2 frames to an `AgentiveController` and test deferral/resume.

    Parameters
    ----------
    frames:
        Timestamped M2 attribution frames.
    requested_output:
        Text the agent wants to speak once the controller is listening.
    controller:
        Optional preconfigured controller.
    request_at_s:
        First frame time at/after which to request output. If omitted, output is
        requested when the controller first enters LISTEN.
    """
    ctrl = controller or AgentiveController()
    decisions: list[ControllerDecision] = []
    accepted_initial_output = True
    requested = False

    for frame in frames:
        decision = ctrl.process_attribution(frame.attribution, current_time=frame.time_s)
        decisions.append(decision)
        should_request = request_at_s is not None and frame.time_s >= request_at_s
        should_request = should_request or (request_at_s is None and not requested and not decision.can_speak)
        if should_request and not requested:
            accepted_initial_output = ctrl.request_output(requested_output, current_time=frame.time_s)
            requested = True

    resumed_output = None
    if requested and ctrl.state.name == "IDLE":
        resumed_output = ctrl.pop_deferred_output()

    return M3BridgeResult(
        decisions=decisions,
        accepted_initial_output=accepted_initial_output,
        deferred_output=ctrl.deferred_utterance,
        resumed_output=resumed_output,
    )

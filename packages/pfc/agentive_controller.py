"""PFC — Agentive Controller for Turn-Taking.

M3: Distinguishes between agentive vocal others and ambient noise.
Manages contingent listening and turn-taking policies.
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from thalamus.attributor import SourceVerdict, AttributionResult

class AgentState(Enum):
    IDLE = 0
    LISTEN = 1  # Hearing world-speech
    SPEAK = 2   # Emitting self-audio

@dataclass
class ControllerDecision:
    state: AgentState
    can_speak: bool
    should_interrupt: bool
    deferred_text: Optional[str] = None
    collision_repair_text: Optional[str] = None

class AgentiveController:
    """The PFC controller responsible for social Contingency.
    
    Decides when to speak and when to listen based on M3 goals.
    Deferred output is represented explicitly as queued text, not as a
    separate controller state.
    """

    def __init__(
        self,
        listen_threshold_ms: float = 300.0,
        cooldown_ms: float = 500.0,
        treat_both_speech_as_agentive: bool = True,
        motor_holdoff_ms: float = 200.0,
        both_external_energy_ratio_threshold: float = 2.0,
        both_external_min_frames: int = 7,
        both_external_listen_min_ms: float = 300.0,
        collision_repair_text: Optional[str] = "Sorry, go ahead.",
    ) -> None:
        self.state = AgentState.IDLE
        self.listen_threshold_ms = listen_threshold_ms
        self.cooldown_ms = cooldown_ms
        self.treat_both_speech_as_agentive = treat_both_speech_as_agentive
        self.motor_holdoff_ms = motor_holdoff_ms
        self.both_external_energy_ratio_threshold = both_external_energy_ratio_threshold
        self.both_external_min_frames = both_external_min_frames
        self.both_external_listen_min_ms = both_external_listen_min_ms
        self.collision_repair_text = collision_repair_text

        self.last_speech_time: float = 0.0
        self.speech_start_time: float = 0.0
        self.last_self_time: float = 0.0
        self.active_utterance: Optional[str] = None
        self.deferred_utterance: Optional[str] = None
        self._request_output_time: float = 0.0
        self._motor_end_time: float = 0.0
        self._was_in_motor_window: bool = False
        self.motor_end_holdoff_ms = motor_holdoff_ms  # reuse same value
        # Overlap detection state
        self._both_external_count: int = 0
        self._overlap_listen_start_time: float = 0.0
        self._was_overlap_interrupted: bool = False

    @property
    def has_deferred_output(self) -> bool:
        """Whether the controller currently has queued output."""
        return self.deferred_utterance is not None

    def pop_deferred_output(self) -> Optional[str]:
        """Return and clear queued output, if any."""
        text = self.deferred_utterance
        self.deferred_utterance = None
        return text

    def process_attribution(
        self, 
        res: AttributionResult, 
        current_time: float = None
    ) -> ControllerDecision:
        """Update internal state based on latest attribution result.
        
        Parameters
        ----------
        res : AttributionResult
            The latest verdict from the Thalamus.
        current_time : float, optional
            Manual clock for testing.
        """
        if current_time is None:
            current_time = time.time()

        # 0. Pre-playback LISTEN recovery: if the controller entered LISTEN due
        # to ambient noise shortly before the agent started playing, and the
        # current frame shows BOTH with high match_score (indicating the agent's
        # own playback is now arriving), recover to SPEAK.
        if (
            self.state == AgentState.LISTEN
            and self._request_output_time > 0
            and self.speech_start_time > 0
            and (self._request_output_time - self.speech_start_time) < (self.motor_holdoff_ms / 1000.0)
            and res.verdict == SourceVerdict.BOTH
            and res.is_speech
            and res.self_match_score >= 0.03  # above noise floor
        ):
            self.state = AgentState.SPEAK
            self.active_utterance = self.deferred_utterance
            self.deferred_utterance = None
            self.speech_start_time = 0
            self.last_speech_time = 0

        # 1. Classify the frame for M3 policy.
        is_world_speech = res.is_speech and res.verdict == SourceVerdict.WORLD
        
        # BOTH frames during self-playback (is_in_motor_window) are treated as
        # self-activity by default. However, if the per-frame energy ratio
        # (observed / expected self RMS) is significantly above 1.0 for a
        # sustained period, this indicates human speech is overlapping.
        #
        # The per-frame energy ratio is computed by the attributor by comparing
        # the observed frame's RMS against the expected self frame's RMS from
        # the agent PCM template.
        #
        both_is_external = False
        if (res.verdict == SourceVerdict.BOTH
            and res.is_in_motor_window
            and res.self_match_score_per_frame is not None):
            if res.self_match_score_per_frame > self.both_external_energy_ratio_threshold:
                self._both_external_count += 1
                if self._both_external_count >= self.both_external_min_frames:
                    both_is_external = True
            else:
                self._both_external_count = 0
        else:
            self._both_external_count = 0
        
        # Track when motor window ends (for post-playback ring-down holdoff)
        # Only trigger on the actual transition from in-window to out-of-window.
        if res.is_in_motor_window:
            self._was_in_motor_window = True
            self._motor_end_time = 0.0
        elif self._was_in_motor_window and self._motor_end_time == 0.0:
            # Motor window just ended — record the time
            self._motor_end_time = current_time
            self._was_in_motor_window = False
        
        is_overlap_speech = (
            self.treat_both_speech_as_agentive
            and res.is_speech
            and res.verdict == SourceVerdict.BOTH
            and (not res.is_in_motor_window or both_is_external)
        )
        is_agentive_external_speech = is_world_speech or is_overlap_speech
        is_self_activity = res.verdict == SourceVerdict.SELF or (
            res.verdict == SourceVerdict.BOTH and not is_overlap_speech
        )

        should_interrupt = False
        just_interrupted = False
        collision_repair_text: Optional[str] = None

        # 2. SPEAK is a real lifecycle: self frames sustain it; external speech
        # interrupts it immediately and queues the active utterance for later.
        if self.state == AgentState.SPEAK:
            # Post-motor holdoff: after the motor window ends, suppress
            # WORLD→LISTEN transitions briefly. This handles the acoustic
            # ring-down of TTS in the room after playback stops.
            # Only suppresses if the "speech" started while the agent was
            # still playing (i.e., it's ring-down, not new human speech).
            in_post_motor_holdoff = (
                res.verdict == SourceVerdict.WORLD
                and self._motor_end_time > 0
                and self.speech_start_time > 0
                and self.speech_start_time < self._motor_end_time
                and (current_time - self._motor_end_time) < (self.motor_end_holdoff_ms / 1000.0)
            )
            if is_agentive_external_speech and not in_post_motor_holdoff:
                should_interrupt = True
                just_interrupted = True
                collision_repair_text = self.collision_repair_text
                if self.active_utterance is not None:
                    self.deferred_utterance = self.active_utterance
                self.active_utterance = None
                self.state = AgentState.LISTEN
                if self.speech_start_time == 0:
                    self.speech_start_time = current_time
                self.last_speech_time = current_time
                pass  # interrupt handled
                if both_is_external:
                    self._overlap_listen_start_time = current_time
                    self._was_overlap_interrupted = True
            elif is_self_activity:
                self.last_self_time = current_time
            else:
                # SILENCE/non-agentive frames during the motor window are just
                # low-energy portions of playback. Only end SPEAK once the
                # motor window has actually ended.
                if (not res.is_in_motor_window) and self._motor_end_time > 0:
                    in_initial_motor_holdoff = (
                        self._request_output_time > 0
                        and (current_time - self._request_output_time) < (self.motor_holdoff_ms / 1000.0)
                    )
                    if not in_initial_motor_holdoff:
                        self.active_utterance = None
                        self.last_self_time = 0
                        self.state = AgentState.IDLE

        # 3. Update external speech timing when not handled by SPEAK interrupt.
        if self.state != AgentState.SPEAK:
            if just_interrupted:
                pass
            elif is_agentive_external_speech:
                # Motor holdoff: after request_output(), suppress WORLD→LISTEN
                # transitions for a brief period. This prevents ambient noise
                # before the agent starts playing from triggering LISTEN.
                in_holdoff = (
                    self._request_output_time > 0
                    and (current_time - self._request_output_time) < (self.motor_holdoff_ms / 1000.0)
                )
                if in_holdoff:
                    # During holdoff, don't start speech timing from WORLD frames.
                    # But do track timing if already listening (sustained speech).
                    if self.speech_start_time > 0:
                        self.last_speech_time = current_time
                else:
                    if self.speech_start_time == 0:
                        self.speech_start_time = current_time
                    self.last_speech_time = current_time
            else:
                # Only reset if we were previously hearing speech and the cooldown has passed.
                if self.last_speech_time > 0 and (current_time - self.last_speech_time) > (self.cooldown_ms / 1000.0):
                    self.speech_start_time = 0
                    self.last_speech_time = 0

            # 3b. Hysteresis: minimum LISTEN duration after overlap interrupt.
            # Once we entered LISTEN due to BOTH-overlap detection, stay in LISTEN
            # for at least both_external_listen_min_ms. This prevents flapping
            # when the energy ratio oscillates around the threshold.
            in_overlap_hysteresis = False
            if (
                self.state == AgentState.LISTEN
                and self._overlap_listen_start_time > 0
            ):
                elapsed = current_time - self._overlap_listen_start_time
                if elapsed < (self.both_external_listen_min_ms / 1000.0):
                    in_overlap_hysteresis = True
                else:
                    # Minimum LISTEN duration satisfied
                    self._overlap_listen_start_time = 0.0

            # 3c. Post-interrupt resume: after any SPEAK→LISTEN interrupt,
            # if the controller is still in LISTEN after the agent finishes
            # and the human speech has ended, return to IDLE.
            #
            # This handles two cases:
            # A) Overlap interrupt (energy ratio-based): resume when ratio
            #    drops below threshold or after timeout
            # B) Agent-finished interrupt (WORLD speech): resume after timeout
            #    if VAD keeps detecting noise but agent is not playing
            post_interrupt_resume = False
            if (
                self.state == AgentState.LISTEN
                and not just_interrupted
                and not in_overlap_hysteresis
            ):
                # Check if we have deferred output (were interrupted)
                if self.deferred_utterance is not None:
                    # Case A: Overlap interrupt — use energy ratio
                    if (
                        self._was_overlap_interrupted
                        and res.self_match_score_per_frame is not None
                        and res.self_match_score_per_frame <= self.both_external_energy_ratio_threshold
                    ):
                        post_interrupt_resume = True
                    # Case B: Any interrupt — time-based fallback
                    # If we've been in LISTEN for > 2x min listen duration
                    # and the agent is not playing, assume human speech ended
                    elif res.verdict != SourceVerdict.BOTH:
                        # Agent is not playing (verdict is WORLD or SILENCE)
                        if self._overlap_listen_start_time > 0:
                            elapsed = current_time - self._overlap_listen_start_time
                        else:
                            elapsed = current_time - self.speech_start_time
                        if elapsed >= (self.both_external_listen_min_ms * 2 / 1000.0):
                            post_interrupt_resume = True
            
            if post_interrupt_resume:
                self.state = AgentState.IDLE
                self.speech_start_time = 0
                self.last_speech_time = 0

            if not just_interrupted and not in_overlap_hysteresis:
                is_hearing_agent = (self.speech_start_time > 0) and \
                                   (current_time - self.speech_start_time) > (self.listen_threshold_ms / 1000.0)

                if is_hearing_agent:
                    self.state = AgentState.LISTEN
                elif self.state == AgentState.LISTEN:
                    self.state = AgentState.IDLE

                # LISTEN→SPEAK recovery: if we're in LISTEN (from IDLE, not from
                # a SPEAK interrupt) and the attributor detects BOTH with high
                # match_score while the agent is actively emitting, the "world
                # speech" is the agent's own playback starting. Transition back
                # to SPEAK so the agent can continue its utterance.
                if (
                    self.state == AgentState.LISTEN
                    and not self._was_overlap_interrupted
                    and res.verdict == SourceVerdict.BOTH
                    and res.is_speech
                    and res.self_match_score >= 0.03  # above noise floor
                    and res.is_in_motor_window  # agent is actively emitting
                ):
                    self.state = AgentState.SPEAK
                    self.active_utterance = self.deferred_utterance
                    self.deferred_utterance = None
            elif in_overlap_hysteresis:
                # During hysteresis, force stay in LISTEN
                self.state = AgentState.LISTEN
                self.active_utterance = None

        # 4. Decision Making
        can_speak = (self.state != AgentState.LISTEN)
        
        return ControllerDecision(
            state=self.state,
            can_speak=can_speak,
            should_interrupt=should_interrupt,
            deferred_text=self.deferred_utterance if not can_speak else None,
            collision_repair_text=collision_repair_text,
        )

    def request_output(self, text: str, current_time: float = None) -> bool:
        """PFC requests to speak. Returns True if accepted, False if deferred."""
        if self.state == AgentState.LISTEN:
            self.deferred_utterance = text
            return False

        self.state = AgentState.SPEAK
        self.active_utterance = text
        self.deferred_utterance = None
        # Use provided time or fall back to wall-clock
        self._request_output_time = current_time if current_time is not None else time.time()
        return True

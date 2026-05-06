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
        both_external_min_frames: int = 3,
    ) -> None:
        self.state = AgentState.IDLE
        self.listen_threshold_ms = listen_threshold_ms
        self.cooldown_ms = cooldown_ms
        self.treat_both_speech_as_agentive = treat_both_speech_as_agentive
        self.motor_holdoff_ms = motor_holdoff_ms
        self.both_external_energy_ratio_threshold = both_external_energy_ratio_threshold
        self.both_external_min_frames = both_external_min_frames

        self.last_speech_time: float = 0.0
        self.speech_start_time: float = 0.0
        self.last_self_time: float = 0.0
        self.active_utterance: Optional[str] = None
        self.deferred_utterance: Optional[str] = None
        self._request_output_time: float = 0.0

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
        # (observed / expected self RMS) is significantly above 1.0, this
        # indicates human speech is overlapping with the self signal.
        #
        # The per-frame energy ratio is computed by comparing the observed
        # frame's RMS against the expected self frame's RMS from the agent
        # PCM template. When human speech overlaps, the observed energy is
        # higher than expected.
        #
        # We require:
        # 1. Per-frame energy ratio is available (self_pcm was provided)
        # 2. Ratio exceeds threshold (e.g., 1.5 = 3.5 dB excess)
        # 3. Sustained excess (N consecutive frames) to avoid noise
        both_is_external = False
        if (res.is_speech and res.verdict == SourceVerdict.BOTH 
            and res.is_in_motor_window
            and res.self_match_score_per_frame is not None):
            # Energy ratio > threshold indicates external speech
            if res.self_match_score_per_frame > self.both_external_energy_ratio_threshold:
                self._both_external_count = getattr(self, '_both_external_count', 0) + 1
            else:
                self._both_external_count = 0
            if self._both_external_count >= self.both_external_min_frames:
                both_is_external = True
        else:
            self._both_external_count = 0
        
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

        # 2. SPEAK is a real lifecycle: self frames sustain it; external speech
        # interrupts it immediately and queues the active utterance for later.
        if self.state == AgentState.SPEAK:
            if is_agentive_external_speech:
                should_interrupt = True
                just_interrupted = True
                if self.active_utterance is not None:
                    self.deferred_utterance = self.active_utterance
                self.active_utterance = None
                self.state = AgentState.LISTEN
                if self.speech_start_time == 0:
                    self.speech_start_time = current_time
                self.last_speech_time = current_time
            elif is_self_activity:
                self.last_self_time = current_time
            else:
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

            if not just_interrupted:
                is_hearing_agent = (self.speech_start_time > 0) and \
                                   (current_time - self.speech_start_time) > (self.listen_threshold_ms / 1000.0)

                if is_hearing_agent:
                    self.state = AgentState.LISTEN
                elif self.state == AgentState.LISTEN:
                    # We were listening, now it's quiet. Return to IDLE.
                    self.state = AgentState.IDLE

                # LISTEN→SPEAK recovery: if we're in LISTEN (from IDLE, not from
                # a SPEAK interrupt) and the attributor detects BOTH with high
                # match_score while the agent is actively emitting, the "world
                # speech" is the agent's own playback starting. Transition back
                # to SPEAK so the agent can continue its utterance.
                if (
                    self.state == AgentState.LISTEN
                    and not just_interrupted  # don't override genuine SPEAK→LISTEN
                    and res.verdict == SourceVerdict.BOTH
                    and res.is_speech
                    and res.self_match_score >= 0.03  # above noise floor
                    and res.is_in_motor_window  # agent is actively emitting
                ):
                    self.state = AgentState.SPEAK
                    self.active_utterance = self.deferred_utterance
                    self.deferred_utterance = None

        # 4. Decision Making
        can_speak = (self.state != AgentState.LISTEN)
        
        return ControllerDecision(
            state=self.state,
            can_speak=can_speak,
            should_interrupt=should_interrupt,
            deferred_text=self.deferred_utterance if not can_speak else None
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

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
    ) -> None:
        self.state = AgentState.IDLE
        self.listen_threshold_ms = listen_threshold_ms
        self.cooldown_ms = cooldown_ms
        self.treat_both_speech_as_agentive = treat_both_speech_as_agentive
        
        self.last_speech_time: float = 0.0
        self.speech_start_time: float = 0.0
        self.last_self_time: float = 0.0
        self.active_utterance: Optional[str] = None
        self.deferred_utterance: Optional[str] = None

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
            
        # 1. Classify the frame for M3 policy.
        is_world_speech = res.is_speech and res.verdict == SourceVerdict.WORLD
        is_overlap_speech = (
            self.treat_both_speech_as_agentive
            and res.is_speech
            and res.verdict == SourceVerdict.BOTH
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

        # 4. Decision Making
        can_speak = (self.state != AgentState.LISTEN)
        
        return ControllerDecision(
            state=self.state,
            can_speak=can_speak,
            should_interrupt=should_interrupt,
            deferred_text=self.deferred_utterance if not can_speak else None
        )

    def request_output(self, text: str) -> bool:
        """PFC requests to speak. Returns True if accepted, False if deferred."""
        if self.state == AgentState.LISTEN:
            self.deferred_utterance = text
            return False
        
        self.state = AgentState.SPEAK
        self.active_utterance = text
        self.deferred_utterance = None
        return True

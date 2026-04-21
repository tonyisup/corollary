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
    ) -> None:
        self.state = AgentState.IDLE
        self.listen_threshold_ms = listen_threshold_ms
        self.cooldown_ms = cooldown_ms
        
        self.last_speech_time: float = 0.0
        self.speech_start_time: float = 0.0
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
            
        # 1. Update Speech Timing
        is_world_speech = res.is_speech and res.verdict == SourceVerdict.WORLD
        
        if is_world_speech:
            if self.speech_start_time == 0:
                self.speech_start_time = current_time
            self.last_speech_time = current_time
        else:
            # Only reset if we were previously hearing speech and the cooldown has passed
            if self.last_speech_time > 0 and (current_time - self.last_speech_time) > (self.cooldown_ms / 1000.0):
                self.speech_start_time = 0
                self.last_speech_time = 0
            
        # 2. State Transitions
        is_hearing_agent = (self.speech_start_time > 0) and \
                           (current_time - self.speech_start_time) > (self.listen_threshold_ms / 1000.0)
        
        if is_hearing_agent:
            self.state = AgentState.LISTEN
        elif self.state == AgentState.LISTEN:
            # We were listening, now it's quiet. Return to IDLE.
            self.state = AgentState.IDLE
            
        # 3. Decision Making
        can_speak = (self.state != AgentState.LISTEN)
        
        # Simple policy: don't interrupt if we are in LISTEN state
        should_interrupt = False
        
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
        self.deferred_utterance = None
        return True

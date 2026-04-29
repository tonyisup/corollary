"""Thalamus — Source Attribution.

M1C: Source attribution logic combining playback state, forward-model 
predictions (efference copy), and acoustic similarity.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

class SourceVerdict(Enum):
    SILENCE = 0
    SELF = 1
    WORLD = 2
    BOTH = 3

@dataclass
class AttributionResult:
    verdict: SourceVerdict
    confidence: float
    self_match_score: float
    is_in_motor_window: bool
    causal_score: float
    energy_db: float
    is_speech: bool = False
    onset_error_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ThalamicAttributor:
    """The source attribution unit of the Thalamus.
    
    Decides whether an acoustic event was self-caused or world-caused.
    Now with speech-awareness (M2).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        energy_threshold_db: float = -55.0,
        causal_tolerance_ms: float = 25.0,
        sr: int = 16000,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.energy_threshold_db = energy_threshold_db
        self.causal_tolerance_ms = causal_tolerance_ms
        self.sr = sr

    def attribute(
        self,
        match_score: float,
        observed_onset_samples: int,
        expected_onset_samples: Optional[int],
        rms_db: float,
        is_playing: bool,
        is_speech: bool = False,
    ) -> AttributionResult:
        """Assign source attribution to an acoustic event.

        Parameters
        ----------
        match_score : float
            Acoustic similarity score (e.g., normalized cross-correlation).
        observed_onset_samples : int
            Measured onset time of the event.
        expected_onset_samples : int, optional
            Expected onset time based on motor command + room delay.
        rms_db : float
            Energy of the observed signal in dB.
        is_playing : bool
            True if Corollary was actively emitting sound during the window.
        is_speech : bool
            True if VAD detects speech in this window (M2).

        Returns
        -------
        AttributionResult
            The source verdict and supporting metrics.
        """
        # 1. Check for silence
        if rms_db < self.energy_threshold_db:
            return AttributionResult(
                verdict=SourceVerdict.SILENCE,
                confidence=1.0,
                self_match_score=match_score,
                is_in_motor_window=is_playing,
                causal_score=0.0,
                energy_db=rms_db,
                is_speech=is_speech
            )

        # 2. Compute causal score (timing alignment)
        causal_score = 0.0
        onset_error_ms = None
        if expected_onset_samples is not None:
            onset_error_ms = abs(observed_onset_samples - expected_onset_samples) * 1000.0 / self.sr
            causal_score = max(0.0, 1.0 - (onset_error_ms / self.causal_tolerance_ms))

        # 3. Decision Logic
        
        # Candidate for SELF
        is_self_like = match_score >= self.similarity_threshold
        is_timely = causal_score > 0.0
        
        if is_playing and is_self_like and is_timely and is_speech:
            # M2 conservatism: speech during playback can contain world speech
            # even when the self template also matches, so preserve it for a
            # soft downstream gate instead of making a hard SELF decision.
            confidence = (match_score + causal_score) / 2.0
            return AttributionResult(
                verdict=SourceVerdict.BOTH,
                confidence=confidence,
                self_match_score=match_score,
                is_in_motor_window=is_playing,
                causal_score=causal_score,
                energy_db=rms_db,
                is_speech=is_speech,
                onset_error_ms=onset_error_ms,
            )

        if is_playing and is_self_like and is_timely:
            # High similarity + motor command + correct timing = SELF
            confidence = (match_score + causal_score) / 2.0
            return AttributionResult(
                verdict=SourceVerdict.SELF,
                confidence=confidence,
                self_match_score=match_score,
                is_in_motor_window=is_playing,
                causal_score=causal_score,
                energy_db=rms_db,
                is_speech=is_speech,
                onset_error_ms=onset_error_ms
            )
        
        if is_playing and (not is_self_like or not is_timely):
            # Energy exists during playback, but it doesn't match our prediction
            # or it arrived at the wrong time.
            verdict = SourceVerdict.BOTH
            if is_speech:
                # If it's speech and doesn't match ME, it's likely WORLD speech.
                confidence = 0.8
            else:
                confidence = 0.7
                
            return AttributionResult(
                verdict=verdict,
                confidence=confidence,
                self_match_score=match_score,
                is_in_motor_window=is_playing,
                causal_score=causal_score,
                energy_db=rms_db,
                is_speech=is_speech,
                onset_error_ms=onset_error_ms
            )

        if not is_playing:
            # Energy exists when we ARE NOT playing.
            verdict = SourceVerdict.WORLD
            confidence = 0.9 if is_speech else 0.8
            return AttributionResult(
                verdict=verdict,
                confidence=confidence,
                self_match_score=match_score,
                is_in_motor_window=is_playing,
                causal_score=causal_score,
                energy_db=rms_db,
                is_speech=is_speech,
                onset_error_ms=onset_error_ms
            )

        # Fallback
        return AttributionResult(
            verdict=SourceVerdict.WORLD,
            confidence=0.5,
            self_match_score=match_score,
            is_in_motor_window=is_playing,
            causal_score=causal_score,
            energy_db=rms_db,
            is_speech=is_speech
        )

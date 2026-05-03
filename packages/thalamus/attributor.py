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
        similarity_threshold: float = 0.04,
        energy_threshold_db: float = -55.0,
        causal_tolerance_ms: float = 25.0,
        sr: int = 16000,
        m4_detector: Optional[Any] = None,
        m4_self_threshold: float = 0.4,
        m4_window_ms: float = 1000.0,
        m4_min_buffer_ms: float = 500.0,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.energy_threshold_db = energy_threshold_db
        self.causal_tolerance_ms = causal_tolerance_ms
        self.sr = sr
        self.m4_detector = m4_detector
        self.m4_self_threshold = m4_self_threshold
        self.m4_window_samples = int(m4_window_ms * sr / 1000.0)
        self.m4_min_buffer_samples = int(m4_min_buffer_ms * sr / 1000.0)
        self._m4_buffer: Optional[np.ndarray] = None

    def attribute(
        self,
        match_score: float,
        observed_onset_samples: int,
        expected_onset_samples: Optional[int],
        rms_db: float,
        is_playing: bool,
        is_speech: bool = False,
        frame_audio: Optional[np.ndarray] = None,
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
            # M4 override: use speaker embedding to distinguish self from world
            # speech. Requires accumulating audio into a buffer since ECAPA-TDNN
            # needs ~500ms+ for reliable classification.
            m4_metadata = {
                'm4_attempted': False,
                'm4_outcome': 'not_attempted',
                'm4_confidence': None,
                'm4_buffer_length': 0,
            }
            if self.m4_detector is not None and frame_audio is not None:
                # Accumulate frame into M4 buffer
                if self._m4_buffer is None:
                    self._m4_buffer = frame_audio.copy()
                else:
                    self._m4_buffer = np.concatenate([self._m4_buffer, frame_audio])
                # Keep only the last m4_window_samples
                if len(self._m4_buffer) > self.m4_window_samples:
                    self._m4_buffer = self._m4_buffer[-self.m4_window_samples:]
                m4_metadata['m4_buffer_length'] = len(self._m4_buffer)
                # Only run M4 when we have enough audio
                m4_metadata['m4_attempted'] = len(self._m4_buffer) >= self.m4_min_buffer_samples
                if m4_metadata['m4_attempted']:
                    try:
                        is_self_voice, m4_confidence = self.m4_detector.is_self(self._m4_buffer)
                        m4_metadata['m4_confidence'] = m4_confidence
                        if is_self_voice and m4_confidence > self.m4_self_threshold:
                            m4_metadata['m4_outcome'] = 'accepted'
                            confidence = (match_score + causal_score + m4_confidence) / 3.0
                            return AttributionResult(
                                verdict=SourceVerdict.SELF,
                                confidence=confidence,
                                self_match_score=match_score,
                                is_in_motor_window=is_playing,
                                causal_score=causal_score,
                                energy_db=rms_db,
                                is_speech=is_speech,
                                onset_error_ms=onset_error_ms,
                                metadata=m4_metadata,
                            )
                        else:
                            m4_metadata['m4_outcome'] = 'below_threshold'
                    except Exception:
                        m4_metadata['m4_outcome'] = 'exception'
                else:
                    m4_metadata['m4_outcome'] = 'buffer_too_short'
            # M2 conservatism (fallback): speech during playback can contain
            # world speech even when the self template matches.
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
                metadata=m4_metadata,
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

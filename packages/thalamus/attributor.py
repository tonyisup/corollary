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
    self_match_score_per_frame: Optional[float] = None  # Per-frame local self-match

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
        m4_self_threshold: float = 0.1,
        m4_window_ms: float = 1000.0,
        m4_min_buffer_ms: float = 500.0,
        per_frame_self_min_rms_db: float = -50.0,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.energy_threshold_db = energy_threshold_db
        self.causal_tolerance_ms = causal_tolerance_ms
        self.sr = sr
        self.m4_detector = m4_detector
        self.m4_self_threshold = m4_self_threshold
        self.m4_window_samples = int(m4_window_ms * sr / 1000.0)
        self.m4_min_buffer_samples = int(m4_min_buffer_ms * sr / 1000.0)
        self.per_frame_self_min_rms_db = per_frame_self_min_rms_db
        self._m4_buffer: Optional[np.ndarray] = None

    def _compute_per_frame_self_match(
        self,
        frame_audio: np.ndarray,
        self_pcm: np.ndarray,
        agent_start_sample: int,
        observed_onset_samples: int,
        expected_onset_samples: int,
    ) -> Optional[float]:
        """Compute per-frame self-match score.
        
        Uses normalized cross-correlation between the observed frame and the
        corresponding slice of the agent PCM template. Returns None if
        self_pcm is not available or frame is out of bounds.
        
        Note: This uses a short window (frame_audio length, typically 30ms)
        which can be noisy. For more stable scoring, a longer window should
        be used (see _compute_per_frame_energy_ratio).
        """
        frame_len = len(frame_audio)
        rel_pos = observed_onset_samples - expected_onset_samples
        self_start = agent_start_sample + rel_pos
        self_end = self_start + frame_len
        
        if self_start < 0 or self_end > len(self_pcm):
            return None
        
        expected_self_frame = self_pcm[self_start:self_end]
        
        obs_norm = float(np.linalg.norm(frame_audio))
        self_norm = float(np.linalg.norm(expected_self_frame))
        
        if obs_norm < 1e-9 or self_norm < 1e-9:
            return 0.0
        
        return float(np.dot(frame_audio, expected_self_frame) / (obs_norm * self_norm))
    
    def _compute_per_frame_energy_ratio(
        self,
        frame_audio: np.ndarray,
        self_pcm: np.ndarray,
        agent_start_sample: int,
        observed_onset_samples: int,
        expected_onset_samples: int,
    ) -> Optional[float]:
        """Compute per-frame energy ratio (observed / expected self).
        
        When human speech overlaps with self-playback, the observed energy
        is higher than the expected self energy. This ratio is more stable
        than per-frame correlation because it's less sensitive to timing
        misalignment and phase differences.
        
        Returns:
            Ratio of observed RMS to expected self RMS.
            None if self_pcm is not available or frame is out of bounds.
            0.0 if expected self energy is near zero.
        """
        frame_len = len(frame_audio)
        rel_pos = observed_onset_samples - expected_onset_samples
        self_start = agent_start_sample + rel_pos
        self_end = self_start + frame_len
        
        if self_start < 0 or self_end > len(self_pcm):
            return None
        
        expected_self_frame = self_pcm[self_start:self_end]
        
        obs_rms = float(np.sqrt(np.mean(frame_audio ** 2)))
        self_rms = float(np.sqrt(np.mean(expected_self_frame ** 2)))
        
        if self_rms < 1e-10:
            return 0.0

        self_rms_db = 20 * np.log10(max(self_rms, 1e-10))
        if self_rms_db < self.per_frame_self_min_rms_db:
            return 0.0
        
        return obs_rms / self_rms

    def attribute(
        self,
        match_score: float,
        observed_onset_samples: int,
        expected_onset_samples: Optional[int],
        rms_db: float,
        is_playing: bool,
        is_speech: bool = False,
        frame_audio: Optional[np.ndarray] = None,
        self_pcm: Optional[np.ndarray] = None,
        agent_start_sample: Optional[int] = None,
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
        frame_audio : np.ndarray, optional
            Raw audio for this frame (used for M4 buffer).
        self_pcm : np.ndarray, optional
            Full agent PCM template for per-frame self-match scoring.
        agent_start_sample : int, optional
            Sample offset in self_pcm where agent playback begins.
        
        Returns
        -------
        AttributionResult
            The source verdict and supporting metrics.
        """
        # Initialize M4 metadata
        m4_metadata = {
            'm4_attempted': False,
            'm4_outcome': 'not_attempted',
            'm4_confidence': None,
            'm4_buffer_length': 0,
        }
        
        # Compute per-frame self-match score when self_pcm is available.
        # Uses energy ratio (observed / expected self RMS) which is more stable
        # than per-frame correlation for short windows.
        self_match_score_per_frame = None
        if self_pcm is not None and agent_start_sample is not None and frame_audio is not None:
            if expected_onset_samples is not None and is_playing:
                self_match_score_per_frame = self._compute_per_frame_energy_ratio(
                    frame_audio, self_pcm, agent_start_sample,
                    observed_onset_samples, expected_onset_samples
                )
        
        # 1. Check for silence
        if rms_db < self.energy_threshold_db:
            return AttributionResult(
                verdict= SourceVerdict.SILENCE,
                confidence=1.0,
                self_match_score=match_score,
                is_in_motor_window=is_playing,
                causal_score=0.0,
                energy_db=rms_db,
                is_speech=is_speech,
                metadata=m4_metadata,
                self_match_score_per_frame=0.0,
            )

        # 2. Compute causal score (timing alignment)
        causal_score = 0.0
        onset_error_ms = None
        if expected_onset_samples is not None:
            onset_error_ms = abs(observed_onset_samples - expected_onset_samples) * 1000.0 / self.sr
            causal_score = max(0.0, 1.0 - (onset_error_ms / self.causal_tolerance_ms))
        
        # Candidate for SELF
        is_self_like = match_score >= self.similarity_threshold
        is_timely = causal_score > 0.0
        
        # M4 attempt
        if is_playing and is_speech and self.m4_detector is not None:
            if frame_audio is not None:
                if self._m4_buffer is None:
                    self._m4_buffer = frame_audio.copy()
                else:
                    self._m4_buffer = np.concatenate([self._m4_buffer, frame_audio])
                if len(self._m4_buffer) > self.m4_window_samples:
                    self._m4_buffer = self._m4_buffer[-self.m4_window_samples:]
                m4_metadata['m4_buffer_length'] = len(self._m4_buffer)
                m4_metadata['m4_attempted'] = len(self._m4_buffer) >= self.m4_min_buffer_samples
                if m4_metadata['m4_attempted']:
                    try:
                        is_self_voice, m4_confidence = self.m4_detector.is_self(self._m4_buffer)
                        m4_metadata['m4_confidence'] = m4_confidence
                        if is_self_voice and m4_confidence > self.m4_self_threshold:
                            m4_metadata['m4_outcome'] = 'accepted'
                        else:
                            m4_metadata['m4_outcome'] = 'below_threshold'
                    except Exception:
                        m4_metadata['m4_outcome'] = 'exception'
                else:
                    m4_metadata['m4_outcome'] = 'buffer_too_short'
        
        # Determine verdict
        verdict = SourceVerdict.WORLD
        confidence = 0.5
        
        # M4-assisted SELF attribution
        if is_playing and is_speech and m4_metadata.get('m4_outcome') == 'accepted':
            m4_conf = m4_metadata.get('m4_confidence', 0.0)
            if is_self_like and m4_conf >= self.m4_self_threshold:
                verdict = SourceVerdict.SELF
                confidence = (match_score + causal_score + m4_conf) / 3.0
                return AttributionResult(
                    verdict=verdict, confidence=confidence,
                    self_match_score=match_score, is_in_motor_window=is_playing,
                    causal_score=causal_score, energy_db=rms_db, is_speech=is_speech,
                    onset_error_ms=onset_error_ms, metadata=m4_metadata,
                    self_match_score_per_frame=self_match_score_per_frame,
                )
        
        # M2 logic
        if is_playing and is_self_like and is_timely and is_speech:
            verdict = SourceVerdict.BOTH
            confidence = (match_score + causal_score) / 2.0
        elif is_playing and is_self_like and is_timely:
            verdict = SourceVerdict.SELF
            confidence = (match_score + causal_score) / 2.0
        elif is_playing:
            verdict = SourceVerdict.BOTH
            confidence = 0.8 if is_speech else 0.7
        else:
            verdict = SourceVerdict.WORLD
            confidence = 0.9 if is_speech else 0.8
        
        return AttributionResult(
            verdict=verdict, confidence=confidence,
            self_match_score=match_score, is_in_motor_window=is_playing,
            causal_score=causal_score, energy_db=rms_db, is_speech=is_speech,
            onset_error_ms=onset_error_ms, metadata=m4_metadata,
            self_match_score_per_frame=self_match_score_per_frame,
        )

"""Thalamus — Source Attribution.

M1C: Source attribution logic combining playback state, forward-model
predictions (efference copy), and acoustic similarity.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter

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
    room_local_score: Optional[float] = None  # Room-colored self-match score

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
        m4_template_match_threshold: float = 0.20,
        per_frame_self_min_rms_db: float = -50.0,
        room_filter: Optional[np.ndarray] = None,
        room_local_threshold: Optional[float] = None,
        room_local_search_ms: float = 250.0,
        speaker_threshold_adapter: Optional[SpeakerThresholdAdapter] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.energy_threshold_db = energy_threshold_db
        self.causal_tolerance_ms = causal_tolerance_ms
        self.sr = sr
        self.m4_detector = m4_detector
        self.m4_self_threshold = m4_self_threshold
        self.m4_template_match_threshold = m4_template_match_threshold
        self.m4_window_samples = int(m4_window_ms * sr / 1000.0)
        self.m4_min_buffer_samples = int(m4_min_buffer_ms * sr / 1000.0)
        self.per_frame_self_min_rms_db = per_frame_self_min_rms_db
        self._m4_buffer: Optional[np.ndarray] = None
        self.room_filter = room_filter
        self.room_local_threshold = room_local_threshold
        self.room_local_search_ms = room_local_search_ms
        # Per-speaker adaptive threshold adapter
        self.speaker_threshold_adapter = speaker_threshold_adapter

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

    def _compute_room_local_score(
        self,
        frame_audio: np.ndarray,
        self_pcm: np.ndarray,
        agent_start_sample: int,
        observed_onset_samples: int,
        expected_onset_samples: int,
    ) -> Optional[float]:
        """Compute room-local self-match score using a pre-fitted room filter.
        
        The room filter is applied to the dry self template to produce a
        room-colored reference, which is then correlated against the observed
        frame.  This is the production-causal path: the filter must be fitted
        *before* the current capture (e.g. from a prior calibration run).
        
        Returns:
            Normalized correlation between observed frame and room-colored
            self reference.  None if room_filter or self_pcm is not available.
        """
        if self.room_filter is None or self_pcm is None:
            return None

        frame_len = len(frame_audio)
        rel_pos = observed_onset_samples - expected_onset_samples
        self_start = agent_start_sample + rel_pos
        self_end = self_start + frame_len

        if self_start < 0 or self_end > len(self_pcm):
            return None

        expected_self_frame = self_pcm[self_start:self_end]

        # Apply room filter to get room-colored reference
        room_colored = np.convolve(expected_self_frame, self.room_filter, mode="same")
        room_colored = room_colored.astype(np.float32)

        # Ensure room_colored matches frame_audio length
        if len(room_colored) != len(frame_audio):
            if len(room_colored) > len(frame_audio):
                # Truncate center portion
                start = (len(room_colored) - len(frame_audio)) // 2
                room_colored = room_colored[start:start + len(frame_audio)]
            else:
                # Pad with zeros
                room_colored = np.pad(room_colored, (0, len(frame_audio) - len(room_colored)))

        obs_norm = float(np.linalg.norm(frame_audio))
        room_norm = float(np.linalg.norm(room_colored))

        if obs_norm < 1e-9 or room_norm < 1e-9:
            return 0.0

        return float(np.dot(frame_audio, room_colored) / (obs_norm * room_norm))

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
        speaker_id: Optional[str] = None,
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
        speaker_id : str, optional
            Identified speaker ID from the STG speaker identification module.
            When provided and a SpeakerThresholdAdapter is configured, the
            M4 self-voice threshold is adapted per-speaker.
        
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
            'm4_template_match_passed': False,
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

        # Compute room-local self-match score when a pre-fitted room filter is
        # available.  This is the production-causal path: the filter is fitted
        # before the current capture (e.g. from a prior calibration run).
        room_local_score = None
        if (
            self.room_filter is not None
            and self_pcm is not None
            and expected_onset_samples is not None
            and is_playing
        ):
            room_local_score = self._compute_room_local_score(
                frame_audio, self_pcm, agent_start_sample,
                observed_onset_samples, expected_onset_samples,
            )

        # 1. Check for silence
        if rms_db < self.energy_threshold_db:
            return AttributionResult(
                verdict=SourceVerdict.SILENCE,
                confidence=1.0,
                self_match_score=match_score,
                is_in_motor_window=is_playing,
                causal_score=0.0,
                energy_db=rms_db,
                is_speech=is_speech,
                metadata=m4_metadata,
                self_match_score_per_frame=0.0,
                room_local_score=0.0,
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

        # Only engage per-speaker adaptation when we have a real identified
        # speaker from upstream STG. The live fallback `speaker_id=None` would
        # otherwise collapse all room-colored playback into a single "unknown"
        # profile and ratchet its threshold upward, self-poisoning the live path.
        adaptive_speaker_id = (
            speaker_id
            if speaker_id not in (None, "", "unknown")
            else None
        )

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
                        # Apply per-speaker adaptive threshold only for real
                        # identified speakers. Non-self profile learning happens
                        # in the runtime from cleaner WORLD-only evidence after
                        # verdict resolution; do not learn here from overlap-prone
                        # attribution frames.
                        if (
                            self.speaker_threshold_adapter is not None
                            and adaptive_speaker_id is not None
                        ):
                            threshold = self.speaker_threshold_adapter.get_threshold(
                                adaptive_speaker_id
                            )
                        else:
                            threshold = self.m4_self_threshold
                        if is_self_voice and m4_confidence > threshold:
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
        
        # M4-assisted SELF attribution.
        # Two paths to SELF:
        #   (a) Strict: ECAPA accepts AND template match >= threshold AND
        #       ECAPA confidence >= self_threshold. This is the original path
        #       for dry/synthetic audio where template correlation is reliable.
        #   (b) Embedding-primary: ECAPA accepts AND ECAPA confidence >= 0.15.
        #       For captured audio, per-frame template correlation is often weak
        #       or negative because the room filter doesn't generalize across
        #       different TTS utterances. ECAPA's embedding-level speaker
        #       discrimination is more robust, so we lower the template
        #       requirement and rely primarily on ECAPA confidence.
        #       We require the match_score to be above a small negative floor
        #       (>= -0.1) as a sanity check — strong anti-correlation should
        #       still block SELF, but weak negative correlation is common with
        #       captured audio due to phase/timing misalignment.
        is_m4_template_match = match_score >= self.m4_template_match_threshold
        is_room_local_match = (
            room_local_score is not None
            and self.room_local_threshold is not None
            and room_local_score >= self.room_local_threshold
        )
        m4_conf = m4_metadata.get("m4_confidence", 0.0) or 0.0
        m4_strict_gate = is_m4_template_match or is_room_local_match
        # For captured audio (room_filter present), use a combined score
        # of ECAPA confidence and template correlation. ECAPA's confidence
        # on room-colored captured audio is calibrated lower than on clean
        # synthetic audio, and the template correlation is weak. A combined
        # score is more robust than either signal alone.
        # Combined = 0.8 * ecapa_conf + 0.2 * max(match_score, 0.0)
        # This weights ECAPA's embedding-level discrimination more heavily
        # than the per-frame template correlation.
        if self.room_filter is not None:
            _combined = 0.8 * m4_conf + 0.2 * max(match_score, 0.0)
            _emb_threshold = self.m4_self_threshold
            if (
                self.speaker_threshold_adapter is not None
                and adaptive_speaker_id is not None
            ):
                _emb_threshold = self.speaker_threshold_adapter.get_threshold(adaptive_speaker_id)
            m4_embedding_gate = _combined >= _emb_threshold
        else:
            # Dry/synthetic benchmark cases have reliable template alignment.
            # In that mode, weak positive correlation is not enough evidence for
            # SELF because same-domain playback-window negatives can still be
            # accepted by ECAPA. Require the strict template gate instead; the
            # embedding-primary path is only for room-colored captured audio.
            m4_embedding_gate = False
        m4_evidence = m4_strict_gate or m4_embedding_gate
        m4_metadata['m4_template_match_passed'] = is_m4_template_match
        m4_metadata['room_local_passed'] = is_room_local_match

        # Use per-speaker threshold for the evidence gate too, but only for
        # real identified speakers. Unknown/live fallback stays on base.
        _gate_threshold = self.m4_self_threshold
        if (
            self.speaker_threshold_adapter is not None
            and adaptive_speaker_id is not None
        ):
            _gate_threshold = self.speaker_threshold_adapter.get_threshold(adaptive_speaker_id)

        if is_playing and is_speech and m4_metadata.get('m4_outcome') == 'accepted':
            if m4_evidence and m4_conf >= _gate_threshold:
                verdict = SourceVerdict.SELF
                confidence = (match_score + causal_score + m4_conf) / 3.0
                return AttributionResult(
                    verdict=verdict, confidence=confidence,
                    self_match_score=match_score, is_in_motor_window=is_playing,
                    causal_score=causal_score, energy_db=rms_db, is_speech=is_speech,
                    onset_error_ms=onset_error_ms, metadata=m4_metadata,
                    self_match_score_per_frame=self_match_score_per_frame,
                    room_local_score=room_local_score,
                )
        
        # Room-local score can boost SELF confidence when the room-colored
        # reference matches better than the dry template.  If the room-local
        # score exceeds the threshold, we treat the frame as more likely SELF.
        room_local_self = (
            room_local_score is not None
            and self.room_local_threshold is not None
            and room_local_score >= self.room_local_threshold
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

        # Room-local override: if the room-colored reference matches well,
        # boost confidence that this is SELF rather than WORLD/BOTH.
        if room_local_self and verdict == SourceVerdict.BOTH:
            # Room-local evidence suggests this is self-playback, not overlap.
            # Only override if the dry match is not already strong.
            if match_score < 0.5:
                verdict = SourceVerdict.SELF
                confidence = (match_score + causal_score + room_local_score) / 3.0
        
        return AttributionResult(
            verdict=verdict, confidence=confidence,
            self_match_score=match_score, is_in_motor_window=is_playing,
            causal_score=causal_score, energy_db=rms_db, is_speech=is_speech,
            onset_error_ms=onset_error_ms, metadata=m4_metadata,
            self_match_score_per_frame=self_match_score_per_frame,
            room_local_score=room_local_score,
        )

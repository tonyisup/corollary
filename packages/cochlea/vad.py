"""Cochlea — Voice Activity Detection (VAD).

Wraps webrtcvad for low-latency frame-level speech detection.
"""

from __future__ import annotations

import collections
import numpy as np

try:
    import webrtcvad
except ImportError:
    webrtcvad = None  # type: ignore[misc, assignment]

# ── Global VAD constants ──────────────────────────────────────────────────
VAD_FRAME_MS = 30  # webrtcvad supports 10, 20, 30 ms
SR = 16000
VAD_FRAME_SAMPLES = int(SR * VAD_FRAME_MS / 1000)


class VAD:
    """Voice Activity Detector.
    
    Uses webrtcvad to detect speech frames.
    Aggregates frame-level decisions to provide a smoothed speech signal.
    """

    def __init__(self, mode: int = 3) -> None:
        """
        Parameters
        ----------
        mode : int
            Aggressiveness mode (0-3). 3 is the most aggressive.
        """
        if webrtcvad is None:
            raise ImportError("webrtcvad is not installed. Run: pip install webrtcvad")
        
        self.vad = webrtcvad.Vad(mode)
        self.frame_duration_ms = VAD_FRAME_MS
        self.sample_rate = SR

    def is_speech(self, pcm: np.ndarray) -> bool:
        """Check if a frame contains speech.
        
        Parameters
        ----------
        pcm : np.ndarray
            float32 or int16 PCM audio. Must be exactly VAD_FRAME_MS long.
        """
        # Convert to 16-bit PCM if needed
        if pcm.dtype == np.float32:
            pcm_int16 = (pcm * 32767).astype(np.int16)
        else:
            pcm_int16 = pcm.astype(np.int16)
        
        return self.vad.is_speech(pcm_int16.tobytes(), self.sample_rate)

    def label_speech(self, pcm: np.ndarray) -> np.ndarray:
        """Return a boolean mask of speech frames for a long audio buffer.
        
        Parameters
        ----------
        pcm : np.ndarray
            Long float32 PCM buffer.
        """
        num_frames = len(pcm) // VAD_FRAME_SAMPLES
        mask = np.zeros(num_frames, dtype=bool)
        
        for i in range(num_frames):
            start = i * VAD_FRAME_SAMPLES
            frame = pcm[start : start + VAD_FRAME_SAMPLES]
            mask[i] = self.is_speech(frame)
            
        return mask


class SmoothedVAD:
    """VAD with temporal smoothing (hysteresis).
    
    Uses a moving window to decide when speech has started or ended.
    """

    def __init__(
        self,
        mode: int = 3,
        window_size: int = 10,
        trigger_threshold: float = 0.7,
        reset_threshold: float = 0.3,
    ) -> None:
        """
        Parameters
        ----------
        mode : int
            VAD aggressiveness.
        window_size : int
            Number of frames to look back for smoothing.
        trigger_threshold : float
            Fraction of frames in window that must be speech to trigger 'ON'.
        reset_threshold : float
            Fraction of frames in window that must be speech to stay 'ON'.
        """
        self.vad = VAD(mode)
        self.window_size = window_size
        self.trigger_threshold = trigger_threshold
        self.reset_threshold = reset_threshold
        
        self.ring_buffer = collections.deque(maxlen=window_size)
        self.is_active = False

    def process_frame(self, frame: np.ndarray) -> bool:
        """Process a single frame and return the smoothed speech state."""
        speech_frame = self.vad.is_speech(frame)
        self.ring_buffer.append(speech_frame)
        
        num_speech = sum(self.ring_buffer)
        fraction = num_speech / len(self.ring_buffer)
        
        if not self.is_active:
            if fraction >= self.trigger_threshold:
                self.is_active = True
        else:
            if fraction < self.reset_threshold:
                self.is_active = False
                
        return self.is_active

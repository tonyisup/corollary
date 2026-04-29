"""STG — Automatic Speech Recognition.

M0: Wraps faster-whisper (CTranslate2 backend) for transcription.
Returns segments with timestamps for efference-copy alignment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[misc, assignment]


# ── Default model ─────────────────────────────────────────────────────────
# tiny = fastest, good for testing
# base = reasonable accuracy/speed trade-off
# medium/large = better accuracy, slower
DEFAULT_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")

# Run on Apple Silicon CPU via CTranslate2 (Metal is not supported)
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")


@dataclass
class Segment:
    """A transcribed segment with timing."""

    id: int
    text: str
    start: float  # seconds
    end: float    # seconds
    confidence: float
    words: Optional[List["Word"]] = None


@dataclass
class Word:
    """A single word within a segment."""

    word: str
    start: float  # seconds
    end: float    # seconds
    probability: float


@dataclass
class TranscriptionResult:
    """Full transcription result."""

    segments: List[Segment]
    language: str

    @property
    def text(self) -> str:
        """Concatenate all segment text."""
        return " ".join(s.text.strip() for s in self.segments if s.text.strip()).strip()


def transcribe(
    audio: np.ndarray,
    sample_rate: int = 16000,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    beam_size: int = 5,
    word_timestamps: bool = False,
    initial_prompt: Optional[str] = None,
    mask: Optional[np.ndarray] = None,
) -> TranscriptionResult:
    """Transcribe audio using faster-whisper.

    Parameters
    ----------
    audio : np.ndarray
        float32 PCM audio at `sample_rate`.
    sample_rate : int
        Sample rate of the audio (must be 16000 for Whisper).
    model_size : str
        Whisper model size: tiny, base, small, medium, large,
        or a HuggingFace model path (e.g. "Systran/faster-whisper-base").
    device : str
        "cpu" or "cuda". Apple Silicon uses CPU (CTranslate2 is fast enough).
    compute_type : str
        "int8", "int8_float16", or "float32".
    beam_size : int
        Beam size for decoding.
    word_timestamps : bool
        Whether to return per-word timestamps.
    initial_prompt : str, optional
        Optional prompt to bias the model.
    mask : np.ndarray, optional
        Boolean mask or numeric gain vector of the same length as audio.
        Boolean values use False = zero, True = keep. Numeric values are
        applied as sample-level gains.

    Returns
    -------
    TranscriptionResult
    """
    if WhisperModel is None:
        raise ImportError(
            "faster-whisper is not installed. "
            "Run: pip install faster-whisper"
        )

    processed_audio = audio.astype(np.float32)
    if mask is not None:
        if len(mask) != len(audio):
            # Try to resample or pad mask if it's frame-level
            if len(mask) < len(audio) and len(audio) % len(mask) == 0:
                # Up-sample frame mask to sample level
                ratio = len(audio) // len(mask)
                mask = np.repeat(mask, ratio)
            
            # Clip to match
            min_len = min(len(mask), len(audio))
            mask = mask[:min_len]
            # Pad mask if shorter
            if len(mask) < len(audio):
                mask = np.pad(mask, (0, len(audio) - len(mask)), constant_values=True)
        
        processed_audio = processed_audio * mask.astype(np.float32)

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments_gen, info = model.transcribe(
        processed_audio,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
    )

    out_segments: List[Segment] = []
    for seg in segments_gen:
        words = None
        if word_timestamps and hasattr(seg, "words") and seg.words is not None:
            words = [
                Word(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
                for w in seg.words
            ]
        # Derive confidence from no_speech_prob and avg_logprob
        confidence = max(0.0, 1.0 - seg.no_speech_prob)
        out_segments.append(
            Segment(
                id=seg.id,
                text=seg.text,
                start=seg.start,
                end=seg.end,
                confidence=confidence,
                words=words,
            )
        )

    return TranscriptionResult(
        segments=out_segments,
        language=info.language,
    )

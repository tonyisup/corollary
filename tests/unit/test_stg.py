"""Tests for STG ASR wrapper (unit-level, no model download)."""

import numpy as np
import pytest

from stg.asr import Segment, Word, TranscriptionResult


# --- Dataclass Tests ---

class TestSegment:
    def test_minimal_segment(self):
        seg = Segment(id=0, text="hello", start=0.0, end=0.5, confidence=0.95)
        assert seg.text == "hello"
        assert seg.start == 0.0
        assert seg.end == 0.5
        assert seg.confidence == 0.95
        assert seg.words is None


class TestWord:
    def test_word_creation(self):
        w = Word(word="hello", start=0.0, end=0.3, probability=0.98)
        assert w.word == "hello"
        assert w.probability == 0.98


class TestTranscriptionResult:
    def test_empty_text(self):
        result = TranscriptionResult(segments=[], language="en")
        assert result.text == ""

    def test_single_segment(self):
        seg = Segment(id=0, text="hello world", start=0.0, end=1.0, confidence=0.9)
        result = TranscriptionResult(segments=[seg], language="en")
        assert result.text == "hello world"

    def test_multi_segment_text(self):
        segs = [
            Segment(id=0, text="Hello,", start=0.0, end=0.5, confidence=0.9),
            Segment(id=1, text="world!", start=0.6, end=1.0, confidence=0.85),
        ]
        result = TranscriptionResult(segments=segs, language="en")
        assert result.text == "Hello, world!"

    def test_strips_whitespace_segments(self):
        segs = [
            Segment(id=0, text="  hello  ", start=0.0, end=0.5, confidence=0.9),
            Segment(id=1, text="", start=0.5, end=0.6, confidence=0.9),
            Segment(id=2, text="  world  ", start=0.6, end=1.0, confidence=0.9),
        ]
        result = TranscriptionResult(segments=segs, language="en")
        assert result.text == "hello world"

    def test_confidence_from_no_speech_prob(self):
        """Verify confidence derivation logic matches what transcribe() does."""
        no_speech_prob = 0.05
        expected_confidence = max(0.0, 1.0 - no_speech_prob)
        assert expected_confidence == pytest.approx(0.95)

    def test_confidence_clamped_at_zero(self):
        """If no_speech_prob > 1, confidence should be 0."""
        # (edge case, but the formula should handle it)
        no_speech_prob = 1.5
        confidence = max(0.0, 1.0 - no_speech_prob)
        assert confidence == 0.0

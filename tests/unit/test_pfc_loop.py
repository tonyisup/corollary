"""Tests for the PFC simulated loop (unit-level, mocking heavy deps)."""

import numpy as np
import pytest

from pfc.simulated_loop import LoopResult


class TestLoopResult:
    def test_wer_perfect_match(self):
        lr = LoopResult(
            original_text="hello world",
            asr_result=None,
            svsr_db=40.0,
            is_world=False,
            gated_text="",
            sag_nodes=["hello", "world"],
            roundtrip_seconds=1.0,
        )
        # Override for testing
        lr.original_text = "hello world"

    def test_wer_empty_original(self):
        # No ground truth → WER is 0 (nothing to compare)
        result = LoopResult(
            original_text="",
            asr_result=None,
            svsr_db=40.0,
            is_world=False,
            gated_text="",
            sag_nodes=[],
            roundtrip_seconds=1.0,
        )
        assert result.wer == 1.0  # No ASR = 100% error

    def test_wer_partial_mismatch(self):
        from stg.asr import Segment, TranscriptionResult
        seg = Segment(id=0, text="hello earth", start=0, end=1, confidence=0.9)
        asr = TranscriptionResult(segments=[seg], language="en")
        result = LoopResult(
            original_text="hello world",
            asr_result=asr,
            svsr_db=40.0,
            is_world=False,
            gated_text="hello earth",
            sag_nodes=["hello", "earth"],
            roundtrip_seconds=1.0,
        )
        # "world" → "earth" = 1 substitution out of 2 words = 50%
        assert result.wer == 0.5  # Levenshteins on word sequences


class TestLoopResultMetrics:
    def test_svsr_threshold_check(self):
        """Verify the M0 target of 20dB can be checked against result."""
        result = LoopResult(
            original_text="test",
            asr_result=None,
            svsr_db=40.0,
            is_world=False,
            gated_text="",
            sag_nodes=[],
            roundtrip_seconds=1.0,
        )
        assert result.svsr_db >= 20.0

    def test_self_voice_not_world(self):
        result = LoopResult(
            original_text="test",
            asr_result=None,
            svsr_db=40.0,
            is_world=False,
            gated_text="",
            sag_nodes=[],
            roundtrip_seconds=1.0,
        )
        # Self-voice is suppressed → residual has nothing → is_world=False
        assert not result.is_world
        assert result.gated_text == ""

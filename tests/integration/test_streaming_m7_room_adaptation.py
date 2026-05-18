"""Integration tests for M7 trust-gated room filter adaptation."""

from __future__ import annotations

import numpy as np

from pfc.streaming_runtime import StreamingConfig, StreamingRuntime
from thalamus.attributor import AttributionResult, SourceVerdict


class FakeRoomAdapter:
    def __init__(self):
        self.calls = []
        self.room_ir = np.array([1.0, 0.25, 0.0], dtype=np.float32)
        self.total_updates = 0

    def update_trust_gated(
        self,
        known_pcm,
        captured_pcm,
        *,
        verdict,
        m4_confidence=None,
        self_match_score=None,
        adapt_confidence_threshold=0.8,
        adapt_match_threshold=0.2,
    ):
        self.calls.append(
            {
                "known_pcm": known_pcm.copy(),
                "captured_pcm": captured_pcm.copy(),
                "verdict": verdict,
                "m4_confidence": m4_confidence,
                "self_match_score": self_match_score,
                "adapt_confidence_threshold": adapt_confidence_threshold,
                "adapt_match_threshold": adapt_match_threshold,
            }
        )
        if verdict == "SELF" and m4_confidence >= adapt_confidence_threshold and self_match_score >= adapt_match_threshold:
            self.total_updates += 1
            self.room_ir = np.array([0.9, 0.3, 0.1], dtype=np.float32)
            return {
                "adapted": True,
                "reason": "trusted_self",
                "error": 0.01,
                "updates": self.total_updates,
            }
        return {
            "adapted": False,
            "reason": "held",
            "error": None,
            "updates": self.total_updates,
        }

    def get_learned_ir(self):
        return self.room_ir.copy()


def _attr(verdict: SourceVerdict, m4_confidence=0.95, room_score=0.45):
    return AttributionResult(
        verdict=verdict,
        confidence=0.9,
        self_match_score=room_score,
        is_in_motor_window=True,
        causal_score=1.0,
        energy_db=-20.0,
        is_speech=True,
        metadata={"m4_outcome": "accepted", "m4_confidence": m4_confidence},
        self_match_score_per_frame=room_score,
        room_local_score=room_score,
    )


class TestStreamingM7RoomAdaptation:
    def test_disabled_by_default_does_not_call_adapter(self):
        adapter = FakeRoomAdapter()
        runtime = StreamingRuntime(
            StreamingConfig(
                tts_pcm=np.zeros(16000, dtype=np.float32),
                room_adapter=adapter,
            )
        )

        result = runtime._update_room_filter_adaptation(
            known_pcm=np.ones(480, dtype=np.float32),
            captured_pcm=np.ones(480, dtype=np.float32),
            attr_result=_attr(SourceVerdict.SELF),
            self_match_score=0.45,
        )

        assert result is None
        assert adapter.calls == []

    def test_holds_on_both_even_with_high_confidence(self):
        adapter = FakeRoomAdapter()
        runtime = StreamingRuntime(
            StreamingConfig(
                tts_pcm=np.zeros(16000, dtype=np.float32),
                enable_room_filter_adaptation=True,
                room_adapter=adapter,
            )
        )

        result = runtime._update_room_filter_adaptation(
            known_pcm=np.ones(480, dtype=np.float32),
            captured_pcm=np.ones(480, dtype=np.float32),
            attr_result=_attr(SourceVerdict.BOTH, m4_confidence=0.99, room_score=0.99),
            self_match_score=0.99,
        )

        assert result["adapted"] is False
        assert adapter.calls[0]["verdict"] == "BOTH"
        assert runtime._last_room_adaptation_result["adapted"] is False

    def test_adapts_on_trusted_self_and_updates_attributor_filter(self):
        adapter = FakeRoomAdapter()
        runtime = StreamingRuntime(
            StreamingConfig(
                tts_pcm=np.zeros(16000, dtype=np.float32),
                enable_room_filter_adaptation=True,
                room_adapter=adapter,
                room_adapt_confidence_threshold=0.8,
                room_adapt_match_threshold=0.2,
            )
        )

        result = runtime._update_room_filter_adaptation(
            known_pcm=np.ones(480, dtype=np.float32),
            captured_pcm=np.ones(480, dtype=np.float32),
            attr_result=_attr(SourceVerdict.SELF, m4_confidence=0.92, room_score=0.45),
            self_match_score=0.45,
        )

        assert result["adapted"] is True
        assert result["reason"] == "trusted_self"
        assert adapter.calls[0]["m4_confidence"] == 0.92
        assert adapter.calls[0]["self_match_score"] == 0.45
        np.testing.assert_array_equal(runtime._attributor.room_filter, adapter.get_learned_ir())

    def test_uses_best_live_self_match_evidence_for_adaptation(self):
        """Live room-local score can be stale/low; strong per-frame evidence should count."""
        adapter = FakeRoomAdapter()
        runtime = StreamingRuntime(
            StreamingConfig(
                tts_pcm=np.zeros(16000, dtype=np.float32),
                enable_room_filter_adaptation=True,
                room_adapter=adapter,
                room_adapt_confidence_threshold=0.8,
                room_adapt_match_threshold=0.2,
            )
        )
        attr = _attr(SourceVerdict.SELF, m4_confidence=0.99, room_score=0.05)
        attr.self_match_score_per_frame = 0.42

        result = runtime._update_room_filter_adaptation(
            known_pcm=np.ones(480, dtype=np.float32),
            captured_pcm=np.ones(480, dtype=np.float32),
            attr_result=attr,
            self_match_score=0.01,
        )

        assert result["adapted"] is True
        assert result["reason"] == "trusted_self"
        assert adapter.calls[0]["self_match_score"] == 0.42

    def test_no_efference_chunk_holds_without_calling_adapter(self):
        adapter = FakeRoomAdapter()
        runtime = StreamingRuntime(
            StreamingConfig(
                tts_pcm=np.zeros(16000, dtype=np.float32),
                enable_room_filter_adaptation=True,
                room_adapter=adapter,
            )
        )

        result = runtime._update_room_filter_adaptation(
            known_pcm=np.zeros(480, dtype=np.float32),
            captured_pcm=np.ones(480, dtype=np.float32),
            attr_result=_attr(SourceVerdict.SELF),
            self_match_score=0.45,
        )

        assert result["adapted"] is False
        assert result["reason"] == "no_efference"
        assert adapter.calls == []

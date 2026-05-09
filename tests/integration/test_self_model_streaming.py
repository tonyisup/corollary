"""Integration test: streaming runtime → SelfModel → SAG.

Verifies that the full M6 pipeline works end-to-end:
streaming runtime produces frame logs → SelfModel processes them →
SAG self-concept subtree accumulates observations.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from pfc.streaming_runtime import StreamingRuntime, StreamingConfig
from pfc.self_model import SelfModel, NODE_ROOT, NODE_AGENT_VOICE, NODE_SPEECH_PATTERNS
from sag.graph import SemanticAbstractionGraph


SR = 16000


def make_test_audio(duration_s=3.0, seed=42):
    """Generate synthetic speech-like audio for testing."""
    rng = np.random.RandomState(seed)
    n = int(SR * duration_s)
    t = np.linspace(0, duration_s, n)
    f0 = 125
    signal = np.zeros(n)
    for h in range(1, 8):
        amp = 1.0 / (h ** 0.7)
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * f0 * h * t + phase)
    signal *= 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 4.0 * t))
    signal += rng.randn(n) * 0.03
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7
    return signal.astype(np.float32)


def test_self_model_wired_into_streaming_runtime():
    """SelfModel should receive frame logs after streaming run completes."""
    sag = SemanticAbstractionGraph()
    model = SelfModel()
    model.initialize_sag(sag)

    tts = make_test_audio(duration_s=3.0)

    config = StreamingConfig(
        tts_pcm=tts,
        tts_text="I was speaking, then paused, and then I continued.",
        sr=SR,
        m4_enable=False,
        gated_playback=False,
        self_model=model,
    )

    runtime = StreamingRuntime(config)
    logs = runtime.run_offline(tts)

    # SelfModel should have processed the frames
    assert model.patterns.session_count == 1
    assert model.patterns.total_speak_frames > 0

    # SAG should have updated nodes
    assert NODE_AGENT_VOICE in sag._nodes
    voice_node = sag._nodes[NODE_AGENT_VOICE]
    assert "mean_self_match" in voice_node.context


def test_self_model_accumulates_across_runs():
    """Multiple streaming runs should accumulate in the SAG."""
    sag = SemanticAbstractionGraph()
    model = SelfModel()
    model.initialize_sag(sag)

    for i in range(3):
        tts = make_test_audio(duration_s=2.0, seed=42 + i)
        config = StreamingConfig(
            tts_pcm=tts,
            sr=SR,
            m4_enable=False,
            gated_playback=False,
            self_model=model,
        )
        runtime = StreamingRuntime(config)
        runtime.run_offline(tts)

    assert model.patterns.session_count == 3
    summary = model.summarize()
    assert summary["patterns"]["session_count"] == 3
    assert summary["sag_nodes"] >= 4


def test_streaming_without_self_model_unchanged():
    """Streaming runtime should work identically without SelfModel."""
    tts = make_test_audio(duration_s=2.0)
    config = StreamingConfig(
        tts_pcm=tts,
        sr=SR,
        m4_enable=False,
        gated_playback=False,
        self_model=None,  # Default
    )
    runtime = StreamingRuntime(config)
    logs = runtime.run_offline(tts)

    assert len(logs) > 0
    summary = runtime.get_summary()
    assert summary["total_frames"] > 0


def test_self_model_query_after_streaming():
    """query_self_concept should return meaningful data after a run."""
    sag = SemanticAbstractionGraph()
    model = SelfModel()
    model.initialize_sag(sag)

    tts = make_test_audio(duration_s=3.0)
    config = StreamingConfig(
        tts_pcm=tts,
        sr=SR,
        m4_enable=False,
        gated_playback=False,
        self_model=model,
    )
    runtime = StreamingRuntime(config)
    runtime.run_offline(tts)

    concept = model.query_self_concept()
    assert NODE_ROOT in concept
    assert NODE_AGENT_VOICE in concept
    assert NODE_SPEECH_PATTERNS in concept

    # Voice node should have observed data
    voice_ctx = concept[NODE_AGENT_VOICE]["context"]
    assert "mean_self_match" in voice_ctx
    assert voice_ctx["total_sessions"] == 1


def test_self_model_with_known_subjects():
    """Known subjects should appear in SAG after being added."""
    sag = SemanticAbstractionGraph()
    model = SelfModel()
    model.initialize_sag(sag)

    model.add_known_subject("human_001", "Human speaker in room A",
                            context={"room": "A"})

    tts = make_test_audio(duration_s=2.0)
    config = StreamingConfig(
        tts_pcm=tts,
        sr=SR,
        m4_enable=False,
        gated_playback=False,
        self_model=model,
    )
    runtime = StreamingRuntime(config)
    runtime.run_offline(tts)

    concept = model.query_self_concept()
    subjects = concept.get("known_subjects", [])
    assert len(subjects) == 1
    assert "Human speaker" in subjects[0]["core"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

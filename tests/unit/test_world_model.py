"""Tests for multi-speaker world model."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))
from hippocampus import Hippocampus
from pfc.world_model import MultiSpeakerWorldModel, SpeakerProfile, SpeakerRegistry
from sag.graph import SemanticAbstractionGraph
from stg.speaker import SpeakerIdentifier


def test_speaker_profile_embedding_update():
    """Running average embedding update works correctly."""
    p = SpeakerProfile(speaker_id="s1", name="test")
    e1 = np.random.randn(192).astype(np.float32)
    e2 = np.random.randn(192).astype(np.float32)

    p.update_embedding(e1)
    assert p.embedding_count == 1
    np.testing.assert_array_almost_equal(p.embedding, e1)

    p.update_embedding(e2)
    assert p.embedding_count == 2
    expected = (e1 + e2) / 2
    np.testing.assert_array_almost_equal(p.embedding, expected)


def test_speaker_profile_similarity():
    """Cosine similarity computation is correct."""
    p = SpeakerProfile(speaker_id="s1", name="test")
    e1 = np.random.randn(192).astype(np.float32)
    p.update_embedding(e1)

    # Same embedding should have similarity ~1.0
    sim = p.similarity_to(e1)
    assert sim > 0.99

    # Orthogonal embedding should have similarity ~0
    e2 = np.random.randn(192).astype(np.float32)
    e2 = e2 - e1 * np.dot(e1, e2) / np.dot(e1, e1)  # orthogonalize
    sim2 = p.similarity_to(e2)
    assert abs(sim2) < 0.1


def test_speaker_profile_roundtrip():
    """SpeakerProfile serializes and deserializes correctly."""
    p = SpeakerProfile(
        speaker_id="s1",
        name="test",
        embedding=np.random.randn(192).astype(np.float32),
        interaction_count=5,
    )
    d = p.to_dict()
    restored = SpeakerProfile.from_dict(d)
    assert restored.speaker_id == "s1"
    assert restored.interaction_count == 5
    assert restored.embedding is not None
    np.testing.assert_array_almost_equal(p.embedding, restored.embedding)


def test_registry_register_and_identify():
    """Register speakers and identify by embedding."""
    registry = SpeakerRegistry(
        identifier=None,
        similarity_threshold=0.5,
    )

    e1 = np.random.randn(192).astype(np.float32)
    registry.register_speaker("s1", "speaker_one", embedding=e1)
    assert registry.speaker_count == 1

    # Identify with same embedding
    matched_id, sim = registry.identify_speaker(e1)
    assert matched_id == "s1"
    assert sim > 0.99

    # Identify with different embedding
    e2 = np.random.randn(192).astype(np.float32)
    matched_id, sim = registry.identify_speaker(e2)
    # May or may not match depending on random vectors
    # Just verify it doesn't crash
    assert matched_id is None or isinstance(matched_id, str)


def test_registry_update_from_frame_log_no_identifier():
    """Without identifier, update_from_frame_log returns empty."""
    registry = SpeakerRegistry(identifier=None)
    results = registry.update_from_frame_log([
        {"verdict": "WORLD", "frame_audio": np.zeros(480)},
    ])
    assert results == []


def test_world_model_process_interaction():
    """Process interaction identifies speakers and encodes to hippocampus."""
    identifier = SpeakerIdentifier(use_real=False)
    registry = SpeakerRegistry(identifier=identifier, similarity_threshold=0.3)
    h = Hippocampus()
    sag = SemanticAbstractionGraph()

    wm = MultiSpeakerWorldModel(registry, hippocampus=h, sag=sag)

    rng = np.random.RandomState(42)
    self_frames = [{"verdict": "SELF", "frame_audio": rng.randn(480).astype(np.float32)} for _ in range(10)]
    world_frames = [{"verdict": "WORLD", "frame_audio": rng.randn(480).astype(np.float32)} for _ in range(10)]
    frame_log = self_frames + world_frames

    results = wm.process_interaction(frame_log, session_id="test_session")

    assert "speakers_identified" in results
    assert "trace_id" in results
    assert results["trace_id"] is not None

    # Hippocampus should have a trace
    assert h.trace_count == 1

    # SAG should have speaker nodes
    assert len(sag._nodes) > 0


def test_world_model_query_speaker():
    """Query speaker returns profile and trace info."""
    registry = SpeakerRegistry(identifier=None)
    h = Hippocampus()
    sag = SemanticAbstractionGraph()

    wm = MultiSpeakerWorldModel(registry, hippocampus=h, sag=sag)

    # Register a speaker manually
    e1 = np.random.randn(192).astype(np.float32)
    registry.register_speaker("s1", "test_speaker", embedding=e1)

    # Encode a trace with this speaker
    h.encode(session_id="s1", known_speakers=["s1"])

    # Query
    info = wm.query_speaker("s1")
    assert info is not None
    assert info["speaker_id"] == "s1"
    assert info["name"] == "test_speaker"
    assert info["recent_traces"] >= 1


def test_world_model_query_unknown_speaker():
    """Query unknown speaker returns None."""
    registry = SpeakerRegistry(identifier=None)
    wm = MultiSpeakerWorldModel(registry)
    assert wm.query_speaker("unknown") is None


def test_world_model_summary():
    """Summary reports correct state."""
    registry = SpeakerRegistry(identifier=None)
    h = Hippocampus()
    sag = SemanticAbstractionGraph()

    wm = MultiSpeakerWorldModel(registry, hippocampus=h, sag=sag)

    e1 = np.random.randn(192).astype(np.float32)
    registry.register_speaker("s1", "speaker_one", embedding=e1)
    registry.register_speaker("s2", "speaker_two", embedding=e1)

    s = wm.summary()
    assert s["known_speakers"] == 2
    assert "s1" in s["speaker_ids"]
    assert "s2" in s["speaker_ids"]


def test_world_model_multiple_interactions():
    """Multiple interactions build up speaker history."""
    identifier = SpeakerIdentifier(use_real=False)
    registry = SpeakerRegistry(identifier=identifier, similarity_threshold=0.3)
    h = Hippocampus()
    sag = SemanticAbstractionGraph()

    wm = MultiSpeakerWorldModel(registry, hippocampus=h, sag=sag)

    # Process 3 interactions
    for i in range(3):
        rng = np.random.RandomState(42 + i)
        frame_log = [
            {"verdict": "SELF", "frame_audio": rng.randn(480).astype(np.float32)}
            for _ in range(5)
        ] + [
            {"verdict": "WORLD", "frame_audio": rng.randn(480).astype(np.float32)}
            for _ in range(5)
        ]
        wm.process_interaction(frame_log, session_id=f"session_{i}")

    # Should have traces
    assert h.trace_count == 3

    # Should have at least one speaker
    assert registry.speaker_count >= 1

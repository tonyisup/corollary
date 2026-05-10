"""Tests for Hippocampus episodic memory."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))
from hippocampus import Hippocampus, EpisodicTrace, ConsolidationCandidate


def test_encode_basic():
    """Basic encoding creates a trace with correct counts."""
    h = Hippocampus()
    frame_log = [
        {"verdict": "SELF"}, {"verdict": "SELF"}, {"verdict": "WORLD"},
        {"verdict": "BOTH"}, {"verdict": "SILENCE"},
    ]

    trace = h.encode(
        session_id="test_session",
        frame_log=frame_log,
        m4_outcomes={"accepted": 2, "below_threshold": 1},
        interaction_events={"interrupts": 1, "defers": 0, "resumes": 0, "collision_repairs": 0},
        known_speakers=["self", "world_speaker_1"],
    )

    assert trace.self_frames == 2
    assert trace.world_frames == 1
    assert trace.both_frames == 1
    assert trace.silence_frames == 1
    assert trace.m4_accepted_frames == 2
    assert trace.m4_attempted_frames == 3
    assert trace.interrupt_events == 1
    assert "self" in trace.known_speakers
    assert "world_speaker_1" in trace.known_speakers
    assert trace.salience_score > 0


def test_encode_empty_frame_log():
    """Encoding with no frame log creates a valid trace."""
    h = Hippocampus()
    trace = h.encode(session_id="empty_session")
    assert trace.self_frames == 0
    assert trace.world_frames == 0
    assert trace.salience_score == 0.0


def test_retrieve_by_session():
    """Retrieve traces filtered by session ID."""
    h = Hippocampus()
    for i in range(3):
        h.encode(session_id="session_a", frame_log=[{"verdict": "SELF"}])
    for i in range(2):
        h.encode(session_id="session_b", frame_log=[{"verdict": "WORLD"}])

    a_traces = h.get_session_traces("session_a")
    b_traces = h.get_session_traces("session_b")
    assert len(a_traces) == 3
    assert len(b_traces) == 2


def test_retrieve_by_salience():
    """Retrieve only salient traces."""
    h = Hippocampus()
    # Low salience: no interaction events
    h.encode(session_id="low", frame_log=[{"verdict": "SELF"}] * 10)
    # High salience: with interrupts
    h.encode(
        session_id="high",
        frame_log=[{"verdict": "SELF"}] * 10,
        interaction_events={"interrupts": 3},
    )

    salient = h.get_salient_traces(threshold=0.1)
    assert len(salient) == 1
    assert salient[0].session_id == "high"


def test_retrieve_with_limit():
    """Retrieve respects the limit parameter."""
    h = Hippocampus()
    for i in range(10):
        h.encode(session_id=f"s{i}")

    results = h.retrieve(limit=3)
    assert len(results) == 3
    # Most recent first
    assert results[0].session_id == "s9"


def test_persistence():
    """Traces persist to disk and can be reloaded."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        h1 = Hippocampus(persist_path=path)
        h1.encode(session_id="persist_test", frame_log=[{"verdict": "SELF"}] * 5)
        assert h1.trace_count == 1

        # Reload
        h2 = Hippocampus(persist_path=path)
        assert h2.trace_count == 1
        assert h2._traces[0].session_id == "persist_test"
        assert h2._traces[0].self_frames == 5
    finally:
        os.unlink(path)


def test_consolidation_candidates_need_min_traces():
    """Consolidation requires minimum number of traces."""
    h = Hippocampus()
    for i in range(2):
        h.encode(
            session_id=f"s{i}",
            known_speakers=["speaker_a"],
            frame_log=[{"verdict": "SELF"}],
        )

    candidates = h.get_consolidation_candidates(min_traces=3)
    assert len(candidates) == 0


def test_consolidation_candidates_speaker_pattern():
    """Speaker attribution pattern is detected."""
    h = Hippocampus()
    for i in range(5):
        h.encode(
            session_id=f"s{i}",
            known_speakers=["speaker_a"],
            frame_log=[{"verdict": "SELF"}] * 10 + [{"verdict": "WORLD"}] * 5,
        )

    candidates = h.get_consolidation_candidates(min_traces=3)
    speaker_candidates = [c for c in candidates if "speaker_a" in c.pattern_id]
    assert len(speaker_candidates) == 1
    assert speaker_candidates[0].trace_count == 5


def test_consolidation_candidates_interaction_pattern():
    """Frequent interruption pattern is detected."""
    h = Hippocampus()
    for i in range(4):
        h.encode(
            session_id=f"s{i}",
            frame_log=[{"verdict": "SELF"}] * 10,
            interaction_events={"interrupts": 2},
        )

    candidates = h.get_consolidation_candidates(min_traces=3)
    interrupt_candidates = [c for c in candidates if "interruption" in c.pattern_id]
    assert len(interrupt_candidates) == 1
    assert interrupt_candidates[0].trace_count == 4


def test_summary():
    """Summary returns correct aggregate stats."""
    h = Hippocampus()
    h.encode(
        session_id="s1",
        frame_log=[{"verdict": "SELF"}] * 10 + [{"verdict": "WORLD"}] * 5,
        interaction_events={"interrupts": 1, "collision_repairs": 1},
        known_speakers=["self", "other"],
    )

    s = h.summary()
    assert s["trace_count"] == 1
    assert s["session_count"] == 1
    assert s["total_self_frames"] == 10
    assert s["total_world_frames"] == 5
    assert s["total_interrupts"] == 1
    assert s["total_repairs"] == 1
    assert "self" in s["known_speakers"]
    assert "other" in s["known_speakers"]


def test_memory_limit():
    """Old traces are evicted when memory limit is exceeded."""
    h = Hippocampus(max_traces_in_memory=5)
    for i in range(10):
        h.encode(session_id=f"s{i}")

    assert h.trace_count == 5
    # Most recent traces are kept
    session_ids = [t.session_id for t in h._traces]
    assert "s5" in session_ids
    assert "s9" in session_ids
    assert "s0" not in session_ids


def test_trace_from_dict():
    """EpisodicTrace round-trips through dict serialization."""
    trace = EpisodicTrace(
        trace_id="test_001",
        timestamp_s=1234567890.0,
        session_id="test",
        self_frames=10,
        world_frames=5,
        m4_accepted_frames=8,
        m4_attempted_frames=10,
        salience_score=0.5,
    )
    d = trace.to_dict()
    restored = EpisodicTrace.from_dict(d)
    assert restored.trace_id == "test_001"
    assert restored.self_frames == 10
    assert restored.salience_score == 0.5

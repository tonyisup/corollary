"""Tests for offline consolidation loop."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))
from hippocampus import Hippocampus, ConsolidationCandidate
from hippocampus.consolidation import OfflineConsolidator
from sag.graph import SemanticAbstractionGraph


def test_consolidate_no_candidates():
    """Consolidation with too few traces returns empty result."""
    h = Hippocampus()
    sag = SemanticAbstractionGraph()
    c = OfflineConsolidator(h, sag, min_traces_for_pattern=3)

    for i in range(2):
        h.encode(session_id=f"s{i}", known_speakers=["speaker_a"])

    result = c.consolidate(force=True)
    assert result["consolidated"] is True
    assert result["candidates_found"] == 0


def test_consolidate_speaker_pattern():
    """Speaker attribution pattern is consolidated into SAG."""
    h = Hippocampus()
    sag = SemanticAbstractionGraph()
    c = OfflineConsolidator(h, sag, min_traces_for_pattern=3)

    for i in range(5):
        h.encode(
            session_id=f"s{i}",
            known_speakers=["speaker_a"],
            frame_log=[{"verdict": "SELF"}] * 10 + [{"verdict": "WORLD"}] * 5,
        )

    result = c.consolidate(force=True)
    assert result["consolidated"] is True
    assert result["candidates_found"] >= 1
    assert result["nodes_added"] >= 1
    assert result["edges_added"] >= 1

    # Check SAG was updated
    assert len(sag._nodes) > 0
    assert "pattern_speaker_speaker_a" in sag._nodes


def test_consolidate_interaction_pattern():
    """Frequent interruption pattern is consolidated."""
    h = Hippocampus()
    sag = SemanticAbstractionGraph()
    c = OfflineConsolidator(h, sag, min_traces_for_pattern=3)

    for i in range(4):
        h.encode(
            session_id=f"s{i}",
            frame_log=[{"verdict": "SELF"}] * 10,
            interaction_events={"interrupts": 2},
        )

    result = c.consolidate(force=True)
    assert result["candidates_found"] >= 1

    # Check for interruption pattern node
    pattern_nodes = [n for n in sag._nodes if "interruption" in n]
    assert len(pattern_nodes) >= 1


def test_consolidate_interval_throttle():
    """Consolidation respects interval throttle."""
    h = Hippocampus()
    sag = SemanticAbstractionGraph()
    c = OfflineConsolidator(h, sag, min_traces_for_pattern=3, consolidation_interval_s=60.0)

    for i in range(5):
        h.encode(session_id=f"s{i}", known_speakers=["speaker_a"])

    # First call should run
    result1 = c.consolidate(force=False)
    assert result1["consolidated"] is True

    # Second call immediately should be throttled
    result2 = c.consolidate(force=False)
    assert result2["consolidated"] is False
    assert result2["reason"] == "interval_not_elapsed"

    # Force should bypass throttle
    result3 = c.consolidate(force=True)
    assert result3["consolidated"] is True


def test_consolidate_persistence():
    """Consolidation summary is persisted to disk."""
    h = Hippocampus()
    sag = SemanticAbstractionGraph()

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        c = OfflineConsolidator(h, sag, min_traces_for_pattern=3, persist_path=path)

        for i in range(5):
            h.encode(session_id=f"s{i}", known_speakers=["speaker_a"])

        c.consolidate(force=True)

        # Check persistence file
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) >= 1
    finally:
        os.unlink(path)


def test_consolidator_summary():
    """Consolidator summary reports correct state."""
    h = Hippocampus()
    sag = SemanticAbstractionGraph()
    c = OfflineConsolidator(h, sag)

    for i in range(3):
        h.encode(session_id=f"s{i}")

    s = c.summary()
    assert s["hippocampus_traces"] == 3
    assert s["consolidation_count"] == 0
    assert s["sag_nodes"] == 0


def test_full_pipeline_integration():
    """End-to-end: hippocampus encoding → consolidation → SAG update."""
    h = Hippocampus()
    sag = SemanticAbstractionGraph()
    c = OfflineConsolidator(h, sag, min_traces_for_pattern=3)

    # Simulate 5 interaction sessions
    for i in range(5):
        frame_log = [{"verdict": "SELF"}] * 20 + [{"verdict": "WORLD"}] * 10
        if i % 2 == 0:
            frame_log += [{"verdict": "BOTH"}] * 5

        h.encode(
            session_id=f"session_{i}",
            frame_log=frame_log,
            m4_outcomes={"accepted": 15, "below_threshold": 5},
            interaction_events={"interrupts": 1 if i % 2 == 0 else 0},
            known_speakers=["self", "speaker_a"] if i % 2 == 0 else ["self"],
        )

    # Run consolidation
    result = c.consolidate(force=True)

    assert result["consolidated"] is True
    assert result["candidates_found"] >= 1
    assert len(sag._nodes) > 0

    # Verify hippocampus state
    assert h.trace_count == 5
    assert h.session_count == 5

    # Verify consolidation found patterns
    candidates = h.get_consolidation_candidates(min_traces=3)
    assert len(candidates) >= 1

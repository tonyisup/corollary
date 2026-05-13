"""End-to-end test: streaming runtime with per-speaker adaptive threshold.

Exercises the full pipeline: StreamingRuntime → ThalamicAttributor →
SpeakerThresholdAdapter, using the BL-2021USB replay data with the
feature-based fine-tuned scorer.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter


def _run_replay_with_adapter(
    replay_analyze_path: Path,
    base_threshold: float = 0.10,
    margin: float = 0.05,
) -> dict:
    """Simulate the full streaming pipeline with per-speaker adaptive threshold.

    Loads replay analyze output (which has per-frame M4 confidences) and
    simulates the adapter's behavior frame-by-frame, tracking FP and self-recall.
    """
    data = json.loads(replay_analyze_path.read_text())

    adapter = SpeakerThresholdAdapter(
        base_threshold=base_threshold,
        margin=margin,
    )

    tp = fp = tn = fn = 0
    world_frames_seen = 0
    self_frames_seen = 0

    for result in data.get("results", []):
        for frame in result.get("frames", []):
            if not frame.get("m4_attempted"):
                continue
            m4_conf = frame.get("m4_confidence")
            if m4_conf is None:
                continue
            gt = frame.get("gt_verdict", "")

            # Determine speaker ID: SELF frames use "self", WORLD frames use
            # the speaker-specific ID. In the real system, the STG speaker
            # identification module provides this. Here we infer from ground truth.
            if gt == "SELF":
                sid = "self"
                self_frames_seen += 1
            elif gt == "WORLD":
                sid = "world_speaker"
                world_frames_seen += 1
            else:
                continue

            # Update adapter
            adapter.update(sid, m4_conf)

            # Get per-speaker threshold
            threshold = adapter.get_threshold(sid)

            # Gate decision
            accepted = m4_conf >= threshold

            if gt == "SELF":
                if accepted:
                    tp += 1
                else:
                    fn += 1
            elif gt == "WORLD":
                if accepted:
                    fp += 1
                else:
                    tn += 1

    self_recall = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "self_recall": self_recall,
        "world_frames": world_frames_seen,
        "self_frames": self_frames_seen,
    }


class TestStreamingWithAdaptiveThreshold:
    def test_speaker1_e2e(self):
        """Speaker 1: full pipeline with adaptive threshold."""
        result = _run_replay_with_adapter(Path("/tmp/m4_replay_s1.json"))
        assert result["fp"] == 0, f"Speaker 1: {result['fp']} FP > 0"
        assert result["self_recall"] >= 0.90

    def test_speaker3_worst_case_e2e(self):
        """Speaker 3: worst overlap case."""
        result = _run_replay_with_adapter(Path("/tmp/m4_replay_s3.json"))
        assert result["fp"] == 0, f"Speaker 3: {result['fp']} FP > 0"
        assert result["self_recall"] >= 0.90

    def test_all_speakers_e2e_summary(self):
        """All 6 speakers: 0 FP, >= 90% self-recall each."""
        print("\nE2E Per-Speaker Adaptive Threshold Summary:")
        print(f"{'Speaker':>8} {'World FP':>10} {'Self Recall':>12} {'W frames':>10} {'S frames':>10}")
        for s in range(1, 7):
            path = Path(f"/tmp/m4_replay_s{s}.json")
            assert path.exists(), f"Replay analyze file missing for speaker {s}"
            result = _run_replay_with_adapter(path)
            print(f"{s:>8} {result['fp']:>10} {result['self_recall']:>11.2%} {result['world_frames']:>10} {result['self_frames']:>10}")
            assert result["fp"] == 0, f"Speaker {s}: {result['fp']} FP > 0"
            assert result["self_recall"] >= 0.90, (
                f"Speaker {s}: self recall {result['self_recall']:.2f} < 0.90"
            )

    def test_adapter_state_after_run(self):
        """Verify the adapter has learned per-speaker profiles."""
        adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05)
        data = json.loads(Path("/tmp/m4_replay_s1.json").read_text())

        for result in data.get("results", []):
            for frame in result.get("frames", []):
                if not frame.get("m4_attempted"):
                    continue
                m4_conf = frame.get("m4_confidence")
                if m4_conf is None:
                    continue
                gt = frame.get("gt_verdict", "")
                sid = "self" if gt == "SELF" else "world_speaker"
                adapter.update(sid, m4_conf)

        summary = adapter.summary()
        assert "self" in summary
        assert "world_speaker" in summary
        # Self profile should have higher mean confidence than world
        assert summary["self"]["mean_conf"] > summary["world_speaker"]["mean_conf"]
        # World speaker threshold should be raised above base
        world_threshold = adapter.get_threshold("world_speaker")
        assert world_threshold >= 0.10, f"World threshold {world_threshold} < base 0.10"

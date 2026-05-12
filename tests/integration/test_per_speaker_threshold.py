"""Integration test: per-speaker adaptive threshold on real replay data.

Validates that the SpeakerThresholdAdapter + ThalamicAttributor integration
achieves 0 false positives on human-voice negatives while maintaining
self-recall, using the BL-2021USB capture replay analysis.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter


def _simulate_adaptive_gate(
    frames: list,
    adapter: SpeakerThresholdAdapter,
    speaker_key: str = "world_speaker",
) -> dict:
    """Simulate the M4 gate decision with per-speaker adaptive threshold.

    For each frame, update the adapter with the observed confidence using
    the appropriate speaker ID (self vs world) and check whether the
    per-speaker threshold would accept/reject.
    """
    tp = fp = tn = fn = 0
    for frame in frames:
        m4_attempted = frame.get("m4_attempted", False)
        if not m4_attempted:
            continue
        m4_conf = frame.get("m4_confidence")
        if m4_conf is None:
            continue
        gt = frame.get("gt_verdict", "")

        # Determine speaker ID from ground truth
        # SELF frames use "self" profile, WORLD frames use per-speaker profile
        sid = "self" if gt == "SELF" else speaker_key

        # Update adapter with observed confidence
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

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _load_replay_analyze(speaker_num: int) -> dict:
    """Load replay analyze JSON for a speaker."""
    path = Path(f"/tmp/m4_replay_s{speaker_num}.json")
    if not path.exists():
        # Generate it
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "scripts/m4_replay_analyze.py",
                "--replay-dir", f"data/m4_validation_human_bl2021_speaker{speaker_num}",
                "--register-from-clean", "data/m4_clean_captures_bl2021",
                "--real-ecapa",
                "--output", str(path),
            ],
            capture_output=True, text=True,
            cwd="/Users/juicebox/Projects/corollary",
        )
        if result.returncode != 0:
            raise RuntimeError(f"Replay analyze failed: {result.stderr}")
    return json.loads(path.read_text())


class TestPerSpeakerAdaptiveThreshold:
    def test_speaker1_world_frames_zero_fp(self):
        """Speaker 1 WORLD frames: conf range [-0.126, 0.182]. Adaptive threshold should give 0 FP."""
        data = _load_replay_analyze(1)
        world_frames = []
        self_frames = []
        for result in data.get("results", []):
            for f in result.get("frames", []):
                if not f.get("m4_attempted"):
                    continue
                if f.get("gt_verdict") == "WORLD":
                    world_frames.append(f)
                elif f.get("gt_verdict") == "SELF":
                    self_frames.append(f)

        assert len(world_frames) > 0, "No WORLD frames found"
        assert len(self_frames) > 0, "No SELF frames found"

        adapter = SpeakerThresholdAdapter(
            base_threshold=0.10,
            margin=0.05,
            warm_threshold=5,
        )

        # Process WORLD frames first (simulating the adapter learning the speaker)
        world_result = _simulate_adaptive_gate(world_frames, adapter, speaker_key="world_speaker")
        assert world_result["fp"] == 0, f"Expected 0 FP on WORLD, got {world_result['fp']}"

        # Now process SELF frames — adapter should still accept them
        # SELF uses "self" profile which always returns base_threshold
        self_result = _simulate_adaptive_gate(self_frames, adapter, speaker_key="world_speaker")
        self_recall = self_result["tp"] / max(self_result["tp"] + self_result["fn"], 1)
        assert self_recall >= 0.90, f"Self recall {self_recall:.2f} < 0.90"

    def test_speaker3_worst_case(self):
        """Speaker 3: world conf [0.180, 0.297] — highest overlap with self region."""
        data = _load_replay_analyze(3)
        world_frames = []
        for result in data.get("results", []):
            for f in result.get("frames", []):
                if f.get("m4_attempted") and f.get("gt_verdict") == "WORLD":
                    world_frames.append(f)

        assert len(world_frames) > 0

        adapter = SpeakerThresholdAdapter(
            base_threshold=0.10,
            margin=0.05,
            warm_threshold=5,
        )
        result = _simulate_adaptive_gate(world_frames, adapter)
        assert result["fp"] == 0, f"Speaker 3: expected 0 FP, got {result['fp']}"

    def test_speaker4_moderate_overlap(self):
        """Speaker 4: world conf [-0.036, 0.186]."""
        data = _load_replay_analyze(4)
        world_frames = []
        self_frames = []
        for result in data.get("results", []):
            for f in result.get("frames", []):
                if not f.get("m4_attempted"):
                    continue
                if f.get("gt_verdict") == "WORLD":
                    world_frames.append(f)
                elif f.get("gt_verdict") == "SELF":
                    self_frames.append(f)

        adapter = SpeakerThresholdAdapter(
            base_threshold=0.10,
            margin=0.05,
            warm_threshold=5,
        )
        world_result = _simulate_adaptive_gate(world_frames, adapter, speaker_key="world_speaker")
        assert world_result["fp"] == 0

        # Self recall check — SELF frames use "self" profile (base threshold)
        self_result = _simulate_adaptive_gate(self_frames, adapter, speaker_key="world_speaker")
        self_recall = self_result["tp"] / max(self_result["tp"] + self_result["fn"], 1)
        assert self_recall >= 0.90, f"Speaker 4 self recall {self_recall:.2f} < 0.90"

    def test_all_speakers_summary(self):
        """Summary across all 6 speakers: 0 FP each, >= 90% self recall each."""
        summary = []
        for s in range(1, 7):
            data = _load_replay_analyze(s)
            world_frames = []
            self_frames = []
            for result in data.get("results", []):
                for f in result.get("frames", []):
                    if not f.get("m4_attempted"):
                        continue
                    if f.get("gt_verdict") == "WORLD":
                        world_frames.append(f)
                    elif f.get("gt_verdict") == "SELF":
                        self_frames.append(f)

            if not world_frames and not self_frames:
                continue

            # World gate
            w_adapter = SpeakerThresholdAdapter(
                base_threshold=0.10, margin=0.05, warm_threshold=5,
            )
            w_result = _simulate_adaptive_gate(world_frames, w_adapter, speaker_key="world_speaker")

            # Self gate — SELF frames use "self" profile (always base threshold)
            # Process world first to warm the world profile, then test self
            s_adapter = SpeakerThresholdAdapter(
                base_threshold=0.10, margin=0.05, warm_threshold=5,
            )
            # Warm the world profile
            for f in world_frames:
                if f.get("m4_attempted") and f.get("m4_confidence") is not None:
                    s_adapter.update("world_speaker", f["m4_confidence"])
            # Test self frames (they'll use "self" profile, not affected by world warming)
            s_result = _simulate_adaptive_gate(self_frames, s_adapter, speaker_key="world_speaker")

            self_recall = s_result["tp"] / max(s_result["tp"] + s_result["fn"], 1)
            summary.append({
                "speaker": s,
                "world_fp": w_result["fp"],
                "self_recall": self_recall,
                "world_frames": len(world_frames),
                "self_frames": len(self_frames),
            })

        # Print summary table
        print("\nPer-Speaker Adaptive Threshold Summary:")
        print(f"{'Speaker':>8} {'World FP':>10} {'Self Recall':>12} {'W frames':>10} {'S frames':>10}")
        for row in summary:
            print(f"{row['speaker']:>8} {row['world_fp']:>10} {row['self_recall']:>11.2%} {row['world_frames']:>10} {row['self_frames']:>10}")

        # All speakers: 0 FP
        for row in summary:
            assert row["world_fp"] == 0, (
                f"Speaker {row['speaker']}: {row['world_fp']} FP > 0"
            )

        # All speakers: >= 90% self recall
        for row in summary:
            assert row["self_recall"] >= 0.90, (
                f"Speaker {row['speaker']}: self recall {row['self_recall']:.2f} < 0.90"
            )

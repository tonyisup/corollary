#!/usr/bin/env python3
"""Per-frame M4 replay analyzer.

Replays a captured-audio artifact through the M4 pipeline and emits a detailed
per-frame breakdown showing:
  - ground-truth verdict
  - match_score (template correlation)
  - m4_outcome (accepted / below_threshold / ...)
  - m4_confidence
  - m4_template_match_passed
  - final verdict

Usage:
    .venv/bin/python scripts/m4_replay_analyze.py \
        --replay-dir data/m4_capture_validation_latest \
        --output /tmp/m4_replay_analysis.json

    .venv/bin/python scripts/m4_replay_analyze.py \
        --replay-dir data/m4_capture_validation_latest \
        --register-from-clean data/m4_clean_captures_fixed \
        --threshold-sweep 0.0 0.05 0.10 0.15 0.20 0.25 0.30 \
        --output /tmp/m4_replay_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from stg.speaker import SpeakerIdentifier, SelfVoiceDetector
from thalamus.attributor import ThalamicAttributor, SourceVerdict

SR = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SR * FRAME_MS / 1000)


def _load_replay_cases(replay_dir: str) -> list[dict]:
    """Load replay cases from a validation artifact directory."""
    replay_path = Path(replay_dir)
    cases = []
    for scenario_dir in sorted(replay_path.iterdir()):
        if not scenario_dir.is_dir():
            continue
        audio_path = scenario_dir / "live_captured.npy"
        agent_path = scenario_dir / "agent_pcm.npy"
        records_path = scenario_dir / "frame_records.json"
        meta_path = scenario_dir / "metadata.json"
        if not audio_path.exists() or not agent_path.exists() or not records_path.exists():
            continue
        audio = np.load(audio_path).astype(np.float32)
        agent_pcm = np.load(agent_path).astype(np.float32)
        with open(records_path) as f:
            records = json.load(f)
        gt_verdicts = [r.get("verdict", "SILENCE") for r in records]
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        cases.append({
            "name": scenario_dir.name,
            "audio": audio,
            "agent_pcm": agent_pcm,
            "gt_verdicts": gt_verdicts,
            "metadata": metadata,
        })
    return cases


def _build_attributor(
    *,
    real_ecapa: bool,
    m4_self_threshold: float = 0.10,
    m4_template_match_threshold: float = 0.20,
    m4_min_buffer_ms: float = 300.0,
    room_filter: Optional[np.ndarray] = None,
    self_embedding: Optional[np.ndarray] = None,
) -> ThalamicAttributor:
    identifier = SpeakerIdentifier(use_real=real_ecapa)
    identifier._load_model()
    m4_detector = SelfVoiceDetector(identifier, threshold=m4_self_threshold)
    if self_embedding is not None:
        m4_detector.register_self_embedding(self_embedding)
    return ThalamicAttributor(
        similarity_threshold=0.04,
        energy_threshold_db=-50.0,
        causal_tolerance_ms=100.0,
        sr=SR,
        m4_detector=m4_detector,
        m4_self_threshold=m4_self_threshold,
        m4_template_match_threshold=m4_template_match_threshold,
        m4_min_buffer_ms=m4_min_buffer_ms,
        room_filter=room_filter,
        room_local_threshold=0.04 if room_filter is not None else None,
    )


def analyze_replay(
    *,
    replay_dir: str,
    real_ecapa: bool = True,
    m4_self_threshold: float = 0.10,
    m4_template_match_threshold: float = 0.20,
    m4_min_buffer_ms: float = 300.0,
    register_from_clean: Optional[str] = None,
    scenario_filter: Optional[str] = None,
) -> dict:
    """Run per-frame analysis on a replay artifact."""
    sys.path.insert(0, os.path.dirname(__file__))
    from m4_f1_benchmark import register_from_clean_capture  # noqa: E402

    cases = _load_replay_cases(replay_dir)
    if scenario_filter:
        cases = [c for c in cases if scenario_filter in c["name"]]

    room_filter = None
    self_embedding = None
    if register_from_clean:
        room_filter, self_embedding = register_from_clean_capture(
            register_from_clean,
            identifier=SpeakerIdentifier(use_real=real_ecapa),
        )

    attributor = _build_attributor(
        real_ecapa=real_ecapa,
        m4_self_threshold=m4_self_threshold,
        m4_template_match_threshold=m4_template_match_threshold,
        m4_min_buffer_ms=m4_min_buffer_ms,
        room_filter=room_filter,
        self_embedding=self_embedding,
    )

    results = []
    for case in cases:
        if hasattr(attributor, "_m4_buffer"):
            attributor._m4_buffer = None
        frame_records = _analyze_case(case, attributor)
        summary = _summarize_case(case["name"], frame_records)
        results.append({"case": case["name"], "summary": summary, "frames": frame_records})

    return {
        "replay_dir": replay_dir,
        "real_ecapa": real_ecapa,
        "m4_self_threshold": m4_self_threshold,
        "m4_template_match_threshold": m4_template_match_threshold,
        "register_from_clean": register_from_clean,
        "results": results,
    }


def _analyze_case(case: dict, attributor: ThalamicAttributor) -> list[dict]:
    audio = case["audio"]
    agent_pcm = case["agent_pcm"]
    gt_verdicts = case["gt_verdicts"]
    n_frames = min(len(gt_verdicts), len(audio) // FRAME_SAMPLES)
    frames = []
    for fi in range(n_frames):
        s_start = fi * FRAME_SAMPLES
        s_end = s_start + FRAME_SAMPLES
        if s_end > len(audio):
            break
        frame_audio = audio[s_start:s_end]
        rms = float(np.sqrt(np.mean(frame_audio ** 2)))
        rms_db = 20 * np.log10(max(rms, 1e-9))

        ref_start = min(s_start, len(agent_pcm) - FRAME_SAMPLES)
        ref_end = ref_start + FRAME_SAMPLES
        if ref_start < 0 or ref_end > len(agent_pcm):
            match_score = 0.0
        else:
            ref_frame = agent_pcm[ref_start:ref_end]
            if attributor.room_filter is not None:
                ref_frame = np.convolve(ref_frame, attributor.room_filter, mode="same").astype(np.float32)
            obs_norm = float(np.linalg.norm(frame_audio))
            ref_norm = float(np.linalg.norm(ref_frame))
            match_score = 0.0 if obs_norm < 1e-9 or ref_norm < 1e-9 else float(
                np.dot(frame_audio, ref_frame) / (obs_norm * ref_norm))

        is_playing = s_start < len(agent_pcm)
        is_speech = rms_db > -40.0

        result = attributor.attribute(
            match_score=match_score,
            observed_onset_samples=s_start,
            expected_onset_samples=s_start,
            rms_db=rms_db,
            is_playing=is_playing,
            is_speech=is_speech,
            frame_audio=frame_audio,
            self_pcm=agent_pcm,
            agent_start_sample=0,
        )

        m4_meta = result.metadata or {}
        gt = gt_verdicts[fi] if fi < len(gt_verdicts) else "SILENCE"
        frames.append({
            "frame_index": fi,
            "timestamp_s": round(s_start / SR, 4),
            "gt_verdict": gt,
            "final_verdict": result.verdict.name,
            "match_score": round(match_score, 4),
            "m4_outcome": m4_meta.get("m4_outcome", "not_attempted"),
            "m4_confidence": round(m4_meta.get("m4_confidence", 0.0), 4) if m4_meta.get("m4_confidence") is not None else None,
            "m4_template_match_passed": bool(m4_meta.get("m4_template_match_passed", False)),
            "m4_attempted": bool(m4_meta.get("m4_attempted", False)),
            "energy_db": round(rms_db, 2),
        })
    return frames


def _summarize_case(name: str, frames: list[dict]) -> dict:
    # Match m4_f1_benchmark replay semantics: score speaker identity only on
    # speech-active frames. Captured artifacts may label a whole scheduled
    # speech window SELF/WORLD, including pauses and trailing room noise.
    scored_frames = [f for f in frames if f.get("energy_db", -120.0) > -40.0]
    tp = sum(1 for f in scored_frames if f["final_verdict"] == "SELF" and f["gt_verdict"] == "SELF")
    fp = sum(1 for f in scored_frames if f["final_verdict"] == "SELF" and f["gt_verdict"] != "SELF")
    fn = sum(1 for f in scored_frames if f["final_verdict"] != "SELF" and f["gt_verdict"] == "SELF")
    tn = sum(1 for f in scored_frames if f["final_verdict"] != "SELF" and f["gt_verdict"] != "SELF")
    m4_accepted = sum(1 for f in frames if f["m4_outcome"] == "accepted")
    m4_attempted = sum(1 for f in frames if f["m4_attempted"])
    template_passed = sum(1 for f in frames if f["m4_template_match_passed"])
    self_frames = [f for f in scored_frames if f["gt_verdict"] == "SELF"]
    neg_frames = [f for f in scored_frames if f["gt_verdict"] == "WORLD"]
    return {
        "name": name,
        "total_frames": len(frames),
        "scored_frames": len(scored_frames),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        "m4_attempted": m4_attempted,
        "m4_accepted": m4_accepted,
        "template_match_passed": template_passed,
        "self_gt_frames": len(self_frames),
        "neg_gt_frames": len(neg_frames),
        "self_m4_accepted": sum(1 for f in self_frames if f["m4_outcome"] == "accepted"),
        "neg_m4_accepted": sum(1 for f in neg_frames if f["m4_outcome"] == "accepted"),
        "self_template_passed": sum(1 for f in self_frames if f["m4_template_match_passed"]),
        "neg_template_passed": sum(1 for f in neg_frames if f["m4_template_match_passed"]),
    }


def sweep_thresholds(
    *,
    replay_dir: str,
    thresholds: list[float],
    real_ecapa: bool = True,
    register_from_clean: Optional[str] = None,
    scenario_filter: Optional[str] = None,
) -> dict:
    """Sweep m4_template_match_threshold and report per-threshold results."""
    all_results = []
    for threshold in thresholds:
        result = analyze_replay(
            replay_dir=replay_dir,
            real_ecapa=real_ecapa,
            m4_template_match_threshold=threshold,
            register_from_clean=register_from_clean,
            scenario_filter=scenario_filter,
        )
        case_summaries = []
        for r in result["results"]:
            s = r["summary"]
            case_summaries.append({
                "case": s["name"],
                "f1": s["f1"],
                "precision": s["precision"],
                "recall": s["recall"],
                "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
                "m4_accepted": s["m4_accepted"],
                "template_passed": s["template_match_passed"],
                "self_m4_accepted": s["self_m4_accepted"],
                "neg_m4_accepted": s["neg_m4_accepted"],
                "self_template_passed": s["self_template_passed"],
                "neg_template_passed": s["neg_template_passed"],
            })
        all_results.append({
            "threshold": threshold,
            "cases": case_summaries,
        })
    return {"replay_dir": replay_dir, "thresholds": all_results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-frame M4 replay analyzer")
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--real-ecapa", action="store_true", default=True)
    parser.add_argument("--mock", action="store_true", help="Use mock embeddings instead of real ECAPA")
    parser.add_argument("--m4-self-threshold", type=float, default=0.10)
    parser.add_argument("--m4-template-match-threshold", type=float, default=0.20)
    parser.add_argument("--register-from-clean", default=None)
    parser.add_argument("--scenario-filter", default=None, help="Only analyze scenarios matching this substring")
    parser.add_argument("--threshold-sweep", nargs="+", type=float, default=None,
                        help="Sweep multiple m4_template_match_threshold values")
    args = parser.parse_args()

    real_ecapa = not args.mock

    if args.threshold_sweep:
        result = sweep_thresholds(
            replay_dir=args.replay_dir,
            thresholds=args.threshold_sweep,
            real_ecapa=real_ecapa,
            register_from_clean=args.register_from_clean,
            scenario_filter=args.scenario_filter,
        )
    else:
        result = analyze_replay(
            replay_dir=args.replay_dir,
            real_ecapa=real_ecapa,
            m4_self_threshold=args.m4_self_threshold,
            m4_template_match_threshold=args.m4_template_match_threshold,
            register_from_clean=args.register_from_clean,
            scenario_filter=args.scenario_filter,
        )

    output_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json)
    print(output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

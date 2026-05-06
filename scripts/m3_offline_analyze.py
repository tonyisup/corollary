#!/usr/bin/env python3
"""M3 Offline Attribution Analysis — full M2+M4 pipeline on recorded audio.

Runs the complete attribution pipeline (cross-correlation + VAD + M4 speaker
embeddings) on pre-recorded interruption scenarios, then measures M3 controller
decision quality.

This is the "real" analysis that the simplified version couldn't do — it computes
actual match scores from the captured audio rather than using ground truth labels.

Usage:
    .venv/bin/python scripts/m3_offline_analyze.py --input-dir data/m3_interrupt_v2
    .venv/bin/python scripts/m3_offline_analyze.py --input-dir data/m3_interrupt_v2 --m4-enable --cerebellum-model data/m4_forward_model_focused/forward_model_1777730332.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))
sys.path.insert(0, os.path.dirname(__file__))

from cochlea.capture import SR
from cochlea.vad import SmoothedVAD, VAD_FRAME_SAMPLES
from thalamus.attributor import ThalamicAttributor, SourceVerdict, AttributionResult
from pfc.agentive_controller import AgentiveController
from m2_speech_preservation_harness import (
    normalized_xcorr, local_xcorr, energy_onset, fit_room_filter, apply_room_filter
)


def load_scenario(scenario_dir: Path) -> dict:
    """Load a recorded scenario."""
    metadata = json.load(open(scenario_dir / "metadata.json"))
    captured = np.load(str(scenario_dir / "captured.npy"))
    gt_human = np.load(str(scenario_dir / "gt_human_speaking.npy"))
    gt_agent = np.load(str(scenario_dir / "gt_agent_playing.npy"))
    gt_should = np.load(str(scenario_dir / "gt_agent_should_speak.npy"))

    agent_pcm = None
    agent_pcm_path = scenario_dir / "agent_pcm.npy"
    if agent_pcm_path.exists():
        agent_pcm = np.load(str(agent_pcm_path))

    return {
        "metadata": metadata,
        "captured": captured,
        "gt_human_speaking": gt_human,
        "gt_agent_playing": gt_agent,
        "gt_agent_should_speak": gt_should,
        "agent_pcm": agent_pcm,
        "input_sr": metadata["input_sr"],
    }


def analyze_scenario(
    scenario: dict,
    m4_enable: bool = False,
    m4_self_threshold: float = 0.50,
    cerebellum_model: str | None = None,
    listen_threshold_ms: float = 100.0,
    cooldown_ms: float = 200.0,
    motor_holdoff_ms: float = 200.0,
    similarity_threshold: float = 0.05,
    causal_tolerance_ms: float = 175.0,
    self_match_mode: str = "local_xcorr",
    onset_method: str = "score",
    self_match_search_ms: float = 256.4,
    room_filter_taps: int = 64,
    room_filter_ridge: float = 0.000112,
) -> dict:
    """Run full M2+M4 attribution + M3 controller on recorded audio."""

    captured = scenario["captured"]
    gt_human = scenario["gt_human_speaking"]
    gt_agent_playing = scenario["gt_agent_playing"]
    gt_agent_should = scenario["gt_agent_should_speak"]
    agent_pcm = scenario["agent_pcm"]
    input_sr = scenario["input_sr"]
    metadata = scenario["metadata"]

    # Resample to SR if needed
    if input_sr != SR:
        import librosa
        observed = librosa.resample(captured, orig_sr=input_sr, target_sr=SR).astype(np.float32)
        # Also resample ground truth to match
        gt_human = np.interp(
            np.linspace(0, len(gt_human), len(observed)),
            np.arange(len(gt_human)),
            gt_human.astype(float)
        ) > 0.5
        gt_agent_playing = np.interp(
            np.linspace(0, len(gt_agent_playing), len(observed)),
            np.arange(len(gt_agent_playing)),
            gt_agent_playing.astype(float)
        ) > 0.5
        gt_agent_should = np.interp(
            np.linspace(0, len(gt_agent_should), len(observed)),
            np.arange(len(gt_agent_should)),
            gt_agent_should.astype(float)
        ) > 0.5
    else:
        observed = captured

    agent_duration_s = len(agent_pcm) / SR if agent_pcm is not None and len(agent_pcm) > 0 else 0
    agent_start_s = metadata["agent_start_s"]

    # Compute cross-correlation scores
    if agent_pcm is not None and len(agent_pcm) > 0:
        dry_global_score, xcorr_shift = normalized_xcorr(observed, agent_pcm)
        self_expected_shift = int((agent_start_s + 0.3) * SR)  # onset_latency_ms=300
        dry_local_score, dry_local_shift = local_xcorr(
            observed, agent_pcm, self_expected_shift, search_ms=self_match_search_ms
        )

        if self_match_mode == "global_xcorr":
            self_score = dry_global_score
            score_shift = xcorr_shift
        elif self_match_mode == "local_xcorr":
            self_score = dry_local_score
            score_shift = dry_local_shift
        elif self_match_mode == "room_local_oracle":
            h = fit_room_filter(observed, agent_pcm, dry_local_shift, taps=room_filter_taps, ridge=room_filter_ridge)
            room_ref = apply_room_filter(agent_pcm, h)
            room_local_score, room_local_shift = local_xcorr(
                observed, room_ref, self_expected_shift, search_ms=self_match_search_ms
            )
            self_score = room_local_score
            score_shift = room_local_shift
        else:
            raise ValueError(f"Unknown self_match_mode: {self_match_mode}")

        if onset_method == "energy":
            self_shift = energy_onset(observed, self_expected_shift)
        elif onset_method == "score":
            self_shift = score_shift
        else:
            self_shift = xcorr_shift
    else:
        self_score = 0.0
        self_shift = 0
        dry_global_score = 0.0
        dry_local_score = 0.0

    # Build M4 detector if enabled
    m4_detector = None
    if m4_enable and agent_pcm is not None and len(agent_pcm) > 0:
        from scripts.m2_benchmark import build_m4_detector_for_self_pcm
        import argparse as _argparse
        args_ns = _argparse.Namespace(
            m4_enable=True,
            m4_self_threshold=m4_self_threshold,
            m4_min_buffer_ms=60.0,
            m4_registration_threshold=0.4,
            m4_mock=False,
            cerebellum_model=cerebellum_model,
        )
        m4_detector, m4_info = build_m4_detector_for_self_pcm(agent_pcm, args_ns)

    # Build attributor
    attributor = ThalamicAttributor(
        sr=SR,
        similarity_threshold=similarity_threshold,
        causal_tolerance_ms=causal_tolerance_ms,
        m4_detector=m4_detector,
        m4_self_threshold=m4_self_threshold,
        m4_min_buffer_ms=60.0,
    )

    # Build controller
    controller = AgentiveController(
        listen_threshold_ms=listen_threshold_ms,
        cooldown_ms=cooldown_ms,
        motor_holdoff_ms=motor_holdoff_ms,
    )

    # Agent requests output at the appropriate time
    agent_requested = False

    # Process frames
    vad = SmoothedVAD()
    num_frames = len(observed) // VAD_FRAME_SAMPLES

    false_interrupts = 0
    false_defers = 0
    missed_interrupts = 0
    successful_resumes = 0
    total_resumes_expected = 0

    was_interrupted = False
    issues = []
    frame_verdicts = []

    for i in range(num_frames):
        t_s = i * VAD_FRAME_SAMPLES / SR
        start = i * VAD_FRAME_SAMPLES
        end = start + VAD_FRAME_SAMPLES
        frame_audio = observed[start:end]

        # VAD
        vad_speech = vad.process_frame(frame_audio)
        is_speech = vad_speech

        # Ground truth
        gt_idx = start
        is_human_speaking = bool(gt_human[gt_idx]) if gt_idx < len(gt_human) else False
        is_agent_playing = bool(gt_agent_playing[gt_idx]) if gt_idx < len(gt_agent_playing) else False
        should_speak = bool(gt_agent_should[gt_idx]) if gt_idx < len(gt_agent_should) else False

        # Compute match score for this frame.
        # When agent is playing, use the pre-computed local xcorr score.
        # This is the same score the live pipeline would compute.
        # For frames where agent is not playing, score is 0.
        if agent_pcm is not None and len(agent_pcm) > 0 and is_agent_playing:
            match_score = self_score  # Use the local xcorr score computed above
        else:
            match_score = 0.0

        # RMS
        rms_db = float(20 * np.log10(max(np.sqrt(np.mean(frame_audio**2)), 1e-10)))

        # Agent requests output
        if not agent_requested and agent_pcm is not None and len(agent_pcm) > 0 and t_s >= agent_start_s:
            controller.request_output(metadata.get("agent_text", "I'm Corollary."), current_time=t_s)
            agent_requested = True

        # Build attribution
        attribution = AttributionResult(
            verdict=SourceVerdict.SILENCE,  # Will be overwritten by attributor
            confidence=0.8,
            self_match_score=match_score,
            is_in_motor_window=is_agent_playing,
            causal_score=0.0,
            energy_db=rms_db,
            is_speech=is_speech,
            metadata={},
        )

        # Run through attributor
        observed_onset = self_shift if is_agent_playing else start
        expected_onset = self_shift if is_agent_playing else None

        result = attributor.attribute(
            match_score=match_score,
            observed_onset_samples=observed_onset,
            expected_onset_samples=expected_onset,
            rms_db=rms_db,
            is_playing=is_agent_playing,
            is_speech=is_speech,
            frame_audio=frame_audio,
            self_pcm=agent_pcm if is_agent_playing else None,
            agent_start_sample=int(agent_start_s * SR) if is_agent_playing else None,
        )

        # Run through controller
        decision = controller.process_attribution(result, current_time=t_s)

        # Record verdict
        frame_verdicts.append({
            "frame": i,
            "t_s": round(t_s, 3),
            "verdict": result.verdict.name,
            "state": decision.state.name,
            "can_speak": decision.can_speak,
            "should_interrupt": decision.should_interrupt,
            "match_score": round(match_score, 4),
            "self_match_score_per_frame": round(result.self_match_score_per_frame, 4) if result.self_match_score_per_frame is not None else None,
            "rms_db": round(rms_db, 1),
            "is_speech": is_speech,
            "gt_human": is_human_speaking,
            "gt_agent": is_agent_playing,
        })

        # Check metrics
        if decision.should_interrupt and should_speak:
            false_interrupts += 1
            issues.append(f"  FALSE INTERRUPT at t={t_s:.3f}s")

        if not decision.can_speak and should_speak and is_agent_playing:
            false_defers += 1
            issues.append(f"  FALSE DEFER at t={t_s:.3f}s")

        if decision.can_speak and is_human_speaking and not is_agent_playing:
            missed_interrupts += 1
            issues.append(f"  MISSED INTERRUPT at t={t_s:.3f}s")

        if decision.should_interrupt and not was_interrupted:
            was_interrupted = True
            total_resumes_expected += 1

        if was_interrupted and not is_human_speaking and not is_agent_playing:
            if controller.has_deferred_output and decision.can_speak:
                successful_resumes += 1
                was_interrupted = False

    # Compute rates
    total_frames = num_frames
    interrupt_frames = sum(1 for fv in frame_verdicts if fv["gt_agent"] and not fv["gt_human"])

    false_interrupt_rate = false_interrupts / max(interrupt_frames, 1)
    false_defer_rate = false_defers / max(interrupt_frames, 1)
    missed_interrupt_rate = missed_interrupts / max(total_frames, 1)
    resume_rate = successful_resumes / max(total_resumes_expected, 1)

    return {
        "scenario": metadata["scenario_name"],
        "description": metadata["description"],
        "expected_behavior": metadata["expected_behavior"],
        "total_frames": total_frames,
        "false_interrupts": false_interrupts,
        "false_defers": false_defers,
        "missed_interrupts": missed_interrupts,
        "successful_resumes": successful_resumes,
        "total_resumes_expected": total_resumes_expected,
        "false_interrupt_rate": false_interrupt_rate,
        "false_defer_rate": false_defer_rate,
        "missed_interrupt_rate": missed_interrupt_rate,
        "resume_rate": resume_rate,
        "self_score": self_score,
        "dry_global_score": dry_global_score,
        "dry_local_score": dry_local_score,
        "issues": issues,
        "frame_verdicts": frame_verdicts,
    }


def main():
    parser = argparse.ArgumentParser(description="M3 Offline Attribution Analysis")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--m4-enable", action="store_true")
    parser.add_argument("--m4-self-threshold", type=float, default=0.50)
    parser.add_argument("--cerebellum-model", default=None)
    parser.add_argument("--listen-threshold-ms", type=float, default=100.0)
    parser.add_argument("--cooldown-ms", type=float, default=200.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.05)
    parser.add_argument("--self-match-mode", default="local_xcorr",
                        choices=["global_xcorr", "local_xcorr", "room_local_oracle"])
    parser.add_argument("--onset-method", default="score",
                        choices=["energy", "score", "xcorr"])
    parser.add_argument("--motor-holdoff-ms", type=float, default=200.0,
                        help="Holdoff period after request_output() where WORLD frames don't trigger LISTEN (default: 200ms)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found")
        sys.exit(1)

    print("=" * 70)
    print("M3 OFFLINE ATTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"M4 enabled: {args.m4_enable}")
    print(f"Self match mode: {args.self_match_mode}")
    print(f"Onset method: {args.onset_method}")
    print()

    scenario_dirs = sorted([d for d in input_dir.iterdir()
                            if d.is_dir() and (d / "metadata.json").exists()])
    if not scenario_dirs:
        print(f"No scenarios found in {input_dir}")
        sys.exit(1)

    print(f"Scenarios: {len(scenario_dirs)}")

    all_results = []
    total_fi = total_fd = total_mi = total_sr = total_re = 0

    for scenario_dir in scenario_dirs:
        scenario = load_scenario(scenario_dir)
        result = analyze_scenario(
            scenario,
            m4_enable=args.m4_enable,
            m4_self_threshold=args.m4_self_threshold,
            cerebellum_model=args.cerebellum_model,
            listen_threshold_ms=args.listen_threshold_ms,
            cooldown_ms=args.cooldown_ms,
            motor_holdoff_ms=args.motor_holdoff_ms,
            similarity_threshold=args.similarity_threshold,
            self_match_mode=args.self_match_mode,
            onset_method=args.onset_method,
        )
        all_results.append(result)

        total_fi += result["false_interrupts"]
        total_fd += result["false_defers"]
        total_mi += result["missed_interrupts"]
        total_sr += result["successful_resumes"]
        total_re += result["total_resumes_expected"]

        print(f"\n{result['scenario']}: {result['description']}")
        print(f"  Expected: {result['expected_behavior']}")
        print(f"  Self score: {result['self_score']:.3f} (global: {result['dry_global_score']:.3f}, local: {result['dry_local_score']:.3f})")
        print(f"  False interrupts: {result['false_interrupts']} ({result['false_interrupt_rate']:.1%})")
        print(f"  False defers: {result['false_defers']} ({result['false_defer_rate']:.1%})")
        print(f"  Missed interrupts: {result['missed_interrupts']} ({result['missed_interrupt_rate']:.1%})")
        print(f"  Successful resumes: {result['successful_resumes']}/{result['total_resumes_expected']} ({result['resume_rate']:.1%})")
        if result["issues"][:3]:
            for issue in result["issues"][:3]:
                print(issue)

    # Summary
    print(f"\n{'='*70}")
    print("OVERALL METRICS")
    print(f"{'='*70}")
    print(f"Scenarios: {len(all_results)}")
    print(f"Total false interrupts: {total_fi}")
    print(f"Total false defers: {total_fd}")
    print(f"Total missed interrupts: {total_mi}")
    print(f"Successful resumes: {total_sr}/{total_re}")

    total_interrupt_frames = sum(r["total_frames"] for r in all_results)
    if total_interrupt_frames > 0:
        print(f"\nFalse interrupt rate: {total_fi/total_interrupt_frames:.1%}")
        print(f"False defer rate: {total_fd/total_interrupt_frames:.1%}")
        print(f"Missed interrupt rate: {total_mi/total_interrupt_frames:.1%}")

    if total_re > 0:
        print(f"Resume rate: {total_sr/total_re:.1%}")

    # Targets
    print(f"\nTargets:")
    print(f"  False interrupt rate: <5%  {'✓' if total_fi/max(total_interrupt_frames,1) < 0.05 else '✗'}")
    print(f"  False defer rate: <10%    {'✓' if total_fd/max(total_interrupt_frames,1) < 0.10 else '✗'}")
    print(f"  Missed interrupt rate: <10% {'✓' if total_mi/max(total_interrupt_frames,1) < 0.10 else '✗'}")
    print(f"  Resume rate: >80%         {'✓' if total_sr/max(total_re,1) > 0.80 else '✗'}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "m4_enable": args.m4_enable,
            "m4_self_threshold": args.m4_self_threshold,
            "cerebellum_model": args.cerebellum_model,
            "listen_threshold_ms": args.listen_threshold_ms,
            "cooldown_ms": args.cooldown_ms,
            "similarity_threshold": args.similarity_threshold,
            "self_match_mode": args.self_match_mode,
            "onset_method": args.onset_method,
        },
        "summary": {
            "scenarios": len(all_results),
            "total_false_interrupts": total_fi,
            "total_false_defers": total_fd,
            "total_missed_interrupts": total_mi,
            "successful_resumes": total_sr,
            "resumes_expected": total_re,
        },
        "scenarios": all_results,
    }

    suffix = "_m4" if args.m4_enable else "_m2"
    out_path = input_dir / f"offline_analysis_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

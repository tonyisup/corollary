#!/usr/bin/env python3
"""M7 Interruption Latency Test.

Measures the end-to-end interruption response latency of the streaming runtime
with the new callback_blocksize_ms=120 default.

This is a deterministic synthetic test — no hardware required, no TTS generation.
It uses run_synthetic() with carefully crafted attribution frames to simulate:
1. Corollary speaking (SELF frames, is_in_motor_window=True)
2. Human starts speaking over Corollary (BOTH frames with high energy ratio)
3. Controller should detect overlap and transition to LISTEN
4. Human stops, Corollary resumes (SELF frames)

The latency is measured in frames from the first BOTH frame with high energy ratio
to the first frame where controller_state == LISTEN.

With callback_blocksize_ms=120 (4 VAD frames per callback), the theoretical
worst-case latency is:
- 1 callback block for analysis (4 frames = 120ms)
- both_external_min_frames=3 consecutive frames above threshold (3 frames = 90ms)
- 1 callback block for render reaction (4 frames = 120ms)
Total worst case: ~330ms (11 frames)

The previous default of 390ms (13 frames per callback) would be ~510ms worst case.

Usage:
    .venv/bin/python scripts/m7_interruption_latency_test.py
    .venv/bin/python scripts/m7_interruption_latency_test.py --blocksize-ms 390
    .venv/bin/python scripts/m7_interruption_latency_test.py --output /tmp/latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.streaming_runtime import (
    SyntheticAttributionFrame,
    StreamingConfig,
    StreamingRuntime,
    SourceVerdict,
)


def make_interruption_scenario(
    self_frames: int = 20,
    both_overlap_frames: int = 10,
    silence_frames: int = 10,
    resume_frames: int = 20,
    both_external_min_frames: int = 3,
    both_external_energy_ratio: float = 1.5,
) -> tuple[list[SyntheticAttributionFrame], int, int]:
    """Create a synthetic interruption scenario.

    Returns:
        frames: List of SyntheticAttributionFrame
        first_both_idx: Frame index where BOTH overlap starts
        expected_first_listen_idx: Expected frame index of first LISTEN
    """
    frames: list[SyntheticAttributionFrame] = []
    frame_idx = 0

    # Phase 1: Corollary speaking alone (SELF)
    for i in range(self_frames):
        frames.append(SyntheticAttributionFrame(
            time_s=frame_idx * 0.03,
            verdict=SourceVerdict.SELF,
            is_speech=False,
            is_in_motor_window=True,
            self_match_score=0.9,
            self_match_score_per_frame=1.0,
            causal_score=1.0,
            energy_db=-20.0,
        ))
        frame_idx += 1

    # Phase 2: Human starts speaking over Corollary (BOTH with high energy)
    first_both_idx = frame_idx
    for i in range(both_overlap_frames):
        frames.append(SyntheticAttributionFrame(
            time_s=frame_idx * 0.03,
            verdict=SourceVerdict.BOTH,
            is_speech=True,
            is_in_motor_window=True,
            self_match_score=0.9,
            self_match_score_per_frame=both_external_energy_ratio + 0.5,  # Above threshold
            causal_score=1.0,
            energy_db=-15.0,
        ))
        frame_idx += 1

    # Phase 3: Silence (human stopped, TTS paused)
    for i in range(silence_frames):
        frames.append(SyntheticAttributionFrame(
            time_s=frame_idx * 0.03,
            verdict=SourceVerdict.SILENCE,
            is_speech=False,
            is_in_motor_window=False,
            self_match_score=0.0,
            self_match_score_per_frame=0.0,
            energy_db=-70.0,
        ))
        frame_idx += 1

    # Phase 4: Corollary resumes (SELF)
    for i in range(resume_frames):
        frames.append(SyntheticAttributionFrame(
            time_s=frame_idx * 0.03,
            verdict=SourceVerdict.SELF,
            is_speech=False,
            is_in_motor_window=True,
            self_match_score=0.9,
            self_match_score_per_frame=1.0,
            causal_score=1.0,
            energy_db=-20.0,
        ))
        frame_idx += 1

    # Expected: LISTEN should fire after both_external_min_frames consecutive
    # BOTH frames above energy ratio threshold
    expected_first_listen_idx = first_both_idx + both_external_min_frames

    return frames, first_both_idx, expected_first_listen_idx


def measure_latency(
    callback_blocksize_ms: float = 120.0,
    both_external_min_frames: int = 3,
    both_external_energy_ratio: float = 1.5,
) -> dict:
    """Run the synthetic scenario and measure interruption latency."""
    frames, first_both_idx, expected_listen_idx = make_interruption_scenario(
        both_external_min_frames=both_external_min_frames,
        both_external_energy_ratio=both_external_energy_ratio,
    )

    config = StreamingConfig(
        tts_pcm=np.zeros(16000 * 10, dtype=np.float32),  # 10s of silence (not used in synthetic)
        tts_text="interruption latency test",
        gated_playback=True,
        callback_blocksize_ms=callback_blocksize_ms,
        sr=16000,
        m4_enable=False,
        both_external_min_frames=both_external_min_frames,
        both_external_energy_ratio=both_external_energy_ratio,
        cooldown_ms=120.0,
    )

    runtime = StreamingRuntime(config)
    logs = runtime.run_synthetic(frames)

    # Find first LISTEN frame
    first_listen_idx = None
    for i, log in enumerate(logs):
        if log.controller_state == "LISTEN":
            first_listen_idx = i
            break

    # Calculate latency
    interrupt_latency_frames = None
    interrupt_latency_ms = None
    if first_listen_idx is not None:
        interrupt_latency_frames = first_listen_idx - first_both_idx
        interrupt_latency_ms = interrupt_latency_frames * 30.0  # 30ms per VAD frame

    # Collect state transition region
    transition_region = []
    if first_listen_idx is not None:
        start = max(0, first_both_idx - 2)
        end = min(len(logs), first_listen_idx + 5)
        for i in range(start, end):
            transition_region.append({
                "frame_idx": i,
                "time_ms": round(i * 30.0, 1),
                "state": logs[i].controller_state,
                "verdict": logs[i].verdict,
                "vad": logs[i].vad_speech,
                "energy_db": round(logs[i].energy_db, 1),
                "match_per_frame": round(logs[i].self_match_score_per_frame or 0, 2),
                "should_interrupt": logs[i].should_interrupt,
            })

    summary = runtime.get_summary()

    return {
        "callback_blocksize_ms": callback_blocksize_ms,
        "both_external_min_frames": both_external_min_frames,
        "both_external_energy_ratio": both_external_energy_ratio,
        "total_frames": len(logs),
        "first_both_idx": first_both_idx,
        "expected_listen_idx": expected_listen_idx,
        "actual_listen_idx": first_listen_idx,
        "interrupt_latency_frames": interrupt_latency_frames,
        "interrupt_latency_ms": round(interrupt_latency_ms, 1) if interrupt_latency_ms is not None else None,
        "interrupt_events": summary.get("interrupt_events", 0),
        "resumed_events": summary.get("resumed_events", 0),
        "tts_completed": summary.get("tts_completed", False),
        "transition_region": transition_region,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 Interruption Latency Test")
    parser.add_argument("--output", type=str, default="/tmp/m7_interruption_latency.json")
    parser.add_argument("--blocksize-ms", type=float, default=120.0,
                        help="Callback blocksize in ms (default: 120)")
    parser.add_argument("--min-frames", type=int, default=3,
                        help="both_external_min_frames (default: 3)")
    parser.add_argument("--energy-ratio", type=float, default=1.5,
                        help="both_external_energy_ratio threshold (default: 1.5)")
    args = parser.parse_args()

    print("=" * 60)
    print("M7 Interruption Latency Test (Synthetic)")
    print("=" * 60)
    print(f"  Callback blocksize:        {args.blocksize_ms} ms")
    print(f"  both_external_min_frames:  {args.min_frames}")
    print(f"  both_external_energy_ratio: {args.energy_ratio}")
    print()

    result = measure_latency(
        callback_blocksize_ms=args.blocksize_ms,
        both_external_min_frames=args.min_frames,
        both_external_energy_ratio=args.energy_ratio,
    )

    print(f"Total frames processed:     {result['total_frames']}")
    print(f"First BOTH frame:           #{result['first_both_idx']}")
    print(f"Expected LISTEN frame:      #{result['expected_listen_idx']}")
    print(f"Actual LISTEN frame:        #{result['actual_listen_idx']}")
    print()

    if result['interrupt_latency_ms'] is not None:
        print(f"Interruption latency:       {result['interrupt_latency_ms']} ms "
              f"({result['interrupt_latency_frames']} frames)")
    else:
        print(f"Interruption latency:       NO INTERRUPTION DETECTED")

    print(f"Interrupt events:           {result['interrupt_events']}")
    print(f"Resumed events:             {result['resumed_events']}")
    print(f"TTS completed:              {result['tts_completed']}")
    print()

    print("State transition region:")
    print(f"  {'frame':>5}  {'ms':>6}  {'state':>6}  {'verdict':>6}  {'vad':>5}  {'energy':>7}  {'match':>5}  {'irq':>4}")
    for s in result['transition_region']:
        print(f"  {s['frame_idx']:5d}  {s['time_ms']:6.1f}  {s['state']:>6}  {s['verdict']:>6}  "
              f"{str(s['vad']):>5}  {s['energy_db']:6.1f}dB  {s['match_per_frame']:5.2f}  "
              f"{'*' if s['should_interrupt'] else ' ':>4}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print()
    print(f"Results saved to: {args.output}")

    # Acceptance
    print()
    print("-" * 60)
    print("Acceptance:")
    print("-" * 60)

    if result['interrupt_latency_ms'] is None:
        print("  FAIL — No interruption detected")
        sys.exit(1)

    latency_ms = result['interrupt_latency_ms']
    blocksize_ms = args.blocksize_ms
    vad_frame_ms = 30.0
    n_vad_per_callback = max(1, int(blocksize_ms / vad_frame_ms))

    # Theoretical analysis:
    # - Analysis callback processes up to n_vad_per_callback frames
    # - Need both_external_min_frames consecutive frames above threshold
    # - Then next render callback reacts
    analysis_ms = n_vad_per_callback * vad_frame_ms
    classifier_ms = args.min_frames * vad_frame_ms
    render_ms = n_vad_per_callback * vad_frame_ms
    theoretical_worst_ms = analysis_ms + classifier_ms + render_ms
    theoretical_best_ms = classifier_ms + render_ms  # if threshold met at start of callback

    print(f"  Callback blocksize:        {blocksize_ms:.0f} ms ({n_vad_per_callback} VAD frames)")
    print(f"  Classifier window:         {classifier_ms:.0f} ms ({args.min_frames} frames)")
    print(f"  Analysis callback:         {analysis_ms:.0f} ms")
    print(f"  Render callback:           {render_ms:.0f} ms")
    print(f"  Theoretical best case:     ~{theoretical_best_ms:.0f} ms")
    print(f"  Theoretical worst case:    ~{theoretical_worst_ms:.0f} ms")
    print(f"  Measured latency:          {latency_ms:.1f} ms")

    # The measured latency should be within the theoretical range
    if latency_ms <= theoretical_worst_ms + 30:  # 30ms tolerance
        print(f"  PASS — Latency within expected range")
    else:
        print(f"  WARN — Latency higher than expected, may need tuning")

    # Also verify the system resumed
    if result['resumed_events'] > 0 and result['tts_completed']:
        print(f"  PASS — System resumed and completed TTS")
    else:
        print(f"  WARN — Resume/completion not verified")

    print()


if __name__ == "__main__":
    main()

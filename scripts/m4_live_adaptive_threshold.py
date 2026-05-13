#!/usr/bin/env python3
"""M4 Live Adaptive Threshold Validation.

Runs the streaming runtime on live hardware with M4 + per-speaker adaptive
threshold enabled. Plays a TTS phrase through the speaker, captures it through
the BL-2021USB mic, and verifies that:
  1. Self-voice detection (M4) fires on the agent's own speech
  2. Speaker identification converges on the captured audio
  3. The adaptive threshold adapter updates per-speaker statistics
  4. Frame logs contain valid speaker_id and m4_outcome fields

Usage:
    .venv/bin/python scripts/m4_live_adaptive_threshold.py \\
        --input-device 1 \\
        --output-device 5 \\
        --duration 8.0 \\
        --text "Hello, I am Corollary. This is a live self-voice detection test." \\
        --output /tmp/m4_live_adaptive.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from pfc.streaming_runtime import StreamingRuntime, StreamingConfig
from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter
from broca.tts import synthesize


def main():
    parser = argparse.ArgumentParser(description="M4 live adaptive threshold validation")
    parser.add_argument("--input-device", type=int, default=1, help="Mic input device index")
    parser.add_argument("--output-device", type=int, default=5, help="Speaker output device index")
    parser.add_argument("--duration", type=float, default=8.0, help="Capture duration in seconds")
    parser.add_argument("--text", type=str, default="Hello, I am Corollary. This is a live self-voice detection test.")
    parser.add_argument("--m4-threshold", type=float, default=0.10, help="Base M4 self-voice threshold")
    parser.add_argument("--m4-buffer-ms", type=float, default=300.0, help="M4 min buffer ms")
    parser.add_argument("--output", type=str, default="/tmp/m4_live_adaptive.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 60)
    print("M4 Live Adaptive Threshold Validation")
    print("=" * 60)
    print(f"  Input device:  {args.input_device} (BL-2021USB)")
    print(f"  Output device: {args.output_device} (Mac mini Speakers)")
    print(f"  Duration:      {args.duration}s")
    print(f"  M4 threshold:  {args.m4_threshold}")
    print(f"  M4 buffer:     {args.m4_buffer_ms}ms")

    # Synthesize TTS
    print("\nSynthesizing TTS...")
    import asyncio
    tts_pcm = asyncio.run(synthesize(args.text)).astype(np.float32)
    tts_duration = len(tts_pcm) / 16000
    print(f"  TTS duration: {tts_duration:.2f}s ({len(tts_pcm)} samples)")

    # Create adaptive threshold adapter
    adapter = SpeakerThresholdAdapter(
        base_threshold=args.m4_threshold,
        margin=0.05,
        adaptation_rate=0.1,
    )
    adapter.register_self("self")

    # Configure streaming runtime
    config = StreamingConfig(
        tts_pcm=tts_pcm,
        tts_text=args.text,
        input_device=args.input_device,
        output_device=args.output_device,
        m4_enable=True,
        gated_playback=False,  # Always play TTS so we can hear it
        m4_self_threshold=args.m4_threshold,
        m4_min_buffer_ms=args.m4_buffer_ms,
        speaker_threshold_adapter=adapter,
        save_path=Path(args.output).with_suffix(".jsonl"),
    )

    runtime = StreamingRuntime(config)

    print(f"\nStarting live streaming ({args.duration}s)...")
    print("  Speak now if you want to test human-voice rejection.")
    print()

    try:
        logs = runtime.run(duration_s=args.duration)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Analyze results
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")

    total_frames = len(logs)
    m4_attempted = sum(1 for l in logs if l.m4_outcome not in ("not_attempted", None))
    m4_accepted = sum(1 for l in logs if l.m4_outcome == "accepted")
    m4_below = sum(1 for l in logs if l.m4_outcome == "below_threshold")
    m4_buffer_short = sum(1 for l in logs if l.m4_outcome == "buffer_too_short")

    speaker_ids = set(l.speaker_id for l in logs if l.speaker_id is not None)
    verdict_counts = {}
    for l in logs:
        verdict_counts[l.verdict] = verdict_counts.get(l.verdict, 0) + 1

    m4_confidences = [l.m4_confidence for l in logs if l.m4_confidence is not None]
    avg_conf = np.mean(m4_confidences) if m4_confidences else 0.0
    max_conf = np.max(m4_confidences) if m4_confidences else 0.0

    print(f"  Total frames:       {total_frames}")
    print(f"  Verdict counts:     {verdict_counts}")
    print(f"  M4 attempted:       {m4_attempted}")
    print(f"  M4 accepted:        {m4_accepted}")
    print(f"  M4 below threshold: {m4_below}")
    print(f"  M4 buffer too short:{m4_buffer_short}")
    print(f"  Avg M4 confidence:  {avg_conf:.4f}")
    print(f"  Max M4 confidence:  {max_conf:.4f}")
    print(f"  Speaker IDs seen:   {speaker_ids if speaker_ids else '(none)'}")

    # Adapter state
    adapter_summary = adapter.summary()
    print(f"\n  Adapter state:")
    for sid, info in adapter_summary.items():
        print(f"    {sid}: count={info['count']}, mean_conf={info.get('mean_conf', 'N/A')}, "
              f"max_conf={info.get('max_conf', 'N/A')}, threshold={info.get('threshold', 'N/A')}")

    # Pass/fail criteria
    print(f"\n{'=' * 60}")
    print("Acceptance Criteria")
    print(f"{'=' * 60}")

    criteria = []

    # C1: M4 fires on self speech
    c1 = m4_attempted > 0
    criteria.append(("M4 attempted on live audio", c1, f"{m4_attempted} frames"))

    # C2: M4 accepts some frames as self
    c2 = m4_accepted > 0
    criteria.append(("M4 accepted self frames", c2, f"{m4_accepted} frames"))

    # C3: Speaker ID is non-None in some frames
    c3 = len(speaker_ids) > 0
    criteria.append(("Speaker identification active", c3, f"IDs: {speaker_ids}"))

    # C4: Adapter has data
    c4 = any(info['count'] > 0 for info in adapter_summary.values())
    criteria.append(("Adapter received updates", c4, f"profiles: {list(adapter_summary.keys())}"))

    all_pass = True
    for name, passed, detail in criteria:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {detail}")

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'=' * 60}")

    # Save verdict
    verdict = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "input_device": args.input_device,
        "output_device": args.output_device,
        "duration_s": args.duration,
        "total_frames": total_frames,
        "m4_attempted": m4_attempted,
        "m4_accepted": m4_accepted,
        "m4_below_threshold": m4_below,
        "m4_buffer_too_short": m4_buffer_short,
        "avg_m4_confidence": round(float(avg_conf), 4),
        "max_m4_confidence": round(float(max_conf), 4),
        "speaker_ids": list(speaker_ids),
        "verdict_counts": verdict_counts,
        "adapter_summary": adapter_summary,
        "criteria": {name: passed for name, passed, _ in criteria},
        "pass": all_pass,
    }

    with open(args.output, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nVerdict saved to: {args.output}")
    print(f"Frame log saved to: {args.output.replace('.json', '.jsonl')}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

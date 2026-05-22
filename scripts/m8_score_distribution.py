#!/usr/bin/env python3
"""M8 Score Distribution Logger.

Runs a longer live session and logs per-frame M8 scores to a JSONL file
for offline threshold calibration.

Usage:
    source .venv/bin/activate
    python scripts/m8_score_distribution.py \
        --input-device 1 \
        --self-output-device 3 \
        --human-output-device 2 \
        --duration-s 30 \
        --output /tmp/m8_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize as _synthesize_tts
from pfc.streaming_runtime import StreamingConfig, StreamingRuntime


def make_tts_pcm(text: str, voice: str = "en-US-AriaNeural", sr: int = 16000) -> np.ndarray:
    import asyncio
    pcm = asyncio.run(_synthesize_tts(text, voice=voice))
    return pcm.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="M8 Score Distribution Logger")
    parser.add_argument("--input-device", type=int, required=True)
    parser.add_argument("--self-output-device", type=int, required=True)
    parser.add_argument("--human-output-device", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--m4-mode", choices=["auto", "bank", "scorer", "arbitration"], default="arbitration")
    parser.add_argument("--m4-scorer", type=str,
                        default="data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt")
    parser.add_argument("--m4-bank", type=str,
                        default="data/m4_prototype_bank_round4_20260520/prototype_bank.npz")
    parser.add_argument("--room-filter", type=str,
                        default="data/m3_broca_room_filter_production.npy")
    parser.add_argument("--output", type=str, default="/tmp/m8_scores.jsonl")
    args = parser.parse_args()

    sr = 16000
    ROOT = Path(__file__).resolve().parents[1]

    self_text = "This is a longer M8 score distribution test. I am speaking continuously to gather salience data across different acoustic conditions."
    self_pcm = make_tts_pcm(self_text, voice="en-US-AriaNeural", sr=sr)

    config = StreamingConfig(
        tts_pcm=self_pcm,
        tts_text="m8 score distribution",
        input_device=args.input_device,
        output_device=args.self_output_device,
        sr=sr,
        callback_blocksize_ms=120.0,
        gated_playback=True,
        m4_enable=True,
        m4_mode=args.m4_mode,
        m4_scorer_path=str(ROOT / args.m4_scorer),
        m4_bank_path=str(ROOT / args.m4_bank),
        m4_self_threshold=0.99,
        room_filter_path=str(ROOT / args.room_filter),
        room_local_threshold=0.0747,
        energy_threshold_db=-55.0,
        similarity_threshold=0.05,
        causal_tolerance_ms=175.0,
        both_external_min_frames=3,
        both_external_energy_ratio=1.5,
        cooldown_ms=200.0,
        m8_enable=True,
        save_path=Path(args.output),
    )

    print("=" * 60)
    print("M8 Score Distribution Logger")
    print("=" * 60)
    print(f"  Duration:    {args.duration_s}s")
    print(f"  M4 mode:     {args.m4_mode}")
    print(f"  Output:      {args.output}")
    print()

    runtime = StreamingRuntime(config)

    print(f"Running for {args.duration_s}s...")
    runtime.run(duration_s=args.duration_s)

    logs = runtime._logs
    print(f"  Collected {len(logs)} frames")

    # Compute score distribution
    scores = [log.m8_score for log in logs if log.m8_score is not None]
    actions = {}
    for log in logs:
        a = log.m8_action or "NONE"
        actions[a] = actions.get(a, 0) + 1

    if scores:
        import statistics
        print(f"\nScore distribution:")
        print(f"  min:    {min(scores):.4f}")
        print(f"  max:    {max(scores):.4f}")
        print(f"  mean:   {statistics.mean(scores):.4f}")
        print(f"  median: {statistics.median(scores):.4f}")
        print(f"  stdev:  {statistics.stdev(scores):.4f}" if len(scores) > 1 else "  stdev:  N/A")

        # Percentiles
        sorted_scores = sorted(scores)
        for p in [25, 50, 75, 90, 95, 99]:
            idx = int(len(sorted_scores) * p / 100)
            idx = min(idx, len(sorted_scores) - 1)
            print(f"  p{p:2d}:    {sorted_scores[idx]:.4f}")

    print(f"\nAction distribution:")
    for action, count in sorted(actions.items(), key=lambda x: -x[1]):
        pct = count / len(logs) * 100
        print(f"  {action:>10s}: {count:5d} ({pct:5.1f}%)")

    print(f"\nScores logged to: {args.output}")


if __name__ == "__main__":
    main()

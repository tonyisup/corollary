#!/usr/bin/env python3
"""M7 Live Interruption Latency Test.

Measures end-to-end interruption response latency on live hardware.

Protocol:
1. Corollary speaks a long TTS utterance through the speaker
2. At a known time (~2s), a "human" voice (different TTS voice) is played
   through the SAME speaker, simulating a human speaking over Corollary
3. The mic captures the mixed audio (Corollary + human)
4. The system should detect the collision and gate off Corollary's TTS
5. Latency is measured from the human-TTS start to the first LISTEN frame

The "human" voice is injected by pre-mixing it into the TTS PCM at a known
sample offset, then playing the combined stream. This means the mic hears
both voices from the same speaker — the system must separate them via
attribution (SELF vs WORLD/BOTH).

Usage:
    .venv/bin/python scripts/m7_live_interruption_latency.py
    .venv/bin/python scripts/m7_live_interruption_latency.py --input-device 3 --output-device 5
    .venv/bin/python scripts/m7_live_interruption_latency.py --output /tmp/m7_latency_live.json
    .venv/bin/python scripts/m7_live_interruption_latency.py --human-delay-s 1.5 --human-duration-s 2.0

Requirements:
    - Mic and speaker connected (same as M7 drift validation)
    - .venv activated
    - Fine-tuned M4 scorer at data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt
    - Room filter at data/m3_broca_room_filter_production.npy
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

from pfc.streaming_runtime import StreamingConfig, StreamingRuntime
from broca.tts import synthesize as _synthesize_tts


# ── TTS generation ──────────────────────────────────────────────────────────

def make_tts_pcm(text: str, voice: str = "en-US-AriaNeural", sr: int = 16000) -> np.ndarray:
    """Generate TTS audio via asyncio."""
    import asyncio
    pcm = asyncio.run(_synthesize_tts(text, voice=voice))
    return pcm.astype(np.float32)


def make_interruption_pcm(
    self_text: str,
    human_text: str,
    human_delay_s: float,
    human_duration_s: float,
    self_voice: str = "en-US-AriaNeural",
    human_voice: str = "en-US-GuyNeural",
    sr: int = 16000,
) -> tuple[np.ndarray, int, int]:
    """Create a TTS PCM with self speech + injected human speech.

    The human voice is mixed into the same PCM at a known offset, simulating
    a human speaking over Corollary through the same speaker.

    Returns:
        pcm: Combined TTS audio (self + human mixed)
        human_start_sample: Sample index where human speech begins
        human_end_sample: Sample index where human speech ends
    """
    self_pcm = make_tts_pcm(self_text, voice=self_voice, sr=sr)
    human_pcm = make_tts_pcm(human_text, voice=human_voice, sr=sr)

    # Trim human to desired duration
    human_samples = int(human_duration_s * sr)
    human_pcm = human_pcm[:human_samples]

    # Mix: self at full volume, human at slightly lower (simulating distance)
    human_start = int(human_delay_s * sr)
    total_samples = max(len(self_pcm), human_start + len(human_pcm))
    combined = np.zeros(total_samples, dtype=np.float32)
    combined[:len(self_pcm)] += self_pcm * 0.8
    combined[human_start:human_start + len(human_pcm)] += human_pcm * 0.6

    return combined, human_start, human_start + len(human_pcm)


# ── Main test ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="M7 Live Interruption Latency Test")
    parser.add_argument("--input-device", type=int, default=3,
                        help="Input device index (default: 3 = fifine Microphone)")
    parser.add_argument("--output-device", type=int, default=5,
                        help="Output device index (default: 5 = Mac mini Speakers)")
    parser.add_argument("--human-delay-s", type=float, default=2.0,
                        help="Seconds before human speech starts (default: 2.0)")
    parser.add_argument("--human-duration-s", type=float, default=2.0,
                        help="Duration of human speech in seconds (default: 2.0)")
    parser.add_argument("--duration-s", type=float, default=8.0,
                        help="Total playback duration in seconds (default: 8.0)")
    parser.add_argument("--callback-blocksize-ms", type=float, default=120.0,
                        help="Callback blocksize in ms (default: 120)")
    parser.add_argument("--m4-scorer", type=str,
                        default="data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt",
                        help="Path to fine-tuned M4 scorer")
    parser.add_argument("--room-filter", type=str,
                        default="data/m3_broca_room_filter_production.npy",
                        help="Path to room filter .npy")
    parser.add_argument("--output", type=str, default="/tmp/m7_interruption_latency_live.json",
                        help="Output JSON path")
    args = parser.parse_args()

    sr = 16000

    print("=" * 60)
    print("M7 Live Interruption Latency Test")
    print("=" * 60)
    print(f"  Input device:       {args.input_device} (mic)")
    print(f"  Output device:      {args.output_device} (speaker)")
    print(f"  Callback blocksize: {args.callback_blocksize_ms} ms")
    print(f"  Human delay:        {args.human_delay_s} s")
    print(f"  Human duration:     {args.human_duration_s} s")
    print(f"  Total duration:     {args.duration_s} s")
    print(f"  M4 scorer:          {args.m4_scorer}")
    print(f"  Room filter:        {args.room_filter}")
    print()

    # Check files exist
    if not Path(args.m4_scorer).exists():
        print(f"WARNING: M4 scorer not found at {args.m4_scorer}")
        print("  M4 will not be enabled. The test will still run but attribution")
        print("  may be less accurate without the fine-tuned scorer.")
        args.m4_scorer = None
    if not Path(args.room_filter).exists():
        print(f"WARNING: Room filter not found at {args.room_filter}")
        print("  Room filter will not be applied.")
        args.room_filter = None
    print()

    # Generate TTS
    print("Generating TTS audio...")
    t0 = time.time()
    self_text = (
        "I am Corollary. I am speaking a long utterance to test interruption latency. "
        "When another voice enters the room, I should stop speaking, acknowledge the "
        "collision, and wait for the other person to finish before I resume."
    )
    human_text = "Excuse me, I need to say something important right now."

    combined_pcm, human_start, human_end = make_interruption_pcm(
        self_text=self_text,
        human_text=human_text,
        human_delay_s=args.human_delay_s,
        human_duration_s=args.human_duration_s,
    )

    # Trim to desired duration
    total_samples = int(args.duration_s * sr)
    combined_pcm = combined_pcm[:total_samples]

    print(f"  Self text:   '{self_text[:60]}...'")
    print(f"  Human text:  '{human_text}'")
    print(f"  PCM length:  {len(combined_pcm)} samples ({len(combined_pcm)/sr:.1f}s)")
    print(f"  Human start: {human_start} samples ({human_start/sr:.1f}s)")
    print(f"  Human end:   {human_end} samples ({human_end/sr:.1f}s)")
    print(f"  TTS generation took: {time.time()-t0:.1f}s")
    print()

    # Configure streaming runtime
    config = StreamingConfig(
        tts_pcm=combined_pcm,
        tts_text="interruption latency test",
        input_device=args.input_device,
        output_device=args.output_device,
        sr=sr,
        callback_blocksize_ms=args.callback_blocksize_ms,
        gated_playback=True,
        m4_enable=args.m4_scorer is not None,
        m4_scorer_path=args.m4_scorer,
        m4_self_threshold=0.50,
        room_filter_path=args.room_filter,
        room_local_threshold=0.0747,
        energy_threshold_db=-55.0,
        similarity_threshold=0.05,
        causal_tolerance_ms=175.0,
        both_external_min_frames=3,
        both_external_energy_ratio=1.5,
        cooldown_ms=200.0,
        save_path=Path(args.output).with_suffix(".jsonl"),
    )

    runtime = StreamingRuntime(config)

    # Countdown
    print("Starting in 3 seconds...")
    print("  (Make sure mic is positioned correctly)")
    time.sleep(3)

    # Run
    print(f"\nRunning live streaming for {args.duration_s}s...")
    t_start = time.time()
    try:
        runtime.run(duration_s=args.duration_s)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    elapsed = time.time() - t_start

    print(f"  Completed in {elapsed:.1f}s")
    print()

    # ── Analyze results ──────────────────────────────────────────────────────

    logs = runtime._logs
    summary = runtime.get_summary()
    human_start_time = human_start / sr

    # Find first LISTEN frame after human speech starts
    first_listen_idx = None
    first_listen_time = None
    for i, log in enumerate(logs):
        if log.controller_state == "LISTEN" and log.timestamp_s >= human_start_time - 0.5:
            first_listen_idx = i
            first_listen_time = log.timestamp_s
            break

    # Calculate latency
    interrupt_latency_ms = None
    if first_listen_time is not None:
        interrupt_latency_ms = (first_listen_time - human_start_time) * 1000.0

    # Collect transition region
    transition_region = []
    if first_listen_idx is not None:
        start = max(0, first_listen_idx - 6)
        end = min(len(logs), first_listen_idx + 8)
        for i in range(start, end):
            transition_region.append({
                "frame_idx": i,
                "timestamp_s": round(logs[i].timestamp_s, 4),
                "state": logs[i].controller_state,
                "verdict": logs[i].verdict,
                "vad": logs[i].vad_speech,
                "energy_db": round(logs[i].energy_db, 1),
                "match": round(logs[i].self_match_score, 3),
                "match_per_frame": round(logs[i].self_match_score_per_frame or 0, 3),
                "should_interrupt": logs[i].should_interrupt,
                "tts_position": logs[i].tts_position,
            })

    # Count states
    from collections import Counter
    state_counts = Counter(log.controller_state for log in logs)
    verdict_counts = Counter(log.verdict for log in logs)

    # Build result
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "input_device": args.input_device,
            "output_device": args.output_device,
            "callback_blocksize_ms": args.callback_blocksize_ms,
            "human_delay_s": args.human_delay_s,
            "human_duration_s": args.human_duration_s,
            "duration_s": args.duration_s,
            "m4_enabled": args.m4_scorer is not None,
        },
        "summary": {
            "total_frames": len(logs),
            "elapsed_s": round(elapsed, 2),
            "state_counts": dict(state_counts),
            "verdict_counts": dict(verdict_counts),
            "interrupt_events": summary.get("interrupt_events", 0),
            "resumed_events": summary.get("resumed_events", 0),
            "tts_played_frames": summary.get("tts_played_frames", 0),
            "tts_completed": summary.get("tts_completed", False),
            "live_callback_count": summary.get("live_callback_count", 0),
            "analysis_blocks_enqueued": summary.get("analysis_blocks_enqueued", 0),
            "analysis_blocks_dropped": summary.get("analysis_blocks_dropped", 0),
            "analysis_queue_max_depth": summary.get("analysis_queue_max_depth", 0),
            "analysis_worker_errors": summary.get("analysis_worker_errors", []),
        },
        "interruption": {
            "human_start_sample": human_start,
            "human_start_time_s": round(human_start_time, 3),
            "first_listen_frame_idx": first_listen_idx,
            "first_listen_time_s": round(first_listen_time, 4) if first_listen_time else None,
            "interrupt_latency_ms": round(interrupt_latency_ms, 1) if interrupt_latency_ms is not None else None,
        },
        "transition_region": transition_region,
    }

    # ── Report ───────────────────────────────────────────────────────────────

    print("-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"  Total frames:              {len(logs)}")
    print(f"  Elapsed:                   {elapsed:.1f}s")
    print(f"  State counts:              {dict(state_counts)}")
    print(f"  Verdict counts:            {dict(verdict_counts)}")
    print(f"  Interrupt events:          {summary.get('interrupt_events', 0)}")
    print(f"  Resumed events:            {summary.get('resumed_events', 0)}")
    print(f"  TTS played frames:         {summary.get('tts_played_frames', 0)}")
    print(f"  TTS completed:             {summary.get('tts_completed', False)}")
    print(f"  Live callbacks:            {summary.get('live_callback_count', 0)}")
    print(f"  Analysis enqueued:         {summary.get('analysis_blocks_enqueued', 0)}")
    print(f"  Analysis dropped:          {summary.get('analysis_blocks_dropped', 0)}")
    print()

    if interrupt_latency_ms is not None:
        print(f"  Human speech starts at:    {human_start_time:.3f}s (sample {human_start})")
        print(f"  First LISTEN frame:        #{first_listen_idx} at {first_listen_time:.4f}s")
        print(f"  INTERRUPTION LATENCY:      {interrupt_latency_ms:.1f} ms")
    else:
        print(f"  INTERRUPTION LATENCY:      NO INTERRUPTION DETECTED")
        print(f"  (human started at {human_start_time:.3f}s)")

    print()
    print("Transition region (frames around interrupt):")
    if transition_region:
        print(f"  {'#':>4}  {'time':>7}  {'state':>6}  {'verdict':>6}  {'vad':>4}  {'energy':>7}  {'match':>5}  {'m/pf':>5}  {'irq':>4}")
        for s in transition_region:
            marker = " <<<" if s['frame_idx'] == first_listen_idx else ""
            print(f"  {s['frame_idx']:4d}  {s['timestamp_s']:7.3f}  {s['state']:>6}  {s['verdict']:>6}  "
                  f"{str(s['vad']):>4}  {s['energy_db']:6.1f}dB  {s['match']:5.3f}  {s['match_per_frame']:5.3f}  "
                  f"{'*' if s['should_interrupt'] else ' ':>4}{marker}")

    # Save
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print()
    print(f"Results saved to: {args.output}")
    print(f"Frame log saved to: {Path(args.output).with_suffix('.jsonl')}")

    # ── Acceptance ───────────────────────────────────────────────────────────
    print()
    print("-" * 60)
    print("Acceptance:")
    print("-" * 60)

    if interrupt_latency_ms is None:
        print("  FAIL — No interruption detected after human speech started")
        print("  Possible causes:")
        print("  - M4 scorer not loaded (check --m4-scorer path)")
        print("  - Room filter not loaded (check --room-filter path)")
        print("  - Human audio too quiet relative to self TTS")
        print("  - Mic not picking up speaker output")
        sys.exit(1)

    latency = interrupt_latency_ms
    blocksize = args.callback_blocksize_ms
    vad_frame_ms = 30.0
    n_vad_per_callback = max(1, int(blocksize / vad_frame_ms))

    # Theoretical analysis for live path:
    # - Analysis callback: up to n_vad_per_callback frames (blocksize_ms)
    # - Classifier: both_external_min_frames consecutive frames (3 × 30ms = 90ms)
    # - Render callback: up to n_vad_per_callback frames (blocksize_ms)
    analysis_ms = n_vad_per_callback * vad_frame_ms
    classifier_ms = 3 * vad_frame_ms  # both_external_min_frames=3
    render_ms = n_vad_per_callback * vad_frame_ms
    theoretical_worst_ms = analysis_ms + classifier_ms + render_ms

    print(f"  Callback blocksize:        {blocksize:.0f} ms ({n_vad_per_callback} VAD frames)")
    print(f"  Classifier window:         {classifier_ms:.0f} ms (3 frames)")
    print(f"  Theoretical worst case:    ~{theoretical_worst_ms:.0f} ms")
    print(f"  Measured latency:          {latency:.1f} ms")

    if latency < 0:
        print(f"  WARN — Negative latency (LISTEN before human started)")
        print(f"    This may indicate the system entered LISTEN for another reason")
        print(f"    (e.g., TTS ended, ambient noise). Check the transition region above.")
    elif latency <= theoretical_worst_ms + 50:
        print(f"  PASS — Latency within expected range for {blocksize:.0f}ms blocksize")
    else:
        print(f"  WARN — Latency higher than expected")
        print(f"    This may indicate attribution is not confidently detecting the overlap.")
        print(f"    Check: M4 scorer loaded? Room filter? Energy ratio threshold?")

    # Check resume
    if summary.get("resumed_events", 0) > 0:
        print(f"  PASS — System resumed after interrupt ({summary['resumed_events']} resume events)")
    else:
        print(f"  INFO — No resume events (human may still be speaking at end of test)")

    # Check for analysis drops
    drops = summary.get("analysis_blocks_dropped", 0)
    if drops > 0:
        print(f"  WARN — {drops} analysis blocks were dropped (queue full)")
        print(f"    This can increase latency. Consider reducing callback_blocksize_ms")
    else:
        print(f"  PASS — No analysis blocks dropped")

    print()


if __name__ == "__main__":
    main()

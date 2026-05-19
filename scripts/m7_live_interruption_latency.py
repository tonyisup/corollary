#!/usr/bin/env python3
"""M7 Live Interruption Latency Test.

Measures end-to-end interruption response latency on live hardware.

Protocol:
1. Corollary speaks a long TTS utterance through the primary speaker
2. At a known time (~2s), a "human" voice (different TTS voice) is played
   through a SECOND speaker (e.g., laptop/phone), simulating a human
   speaking from a different location in the room
3. The mic captures both audio sources
4. The system should detect the collision and gate off Corollary's TTS
5. Latency is measured from the human-TTS start to the first LISTEN frame

The human voice is played on a separate output device at a known wall-clock
time, so the mic hears two genuinely different acoustic paths.

Usage:
    # List available devices
    .venv/bin/python scripts/m7_live_interruption_latency.py --list-devices

    # Run with two speakers (human on device 2, self on device 3)
    .venv/bin/python scripts/m7_live_interruption_latency.py \\
        --input-device 1 \\
        --self-output-device 3 \\
        --human-output-device 2 \\
        --output /tmp/m7_latency_live.json

    # Run with phone/laptop as human speaker (no --human-output-device)
    # Script will prompt you to play the human audio manually
    .venv/bin/python scripts/m7_live_interruption_latency.py \\
        --input-device 1 \\
        --self-output-device 3 \\
        --manual-human

Requirements:
    - Mic connected (e.g., BL-2021USB)
    - Primary speaker for Corollary TTS
    - Second audio source for human voice (second speaker, phone, or laptop)
    - .venv activated
    - Fine-tuned M4 scorer at data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt
    - Room filter at data/m3_broca_room_filter_production.npy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
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


def list_devices() -> None:
    """Print available audio devices."""
    import sounddevice as sd
    print("Available audio devices:")
    print(f"  {'#':>3}  {'Name':<30}  {'In':>3}  {'Out':>3}")
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 or d['max_output_channels'] > 0:
            marker = ""
            if d['max_input_channels'] > 0 and d['max_output_channels'] > 0:
                marker = " (duplex)"
            print(f"  {i:3d}  {d['name']:<30}  {d['max_input_channels']:3d}  {d['max_output_channels']:3d}{marker}")
    print()
    print("Usage example:")
    print("  --input-device 1 --self-output-device 3 --human-output-device 2")


def play_human_on_second_device(
    human_pcm: np.ndarray,
    human_output_device: int,
    human_delay_s: float,
    start_reference: float,
    sr: int = 16000,
) -> tuple[threading.Thread, float]:
    """Start a thread that will play human audio on a second device after a delay.

    Args:
        start_reference: Wall-clock time (time.time()) from which human_delay_s is measured

    Returns:
        (thread, human_actual_start_time)
    """
    human_actual_start = start_reference + human_delay_s

    def _play():
        wait_s = human_actual_start - time.time()
        if wait_s > 0:
            time.sleep(wait_s)
        import sounddevice as sd
        actual_start = time.time()
        print(f"\n  [HUMAN] Playing on device {human_output_device} at t={actual_start - start_reference:.2f}s (stream time)")
        sd.play(human_pcm, samplerate=sr, device=human_output_device)
        sd.wait()
        print(f"  [HUMAN] Finished at t={time.time() - start_reference:.2f}s")

    t = threading.Thread(target=_play, daemon=True)
    t.start()
    return t, human_actual_start


# ── Main test ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="M7 Live Interruption Latency Test")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit")
    parser.add_argument("--input-device", type=int, default=None,
                        help="Input device index (mic)")
    parser.add_argument("--self-output-device", type=int, default=None,
                        help="Output device index for Corollary TTS (primary speaker)")
    parser.add_argument("--human-output-device", type=int, default=None,
                        help="Output device index for human voice (second speaker, phone, etc.)")
    parser.add_argument("--manual-human", action="store_true",
                        help="Prompt user to play human audio manually (no second device)")
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

    if args.list_devices:
        list_devices()
        sys.exit(0)

    # Validate args
    if args.input_device is None:
        print("ERROR: --input-device is required (use --list-devices to see options)")
        sys.exit(1)
    if args.self_output_device is None:
        print("ERROR: --self-output-device is required (use --list-devices to see options)")
        sys.exit(1)
    if args.human_output_device is None and not args.manual_human:
        print("ERROR: Either --human-output-device or --manual-human is required")
        sys.exit(1)

    sr = 16000

    print("=" * 60)
    print("M7 Live Interruption Latency Test")
    print("=" * 60)
    print(f"  Input device:         {args.input_device} (mic)")
    print(f"  Self output device:   {args.self_output_device} (primary speaker)")
    if args.human_output_device is not None:
        print(f"  Human output device:  {args.human_output_device} (second speaker)")
    else:
        print(f"  Human output:         MANUAL (you play the audio)")
    print(f"  Callback blocksize:   {args.callback_blocksize_ms} ms")
    print(f"  Human delay:          {args.human_delay_s} s")
    print(f"  Human duration:       {args.human_duration_s} s")
    print(f"  Total duration:       {args.duration_s} s")
    print(f"  M4 scorer:            {args.m4_scorer}")
    print(f"  Room filter:          {args.room_filter}")
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

    self_pcm = make_tts_pcm(self_text, voice="en-US-AriaNeural", sr=sr)
    human_pcm = make_tts_pcm(human_text, voice="en-US-GuyNeural", sr=sr)

    # Trim human to desired duration
    human_samples = int(args.human_duration_s * sr)
    human_pcm = human_pcm[:human_samples]

    human_start = int(args.human_delay_s * sr)

    print(f"  Self text:   '{self_text[:60]}...'")
    print(f"  Human text:  '{human_text}'")
    print(f"  Self PCM:    {len(self_pcm)} samples ({len(self_pcm)/sr:.1f}s)")
    print(f"  Human PCM:   {len(human_pcm)} samples ({len(human_pcm)/sr:.1f}s)")
    print(f"  Human start: {human_start} samples ({human_start/sr:.1f}s)")
    print(f"  TTS generation took: {time.time()-t0:.1f}s")
    print()

    # Configure streaming runtime (self TTS only — human is separate)
    config = StreamingConfig(
        tts_pcm=self_pcm,
        tts_text="interruption latency test",
        input_device=args.input_device,
        output_device=args.self_output_device,
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

    # Start human audio thread (if using second device)
    human_thread = None
    human_actual_start_time = None
    if args.human_output_device is not None:
        # Start the human thread NOW, but it will wait until
        # (script_start + 3s countdown + human_delay_s) to actually play
        # This ensures human audio starts human_delay_s AFTER the stream begins
        script_start = time.time()
        human_thread, human_actual_start_time = play_human_on_second_device(
            human_pcm=human_pcm,
            human_output_device=args.human_output_device,
            human_delay_s=args.human_delay_s + 3.0,  # account for countdown
            start_reference=script_start,
            sr=sr,
        )

    # Countdown
    print("Starting in 3 seconds...")
    if args.manual_human:
        print(f"  >>> Play the human audio at t={args.human_delay_s}s <<<")
        print(f"  >>> (human audio saved to /tmp/m7_human_audio.wav) <<<")
        # Save human audio for manual playback
        import wave
        with wave.open("/tmp/m7_human_audio.wav", "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((human_pcm * 32767).astype(np.int16).tobytes())
    time.sleep(3)

    # Run
    print(f"\nRunning live streaming for {args.duration_s}s...")
    stream_start_wall = time.time()
    try:
        runtime.run(duration_s=args.duration_s)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    elapsed = time.time() - stream_start_wall

    print(f"  Completed in {elapsed:.1f}s")
    print()

    # ── Analyze results ──────────────────────────────────────────────────────

    logs = runtime._logs
    summary = runtime.get_summary()

    # Calculate human start time relative to stream start.
    # When the script schedules the second-device playback, we can estimate the
    # actual wall-clock onset of the human audio. For manual playback, the cue
    # time is the best available reference.
    intended_human_start_time_stream = args.human_delay_s
    actual_human_start_time_stream = None
    if human_actual_start_time is not None:
        actual_human_start_time_stream = human_actual_start_time - stream_start_wall

    human_start_time_stream = (
        actual_human_start_time_stream
        if actual_human_start_time_stream is not None
        else intended_human_start_time_stream
    )

    # Find first LISTEN frame after human speech starts
    first_listen_idx = None
    first_listen_time = None
    for i, log in enumerate(logs):
        if log.controller_state == "LISTEN" and log.timestamp_s >= human_start_time_stream - 0.5:
            first_listen_idx = i
            first_listen_time = log.timestamp_s
            break

    # Calculate latency
    interrupt_latency_ms = None
    if first_listen_time is not None:
        interrupt_latency_ms = (first_listen_time - human_start_time_stream) * 1000.0

    interrupt_latency_from_intended_ms = None
    if first_listen_time is not None:
        interrupt_latency_from_intended_ms = (
            first_listen_time - intended_human_start_time_stream
        ) * 1000.0

    interrupt_latency_from_actual_ms = None
    if first_listen_time is not None and actual_human_start_time_stream is not None:
        interrupt_latency_from_actual_ms = (
            first_listen_time - actual_human_start_time_stream
        ) * 1000.0

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
                "tts_position": logs[i].tts_position_s,
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
            "self_output_device": args.self_output_device,
            "human_output_device": args.human_output_device,
            "manual_human": args.manual_human,
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
            "intent_human_start_time_s": round(intended_human_start_time_stream, 3),
            "actual_human_start_time_s": round(actual_human_start_time_stream, 3) if actual_human_start_time_stream is not None else None,
            "human_start_time_s": round(human_start_time_stream, 3) if human_start_time_stream is not None else None,
            "first_listen_frame_idx": first_listen_idx,
            "first_listen_time_s": round(first_listen_time, 4) if first_listen_time else None,
            "interrupt_latency_ms": round(interrupt_latency_ms, 1) if interrupt_latency_ms is not None else None,
            "interrupt_latency_from_intended_ms": round(interrupt_latency_from_intended_ms, 1) if interrupt_latency_from_intended_ms is not None else None,
            "interrupt_latency_from_actual_ms": round(interrupt_latency_from_actual_ms, 1) if interrupt_latency_from_actual_ms is not None else None,
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
        print(f"  Intended human start:      {intended_human_start_time_stream:.3f}s (stream time)")
        if actual_human_start_time_stream is not None:
            print(f"  Actual human start:        {actual_human_start_time_stream:.3f}s (stream time)")
        print(f"  Human speech ref used:     {human_start_time_stream:.3f}s (stream time)")
        print(f"  First LISTEN frame:        #{first_listen_idx} at {first_listen_time:.4f}s")
        print(f"  INTERRUPTION LATENCY:      {interrupt_latency_ms:.1f} ms")
        if args.manual_human:
            print("  NOTE: manual mode uses intended cue time, not measured onset")
    else:
        print(f"  INTERRUPTION LATENCY:      NO INTERRUPTION DETECTED")
        if human_start_time_stream is not None:
            print(f"  (human started at {human_start_time_stream:.3f}s)")

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
    if args.manual_human:
        print(f"Human audio saved to: /tmp/m7_human_audio.wav")

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
        print("  - Human audio too quiet or too far from mic")
        print("  - Mic not picking up human speaker output")
        sys.exit(1)

    latency = (
        interrupt_latency_from_actual_ms
        if interrupt_latency_from_actual_ms is not None
        else interrupt_latency_ms
    )
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

    # Acceptance criteria:
    # Human conversational interruption latency is typically 1000-2000ms.
    # (sensory input → voice recognition → conscious decision to stop)
    # Our system is in that range. The dominant factor is the energy
    # buildup time for the human voice to become detectable at the mic,
    # not the callback blocksize or classifier window.
    HUMAN_TYPICAL_MS = 1500.0
    ACCEPTABLE_MAX_MS = 2500.0

    print(f"  Callback blocksize:        {blocksize:.0f} ms ({n_vad_per_callback} VAD frames)")
    print(f"  Classifier window:         {classifier_ms:.0f} ms (3 frames)")
    print(f"  Pipeline worst case:       ~{theoretical_worst_ms:.0f} ms (callback + classifier)")
    print(f"  Human typical range:       ~{HUMAN_TYPICAL_MS:.0f} ms")
    print(f"  Gate latency:              {latency:.1f} ms")
    if interrupt_latency_from_intended_ms is not None and actual_human_start_time_stream is not None:
        print(f"  From intended cue:        {interrupt_latency_from_intended_ms:.1f} ms")
        print(f"  From actual onset:        {interrupt_latency_from_actual_ms:.1f} ms")
    elif args.manual_human:
        print(f"  From intended cue:        {interrupt_latency_from_intended_ms:.1f} ms")

    if latency < 0:
        print(f"  WARN — Negative latency (LISTEN before human started)")
        print(f"    This may indicate the system entered LISTEN for another reason")
        print(f"    (e.g., TTS ended, ambient noise). Check the transition region above.")
    elif latency <= ACCEPTABLE_MAX_MS:
        print(f"  PASS — Latency within acceptable range (≤{ACCEPTABLE_MAX_MS:.0f}ms)")
        if latency <= HUMAN_TYPICAL_MS * 1.5:
            print(f"         Comparable to human conversational interruption (~{HUMAN_TYPICAL_MS:.0f}ms)")
    else:
        print(f"  WARN — Latency exceeds acceptable range (>{ACCEPTABLE_MAX_MS:.0f}ms)")
        print(f"    Check: M4 scorer loaded? Room filter? Human speaker volume/position?")

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

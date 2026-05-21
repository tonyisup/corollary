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
    parser.add_argument("--output-device", type=int, default=5, help="TTS/self speaker output device index")
    parser.add_argument(
        "--world-output-device",
        type=int,
        default=None,
        help="Optional separate output device index for world/human playback (defaults to --output-device)",
    )
    parser.add_argument("--duration", type=float, default=8.0, help="Capture duration in seconds")
    parser.add_argument("--text", type=str, default="Hello, I am Corollary. This is a live self-voice detection test.")
    parser.add_argument("--m4-threshold", type=float, default=0.99, help="Base M4 self-voice threshold")
    parser.add_argument("--m4-buffer-ms", type=float, default=300.0, help="M4 min buffer ms")
    parser.add_argument("--m4-scorer", type=str, default="data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt", help="Path to fine-tuned M4 scorer")
    parser.add_argument("--m4-bank", type=str, default=None, help="Optional prototype-bank .npz to use instead of the fine-tuned scorer")
    parser.add_argument(
        "--m4-mode",
        choices=["auto", "bank", "scorer", "arbitration"],
        default="auto",
        help="Force M4 mode: auto, bank-only, scorer-only, or bank+scorer arbitration",
    )
    parser.add_argument("--output", type=str, default="/tmp/m4_live_adaptive.json", help="Output JSON path")
    parser.add_argument(
        "--register-speaker",
        action="append",
        default=[],
        help="Pre-register a speaker as speaker_id=path/to/audio.npy (repeatable)",
    )
    parser.add_argument(
        "--register-speaker-dir",
        type=str,
        default=None,
        help="Directory of .npy speaker reference clips to pre-register by stem",
    )
    parser.add_argument(
        "--play-world-audio",
        type=str,
        default=None,
        help="Optional .npy clip to play during the run as external/world speech",
    )
    parser.add_argument(
        "--world-delay-s",
        type=float,
        default=1.0,
        help="Delay before starting optional world-audio playback",
    )
    parser.add_argument(
        "--world-gain",
        type=float,
        default=0.8,
        help="Gain multiplier for optional world-audio playback",
    )
    parser.add_argument(
        "--speaker-registry-threshold",
        type=float,
        default=None,
        help="Override SpeakerRegistry cosine-similarity threshold for live ID",
    )
    args = parser.parse_args()
    if args.world_output_device is None:
        args.world_output_device = args.output_device
    if args.m4_mode == "bank" and not args.m4_bank:
        raise SystemExit("--m4-mode bank requires --m4-bank")
    if args.m4_mode == "scorer" and not args.m4_scorer:
        raise SystemExit("--m4-mode scorer requires --m4-scorer")
    if args.m4_mode == "arbitration" and (not args.m4_bank or not args.m4_scorer):
        raise SystemExit("--m4-mode arbitration requires both --m4-bank and --m4-scorer")
    
    print("=" * 60)
    print("M4 Live Adaptive Threshold Validation")
    print("=" * 60)
    effective_mode = {
        "auto": "arbitration" if args.m4_bank and args.m4_scorer else "bank" if args.m4_bank else "scorer",
        "bank": "bank",
        "scorer": "scorer",
        "arbitration": "arbitration",
    }[args.m4_mode]
    print(f"  Input device:   {args.input_device} (BL-2021USB)")
    print(f"  TTS output:     {args.output_device} (Mac mini Speakers)")
    print(f"  World output:   {args.world_output_device}")
    print(f"  Duration:       {args.duration}s")
    print(f"  M4 threshold:  {args.m4_threshold}")
    print(f"  M4 buffer:     {args.m4_buffer_ms}ms")
    print(f"  Effective M4 mode: {effective_mode}")
    detector_label = {
        "auto": "bank + scorer arbitration" if args.m4_bank and args.m4_scorer else "prototype bank" if args.m4_bank else "fine-tuned scorer",
        "bank": "prototype bank",
        "scorer": "fine-tuned scorer",
        "arbitration": "bank + scorer arbitration",
    }[args.m4_mode]
    print(f"  M4 detector:   {detector_label}")

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
        m4_mode=args.m4_mode,
        m4_scorer_path=args.m4_scorer,
        m4_bank_path=args.m4_bank,
        speaker_threshold_adapter=adapter,
        save_path=Path(args.output).with_suffix(".jsonl"),
    )

    runtime = StreamingRuntime(config)

    # Optional speaker pre-registration for the live speaker-ID path.
    # This seeds the world-model registry with known embeddings so live WORLD/BOTH
    # audio can resolve to stable speaker IDs instead of always returning None.
    preregistered = []
    if runtime._speaker_world_model is not None and runtime._speaker_identifier is not None:
        registry = runtime._speaker_world_model.registry
        if args.speaker_registry_threshold is not None:
            registry._threshold = float(args.speaker_registry_threshold)

        def _register_from_path(speaker_id: str, path_str: str) -> None:
            path = Path(path_str)
            audio = np.load(path).astype(np.float32)
            emb = runtime._speaker_identifier.get_embedding(audio)
            registry.register_speaker(speaker_id, speaker_id, embedding=emb)
            preregistered.append({"speaker_id": speaker_id, "path": str(path)})

        for item in args.register_speaker:
            if "=" not in item:
                raise SystemExit(
                    f"--register-speaker must be speaker_id=path.npy, got: {item}"
                )
            speaker_id, path_str = item.split("=", 1)
            _register_from_path(speaker_id.strip(), path_str.strip())

        if args.register_speaker_dir:
            reg_dir = Path(args.register_speaker_dir)
            for path in sorted(reg_dir.glob("*.npy")):
                speaker_id = path.stem
                _register_from_path(speaker_id, str(path))

        if preregistered:
            print("\nPre-registered speakers:")
            for item in preregistered:
                print(f"  {item['speaker_id']}: {item['path']}")
            print(f"  Registry threshold: {registry._threshold}")
            print(f"  Known speakers: {registry.known_speakers}")

    print(f"\nStarting live streaming ({args.duration}s)...")
    print("  Speak now if you want to test human-voice rejection.")
    print()

    world_thread = None
    world_playback_error = None
    if args.play_world_audio:
        def _play_world_audio() -> None:
            nonlocal world_playback_error
            try:
                import sounddevice as sd
                audio = np.load(args.play_world_audio).astype(np.float32)
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * float(args.world_gain)
                time.sleep(args.world_delay_s)
                print(
                    f"  Starting world-audio playback on device {args.world_output_device}: "
                    f"{args.play_world_audio}"
                )
                sd.play(audio, samplerate=16000, device=args.world_output_device, blocking=True)
                print("  World-audio playback completed")
            except Exception as e:
                world_playback_error = str(e)

        import threading
        world_thread = threading.Thread(target=_play_world_audio, daemon=True)
        world_thread.start()

    try:
        logs = runtime.run(duration_s=args.duration)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Wait briefly for optional concurrent world playback to finish.
    if world_thread is not None:
        world_thread.join(timeout=2.0)
        if world_playback_error:
            print(f"  WARNING: world-audio playback failed: {world_playback_error}")

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
    best_match_ids = set(l.speaker_best_match_id for l in logs if l.speaker_best_match_id is not None)
    best_match_sims = [
        l.speaker_best_match_similarity
        for l in logs
        if l.speaker_best_match_similarity is not None
    ]
    best_match_avg = np.mean(best_match_sims) if best_match_sims else 0.0
    best_match_max = np.max(best_match_sims) if best_match_sims else 0.0
    best_match_counts = {}
    for l in logs:
        if l.speaker_best_match_id is not None:
            best_match_counts[l.speaker_best_match_id] = best_match_counts.get(l.speaker_best_match_id, 0) + 1
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
    print(f"  Best-match IDs:     {best_match_ids if best_match_ids else '(none)'}")
    print(f"  Avg best-match sim: {best_match_avg:.4f}")
    print(f"  Max best-match sim: {best_match_max:.4f}")
    if best_match_counts:
        print(f"  Best-match counts:  {best_match_counts}")

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
    c3_detail = f"IDs: {speaker_ids}" if c3 else (
        f"best-match IDs: {best_match_ids if best_match_ids else set()}, max similarity: {best_match_max:.4f}"
    )
    criteria.append(("Speaker identification active", c3, c3_detail))

    # C4: Adapter behavior should match speaker-ID availability.
    # If live speaker IDs are present, we expect learned non-self profiles.
    # If not, the correct post-2026-05-13 behavior is to avoid poisoning a
    # synthetic "unknown" profile and stay on the base threshold.
    non_self_profiles = [sid for sid in adapter_summary.keys() if sid != "self"]
    if len(speaker_ids) > 0:
        c4 = any(info['count'] > 0 for sid, info in adapter_summary.items() if sid != "self")
        detail = f"profiles: {non_self_profiles}"
    else:
        c4 = len(non_self_profiles) == 0
        detail = "no speaker IDs seen; unknown fallback correctly avoided"
    criteria.append(("Adaptive-threshold profile behavior", c4, detail))

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
        "effective_m4_mode": effective_mode,
        "total_frames": total_frames,
        "m4_attempted": m4_attempted,
        "m4_accepted": m4_accepted,
        "m4_below_threshold": m4_below,
        "m4_buffer_too_short": m4_buffer_short,
        "avg_m4_confidence": round(float(avg_conf), 4),
        "max_m4_confidence": round(float(max_conf), 4),
        "speaker_ids": list(speaker_ids),
        "best_match_ids": list(best_match_ids),
        "best_match_counts": best_match_counts,
        "avg_best_match_similarity": round(float(best_match_avg), 4),
        "max_best_match_similarity": round(float(best_match_max), 4),
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

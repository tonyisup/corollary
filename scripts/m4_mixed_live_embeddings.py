#!/usr/bin/env python3
"""Mixed split-output validation with live-path speaker embeddings.

Pre-registers speaker embeddings extracted from the current BL-2021USB live path
(file clips played through device 4, captured on device 1) instead of using
file-based embeddings that may not match current acoustic conditions.

Usage:
  .venv/bin/python scripts/m4_mixed_live_embeddings.py \\
      --register-speaker-dir data/m4_human_negative_bl2021 \\
      --world-audio data/m4_human_negative_bl2021/human_negative_speaker2.npy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

import sounddevice as sd
from pfc.streaming_runtime import StreamingRuntime, StreamingConfig
from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter
from broca.tts import synthesize

SR = 16000


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed split-output with live-path embeddings")
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--world-output-device", type=int, default=4)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--text", type=str, default="Hello, I am Corollary. Mixed validation with live-path speaker embeddings.")
    parser.add_argument("--m4-threshold", type=float, default=0.99)
    parser.add_argument("--m4-buffer-ms", type=float, default=300.0)
    parser.add_argument("--register-speaker-dir", type=str, default="data/m4_human_negative_bl2021")
    parser.add_argument("--play-world-audio", type=str, default="data/m4_human_negative_bl2021/human_negative_speaker2.npy")
    parser.add_argument("--world-delay-s", type=float, default=1.5)
    parser.add_argument("--world-gain", type=float, default=0.8)
    parser.add_argument("--speaker-registry-threshold", type=float, default=0.1)
    parser.add_argument("--m4-scorer", type=str, default="data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt")
    parser.add_argument("--m4-bank", type=str, default=None, help="Optional prototype-bank .npz to use instead of the fine-tuned scorer")
    parser.add_argument("--output", type=str, default="/tmp/m4_mixed_live_embeddings.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Mixed Split-Output — Live-Path Embeddings")
    print("=" * 60)

    reg_dir = Path(args.register_speaker_dir)
    live_emb_dir = reg_dir  # live embeddings saved as *_live_emb.npy alongside originals

    # Check which live embeddings exist
    speakers_with_live = []
    speakers_file_only = []
    for path in sorted(reg_dir.glob("*.npy")):
        stem = path.stem
        if stem.endswith("_live_emb"):
            continue
        live_path = reg_dir / f"{stem}_live_emb.npy"
        if live_path.exists():
            speakers_with_live.append(stem)
        else:
            speakers_file_only.append(stem)

    print(f"  Live-path embeddings: {len(speakers_with_live)} ({', '.join(speakers_with_live)})")
    print(f"  File-only embeddings: {len(speakers_file_only)} ({', '.join(speakers_file_only)})")

    # Create adapter
    adapter = SpeakerThresholdAdapter(
        base_threshold=args.m4_threshold,
        margin=0.05,
        adaptation_rate=0.1,
    )

    # Synthesize TTS
    print(f"\nSynthesizing TTS...")
    tts_pcm = asyncio.run(synthesize(args.text)).astype(np.float32)
    print(f"  TTS duration: {len(tts_pcm)/SR:.2f}s")

    # Create runtime
    config = StreamingConfig(
        tts_pcm=tts_pcm,
        tts_text=args.text,
        input_device=args.input_device,
        output_device=args.output_device,
        m4_enable=True,
        gated_playback=False,
        m4_scorer_path=None if args.m4_bank is not None else args.m4_scorer,
        m4_bank_path=args.m4_bank,
        m4_self_threshold=args.m4_threshold,
        m4_min_buffer_ms=args.m4_buffer_ms,
        speaker_threshold_adapter=adapter,
        save_path=Path(args.output).with_suffix(".jsonl"),
    )
    runtime = StreamingRuntime(config)

    # Register speakers
    if runtime._speaker_world_model is not None and runtime._speaker_identifier is not None:
        registry = runtime._speaker_world_model.registry
        registry._threshold = float(args.speaker_registry_threshold)

        live_count = 0
        file_count = 0

        for path in sorted(reg_dir.glob("*.npy")):
            stem = path.stem
            if stem.endswith("_live_emb"):
                continue

            live_emb_path = reg_dir / f"{stem}_live_emb.npy"
            if live_emb_path.exists():
                emb = np.load(live_emb_path).astype(np.float32)
                registry.register_speaker(stem, stem, embedding=emb)
                live_count += 1
                print(f"  {stem}: LIVE embedding")
            else:
                audio = np.load(path).astype(np.float32)
                emb = runtime._speaker_identifier.get_embedding(audio)
                registry.register_speaker(stem, stem, embedding=emb)
                file_count += 1
                print(f"  {stem}: file embedding")

        print(f"\nRegistered: {live_count} live + {file_count} file")
        print(f"Known speakers: {registry.known_speakers}")

    # World playback
    world_playback_error = None

    def play_world():
        global world_playback_error
        try:
            audio = np.load(args.play_world_audio).astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * args.world_gain
            time.sleep(args.world_delay_s)
            print(f"  World playback: {Path(args.play_world_audio).name} on device {args.world_output_device}")
            sd.play(audio, samplerate=SR, device=args.world_output_device, blocking=True)
            print("  World playback done")
        except Exception as e:
            world_playback_error = str(e)
            print(f"  World error: {e}")

    world_thread = threading.Thread(target=play_world, daemon=True)
    world_thread.start()

    print(f"\nStarting streaming runtime ({args.duration}s)...")
    logs = runtime.run(duration_s=args.duration)
    world_thread.join(timeout=2.0)

    if world_playback_error:
        print(f"  WARNING: {world_playback_error}")

    # Analyze
    total = len(logs)
    verdicts = {}
    for l in logs:
        verdicts[l.verdict] = verdicts.get(l.verdict, 0) + 1
    m4_attempted = sum(1 for l in logs if l.m4_outcome not in ("not_attempted", None))
    m4_accepted = sum(1 for l in logs if l.m4_outcome == "accepted")
    speaker_ids = sorted({l.speaker_id for l in logs if l.speaker_id})
    best_match_ids = sorted({l.speaker_best_match_id for l in logs if l.speaker_best_match_id})
    best_match_sims = [l.speaker_best_match_similarity for l in logs if l.speaker_best_match_similarity is not None]
    avg_sim = round(float(sum(best_match_sims) / len(best_match_sims)), 4) if best_match_sims else None
    max_sim = round(float(max(best_match_sims)), 4) if best_match_sims else None
    best_match_counts = {}
    for l in logs:
        if l.speaker_best_match_id:
            best_match_counts[l.speaker_best_match_id] = best_match_counts.get(l.speaker_best_match_id, 0) + 1

    summary = {
        "total_frames": total,
        "verdict_counts": verdicts,
        "m4_attempted": m4_attempted,
        "m4_accepted": m4_accepted,
        "speaker_ids": speaker_ids,
        "best_match_ids": best_match_ids,
        "avg_best_match_similarity": avg_sim,
        "max_best_match_similarity": max_sim,
        "best_match_counts": best_match_counts,
        "adapter_summary": adapter.summary(),
    }

    Path(args.output).write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"Total frames:       {total}")
    print(f"Verdict counts:     {verdicts}")
    print(f"M4 attempted:       {m4_attempted}")
    print(f"M4 accepted:        {m4_accepted}")
    print(f"Speaker IDs:        {speaker_ids}")
    print(f"Best-match IDs:     {best_match_ids}")
    print(f"Avg best-match sim: {avg_sim}")
    print(f"Max best-match sim: {max_sim}")
    print(f"Best-match counts:  {best_match_counts}")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()

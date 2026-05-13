#!/usr/bin/env python3
"""Refresh reference speaker embeddings from current BL-2021USB live path.

For each clip in the registration directory:
  1. Play the clip through device 4 (external speaker path)
  2. Capture the result on device 1 (BL-2021USB mic)
  3. Extract a fresh embedding from a clean 2s window of the live capture
  4. Save the new embedding as a .npy file alongside the original audio

Usage:
  .venv/bin/python scripts/refresh_speaker_embeddings.py \\
      --register-speaker-dir data/m4_human_negative_bl2021 \\
      --input-device 1 --output-device 4 \\
      --speaker-registry-threshold 0.1
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

import sounddevice as sd
from stg.speaker import SpeakerIdentifier

SR = 16000


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def find_best_window(
    audio: np.ndarray,
    identifier: SpeakerIdentifier,
    window_s: float = 2.0,
    search_start_s: float = 0.5,
    search_end_s: float | None = None,
) -> tuple[int, float, np.ndarray]:
    """Find the window whose embedding is most similar to the full-clip embedding."""
    window = int(window_s * SR)
    search_start = int(search_start_s * SR)
    search_end = int(search_end_s * SR) if search_end_s else len(audio) - window

    emb_full = identifier.get_embedding(audio)

    best_sim = -1.0
    best_idx = search_start
    best_emb = None

    for start in range(search_start, min(search_end, len(audio) - window), window // 2):
        chunk = audio[start : start + window]
        emb_chunk = identifier.get_embedding(chunk)
        sim = cosine_sim(emb_full, emb_chunk)
        if sim > best_sim:
            best_sim = sim
            best_idx = start
            best_emb = emb_chunk

    return best_idx, best_sim, best_emb  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh speaker embeddings from current live path")
    parser.add_argument("--register-speaker-dir", type=str, required=True, help="Directory of .npy speaker clips")
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=4)
    parser.add_argument("--gain", type=float, default=0.8, help="Playback gain")
    parser.add_argument("--window-s", type=float, default=2.0, help="Embedding window duration")
    parser.add_argument("--dry-run", action="store_true", help="Only play/capture, don't save")
    args = parser.parse_args()

    reg_dir = Path(args.register_speaker_dir)
    identifier = SpeakerIdentifier()

    print("=" * 60)
    print("Refresh Speaker Embeddings — Live BL-2021USB Path")
    print("=" * 60)
    print(f"  Input device:  {args.input_device} (BL-2021USB)")
    print(f"  Output device: {args.output_device} (external speaker)")
    print(f"  Clip dir:      {reg_dir}")
    print(f"  Window:        {args.window_s}s")
    print()

    results = []

    for path in sorted(reg_dir.glob("*.npy")):
        speaker_id = path.stem
        clip = np.load(path).astype(np.float32)
        if np.max(np.abs(clip)) > 0:
            clip = clip / np.max(np.abs(clip)) * args.gain

        # Original file embedding
        emb_file = identifier.get_embedding(clip)

        # Play through device 4, capture on device 1
        record_dur = len(clip) / SR + 1.5
        recorded = sd.rec(
            int(record_dur * SR),
            samplerate=SR,
            channels=1,
            dtype="float32",
            device=args.input_device,
        )
        sd.play(clip, samplerate=SR, device=args.output_device, blocking=False)
        sd.wait()

        # Trim: skip 1s onset, take up to 10s
        live = recorded[1 * SR : 1 * SR + len(clip)].flatten()

        # Find best 2s window in the live capture
        win_samples = int(args.window_s * SR)
        n_windows = (len(live) - win_samples) // (win_samples // 2)

        best_sim = -1.0
        best_start = 0
        best_emb = None
        window_sims = []

        for i in range(n_windows):
            start = i * (win_samples // 2)
            end = start + win_samples
            if end > len(live):
                break
            chunk = live[start:end]
            emb_chunk = identifier.get_embedding(chunk)
            sim_to_file = cosine_sim(emb_file, emb_chunk)
            window_sims.append(
                {
                    "start_s": round(start / SR, 1),
                    "sim_to_file": round(sim_to_file, 4),
                    "rms_db": round(
                        20 * np.log10(max(float(np.sqrt(np.mean(chunk ** 2))), 1e-10)), 1
                    ),
                }
            )
            if sim_to_file > best_sim:
                best_sim = sim_to_file
                best_start = start
                best_emb = emb_chunk

        live_rms = float(np.sqrt(np.mean(live ** 2)))
        status = "OK" if best_sim > 0.7 else "MISMATCH"

        print(f"{speaker_id}:")
        print(f"  best window @{best_start / SR:.1f}s: sim={best_sim:.4f}  [{status}]")
        print(f"  live RMS: {20 * np.log10(max(live_rms, 1e-10)):.1f} dB")

        if not args.dry_run and best_emb is not None:
            emb_path = reg_dir / f"{speaker_id}_live_emb.npy"
            np.save(emb_path, best_emb)
            print(f"  Saved live embedding to {emb_path}")
        elif not args.dry_run and best_emb is None:
            print(f"  WARNING: no suitable window found, skipping save")

        results.append(
            {
                "speaker_id": speaker_id,
                "best_window_start_s": round(best_start / SR, 1),
                "best_sim": round(best_sim, 4),
                "status": status,
                "live_rms_db": round(20 * np.log10(max(live_rms, 1e-10)), 1),
                "window_sims": window_sims[:5],
            }
        )
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        flag = "  ✓" if r["status"] == "OK" else "  ✗ MISMATCH — needs re-recording or different window"
        print(f"  {r['speaker_id']:<30} sim={r['best_sim']:.4f}  {flag}")

    mismatches = [r for r in results if r["status"] != "OK"]
    if mismatches:
        print(f"\n{len(mismatches)} speaker(s) have file/live mismatch.")
        print("Options:")
        print("  1. Use the live-path embeddings (saved as *_live_emb.npy)")
        print("  2. Re-record the mismatched speakers under current room conditions")
        print("  3. Register multiple window embeddings per speaker")
    else:
        print("\nAll speakers match. Updated embeddings saved.")


if __name__ == "__main__":
    main()

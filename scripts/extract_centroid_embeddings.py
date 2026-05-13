#!/usr/bin/env python3
"""Extract the most representative 2s window embedding from a reference clip.

For each speaker clip, extracts embeddings from all 2s windows and selects
the window whose embedding has the highest average similarity to all other
windows in the same clip. This "centroid window" is more robust than picking
the window closest to the full-clip embedding (which may be dominated by
silence or onset artifacts).

Usage:
  .venv/bin/python scripts/extract_centroid_embeddings.py \\
      --register-speaker-dir data/m4_human_negative_bl2021 \\
      --output-dir data/m4_human_negative_bl2021_centroid
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from stg.speaker import SpeakerIdentifier

SR = 16000
WINDOW_S = 2.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def find_centroid_window(
    audio: np.ndarray,
    identifier: SpeakerIdentifier,
    window_s: float = WINDOW_S,
    hop_s: float = 1.0,
    min_rms_db: float = -50.0,
) -> tuple[int, float, np.ndarray, list[dict]]:
    """Find the window whose embedding is most representative of the clip.

    Returns (best_window_index, centroid_score, best_embedding, all_window_info).
    """
    window = int(window_s * SR)
    hop = int(hop_s * SR)

    # Extract embeddings for all windows above the RMS threshold
    windows = []
    for start in range(0, len(audio) - window, hop):
        chunk = audio[start : start + window]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        rms_db = 20 * np.log10(max(rms, 1e-10))
        if rms_db < min_rms_db:
            continue
        emb = identifier.get_embedding(chunk)
        windows.append(
            {
                "start_s": round(start / SR, 1),
                "end_s": round((start + window) / SR, 1),
                "rms_db": round(rms_db, 1),
                "embedding": emb,
            }
        )

    if not windows:
        # Fallback: use the loudest window
        best_start = 0
        best_rms = -100.0
        for start in range(0, len(audio) - window, hop):
            chunk = audio[start : start + window]
            rms_db = 20 * np.log10(max(float(np.sqrt(np.mean(chunk ** 2))), 1e-10))
            if rms_db > best_rms:
                best_rms = rms_db
                best_start = start
        chunk = audio[best_start : best_start + window]
        emb = identifier.get_embedding(chunk)
        return 0, 0.0, emb, [
            {
                "start_s": round(best_start / SR, 1),
                "end_s": round((best_start + window) / SR, 1),
                "rms_db": round(best_rms, 1),
                "embedding": emb,
            }
        ]

    # Compute pairwise similarities and find centroid
    n = len(windows)
    best_idx = 0
    best_score = -1.0

    for i in range(n):
        total_sim = 0.0
        for j in range(n):
            if i != j:
                total_sim += cosine_sim(windows[i]["embedding"], windows[j]["embedding"])
        avg_sim = total_sim / (n - 1) if n > 1 else 0.0
        windows[i]["centroid_score"] = round(avg_sim, 4)
        if avg_sim > best_score:
            best_score = avg_sim
            best_idx = i

    return best_idx, best_score, windows[best_idx]["embedding"], windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract centroid window embeddings from reference clips")
    parser.add_argument("--register-speaker-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir (defaults to input dir)")
    parser.add_argument("--window-s", type=float, default=WINDOW_S)
    parser.add_argument("--hop-s", type=float, default=1.0)
    parser.add_argument("--min-rms-db", type=float, default=-50.0)
    args = parser.parse_args()

    reg_dir = Path(args.register_speaker_dir)
    out_dir = Path(args.output_dir) if args.output_dir else reg_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    identifier = SpeakerIdentifier()

    print("=" * 60)
    print("Centroid Window Embedding Extraction")
    print("=" * 60)
    print(f"  Input dir:   {reg_dir}")
    print(f"  Output dir:  {out_dir}")
    print(f"  Window:      {args.window_s}s, hop: {args.hop_s}s")
    print()

    results = []

    for path in sorted(reg_dir.glob("*.npy")):
        stem = path.stem
        if stem.endswith("_live_emb") or stem.endswith("_centroid"):
            continue

        audio = np.load(path).astype(np.float32)

        best_idx, best_score, best_emb, windows = find_centroid_window(
            audio, identifier,
            window_s=args.window_s,
            hop_s=args.hop_s,
            min_rms_db=args.min_rms_db,
        )

        # Save centroid embedding
        emb_path = out_dir / f"{stem}_centroid_emb.npy"
        np.save(emb_path, best_emb)

        # Also save metadata
        meta = {
            "speaker_id": stem,
            "source_clip": str(path),
            "centroid_window_idx": best_idx,
            "centroid_window_start_s": windows[best_idx]["start_s"],
            "centroid_window_end_s": windows[best_idx]["end_s"],
            "centroid_score": round(best_score, 4),
            "centroid_rms_db": windows[best_idx]["rms_db"],
            "num_windows": len(windows),
            "all_windows": [
                {
                    "start_s": w["start_s"],
                    "end_s": w["end_s"],
                    "rms_db": w["rms_db"],
                    "centroid_score": w.get("centroid_score", None),
                }
                for w in windows
            ],
        }
        meta_path = out_dir / f"{stem}_centroid_meta.json"
        import json
        meta_path.write_text(json.dumps(meta, indent=2))

        print(f"{stem}:")
        print(f"  Centroid window: {windows[best_idx]['start_s']}-{windows[best_idx]['end_s']}s")
        print(f"  Centroid score:  {best_score:.4f} (avg sim to other windows)")
        print(f"  Window RMS:      {windows[best_idx]['rms_db']:.1f} dB")
        print(f"  Total windows:   {len(windows)}")
        print(f"  Saved: {emb_path.name}")
        print()

        results.append(meta)

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(
            f"  {r['speaker_id']:<30} "
            f"window={r['centroid_window_start_s']:.1f}-{r['centroid_window_end_s']:.1f}s  "
            f"score={r['centroid_score']:.4f}  "
            f"rms={r['centroid_rms_db']:.1f}dB"
        )


if __name__ == "__main__":
    main()

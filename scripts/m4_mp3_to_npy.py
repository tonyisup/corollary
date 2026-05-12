#!/usr/bin/env python3
"""Convert MP3 speech segments to .npy files for M4 negative training.

Extracts speech segments from MP3 files and saves them as 16kHz mono
float32 numpy arrays, ready for the M4 fine-tuning pipeline.

Usage:
  # Single segment from a podcast episode
  .venv/bin/python scripts/m4_mp3_to_npy.py \
    --input episode_42.mp3 \
    --start 120.5 \
    --duration 5.0 \
    --output-dir data/m4_human_negative_bl2021_artifacts/speaker7

  # Multiple segments from a single file (one file per segment)
  .venv/bin/python scripts/m4_mp3_to_npy.py \
    --input episode_42.mp3 \
    --segments "120.5:5.0,180.2:5.0,240.8:5.0" \
    --output-dir data/m4_human_negative_bl2021_artifacts/speaker7 \
    --prefix guest_name

  # Batch: process multiple MP3s from a directory
  .venv/bin/python scripts/m4_mp3_to_npy.py \
    --input-dir /path/to/podcasts/ \
    --segments-file segments.csv \
    --output-dir data/m4_human_negative_bl2021_artifacts

The output structure matches what m4_finetune_ecapa.py expects:
  <output-dir>/same_domain_negative_playback_window/live_captured.npy

If you provide multiple segments, each gets its own speakerN subdirectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
    import librosa
except ImportError:
    print("ERROR: Install soundfile and librosa:")
    print("  pip install soundfile librosa")
    sys.exit(1)

SR = 16000  # Corollary standard sample rate


def load_mp3_segment(
    filepath: str | Path,
    start_s: float,
    duration_s: float,
) -> np.ndarray:
    """Load a segment from an MP3 file as 16kHz mono float32 PCM.

    Uses librosa for decoding (handles MP3 via audioread/ffmpeg).
    """
    filepath = str(filepath)
    y, sr = librosa.load(
        filepath,
        sr=SR,
        mono=True,
        offset=start_s,
        duration=duration_s,
        dtype=np.float32,
    )
    return y


def save_as_npy(
    pcm: np.ndarray,
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """Save PCM as .npy in the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"speech_{ts}.npy"

    path = output_dir / filename
    np.save(path, pcm.astype(np.float32))
    return path


def build_artifact_structure(
    output_dir: str | Path,
    segment_pcms: list[np.ndarray],
    speaker_label: str | None = None,
) -> dict:
    """Build the same-domain-negative artifact structure that the fine-tuning
    pipeline expects.

    Creates:
      <output_dir>/same_domain_negative_playback_window/live_captured.npy

    If multiple segments are provided, they are concatenated into one file.
    """
    output_dir = Path(output_dir)
    scenario_dir = output_dir / "same_domain_negative_playback_window"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    if len(segment_pcms) == 1:
        combined = segment_pcms[0]
    else:
        combined = np.concatenate(segment_pcms)

    live_path = scenario_dir / "live_captured.npy"
    np.save(live_path, combined.astype(np.float32))

    # Write a minimal metadata file
    meta = {
        "scenario": "same_domain_negative_playback_window",
        "description": "Pre-recorded human guest speech (negative for M4 fine-tuning)",
        "sr": SR,
        "samples": len(combined),
        "duration_s": round(len(combined) / SR, 3),
        "rms": round(float(np.sqrt(np.mean(combined ** 2))), 6),
        "peak": round(float(np.max(np.abs(combined))), 6),
        "speech_verdict": "WORLD",
        "dry_run": False,
        "speaker_label": speaker_label or "unknown",
        "source_segments": len(segment_pcms),
    }
    meta_path = scenario_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Write a minimal frame_records.json (empty-compatible for replay)
    # Note: the fine-tuner loads live_captured.npy directly, so this is for
    # structural completeness only.
    frame_path = scenario_dir / "frame_records.json"
    frame_path.write_text("[]")

    return meta


def parse_segments_arg(segments_str: str) -> list[tuple[float, float]]:
    """Parse 'start:duration,start:duration' format."""
    segments = []
    for part in segments_str.split(","):
        part = part.strip()
        if ":" in part:
            start_str, dur_str = part.split(":", 1)
            segments.append((float(start_str.strip()), float(dur_str.strip())))
        else:
            # Just a start time, default to 5s
            segments.append((float(part.strip()), 5.0))
    return segments


def parse_segments_csv(filepath: str | Path) -> list[dict]:
    """Parse a CSV file with columns: mp3_path, start_s, duration_s [, speaker_label].

    Example CSV:
        mp3_path,start_s,duration_s,speaker_label
        /podcasts/ep42.mp3,120.5,5.0,dr_smith
        /podcasts/ep42.mp3,180.2,5.0,dr_smith
        /podcasts/ep99.mp3,45.0,5.0,prof_jones
    """
    rows = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "mp3_path": row["mp3_path"].strip(),
                "start_s": float(row["start_s"].strip()),
                "duration_s": float(row.get("duration_s", "5.0").strip()),
                "speaker_label": row.get("speaker_label", "").strip(),
            })
    return rows


def check_audio_health(pcm: np.ndarray, label: str = "") -> list[str]:
    """Return a list of warnings for an audio segment."""
    warnings = []
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    peak = float(np.max(np.abs(pcm)))
    rms_db = 20 * np.log10(max(rms, 1e-9))

    if rms_db < -45:
        warnings.append(f"{label}: near-silent ({rms_db:.1f} dB)")
    if peak > 0.99:
        warnings.append(f"{label}: clipping (peak={peak:.3f})")
    if len(pcm) / SR < 2.0:
        warnings.append(f"{label}: too short ({len(pcm)/SR:.1f}s)")
    if len(pcm) / SR > 30.0:
        warnings.append(f"{label}: very long ({len(pcm)/SR:.1f}s — consider trimming)")

    # Check for near-zero energy (silence or corrupt decode)
    zero_ratio = float(np.sum(np.abs(pcm) < 1e-6)) / len(pcm)
    if zero_ratio > 0.5:
        warnings.append(f"{label}: {zero_ratio:.0%} of samples near zero — possible decode failure")

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert MP3 speech segments to .npy for M4 fine-tuning"
    )
    parser.add_argument("--input", "-i", help="Path to a single MP3 file")
    parser.add_argument("--input-dir", "-I", help="Directory of MP3s (batch mode)")
    parser.add_argument("--output-dir", "-o", default="data/m4_human_negative_bl2021_artifacts")
    parser.add_argument("--start", type=float, help="Start time in seconds (single file mode)")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds (default: 5)")
    parser.add_argument(
        "--segments",
        help="Multiple segments as 'start:duration,start:duration' (single file mode)",
    )
    parser.add_argument("--prefix", default="speaker", help="Speaker directory prefix")
    parser.add_argument("--speakers-file", help="CSV with columns: mp3_path, start_s, duration_s [, speaker_label]")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")
    args = parser.parse_args()

    if not any([args.input, args.input_dir, args.speakers_file]):
        parser.print_help()
        print("\nERROR: Provide --input, --input-dir, or --speakers-file")
        return 2

    all_warnings = []
    speakers_processed = 0

    if args.speakers_file:
        # Batch mode from CSV
        rows = parse_segments_csv(args.speakers_file)
        # Group by speaker
        by_speaker: dict[str, list] = {}
        for row in rows:
            label = row["speaker_label"] or row["mp3_path"].split("/")[-1].replace(".mp3", "")
            by_speaker.setdefault(label, []).append(row)

        for speaker_label, segments in by_speaker.items():
            speaker_idx = speakers_processed + 1
            speaker_dir = Path(args.output_dir) / f"speaker{speaker_idx:02d}_{speaker_label}"
            segment_pcms = []

            for seg in segments:
                pcm = load_mp3_segment(seg["mp3_path"], seg["start_s"], seg["duration_s"])
                segment_pcms.append(pcm)
                all_warnings.extend(check_audio_health(pcm, f"{speaker_label} @ {seg['start_s']}s"))

            if not args.dry_run:
                meta = build_artifact_structure(speaker_dir, segment_pcms, speaker_label)
                print(f"  {speaker_label}: {meta['duration_s']:.1f}s → {speaker_dir}/")
            else:
                total_dur = sum(len(p) / SR for p in segment_pcms)
                print(f"  {speaker_label}: {total_dur:.1f}s (dry run)")

            speakers_processed += 1

    elif args.input and not args.segments:
        # Single file, single segment
        start = args.start if args.start is not None else 0.0
        pcm = load_mp3_segment(args.input, start, args.duration)
        all_warnings.extend(check_audio_health(pcm))

        speaker_dir = Path(args.output_dir) / f"{args.prefix}1"
        if not args.dry_run:
            meta = build_artifact_structure(speaker_dir, [pcm])
            print(f"  Saved: {meta['duration_s']:.1f}s → {speaker_dir}/")
            print(f"  RMS: {meta['rms_db']:.1f} dB, peak: {meta['peak']}")
        else:
            print(f"  Would save: {len(pcm)/SR:.1f}s to {speaker_dir}/")

        speakers_processed = 1

    elif args.input and args.segments:
        # Single file, multiple segments → multiple speaker dirs
        segments = parse_segments_arg(args.segments)
        for idx, (start, dur) in enumerate(segments):
            pcm = load_mp3_segment(args.input, start, dur)
            speaker_dir = Path(args.output_dir) / f"{args.prefix}{idx + 1:02d}"
            if not args.dry_run:
                meta = build_artifact_structure(speaker_dir, [pcm])
                all_warnings.extend(check_audio_health(pcm, f"segment {idx}"))
                print(f"  Segment {idx}: {meta['duration_s']:.1f}s (start={start}s) → {speaker_dir}/")
            else:
                print(f"  Segment {idx}: {dur}s (dry run)")
            speakers_processed += 1

    elif args.input_dir:
        # Batch: find all MP3s, extract a 5s segment from the middle of each
        mp3_files = sorted(Path(args.input_dir).glob("**/*.mp3"))
        if not mp3_files:
            print(f"No MP3 files found in {args.input_dir}")
            return 2

        for idx, mp3_path in enumerate(mp3_files):
            # Use 5s from 25% through the file (usually past intros)
            info = librosa.get_duration(path=str(mp3_path))
            start = max(0, info * 0.25)
            pcm = load_mp3_segment(str(mp3_path), start, args.duration)
            speaker_dir = Path(args.output_dir) / f"{args.prefix}{idx + 1:02d}_{mp3_path.stem[:20]}"
            if not args.dry_run:
                meta = build_artifact_structure(speaker_dir, [pcm])
                all_warnings.extend(check_audio_health(pcm, mp3_path.stem))
                print(f"  {mp3_path.stem}: {meta['duration_s']:.1f}s (start={start:.1f}s) → {speaker_dir}/")
            else:
                print(f"  {mp3_path.stem}: {args.duration}s (dry run)")
            speakers_processed += 1

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Processed {speakers_processed} speaker(s)")
    print(f"Output: {args.output_dir}")

    if all_warnings:
        print(f"\n⚠  WARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  - {w}")
    else:
        print("\n✓ No warnings")

    if args.dry_run:
        print("\n(dry run — no files written)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
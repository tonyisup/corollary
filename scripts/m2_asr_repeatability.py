#!/usr/bin/env python3
"""Measure ASR repeatability on saved M2 audio artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from m2_speech_preservation_harness import word_error_rate
from stg.asr import transcribe

EXPECTED_TEXT = "this is a test of world speech conservation"


def run_repeats(audio: np.ndarray, expected_text: str, repeats: int, model_size: str) -> list[dict[str, Any]]:
    rows = []
    for i in range(repeats):
        result = transcribe(audio, model_size=model_size)
        wer = word_error_rate(expected_text, result.text)
        rows.append({"repeat": i, "text": result.text, "wer": wer})
        print(f"repeat {i}: WER={wer:.3f} text={result.text!r}", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wers = [float(row["wer"]) for row in rows]
    texts = [str(row["text"]) for row in rows]
    return {
        "repeats": len(rows),
        "unique_texts": len(set(texts)),
        "min_wer": min(wers) if wers else None,
        "max_wer": max(wers) if wers else None,
        "mean_wer": sum(wers) / len(wers) if wers else None,
        "wer_range": (max(wers) - min(wers)) if wers else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat ASR on identical M2 artifact audio")
    parser.add_argument("audio", help="Path to observed.npy or processed.npy")
    parser.add_argument("--expected-text", default=EXPECTED_TEXT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--model-size", default="base")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    audio = np.load(audio_path).astype(np.float32)
    rows = run_repeats(audio, args.expected_text, args.repeats, args.model_size)
    out = {"audio": str(audio_path), "summary": summarize(rows), "rows": rows}
    print(json.dumps(out["summary"], indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

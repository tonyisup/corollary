#!/usr/bin/env python3
"""Reproducible hardware-repair validation for M3 streaming runs.

Reads the streaming_log.jsonl and summary.json from a live hardware run
directory, re-derives the repair-validation criteria, and writes
hardware_repair_validation.json. No hardware needed — this is a post-hoc
acceptance gate for saved live artifacts.

Usage:
  .venv/bin/python scripts/m3_streaming_repair_hardware_acceptance.py \\
    --run-dir data/m3_streaming_repair_live_20260509_072237 \\
    --output /tmp/hw_repair_verdict.json

Exit code:
  0 = all repair criteria passed
  1 = one or more criteria failed
  2 = setup/input missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def validate_repair(logs: list[dict], summary: dict) -> dict:
    """Derive repair-validation criteria from streaming logs and summary.

    Returns a dict with:
      - pass: bool
      - criteria: dict of individual bool criteria
      - timing: dict of key timestamps
      - summary: copy of the run summary
      - output_source_counts: frame counts by output source
    """
    if not logs:
        return {"pass": False, "error": "no log entries"}

    # --- Criteria derivation ---

    # interrupt_once: exactly one frame with should_interrupt=True
    interrupt_frames = [l for l in logs if l.get("should_interrupt")]
    interrupt_once = len(interrupt_frames) == 1

    # collision_repair_once: exactly one frame with collision_repair_text set
    repair_text_frames = [l for l in logs if l.get("collision_repair_text")]
    collision_repair_once = len(repair_text_frames) == 1

    # repair_audio_completed: summary says so
    repair_audio_completed = summary.get("repair_audio_completed", False)

    # repair_audio_played: at least one frame with output_source="repair"
    repair_played_frames = [l for l in logs if l.get("output_source") == "repair"]
    repair_audio_played = len(repair_played_frames) > 0

    # repair_not_during_listen: no repair frame has controller_state="LISTEN"
    repair_during_listen = [
        l for l in repair_played_frames if l.get("controller_state") == "LISTEN"
    ]
    repair_not_during_listen = len(repair_during_listen) == 0

    # original_tts_held_during_repair: tts_position_s is constant across all repair frames
    if repair_played_frames:
        tts_positions_during_repair = {
            l.get("tts_position_s") for l in repair_played_frames
        }
        original_tts_held_during_repair = len(tts_positions_during_repair) == 1
    else:
        original_tts_held_during_repair = False

    # tts_resumed_after_repair: at least one resumed frame after the last repair frame
    if repair_played_frames:
        last_repair_ts = max(l.get("timestamp_s", 0) for l in repair_played_frames)
        resumed_after = [
            l for l in logs
            if l.get("resumed") and l.get("timestamp_s", 0) > last_repair_ts
        ]
        tts_resumed_after_repair = len(resumed_after) > 0
    else:
        tts_resumed_after_repair = False

    # tts_completed: summary says so
    tts_completed = summary.get("tts_completed", False)

    # final_idle: last log frame has controller_state="IDLE"
    final_idle = logs[-1].get("controller_state") == "IDLE" if logs else False

    criteria = {
        "interrupt_once": interrupt_once,
        "collision_repair_once": collision_repair_once,
        "repair_audio_completed": repair_audio_completed,
        "repair_audio_played": repair_audio_played,
        "repair_not_during_listen": repair_not_during_listen,
        "original_tts_held_during_repair": original_tts_held_during_repair,
        "tts_resumed_after_repair": tts_resumed_after_repair,
        "tts_completed": tts_completed,
        "final_idle": final_idle,
    }

    # --- Timing extraction ---
    interrupt_s = [l["timestamp_s"] for l in interrupt_frames]
    repair_text_s = [l["timestamp_s"] for l in repair_text_frames]

    # LISTEN range: contiguous span from first to last LISTEN frame
    listen_frames = [l for l in logs if l.get("controller_state") == "LISTEN"]
    listen_range_s = (
        [listen_frames[0]["timestamp_s"], listen_frames[-1]["timestamp_s"]]
        if len(listen_frames) >= 2
        else []
    )

    # repair audio range: first to last repair output frame
    repair_audio_range_s = (
        [repair_played_frames[0]["timestamp_s"], repair_played_frames[-1]["timestamp_s"]]
        if len(repair_played_frames) >= 2
        else []
    )

    # resume timestamps
    resume_s = [l["timestamp_s"] for l in logs if l.get("resumed")]

    # first TTS frame after repair
    tts_after_repair_first_s = None
    if repair_played_frames and resume_s:
        last_repair_ts = max(l["timestamp_s"] for l in repair_played_frames)
        for l in logs:
            if (
                l.get("output_source") == "tts"
                and l.get("timestamp_s", 0) > last_repair_ts
            ):
                tts_after_repair_first_s = l["timestamp_s"]
                break

    # --- Output source counts ---
    source_counts: dict[str, int] = {}
    for l in logs:
        src = l.get("output_source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    all_passed = all(criteria.values())

    return {
        "pass": all_passed,
        "criteria": criteria,
        "summary": summary,
        "output_source_counts": source_counts,
        "timing": {
            "interrupt_s": interrupt_s,
            "repair_text_s": repair_text_s,
            "listen_range_s": listen_range_s,
            "repair_audio_range_s": repair_audio_range_s,
            "resume_s": resume_s,
            "tts_after_repair_first_s": tts_after_repair_first_s,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible M3 hardware-repair validation"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to live run directory containing streaming_log.jsonl and summary.json",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write verdict JSON (default: <run-dir>/hardware_repair_validation.json)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = run_dir / "streaming_log.jsonl"
    summary_path = run_dir / "summary.json"

    if not log_path.exists():
        print(f"ERROR: streaming_log.jsonl not found in {run_dir}", file=sys.stderr)
        return 2
    if not summary_path.exists():
        print(f"ERROR: summary.json not found in {run_dir}", file=sys.stderr)
        return 2

    logs = load_jsonl(log_path)
    with open(summary_path) as f:
        summary = json.load(f)

    verdict = validate_repair(logs, summary)
    verdict["run_dir"] = str(run_dir)

    # Write verdict
    output_path = Path(args.output) if args.output else run_dir / "hardware_repair_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    # Report
    if verdict["pass"]:
        print(f"PASS: M3 hardware-repair validation criteria met.")
        print(f"  Written to: {output_path}")
        return 0
    else:
        print(f"FAIL: M3 hardware-repair validation criteria NOT met.")
        failed = [k for k, v in verdict["criteria"].items() if not v]
        print(f"  Failed criteria: {', '.join(failed)}")
        print(f"  Written to: {output_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Analyze M3 controller-gated playback artifacts.

The live harness saves per-scenario folders containing metrics.json,
frame_records.json, and live_captured.npy. This analyzer answers the specific
post-run question: did the gated run physically suppress TTS during scheduled
human overlap windows and resume afterward?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SR = 16000
EPS = 1e-10


def compute_window_rms_db(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
    """Return RMS dBFS for a time window, clamped for empty/silent windows."""
    start = max(0, int(round(start_s * sr)))
    end = min(len(audio), int(round(end_s * sr)))
    if end <= start:
        return -200.0
    window = audio[start:end].astype(np.float32)
    rms = float(np.sqrt(np.mean(window ** 2))) if len(window) else 0.0
    return float(20.0 * np.log10(max(rms, EPS)))


def _merge_times(times: list[float], frame_s: float = 0.03, gap_s: float = 0.045) -> list[dict[str, float]]:
    if not times:
        return []
    times = sorted(times)
    windows: list[dict[str, float]] = []
    start = prev = times[0]
    for t in times[1:]:
        if t - prev <= gap_s:
            prev = t
            continue
        windows.append({"start_s": round(start, 6), "end_s": round(prev + frame_s, 6)})
        start = prev = t
    windows.append({"start_s": round(start, 6), "end_s": round(prev + frame_s, 6)})
    return windows


def human_windows_from_records(frame_records: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Find scheduled human-speech windows from annotated frame records."""
    times = [
        float(fr.get("start_s", 0.0))
        for fr in frame_records
        if fr.get("gt_human_speaking") is True
    ]
    return _merge_times(times)


def blocked_windows_from_records(frame_records: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Find scheduled human-overlap windows from annotated frame records."""
    times = [
        float(fr.get("start_s", 0.0))
        for fr in frame_records
        if fr.get("gt_human_speaking") is True and fr.get("is_in_motor_window") is True
    ]
    return _merge_times(times)


def _segment_windows(segments: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [
        {
            "start_s": float(seg.get("dst_start_s", 0.0)),
            "end_s": float(seg.get("dst_start_s", 0.0)) + float(seg.get("duration_s", 0.0)),
        }
        for seg in segments
        if float(seg.get("duration_s", 0.0)) > 0
    ]


def _max_window_rms(audio: np.ndarray, sr: int, windows: list[dict[str, float]]) -> float | None:
    if not windows:
        return None
    return max(compute_window_rms_db(audio, sr, w["start_s"], w["end_s"]) for w in windows)


def _windows_overlap(a: dict[str, float], b: dict[str, float]) -> bool:
    return a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"]


def _segments_avoid_human_windows(
    playback_windows: list[dict[str, float]],
    human_windows: list[dict[str, float]],
) -> bool | None:
    if not human_windows:
        return None
    for playback in playback_windows:
        for human in human_windows:
            if _windows_overlap(playback, human):
                return False
    return True


def _analyze_scenario(path: Path, sr: int, suppression_margin_threshold_db: float) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    records_path = path / "frame_records.json"
    audio_path = path / "live_captured.npy"

    metrics = json.loads(metrics_path.read_text())
    frame_records = json.loads(records_path.read_text()) if records_path.exists() else []
    audio = np.load(audio_path).astype(np.float32) if audio_path.exists() else np.zeros(0, dtype=np.float32)

    expected_behavior = metrics.get("expected_behavior")
    segments = metrics.get("tts_segments", []) or []
    playback_windows = _segment_windows(segments)
    human_windows = human_windows_from_records(frame_records)
    blocked_windows = (
        blocked_windows_from_records(frame_records)
        if expected_behavior == "interrupt_defer_resume"
        else []
    )

    playback_rms_db = _max_window_rms(audio, sr, playback_windows)
    blocked_rms_db = _max_window_rms(audio, sr, blocked_windows)
    segment_overlap_passed = (
        _segments_avoid_human_windows(playback_windows, human_windows)
        if expected_behavior in {"defer", "interrupt_defer_resume"}
        else None
    )
    suppression_margin_db = None
    if playback_rms_db is not None and blocked_rms_db is not None:
        suppression_margin_db = playback_rms_db - blocked_rms_db

    notes: list[str] = []
    if not metrics.get("gated_playback", False):
        notes.append("gated_playback is false")
    if not segments:
        notes.append("missing tts_segments")
    if not audio_path.exists():
        notes.append("missing live_captured.npy")
    if not blocked_windows and expected_behavior == "interrupt_defer_resume":
        notes.append("no blocked human-overlap windows found")

    defer_passed = None
    if expected_behavior == "defer":
        if not segments:
            defer_passed = False
        elif human_windows:
            last_human_end_s = max(w["end_s"] for w in human_windows)
            first_tts_start_s = min(float(seg.get("dst_start_s", 0.0)) for seg in segments)
            defer_passed = first_tts_start_s >= last_human_end_s
            if not defer_passed:
                notes.append("first TTS segment starts before human speech ends")
        else:
            defer_passed = bool(segments)

    suppression_passed = bool(metrics.get("gated_playback", False) and segments)
    if segment_overlap_passed is not None:
        suppression_passed = suppression_passed and segment_overlap_passed
    elif blocked_windows:
        suppression_passed = suppression_passed and suppression_margin_db is not None
        suppression_passed = suppression_passed and suppression_margin_db >= suppression_margin_threshold_db
    if defer_passed is not None:
        suppression_passed = suppression_passed and defer_passed

    return {
        "scenario": metrics.get("scenario", path.name),
        "path": str(path),
        "gated_playback": bool(metrics.get("gated_playback", False)),
        "false_interrupts": int(metrics.get("false_interrupts", 0)),
        "false_defers": int(metrics.get("false_defers", 0)),
        "missed_interrupts": int(metrics.get("missed_interrupts", 0)),
        "true_interrupts": int(metrics.get("true_interrupts", 0)),
        "resume_rate": metrics.get("resume_rate", "n/a"),
        "tts_segment_count": len(segments),
        "tts_segments": segments,
        "human_window_count": len(human_windows),
        "human_windows": human_windows,
        "blocked_window_count": len(blocked_windows),
        "blocked_windows": blocked_windows,
        "playback_rms_db": None if playback_rms_db is None else round(playback_rms_db, 2),
        "blocked_rms_db": None if blocked_rms_db is None else round(blocked_rms_db, 2),
        "suppression_margin_db": None if suppression_margin_db is None else round(suppression_margin_db, 2),
        "segment_overlap_passed": segment_overlap_passed,
        "defer_passed": defer_passed,
        "suppression_passed": bool(suppression_passed),
        "notes": "; ".join(notes),
    }


def analyze_run(
    run_dir: str | Path,
    sr: int = DEFAULT_SR,
    suppression_margin_threshold_db: float = 12.0,
) -> dict[str, Any]:
    """Analyze a gated playback run directory."""
    root = Path(run_dir)
    scenario_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "metrics.json").exists())
    scenarios = [_analyze_scenario(p, sr, suppression_margin_threshold_db) for p in scenario_dirs]
    totals = {
        "false_interrupts": sum(s["false_interrupts"] for s in scenarios),
        "false_defers": sum(s["false_defers"] for s in scenarios),
        "missed_interrupts": sum(s["missed_interrupts"] for s in scenarios),
        "true_interrupts": sum(s["true_interrupts"] for s in scenarios),
    }
    return {
        "run_dir": str(root),
        "scenario_count": len(scenarios),
        "suppression_margin_threshold_db": suppression_margin_threshold_db,
        "totals": totals,
        "all_suppression_passed": all(s["suppression_passed"] for s in scenarios) if scenarios else False,
        "scenarios": scenarios,
    }


def _print_text_report(summary: dict[str, Any]) -> None:
    print(f"Run: {summary['run_dir']}")
    print(f"Scenarios: {summary['scenario_count']}")
    print(f"Totals: {summary['totals']}")
    print(f"All suppression passed: {summary['all_suppression_passed']}")
    print("")
    for sc in summary["scenarios"]:
        print(
            f"{sc['scenario']}: gated={sc['gated_playback']} "
            f"FI={sc['false_interrupts']} FD={sc['false_defers']} "
            f"MI={sc['missed_interrupts']} TI={sc['true_interrupts']} "
            f"resume={sc['resume_rate']} segments={sc['tts_segment_count']} "
            f"blocked={sc['blocked_window_count']} margin={sc['suppression_margin_db']} "
            f"pass={sc['suppression_passed']}"
        )
        if sc["notes"]:
            print(f"  notes: {sc['notes']}")
        for seg in sc["tts_segments"]:
            print(
                "  segment src={src_start_s:.2f}s dst={dst_start_s:.2f}s dur={duration_s:.2f}s".format(**seg)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze M3 gated playback artifacts")
    parser.add_argument("run_dir", help="Directory produced by m3_live_e2e_harness.py --save-dir")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="Analysis sample rate for live_captured.npy")
    parser.add_argument("--suppression-margin-threshold-db", type=float, default=12.0)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text report")
    parser.add_argument("--output", help="Optional path to write summary JSON")
    args = parser.parse_args()

    summary = analyze_run(
        args.run_dir,
        sr=args.sr,
        suppression_margin_threshold_db=args.suppression_margin_threshold_db,
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_text_report(summary)


if __name__ == "__main__":
    main()

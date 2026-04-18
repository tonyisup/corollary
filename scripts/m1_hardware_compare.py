#!/usr/bin/env python3
"""Repeated hardware-facing M1A probe comparison runner.

Runs a bounded alternating comparison between primitive probe stimuli on a real
speaker+mic path, then prints a compact summary and appends JSONL results.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from statistics import mean, median, pstdev
from typing import Any

from m1_probe_capture import run_probe, run_paired_probe

DEFAULT_RESULTS_PATH = os.path.join("results", "m1_hardware_compare.jsonl")


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(rows: list[dict[str, Any]], paired: bool = False) -> dict[str, Any]:
    if not rows:
        return {}
    if paired:
        match_scores = [float(r["mean_match_score"]) for r in rows]
        onset_errors = [float(r["mean_onset_error_ms"]) for r in rows]
        envelope_onset_errors = [float(r["mean_envelope_onset_error_ms"]) for r in rows]
        quality_key = "paired_quality"
        strong_match_runs = sum(1 for x in match_scores if x >= 0.5)
        visible_match_runs = sum(1 for x in match_scores if x >= 0.2)
        stable_onset_runs = sum(1 for x in onset_errors if x <= 50.0)
        good_runs = sum(1 for r in rows if r.get(quality_key) == "good")
        borderline_runs = sum(1 for r in rows if r.get(quality_key) == "borderline")
        bad_runs = sum(1 for r in rows if r.get(quality_key) == "bad")
        return {
            "runs": len(rows),
            "good_runs": good_runs,
            "borderline_runs": borderline_runs,
            "bad_runs": bad_runs,
            "mean_match_score": round(mean(match_scores), 6),
            "median_match_score": round(median(match_scores), 6),
            "match_score_std": round(pstdev(match_scores), 6) if len(match_scores) > 1 else 0.0,
            "best_match_score": round(max(match_scores), 6),
            "strong_match_runs": strong_match_runs,
            "visible_match_runs": visible_match_runs,
            "mean_onset_error_ms": round(mean(onset_errors), 4),
            "median_onset_error_ms": round(median(onset_errors), 4),
            "onset_error_std_ms": round(pstdev(onset_errors), 4) if len(onset_errors) > 1 else 0.0,
            "best_onset_error_ms": round(min(onset_errors), 4),
            "mean_envelope_onset_error_ms": round(mean(envelope_onset_errors), 4),
            "median_envelope_onset_error_ms": round(median(envelope_onset_errors), 4),
            "envelope_onset_error_std_ms": round(pstdev(envelope_onset_errors), 4) if len(envelope_onset_errors) > 1 else 0.0,
            "best_envelope_onset_error_ms": round(min(envelope_onset_errors), 4),
            "stable_onset_runs": stable_onset_runs,
        }

    match_scores = [float(r["match_score"]) for r in rows]
    onset_errors = [float(r["onset_error_ms"]) for r in rows]
    capture_rms = [float(r["capture_rms_db"]) for r in rows]
    usable_rows = [r for r in rows if r.get("quality") != "bad"]
    usable_match_scores = [float(r["match_score"]) for r in usable_rows]
    usable_onset_errors = [float(r["onset_error_ms"]) for r in usable_rows]
    strong_match_runs = sum(1 for x in match_scores if x >= 0.5)
    visible_match_runs = sum(1 for x in match_scores if x >= 0.2)
    stable_onset_runs = sum(1 for x in onset_errors if x <= 50.0)
    good_runs = sum(1 for r in rows if r.get("quality") == "good")
    borderline_runs = sum(1 for r in rows if r.get("quality") == "borderline")
    bad_runs = sum(1 for r in rows if r.get("quality") == "bad")
    return {
        "runs": len(rows),
        "good_runs": good_runs,
        "borderline_runs": borderline_runs,
        "bad_runs": bad_runs,
        "mean_match_score": round(mean(match_scores), 6),
        "median_match_score": round(median(match_scores), 6),
        "match_score_std": round(pstdev(match_scores), 6) if len(match_scores) > 1 else 0.0,
        "best_match_score": round(max(match_scores), 6),
        "usable_mean_match_score": round(mean(usable_match_scores), 6) if usable_match_scores else None,
        "strong_match_runs": strong_match_runs,
        "visible_match_runs": visible_match_runs,
        "mean_onset_error_ms": round(mean(onset_errors), 4),
        "median_onset_error_ms": round(median(onset_errors), 4),
        "onset_error_std_ms": round(pstdev(onset_errors), 4) if len(onset_errors) > 1 else 0.0,
        "best_onset_error_ms": round(min(onset_errors), 4),
        "usable_mean_onset_error_ms": round(mean(usable_onset_errors), 4) if usable_onset_errors else None,
        "stable_onset_runs": stable_onset_runs,
        "mean_capture_rms_db": round(mean(capture_rms), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated hardware M1A probe comparisons")
    parser.add_argument("--input-device", type=int, required=True)
    parser.add_argument("--output-device", type=int, required=True)
    parser.add_argument("--stimuli", nargs="+", default=["chirp", "vowel_ah"])
    parser.add_argument("--stimulus-seed", type=int, default=0)
    parser.add_argument("--setup-label", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration-ms", type=int, default=300)
    parser.add_argument("--amplitude", type=float, default=0.7)
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.0)
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--pair-gap-s", type=float, default=0.35)
    args = parser.parse_args()

    ordered: list[str] = []
    for _ in range(args.repeats):
        ordered.extend(args.stimuli)

    rows_by_stimulus: dict[str, list[dict[str, Any]]] = {s: [] for s in args.stimuli}

    print("\nRunning hardware probe comparison:\n")
    print(f"  input device:  {args.input_device}")
    print(f"  output device: {args.output_device}")
    print(f"  stimuli:       {args.stimuli}")
    print(f"  repeats:       {args.repeats}\n")

    for idx, stimulus in enumerate(ordered, start=1):
        print(f"=== run {idx}/{len(ordered)}: {stimulus} ===")
        row = (
            run_paired_probe(
                stimulus=stimulus,
                duration_ms=args.duration_ms,
                amplitude=args.amplitude,
                input_device=args.input_device,
                output_device=args.output_device,
                lead_in_s=args.lead_in_s,
                tail_s=args.tail_s,
                gap_s=args.pair_gap_s,
                stimulus_seed=args.stimulus_seed,
                setup_label=args.setup_label,
            )
            if args.paired
            else run_probe(
                stimulus=stimulus,
                duration_ms=args.duration_ms,
                amplitude=args.amplitude,
                input_device=args.input_device,
                output_device=args.output_device,
                lead_in_s=args.lead_in_s,
                tail_s=args.tail_s,
                stimulus_seed=args.stimulus_seed,
                setup_label=args.setup_label,
            )
        )
        row.update({
            "logged_at": time.time(),
            "input_device": args.input_device,
            "output_device": args.output_device,
            "duration_ms": args.duration_ms,
            "amplitude": args.amplitude,
            "lead_in_s": args.lead_in_s,
            "tail_s": args.tail_s,
            "stimulus_seed": args.stimulus_seed,
            "setup_label": args.setup_label,
        })
        rows_by_stimulus[stimulus].append(row)
        append_jsonl(args.results, row)
        if args.paired:
            print(
                f"summary: mean_match={row['mean_match_score']:.4f} mean_onset_error_ms={row['mean_onset_error_ms']:.2f} "
                f"mean_envelope_onset_error_ms={row['mean_envelope_onset_error_ms']:.2f} paired_quality={row.get('paired_quality', 'unknown')}"
            )
        else:
            print(
                f"summary: match={row['match_score']:.4f} onset_error_ms={row['onset_error_ms']:.2f} "
                f"capture_rms_db={row['capture_rms_db']:.2f} quality={row.get('quality', 'unknown')}"
            )
        print()

    print("Final summary:\n")
    summaries = {stimulus: summarize(rows, paired=args.paired) for stimulus, rows in rows_by_stimulus.items()}
    for stimulus, stats in summaries.items():
        print(f"{stimulus}: {stats}")

    if len(args.stimuli) >= 2:
        ranked = sorted(
            ((stimulus, stats) for stimulus, stats in summaries.items()),
            key=lambda item: (
                item[1].get("good_runs", 0),
                item[1].get("visible_match_runs", 0),
                item[1].get("stable_onset_runs", 0),
                item[1].get("median_match_score", 0.0),
                -item[1].get("median_onset_error_ms", 1e9),
            ),
            reverse=True,
        )
        print("\nCurrent ranking:")
        for i, (stimulus, stats) in enumerate(ranked, start=1):
            print(
                f"  {i}. {stimulus} | good={stats.get('good_runs', 0)} borderline={stats.get('borderline_runs', 0)} bad={stats.get('bad_runs', 0)} "
                f"visible={stats.get('visible_match_runs', 0)}/{stats.get('runs', 0)} "
                f"stable_onset={stats.get('stable_onset_runs', 0)}/{stats.get('runs', 0)} "
                f"median_match={stats.get('median_match_score', 0.0):.4f} "
                f"median_onset_error_ms={stats.get('median_onset_error_ms', 0.0):.2f}"
            )

        top_stimulus, top_stats = ranked[0]
        print("\nRecommendation:")
        if top_stats.get("good_runs", 0) > 0:
            print(
                f"  Current best hardware-facing probe on this setup: {top_stimulus} "
                f"(good_runs={top_stats.get('good_runs', 0)}, median_match={top_stats.get('median_match_score', 0.0):.4f}, "
                f"median_onset_error_ms={top_stats.get('median_onset_error_ms', 0.0):.2f})"
            )
        else:
            print(
                f"  No clearly good hardware-facing probe yet. Current least-bad candidate: {top_stimulus}"
            )


if __name__ == "__main__":
    main()

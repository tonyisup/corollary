#!/usr/bin/env python3
"""Deterministic acceptance check for M4 captured replay validation.

Runs the captured-audio replay F1 benchmark with a clean self-capture
registration and fine-tuned scorer, then emits a compact pass/fail JSON verdict.
No hardware is required; this is the replay gate for existing M4 capture artifacts.

Usage:
  .venv/bin/python scripts/m4_capture_replay_acceptance.py \
    --replay-dir data/m4_capture_validation_heldout_20260509 \
    --register-from-clean data/m4_clean_captures_fixed \
    --fine-tuned-scorer data/m4_finetuned_ecapa_aug/fine_tuned/scorer.pt \
    --output /tmp/m4_capture_replay_acceptance.json

Exit code:
  0 = captured replay acceptance criteria met
  1 = one or more acceptance criteria failed
  2 = invalid inputs / setup error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_f1_benchmark import run_benchmark  # noqa: E402


REQUIRED_CASES = {
    "replay_self_only",
    "replay_same_domain_negative_playback_window",
}


def _case_by_name(result: Any) -> dict[str, dict]:
    return {case.get("name", ""): case for case in getattr(result, "cases", [])}


def build_verdict(
    result: Any,
    *,
    replay_dir: str,
    register_from_clean: str,
    fine_tuned_scorer: str,
    threshold: float,
    min_aggregate_f1: float,
    min_eligible_f1: float,
    max_negative_fp: int,
) -> dict:
    """Build compact acceptance verdict from a BenchmarkResult-like object."""
    cases = _case_by_name(result)
    required_cases_present = REQUIRED_CASES.issubset(cases.keys())
    negative = cases.get("replay_same_domain_negative_playback_window", {})

    aggregate_f1 = float(getattr(result, "aggregate_f1", 0.0))
    eligible_f1 = float(getattr(result, "aggregate_eligible_f1", 0.0))
    negative_fp = int(negative.get("fp", 10**9))
    negative_eligible_fp = int(negative.get("eligible_fp", 10**9))

    criteria = {
        "required_cases_present": required_cases_present,
        "aggregate_f1_passed": aggregate_f1 >= min_aggregate_f1,
        "eligible_f1_passed": eligible_f1 >= min_eligible_f1,
        "negative_fp_passed": negative_fp <= max_negative_fp,
        "negative_eligible_fp_passed": negative_eligible_fp <= max_negative_fp,
    }

    verdict = {
        "replay_dir": replay_dir,
        "register_from_clean": register_from_clean,
        "fine_tuned_scorer": fine_tuned_scorer,
        "threshold": threshold,
        "acceptance_thresholds": {
            "min_aggregate_f1": min_aggregate_f1,
            "min_eligible_f1": min_eligible_f1,
            "max_negative_fp": max_negative_fp,
        },
        "aggregate_f1": aggregate_f1,
        "aggregate_eligible_f1": eligible_f1,
        "negative_fp": negative_fp,
        "negative_eligible_fp": negative_eligible_fp,
        "cases": list(getattr(result, "cases", [])),
        "acceptance_criteria": criteria,
    }
    verdict["pass"] = all(criteria.values())
    return verdict


def run_acceptance(
    *,
    replay_dir: str,
    register_from_clean: str,
    fine_tuned_scorer: str,
    threshold: float = 0.70,
    m4_min_buffer_ms: float = 300.0,
    min_aggregate_f1: float = 0.95,
    min_eligible_f1: float = 0.95,
    max_negative_fp: int = 0,
) -> dict:
    """Run the M4 captured replay benchmark and return compact verdict."""
    result = run_benchmark(
        threshold=threshold,
        real_ecapa=True,
        m4_min_buffer_ms=m4_min_buffer_ms,
        replay_dir=replay_dir,
        register_from_clean=register_from_clean,
        backend="ecapa",
        fine_tuned_scorer=fine_tuned_scorer,
    )
    return build_verdict(
        result,
        replay_dir=replay_dir,
        register_from_clean=register_from_clean,
        fine_tuned_scorer=fine_tuned_scorer,
        threshold=threshold,
        min_aggregate_f1=min_aggregate_f1,
        min_eligible_f1=min_eligible_f1,
        max_negative_fp=max_negative_fp,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic M4 captured replay acceptance check"
    )
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--register-from-clean", required=True)
    parser.add_argument("--fine-tuned-scorer", required=True)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--m4-min-buffer-ms", type=float, default=300.0)
    parser.add_argument("--min-aggregate-f1", type=float, default=0.95)
    parser.add_argument("--min-eligible-f1", type=float, default=0.95)
    parser.add_argument("--max-negative-fp", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()

    for path_arg, label in (
        (args.replay_dir, "replay_dir"),
        (args.register_from_clean, "register_from_clean"),
        (args.fine_tuned_scorer, "fine_tuned_scorer"),
    ):
        if not Path(path_arg).exists():
            print(f"ERROR: {label} not found: {path_arg}", file=sys.stderr)
            return 2

    verdict = run_acceptance(
        replay_dir=args.replay_dir,
        register_from_clean=args.register_from_clean,
        fine_tuned_scorer=args.fine_tuned_scorer,
        threshold=args.threshold,
        m4_min_buffer_ms=args.m4_min_buffer_ms,
        min_aggregate_f1=args.min_aggregate_f1,
        min_eligible_f1=args.min_eligible_f1,
        max_negative_fp=args.max_negative_fp,
    )

    output_json = json.dumps(verdict, indent=2)
    print(output_json)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output_json)

    if verdict["pass"]:
        print("\nPASS: M4 captured replay acceptance criteria met.", file=sys.stderr)
        return 0

    failed = [k for k, v in verdict["acceptance_criteria"].items() if not v]
    print("\nFAIL: M4 captured replay acceptance criteria NOT met.", file=sys.stderr)
    print(f"  Failed checks: {', '.join(failed)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Deterministic acceptance check for M3 controller-gated playback.

Re-runs the gated playback analyzer on a saved artifact directory
and emits a compact pass/fail JSON verdict. No hardware required.

Usage:
  .venv/bin/python scripts/m3_gated_playback_acceptance.py \\
    --run-dir data/m3_gated_playback_test \\
    --output /tmp/m3_gated_acceptance_verdict.json

Exit code:
  0 = all suppression checks passed
  1 = one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_gated_playback_analyze import analyze_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic M3 gated playback acceptance check"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Saved gated playback run directory (from m3_live_e2e_harness.py --save-dir)",
    )
    parser.add_argument(
        "--suppression-margin-threshold-db",
        type=float,
        default=12.0,
        help="Minimum playback-vs-blocked RMS margin in dB (default: 12.0)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the compact verdict JSON",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run_dir not found: {run_dir}", file=sys.stderr)
        return 2

    summary = analyze_run(
        run_dir,
        suppression_margin_threshold_db=args.suppression_margin_threshold_db,
    )

    # Build compact verdict
    all_passed = bool(summary.get("all_suppression_passed"))
    scenario_verdicts = []
    for sc in summary.get("scenarios", []):
        v = {
            "scenario": sc["scenario"],
            "gated_playback": sc["gated_playback"],
            "suppression_passed": sc["suppression_passed"],
            "segment_overlap_passed": sc["segment_overlap_passed"],
            "defer_passed": sc["defer_passed"],
            "tts_segment_count": sc["tts_segment_count"],
            "notes": sc["notes"],
        }
        scenario_verdicts.append(v)

    verdict = {
        "run_dir": str(run_dir),
        "all_suppression_passed": all_passed,
        "scenario_count": summary["scenario_count"],
        "scenarios": scenario_verdicts,
        "acceptance_criteria": {
            "all_suppression_passed": all_passed,
            "all_gated_playback": all(
                v["gated_playback"] for v in scenario_verdicts
            ),
            "all_deferred_or_non_overlapping": all(
                v["defer_passed"] is None or v["defer_passed"]
                for v in scenario_verdicts
            ),
            "all_segment_overlap_passed_or_na": all(
                v["segment_overlap_passed"] is None
                or v["segment_overlap_passed"]
                for v in scenario_verdicts
            ),
        },
    }
    verdict["pass"] = all(
        verdict["acceptance_criteria"].values()
    )

    output_json = json.dumps(verdict, indent=2)
    print(output_json)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output_json)

    if verdict["pass"]:
        print("\nPASS: M3 gated playback acceptance criteria met.", file=sys.stderr)
        return 0
    else:
        print("\nFAIL: M3 gated playback acceptance criteria NOT met.", file=sys.stderr)
        failed = [
            k for k, v in verdict["acceptance_criteria"].items() if not v
        ]
        print(f"  Failed checks: {', '.join(failed)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

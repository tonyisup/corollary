#!/usr/bin/env python3
"""Multi-artifact M4 captured replay acceptance.

Runs the M4 captured replay acceptance gate across one or more artifact
directories and emits a compact per-artifact + aggregate verdict. This is
the promotion gate for broader held-out capture validation across room/device
conditions.

Usage:
  .venv/bin/python scripts/m4_multi_artifact_acceptance.py \
    --artifact-dirs \
      data/m4_capture_validation_heldout_20260509 \
      data/m4_capture_validation_room2_20260510 \
    --register-from-clean data/m4_clean_captures_fixed \
    --fine-tuned-scorer data/m4_finetuned_ecapa_aug/fine_tuned/scorer.pt \
    --output /tmp/m4_multi_artifact_acceptance.json

  # Also include the synthetic dry-run battery:
  .venv/bin/python scripts/m4_multi_artifact_acceptance.py \
    --artifact-dirs data/m4_capture_validation_heldout_20260509 \
    --include-synthetic \
    --register-from-clean data/m4_clean_captures_fixed \
    --fine-tuned-scorer data/m4_finetuned_ecapa_aug/fine_tuned/scorer.pt \
    --output /tmp/m4_multi_artifact_acceptance.json

Exit code:
  0 = all artifacts pass acceptance criteria
  1 = one or more artifacts failed
  2 = invalid inputs / setup error
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_capture_replay_acceptance import run_acceptance  # noqa: E402
from m4_capture_validation import create_dry_run_artifact  # noqa: E402


def run_multi_artifact_acceptance(
    *,
    artifact_dirs: list[str],
    register_from_clean: str,
    fine_tuned_scorer: str,
    threshold: float = 0.70,
    m4_min_buffer_ms: float = 300.0,
    min_aggregate_f1: float = 0.95,
    min_eligible_f1: float = 0.95,
    max_negative_fp: int = 0,
    include_synthetic: bool = False,
) -> dict:
    """Run acceptance gate across multiple artifact directories.

    Each artifact dir is expected to contain replay-compatible scenario
    subdirectories. The gate checks:
    - required cases present (at least one self + one negative scenario)
    - eligible-frame F1 >= min_eligible_f1
    - total negative FP <= max_negative_fp

    Aggregate F1 is reported as diagnostic but is not part of the pass gate.

    When include_synthetic is True, a synthetic dry-run battery is prepended.
    Synthetic artifacts are reported separately and do not affect the overall
    pass/fail gate (the ECAPA fine-tuned scorer may not generalize to synthetic
    room/device transforms).
    """
    results = []

    # Optionally prepend a synthetic dry-run battery (informational only)
    synthetic_results = []
    if include_synthetic:
        synthetic_dir = Path(tempfile.mkdtemp(prefix="m4_synthetic_"))
        create_dry_run_artifact(
            output_dir=synthetic_dir,
            include_variations=True,
        )
        # Run synthetic acceptance but don't let it fail the gate
        try:
            synth_verdict = run_acceptance(
                replay_dir=str(synthetic_dir),
                register_from_clean=register_from_clean,
                fine_tuned_scorer=fine_tuned_scorer,
                threshold=threshold,
                m4_min_buffer_ms=m4_min_buffer_ms,
                min_aggregate_f1=min_aggregate_f1,
                min_eligible_f1=min_eligible_f1,
                max_negative_fp=max_negative_fp,
            )
            synthetic_results.append({
                "artifact_dir": str(synthetic_dir),
                "pass": synth_verdict["pass"],
                "aggregate_f1": synth_verdict["aggregate_f1"],
                "aggregate_eligible_f1": synth_verdict["aggregate_eligible_f1"],
                "negative_fp": synth_verdict["negative_fp"],
                "acceptance_criteria": synth_verdict["acceptance_criteria"],
                "cases": synth_verdict["cases"],
                "simulated": True,
                "gate_exempt": True,
            })
        except Exception as e:
            synthetic_results.append({
                "artifact_dir": str(synthetic_dir),
                "error": str(e),
                "pass": False,
                "simulated": True,
                "gate_exempt": True,
            })

    for artifact_dir in artifact_dirs:
        try:
            verdict = run_acceptance(
                replay_dir=artifact_dir,
                register_from_clean=register_from_clean,
                fine_tuned_scorer=fine_tuned_scorer,
                threshold=threshold,
                m4_min_buffer_ms=m4_min_buffer_ms,
                min_aggregate_f1=min_aggregate_f1,
                min_eligible_f1=min_eligible_f1,
                max_negative_fp=max_negative_fp,
            )
            results.append({
                "artifact_dir": artifact_dir,
                "pass": verdict["pass"],
                "aggregate_f1": verdict["aggregate_f1"],
                "aggregate_eligible_f1": verdict["aggregate_eligible_f1"],
                "negative_fp": verdict["negative_fp"],
                "acceptance_criteria": verdict["acceptance_criteria"],
                "cases": verdict["cases"],
                "simulated": False,
            })
        except Exception as e:
            results.append({
                "artifact_dir": artifact_dir,
                "error": str(e),
                "pass": False,
            })

    all_passed = all(r.get("pass", False) for r in results)
    any_errors = any("error" in r for r in results)

    return {
        "artifact_count": len(results),
        "all_passed": all_passed,
        "any_errors": any_errors,
        "threshold": threshold,
        "acceptance_thresholds": {
            "min_aggregate_f1": min_aggregate_f1,
            "min_eligible_f1": min_eligible_f1,
            "max_negative_fp": max_negative_fp,
        },
        "results": results,
        "synthetic_results": synthetic_results,
        "pass": all_passed and not any_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-artifact M4 captured replay acceptance gate"
    )
    parser.add_argument(
        "--artifact-dirs",
        nargs="+",
        required=True,
        help="One or more captured validation artifact directories",
    )
    parser.add_argument("--register-from-clean", required=True)
    parser.add_argument("--fine-tuned-scorer", required=True)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--m4-min-buffer-ms", type=float, default=300.0)
    parser.add_argument("--min-aggregate-f1", type=float, default=0.95)
    parser.add_argument("--min-eligible-f1", type=float, default=0.95)
    parser.add_argument("--max-negative-fp", type=int, default=0)
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Prepend a synthetic dry-run battery to the artifact list",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    for path_arg, label in (
        (args.register_from_clean, "register_from_clean"),
        (args.fine_tuned_scorer, "fine_tuned_scorer"),
    ):
        if not Path(path_arg).exists():
            print(f"ERROR: {label} not found: {path_arg}", file=sys.stderr)
            return 2

    verdict = run_multi_artifact_acceptance(
        artifact_dirs=args.artifact_dirs,
        register_from_clean=args.register_from_clean,
        fine_tuned_scorer=args.fine_tuned_scorer,
        threshold=args.threshold,
        m4_min_buffer_ms=args.m4_min_buffer_ms,
        min_aggregate_f1=args.min_aggregate_f1,
        min_eligible_f1=args.min_eligible_f1,
        max_negative_fp=args.max_negative_fp,
        include_synthetic=args.include_synthetic,
    )

    output_json = json.dumps(verdict, indent=2)
    print(output_json)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output_json)

    if verdict["pass"]:
        print(
            f"\nPASS: All {verdict['artifact_count']} artifact(s) passed M4 multi-artifact acceptance.",
            file=sys.stderr,
        )
        return 0

    failed = [r for r in verdict["results"] if not r.get("pass", False)]
    print(
        f"\nFAIL: {len(failed)} of {verdict['artifact_count']} artifact(s) failed.",
        file=sys.stderr,
    )
    for f in failed:
        reason = f.get("error", "acceptance criteria not met")
        print(f"  {f['artifact_dir']}: {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

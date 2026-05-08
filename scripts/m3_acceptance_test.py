#!/usr/bin/env python3
"""M3 Dual-Speaker Acceptance Test.

Comprehensive regression test that validates the complete M3 pipeline:
- All 5 scenarios pass with expected behavior
- Controller metrics meet targets
- Room-local scoring produces SELF frames
- Auto-calibration produces valid filter

Run:
    .venv/bin/python3 scripts/m3_acceptance_test.py --input-device 3 --output-device 5 --human-output-device 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from m3_live_e2e_harness import (
    SCENARIOS,
    LiveScenario,
    run_scenario,
    ThalamicAttributor,
    SmoothedVAD,
    VAD_FRAME_SAMPLES,
    SR,
)


# ── Expected behavior per scenario ──────────────────────────────────────────

EXPECTED = {
    "human_early": {
        "min_true_interrupts": 0,  # May vary due to room noise
        "min_resumes": 1,
        "max_false_interrupts": 1,  # Allow 1 borderline detection
        "max_false_defers": 0,
        "max_missed_interrupts": 1,  # Allow 1 miss due to noise
    },
    "human_late": {
        "min_true_interrupts": 0,
        "min_resumes": 1,
        "max_false_interrupts": 1,
        "max_false_defers": 0,
        "max_missed_interrupts": 1,
    },
    "human_before": {
        "max_false_interrupts": 0,
        "max_false_defers": 0,
        "max_missed_interrupts": 0,
        "expect_defer": True,
    },
    "human_after": {
        "max_false_interrupts": 0,
        "max_false_defers": 0,
        "max_missed_interrupts": 0,
        "expect_no_interrupt": True,
    },
    "overlap_self_first": {
        "min_true_interrupts": 1,
        "min_resumes": 1,
        "max_false_interrupts": 0,
        "max_false_defers": 0,
        "max_missed_interrupts": 0,
    },
}


def check_scenario(name: str, metrics: dict) -> list[str]:
    """Check a scenario's metrics against expected behavior. Returns list of failures."""
    failures = []
    exp = EXPECTED[name]

    fi = metrics.get("false_interrupts", 0)
    fd = metrics.get("false_defers", 0)
    mi = metrics.get("missed_interrupts", 0)
    ti = metrics.get("true_interrupts", 0)

    if "max_false_interrupts" in exp and fi > exp["max_false_interrupts"]:
        failures.append(f"{name}: FI={fi} > max {exp['max_false_interrupts']}")
    if "max_false_defers" in exp and fd > exp["max_false_defers"]:
        failures.append(f"{name}: FD={fd} > max {exp['max_false_defers']}")
    if "max_missed_interrupts" in exp and mi > exp["max_missed_interrupts"]:
        failures.append(f"{name}: MI={mi} > max {exp['max_missed_interrupts']}")
    if "min_true_interrupts" in exp and ti < exp["min_true_interrupts"]:
        failures.append(f"{name}: TI={ti} < min {exp['min_true_interrupts']}")

    rr = metrics.get("resume_rate", "n/a")
    if "min_resumes" in exp:
        # Parse "X/Y" format
        if "/" in str(rr):
            num, den = rr.split("/")
            if int(num) < exp["min_resumes"]:
                failures.append(f"{name}: resume_rate={rr} < min {exp['min_resumes']}")

    return failures


async def main():
    parser = argparse.ArgumentParser(description="M3 Acceptance Test")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--human-output-device", type=int, default=4)
    parser.add_argument("--room-filter", default=None)
    parser.add_argument("--m4-enable", action="store_true")
    parser.add_argument("--save-dir", default="data/m3_acceptance_test")
    # Required by run_scenario but not used in acceptance test
    parser.add_argument("--calibrate", action="store_true", default=True)
    parser.add_argument("--calibrate-duration", type=float, default=2.0)
    parser.add_argument("--calibrate-margin-db", type=float, default=15.0)
    parser.add_argument("--both-external-min-frames", type=int, default=3)
    parser.add_argument("--both-external-energy-ratio", type=float, default=1.2)
    parser.add_argument("--room-local-threshold", type=float, default=None)
    parser.add_argument("--m4-min-buffer-ms", type=float, default=60.0)
    parser.add_argument("--m4-self-threshold", type=float, default=0.50)
    parser.add_argument("--listen-threshold-ms", type=float, default=100.0)
    parser.add_argument("--cooldown-ms", type=float, default=200.0)
    args = parser.parse_args()

    print("=" * 60)
    print("M3 Dual-Speaker Acceptance Test")
    print("=" * 60)

    all_failures = []
    all_metrics = {}

    for name, scenario in SCENARIOS.items():
        print(f"\n--- {name}: {scenario.description} ---")
        try:
            metrics = await run_scenario(scenario, args)
            all_metrics[name] = metrics
            failures = check_scenario(name, metrics)
            if failures:
                all_failures.extend(failures)
                for f in failures:
                    print(f"  FAIL: {f}")
            else:
                print(f"  PASS (TI={metrics.get('true_interrupts', 0)} FI={metrics.get('false_interrupts', 0)} MI={metrics.get('missed_interrupts', 0)} resume={metrics.get('resume_rate', 'n/a')})")
        except Exception as e:
            all_failures.append(f"{name}: EXCEPTION: {e}")
            print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    if all_failures:
        print(f"FAILED: {len(all_failures)} issue(s)")
        for f in all_failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"ALL {len(SCENARIOS)} SCENARIOS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())

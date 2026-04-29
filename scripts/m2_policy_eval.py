#!/usr/bin/env python3
"""Evaluate a conservative learned/rule M2 BOTH-frame policy gate."""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "packages")

from m2_policy.dataset import load_many
from m2_policy.risk_gate import StrategyMeanGate, leave_one_seed_out_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M2 BOTH-frame policy risk gates")
    parser.add_argument("summaries", nargs="+", help="offline_summary.json paths")
    parser.add_argument("--fallback-strategy", default="gain_1p0")
    args = parser.parse_args()

    rows = load_many(args.summaries)
    gate = StrategyMeanGate.fit(rows, fallback_strategy=args.fallback_strategy)
    loo = leave_one_seed_out_eval(rows, fallback_strategy=args.fallback_strategy)
    out = {
        "rows": len(rows),
        "strategy_mean_delta": gate.strategy_mean_delta,
        "leave_one_seed_out": loo,
        "interpretation": "Promote only if chosen mean delta beats fallback and regressions_vs_fallback is 0.",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

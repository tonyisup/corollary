#!/usr/bin/env python3
"""
M4 vs No-M4 Comparison: Parse frame_verdicts.jsonl and compare metrics.

Compares:
1. SELF frame count (M4 should increase this)
2. BOTH frame count (M4 should decrease this for self-speech)
3. M4 metadata (attempted, outcome, confidence)
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any


def parse_frame_verdicts(turn_dir: str) -> dict[str, Any]:
    """Parse frame_verdicts.jsonl and return verdict counts + M4 stats."""
    verdicts_path = os.path.join(turn_dir, "frame_verdicts.jsonl")
    if not os.path.exists(verdicts_path):
        return {}
    
    verdicts = []
    m4_stats = {
        "attempted": 0,
        "accepted": 0,
        "below_threshold": 0,
        "buffer_too_short": 0,
        "not_attempted": 0,
        "exception": 0,
        "confidence_sum": 0.0,
        "confidence_count": 0,
    }
    
    with open(verdicts_path, "r") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                verdicts.append(rec)
                
                # M4 stats
                if rec.get("m4_attempted"):
                    m4_stats["attempted"] += 1
                    outcome = rec.get("m4_outcome", "unknown")
                    if outcome in m4_stats:
                        m4_stats[outcome] += 1
                    conf = rec.get("m4_confidence")
                    if conf is not None:
                        m4_stats["confidence_sum"] += conf
                        m4_stats["confidence_count"] += 1
    
    verdict_counts = Counter(v["verdict"] for v in verdicts)
    
    return {
        "total_frames": len(verdicts),
        "self_frames": verdict_counts.get("SELF", 0),
        "both_frames": verdict_counts.get("BOTH", 0),
        "world_frames": verdict_counts.get("WORLD", 0),
        "silence_frames": verdict_counts.get("SILENCE", 0),
        "m4_stats": m4_stats,
        "avg_m4_confidence": (
            m4_stats["confidence_sum"] / m4_stats["confidence_count"]
            if m4_stats["confidence_count"] > 0 else 0.0
        ),
    }


def compare_conditions(baseline_dir: str, m4_dir: str) -> dict[str, Any]:
    """Compare baseline (M4 off) vs M4 on metrics."""
    baseline_results = {}
    m4_results = {}
    
    # Parse all turns in each directory
    for turn_dir in sorted(os.listdir(baseline_dir)):
        if turn_dir.startswith("turn_"):
            full_path = os.path.join(baseline_dir, turn_dir)
            if os.path.isdir(full_path):
                baseline_results[turn_dir] = parse_frame_verdicts(full_path)
    
    for turn_dir in sorted(os.listdir(m4_dir)):
        if turn_dir.startswith("turn_"):
            full_path = os.path.join(m4_dir, turn_dir)
            if os.path.isdir(full_path):
                m4_results[turn_dir] = parse_frame_verdicts(full_path)
    
    # Aggregate
    comparison = {
        "baseline": {
            "total_self": sum(r.get("self_frames", 0) for r in baseline_results.values()),
            "total_both": sum(r.get("both_frames", 0) for r in baseline_results.values()),
            "total_world": sum(r.get("world_frames", 0) for r in baseline_results.values()),
        },
        "m4_on": {
            "total_self": sum(r.get("self_frames", 0) for r in m4_results.values()),
            "total_both": sum(r.get("both_frames", 0) for r in m4_results.values()),
            "total_world": sum(r.get("world_frames", 0) for r in m4_results.values()),
            "m4_accepted": sum(
                r.get("m4_stats", {}).get("accepted", 0)
                for r in m4_results.values()
            ),
            "avg_m4_confidence": (
                sum(r.get("avg_m4_confidence", 0) for r in m4_results.values())
                / len(m4_results) if m4_results else 0.0
            ),
        },
        "turns": {},
    }
    
    # Per-turn comparison
    for turn_key in sorted(set(baseline_results.keys()) | set(m4_results.keys())):
        comparison["turns"][turn_key] = {
            "baseline": baseline_results.get(turn_key, {}),
            "m4_on": m4_results.get(turn_key, {}),
        }
    
    return comparison


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare M4 vs No-M4 frame verdicts")
    parser.add_argument("--baseline-dir", required=True, help="Directory with M4-off results")
    parser.add_argument("--m4-dir", required=True, help="Directory with M4-on results")
    parser.add_argument("--output", default="data/m4_comparison.json", help="Output file")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("M4 vs No-M4 COMPARISON")
    print(f"{'='*60}\n")
    
    comparison = compare_conditions(args.baseline_dir, args.m4_dir)
    
    # Print summary
    print("AGGREGATE RESULTS:")
    print(f"{'='*60}")
    print(f"Baseline (M4 OFF):")
    print(f"  SELF frames: {comparison['baseline']['total_self']}")
    print(f"  BOTH frames: {comparison['baseline']['total_both']}")
    print(f"  WORLD frames: {comparison['baseline']['total_world']}")
    
    print(f"\nM4 ON:")
    print(f"  SELF frames: {comparison['m4_on']['total_self']}")
    print(f"  BOTH frames: {comparison['m4_on']['total_both']}")
    print(f"  WORLD frames: {comparison['m4_on']['total_world']}")
    print(f"  M4 ACCEPTED frames: {comparison['m4_on']['m4_accepted']}")
    print(f"  Avg M4 confidence: {comparison['m4_on']['avg_m4_confidence']:.3f}")
    
    # Calculate improvement
    if comparison['baseline']['total_self'] > 0:
        improvement = (
            (comparison['m4_on']['total_self'] - comparison['baseline']['total_self'])
            / comparison['baseline']['total_self'] * 100
        )
        print(f"\nSELF frame improvement: {improvement:+.1f}%")
    
    # Save detailed results
    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nSaved detailed comparison to: {args.output}")


if __name__ == "__main__":
    main()

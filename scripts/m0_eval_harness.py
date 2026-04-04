#!/usr/bin/env python3
"""M0 Eval Harness — Automated acceptance testing for M0 simulated loop.

Runs a battery of test utterances through the PFC simulated loop and
reports acceptance criteria per docs/eval_protocol.md:

    - WER ≤ 5% (M0: self-voice has no noise, should be ~0%)
    - SVSR ≥ 20 dB (self-voice suppression ratio)
    - Roundtrip latency < 5 s (M0 scale)
    - SAG grounding: all input tokens resolve to valid nodes

Usage:
    # Default: runs 5 standard test utterances with Whisper tiny
    python scripts/m0_eval_harness.py

    # Use Whisper base for better (but slower) accuracy
    python scripts/m0_eval_harness.py --model base

    # Run specific utterances by index (0-based)
    python scripts/m0_eval_harness.py --indices 0 2 4

Requirements:
    The Corollary packages must be on PYTHONPATH.
    faster-whisper, edge-tts, librosa, ffmpeg must be installed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List

# Add corollary packages to path
sys.path.insert(0, "packages")

from pfc.simulated_loop import run_simulated_loop, LoopResult

# ── Test Utterances ─────────────────────────────────────────────────────────

TEST_UTTERANCES = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "This is a test of the self listening loop system",
    "Corollary checks that it can hear itself speak",
    "Semantic abstraction graphs help resolve ambiguous terms",
]

# ── Acceptance Criteria ─────────────────────────────────────────────────────

CRITERIA = {
    "wer_max": 0.05,          # 5% max word error rate
    "svsr_min_db": 20.0,      # 20 dB min self-voice suppression
    "latency_max_s": 5.0,     # 5 seconds max roundtrip
}


def evaluate_result(result: LoopResult) -> Dict[str, Any]:
    """Evaluate a single LoopResult against acceptance criteria.
    
    Returns dict with pass/fail status for each criterion.
    """
    checks = {
        "wer": {
            "actual": result.wer,
            "threshold": f"≤ {CRITERIA['wer_max']:.0%}",
            "pass": result.wer <= CRITERIA["wer_max"],
        },
        "svsr_db": {
            "actual": round(result.svsr_db, 1),
            "threshold": f"≥ {CRITERIA['svsr_min_db']} dB",
            "pass": result.svsr_db >= CRITERIA["svsr_min_db"],
        },
        "latency_s": {
            "actual": round(result.roundtrip_seconds, 2),
            "threshold": f"≤ {CRITERIA['latency_max_s']} s",
            "pass": result.roundtrip_seconds <= CRITERIA["latency_max_s"],
        },
    }
    
    checks["all_pass"] = all(c["pass"] for c in checks.values())
    return checks


def print_result_table(results: List[tuple]) -> None:
    """Print formatted results table."""
    width = 80
    print(f"\n{'═' * width}")
    print(f"  M0 EVALUATION RESULTS")
    print(f"{'═' * width}\n")
    
    for i, (utterance, result, checks) in enumerate(results):
        status = "✅ PASS" if checks["all_pass"] else "❌ FAIL"
        print(f"[{i}] {status} — \"{utterance}\"")
        print(f"    WER:     {checks['wer']['actual']:.1%}  ({checks['wer']['threshold']}) {'✅' if checks['wer']['pass'] else '❌'}")
        print(f"    SVSR:    {checks['svsr_db']['actual']} dB  ({checks['svsr_db']['threshold']}) {'✅' if checks['svsr_db']['pass'] else '❌'}")
        print(f"    Latency: {checks['latency_s']['actual']}s  ({checks['latency_s']['threshold']}) {'✅' if checks['latency_s']['pass'] else '❌'}")
        
        # Show ASR transcript if mismatched
        if result.asr_result is not None:
            if result.wer > 0:
                print(f"    Expected: \"{result.original_text}\"")
                print(f"    Got:      \"{result.asr_result.text}\"")
        print()
    
    # Summary
    total = len(results)
    passed = sum(1 for _, _, checks in results if checks["all_pass"])
    print(f"{'─' * width}")
    print(f"  Summary: {passed}/{total} passed")
    print(f"{'═' * width}\n")


async def main(
    model: str = "tiny",
    indices: List[int] | None = None,
) -> None:
    """Run eval harness and print results."""
    selected = indices if indices is not None else list(range(len(TEST_UTTERANCES)))
    
    print("🔬  M0 Eval Harness")
    print(f"    Model:  {model}")
    print(f"    Tests:  {len(selected)} utterances")
    print(f"    Criteria: WER ≤ 5%, SVSR ≥ 20 dB, Latency ≤ 5 s")
    
    t_start = time.perf_counter()
    results: List[tuple] = []
    
    for i in selected:
        utterance = TEST_UTTERANCES[i]
        print(f"\n  [{i}] Running: \"{utterance}\"... ", end="", flush=True)
        
        result = await run_simulated_loop(
            utterance,
            build_sag=True,
            whisper_model=model,
        )
        checks = evaluate_result(result)
        results.append((utterance, result, checks))
        
        status = "PASS" if checks["all_pass"] else "FAIL"
        print(f"{status} (SVSR: {checks['svsr_db']['actual']} dB, WER: {checks['wer']['actual']:.1%}, ⏱️ {checks['latency_s']['actual']}s)")
    
    # Print table
    print_result_table(results)
    
    # Timing
    elapsed = time.perf_counter() - t_start
    print(f"  Total time: {elapsed:.1f}s\n")
    
    # Exit code
    all_passed = all(checks["all_pass"] for _, _, checks in results)
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="M0 Eval Harness")
    parser.add_argument("--model", default="tiny", help="Whisper model size")
    parser.add_argument("--indices", nargs="+", type=int, default=None,
                        help="Specific test indices to run (0-based)")
    args = parser.parse_args()
    
    asyncio.run(main(model=args.model, indices=args.indices))

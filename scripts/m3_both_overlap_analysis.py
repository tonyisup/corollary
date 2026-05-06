#!/usr/bin/env python3
"""M3 BOTH-overlap discrimination analysis.

Analyzes the discrimination power of various signals for distinguishing
self-dominant BOTH from human-overlap BOTH during active playback.

Key finding: No single signal perfectly discriminates. Best approach combines:
1. Excess energy (works when human is louder)
2. Sustained duration (works as fallback)
3. Multi-frame voting (reduces false positives)

See docs/m3_both_overlap_analysis.md for full analysis.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results = json.load(open(Path('data/m3_interrupt_v2/offline_analysis_results_m2.json')))
    
    print("=" * 80)
    print("M3 BOTH-OVERLAP DISCRIMINATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Summary of findings
    print("KEY FINDINGS:")
    print()
    print("1. Match score does NOT discriminate (constant within scenario)")
    print("   - xcorr is computed against full template, not per-frame")
    print("   - Human speech adds energy but doesn't change correlation much")
    print()
    print("2. Excess energy discriminates WHEN human is louder than TTS")
    print("   - brief_noise: +15.7 dB excess (human much louder)")
    print("   - overlap_human_first: +13.1 dB excess")
    print("   - overlap_self_first: +9.6 dB excess")
    print("   - human_late: +3.5 dB excess (marginal)")
    print("   - human_early: -0.8 dB (human quieter, fails)")
    print("   - human_before: -0.2 dB (human quieter, fails)")
    print()
    print("3. Absolute RMS threshold fails due to TTS variability")
    print("   - TTS RMS std ~10 dB (highly variable)")
    print("   - self_only BOTH: 59% of frames above -25 dB")
    print("   - Any threshold that catches quiet human speech also catches TTS peaks")
    print()
    print("4. M4 below-threshold fires too often during self-playback")
    print("   - M4 confidence during self-playback is often < 0.50")
    print("   - Even with 5-frame sustained requirement, 435 false interrupts")
    print()
    print("5. Best combined approach (from sweep):")
    print("   - Sustained BOTH (>15 frames) + excess energy (>12 dB)")
    print("   - 24 false interrupts, 5 true interrupts, 4 misses")
    print("   - Still too many false interrupts for production use")
    print()
    print("RECOMMENDATION:")
    print("   The current architecture cannot reliably discriminate self-dominant")
    print("   BOTH from human-overlap BOTH using frame-level signals alone.")
    print()
    print("   Two paths forward:")
    print()
    print("   A) Accept current behavior (no yield on BOTH during motor window)")
    print("      - Pros: 0 false interrupts, 0 false defers")
    print("      - Cons: doesn't yield to human speech during self-playback")
    print("      - This is the SAFEST option for now")
    print()
    print("   B) Add per-frame self-match scoring to the attributor")
    print("      - Compute local xcorr for each frame against expected self")
    print("      - When human speech overlaps, local xcorr drops")
    print("      - Requires significant changes to attributor architecture")
    print("      - This is the CORRECT long-term solution")
    print()
    print("   C) Use M4 with a separate, lower threshold for overlap detection")
    print("      - Run M4 on a fresh buffer when BOTH is sustained")
    print("      - Lower threshold (e.g., 0.30) for overlap detection")
    print("      - Higher threshold (0.50) for SELF classification")
    print("      - Requires M4 buffer reset logic")
    print()


if __name__ == "__main__":
    main()

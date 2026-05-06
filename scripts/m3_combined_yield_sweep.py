#!/usr/bin/env python3
"""M3 combined yield policy sweep: sustained BOTH + excess energy.

Tests a controller policy where, during SPEAK, the controller yields to LISTEN
only if BOTH is sustained for > N frames AND excess energy above self baseline
is detected.

This combines two signals:
1. Sustained BOTH: ensures we don't react to brief transients
2. Excess energy: ensures we only yield when there's evidence of external speech

Usage:
    .venv/bin/python scripts/m3_combined_yield_sweep.py \\
        --results data/m3_interrupt_v2/offline_analysis_results_m2.json
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def compute_self_baseline(fvs: list[dict]) -> float:
    """Compute mean self-only BOTH RMS."""
    self_rms = [f['rms_db'] for f in fvs 
                if f['verdict'] == 'BOTH' and f.get('is_speech') 
                and f.get('gt_agent') and not f.get('gt_human')]
    return float(np.mean(self_rms)) if self_rms else -30.0


def simulate_combined_yield_policy(
    fvs: list[dict], 
    yield_threshold_frames: int,
    excess_db_threshold: float,
    self_baseline_rms: float,
) -> dict:
    """Simulate the combined yield policy.
    
    Yields to LISTEN only if:
    1. BOTH is sustained for >= yield_threshold_frames consecutive frames
    2. Excess RMS above self baseline > excess_db_threshold
    """
    state = "IDLE"
    both_count = 0
    false_interrupts = 0
    true_interrupts = 0
    missed_interrupts = 0
    total_yield_events = 0
    response_delays = []
    human_overlap_started = False
    human_overlap_start_frame = None
    yielded_during_current_overlap = False
    
    for i, f in enumerate(fvs):
        verdict = f.get('verdict', 'SILENCE')
        is_speech = f.get('is_speech', False)
        gt_human = f.get('gt_human', False)
        gt_agent = f.get('gt_agent', False)
        rms_db = f.get('rms_db', -60.0)
        
        # Track human overlap
        if gt_human and gt_agent and not human_overlap_started:
            human_overlap_started = True
            human_overlap_start_frame = i
            yielded_during_current_overlap = False
        
        if not gt_human and human_overlap_started:
            if not yielded_during_current_overlap:
                missed_interrupts += 1
            human_overlap_started = False
            human_overlap_start_frame = None
        
        # State machine with combined yield policy
        if state == "SPEAK":
            if verdict == "BOTH" and is_speech and gt_agent:
                both_count += 1
                
                # Check excess energy
                excess_db = rms_db - self_baseline_rms
                
                if both_count >= yield_threshold_frames and excess_db > excess_db_threshold:
                    # Yield to LISTEN
                    state = "LISTEN"
                    total_yield_events += 1
                    both_count = 0
                    
                    if gt_human:
                        true_interrupts += 1
                        yielded_during_current_overlap = True
                        if human_overlap_start_frame is not None:
                            response_delays.append(i - human_overlap_start_frame)
                    else:
                        false_interrupts += 1
            else:
                both_count = 0
                
        elif state == "LISTEN":
            if verdict in ("SELF", "BOTH") and is_speech:
                state = "SPEAK"
                both_count = 0
            elif verdict == "WORLD" and is_speech:
                pass
            elif verdict == "SILENCE" or not is_speech:
                state = "IDLE"
                both_count = 0
                
        elif state == "IDLE":
            if verdict in ("SELF", "BOTH") and is_speech and gt_agent:
                state = "SPEAK"
                both_count = 0
            elif verdict == "WORLD" and is_speech:
                state = "LISTEN"
                both_count = 0
    
    avg_delay = sum(response_delays) / len(response_delays) if response_delays else None
    
    return {
        'yield_threshold_frames': yield_threshold_frames,
        'excess_db_threshold': excess_db_threshold,
        'false_interrupts': false_interrupts,
        'true_interrupts': true_interrupts,
        'missed_interrupts': missed_interrupts,
        'total_yield_events': total_yield_events,
        'avg_response_delay_frames': avg_delay,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=Path("data/m3_interrupt_v2/offline_analysis_results_m2.json"))
    parser.add_argument("--yield-thresholds", type=int, nargs="+",
                        default=[3, 5, 8, 10, 15])
    parser.add_argument("--excess-db-thresholds", type=float, nargs="+",
                        default=[0, 3, 6, 9, 12])
    args = parser.parse_args()
    
    results = load_results(args.results)
    
    print("=" * 80)
    print("M3 COMBINED YIELD POLICY SWEEP (sustained BOTH + excess energy)")
    print("=" * 80)
    print(f"Source: {args.results}")
    print(f"Yield thresholds: {args.yield_thresholds}")
    print(f"Excess dB thresholds: {args.excess_db_thresholds}")
    print()
    
    # Categorize scenarios
    overlap_scenarios = []
    self_only_scenarios = []
    
    for s in results['scenarios']:
        fvs = s.get('frame_verdicts', [])
        has_overlap = any(f.get('gt_human') and f.get('gt_agent') and f['verdict'] == 'BOTH' for f in fvs)
        has_self = any(f.get('gt_agent') and f['verdict'] == 'BOTH' for f in fvs)
        
        if has_overlap:
            overlap_scenarios.append(s)
        elif has_self:
            self_only_scenarios.append(s)
    
    print(f"Overlap scenarios: {len(overlap_scenarios)}")
    print(f"Self-only scenarios: {len(self_only_scenarios)}")
    print()
    
    # Compute self baseline from self-only scenarios
    all_self_baselines = []
    for s in self_only_scenarios:
        fvs = s.get('frame_verdicts', [])
        baseline = compute_self_baseline(fvs)
        all_self_baselines.append(baseline)
    
    global_self_baseline = float(np.mean(all_self_baselines)) if all_self_baselines else -30.0
    print(f"Self baseline RMS: {global_self_baseline:.1f} dB")
    print()
    
    # Sweep
    best_config = None
    best_score = -1
    
    for yt in args.yield_thresholds:
        for edt in args.excess_db_thresholds:
            total_fi = 0
            total_ti = 0
            total_mi = 0
            all_delays = []
            
            for s in overlap_scenarios:
                fvs = s.get('frame_verdicts', [])
                # Use scenario-specific baseline
                scenario_baseline = compute_self_baseline(fvs)
                baseline = scenario_baseline if scenario_baseline is not None else global_self_baseline
                
                r = simulate_combined_yield_policy(fvs, yt, edt, baseline)
                total_fi += r['false_interrupts']
                total_ti += r['true_interrupts']
                total_mi += r['missed_interrupts']
                if r['avg_response_delay_frames'] is not None:
                    all_delays.append(r['avg_response_delay_frames'])
            
            for s in self_only_scenarios:
                fvs = s.get('frame_verdicts', [])
                r = simulate_combined_yield_policy(fvs, yt, edt, global_self_baseline)
                total_fi += r['false_interrupts']
            
            # Score: maximize true interrupts, minimize false interrupts and misses
            # Weight: FI * 2 + MI * 3 vs TI * 1
            score = total_ti - total_fi * 2 - total_mi * 3
            avg_delay = sum(all_delays) / len(all_delays) if all_delays else None
            
            if score > best_score:
                best_score = score
                best_config = {
                    'yield_threshold': yt,
                    'excess_db': edt,
                    'false_interrupts': total_fi,
                    'true_interrupts': total_ti,
                    'missed_interrupts': total_mi,
                    'score': score,
                    'avg_delay': avg_delay,
                }
            
            print(f"yt={yt:2f} edt={edt:2.0f}dB: FI={total_fi:3d} TI={total_ti:3d} MI={total_mi:2d} score={score:4d}" +
                  (f" delay={avg_delay:.1f}f" if avg_delay else ""))
    
    print()
    print("=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    if best_config:
        print(f"  Yield threshold: {best_config['yield_threshold']} frames ({best_config['yield_threshold'] * 30}ms)")
        print(f"  Excess dB threshold: {best_config['excess_db']:.0f} dB")
        print(f"  False interrupts: {best_config['false_interrupts']}")
        print(f"  True interrupts: {best_config['true_interrupts']}")
        print(f"  Missed interrupts: {best_config['missed_interrupts']}")
        if best_config['avg_delay']:
            print(f"  Avg response delay: {best_config['avg_delay']:.1f} frames ({best_config['avg_delay'] * 30:.0f}ms)")


if __name__ == "__main__":
    main()

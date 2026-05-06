#!/usr/bin/env python3
"""M3 yield-on-sustained-overlap sweep.

Tests a controller policy where, during SPEAK, if BOTH frames are sustained
for more than N consecutive frames, the controller briefly yields to LISTEN
to check for external speech.

This sweep evaluates the policy offline using existing analysis results.
It does NOT modify the controller — it post-processes the frame verdicts
to simulate what would happen with the yield policy.

For each scenario and each N:
- Simulate the yield policy
- Count false interrupts (yield during self-only BOTH)
- Count true interrupts (yield during human-overlap BOTH)
- Measure response delay (frames between human start and yield)

Usage:
    .venv/bin/python scripts/m3_yield_sweep.py \\
        --results data/m3_interrupt_v2/offline_analysis_results_m2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def simulate_yield_policy(fvs: list[dict], yield_threshold_frames: int) -> dict:
    """Simulate the yield-on-sustained-overlap policy.
    
    Returns dict with:
    - false_interrupts: number of times we yield during self-only BOTH
    - true_interrupts: number of times we yield during human-overlap BOTH
    - missed_interrupts: human overlap frames where we didn't yield
    - response_delay_frames: frames between first human overlap and first yield
    - total_yield_events: total number of yield events
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
        
        # Track human overlap
        if gt_human and gt_agent and not human_overlap_started:
            human_overlap_started = True
            human_overlap_start_frame = i
            yielded_during_current_overlap = False
        
        if not gt_human and human_overlap_started:
            # Human overlap ended
            if not yielded_during_current_overlap:
                missed_interrupts += 1
            human_overlap_started = False
            human_overlap_start_frame = None
        
        # State machine with yield policy
        if state == "SPEAK":
            if verdict == "BOTH" and is_speech and gt_agent:
                both_count += 1
                if both_count >= yield_threshold_frames:
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
                # Self resumed, go back to SPEAK
                state = "SPEAK"
                both_count = 0
            elif verdict == "WORLD" and is_speech:
                # External speech, stay in LISTEN
                pass
            elif verdict == "SILENCE" or not is_speech:
                # Silence, return to IDLE
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
        'yield_threshold': yield_threshold_frames,
        'false_interrupts': false_interrupts,
        'true_interrupts': true_interrupts,
        'missed_interrupts': missed_interrupts,
        'total_yield_events': total_yield_events,
        'avg_response_delay_frames': avg_delay,
        'n_response_delays': len(response_delays),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=Path("data/m3_interrupt_v2/offline_analysis_results_m2.json"))
    parser.add_argument("--yield-thresholds", type=int, nargs="+",
                        default=[3, 5, 8, 10, 15, 20, 30])
    args = parser.parse_args()
    
    results = load_results(args.results)
    
    print("=" * 80)
    print("M3 YIELD-ON-SUSTAINED-OVERLAP SWEEP")
    print("=" * 80)
    print(f"Source: {args.results}")
    print(f"Yield thresholds: {args.yield_thresholds}")
    print()
    
    # Identify scenarios with human overlap during agent playback
    overlap_scenarios = []
    for s in results['scenarios']:
        fvs = s.get('frame_verdicts', [])
        has_overlap = any(f.get('gt_human') and f.get('gt_agent') and f['verdict'] == 'BOTH'
                         for f in fvs)
        if has_overlap:
            overlap_scenarios.append(s)
    
    print(f"Scenarios with human overlap during agent: {len(overlap_scenarios)}")
    for s in overlap_scenarios:
        print(f"  - {s['scenario']}")
    print()
    
    # Also identify self-only scenarios (for false interrupt measurement)
    self_only_scenarios = []
    for s in results['scenarios']:
        fvs = s.get('frame_verdicts', [])
        has_self = any(f.get('gt_agent') and f['verdict'] == 'BOTH' for f in fvs)
        has_human = any(f.get('gt_human') and f.get('gt_agent') for f in fvs)
        if has_self and not has_human:
            self_only_scenarios.append(s)
    
    print(f"Self-only scenarios: {len(self_only_scenarios)}")
    for s in self_only_scenarios:
        print(f"  - {s['scenario']}")
    print()
    
    # Sweep
    all_results = {}
    for threshold in args.yield_thresholds:
        total_fi = 0
        total_ti = 0
        total_mi = 0
        total_yields = 0
        all_delays = []
        
        for s in overlap_scenarios:
            fvs = s.get('frame_verdicts', [])
            r = simulate_yield_policy(fvs, threshold)
            total_fi += r['false_interrupts']
            total_ti += r['true_interrupts']
            total_mi += r['missed_interrupts']
            total_yields += r['total_yield_events']
            if r['avg_response_delay_frames'] is not None:
                all_delays.append(r['avg_response_delay_frames'])
        
        # Also check self-only scenarios for false interrupts
        for s in self_only_scenarios:
            fvs = s.get('frame_verdicts', [])
            r = simulate_yield_policy(fvs, threshold)
            total_fi += r['false_interrupts']
        
        avg_delay = sum(all_delays) / len(all_delays) if all_delays else None
        
        all_results[threshold] = {
            'false_interrupts': total_fi,
            'true_interrupts': total_ti,
            'missed_interrupts': total_mi,
            'total_yields': total_yields,
            'avg_response_delay_frames': avg_delay,
        }
        
        print(f"Threshold {threshold:3d} frames ({threshold * 30}ms):")
        print(f"  False interrupts: {total_fi}")
        print(f"  True interrupts: {total_ti}")
        print(f"  Missed interrupts: {total_mi}")
        print(f"  Total yield events: {total_yields}")
        if avg_delay is not None:
            print(f"  Avg response delay: {avg_delay:.1f} frames ({avg_delay * 30:.0f}ms)")
        print()
    
    # Save results
    out_path = args.results.parent / 'yield_sweep.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()

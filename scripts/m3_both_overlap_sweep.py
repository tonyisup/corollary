#!/usr/bin/env python3
"""M3 BOTH-overlap discrimination sweep.

Tests whether excess energy or other signals can discriminate self-dominant BOTH
from human-overlap BOTH during active playback.

Uses existing offline analysis results (no new recording needed).

For each scenario with both self-only and overlap BOTH frames, computes:
- excess_rms_db: observed RMS - expected self RMS
- match_score delta from scenario baseline
- sustained overlap duration

Sweeps thresholds and reports:
- True positives: overlap frames correctly identified
- False positives: self-only frames incorrectly flagged
- Missed: overlap frames not detected

Usage:
    .venv/bin/python scripts/m3_both_overlap_sweep.py \\
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


def compute_self_baseline(fvs: list[dict], require_speech: bool = True) -> dict:
    """Compute self-only BOTH statistics from frames where gt_agent=True, gt_human=False."""
    rms_values = []
    match_values = []
    for f in fvs:
        if f['verdict'] != 'BOTH':
            continue
        if require_speech and not f.get('is_speech'):
            continue
        if f.get('gt_agent') and not f.get('gt_human'):
            rms_values.append(f['rms_db'])
            match_values.append(f['match_score'])
    
    if not rms_values:
        return {'mean_rms': None, 'std_rms': None, 'median_rms': None, 'mean_match': None, 'n': 0}
    
    import numpy as np
    return {
        'mean_rms': float(np.mean(rms_values)),
        'std_rms': float(np.std(rms_values)),
        'median_rms': float(np.median(rms_values)),
        'mean_match': float(np.mean(match_values)),
        'n': len(rms_values),
    }


def sweep_scenario(fvs: list[dict], self_baseline: dict, 
                   excess_db_thresholds: list[float],
                   min_consecutive_frames: int) -> dict:
    """Sweep excess dB thresholds for one scenario."""
    import numpy as np
    
    results = {}
    for threshold in excess_db_thresholds:
        for min_frames in min_consecutive_frames:
            key = f"excess_{threshold:.0f}db_min_{min_frames}f"
            
            tp = 0  # overlap frames correctly flagged
            fp = 0  # self-only frames incorrectly flagged
            fn = 0  # overlap frames missed
            total_overlap = 0
            total_self_only = 0
            
            # Sliding window for sustained detection
            consecutive_above = 0
            flagged_frames = set()
            
            for i, f in enumerate(fvs):
                if f['verdict'] != 'BOTH' or not f.get('is_speech'):
                    consecutive_above = 0
                    continue
                
                if not f.get('gt_agent'):
                    consecutive_above = 0
                    continue
                
                is_overlap = f.get('gt_human', False)
                
                if is_overlap:
                    total_overlap += 1
                else:
                    total_self_only += 1
                
                # Compute excess dB
                if self_baseline['mean_rms'] is not None:
                    excess = f['rms_db'] - self_baseline['mean_rms']
                else:
                    excess = 0
                
                if excess > threshold:
                    consecutive_above += 1
                else:
                    consecutive_above = 0
                
                if consecutive_above >= min_frames:
                    flagged_frames.add(i)
            
            # Count TP/FP/FN
            for i, f in enumerate(fvs):
                if f['verdict'] != 'BOTH' or not f.get('is_speech') or not f.get('gt_agent'):
                    continue
                
                is_overlap = f.get('gt_human', False)
                is_flagged = i in flagged_frames
                
                if is_overlap and is_flagged:
                    tp += 1
                elif not is_overlap and is_flagged:
                    fp += 1
                elif is_overlap and not is_flagged:
                    fn += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[key] = {
                'threshold_db': threshold,
                'min_frames': min_frames,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'total_overlap': total_overlap,
                'total_self_only': total_self_only,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
    
    return results


def main() -> None:
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, 
                        default=Path("data/m3_interrupt_v2/offline_analysis_results_m2.json"))
    parser.add_argument("--excess-db", type=float, nargs="+", 
                        default=[0, 3, 6, 9, 12, 15])
    parser.add_argument("--min-frames", type=int, nargs="+",
                        default=[2, 3, 5, 8])
    args = parser.parse_args()
    
    results = load_results(args.results)
    
    print("=" * 80)
    print("M3 BOTH-OVERLAP DISCRIMINATION SWEEP")
    print("=" * 80)
    print(f"Source: {args.results}")
    print(f"Excess dB thresholds: {args.excess_db}")
    print(f"Min consecutive frames: {args.min_frames}")
    print()
    
    all_scenario_results = {}
    
    for s in results['scenarios']:
        name = s['scenario']
        fvs = s.get('frame_verdicts', [])
        
        self_baseline = compute_self_baseline(fvs)
        
        if self_baseline['n'] == 0:
            continue
        
        # Check if there are overlap frames
        has_overlap = any(f.get('gt_human') and f.get('gt_agent') and f['verdict'] == 'BOTH' 
                         for f in fvs)
        if not has_overlap:
            continue
        
        print(f"\n--- {name} ---")
        print(f"  Self-only BOTH: n={self_baseline['n']}, mean_rms={self_baseline['mean_rms']:.1f}, std={self_baseline['std_rms']:.1f}")
        
        scenario_results = sweep_scenario(fvs, self_baseline, args.excess_db, args.min_frames)
        all_scenario_results[name] = scenario_results
        
        # Print best F1 for this scenario
        best_key = max(scenario_results.keys(), key=lambda k: scenario_results[k]['f1'])
        best = scenario_results[best_key]
        print(f"  Best F1: {best['f1']:.2f} @ {best['threshold_db']:.0f}dB, {best['min_frames']}f")
        print(f"    TP={best['tp']}, FP={best['fp']}, FN={best['fn']}")
        print(f"    Precision={best['precision']:.2f}, Recall={best['recall']:.2f}")
    
    # Aggregate across scenarios
    print("\n" + "=" * 80)
    print("AGGREGATE ACROSS SCENARIOS")
    print("=" * 80)
    
    # Collect all threshold combinations
    all_keys = set()
    for sr in all_scenario_results.values():
        all_keys.update(sr.keys())
    
    aggregate = {}
    for key in sorted(all_keys):
        total_tp = sum(sr.get(key, {}).get('tp', 0) for sr in all_scenario_results.values())
        total_fp = sum(sr.get(key, {}).get('fp', 0) for sr in all_scenario_results.values())
        total_fn = sum(sr.get(key, {}).get('fn', 0) for sr in all_scenario_results.values())
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        aggregate[key] = {
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': precision, 'recall': recall, 'f1': f1,
        }
    
    # Print top 5 by F1
    sorted_keys = sorted(aggregate.keys(), key=lambda k: aggregate[k]['f1'], reverse=True)[:5]
    print("\nTop 5 configurations by F1:")
    for key in sorted_keys:
        a = aggregate[key]
        print(f"  {key}: F1={a['f1']:.2f}, P={a['precision']:.2f}, R={a['recall']:.2f} (TP={a['tp']}, FP={a['fp']}, FN={a['fn']})")
    
    # Save results
    output = {
        'per_scenario': all_scenario_results,
        'aggregate': aggregate,
    }
    out_path = args.results.parent / 'both_overlap_sweep.json'
    with out_path.open('w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

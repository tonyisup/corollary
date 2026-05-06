#!/usr/bin/env python3
"""Compute per-frame self-match scores for recorded interruption scenarios.

For each frame during agent playback, computes a local self-match score by
comparing the observed audio against the expected agent PCM at that time point.
When human speech overlaps, the local self-match should drop because the
captured signal doesn't match the expected self template as well.

Outputs per-frame scores and summary statistics for each scenario.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

SR = 16000
VAD_FRAME_SAMPLES = int(0.03 * SR)  # 30ms frames


def compute_per_frame_self_match(
    observed: np.ndarray,
    agent_pcm: np.ndarray,
    agent_start_s: float,
    frame_samples: int = VAD_FRAME_SAMPLES,
) -> list[dict]:
    """Compute per-frame self-match scores.
    
    For each frame during agent playback, extracts the corresponding slice
    of the agent PCM and computes normalized cross-correlation between the
    observed frame and the expected self audio.
    
    Returns list of per-frame score dicts.
    """
    num_frames = len(observed) // frame_samples
    agent_start_sample = int(agent_start_s * SR)
    scores = []
    
    for i in range(num_frames):
        frame_start = i * frame_samples
        frame_end = frame_start + frame_samples
        obs_frame = observed[frame_start:frame_end]
        
        # Corresponding agent PCM slice
        agent_start = agent_start_sample + frame_start
        agent_end = agent_start + frame_samples
        
        if agent_start < 0 or agent_end > len(agent_pcm) or frame_end > len(observed):
            scores.append({
                'frame': i,
                't_s': frame_start / SR,
                'self_match': 0.0,
                'obs_rms': float(20 * np.log10(max(np.sqrt(np.mean(obs_frame**2)), 1e-10))),
                'agent_active': False,
            })
            continue
        
        agent_frame = agent_pcm[agent_start:agent_end]
        
        # Normalized cross-correlation
        obs_norm = float(np.linalg.norm(obs_frame))
        agent_norm = float(np.linalg.norm(agent_frame))
        
        if obs_norm < 1e-9 or agent_norm < 1e-9:
            score = 0.0
        else:
            score = float(np.dot(obs_frame, agent_frame) / (obs_norm * agent_norm))
        
        scores.append({
            'frame': i,
            't_s': frame_start / SR,
            'self_match': max(0.0, score),
            'obs_rms': float(20 * np.log10(max(np.sqrt(np.mean(obs_frame**2)), 1e-10))),
            'agent_rms': float(20 * np.log10(max(np.sqrt(np.mean(agent_frame**2)), 1e-10))),
            'agent_active': True,
        })
    
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/m3_interrupt_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/m3_interrupt_v2/per_frame_scores"))
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    
    for scenario_dir in sorted(args.input_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        meta_path = scenario_dir / "metadata.json"
        if not meta_path.exists():
            continue
        
        name = scenario_dir.name
        meta = json.load(open(meta_path))
        
        captured_path = scenario_dir / "captured.npy"
        agent_path = scenario_dir / "agent_pcm.npy"
        
        if not captured_path.exists() or not agent_path.exists():
            continue
        
        captured = np.load(str(captured_path)).astype(np.float32)
        agent_pcm = np.load(str(agent_path)).astype(np.float32)
        
        agent_start_s = meta.get("agent_start_s", 0.5)
        
        # Compute per-frame scores
        scores = compute_per_frame_self_match(captured, agent_pcm, agent_start_s)
        
        # Load ground truth
        gt_human_path = scenario_dir / "gt_human_speaking.npy"
        gt_agent_path = scenario_dir / "gt_agent_playing.npy"
        gt_human = np.load(str(gt_human_path)) if gt_human_path.exists() else np.zeros(len(captured), dtype=bool)
        gt_agent = np.load(str(gt_agent_path)) if gt_agent_path.exists() else np.zeros(len(captured), dtype=bool)
        
        # Add ground truth to scores
        for s in scores:
            idx = int(s['t_s'] * SR)
            s['gt_human'] = bool(gt_human[min(idx, len(gt_human)-1)])
            s['gt_agent'] = bool(gt_agent[min(idx, len(gt_agent)-1)])
        
        # Compute statistics
        self_only_scores = [s['self_match'] for s in scores if s['agent_active'] and not s['gt_human']]
        overlap_scores = [s['self_match'] for s in scores if s['agent_active'] and s['gt_human']]
        
        self_only_rms = [s['obs_rms'] for s in scores if s['agent_active'] and not s['gt_human']]
        overlap_rms = [s['obs_rms'] for s in scores if s['agent_active'] and s['gt_human']]
        
        print(f"\n{name}:")
        if self_only_scores:
            print(f"  self-only: n={len(self_only_scores)}, match_mean={np.mean(self_only_scores):.3f}, rms_mean={np.mean(self_only_rms):.1f}")
        if overlap_scores:
            print(f"  overlap:   n={len(overlap_scores)}, match_mean={np.mean(overlap_scores):.3f}, rms_mean={np.mean(overlap_rms):.1f}")
        if self_only_scores and overlap_scores:
            match_diff = np.mean(overlap_scores) - np.mean(self_only_scores)
            rms_diff = np.mean(overlap_rms) - np.mean(self_only_rms)
            print(f"  match diff (overlap - self): {match_diff:+.3f}")
            print(f"  rms diff (overlap - self):   {rms_diff:+.1f} dB")
        
        # Save per-frame scores
        out_path = args.output_dir / f"{name}_per_frame.json"
        with open(out_path, 'w') as f:
            json.dump(scores, f, indent=2)
        
        summary[name] = {
            'n_self_only': len(self_only_scores),
            'n_overlap': len(overlap_scores),
            'self_match_mean': float(np.mean(self_only_scores)) if self_only_scores else None,
            'overlap_match_mean': float(np.mean(overlap_scores)) if overlap_scores else None,
            'match_diff': float(np.mean(overlap_scores) - np.mean(self_only_scores)) if (self_only_scores and overlap_scores) else None,
            'self_rms_mean': float(np.mean(self_only_rms)) if self_only_rms else None,
            'overlap_rms_mean': float(np.mean(overlap_rms)) if overlap_rms else None,
            'rms_diff': float(np.mean(overlap_rms) - np.mean(self_only_rms)) if (self_only_rms and overlap_rms) else None,
        }
    
    # Save summary
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nPer-frame scores saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

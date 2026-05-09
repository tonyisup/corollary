#!/usr/bin/env python3
"""Post-hoc analysis of streaming runtime JSONL logs.

Reads the structured logs produced by StreamingRuntime and produces a
compact summary of the turn-taking behavior.

Usage:
  .venv/bin/python scripts/m3_streaming_analyze.py /tmp/streaming_demo_log.jsonl
  .venv/bin/python scripts/m3_streaming_analyze.py /tmp/streaming_demo_log.jsonl --output /tmp/analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_logs(path: str | Path) -> list[dict]:
    """Load JSONL log entries."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def analyze_logs(logs: list[dict]) -> dict:
    """Produce a compact analysis of streaming logs."""
    if not logs:
        return {"error": "no logs"}

    total = len(logs)

    # Frame counts
    speech_frames = sum(1 for l in logs if l.get("vad_speech"))
    tts_played = sum(1 for l in logs if l.get("tts_playing"))
    tts_muted = sum(1 for l in logs if not l.get("tts_playing") and l.get("tts_position_s", 0) < l.get("tts_total_duration_s", 0))
    interrupts = sum(1 for l in logs if l.get("should_interrupt"))
    deferreds = sum(1 for l in logs if l.get("deferred"))
    resumes = sum(1 for l in logs if l.get("resumed"))

    # Verdict distribution
    verdict_counts: dict[str, int] = {}
    for l in logs:
        v = l.get("verdict", "UNKNOWN")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    # Controller state distribution
    state_counts: dict[str, int] = {}
    for l in logs:
        s = l.get("controller_state", "UNKNOWN")
        state_counts[s] = state_counts.get(s, 0) + 1

    # State transitions
    transitions: list[dict] = []
    prev_state = None
    for l in logs:
        state = l.get("controller_state")
        if state != prev_state:
            transitions.append({
                "frame": l["frame_index"],
                "timestamp_s": l["timestamp_s"],
                "from": prev_state,
                "to": state,
            })
            prev_state = state

    # TTS coverage
    total_duration = logs[0].get("tts_total_duration_s", 0) if logs else 0
    last_position = logs[-1].get("tts_position_s", 0) if logs else 0
    tts_completed = last_position >= total_duration * 0.99 if total_duration > 0 else False

    # M4 stats
    m4_attempted = sum(1 for l in logs if l.get("m4_outcome") not in ("not_attempted", None))
    m4_accepted = sum(1 for l in logs if l.get("m4_outcome") == "accepted")

    # Energy stats
    energy_values = [l.get("energy_db", -100) for l in logs]
    self_match_values = [l.get("self_match_score", 0) for l in logs if l.get("self_match_score") is not None]

    return {
        "total_frames": total,
        "speech_frames": speech_frames,
        "speech_fraction": round(speech_frames / total, 4) if total else 0,
        "tts_played_frames": tts_played,
        "tts_muted_frames": tts_muted,
        "tts_total_duration_s": total_duration,
        "tts_completed": tts_completed,
        "interrupt_events": interrupts,
        "deferred_events": deferreds,
        "resumed_events": resumes,
        "verdict_counts": verdict_counts,
        "controller_state_counts": state_counts,
        "state_transitions": transitions[:20],  # cap for readability
        "total_state_transitions": len(transitions),
        "m4_attempted_frames": m4_attempted,
        "m4_accepted_frames": m4_accepted,
        "energy_db": {
            "min": round(min(energy_values), 2) if energy_values else None,
            "max": round(max(energy_values), 2) if energy_values else None,
            "mean": round(sum(energy_values) / len(energy_values), 2) if energy_values else None,
        },
        "self_match_score": {
            "min": round(min(self_match_values), 4) if self_match_values else None,
            "max": round(max(self_match_values), 4) if self_match_values else None,
            "mean": round(sum(self_match_values) / len(self_match_values), 4) if self_match_values else None,
        },
        "final_frame": logs[-1] if logs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze streaming runtime logs")
    parser.add_argument("log_file", help="Path to JSONL log file")
    parser.add_argument("--output", help="Optional path to write analysis JSON")
    parser.add_argument("--full", action="store_true", help="Print full analysis (not just summary)")
    args = parser.parse_args()

    logs = load_logs(args.log_file)
    analysis = analyze_logs(logs)

    if args.full:
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print(f"\nStreaming Log Analysis: {args.log_file}")
        print(f"  Total frames:      {analysis['total_frames']}")
        print(f"  Speech frames:     {analysis['speech_frames']} ({analysis['speech_fraction']:.1%})")
        print(f"  TTS played:        {analysis['tts_played_frames']}")
        print(f"  TTS muted:         {analysis['tts_muted_frames']}")
        print(f"  TTS duration:      {analysis['tts_total_duration_s']:.2f}s")
        print(f"  TTS completed:     {analysis['tts_completed']}")
        print(f"  Interrupts:        {analysis['interrupt_events']}")
        print(f"  Deferred events:   {analysis['deferred_events']}")
        print(f"  Resumed events:    {analysis['resumed_events']}")
        print(f"  State transitions: {analysis['total_state_transitions']}")
        print(f"  M4 attempted:      {analysis['m4_attempted_frames']}")
        print(f"  M4 accepted:       {analysis['m4_accepted_frames']}")
        print(f"  Verdicts:          {analysis['verdict_counts']}")
        print(f"  Controller states: {analysis['controller_state_counts']}")
        if analysis['state_transitions']:
            print(f"  First transitions:")
            for t in analysis['state_transitions'][:5]:
                print(f"    frame {t['frame']} @ {t['timestamp_s']:.3f}s: {t['from']} → {t['to']}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nSaved analysis to: {out}")


if __name__ == "__main__":
    main()

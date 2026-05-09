#!/usr/bin/env python3
"""Deterministic acceptance check for M3 streaming runtime.

This checks two layers:
1. Synthetic attribution-frame validation of online controller/TTS gating.
   This is the behavior gate: pre-agent defer and overlap interrupt must hold
   TTS position and resume later.
2. Offline audio replay smoke on saved gated hardware artifacts. This validates
   callback/log plumbing on real captured audio, but is intentionally NOT used
   as the overlap behavior gate because room-colored captures can under-match
   dry TTS templates in xcorr.

Exit code:
  0 = acceptance criteria met
  1 = acceptance criteria failed
  2 = setup/input missing
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.streaming_runtime import (  # noqa: E402
    StreamingConfig,
    StreamingRuntime,
    SyntheticAttributionFrame,
)
from thalamus.attributor import SourceVerdict  # noqa: E402


SCENARIOS = ["human_before", "human_early", "human_late", "human_after", "overlap_self_first"]


def _make_test_pcm(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _run_synthetic_pre_agent_defer() -> dict:
    runtime = StreamingRuntime(StreamingConfig(
        tts_pcm=_make_test_pcm(0.6),
        tts_text="synthetic defer",
        gated_playback=True,
        output_request_s=0.30,
        listen_threshold_ms=90.0,
        cooldown_ms=120.0,
    ))
    frames: list[SyntheticAttributionFrame] = []
    for i in range(60):
        t = i * 0.03
        if t < 0.60:
            frames.append(SyntheticAttributionFrame(
                time_s=t,
                verdict=SourceVerdict.WORLD,
                is_speech=True,
                energy_db=-25.0,
            ))
        else:
            frames.append(SyntheticAttributionFrame(
                time_s=t,
                verdict=SourceVerdict.SILENCE,
                is_speech=False,
                energy_db=-70.0,
            ))
    logs = runtime.run_synthetic(frames)
    summary = runtime.get_summary()
    first_defer = next((l for l in logs if l.deferred), None)
    passed = (
        summary["deferred_events"] >= 1
        and summary["resumed_events"] >= 1
        and summary["tts_played_frames"] > 0
        and summary["tts_completed"]
        and first_defer is not None
        and first_defer.tts_position_s == 0.0
    )
    return {
        "name": "synthetic_pre_agent_defer",
        "pass": passed,
        "summary": summary,
        "first_defer_tts_position_s": None if first_defer is None else first_defer.tts_position_s,
    }


def _run_synthetic_overlap_interrupt() -> dict:
    runtime = StreamingRuntime(StreamingConfig(
        tts_pcm=_make_test_pcm(0.8),
        repair_pcm=_make_test_pcm(0.12),
        tts_text="synthetic interrupt",
        gated_playback=True,
        output_request_s=0.0,
        both_external_min_frames=2,
        both_external_energy_ratio=1.2,
        cooldown_ms=120.0,
    ))
    frames: list[SyntheticAttributionFrame] = []
    for i in range(70):
        t = i * 0.03
        if i < 5:
            frames.append(SyntheticAttributionFrame(
                time_s=t,
                verdict=SourceVerdict.SELF,
                is_speech=False,
                is_in_motor_window=True,
                self_match_score=0.9,
                self_match_score_per_frame=1.0,
                causal_score=1.0,
            ))
        elif i < 10:
            frames.append(SyntheticAttributionFrame(
                time_s=t,
                verdict=SourceVerdict.BOTH,
                is_speech=True,
                is_in_motor_window=True,
                self_match_score=0.9,
                self_match_score_per_frame=2.0,
                causal_score=1.0,
            ))
        elif i < 16:
            frames.append(SyntheticAttributionFrame(
                time_s=t,
                verdict=SourceVerdict.SILENCE,
                is_speech=False,
                is_in_motor_window=False,
                self_match_score_per_frame=0.0,
                energy_db=-70.0,
            ))
        else:
            frames.append(SyntheticAttributionFrame(
                time_s=t,
                verdict=SourceVerdict.SELF,
                is_speech=False,
                is_in_motor_window=True,
                self_match_score=0.9,
                self_match_score_per_frame=1.0,
                causal_score=1.0,
            ))
    logs = runtime.run_synthetic(frames)
    summary = runtime.get_summary()
    listen_positions = [l.tts_position_s for l in logs if l.controller_state == "LISTEN"]
    position_held = bool(listen_positions) and (max(listen_positions) - min(listen_positions) <= 0.031)
    repair_texts = [l.collision_repair_text for l in logs if l.collision_repair_text]
    repair_logs = [l for l in logs if l.output_source == "repair"]
    tts_after_repair = [l for l in logs if repair_logs and l.output_source == "tts" and l.timestamp_s > repair_logs[-1].timestamp_s]
    original_position_held_during_repair = (
        bool(repair_logs)
        and max(l.tts_position_s for l in repair_logs) == min(l.tts_position_s for l in repair_logs)
    )
    collision_repair_passed = repair_texts == ["Sorry, go ahead."]
    repair_audio_passed = (
        summary["repair_played_frames"] > 0
        and summary["repair_audio_completed"]
        and bool(tts_after_repair)
        and original_position_held_during_repair
        and all(l.controller_state != "LISTEN" for l in repair_logs)
    )
    passed = (
        summary["interrupt_events"] >= 1
        and summary["resumed_events"] >= 1
        and summary["collision_repair_events"] == 1
        and collision_repair_passed
        and repair_audio_passed
        and summary["tts_played_frames"] > 0
        and summary["tts_completed"]
        and position_held
    )
    return {
        "name": "synthetic_overlap_interrupt_resume",
        "pass": passed,
        "summary": summary,
        "listen_position_held": position_held,
        "collision_repair_passed": collision_repair_passed,
        "collision_repair_texts": repair_texts,
        "repair_audio_passed": repair_audio_passed,
        "repair_audio_frames": len(repair_logs),
        "original_position_held_during_repair": original_position_held_during_repair,
        "tts_after_repair_frames": len(tts_after_repair),
        "listen_tts_position_range_s": (
            None if not listen_positions else round(max(listen_positions) - min(listen_positions), 6)
        ),
    }


def _run_offline_audio_smoke(run_dir: Path) -> dict:
    scenario_results = []
    for scenario in SCENARIOS:
        scenario_dir = run_dir / scenario
        input_audio_path = scenario_dir / "live_captured.npy"
        agent_pcm_path = scenario_dir / "agent_pcm.npy"
        if not input_audio_path.exists() or not agent_pcm_path.exists():
            scenario_results.append({
                "scenario": scenario,
                "pass": False,
                "error": "missing live_captured.npy or agent_pcm.npy",
            })
            continue
        input_audio = np.load(input_audio_path).astype(np.float32)
        agent_pcm = np.load(agent_pcm_path).astype(np.float32)
        runtime = StreamingRuntime(StreamingConfig(
            tts_pcm=agent_pcm,
            tts_text=f"offline smoke {scenario}",
            gated_playback=True,
            m4_enable=False,
        ))
        with contextlib.redirect_stdout(sys.stderr):
            logs = runtime.run_offline(input_audio)
        summary = runtime.get_summary()
        scenario_results.append({
            "scenario": scenario,
            "pass": bool(logs and summary.get("total_frames", 0) > 0),
            "summary": summary,
        })
    return {
        "name": "offline_audio_replay_smoke",
        "pass": all(r.get("pass") for r in scenario_results),
        "note": (
            "Smoke only: room-colored captured audio validates callback/log plumbing, "
            "not overlap promotion behavior. Synthetic frames are the behavior gate."
        ),
        "scenarios": scenario_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 streaming runtime deterministic acceptance")
    parser.add_argument("--run-dir", default="data/m3_gated_playback_test")
    parser.add_argument("--output", help="Optional path to write verdict JSON")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run_dir not found: {run_dir}", file=sys.stderr)
        return 2

    checks = [
        _run_synthetic_pre_agent_defer(),
        _run_synthetic_overlap_interrupt(),
        _run_offline_audio_smoke(run_dir),
    ]
    verdict = {
        "pass": all(c["pass"] for c in checks),
        "run_dir": str(run_dir),
        "checks": checks,
        "acceptance_criteria": {
            "synthetic_pre_agent_defer_passed": checks[0]["pass"],
            "synthetic_overlap_interrupt_resume_passed": checks[1]["pass"],
            "synthetic_collision_repair_passed": checks[1].get("collision_repair_passed", False),
            "synthetic_repair_audio_passed": checks[1].get("repair_audio_passed", False),
            "offline_audio_replay_smoke_passed": checks[2]["pass"],
        },
        "interpretation": (
            "Streaming behavior is validated by synthetic attribution-frame injection; "
            "saved room-captured audio replay is retained as a callback/logging smoke test."
        ),
    }

    text = json.dumps(verdict, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)

    if verdict["pass"]:
        print("\nPASS: M3 streaming runtime acceptance criteria met.", file=sys.stderr)
        return 0
    print("\nFAIL: M3 streaming runtime acceptance criteria NOT met.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

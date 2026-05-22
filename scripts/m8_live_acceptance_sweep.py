#!/usr/bin/env python3
"""M8 Live Acceptance Sweep.

Runs 3 controlled live scenarios on hardware, logs per-frame M8 decisions,
and prints a compact acceptance summary:

  1. SELF_ONLY      – Corollary speaks, no human. M8 should stay CONTINUE.
  2. HIGH_SALIENCE  – Human speaks over Corollary. M8 should INTERRUPT/WARN.
  3. SILENCE        – No TTS, no human. M8 should stay CONTINUE only.

Usage:
    source .venv/bin/activate
    python scripts/m8_live_acceptance_sweep.py \\
        --input-device 1 \\
        --self-output-device 3 \\
        --human-output-device 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize as _synthesize_tts
from pfc.streaming_runtime import StreamingConfig, StreamingRuntime


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_tts_pcm(text: str, voice: str = "en-US-AriaNeural", sr: int = 16000) -> np.ndarray:
    import asyncio
    pcm = asyncio.run(_synthesize_tts(text, voice=voice))
    return pcm.astype(np.float32)


def play_on_device(pcm, device: int, delay_s: float, reference: float, sr: int = 16000):
    target = reference + delay_s
    def _play():
        wait = target - time.time()
        if wait > 0:
            time.sleep(wait)
        import sounddevice as sd
        sd.play(pcm, samplerate=sr, device=device)
        sd.wait()
    t = threading.Thread(target=_play, daemon=True)
    t.start()
    return t, target


def summarise(logs, label: str) -> dict:
    total = len(logs)
    if total == 0:
        return {"scenario": label, "total_frames": 0}

    action_counts = {}
    for log in logs:
        a = getattr(log, "m8_action", None) or "NONE"
        action_counts[a] = action_counts.get(a, 0) + 1

    state_counts = {}
    for log in logs:
        s = getattr(log, "controller_state", "?")
        state_counts[s] = state_counts.get(s, 0) + 1

    interrupt_frames = sum(1 for l in logs if getattr(l, "should_interrupt", False))
    non_continue = [l for l in logs if getattr(l, "m8_action", None) not in (None, "CONTINUE", "NONE")]

    sample = [
        {
            "frame": l.frame_index,
            "verdict": l.verdict,
            "state": l.controller_state,
            "action": l.m8_action,
            "score": l.m8_score,
            "urgency": l.m8_urgency,
            "reason": l.m8_reason,
        }
        for l in non_continue[:5]
    ]

    return {
        "scenario": label,
        "total_frames": total,
        "action_counts": action_counts,
        "state_counts": state_counts,
        "interrupt_frames": interrupt_frames,
        "non_continue_frames": len(non_continue),
        "sample_decisions": sample,
    }


def run_scenario(name: str, config: StreamingConfig, duration_s: float,
                 human_pcm=None, human_device: int = None) -> list:
    """Run one scenario and return frame logs."""
    runtime = StreamingRuntime(config)
    human_thread = None

    if human_pcm is not None and human_device is not None:
        ref = time.time() + 2.0
        human_thread, _ = play_on_device(human_pcm, human_device, delay_s=2.0, reference=ref, sr=config.sr)
        print(f"  Human audio scheduled at t=2s (device {human_device})")
        time.sleep(2)

    print(f"  Running for {duration_s}s...")
    try:
        runtime.run(duration_s=duration_s)
    except KeyboardInterrupt:
        print("  Interrupted")

    if human_thread is not None:
        human_thread.join(timeout=2.0)

    return runtime._logs


def check_acceptance(results: list[dict]) -> list[str]:
    checks = []
    for r in results:
        label = r["scenario"]
        ac = r.get("action_counts", {})

        if label == "SELF_ONLY":
            bad = ac.get("INTERRUPT", 0) + ac.get("ESCALATE", 0)
            checks.append(f"  SELF_ONLY    : interrupts+escalates={bad}  {'PASS' if bad == 0 else 'FAIL'}")

        elif label == "HIGH_SALIENCE":
            good = ac.get("INTERRUPT", 0) + ac.get("ESCALATE", 0) + ac.get("WARN", 0)
            checks.append(f"  HIGH_SALIENCE: fast-track decisions={good}  {'PASS' if good > 0 else 'FAIL'}")

        elif label == "SILENCE":
            bad = ac.get("INTERRUPT", 0) + ac.get("ESCALATE", 0) + ac.get("WARN", 0) + ac.get("DEFER", 0)
            checks.append(f"  SILENCE      : false fast-tracks={bad}  {'PASS' if bad == 0 else 'FAIL'}")

    return checks


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="M8 Live Acceptance Sweep")
    parser.add_argument("--input-device", type=int, required=True)
    parser.add_argument("--self-output-device", type=int, required=True)
    parser.add_argument("--human-output-device", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--callback-blocksize-ms", type=float, default=120.0)
    parser.add_argument("--m4-mode", choices=["auto", "bank", "scorer", "arbitration"], default="arbitration")
    parser.add_argument("--m4-scorer", type=str,
                        default="data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt")
    parser.add_argument("--m4-bank", type=str,
                        default="data/m4_prototype_bank_round4_20260520/prototype_bank.npz")
    parser.add_argument("--room-filter", type=str,
                        default="data/m3_broca_room_filter_production.npy")
    parser.add_argument("--output-dir", type=str, default="/tmp/m8_sweep")
    parser.add_argument("--scenarios", nargs="+",
                        default=["SELF_ONLY", "HIGH_SALIENCE", "SILENCE"],
                        choices=["SELF_ONLY", "HIGH_SALIENCE", "SILENCE"])
    args = parser.parse_args()

    if "HIGH_SALIENCE" in args.scenarios and not args.human_output_device:
        print("ERROR: HIGH_SALIENCE requires --human-output-device")
        sys.exit(1)

    sr = 16000
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ROOT = Path(__file__).resolve().parents[1]

    print("=" * 60)
    print("M8 Live Acceptance Sweep")
    print("=" * 60)
    print(f"  Input device:       {args.input_device}")
    print(f"  Self output device: {args.self_output_device}")
    print(f"  Human device:       {args.human_output_device or 'N/A'}")
    print(f"  M4 mode:            {args.m4_mode}")
    print(f"  Scenarios:          {', '.join(args.scenarios)}")
    print(f"  Duration each:      {args.duration_s}s")
    print(f"  Output dir:         {out_dir}")
    print()

    results = []

    for scenario in args.scenarios:
        print(f"{'─' * 40}")
        print(f"  Scenario: {scenario}")
        print(f"{'─' * 40}")

        if scenario == "SELF_ONLY":
            self_text = "This is a self-only M8 acceptance test. I am speaking alone."
            self_pcm = make_tts_pcm(self_text, voice="en-US-AriaNeural", sr=sr)
            config = StreamingConfig(
                tts_pcm=self_pcm,
                tts_text="self only",
                input_device=args.input_device,
                output_device=args.self_output_device,
                sr=sr,
                callback_blocksize_ms=args.callback_blocksize_ms,
                gated_playback=True,
                m4_enable=True,
                m4_mode=args.m4_mode,
                m4_scorer_path=str(ROOT / args.m4_scorer),
                m4_bank_path=str(ROOT / args.m4_bank),
                m4_self_threshold=0.99,
                room_filter_path=str(ROOT / args.room_filter),
                room_local_threshold=0.0747,
                energy_threshold_db=-55.0,
                similarity_threshold=0.05,
                causal_tolerance_ms=175.0,
                both_external_min_frames=3,
                both_external_energy_ratio=1.5,
                cooldown_ms=200.0,
                m8_enable=True,
                save_path=out_dir / "self_only.jsonl",
            )
            logs = run_scenario(scenario, config, args.duration_s)

        elif scenario == "HIGH_SALIENCE":
            self_text = "Corollary speaking normally during this high salience test."
            human_text = "Hey Corollary, I need to interrupt you right now."
            self_pcm = make_tts_pcm(self_text, voice="en-US-AriaNeural", sr=sr)
            human_pcm = make_tts_pcm(human_text, voice="en-US-GuyNeural", sr=sr)
            config = StreamingConfig(
                tts_pcm=self_pcm,
                tts_text="high salience",
                input_device=args.input_device,
                output_device=args.self_output_device,
                sr=sr,
                callback_blocksize_ms=args.callback_blocksize_ms,
                gated_playback=True,
                m4_enable=True,
                m4_mode=args.m4_mode,
                m4_scorer_path=str(ROOT / args.m4_scorer),
                m4_bank_path=str(ROOT / args.m4_bank),
                m4_self_threshold=0.99,
                room_filter_path=str(ROOT / args.room_filter),
                room_local_threshold=0.0747,
                energy_threshold_db=-55.0,
                similarity_threshold=0.05,
                causal_tolerance_ms=175.0,
                both_external_min_frames=3,
                both_external_energy_ratio=1.5,
                cooldown_ms=200.0,
                m8_enable=True,
                save_path=out_dir / "high_salience.jsonl",
            )
            logs = run_scenario(scenario, config, args.duration_s,
                                human_pcm=human_pcm, human_device=args.human_output_device)

        elif scenario == "SILENCE":
            config = StreamingConfig(
                tts_pcm=np.zeros(sr, dtype=np.float32),
                tts_text="",
                input_device=args.input_device,
                output_device=args.self_output_device,
                sr=sr,
                callback_blocksize_ms=args.callback_blocksize_ms,
                gated_playback=True,
                m4_enable=True,
                m4_mode=args.m4_mode,
                m4_scorer_path=str(ROOT / args.m4_scorer),
                m4_bank_path=str(ROOT / args.m4_bank),
                m4_self_threshold=0.99,
                room_filter_path=str(ROOT / args.room_filter),
                room_local_threshold=0.0747,
                energy_threshold_db=-55.0,
                similarity_threshold=0.05,
                causal_tolerance_ms=175.0,
                both_external_min_frames=3,
                both_external_energy_ratio=1.5,
                cooldown_ms=200.0,
                m8_enable=True,
                save_path=out_dir / "silence.jsonl",
            )
            logs = run_scenario(scenario, config, min(args.duration_s, 4.0))

        summary = summarise(logs, scenario)
        results.append(summary)
        print(f"  Frames: {summary['total_frames']}")
        print(f"  Actions: {summary['action_counts']}")
        print(f"  States:  {summary['state_counts']}")
        if summary.get("sample_decisions"):
            for d in summary["sample_decisions"]:
                print(f"    frame {d['frame']:4d} | {d['action']:>10s} | score={d['score']:.3f} | "
                      f"urgency={d['urgency']:4s} | verdict={d['verdict']:6s} | state={d['state']:6s}")
        print()

    # ── Acceptance Summary ──────────────────────────────────────────────────
    print("=" * 60)
    print("Acceptance Summary")
    print("=" * 60)
    acceptance = check_acceptance(results)
    for line in acceptance:
        print(line)

    overall = all("PASS" in line for line in acceptance)
    print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")
    print(f"  Logs in: {out_dir}")
    print()

    # Save JSON summary
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({"results": results, "acceptance": acceptance,
                                         "overall": "PASS" if overall else "FAIL"}, indent=2))
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()

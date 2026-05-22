#!/usr/bin/env python3
"""M8 Multi-Turn Conversation Test.

Simulates a back-and-forth conversation between Corollary and a human speaker
to test M8 fast-track behavior across turn boundaries.

Protocol:
  1. Corollary speaks a long utterance (self-only, M8 should stay CONTINUE)
  2. Human interrupts at ~3s (M8 should WARN/INTERRUPT)
  3. Corollary pauses, human continues (M8 should DEFER)
  4. Human stops, Corollary resumes (M8 should return to CONTINUE)
  5. Repeat for 2-3 turns

Usage:
    source .venv/bin/activate
    python scripts/m8_multi_turn_test.py \\
        --input-device 1 \\
        --self-output-device 3 \\
        --human-output-device 2 \\
        --turns 3
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


def analyse_turns(logs, turn_boundaries):
    """Analyse M8 behaviour at each turn boundary."""
    results = []
    for i, (start_s, end_s, label, expected) in enumerate(turn_boundaries):
        frames = [l for l in logs if start_s <= l.timestamp_s < end_s]
        if not frames:
            results.append({"turn": i, "label": label, "frames": 0, "actions": {}, "pass": "SKIP"})
            continue

        actions = {}
        for f in frames:
            a = f.m8_action or "NONE"
            actions[a] = actions.get(a, 0) + 1

        # Determine pass/fail based on expected behaviour
        pass_fail = "PASS"
        if "no_interrupt" in expected:
            bad = actions.get("INTERRUPT", 0) + actions.get("ESCALATE", 0)
            if bad > 0:
                pass_fail = "FAIL"
        if "should_warn" in expected:
            good = actions.get("WARN", 0) + actions.get("INTERRUPT", 0) + actions.get("ESCALATE", 0)
            if good == 0:
                pass_fail = "FAIL"
        if "should_defer" in expected:
            good = actions.get("DEFER", 0) + actions.get("WARN", 0)
            if good == 0:
                pass_fail = "FAIL"
        if "should_continue_or_react" in expected:
            # During resume, human audio may still be playing.
            # Some INTERRUPT/ESCALATE is expected if human is still speaking.
            # Only fail if the entire segment is dominated by fast-track actions.
            total = len(frames)
            bad = actions.get("INTERRUPT", 0) + actions.get("ESCALATE", 0)
            if total > 0 and (bad / total) > 0.1:  # More than 10% fast-track
                pass_fail = "FAIL"

        results.append({
            "turn": i,
            "label": label,
            "frames": len(frames),
            "actions": actions,
            "expected": expected,
            "pass": pass_fail,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="M8 Multi-Turn Conversation Test")
    parser.add_argument("--input-device", type=int, required=True)
    parser.add_argument("--self-output-device", type=int, required=True)
    parser.add_argument("--human-output-device", type=int, default=None)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--m4-mode", choices=["auto", "bank", "scorer", "arbitration"], default="arbitration")
    parser.add_argument("--m4-scorer", type=str,
                        default="data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt")
    parser.add_argument("--m4-bank", type=str,
                        default="data/m4_prototype_bank_round4_20260520/prototype_bank.npz")
    parser.add_argument("--room-filter", type=str,
                        default="data/m3_broca_room_filter_production.npy")
    parser.add_argument("--output-dir", type=str, default="/tmp/m8_multi_turn")
    args = parser.parse_args()

    if not args.human_output_device:
        print("ERROR: --human-output-device is required")
        sys.exit(1)

    sr = 16000
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ROOT = Path(__file__).resolve().parents[1]

    # Build conversation segments
    self_voice = "en-US-AriaNeural"
    human_voice = "en-US-GuyNeural"

    self_utterances = [
        "Let me explain how the self-listening pipeline works. First, the cochlea captures audio from the microphone.",
        "Now, the thalamus acts as a gate, using the forward model to predict what the agent will hear from itself.",
        "Finally, the prefrontal cortex makes decisions about when to speak and when to listen based on the attribution results.",
    ]
    human_utterances = [
        "Wait, can you clarify the gating mechanism?",
        "Okay, but how does the forward model adapt to different room acoustics?",
        "That makes sense. What happens when there are multiple speakers?",
    ]

    # Generate all TTS upfront
    print("Synthesizing TTS...")
    self_pcms = [make_tts_pcm(t, self_voice, sr) for t in self_utterances[:args.turns]]
    human_pcms = [make_tts_pcm(t, human_voice, sr) for t in human_utterances[:args.turns]]

    # Build a long self-TTS stream with gaps for human interruption
    # Each turn: self speaks for ~8s, human interrupts at ~3s for ~4s, gap ~2s, self resumes
    turn_duration = 16  # seconds per turn
    total_duration = turn_duration * args.turns

    # We'll use the streaming runtime with the first self utterance as the main TTS
    # and schedule human audio on the second device at the right times
    # For a proper multi-turn test, we concatenate self utterances with silence gaps

    gap_samples = int(2.0 * sr)  # 2s silence between utterances
    combined_self = np.array([], dtype=np.float32)
    for i, pcm in enumerate(self_pcms):
        combined_self = np.concatenate([combined_self, pcm])
        if i < len(self_pcms) - 1:
            combined_self = np.concatenate([combined_self, np.zeros(gap_samples, dtype=np.float32)])

    print(f"  Combined self audio: {len(combined_self)/sr:.1f}s")
    print(f"  Total test duration: {total_duration}s")
    print()

    config = StreamingConfig(
        tts_pcm=combined_self,
        tts_text="multi turn conversation",
        input_device=args.input_device,
        output_device=args.self_output_device,
        sr=sr,
        callback_blocksize_ms=120.0,
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
        save_path=out_dir / "multi_turn.jsonl",
    )

    runtime = StreamingRuntime(config)

    # Schedule human audio for each turn
    # Human interrupts at turn_start + 3s, speaks for ~4s
    human_threads = []
    ref_time = time.time() + 3.0  # 3s countdown before start

    print("Scheduling human interruptions...")
    for turn in range(args.turns):
        turn_start = turn * turn_duration
        human_delay = turn_start + 3.0  # interrupt 3s into each turn
        t, _ = play_on_device(
            human_pcms[turn], args.human_output_device,
            delay_s=human_delay, reference=ref_time, sr=sr
        )
        human_threads.append(t)
        print(f"  Turn {turn+1}: human at t={human_delay:.1f}s ({human_pcms[turn].shape[0]/sr:.1f}s duration)")

    # Countdown
    print(f"\nStarting in 3 seconds...")
    time.sleep(3)

    print(f"Running multi-turn conversation for {total_duration}s...")
    stream_start = time.time()
    runtime.run(duration_s=total_duration)
    elapsed = time.time() - stream_start

    for t in human_threads:
        t.join(timeout=2.0)

    logs = runtime._logs
    print(f"  Completed in {elapsed:.1f}s, {len(logs)} frames")

    # Define turn boundaries for analysis
    turn_boundaries = []
    for turn in range(args.turns):
        t_start = turn * turn_duration
        t_self_end = t_start + 3.0  # self speaking before interrupt
        t_human_start = t_start + 3.0  # human starts
        t_human_end = t_start + 7.0  # human finishes
        t_resume = t_start + 7.0  # self resumes
        t_end = (turn + 1) * turn_duration

        turn_boundaries.append((t_start, t_self_end, f"turn_{turn+1}_self", "should_continue no_interrupt"))
        turn_boundaries.append((t_human_start, t_human_end, f"turn_{turn+1}_human", "should_warn should_defer"))
        turn_boundaries.append((t_resume, t_end, f"turn_{turn+1}_resume", "should_continue_or_react"))

    # Analyse
    results = analyse_turns(logs, turn_boundaries)

    print(f"\n{'=' * 60}")
    print("Multi-Turn M8 Analysis")
    print("=" * 60)

    overall = True
    for r in results:
        status = "PASS" if r["pass"] == "PASS" else "FAIL" if r["pass"] == "FAIL" else "SKIP"
        if r["pass"] == "FAIL":
            overall = False
        print(f"\n  {r['label']:30s} | {status}")
        print(f"    Frames: {r['frames']}, Actions: {r.get('actions', {})}")
        if r['frames'] > 0 and r.get('actions'):
            # Show sample decisions
            segment_frames = [l for l in logs
                              if r['label'].split('_')[0] == 'turn' and
                              any(l.m8_action == a for a in ['WARN', 'INTERRUPT', 'ESCALATE', 'DEFER'])]
            shown = 0
            for l in segment_frames[:3]:
                if l.m8_action in ('WARN', 'INTERRUPT', 'ESCALATE', 'DEFER'):
                    print(f"    frame {l.frame_index:4d} | {l.m8_action:>10s} | score={l.m8_score:.3f} | "
                          f"verdict={l.verdict:6s} | state={l.controller_state:6s}")
                    shown += 1
            if shown == 0:
                print(f"    (all CONTINUE in this segment)")

    # Overall action distribution
    all_actions = {}
    for l in logs:
        a = l.m8_action or "NONE"
        all_actions[a] = all_actions.get(a, 0) + 1

    print(f"\n  Overall action distribution:")
    for action, count in sorted(all_actions.items(), key=lambda x: -x[1]):
        pct = count / len(logs) * 100
        print(f"    {action:>10s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")
    print(f"  Logs in: {out_dir}")

    # Save summary
    summary = {
        "turns": args.turns,
        "total_frames": len(logs),
        "elapsed_s": elapsed,
        "turn_results": results,
        "action_distribution": all_actions,
        "overall": "PASS" if overall else "FAIL",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

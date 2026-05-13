#!/usr/bin/env python3
"""Conversational speaker registration — version 2.

Uses the streaming runtime's built-in speaker identification to accumulate
evidence about who is speaking. After enough turns, the accumulated
speaker_best_match data is used to register the speaker.

This version works with the existing StreamingFrameLog schema (no frame_audio needed).

Usage:
  .venv/bin/python scripts/conversational_registration_v2.py \\
      --input-device 1 --output-device 5 \\
      --speaker-name "User"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.streaming_runtime import StreamingRuntime, StreamingConfig, StreamingFrameLog
from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter
from pfc.world_model import SpeakerRegistry, MultiSpeakerWorldModel
from broca.tts import synthesize

SR = 16000


def analyze_registration_readiness(
    all_logs: list[StreamingFrameLog],
    registry: SpeakerRegistry,
) -> tuple[bool, str, float, int]:
    """Analyze accumulated logs to determine if we have enough evidence to register.

    Looks at frames where speaker_match_passed was True (i.e., the runtime
    found a matching speaker). If we see consistent matches for the same
    speaker_id across multiple frames, we have a reliable registration.

    Returns (ready, best_speaker_id, confidence, matched_frame_count).
    """
    # Count matched frames per speaker_id
    match_counts: dict[str, int] = {}
    match_sims: dict[str, float] = {}

    for log_entry in all_logs:
        sid = log_entry.speaker_id
        if sid is None or sid == "self":
            continue
        sim = log_entry.speaker_match_passed
        if sim is None:
            sim = 0.0
        match_counts[sid] = match_counts.get(sid, 0) + 1
        if sid not in match_sims:
            match_sims[sid] = 0.0
        if log_entry.speaker_best_match_similarity:
            match_sims[sid] += log_entry.speaker_best_match_similarity

    if not match_counts:
        return False, "", 0.0, 0

    # Find the speaker with the most matched frames
    best_id = max(match_counts, key=lambda k: match_counts[k])
    best_count = match_counts[best_id]
    avg_sim = match_sims[best_id] / best_count if best_count > 0 else 0.0

    # Need at least 30 matched frames (~3 seconds) and avg sim > 0.3
    ready = best_count >= 30 and avg_sim > 0.3

    return ready, best_id, avg_sim, best_count


async def main() -> None:
    parser = argparse.ArgumentParser(description="Conversational registration v2")
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--speaker-name", type=str, default="User")
    parser.add_argument("--opener-duration", type=float, default=6.0)
    parser.add_argument("--output", type=str, default="/tmp/conversational_registration_v2.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Conversational Registration v2 — Streaming Runtime")
    print("=" * 60)

    adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05, adaptation_rate=0.1)

    openers = [
        "Hello! I'm Corollary. What's your name?",
        "Nice to meet you! How's your day going?",
        "That's wonderful. What are you working on today?",
        "Is there anything specific you'd like to talk about?",
    ]

    registered = False
    registration_turn = 0
    all_logs: list[StreamingFrameLog] = []

    for i, opener_text in enumerate(openers):
        print(f"\n--- Turn {i + 1}/{len(openers)} ---")

        # Synthesize TTS
        tts_pcm = (await synthesize(opener_text)).astype(np.float32)
        print(f'  TTS: "{opener_text}" ({len(tts_pcm)/SR:.2f}s)')

        config = StreamingConfig(
            tts_pcm=tts_pcm,
            tts_text=opener_text,
            input_device=args.input_device,
            output_device=args.output_device,
            m4_enable=True,
            gated_playback=False,
            m4_self_threshold=0.10,
            m4_min_buffer_ms=300.0,
            speaker_threshold_adapter=adapter,
        )

        runtime = StreamingRuntime(config)
        logs = runtime.run(duration_s=args.opener_duration)
        all_logs.extend(logs)

        # Analyze this turn
        verdicts = {}
        for l in logs:
            verdicts[l.verdict] = verdicts.get(l.verdict, 0) + 1
        m4_attempted = sum(1 for l in logs if l.m4_outcome not in ("not_attempted", None))
        m4_accepted = sum(1 for l in logs if l.m4_outcome == "accepted")

        # Count human speech frames
        human_frames = sum(1 for l in logs if l.verdict in ("WORLD", "BOTH") and l.vad_speech)
        matched_frames = sum(1 for l in logs if l.speaker_id is not None and l.speaker_id != "self")

        print(f"  Frames: {len(logs)} | Verdicts: {verdicts}")
        print(f"  M4: {m4_attempted} att, {m4_accepted} ok | Human speech: {human_frames} frames")
        print(f"  Speaker-matched frames: {matched_frames}")

        # Check registration readiness
        if not registered:
            ready, best_id, conf, count = analyze_registration_readiness(
                all_logs, runtime._speaker_world_model.registry if runtime._speaker_world_model else None  # type: ignore[arg-type]
            )
            # Also check: do we have enough human speech and consistent matches?
            total_human = sum(1 for l in all_logs if l.verdict in ("WORLD", "BOTH") and l.vad_speech)
            if total_human >= 30 and matched_frames >= 5:
                # Register using the runtime's accumulated evidence
                # Use the speaker_id from the most recent matched frames
                recent_matches = [
                    l.speaker_best_match_id
                    for l in logs[-50:]
                    if l.speaker_best_match_id and l.speaker_match_passed
                ]
                if recent_matches:
                    from collections import Counter
                    most_common = Counter(recent_matches).most_common(1)[0]
                    if most_common[1] >= 3:
                        registered = True
                        registration_turn = i + 1
                        print(f"\n  *** Registration evidence found on turn {registration_turn}! ***")
                        print(f"  Most common match: {most_common[0]} ({most_common[1]} frames)")

    # Final summary
    total_human = sum(1 for l in all_logs if l.verdict in ("WORLD", "BOTH") and l.vad_speech)
    total_matched = sum(1 for l in all_logs if l.speaker_id is not None and l.speaker_id != "self")

    all_verdicts = {}
    for l in all_logs:
        all_verdicts[l.verdict] = all_verdicts.get(l.verdict, 0) + 1

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Registered:       {registered}")
    if registered:
        print(f"  Turn registered:  {registration_turn}")
    print(f"  Total turns:      {len(openers)}")
    print(f"  Total frames:     {len(all_logs)}")
    print(f"  Human speech:     {total_human} frames")
    print(f"  Speaker-matched:  {total_matched} frames")
    print(f"  Verdict dist:     {all_verdicts}")

    results = {
        "registered": registered,
        "registration_turn": registration_turn,
        "speaker_name": args.speaker_name,
        "total_turns": len(openers),
        "total_frames": len(all_logs),
        "human_speech_frames": total_human,
        "speaker_matched_frames": total_matched,
        "verdict_distribution": all_verdicts,
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Conversational registration integrated with the streaming runtime.

This extends the standard streaming runtime with automatic speaker registration
during the first few turns of conversation. Instead of separate play/capture
calls, it uses the runtime's own audio pipeline — TTS output goes through
device 5, mic input comes from device 1, and the attribution pipeline classifies
each frame. When enough WORLD speech is accumulated from an unknown speaker,
it extracts an embedding and registers them.

This is the "production" version of conversational registration.

Usage:
  .venv/bin/python scripts/conversational_registration_runtime.py \\
      --input-device 1 --output-device 5
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
from stg.speaker import SpeakerIdentifier
from pfc.world_model import SpeakerRegistry, MultiSpeakerWorldModel
from broca.tts import synthesize

SR = 16000
MIN_WORLD_SAMPLES = int(2.0 * SR)  # 2 seconds of WORLD speech for registration


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


async def generate_opener_pcm(text: str) -> np.ndarray:
    """Synthesize TTS for an opener utterance."""
    return (await synthesize(text)).astype(np.float32)


def try_register_speaker(
    logs: list[StreamingFrameLog],
    registry: SpeakerRegistry,
    identifier: SpeakerIdentifier,
    speaker_name: str,
) -> tuple[bool, str, float]:
    """Try to register a speaker from accumulated WORLD/BOTH frames.

    Returns (success_or_updated, speaker_id, best_similarity).
    """
    # Extract WORLD/BOTH audio from frame logs
    world_audio = []
    for log_entry in logs:
        if log_entry.verdict in ("WORLD", "BOTH") and log_entry.frame_audio is not None:
            world_audio.append(log_entry.frame_audio)

    if not world_audio:
        return False, "", 0.0

    combined = np.concatenate(world_audio)
    if len(combined) < MIN_WORLD_SAMPLES:
        return False, "", 0.0

    try:
        embedding = identifier.get_embedding(combined)
    except Exception as e:
        print(f"  Embedding extraction failed: {e}")
        return False, "", 0.0

    # Check against existing speakers
    best_id, best_sim = registry.identify_speaker(embedding)
    if best_id is not None:
        # Update existing speaker
        profile = registry.get_profile(best_id)
        if profile:
            profile.update_embedding(embedding)
        return True, best_id, best_sim

    # Register new speaker
    speaker_id = f"human_{speaker_name.lower().replace(' ', '_')}"
    registry.register_speaker(speaker_id, speaker_name, embedding=embedding)
    return True, speaker_id, 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversational registration via streaming runtime")
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--speaker-name", type=str, default="User")
    parser.add_argument("--opener-duration", type=float, default=6.0, help="Duration per opener turn")
    parser.add_argument("--output", type=str, default="/tmp/conversational_registration_rt.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Conversational Registration — Streaming Runtime")
    print("=" * 60)

    # Set up registry
    adapter = SpeakerThresholdAdapter(base_threshold=0.10, margin=0.05, adaptation_rate=0.1)
    adapter.register_self("self")

    identifier = SpeakerIdentifier()
    registry = SpeakerRegistry(identifier=identifier, similarity_threshold=0.1)
    registry.register_speaker("self", "self")
    world_model = MultiSpeakerWorldModel(registry=registry)

    # Define openers
    openers = [
        "Hello! I'm Corollary. What's your name?",
        "Nice to meet you! How's your day going?",
        "That's wonderful. What are you working on today?",
    ]

    registered = False
    registration_turn = 0
    all_logs: list[StreamingFrameLog] = []
    turn_logs: list[StreamingFrameLog] = []
    last_registry = registry  # Track the last runtime's registry

    for i, opener_text in enumerate(openers):
        print(f"\n--- Turn {i + 1}/{len(openers)} ---")

        # Synthesize TTS
        tts_pcm = asyncio.run(generate_opener_pcm(opener_text))
        print(f'  Corollary: "{opener_text}"')
        print(f"  TTS duration: {len(tts_pcm)/SR:.2f}s")

        # Create streaming config for this turn
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

        # The runtime creates its own SpeakerRegistry when speaker_threshold_adapter is set.
        # Access it to register speakers and perform identification.
        rt_registry = runtime._speaker_world_model.registry
        rt_identifier = runtime._speaker_identifier
        last_registry = rt_registry

        # Run the streaming loop for this turn
        print(f"  Running streaming loop ({args.opener_duration}s)...")
        logs = runtime.run(duration_s=args.opener_duration)
        turn_logs = logs
        all_logs.extend(logs)

        # Analyze the turn
        verdicts = {}
        for l in logs:
            verdicts[l.verdict] = verdicts.get(l.verdict, 0) + 1
        m4_attempted = sum(1 for l in logs if l.m4_outcome not in ("not_attempted", None))
        m4_accepted = sum(1 for l in logs if l.m4_outcome == "accepted")

        print(f"  Frames: {len(logs)}, Verdicts: {verdicts}")
        print(f"  M4: {m4_attempted} attempted, {m4_accepted} accepted")

        # Try registration from all accumulated logs
        if not registered:
            registered_now, speaker_id, sim = try_register_speaker(
                all_logs, rt_registry, rt_identifier, args.speaker_name
            )
            if registered_now:
                registered = True
                registration_turn = i + 1
                print(f"\n  *** Speaker registered: {speaker_id} (sim={sim:.4f}) ***")

        if registered:
            # Try to identify the speaker in this turn
            recent_world = []
            for l in turn_logs:
                if l.verdict in ("WORLD", "BOTH") and l.frame_audio is not None:
                    recent_world.append(l.frame_audio)
            if recent_world:
                combined = np.concatenate(recent_world)
                if len(combined) >= int(1.0 * SR):
                    try:
                        emb = identifier.get_embedding(combined)
                        best_id, best_sim = rt_registry.identify_speaker(emb)
                        if best_id:
                            print(f"  Identification: {best_id} (sim={best_sim:.4f})")
                        else:
                            print(f"  Identification: no match (best={best_sim:.4f})")
                    except Exception:
                        pass

    # Summary
    print(f"\n{'=' * 60}")
    print("Registration Summary")
    print(f"{'=' * 60}")
    print(f"  Registered:       {registered}")
    if registered:
        print(f"  Turn registered:  {registration_turn}")
        print(f"  Speaker:          human_{args.speaker_name.lower().replace(' ', '_')}")
    print(f"  Total frames:     {len(all_logs)}")
    # Use the last runtime's registry for the summary
    summary_registry = last_registry

    print(f"  Known speakers:   {summary_registry.known_speakers}")

    # Final verdict distribution on all logs
    all_verdicts = {}
    for l in all_logs:
        all_verdicts[l.verdict] = all_verdicts.get(l.verdict, 0) + 1
    print(f"  All verdicts:     {all_verdicts}")

    results = {
        "registered": registered,
        "registration_turn": registration_turn,
        "speaker_name": args.speaker_name,
        "speaker_id": f"human_{args.speaker_name.lower().replace(' ', '_')}" if registered else None,
        "total_turns": len(openers),
        "total_frames": len(all_logs),
        "known_speakers": summary_registry.known_speakers,
        "verdict_distribution": all_verdicts,
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()

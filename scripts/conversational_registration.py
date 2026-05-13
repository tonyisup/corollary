#!/usr/bin/env python3
"""Conversational speaker registration prototype.

Demonstrates organic speaker registration through the first few turns of
conversation. Corollary opens with canned elicitation questions, captures
the human's responses, and registers the speaker — no separate training
phase needed.

Flow:
  1. Corollary plays opener TTS through device 5
  2. Human responds (captured on device 1)
  3. System accumulates WORLD/BOTH audio from human responses
  4. Once enough clean speech is collected, extract embedding and register
  5. Continue conversation with speaker identified

Usage:
  .venv/bin/python scripts/conversational_registration.py \\
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

import sounddevice as sd
from pfc.streaming_runtime import StreamingRuntime, StreamingConfig
from thalamus.speaker_threshold_adapter import SpeakerThresholdAdapter
from stg.speaker import SpeakerIdentifier
from pfc.world_model import SpeakerRegistry, MultiSpeakerWorldModel
from broca.tts import synthesize

SR = 16000
MIN_REGISTRATION_SAMPLES = int(2.0 * SR)  # 2 seconds of speech minimum


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


async def play_tts(text: str, device: int) -> np.ndarray:
    """Synthesize and play TTS, returning the PCM."""
    pcm = (await synthesize(text)).astype(np.float32)
    print(f'  Corollary: "{text}"')
    sd.play(pcm, samplerate=SR, device=device, blocking=True)
    return pcm


def capture_response(device: int, duration: float = 5.0) -> np.ndarray:
    """Capture audio from the mic for a fixed duration."""
    print(f"  Listening for response ({duration:.0f}s)...")
    audio = sd.rec(
        int(duration * SR),
        samplerate=SR,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio.flatten()


def extract_speech_segments(
    audio: np.ndarray,
    identifier: SpeakerIdentifier,
    min_rms_db: float = -50.0,
) -> list[np.ndarray]:
    """Extract segments of speech from captured audio using VAD-like energy gating."""
    frame_samples = int(SR * 0.03)  # 30ms frames
    segments = []
    current_segment = []

    for i in range(0, len(audio) - frame_samples, frame_samples):
        frame = audio[i : i + frame_samples]
        rms_db = 20 * np.log10(max(float(np.sqrt(np.mean(frame ** 2))), 1e-10))
        if rms_db > min_rms_db:
            current_segment.append(frame)
        else:
            if current_segment:
                segments.append(np.concatenate(current_segment))
                current_segment = []

    if current_segment:
        segments.append(np.concatenate(current_segment))

    return segments


def register_from_audio(
    audio: np.ndarray,
    speaker_name: str,
    registry: SpeakerRegistry,
    identifier: SpeakerIdentifier,
) -> tuple[bool, float]:
    """Attempt to register a speaker from captured audio.

    Returns (success, best_similarity).
    """
    segments = extract_speech_segments(audio, identifier)

    if not segments:
        print("  No speech detected in capture")
        return False, 0.0

    # Concatenate all speech segments
    all_speech = np.concatenate(segments)
    total_duration = len(all_speech) / SR
    print(f"  Detected {len(segments)} speech segments, {total_duration:.1f}s total")

    if len(all_speech) < MIN_REGISTRATION_SAMPLES:
        print(f"  Insufficient speech ({total_duration:.1f}s < {MIN_REGISTRATION_SAMPLES/SR:.1f}s minimum)")
        return False, 0.0

    # Extract embedding from the full speech
    try:
        embedding = identifier.get_embedding(all_speech)
    except Exception as e:
        print(f"  Embedding extraction failed: {e}")
        return False, 0.0

    # Check if this speaker is already registered
    existing_id = f"human_{speaker_name.lower().replace(' ', '_')}"
    if existing_id in registry.known_speakers:
        profile = registry.get_profile(existing_id)
        if profile and profile.embedding is not None:
            sim = cosine_sim(embedding, profile.embedding)
            print(f"  Speaker already registered (existing sim={sim:.4f}), updating...")
            profile.update_embedding(embedding)
            return True, sim

    # Register new speaker
    registry.register_speaker(existing_id, speaker_name, embedding=embedding)
    print(f"  Registered new speaker: {existing_id}")
    return True, 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversational speaker registration")
    parser.add_argument("--input-device", type=int, default=1, help="Mic input device")
    parser.add_argument("--output-device", type=int, default=5, help="Speaker output device")
    parser.add_argument("--speaker-name", type=str, default="User", help="Name for the registered speaker")
    parser.add_argument("--openers", type=str, nargs="+", default=[
        "Hello! I'm Corollary. What's your name?",
        "Nice to meet you! How's your day going?",
        "That's great! What brings you here today?",
    ], help="Canned opener questions")
    parser.add_argument("--response-duration", type=float, default=5.0, help="Max response capture duration per opener")
    parser.add_argument("--output", type=str, default="/tmp/conversational_registration.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Conversational Speaker Registration")
    print("=" * 60)
    print(f"  Input device:  {args.input_device} (BL-2021USB)")
    print(f"  Output device: {args.output_device} (Mac mini Speakers)")
    print(f"  Speaker name:  {args.speaker_name}")
    print(f"  Openers:       {len(args.openers)}")
    print()

    # Set up the speaker registry
    identifier = SpeakerIdentifier()
    registry = SpeakerRegistry(identifier=identifier, similarity_threshold=0.1)
    registry.register_speaker("self", "self")
    world_model = MultiSpeakerWorldModel(registry=registry)

    # Accumulated human speech for registration
    accumulated_speech: list[np.ndarray] = []
    registered = False
    registration_turn = 0

    # Play openers and capture responses
    for i, opener in enumerate(args.openers):
        print(f"\n--- Turn {i + 1}/{len(args.openers)} ---")

        # Play opener
        asyncio.run(play_tts(opener, args.output_device))

        # Small gap
        time.sleep(0.5)

        # Capture response
        response = capture_response(args.input_device, args.response_duration)
        accumulated_speech.append(response)

        # Check if we have enough speech to register
        total_samples = sum(len(s) for s in accumulated_speech)
        total_duration = total_samples / SR
        print(f"  Accumulated speech: {total_duration:.1f}s")

        if not registered and total_duration >= 2.0:
            all_audio = np.concatenate(accumulated_speech)
            success, sim = register_from_audio(
                all_audio, args.speaker_name, registry, identifier
            )
            if success:
                registered = True
                registration_turn = i + 1
                print(f"\n  *** Speaker registered on turn {registration_turn}! ***")
                print(f"  Known speakers: {registry.known_speakers}")

        if registered:
            # Try to identify the speaker in this response
            segments = extract_speech_segments(response, identifier)
            if segments:
                seg_audio = np.concatenate(segments)
                if len(seg_audio) >= int(1.0 * SR):  # At least 1 second
                    try:
                        emb = identifier.get_embedding(seg_audio)
                        best_id, best_sim = registry.identify_speaker(emb)
                        if best_id:
                            print(f"  Identification: {best_id} (sim={best_sim:.4f})")
                        else:
                            print(f"  Identification: no match (best sim={best_sim:.4f})")
                    except Exception:
                        pass

    # Summary
    print(f"\n{'=' * 60}")
    print("Registration Summary")
    print(f"{'=' * 60}")
    print(f"  Registered:     {registered}")
    if registered:
        print(f"  Registered on:  turn {registration_turn}")
        print(f"  Speaker ID:     human_{args.speaker_name.lower().replace(' ', '_')}")
    print(f"  Total turns:    {len(args.openers)}")
    print(f"  Known speakers: {registry.known_speakers}")

    # Save results
    results = {
        "registered": registered,
        "registration_turn": registration_turn,
        "speaker_name": args.speaker_name,
        "speaker_id": f"human_{args.speaker_name.lower().replace(' ', '_')}" if registered else None,
        "total_turns": len(args.openers),
        "known_speakers": registry.known_speakers,
        "accumulated_speech_s": sum(len(s) for s in accumulated_speech) / SR,
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()

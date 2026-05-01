#!/usr/bin/env python3
"""M4 — Self-Voice Detector Prototype.

Demonstrates the use of speaker embeddings to distinguish between
'Self' (Corollary) and 'Other' (Human).

Now supports room-colored self registration via a saved cerebellum model.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import numpy as np

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from broca.tts import synthesize
from cerebellum.room_coloring import predict_room_colored_self
from stg.speaker import SpeakerIdentifier, SelfVoiceDetector


async def main() -> None:
    parser = argparse.ArgumentParser(description="M4 self-voice prototype")
    parser.add_argument(
        "--cerebellum-model",
        default=None,
        help="Optional saved BabblingCerebellum JSON. If provided, register room-colored predicted self instead of dry TTS.",
    )
    parser.add_argument(
        "--reference-text",
        default="My name is Corollary. This is my reference voice.",
    )
    parser.add_argument(
        "--probe-text",
        default="I am speaking again. Do you recognize me?",
    )
    args = parser.parse_args()

    print("--- M4 Self-Voice Detector Prototype ---")

    identifier = SpeakerIdentifier()
    detector = SelfVoiceDetector(identifier)

    print("\n[M4] Synthesizing dry self reference...")
    dry_self_pcm = await synthesize(args.reference_text)

    if args.cerebellum_model:
        print(f"[M4] Applying room-colored forward model: {args.cerebellum_model}")
        registered_pcm = predict_room_colored_self(dry_self_pcm, args.cerebellum_model)
        registration_mode = "room_colored_prediction"
    else:
        registered_pcm = dry_self_pcm
        registration_mode = "dry_self"

    print(f"[M4] Registering self template ({registration_mode})...")
    detector.register_self(registered_pcm)
    print("Registration complete.")

    print("\n[M4] Testing self recognition...")
    probe_dry_pcm = await synthesize(args.probe_text)
    if args.cerebellum_model:
        probe_pcm = predict_room_colored_self(probe_dry_pcm, args.cerebellum_model)
        probe_mode = "room_colored_prediction"
    else:
        probe_pcm = probe_dry_pcm
        probe_mode = "dry_self"
    is_self, score = detector.is_self(probe_pcm)
    print(f"Probe mode: {probe_mode}")
    print(f"Match Score: {score:.4f} | Verdict: {'SELF' if is_self else 'OTHER'}")

    print("\n[M4] Testing 'Other' recognition (noise placeholder)...")
    human_pcm = np.random.randn(16000 * 2).astype(np.float32) * 0.1
    is_self, score = detector.is_self(human_pcm)
    print(f"Match Score: {score:.4f} | Verdict: {'SELF' if is_self else 'OTHER'}")


if __name__ == "__main__":
    asyncio.run(main())

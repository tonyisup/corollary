#!/usr/bin/env python3
"""Generate a longer, phonetically diverse world speech sample for M2 benchmarking.

The existing world_speech_sample.npy is only 1.1s — too short for meaningful WER
discrimination. This script generates a ~4s sample using edge-tts with a
phonetically rich text that gives Whisper more to work with.

Usage:
  ./.venv/bin/python scripts/gen_world_speech.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
import numpy as np

# Phonetically diverse text that covers a wide range of English phonemes
# and gives Whisper enough context for reliable transcription
WORLD_TEXT = (
    "the quick brown fox jumps over the lazy dog and then "
    "runs through the forest to find a quiet place near the river"
)

OUTPUT_PATH = "data/world_speech_benchmark.npy"


async def main():
    print(f"Generating world speech sample...")
    print(f"Text: \"{WORLD_TEXT}\"")
    print(f"Voice: en-US-GuyNeural (male, distinct from self AriaNeural)")

    pcm = await synthesize(WORLD_TEXT, voice="en-US-GuyNeural", rate="+0%")
    pcm = np.asarray(pcm, dtype=np.float32)

    # Normalize to a reasonable level
    peak = np.max(np.abs(pcm))
    if peak > 0:
        pcm = pcm / peak * 0.7

    np.save(OUTPUT_PATH, pcm)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Duration: {len(pcm)/16000:.2f}s")
    print(f"RMS: {np.sqrt(np.mean(pcm**2)):.4f}")
    print(f"Peak: {np.max(np.abs(pcm)):.4f}")


if __name__ == "__main__":
    asyncio.run(main())

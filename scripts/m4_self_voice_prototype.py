#!/usr/bin/env python3
"""M4 — Self-Voice Detector Prototype.

Demonstrates the use of speaker embeddings to distinguish between 
'Self' (Corollary) and 'Other' (Human).
"""

import asyncio
import os
import sys
import numpy as np

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from stg.speaker import SpeakerIdentifier, SelfVoiceDetector
from broca.tts import synthesize

async def main():
    print("--- M4 Self-Voice Detector Prototype ---")
    
    # 1. Initialize Detector
    identifier = SpeakerIdentifier()
    detector = SelfVoiceDetector(identifier)
    
    # 2. Register 'Self' Voice
    print("\n[M4] Registering 'Self' voice via TTS...")
    self_pcm = await synthesize("My name is Corollary. This is my reference voice.")
    detector.register_self(self_pcm)
    print("Registration complete.")
    
    # 3. Test 'Self' Recognition
    print("\n[M4] Testing 'Self' recognition...")
    another_self_pcm = await synthesize("I am speaking again. Do you recognize me?")
    is_self, score = detector.is_self(another_self_pcm)
    print(f"Match Score: {score:.4f} | Verdict: {'SELF' if is_self else 'OTHER'}")
    
    # 4. Test 'Other' Recognition
    print("\n[M4] Testing 'Other' recognition (Human placeholder)...")
    # Simulate a human voice with a different tone/text (different hash in our mock)
    # In real world, this would be a microphone capture of a human.
    human_pcm = np.random.randn(16000 * 2).astype(np.float32) * 0.1
    is_self, score = detector.is_self(human_pcm)
    print(f"Match Score: {score:.4f} | Verdict: {'SELF' if is_self else 'OTHER'}")

if __name__ == "__main__":
    asyncio.run(main())

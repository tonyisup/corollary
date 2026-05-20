#!/usr/bin/env python3
"""Quick negative test: verify fine-tuned scorer rejects other TTS voices."""

import asyncio
import json
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from stg.speaker import SpeakerIdentifier, FineTunedSelfVoiceDetector
from broca.tts import synthesize

async def main():
    scorer_path = "data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt"
    
    identifier = SpeakerIdentifier(use_real=True, backend="ecapa")
    # Hardened live-domain threshold: the same-family Jenny negative that still
    # slips at 0.5 is rejected here, while Aria self remains accepted.
    detector = FineTunedSelfVoiceDetector(identifier, threshold=0.99)
    detector.load_scorer(scorer_path)
    
    # Register self from AriaNeural
    tts_self = await synthesize("Hello, I am Corollary. This is my voice.", voice="en-US-AriaNeural")
    tts_self = tts_self.astype(np.float32)
    mid = len(tts_self) // 2
    detector.register_self(tts_self[mid:mid + 2 * 16000])
    
    print("=== Negative Test: Other TTS voices ===\n")
    
    test_voices = [
        ("en-US-GuyNeural", "Guy"),
        ("en-US-ChristopherNeural", "Christopher"),
        ("en-US-JennyNeural", "Jenny"),
        ("en-US-AriaNeural", "Aria (self)"),
    ]
    
    for voice, name in test_voices:
        tts = await synthesize(f"Hello, I am {name}. This is a test of the fine-tuned scorer.", voice=voice)
        tts = tts.astype(np.float32)
        
        # Test in 2s windows
        seg_samples = 2 * 16000
        hop = 1 * 16000
        results = []
        for start in range(0, len(tts) - seg_samples, hop):
            seg = tts[start:start + seg_samples]
            is_self, conf = detector.is_self(seg)
            results.append({"is_self": is_self, "confidence": round(float(conf), 4)})
        
        avg_conf = np.mean([r["confidence"] for r in results])
        pct_self = np.mean([r["is_self"] for r in results]) * 100
        label = "SELF" if name == "Aria (self)" else "WORLD"
        status = "✅" if (name == "Aria (self)" and pct_self > 80) or (name != "Aria (self)" and pct_self < 20) else "❌"
        print(f"  {status} {name:20s}: {len(results)} segs, avg_conf={avg_conf:.4f}, {pct_self:.0f}% SELF (expected {label})")

if __name__ == "__main__":
    asyncio.run(main())

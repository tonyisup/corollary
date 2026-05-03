#!/usr/bin/env python3
"""True false-positive test: M4 registered with self-stim, tested on WORLD speech."""
import sys, os, numpy as np
sys.path.insert(0, '.')

from packages.stg.speaker import SelfVoiceDetector, SpeakerIdentifier

# 1. Load self-stim from benchmark output (what M4 actually registered with)
print("Loading self-stim from benchmark output...")
self_pcm_path = "data/m4_controls/false_positive_world_only/benchmark_20260502_233144/seed_0/self_pcm.npy"
if not os.path.exists(self_pcm_path):
    # Try to find it
    for root, dirs, files in os.walk('data'):
        if 'self_pcm.npy' in files:
            self_pcm_path = os.path.join(root, 'self_pcm.npy')
            break

if not os.path.exists(self_pcm_path):
    print(f"ERROR: self_pcm.npy not found")
    sys.exit(1)

self_pcm = np.load(self_pcm_path)
print(f"Self-stim length: {len(self_pcm)} samples ({len(self_pcm)/16000:.1f}s)")

# 2. Create M4 detector with WINNER config (threshold=0.05)
print("\nCreating M4 detector (threshold=0.05)...")
identifier = SpeakerIdentifier(use_real=True)
detector = SelfVoiceDetector(identifier, threshold=0.05)
detector.register_self(self_pcm)
print("M4 registered with self-stim.")

# 3. Load WORLD speech (different voice - should NOT be identified as self)
world_paths = [
    "data/world_speech_gain_ladder/world_speech_gain_1p0.npy",
    "data/world_speech_gain_ladder/world_speech_same_text_guy.npy",
]
world_pcm = None
for wp in world_paths:
    if os.path.exists(wp):
        world_pcm = np.load(wp)
        print(f"\nLoaded world speech: {wp}")
        print(f"World speech length: {len(world_pcm)} samples ({len(world_pcm)/16000:.1f}s)")
        break

if world_pcm is None:
    print("ERROR: No world speech file found")
    sys.exit(1)

# 4. Test M4 on world speech (700ms chunks like benchmark)
print("\nTesting M4 on WORLD speech (should be NEAR-ZERO false positives)...")
chunk_size = int(0.7 * 16000)  # 700ms
accepted = 0
total = 0
confidences = []

for i in range(0, len(world_pcm) - chunk_size, chunk_size):
    chunk = world_pcm[i:i+chunk_size]
    is_self, conf = detector.is_self(chunk)
    total += 1
    confidences.append(conf)
    if is_self:
        accepted += 1
        if accepted <= 3:  # Print first 3 false positives
            print(f"  FALSE POSITIVE at chunk {i//chunk_size}: conf={conf:.4f}")

fp_rate = accepted / total * 100 if total > 0 else 0
print(f"\n=== False-Positive Results ===")
print(f"Accepted as SELF: {accepted}/{total} chunks")
print(f"False-positive rate: {fp_rate:.1f}%")
print(f"Mean confidence: {np.mean(confidences):.4f}")
print(f"Confidence range: {min(confidences):.4f} to {max(confidences):.4f}")

if fp_rate > 5.0:
    print(f"\n*** WARNING: HIGH FALSE-POSITIVE RATE ({fp_rate:.1f}%) ***")
    print("*** M4 is NOT ready for production. ***\n")
    sys.exit(1)
elif fp_rate > 0:
    print(f"\n*** CAUTION: Some false positives ({fp_rate:.1f}%) ***\n")
    sys.exit(0)
else:
    print(f"\n*** SUCCESS: Zero false positives! M4 correctly rejects world speech. ***\n")
    sys.exit(0)

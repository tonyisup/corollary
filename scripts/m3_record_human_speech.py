#!/usr/bin/env python3
"""Record clean human speech samples for dual-speaker testing.

Records your voice through the mic with no TTS playback.
These samples are then played through the external speaker during
dual-speaker live E2E testing.

Usage:
    python scripts/m3_record_human_speech.py
"""

import argparse
import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice not installed")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))
from cochlea.capture import SR
from cochlea.vad import VAD_FRAME_SAMPLES

# Phrases to record (matching the v3 scenario speak windows)
PHRASES = [
    "Excuse me, I have a question.",
    "Sorry, could you repeat that?",
    "Hello, can I ask something?",
    "Okay, thanks for explaining.",
    "Hold on, I need to say something.",
]


def record_phrase(phrase: str, input_device: int, output_device: int, 
                  output_dir: Path, phrase_idx: int) -> bool:
    """Record a single phrase with visual guidance."""
    
    print(f"\n{'='*60}")
    print(f"Phrase {phrase_idx + 1}/{len(PHRASES)}:")
    print(f"  \"{phrase}\"")
    print(f"{'='*60}")
    
    # Lead-in and tail
    lead_in_s = 0.5
    max_duration_s = 4.0
    total_duration_s = lead_in_s + max_duration_s
    
    input_sr = int(sd.query_devices(input_device)["default_samplerate"])
    total_samples = int(total_duration_s * input_sr)
    capture = np.zeros(total_samples, dtype=np.float32)
    
    samples_captured = [0]
    recording = [False]
    
    def callback(indata, outdata, frames, time_info, status):
        # Output silence
        outdata[:] = 0
        
        # Capture
        start = samples_captured[0]
        end = min(start + frames, total_samples)
        if start < total_samples:
            n = end - start
            capture[start:end] = indata[:n, 0]
            samples_captured[0] = end
    
    print("\n  Get ready...")
    time.sleep(1)
    
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    print(f"\n  >>> RECORD! <<<")
    print(f"  Speak now: \"{phrase}\"")
    
    t0 = time.time()
    with sd.Stream(
        samplerate=input_sr,
        blocksize=int(VAD_FRAME_SAMPLES * input_sr / SR),
        dtype="float32",
        channels=1,
        callback=callback,
        device=(input_device, output_device),
    ):
        # Wait for speech to start, then stop after silence
        time.sleep(total_duration_s)
    
    elapsed = time.time() - t0
    
    # Trim
    capture = capture[:samples_captured[0]]
    
    # Check quality
    from cochlea.vad import SmoothedVAD
    vad = SmoothedVAD()
    speech_frames = 0
    for i in range(0, len(capture) - VAD_FRAME_SAMPLES, VAD_FRAME_SAMPLES):
        frame = capture[i:i + VAD_FRAME_SAMPLES]
        if vad.process_frame(frame):
            speech_frames += 1
    
    # Resample to SR
    if input_sr != SR:
        from scipy.signal import resample as scipy_resample
        capture_16k = scipy_resample(capture, int(len(capture) * SR / input_sr)).astype(np.float32)
    else:
        capture_16k = capture
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"human_speech_{phrase_idx:02d}.npy"
    np.save(output_dir / filename, capture_16k)
    
    metadata = {
        "phrase": phrase,
        "phrase_idx": phrase_idx,
        "duration_s": len(capture_16k) / SR,
        "speech_frames": speech_frames,
        "input_sr": input_sr,
        "recorded_at": datetime.now().isoformat(),
        "quality_ok": speech_frames > 5,
    }
    
    with open(output_dir / f"human_speech_{phrase_idx:02d}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Captured: {len(capture_16k)} samples ({len(capture_16k)/SR:.2f}s)")
    print(f"  Speech frames: {speech_frames}")
    print(f"  Quality: {'OK' if speech_frames > 5 else 'LOW'}")
    print(f"  Saved: {output_dir / filename}")
    
    return speech_frames > 5


def main():
    parser = argparse.ArgumentParser(description="Record clean human speech samples")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--output-dir", default="data/m3_human_speech")
    parser.add_argument("--redo", action="store_true")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    print(f"\nRecording {len(PHRASES)} human speech phrases")
    print(f"  Mic: device {args.input_device}")
    print(f"  Output: device {args.output_device} (for monitoring)")
    
    for i, phrase in enumerate(PHRASES):
        # Check if already recorded
        if not args.redo and (output_dir / f"human_speech_{i:02d}.npy").exists():
            print(f"\nSkipping phrase {i+1} (already exists, use --redo)")
            continue
        
        try:
            success = record_phrase(phrase, args.input_device, args.output_device, output_dir, i)
            if not success:
                print(f"  WARNING: Low quality recording for phrase {i+1}")
        except KeyboardInterrupt:
            print(f"\n  Cancelled")
            break
    
    print(f"\n\nDone! Recorded phrases saved to: {output_dir}")


if __name__ == "__main__":
    main()

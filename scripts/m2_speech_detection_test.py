#!/usr/bin/env python3
"""M2 — Speech vs Non-Speech Verification.

Captures audio and runs VAD to detect speech frames.
Tests whether Corollary can distinguish between environmental noise and 
human speech (M2 milestone).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    sys.exit(1)

# Add corollary packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cochlea.capture import SR
from cochlea.vad import SmoothedVAD, VAD_FRAME_SAMPLES

def main():
    parser = argparse.ArgumentParser(description="M2 Speech vs Non-Speech Hardware Test")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--duration", type=float, default=5.0, help="Capture duration in seconds")
    parser.add_argument("--mode", type=int, default=3, help="VAD aggressiveness (0-3)")
    args = parser.parse_args()

    input_sr = SR
    if args.device is not None:
        input_sr = int(sd.query_devices(args.device)["default_samplerate"] or SR)

    print(f"\n--- M2 Speech Detection Test ---")
    print(f"Device: {args.device if args.device is not None else 'default'}")
    print(f"Duration: {args.duration}s")
    print(f"VAD Mode: {args.mode}")
    print(f"Please speak or make noise during the capture...")
    
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    vad = SmoothedVAD(mode=args.mode)
    num_samples = int(args.duration * input_sr)
    captured_audio = np.zeros(num_samples, dtype=np.float32)

    stream = sd.InputStream(device=args.device, channels=1, samplerate=input_sr, dtype="float32", blocksize=1024)
    stream.start()

    pos = 0
    speech_frames = 0
    total_frames = 0
    
    # We'll process in real-time if possible, or just capture and process.
    # To keep it simple, capture first then process.
    while pos < num_samples:
        data, _ = stream.read(min(1024, num_samples - pos))
        n = min(len(data), num_samples - pos)
        captured_audio[pos:pos+n] = data[:n, 0]
        pos += n

    stream.stop()
    stream.close()

    print("\nProcessing VAD...")
    
    # Resample to 16kHz if needed for VAD
    if input_sr != SR:
        import librosa
        audio_16k = librosa.resample(captured_audio, orig_sr=input_sr, target_sr=SR)
    else:
        audio_16k = captured_audio

    num_vad_frames = len(audio_16k) // VAD_FRAME_SAMPLES
    speech_mask = np.zeros(num_vad_frames, dtype=bool)
    
    for i in range(num_vad_frames):
        start = i * VAD_FRAME_SAMPLES
        frame = audio_16k[start : start + VAD_FRAME_SAMPLES]
        is_speech = vad.process_frame(frame)
        speech_mask[i] = is_speech
        if is_speech:
            speech_frames += 1
        total_frames += 1

    speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
    
    print(f"\nResults:")
    print(f"  Total Frames:  {total_frames}")
    print(f"  Speech Frames: {speech_frames}")
    print(f"  Speech Ratio:  {speech_ratio:.1%}")
    
    if speech_ratio > 0.1:
        print("\n✅ SPEECH DETECTED")
    else:
        print("\n❌ NO SIGNIFICANT SPEECH DETECTED")

    # Visual representation
    print("\nTimeline (S=Speech, .=Silence):")
    viz = "".join("S" if m else "." for m in speech_mask)
    # Chunk viz to fit terminal
    width = 60
    for i in range(0, len(viz), width):
        print(viz[i:i+width])

if __name__ == "__main__":
    main()

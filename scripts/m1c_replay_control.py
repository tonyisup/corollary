#!/usr/bin/env python3
"""Hardware-facing M1C Replayed-Self Negative Control.

Tests whether the system correctly identifies self-like audio as "WORLD" 
when it was replayed externally (without an active motor command).
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cochlea.capture import SR
from thalamus.attributor import ThalamicAttributor, SourceVerdict

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        factor = max(int(orig_sr / target_sr), 1)
        return audio[::factor].astype(np.float32)

def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-10))

def normalized_xcorr(observed: np.ndarray, template: np.ndarray) -> tuple[float, int]:
    if len(observed) < len(template):
        return 0.0, 0
    template_norm = float(np.linalg.norm(template))
    if template_norm < 1e-9:
        return 0.0, 0
    best_score = -1.0
    best_shift = 0
    step = max(len(template) // 80, 1)
    for shift in range(0, len(observed) - len(template) + 1, step):
        window = observed[shift:shift + len(template)]
        denom = float(np.linalg.norm(window) * template_norm)
        score = 0.0 if denom < 1e-9 else float(np.dot(window, template) / denom)
        if score > best_score:
            best_score = score
            best_shift = shift
    return max(0.0, best_score), best_shift

def main():
    parser = argparse.ArgumentParser(description="M1C Replayed-Self Control")
    parser.add_argument("npz_file", help="Path to a recorded probe .npz file")
    parser.add_argument("--input-device", type=int, default=2)
    parser.add_argument("--noise-device", type=int, default=3)
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.0)
    args = parser.parse_args()

    data = np.load(args.npz_file, allow_pickle=True)
    known_pcm = data["known_pcm"]
    
    # Use synthetic known_pcm for loud playback
    playback_audio = known_pcm
    playback_duration = len(playback_audio) / SR
    
    input_sr = int(sd.query_devices(args.input_device)["default_samplerate"] or SR)
    noise_sr = int(sd.query_devices(args.noise_device)["default_samplerate"] or SR)
    
    capture_duration = args.lead_in_s + playback_duration + args.tail_s
    total_samples = int(capture_duration * input_sr)
    captured_audio = np.zeros(total_samples, dtype=np.float32)
    
    print(f"Replaying {args.npz_file} on device {args.noise_device}")
    
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
        
    stream = sd.InputStream(device=args.input_device, channels=1, samplerate=input_sr, dtype="float32", blocksize=1024)
    stream.start()
    
    lead_in_samples = int(args.lead_in_s * input_sr)
    pos = 0
    playback_started = False
    while pos < total_samples:
        if not playback_started and pos >= lead_in_samples:
            sd.play(playback_audio, samplerate=SR, device=args.noise_device, blocking=False)
            playback_started = True
        data_in, _ = stream.read(min(1024, total_samples - pos))
        n = min(len(data_in), total_samples - pos)
        captured_audio[pos:pos+n] = data_in[:n, 0]
        pos += n
        
    stream.stop()
    stream.close()
    sd.wait()
    
    if input_sr != SR:
        observed = resample_audio(captured_audio, input_sr, SR)
    else:
        observed = captured_audio
        
    # Analysis
    attributor = ThalamicAttributor(sr=SR)
    score, shift = normalized_xcorr(observed, known_pcm)
    rms = float(np.sqrt(np.mean(observed**2)))
    rms_db = db(rms)

    # is_playing=False because we ARE NOT using our "self" output device (device 4)
    res = attributor.attribute(
        match_score=score,
        observed_onset_samples=shift,
        expected_onset_samples=None,
        rms_db=rms_db,
        is_playing=False
    )

    print("\nResults:")
    print(f"  Match Score: {score:.6f}")
    print(f"  RMS DB:      {rms_db:.2f}")
    print(f"  Verdict:     {res.verdict.name}")
    print(f"  Confidence:  {res.confidence:.4f}")
    
    if res.verdict == SourceVerdict.WORLD:
        print("  ✅ SUCCESS: Replayed self correctly identified as WORLD (absence of motor command).")
    else:
        print("  ❌ FAILURE: Replayed self was incorrectly attributed.")

if __name__ == "__main__":
    main()

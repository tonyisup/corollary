#!/usr/bin/env python3
"""M2 — Speech-Aware Self/World Separation Harness.

Goal: Suppress "Self" audio while preserving "World" speech.

Execution:
1. Play Self-Babble (Device 4).
2. Play World-Speech (Device 3).
3. Capture and use ThalamicAttributor + VAD to label segments.
4. Run Whisper ASR on both raw and gated audio to measure preservation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from typing import Any

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
from thalamus.attributor import ThalamicAttributor, SourceVerdict
from stg.asr import transcribe

sys.path.insert(0, os.path.dirname(__file__))
from m1_tone_attribution import generate_stimulus_with_metadata

def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-10))

def play_array_blocking(data: np.ndarray, device: int, samplerate: float) -> None:
    if data.ndim == 1:
        data = data[:, np.newaxis]
    with sd.OutputStream(device=device, channels=data.shape[1], samplerate=samplerate, dtype="float32") as stream:
        stream.write(np.asarray(data, dtype=np.float32))

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

def run_m2_trial(
    self_stim: str,
    world_speech_file: str,
    input_device: int,
    output_device: int,
    noise_device: int,
    lead_in_s: float = 0.5,
    tail_s: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    # 1. Generate/Load Stimuli
    self_meta = generate_stimulus_with_metadata(self_stim, 350, 0.7, seed=seed)
    self_pcm = self_meta["pcm"]
    
    if world_speech_file.endswith(".npy"):
        world_pcm = np.load(world_speech_file)
    else:
        # Fallback to a simple tone if no file
        world_pcm = np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR).astype(np.float32) * 0.5

    self_duration = len(self_pcm) / SR
    world_duration = len(world_pcm) / SR
    
    # Schedule: World speech starts, Self babble happens in the middle
    world_start_s = lead_in_s
    self_start_s = world_start_s + (world_duration / 2.0) - (self_duration / 2.0)
    total_duration = max(world_start_s + world_duration, self_start_s + self_duration) + tail_s

    input_sr = int(sd.query_devices(input_device)["default_samplerate"] or SR)
    output_sr = int(sd.query_devices(output_device)["default_samplerate"] or SR)
    noise_sr = int(sd.query_devices(noise_device)["default_samplerate"] or SR)

    def get_padded_pcm(pcm: np.ndarray, start_s: float, total_s: float, target_sr: float) -> np.ndarray:
        padded = np.zeros(int(total_s * target_sr), dtype=np.float32)
        start_idx = int(start_s * target_sr)
        # Simple resample if needed
        if abs(target_sr - SR) > 1:
            import librosa
            pcm = librosa.resample(pcm, orig_sr=SR, target_sr=int(target_sr))
        end_idx = min(len(padded), start_idx + len(pcm))
        padded[start_idx:end_idx] = pcm[:end_idx-start_idx]
        return padded

    self_padded = get_padded_pcm(self_pcm, self_start_s, total_duration, output_sr)
    world_padded = get_padded_pcm(world_pcm, world_start_s, total_duration, noise_sr)

    print(f"\n--- M2 Speech Preservation Trial ---")
    print(f"Self: {self_stim} at {self_start_s:.2f}s")
    print(f"World Speech: {world_duration:.2f}s at {world_start_s:.2f}s")
    
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    # 2. Hardware Capture
    stream = sd.InputStream(device=input_device, channels=1, samplerate=input_sr, dtype="float32", blocksize=1024)
    stream.start()
    
    self_thread = threading.Thread(target=play_array_blocking, args=(self_padded, output_device, output_sr))
    world_thread = threading.Thread(target=play_array_blocking, args=(world_padded, noise_device, noise_sr))
    
    self_thread.start()
    world_thread.start()
    
    total_samples = int(total_duration * input_sr)
    captured_audio = np.zeros(total_samples, dtype=np.float32)
    pos = 0
    while pos < total_samples:
        data, _ = stream.read(min(1024, total_samples - pos))
        n = min(len(data), total_samples - pos)
        captured_audio[pos:pos+n] = data[:n, 0]
        pos += n
        
    self_thread.join()
    world_thread.join()
    stream.stop()
    stream.close()

    if input_sr != SR:
        import librosa
        observed = librosa.resample(captured_audio, orig_sr=input_sr, target_sr=SR)
    else:
        observed = captured_audio

    # 3. M2 Analysis: Attribution + VAD + Gating
    attributor = ThalamicAttributor(sr=SR)
    vad = SmoothedVAD()
    
    # Match score for self
    self_score, self_shift = normalized_xcorr(observed, self_pcm)
    self_expected_shift = int(self_start_s * SR)
    
    # Frame-level processing for gating
    num_frames = len(observed) // VAD_FRAME_SAMPLES
    mask = np.ones(len(observed), dtype=bool)
    
    verdicts = []
    speech_frames = 0
    
    for i in range(num_frames):
        start = i * VAD_FRAME_SAMPLES
        end = start + VAD_FRAME_SAMPLES
        frame = observed[start:end]
        
        is_speech = vad.process_frame(frame)
        if is_speech: speech_frames += 1
        
        rms = float(np.sqrt(np.mean(frame**2)))
        rms_db = db(rms)
        
        # Is this frame within our motor window?
        is_playing = (start >= self_expected_shift) and (end <= self_expected_shift + len(self_pcm))
        
        # Frame-level match score is expensive, so we use a simplified check or the global one
        # For M2 prototype, we use the global match_score if is_playing is True.
        res = attributor.attribute(
            match_score=self_score if is_playing else 0.0,
            observed_onset_samples=self_shift,
            expected_onset_samples=self_expected_shift if is_playing else None,
            rms_db=rms_db,
            is_playing=is_playing,
            is_speech=is_speech
        )
        verdicts.append(res.verdict)
        
        # GATE LOGIC: If verdict is SELF, zero it out.
        # If verdict is BOTH, we might want to keep it if is_speech is True (M2 preservation).
        if res.verdict == SourceVerdict.SELF:
            mask[start:end] = False
        elif res.verdict == SourceVerdict.BOTH:
            # M2: If it's contaminated, but there's speech, DON'T zero it out completely.
            # Maybe attenuate instead? For now, let's keep it to allow ASR to try.
            mask[start:end] = True 

    # 4. ASR Comparison
    print("\nRunning ASR on Raw Audio...")
    raw_res = transcribe(observed, model_size="base")
    print(f"Raw ASR: \"{raw_res.text}\"")
    
    print("\nRunning ASR on Gated Audio...")
    gated_res = transcribe(observed, mask=mask, model_size="base")
    print(f"Gated ASR: \"{gated_res.text}\"")

    return {
        "raw_text": raw_res.text,
        "gated_text": gated_res.text,
        "self_score": self_score,
        "speech_ratio": speech_frames / num_frames,
        "self_frames": sum(1 for v in verdicts if v == SourceVerdict.SELF),
        "both_frames": sum(1 for v in verdicts if v == SourceVerdict.BOTH),
        "world_frames": sum(1 for v in verdicts if v == SourceVerdict.WORLD),
    }

def main():
    parser = argparse.ArgumentParser(description="M2 Speech Preservation Harness")
    parser.add_argument("--self-stim", default="babble_ba_repeat_short_tight_var")
    parser.add_argument("--world-speech", default="data/world_speech_sample.npy")
    parser.add_argument("--input-device", type=int, default=2)
    parser.add_argument("--output-device", type=int, default=4)
    parser.add_argument("--noise-device", type=int, default=3)
    args = parser.parse_args()

    res = run_m2_trial(
        self_stim=args.self_stim,
        world_speech_file=args.world_speech,
        input_device=args.input_device,
        output_device=args.output_device,
        noise_device=args.noise_device
    )
    
    print("\n--- M2 Final Verdict ---")
    print(f"Self Match Score: {res['self_score']:.4f}")
    print(f"Speech Ratio: {res['speech_ratio']:.1%}")
    print(f"Self/Both/World Frames: {res['self_frames']}/{res['both_frames']}/{res['world_frames']}")
    
    if len(res['gated_text']) > 5:
        print("\n✅ SUCCESS: World speech preserved after self-voice gating!")
    else:
        print("\n❌ FAILURE: World speech was lost or masked.")

if __name__ == "__main__":
    main()

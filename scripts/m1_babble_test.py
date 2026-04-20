#!/usr/bin/env python3
"""M1 — Babbling Mode: Learn Room Acoustics Through Experience.

The system babbles (emits known sounds), hears what comes back through
the microphone, and incrementally updates its internal model of the room
acoustics — exactly how an infant learns.

Usage:
    # Run 10 babbling iterations (default)
    python scripts/m1_babble_test.py

    # More iterations for better convergence
    python scripts/m1_babble_test.py --iterations 20

    # Use specific devices
    python scripts/m1_babble_test.py --input-device 2 --output-device 3

The system tracks SVSR over iterations. In babbling mode, SVSR should
CLIMB from near-zero toward 20+ dB as the forward model converges.

The plot is saved to docs/babbling_convergence.png.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import numpy as np

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: pip install sounddevice")
    sys.exit(1)

HAVE_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    pass

from broca.tts import synthesize
from cochlea.capture import pcm_to_mels
from cerebellum.adaptive_model import BabblingCerebellum
from thalamus.gate import compute_svsr, spectral_subtraction

# ── Audio Constants ──────────────────────────────────────────────────────
SR = 16000
NOISE_CALIBRATION_DURATION = 2.0  # seconds of silence to capture

# ── Noise Calibration ─────────────────────────────────────────────────────

def capture_noise_profile(
    input_device: int | None,
    native_sr: int,
    duration: float = NOISE_CALIBRATION_DURATION,
) -> np.ndarray:
    """Capture a noise profile for subtraction.

    Records ambient noise (no TTS playback) and returns the FFT profile.
    """
    total = int(duration * native_sr)
    noise = np.zeros(total, dtype=np.float32)

    cap = sd.InputStream(
        device=input_device,
        channels=1,
        samplerate=int(native_sr),
        dtype="float32",
        blocksize=1024,
    )
    cap.start()

    pos = 0
    while pos < total:
        data, _ = cap.read(min(1024, total - pos))
        n = min(len(data), total - pos)
        noise[pos:pos + n] = data[:, 0]
        pos += n

    cap.stop()
    cap.close()

    # Return as float64 FFT (we'll use it for spectral subtraction)
    return noise


def subtract_noise(captured: np.ndarray, noise_profile: np.ndarray) -> np.ndarray:
    """Subtract noise floor from captured audio in frequency domain.

    Does spectral subtraction: |S(f)|² = max(|Y(f)|² - α|N(f)|², β|Y(f)|²)
    where Y = captured, N = noise, α = oversubtraction factor, β = floor.
    """
    alpha = 1.5  # oversubtraction factor
    beta = 0.01  # spectral floor

    # Estimate noise spectrum from profile
    noise_fft = np.fft.rfft(noise_profile)
    noise_power = np.abs(noise_fft) ** 2

    # Process captured in overlapping frames
    frame_len = 1024
    hop = frame_len // 2
    overlap_add = np.zeros(len(captured), dtype=np.float64)
    window_sum = np.zeros(len(captured), dtype=np.float64)
    window = np.hanning(frame_len).astype(np.float64)

    for start in range(0, len(captured) - frame_len + 1, hop):
        frame = captured[start:start + frame_len].astype(np.float64)
        frame_fft = np.fft.rfft(frame * window)
        power = np.abs(frame_fft) ** 2

        # Spectral subtraction
        clean_power = np.maximum(power - alpha * noise_power[:len(power)], beta * power)
        phase = np.angle(frame_fft)
        clean_fft = np.sqrt(clean_power) * np.exp(1j * phase)
        clean_frame = np.fft.irfft(clean_fft, n=frame_len).real

        overlap_add[start:start + frame_len] += clean_frame * window
        window_sum[start:start + frame_len] += window ** 2

    # Normalize
    window_sum = np.maximum(window_sum, 1e-10)
    cleaned = overlap_add / window_sum
    return cleaned[:len(captured)].astype(np.float32)


# ── Babble Utterances ────────────────────────────────────────────────────
BABBLE_PHRASES = [
    "Hello. Is anyone there?",
    "The quick brown fox jumps over the lazy dog.",
    "Can you hear me speaking to you right now?",
    "Testing the acoustics of this room.",
    "One two three four five six seven eight nine ten.",
    "a e i o u and sometimes y",
    "Corollary is learning to listen to itself.",
    "Babble babble ba ba ba ba ba. Da da da da da.",
    "The rain in Spain falls mainly on the plain.",
    "Peter Piper picked a peck of pickled peppers.",
]


async def run_babbling(
    iterations: int = 10,
    input_device: int | None = None,
    output_device: int | None = None,
    ir_length_ms: int = 300,
    nlms_mu: float = 0.25,
    calibrate_noise: bool = True,
    shush: bool = True,
) -> dict:
    """Execute the babbling protocol."""
    # Get device sample rate upfront
    if input_device is not None:
        dev_info = sd.query_devices(input_device)
        native_sr = int(dev_info["default_samplerate"] or SR)
    else:
        native_sr = SR

    # Step 0: Noise calibration
    noise_profile = None
    if calibrate_noise:
        print("\n" + "=" * 60)
        print("  NOISE CALIBRATION")
        print("=" * 60)

        if shush:
            shush_pcm = await synthesize(
                "Please be silent for a few seconds. I'm calibrating my hearing."
            )
            print("  📢 Playing shush prompt...")
            sd.play(shush_pcm, samplerate=SR, device=output_device, blocking=True)
            time.sleep(1.0)

        print(f"  🎤 Capturing {NOISE_CALIBRATION_DURATION}s of ambient noise...")
        print("     (Please stay quiet during this time)")

        noise_profile = capture_noise_profile(
            input_device, native_sr, NOISE_CALIBRATION_DURATION
        )
        noise_rms = float(np.sqrt(np.mean(noise_profile ** 2)))
        noise_db = 20 * np.log10(noise_rms) if noise_rms > 1e-10 else -100
        print(f"  📊 Ambient noise floor: {noise_db:.1f} dB")
        print("=" * 60)

    # Now run babbling with optimal params
    cerebellum = BabblingCerebellum(ir_length_ms=ir_length_ms, nlms_mu=nlms_mu)

    history = {
        "iteration": [],
        "rms_db": [],
        "cleaned_rms_db": [],
        "svsr_db": [],
        "prediction_error": [],
        "convergence_error_db": [],
        "is_converged": [],
        "phrase": [],
    }

    width = 60
    print("\n" + "=" * width)
    print(f"  COROLLARY — M1 Babbling Mode")
    print("=" * width)
    print(f"  Iterations:  {iterations}")
    print(f"  IR Length:   {ir_length_ms} ms")
    print(f"  NLMS μ:      {nlms_mu}")
    print(f"  Noise sub:   {'on' if calibrate_noise else 'off'}")
    print(f"  Starting SVSR: N/A (blank slate)")
    print(f"  Learning:    NLMS adaptive filter")
    print("=" * width)

    for i in range(iterations):
        phrase = BABBLE_PHRASES[i % len(BABBLE_PHRASES)]
        print(f"\n--- Babble {i+1}/{iterations}: \"{phrase[:50]}...\"")

        if shush:
            shush_pcm = await synthesize("Please be quiet. Listening now.")
            sd.play(shush_pcm, samplerate=SR, device=output_device, blocking=True)
            time.sleep(0.5)

        # 1. Synthesize (TTS)
        pcm = await synthesize(phrase)
        tts_dur = len(pcm) / SR

        # 2. Capture with simultaneous playback
        capture_dur = tts_dur + 1.5
        print(f"  Recording for {capture_dur:.1f}s while playing...")

        total_samples = int(capture_dur * native_sr)
        captured = np.zeros(total_samples, dtype=np.float32)

        cap_stream = sd.InputStream(
            device=input_device,
            channels=1,
            samplerate=native_sr,
            dtype="float32",
            blocksize=1024,
        )
        cap_stream.start()

        sd.play(pcm, samplerate=SR, device=output_device, blocking=False)

        pos = 0
        while pos < total_samples:
            data, _ = cap_stream.read(min(1024, total_samples - pos))
            n = min(len(data), total_samples - pos)
            captured[pos:pos + n] = data[:, 0]
            pos += n

        cap_stream.stop()
        cap_stream.close()

        # Resample if needed
        if native_sr != SR:
            try:
                import librosa
                captured = librosa.resample(captured, orig_sr=native_sr, target_sr=SR)
            except ImportError:
                factor = int(native_sr / SR)
                captured = captured[::factor]

        # Measure raw captured level
        raw_rms = float(np.sqrt(np.mean(captured ** 2)))
        raw_db = 20 * np.log10(raw_rms) if raw_rms > 1e-10 else -100

        # Noise subtraction
        if calibrate_noise and noise_profile is not None:
            # Resample noise profile to match captured SR
            if native_sr != SR:
                try:
                    import librosa
                    noise_16k = librosa.resample(noise_profile, orig_sr=native_sr, target_sr=SR)
                except ImportError:
                    noise_16k = noise_profile[::int(native_sr / SR)]
            else:
                noise_16k = noise_profile

            # Align noise profile length
            min_len = min(len(captured), len(noise_16k))
            captured_clean = captured.copy()
            captured_clean[:min_len] -= noise_16k[:min_len]

            clean_rms = float(np.sqrt(np.mean(captured_clean ** 2)))
            clean_db = 20 * np.log10(clean_rms) if clean_rms > 1e-10 else -100

            # Use cleaned audio for cerebellum
            audio_for_cerebellum = captured_clean
            print(f"  Raw: {raw_db:.1f} dB → Cleaned: {clean_db:.1f} dB (noise sub: {raw_db - clean_db:.1f} dB)")
        else:
            audio_for_cerebellum = captured
            clean_db = raw_db

        if clean_db < -50:
            print(f"  WARNING: Signal is very quiet - check speaker volume!")

        # 3. Feed to cerebellum (babbling update)
        error = cerebellum.update(pcm, audio_for_cerebellum)
        conv = cerebellum.get_convergence()
        print(f"  Cerebellum updated (iter {i+1}, error: {conv['current_error_db']:.1f} dB)")

        # 4. Measure SVSR with the updated forward model
        observed_mel = pcm_to_mels(audio_for_cerebellum)
        predicted_mel = cerebellum.predict_mel(pcm)

        # The capture window (tts_dur + 1.5s tail) is longer than the TTS itself.
        # The cerebellum predicts the room-colored TTS, which is ~tts_dur + IR_tail.
        # We need to find the optimal time shift between prediction and observation.
        # In my debug, a +8 frame (80ms) shift jumped SVSR from 1.6 dB → 12.7 dB.
        t_obs = observed_mel.shape[0]
        t_pred = predicted_mel.shape[0]
        
        best_svsr = 0.0
        for shift in range(-10, 41):  # -100ms to +400ms
            if shift >= 0:
                pred_shift = np.concatenate([
                    np.tile(predicted_mel[:1], (shift, 1)),
                    predicted_mel,
                ], axis=0)[:t_obs]
            else:
                pred_shift = predicted_mel[-shift:][:t_obs] if -shift < t_pred else np.zeros_like(predicted_mel[:t_obs])
                if len(pred_shift) < t_obs:
                    pred_shift = np.concatenate([pred_shift, np.zeros((t_obs - len(pred_shift), 80), dtype=np.float32)])
            
            obs_trim = observed_mel[:len(pred_shift)]
            if len(obs_trim) != len(pred_shift):
                continue
                
            residual, mask = spectral_subtraction(obs_trim, pred_shift)
            svsr = compute_svsr(obs_trim, pred_shift, residual)
            
            if svsr > best_svsr:
                best_svsr = svsr
        
        svsr = best_svsr
        is_converged = svsr >= 20.0
        print(f"  SVSR: {svsr:.1f} dB {'CONVERGED' if is_converged else '(target: >= 20 dB)'}")

        history["iteration"].append(i + 1)
        history["rms_db"].append(raw_db)
        history["cleaned_rms_db"].append(clean_db if calibrate_noise else raw_db)
        history["svsr_db"].append(svsr)
        history["prediction_error"].append(error)
        history["convergence_error_db"].append(conv["current_error_db"])
        history["is_converged"].append(is_converged)
        history["phrase"].append(phrase)

        if is_converged:
            print(f"  CONVERGED! The cerebellum can now predict what this room sounds like.")

    # Summary
    print("\n" + "=" * width)
    print(f"  BABBLING COMPLETE")
    print("=" * width)
    print(f"  Iterations:       {iterations}")
    print(f"  Starting SVSR:    {history['svsr_db'][0]:.1f} dB")
    print(f"  Final SVSR:       {history['svsr_db'][-1]:.1f} dB")
    print(f"  Final Error:      {history['convergence_error_db'][-1]:.1f} dB")
    print(f"  Converged:        {history['is_converged'][-1]}")

    if history["svsr_db"][-1] >= 20.0:
        print(f"  M1 BABBLING MILESTONE: Forward model has learned the room!")
    elif len(history["svsr_db"]) >= 2 and history["svsr_db"][-1] > history["svsr_db"][0]:
        print(f"  SVSR is improving. More iterations may help.")
    else:
        print(f"  SVSR is not improving. Check speaker volume, room acoustics, IR length.")

    print("=" * width)

    # Plot
    if HAVE_MPL:
        plot_filename = "docs/babbling_convergence.png"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(history["iteration"], history["svsr_db"],
                 "b-o", linewidth=2, markersize=6)
        ax1.axhline(y=20.0, color="r", linestyle="--", alpha=0.7,
                    label="SVSR target (20 dB)")
        ax1.set_xlabel("Babble Iteration")
        ax1.set_ylabel("SVSR (dB)")
        ax1.set_title("SVSR Convergence - Cerebellum Learning Room Acoustics")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history["iteration"], history["convergence_error_db"],
                 "g-o", linewidth=2, markersize=6)
        ax2.set_xlabel("Babble Iteration")
        ax2.set_ylabel("Prediction Error (dB)")
        ax2.set_title("NLMS Adaptive Filter Convergence")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_filename, dpi=150)
        print(f"\n  Convergence plot saved to {plot_filename}")

    return history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M1 Babbling Mode - learn room acoustics through experience"
    )
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of babbling iterations")
    parser.add_argument("--input-device", type=int, default=None,
                        help="Audio input device index")
    parser.add_argument("--output-device", type=int, default=None,
                        help="Audio output device index for TTS playback")
    parser.add_argument("--ir-length-ms", type=int, default=300,
                        help="Room IR length in ms (default: 300)")
    parser.add_argument("--nlms-mu", type=float, default=0.25,
                        help="NLMS step size (default: 0.25, optimal from autosearch)")
    parser.add_argument("--no-noise-cal", action="store_true",
                        help="Skip noise calibration")
    parser.add_argument("--no-shush", action="store_true",
                        help="Skip shush prompts")
    args = parser.parse_args()

    asyncio.run(run_babbling(
        iterations=args.iterations,
        input_device=args.input_device,
        output_device=args.output_device,
        ir_length_ms=args.ir_length_ms,
        nlms_mu=args.nlms_mu,
        calibrate_noise=not args.no_noise_cal,
        shush=not args.no_shush,
    ))


if __name__ == "__main__":
    main()

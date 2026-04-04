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
    python scripts/m1_babble_test.py --input-device 2 --output-device 0

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

# ── Babble Utterances ────────────────────────────────────────────────────
BABBLE_PHRASES = [
    "Hello. Is anyone there?",
    "The quick brown fox jumps over the lazy dog.",
    "Can you hear me speaking to you right now?",
    "Testing the acoustics of this room.",
    "One two three four five six seven eight nine ten.",
    "a e i o u",
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
) -> dict:
    """Execute the babbling protocol."""
    cerebellum = BabblingCerebellum(ir_length_ms=ir_length_ms)

    history = {
        "iteration": [],
        "rms_db": [],
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
    print(f"  Starting SVSR: N/A (blank slate)")
    print(f"  Learning:    NLMS adaptive filter")
    print("=" * width)

    for i in range(iterations):
        phrase = BABBLE_PHRASES[i % len(BABBLE_PHRASES)]
        print(f"\n--- Babble {i+1}/{iterations}: \"{phrase[:50]}...\"")

        # 1. Synthesize (TTS)
        pcm = await synthesize(phrase)
        tts_dur = len(pcm) / SR

        # 2. Capture with simultaneous playback
        capture_dur = tts_dur + 1.5
        print(f"  Recording for {capture_dur:.1f}s while playing...")

        # Device sample rate
        if input_device is not None:
            dev_info = sd.query_devices(input_device)
            native_sr = dev_info["default_samplerate"] or SR
        else:
            native_sr = SR

        total_samples = int(capture_dur * native_sr)
        captured = np.zeros(total_samples, dtype=np.float32)

        cap_stream = sd.InputStream(
            device=input_device,
            channels=1,
            samplerate=int(native_sr),
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
                captured = librosa.resample(captured, orig_sr=int(native_sr), target_sr=SR)
            except ImportError:
                factor = int(native_sr / SR)
                captured = captured[::factor]

        # Measure captured audio level
        rms = float(np.sqrt(np.mean(captured ** 2)))
        rms_db = 20 * np.log10(rms) if rms > 1e-10 else -100
        print(f"  Captured RMS: {rms_db:.1f} dB")

        if rms_db < -50:
            print(f"  WARNING: Signal is very quiet - check speaker volume!")

        # 3. Feed to cerebellum (babbling update)
        error = cerebellum.update(pcm, captured)
        conv = cerebellum.get_convergence()
        print(f"  Cerebellum updated (iter {i+1}, error: {conv['current_error_db']:.1f} dB)")

        # 4. Measure SVSR with the updated forward model
        observed_mel = pcm_to_mels(captured)
        t_obs = observed_mel.shape[0]

        # Use cerebellum to predict what the mic should hear
        # The cerebellum's predict_mel accounts for the learned room IR
        predicted_mel = cerebellum.predict_mel_aligned(pcm, t_obs)

        residual, mask = spectral_subtraction(observed_mel, predicted_mel)
        svsr = compute_svsr(observed_mel, predicted_mel, residual)

        is_converged = svsr >= 20.0
        print(f"  SVSR: {svsr:.1f} dB {'CONVERGED' if is_converged else '(target: >= 20 dB)'}")

        history["iteration"].append(i + 1)
        history["rms_db"].append(rms_db)
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
    args = parser.parse_args()

    asyncio.run(run_babbling(
        iterations=args.iterations,
        input_device=args.input_device,
        output_device=args.output_device,
        ir_length_ms=args.ir_length_ms,
    ))


if __name__ == "__main__":
    main()

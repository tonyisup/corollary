#!/usr/bin/env python3
"""M1 — Wandering Babble: Learn the Room From Multiple Positions.

Instead of babbling from one static position, the system prompts the user
to move the mic (or reposition themselves), then babbles again. The
cerebellum learns a composite room model that generalizes across positions.

The trust model tracks how well each new position matches the existing model.
If trust drops, the cerebellum re-babbles more aggressively to adapt.

Usage:
    # Interactive walking mode (prompts you to move between babbles)
    python -m scripts.m1_wandering_babble

    # Auto babble (no movement prompts, just continuous)
    python -m scripts.m1_wandering_babble --auto --positions 20

    # Custom parameters
    python -m scripts.m1_wandering_babble --input-device 2 --output-device 3 \
        --positions 10 --babbles-per-position 3
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

from broca.tts import synthesize
from cochlea.capture import pcm_to_mels
from cerebellum.adaptive_model import BabblingCerebellum
from cerebellum.trust_model import TrustTracker
from thalamus.gate import spectral_subtraction, compute_svsr

SR = 16000

# ── Movement Prompts ──────────────────────────────────────────────────
PROMPTS = [
    "Move the microphone to a different spot in the room. I'll babble when you're ready.",
    "Try a new position. Different height or angle would help me learn.",
    "I want to hear the room from somewhere else. Move me and say 'done' when ready.",
    "Change the acoustics. Closer or farther from the speakers — anywhere.",
    "One more position helps me understand this room. Move me around.",
]


async def capture_once(
    pcm: np.ndarray,
    input_device: int,
    output_device: int,
    native_sr: int,
) -> np.ndarray:
    """Play TTS and capture mic response in one shot."""
    tts_dur = len(pcm) / SR
    capture_dur = tts_dur + 1.5
    total = int(capture_dur * native_sr)
    captured = np.zeros(total, dtype=np.float32)

    cap = sd.InputStream(
        device=input_device, channels=1,
        samplerate=int(native_sr), dtype="float32", blocksize=1024,
    )
    cap.start()
    sd.play(pcm, samplerate=SR, device=output_device, blocking=False)

    pos = 0
    while pos < total:
        data, _ = cap.read(min(1024, total - pos))
        n = min(len(data), total - pos)
        captured[pos:pos+n] = data[:, 0]
        pos += n

    cap.stop()
    cap.close()

    # Resample if needed
    if native_sr != SR:
        try:
            import librosa
            captured = librosa.resample(captured, orig_sr=int(native_sr), target_sr=SR)
        except ImportError:
            factor = int(native_sr / SR)
            captured = captured[::factor]

    return captured


def compute_svsr_aligned(
    captured: np.ndarray,
    pcm: np.ndarray,
    cerebellum: BabblingCerebellum,
    search_range: int = 40,
) -> float:
    """Compute SVSR with time-alignment search."""
    observed_mel = pcm_to_mels(captured)
    predicted_mel = cerebellum.predict_mel(pcm)

    t_obs = observed_mel.shape[0]
    t_pred = predicted_mel.shape[0]

    best_svsr = 0.0
    for shift in range(-20, search_range):
        if shift >= 0:
            pred_shift = np.concatenate([
                np.tile(predicted_mel[:1], (shift, 1)),
                predicted_mel,
            ], axis=0)[:t_obs]
        else:
            start = -shift
            end = min(t_pred, t_obs + start)
            pred_shift = predicted_mel[start:end]
            if len(pred_shift) < t_obs:
                pred_shift = np.concatenate([
                    pred_shift,
                    np.zeros((t_obs - len(pred_shift), 80), dtype=np.float32),
                ])

        min_t = min(t_obs, len(pred_shift))
        res, _ = spectral_subtraction(observed_mel[:min_t], pred_shift[:min_t])
        svsr = compute_svsr(observed_mel[:min_t], pred_shift[:min_t], res)

        if svsr > best_svsr:
            best_svsr = svsr

    return best_svsr


async def run_wandering_babble(
    positions: int = 10,
    babbles_per_position: int = 5,
    input_device: int | None = None,
    output_device: int | None = None,
    auto_mode: bool = False,
) -> None:
    """Wandering babble: learn the room from multiple positions."""
    if input_device is not None:
        dev_info = sd.query_devices(input_device)
        native_sr = int(dev_info["default_samplerate"] or SR)
    else:
        native_sr = SR

    cerebellum = BabblingCerebellum(ir_length_ms=500, nlms_mu=0.25)

    phrasing = [
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

    print("\n" + "=" * 60)
    print("  COROLLARY — M1 Wandering Babble Mode")
    print("=" * 60)
    print(f"  Positions: {positions}")
    print(f"  Babbles per position: {babbles_per_position}")
    print(f"  Learning rate: mu=0.25, IR=500ms")
    print(f"  Strategy: explore → babble → update trust → move")
    print("=" * 60)

    all_svsr = []
    svsr_history = []

    for pos in range(positions):
        if auto_mode:
            print(f"\n=== Position {pos + 1}/{positions} ===")
        else:
            prompt = PROMPTS[pos % len(PROMPTS)]
            print(f"\n{'=' * 60}")
            print(f"  🚶 POSITION {pos + 1}/{positions}")
            print(f"  {prompt}")
            print(f"{'=' * 60}")
            input("  Press Enter when you're in position...")

        for babble in range(babbles_per_position):
            phrase = phrasing[(pos * babbles_per_position + babble) % len(phrasing)]
            print(f"  Babble {pos + 1}.{babble + 1}: \"{phrase[:40]}...\"")

            pcm = await synthesize(phrase)
            captured = await capture_once(pcm, input_device, output_device, native_sr)

            rms = float(np.sqrt(np.mean(captured ** 2)))
            rms_db = 20 * np.log10(rms) if rms > 1e-10 else -100

            # Update the cerebellum (it accumulates across all positions)
            err = cerebellum.update(pcm, captured)

            # Compute SVSR
            svsr = compute_svsr_aligned(captured, pcm, cerebellum)
            all_svsr.append(svsr)
            trust = cerebellum.trust_status()

            print(f"    RMS={rms_db:.1f} dB | SVSR={svsr:.1f} dB | "
                  f"Trust={trust['trust']:.3f} | Error={20*np.log10(max(err, 1e-10)):.1f} dB")

            # Trust drops at position change → the cerebellum knows the room shifted
            svsr_history.append({
                "position": pos + 1,
                "babble": babble + 1,
                "svsr": svsr,
                "trust": trust["trust"],
                "rms_db": rms_db,
            })

    # Summary
    print(f"\n{'=' * 60}")
    print("  WANDERING BABBLE COMPLETE")
    print("=" * 60)
    print(f"  Total babbles: {len(all_svsr)}")
    print(f"  Peak SVSR: {max(all_svsr):.1f} dB")
    print(f"  Mean SVSR: {np.mean(all_svsr):.1f} dB")
    print(f"  Final SVSR: {all_svsr[-1]:.1f} dB")
    print(f"  Final Trust: {cerebellum.trust_status()['trust']:.3f}")
    print(f"  SVSR range: {min(all_svsr):.1f} dB → {max(all_svsr):.1f} dB")
    print(f"  Trust range: {min(h['trust'] for h in svsr_history):.3f} → "
          f"{max(h['trust'] for h in svsr_history):.3f}")

    # Show per-position averages
    print(f"\n  Per-position mean SVSR:")
    by_pos = {}
    for h in svsr_history:
        p = h["position"]
        if p not in by_pos:
            by_pos[p] = []
        by_pos[p].append(h["svsr"])

    for p in sorted(by_pos.keys()):
        avg = np.mean(by_pos[p])
        max_s = max(by_pos[p])
        min_s = min(by_pos[p])
        print(f"    Position {p}: avg={avg:.1f} dB (range: {min_s:.1f}-{max_s:.1f} dB)")

    # Save the learned model
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    path = cerebellum.save_model(
        path=os.path.join(save_dir, f"wandering_room_pos{positions}.json"),
        metadata={
            "mode": "wandering_babble",
            "positions": positions,
            "babbles_per_position": babbles_per_position,
            "peak_svsr": max(all_svsr),
            "mean_svsr": float(np.mean(all_svsr)),
        },
    )
    print(f"\n  🧠 Room model saved to {path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="M1 Wandering Babble Mode")
    parser.add_argument("--positions", type=int, default=10)
    parser.add_argument("--babbles-per-position", type=int, default=5)
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--auto", action="store_true", help="Skip position prompts, just run")
    args = parser.parse_args()

    asyncio.run(run_wandering_babble(
        positions=args.positions,
        babbles_per_position=args.babbles_per_position,
        input_device=args.input_device,
        output_device=args.output_device,
        auto_mode=args.auto,
    ))


if __name__ == "__main__":
    main()

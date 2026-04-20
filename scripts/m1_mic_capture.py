#!/usr/bin/env python3
"""M1 — Microphone Capture Verification.

Tests real microphone input to prepare for the M1 hardware-in-the-loop.

Checks:
    1. Lists available audio input devices
    2. Captures a few seconds of audio from a chosen device
    3. Reports basic stats (RMS level, clipping, sample rate, etc.)
    4. Optionally plays the recording back through speakers

Usage:
    # List devices and pick one interactively
    python scripts/m1_mic_capture.py

    # Skip device selection, use device index 2 directly
    python scripts/m1_mic_capture.py --device 2

    # Skip playback, just capture and analyze
    python scripts/m1_mic_capture.py --no-playback

    # Capture longer (default 5 seconds)
    python scripts/m1_mic_capture.py --duration 10

Requirements:
    pip install sounddevice numpy
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# Add packages/ to sys.path so imports work when running scripts directly
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    print("Install with: pip install sounddevice")
    sys.exit(1)

from cochlea.capture import SR, HOP, WIN, N_MELS, pcm_to_mels


# ── Device Listing ─────────────────────────────────────────────────────────

def list_input_devices() -> list[dict]:
    """List all available audio input devices."""
    devices = sd.query_devices()
    inputs = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            inputs.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "default_sr": dev["default_samplerate"],
                "is_default": i == sd.default.device[0],
            })
    return inputs

def list_output_devices() -> list[dict]:
    """List all available audio output devices."""
    devices = sd.query_devices()
    outputs = []
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            outputs.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_output_channels"],
                "default_sr": dev["default_samplerate"],
                "is_default": i == sd.default.device[1],
            })
    return outputs

def print_input_devices_table(devices: list[dict]) -> None:
    """Print a formatted table of input devices."""
    print("\nAvailable audio INPUT devices:\n")
    print(f"{'Index':<8} {'Name':<50} {'Ch':<4} {'SR':<10} {'Default'}")
    print("-" * 80)
    for dev in devices:
        marker = " ← DEFAULT" if dev["is_default"] else ""
        sr = f"{dev['default_sr']:.0f}" if dev["default_sr"] else "N/A"
        print(
            f"{dev['index']:<8} {dev['name']:<50} {dev['channels']:<4} {sr:<10}{marker}"
        )
    print()


def print_output_devices_table(devices: list[dict]) -> None:
    """Print a formatted table of output devices."""
    print("\nAvailable audio OUTPUT devices:\n")
    print(f"{'Index':<8} {'Name':<50} {'Ch':<4} {'SR':<10} {'Default'}")
    print("-" * 80)
    for dev in devices:
        marker = " ← DEFAULT" if dev["is_default"] else ""
        sr = f"{dev['default_sr']:.0f}" if dev["default_sr"] else "N/A"
        print(
            f"{dev['index']:<8} {dev['name']:<50} {dev['channels']:<4} {sr:<10}{marker}"
        )
    print()




# ── Capture ────────────────────────────────────────────────────────────────

def capture_audio(
    device: int,
    duration: float = 5.0,
    sample_rate: int = SR,
    channels: int = 1,
) -> tuple[np.ndarray, int]:
    """Capture audio from a specific device.

    Returns:
        (audio_array, actual_sample_rate)
    """
    print(f"\n🎤 Recording for {duration}s from device {device}...")
    print(f"   Sample rate: {sample_rate} Hz | Channels: {channels}")
    print(f"   Speak now or make some noise!\n")

    # Countdown
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    print("   🎙️  RECORDING...\n")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    actual_sr = sd.query_devices(device)["default_samplerate"] or sample_rate

    print("   ✅ Recording complete.\n")
    return audio.flatten(), int(actual_sr)


# ── Analysis ───────────────────────────────────────────────────────────────

def analyze_audio(audio: np.ndarray, sample_rate: int) -> dict:
    """Compute basic audio statistics."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    dc_offset = float(np.mean(audio))
    rms_db = 20 * np.log10(rms) if rms > 0 else float("-inf")
    peak_db = 20 * np.log10(peak) if peak > 0 else float("-inf")

    # Check for clipping (samples at or very near ±1.0)
    clipping = float(np.mean(np.abs(audio) > 0.99)) * 100

    # Duration
    duration = len(audio) / sample_rate

    # Compute mel spectrogram as a sanity check
    mels = pcm_to_mels(audio)
    avg_mel_db = float(np.mean(mels))
    peak_mel_db = float(np.max(mels))

    return {
        "samples": len(audio),
        "duration_s": duration,
        "sample_rate": sample_rate,
        "rms": rms,
        "rms_db": rms_db,
        "peak": peak,
        "peak_db": peak_db,
        "dc_offset": dc_offset,
        "clipping_pct": clipping,
        "avg_mel_db": avg_mel_db,
        "peak_mel_db": peak_mel_db,
        "mel_shape": mels.shape,
    }


def print_analysis(stats: dict) -> None:
    """Print analysis results in a readable format."""
    print("\n📊 Audio Analysis:\n")
    print(f"  Duration:     {stats['duration_s']:.2f}s ({stats['samples']} samples)")
    print(f"  Sample Rate:  {stats['sample_rate']} Hz")
    print(f"  RMS Level:    {stats['rms_db']:.1f} dB (linear: {stats['rms']:.6f})")
    print(f"  Peak Level:   {stats['peak_db']:.1f} dB (linear: {stats['peak']:.6f})")
    print(f"  DC Offset:    {stats['dc_offset']:.6f}")
    print(f"  Clipping:     {stats['clipping_pct']:.2f}%")
    print(f"  Avg Mel dB:   {stats['avg_mel_db']:.1f} dB")
    print(f"  Peak Mel dB:  {stats['peak_mel_db']:.1f} dB")
    print(f"  Mel Shape:    {stats['mel_shape']}  (time frames × {N_MELS} bands)\n")

    # Health checks
    print("🩺 Health Checks:")

    if stats["rms_db"] < -50:
        print("  ⚠️  Signal is VERY quiet — increase mic gain or speak closer")
    elif stats["rms_db"] < -30:
        print("  ⚠️  Signal is quiet — consider increasing mic gain")
    elif stats["rms_db"] < -12:
        print("  ✅ Signal level is good")
    else:
        print("  ⚠️  Signal is loud — risk of clipping, reduce gain")

    if stats["clipping_pct"] > 0.1:
        print(f"  ❌ Clipping detected ({stats['clipping_pct']:.2f}%) — reduce gain!")
    else:
        print("  ✅ No clipping")

    if stats["peak_mel_db"] < -40:
        print("  ⚠️  Mel spectrogram is near silence — check audio chain")
    else:
        print("  ✅ Mel spectrogram has meaningful energy")

    # Latency note
    print(f"\n  📝 Latency: Corollary needs {'<50ms' if stats['duration_s'] < 0.05 else 'sub-100ms'} for M1 AEC")
    print(f"  📝 Cardioid mic: point the BACK toward speaker for initial AEC testing")


# ── Playback ───────────────────────────────────────────────────────────────

def playback_audio(
    audio: np.ndarray,
    device: int | None = None,
    sample_rate: int = SR,
) -> None:
    """Play audio through system speakers."""
    print("\n🔊 Playing back recording (3s for demo)...\n")
    playback = audio[: min(int(3 * sample_rate), len(audio))]
    sd.play(playback, samplerate=sample_rate, device=device)
    sd.wait()
    print("   ✅ Playback complete.\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="M1 Microphone Capture Verification")
    parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (default: interactive prompt)",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Recording duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--no-playback", action="store_true",
        help="Skip audio playback after recording",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Only list devices, don't capture",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  COROLLARY — M1 Microphone Capture Verification")
    print("=" * 60)

    # List devices
    devices = list_input_devices()
    if not devices:
        print("\n❌ No audio input devices found!")
        sys.exit(1)

    print_input_devices_table(devices)


    devices = list_output_devices()
    if not devices:
        print("\n❌ No audio output devices found!")
        sys.exit(1)

    print_output_devices_table(devices)
    if args.list_only:
        return

    # Select device
    device = args.device
    if device is None:
        # Default to first device, or prompt if there are multiple
        if len(devices) == 1:
            device = devices[0]["index"]
            print(f"Using device: {device} ({devices[0]['name']})")
        else:
            try:
                device = int(
                    input("Enter device index to test [Enter for first]: ")
                    or str(devices[0]["index"])
                )
            except ValueError:
                print("Invalid device index. Exiting.")
                sys.exit(1)

    # Determine sample rate
    dev_info = sd.query_devices(device)
    hw_sr = dev_info["default_samplerate"] or SR
    print(f"Device sample rate: {hw_sr} Hz")

    # Always capture at device native rate (avoids ALSA/CoreAudio resample bugs)
    # We'll resample for analysis if needed
    if hw_sr != SR:
        print(f"⚠️  Device native SR is {hw_sr}, Corollary expects {SR}. Will resample for analysis.")

    # Capture at native device rate
    audio, actual_sr = capture_audio(device, duration=args.duration, sample_rate=int(hw_sr))

    # Resample to 16kHz for Corollary analysis
    if actual_sr != SR:
        print(f"  Resampling {actual_sr} → {SR} Hz for analysis...")
        try:
            librosa = __import__("librosa")
            audio = librosa.resample(audio, orig_sr=actual_sr, target_sr=SR)
            actual_sr = SR
        except ImportError:
            # Fallback: simple decimation
            factor = int(actual_sr / SR)
            audio = audio[::factor]
            actual_sr = SR
            print(f"  ⚠️  librosa not available, using simple decimation")

    # Analyze
    stats = analyze_audio(audio, actual_sr)
    print_analysis(stats)

    # Playback
    if not args.no_playback:
        playback_audio(audio, sample_rate=actual_sr)

    print("=" * 60)
    print("  ✅ Capture verification complete.")
    print("  Next step: run the TTS → speaker → mic loop")
    print("=" * 60)


if __name__ == "__main__":
    main()

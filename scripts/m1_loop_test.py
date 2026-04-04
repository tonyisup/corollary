#!/usr/bin/env python3
"""M1 — Full TTS-to-Mic Loop Verification.

Tests the complete hardware-in-the-loop:
    1. Synthesizes speech via TTS (Broca)
    2. Plays it through speakers (system audio)
    3. Records simultaneously (or immediately after) via mic
    4. Runs the full Corollary pipeline on the captured audio:
       Cochlea → STG (ASR) → Thalamus (gate) → compare to original

This is the critical M1 milestone: does the system hear itself correctly?

Usage:
    # Full loop test (TTS → speaker → mic → ASR → gate)
    python scripts/m1_loop_test.py

    # Use specific input device (list with m1_mic_capture.py --list-only)
    python scripts/m1_loop_test.py --input-device 2

    # Use specific output device for TTS playback
    python scripts/m1_loop_test.py --output-device 3

    # Skip the ASR/gating, just measure capture quality
    python scripts/m1_loop_test.py --capture-only

    # Longer test utterance
    python scripts/m1_loop_test.py --text "The quick brown fox jumps over the lazy dog"

Requirements:
    pip install sounddevice numpy faster-whisper edge-tts

Hardware setup:
    - USB microphone connected and selected as input
    - Speakers playing TTS output so mic can hear it
    - Mic's cardioid null (back) should face speaker initially
    - Measure and note the mic-to-speaker distance (for room IR config)

Expected result:
    - ASR should transcribe something close to the original text
    - SVSR should be high (self-voice suppressed by efference copy)
    - Roundtrip time should be <5s for M0-scale models
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    print("Install with: pip install sounddevice")
    sys.exit(1)

# Add corollary packages to path regardless of cwd
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from broca.tts import synthesize
from cochlea.capture import SR, pcm_to_mels, EfferenceBuffer
from stg.asr import transcribe, DEFAULT_MODEL_SIZE
from thalamus.gate import ThalamicGate


# ── Default Test Utterance ────────────────────────────────────────────────

DEFAULT_TEXT = "Hello, this is Corollary testing the self-listening loop."


# ── TTS Playback ─────────────────────────────────────────────────────────

def play_audio_sync(
    audio: np.ndarray,
    device: int | None = None,
    sample_rate: int = SR,
) -> float:
    """Play audio synchronously and return duration in seconds."""
    sd.play(audio, samplerate=sample_rate, device=device, blocking=False)
    duration = len(audio) / sample_rate
    sd.wait()  # Blocks until playback completes
    return duration


# ── Capture ───────────────────────────────────────────────────────────────

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        factor = int(orig_sr / target_sr)
        return audio[::factor]


def capture_mic(
    duration: float,
    device: int | None = None,
    sample_rate: int = SR,
    channels: int = 1,
) -> tuple[np.ndarray, float]:
    """Record audio for the given duration.

    Returns:
        (audio_array, capture_start_time)
    """
    # Use the device's native sample rate, then resample to Corollary's SR
    if device is not None:
        dev_info = sd.query_devices(device)
        native_sr = dev_info["default_samplerate"] or sample_rate
    else:
        native_sr = sample_rate

    if native_sr != sample_rate:
        print(f"  ⚠️  Device native SR is {native_sr}, capturing at native rate...")

    num_samples = int(duration * native_sr)
    audio = np.zeros(num_samples, dtype=np.float32)

    t_start = time.perf_counter()

    stream = sd.InputStream(
        device=device,
        channels=channels,
        samplerate=int(native_sr),
        dtype="float32",
        blocksize=1024,
    )
    stream.start()

    pos = 0
    while pos < num_samples:
        data, _ = stream.read(min(1024, num_samples - pos))
        n = min(len(data), num_samples - pos)
        audio[pos:pos + n] = data[:, 0]  # Mono
        pos += n

    stream.stop()
    stream.close()

    t_end = time.perf_counter()

    # Resample to Corollary's SR if needed
    if native_sr != sample_rate:
        audio = resample_audio(audio, int(native_sr), sample_rate)

    return audio, t_start


# ── Analysis Helpers ─────────────────────────────────────────────────────_

def compute_similarity(original: str, transcript: str) -> float:
    """Simple word-level similarity (Jaccard-ish)."""
    orig_words = set(original.lower().split())
    trans_words = set(transcript.lower().split())
    if not orig_words and not trans_words:
        return 1.0
    if not orig_words or not trans_words:
        return 0.0
    intersection = orig_words & trans_words
    union = orig_words | trans_words
    return len(intersection) / len(union)


def print_header(text: str, char: str = "=") -> None:
    width = 60
    print(f"\n{'':{char}^{width}}")
    print(f"  {text}")
    print(f"{'':{char}^{width}}")


# ── Main Loop ──────────────────────────────────────────────────────────────

async def run_full_loop(
    text: str = DEFAULT_TEXT,
    input_device: int | None = None,
    output_device: int | None = None,
    capture_only: bool = False,
    whisper_model: str = DEFAULT_MODEL_SIZE,
) -> None:
    """Execute the TTS → speaker → mic → ASR → gate pipeline."""

    print_header("COROLLARY — M1 Full TTS-to-Mic Loop Test")
    print(f"\n  Utterance: \"{text}\"")
    t0 = time.perf_counter()

    # ── Step 1: TTS Synthesis ───────────────────────────────────────
    print("\n🗣️  Step 1: Synthesizing speech (TTS)...")
    tts_start = time.perf_counter()
    pcm = await synthesize(text)
    tts_s = time.perf_counter() - tts_start
    tts_dur = len(pcm) / SR
    print(f"  ✅ Generated {len(pcm)} samples ({tts_dur:.2f}s audio) in {tts_s:.2f}s")

    # ── Step 2: Store efference copy ─────────────────────────────────
    efference = EfferenceBuffer(max_ms=int(tts_dur * 1000 + 2000))
    efference.append(pcm)
    predicted_mel = pcm_to_mels(pcm)
    print(f"  📋 Efference copy stored, predicted mel: {predicted_mel.shape}")

    # ── Step 3: Capture with simultaneous playback ───────────────────
    # Add padding before/after to capture the full playback
    capture_duration = tts_dur + 2.0  # 1s lead + audio + 1s tail
    print(f"\n🎤 Step 2: Playing TTS and capturing mic for {capture_duration:.1f}s...")
    print(f"  📝 TTS should play through speakers NOW")
    print(f"  🔊 Speaker playback duration: {tts_dur:.2f}s")
    print(f"  ⏳ Speak only after playback to test gating...")

    # Countdown
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    # Determine mic's native sample rate
    if input_device is not None:
        dev_info = sd.query_devices(input_device)
        native_sr = dev_info["default_samplerate"] or SR
    else:
        native_sr = SR

    if native_sr != SR:
        print(f"  📝 Mic native SR: {native_sr} Hz (capturing at native, resampling after)")

    # Start capture (non-blocking)
    t_capture_start = time.perf_counter()
    cap_stream = sd.InputStream(
        device=input_device,
        channels=1,
        samplerate=int(native_sr),
        dtype="float32",
        blocksize=1024,
    )
    cap_stream.start()

    # Play TTS through speakers (non-blocking, so mic can hear it)
    sd.play(pcm, samplerate=SR, device=output_device, blocking=False)
    print(f"  🔊 TTS playback started...")

    # Record mic for capture_duration
    total_samples = int(capture_duration * native_sr)
    audio = np.zeros(total_samples, dtype=np.float32)
    pos = 0
    while pos < total_samples:
        data, _ = cap_stream.read(min(1024, total_samples - pos))
        n = min(len(data), total_samples - pos)
        audio[pos:pos + n] = data[:, 0]
        pos += n

    cap_stream.stop()
    cap_stream.close()

    # Resample to 16kHz if needed
    if native_sr != SR:
        audio = resample_audio(audio, int(native_sr), SR)

    capture_elapsed = time.perf_counter() - t_capture_start

    # Measure captured audio level
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    rms_db = 20 * np.log10(rms) if rms > 1e-10 else -100
    print(f"\n  ✅ Capture complete ({len(audio)} samples)")
    print(f"  📊 RMS: {rms_db:.1f} dB | Peak: {20*np.log10(peak) if peak > 1e-10 else -100:.1f} dB")

    if rms_db < -50:
        print("  ⚠️  Signal is very quiet — ensure speaker volume is up and mic is close")

    if capture_only:
        print("\n" + "=" * 60)
        print("  ✅ Capture-only mode. Skipping ASR and gating.")
        print("=" * 60)
        return

    # ── Step 4: ASR on captured audio ───────────────────────────────
    print(f"\n👂 Step 3: Transcribing with faster-whisper ({whisper_model} model)...")
    asr_start = time.perf_counter()
    asr_result = transcribe(audio, sample_rate=SR, model_size=whisper_model)
    asr_elapsed = time.perf_counter() - asr_start

    print(f"  ✅ Transcription ({asr_elapsed:.2f}s):")
    print(f'  📝 "{asr_result.text}"')

    if asr_result.segments:
        for seg in asr_result.segments:
            print(
                f"     [{seg.start:.2f}s → {seg.end:.2f}s] "
                f'"{seg.text.strip()}" (conf: {seg.confidence:.3f})'
            )

    # ── Step 5: Thalamic Gating ─────────────────────────────────────
    print(f"\n🚪 Step 4: Thalamic gating (efference copy subtraction)...")

    # Check if we have meaningful audio before running mel
    if rms_db < -60:
        print(f"  ⚠️  Captured audio is near silence ({rms_db:.1f} dB)")
        print(f"  The TTS was likely NOT heard by the mic.")
        print(f"  💡 Make sure speaker volume is UP and mic is nearby.")
        print(f"  Skipping gating (no signal to process).")
        return

    observed_mel = pcm_to_mels(audio)

    # Pad efference mel to match observed time length
    pcm_frames = predicted_mel.shape[0]
    obs_frames = observed_mel.shape[0]
    if pcm_frames < obs_frames:
        pad = np.zeros((obs_frames - pcm_frames, predicted_mel.shape[1]), dtype=np.float32)
        full_efference_mel = np.concatenate([predicted_mel, pad], axis=0)
    else:
        full_efference_mel = predicted_mel[:obs_frames]

    gate = ThalamicGate()
    result = gate.gate(observed_mel, full_efference_mel)
    svsr = result["svsr"]
    is_world = result["is_world"]

    print(f"  📊 SVSR: {svsr:.1f} dB (target: ≥20 dB)")
    print(f"  🏷️  Gate verdict: {'🌍 WORLD VOICE' if is_world else '🗣️  SELF VOICE SUPPRESSED'}")

    if not is_world:
        print(f"  ✅ Self-voice successfully gated. Residual: nearly zero.")
    else:
        print(f"  ⚠️  Significant residual — check:")
        print(f"      - Timing alignment between TTS and capture")
        print(f"      - Speaker volume vs mic sensitivity")
        print(f"      - Room acoustics (too much reverb)")

    # ── Summary ──────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t0
    similarity = compute_similarity(text, asr_result.text)

    print_header("RESULTS")
    print(f"\n  Original:      \"{text}\"")
    print(f"  ASR Output:    \"{asr_result.text}\"")
    print(f"  Word Similarity: {similarity:.1%}")
    print(f"  SVSR:            {svsr:.1f} dB {'✅' if svsr >= 20 else '⚠️'}")
    print(f"  Gate Verdict:    {'World Voice' if is_world else 'Self-Voice Suppressed'}")
    print(f"\n  Timing:")
    print(f"    TTS Synthesis:     {tts_s:.2f}s")
    print(f"    Capture:           {capture_elapsed:.2f}s")
    print(f"    ASR Transcription: {asr_elapsed:.2f}")
    print(f"    Total Roundtrip:   {total_elapsed:.2f}s")
    print("\n  " + "=" * 60)

    # Verdict
    if similarity > 0.5 and svsr >= 20:
        print("  🎉 M1 MILESTONE ACHIEVED:")
        print("     TTS → speaker → mic → ASR → gate pipeline is functional!")
    elif similarity > 0.3:
        print("  📊 Promising but needs tuning:")
        print("     Some capture worked, check timing alignment and audio levels")
    else:
        print("  🔧 Needs work:")
        print("     Check hardware setup, audio routing, and timing")
    print("  " + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 Full TTS-to-Mic Loop Verification")
    parser.add_argument(
        "--text", type=str, default=DEFAULT_TEXT,
        help="Test utterance to synthesize and capture",
    )
    parser.add_argument(
        "--input-device", type=int, default=None,
        help="Audio input device index",
    )
    parser.add_argument(
        "--output-device", type=int, default=None,
        help="Audio output device index for TTS playback",
    )
    parser.add_argument(
        "--capture-only", action="store_true",
        help="Only capture mic during TTS playback, skip ASR/gating",
    )
    parser.add_argument(
        "--whisper-model", type=str, default=DEFAULT_MODEL_SIZE,
        help="Whisper model size (tiny, base, small, medium)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_full_loop(
            text=args.text,
            input_device=args.input_device,
            output_device=args.output_device,
            capture_only=args.capture_only,
            whisper_model=args.whisper_model,
        )
    )


if __name__ == "__main__":
    main()

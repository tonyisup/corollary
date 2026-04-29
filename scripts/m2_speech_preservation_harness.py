#!/usr/bin/env python3
"""M2 — Speech-Aware Self/World Separation Harness.

Goal: Suppress "Self" audio while preserving "World" speech.

Execution:
1. Play Self-Babble (Device 4).
2. Play World-Speech (Device 3).
3. Capture and use ThalamicAttributor + VAD to label segments.
4. Run Whisper ASR on both raw and gated audio to measure preservation.
5. Optionally score against a reference transcript and save artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def levenshtein(seq1: list[str], seq2: list[str]) -> int:
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1
    prev = list(range(len(seq2) + 1))
    for i, a in enumerate(seq1, start=1):
        curr = [i]
        for j, b in enumerate(seq2, start=1):
            cost = 0 if a == b else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


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


def build_gating_gain(
    verdicts: list[SourceVerdict],
    frame_samples: int,
    total_samples: int,
    both_gain: float,
) -> np.ndarray:
    """Build a sample-level gain vector from frame verdicts."""
    gain = np.ones(total_samples, dtype=np.float32)
    both_gain = float(np.clip(both_gain, 0.0, 1.0))

    for i, verdict in enumerate(verdicts):
        start = i * frame_samples
        end = min(start + frame_samples, total_samples)
        if verdict == SourceVerdict.SELF:
            gain[start:end] = 0.0
        elif verdict == SourceVerdict.BOTH:
            gain[start:end] = both_gain

    return gain


def run_m2_trial(
    self_stim: str,
    world_speech_file: str,
    input_device: int,
    output_device: int,
    noise_device: int,
    lead_in_s: float = 0.5,
    tail_s: float = 1.0,
    seed: int = 0,
    expected_text: str | None = None,
    both_gain: float = 0.5,
    self_duration_ms: int = 350,
    self_amplitude: float = 0.7,
) -> dict[str, Any]:
    self_meta = generate_stimulus_with_metadata(self_stim, self_duration_ms, self_amplitude, seed=seed)
    self_pcm = self_meta["pcm"]

    if world_speech_file.endswith(".npy"):
        world_pcm = np.load(world_speech_file)
    else:
        world_pcm = np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR).astype(np.float32) * 0.5

    self_duration = len(self_pcm) / SR
    world_duration = len(world_pcm) / SR

    world_start_s = lead_in_s
    self_start_s = world_start_s + (world_duration / 2.0) - (self_duration / 2.0)
    total_duration = max(world_start_s + world_duration, self_start_s + self_duration) + tail_s

    input_sr = int(sd.query_devices(input_device)["default_samplerate"] or SR)
    output_sr = int(sd.query_devices(output_device)["default_samplerate"] or SR)
    noise_sr = int(sd.query_devices(noise_device)["default_samplerate"] or SR)

    def get_padded_pcm(pcm: np.ndarray, start_s: float, total_s: float, target_sr: float) -> np.ndarray:
        padded = np.zeros(int(total_s * target_sr), dtype=np.float32)
        start_idx = int(start_s * target_sr)
        if abs(target_sr - SR) > 1:
            import librosa
            pcm = librosa.resample(pcm, orig_sr=SR, target_sr=int(target_sr))
        end_idx = min(len(padded), start_idx + len(pcm))
        padded[start_idx:end_idx] = pcm[: end_idx - start_idx]
        return padded

    self_padded = get_padded_pcm(self_pcm, self_start_s, total_duration, output_sr)
    world_padded = get_padded_pcm(world_pcm, world_start_s, total_duration, noise_sr)

    print(f"\n--- M2 Speech Preservation Trial ---")
    print(f"Self: {self_stim} at {self_start_s:.2f}s")
    print(f"World Speech: {world_duration:.2f}s at {world_start_s:.2f}s")

    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

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
        captured_audio[pos:pos + n] = data[:n, 0]
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

    attributor = ThalamicAttributor(sr=SR)
    vad = SmoothedVAD()

    self_score, self_shift = normalized_xcorr(observed, self_pcm)
    self_expected_shift = int(self_start_s * SR)

    num_frames = len(observed) // VAD_FRAME_SAMPLES
    verdicts = []
    speech_frames = 0

    for i in range(num_frames):
        start = i * VAD_FRAME_SAMPLES
        end = start + VAD_FRAME_SAMPLES
        frame = observed[start:end]

        is_speech = vad.process_frame(frame)
        if is_speech:
            speech_frames += 1

        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_db = db(rms)
        is_playing = (start >= self_expected_shift) and (end <= self_expected_shift + len(self_pcm))

        res = attributor.attribute(
            match_score=self_score if is_playing else 0.0,
            observed_onset_samples=self_shift,
            expected_onset_samples=self_expected_shift if is_playing else None,
            rms_db=rms_db,
            is_playing=is_playing,
            is_speech=is_speech,
        )
        verdicts.append(res.verdict)

    gain = build_gating_gain(verdicts, VAD_FRAME_SAMPLES, len(observed), both_gain)
    mask = gain > 0.0

    print("\nRunning ASR on Raw Audio...")
    raw_res = transcribe(observed, model_size="base")
    print(f"Raw ASR: \"{raw_res.text}\"")

    print("\nRunning ASR on Gated Audio...")
    gated_res = transcribe(observed, mask=gain, model_size="base")
    print(f"Gated ASR: \"{gated_res.text}\"")

    metrics: dict[str, Any] = {}
    if expected_text:
        raw_wer = word_error_rate(expected_text, raw_res.text)
        gated_wer = word_error_rate(expected_text, gated_res.text)
        metrics = {
            "expected_text": expected_text,
            "raw_wer": raw_wer,
            "gated_wer": gated_wer,
            "wer_delta": gated_wer - raw_wer,
        }
        print("\nReference Transcript Metrics:")
        print(f"Expected: \"{expected_text}\"")
        print(f"Raw WER:   {raw_wer:.3f}")
        print(f"Gated WER: {gated_wer:.3f}")

    return {
        "raw_text": raw_res.text,
        "gated_text": gated_res.text,
        "self_score": self_score,
        "speech_ratio": speech_frames / num_frames,
        "self_frames": sum(1 for v in verdicts if v == SourceVerdict.SELF),
        "both_frames": sum(1 for v in verdicts if v == SourceVerdict.BOTH),
        "world_frames": sum(1 for v in verdicts if v == SourceVerdict.WORLD),
        "metrics": metrics,
        "observed": observed,
        "mask": mask,
        "gain": gain,
        "both_gain": both_gain,
        "seed": seed,
        "self_duration_ms": self_duration_ms,
        "self_amplitude": self_amplitude,
    }


def save_trial_artifacts(save_dir: str, result: dict[str, Any]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "observed.npy"), result["observed"])
    np.save(os.path.join(save_dir, "gating_mask.npy"), result["mask"].astype(np.uint8))
    np.save(os.path.join(save_dir, "gating_gain.npy"), result["gain"].astype(np.float32))

    summary = {
        "raw_text": result["raw_text"],
        "gated_text": result["gated_text"],
        "self_score": result["self_score"],
        "speech_ratio": result["speech_ratio"],
        "self_frames": result["self_frames"],
        "both_frames": result["both_frames"],
        "world_frames": result["world_frames"],
        "both_gain": result["both_gain"],
        "seed": result["seed"],
        "self_duration_ms": result["self_duration_ms"],
        "self_amplitude": result["self_amplitude"],
        "metrics": result["metrics"],
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="M2 Speech Preservation Harness")
    parser.add_argument("--self-stim", default="babble_ba_repeat_short_tight_var")
    parser.add_argument("--world-speech", default="data/world_speech_sample.npy")
    parser.add_argument("--input-device", type=int, default=2)
    parser.add_argument("--output-device", type=int, default=4)
    parser.add_argument("--noise-device", type=int, default=3)
    parser.add_argument("--expected-text", default=None, help="Reference transcript for WER measurement")
    parser.add_argument("--save-dir", default=None, help="Optional directory to save observed audio, mask, and summary")
    parser.add_argument("--both-gain", type=float, default=0.5, help="Soft gain applied to BOTH frames before ASR")
    parser.add_argument("--seed", type=int, default=0, help="Stimulus seed")
    parser.add_argument("--self-duration-ms", type=int, default=350, help="Nominal self-stimulus duration in milliseconds")
    parser.add_argument("--self-amplitude", type=float, default=0.7, help="Self-stimulus playback amplitude")
    args = parser.parse_args()

    res = run_m2_trial(
        self_stim=args.self_stim,
        world_speech_file=args.world_speech,
        input_device=args.input_device,
        output_device=args.output_device,
        noise_device=args.noise_device,
        seed=args.seed,
        expected_text=args.expected_text,
        both_gain=args.both_gain,
        self_duration_ms=args.self_duration_ms,
        self_amplitude=args.self_amplitude,
    )

    if args.save_dir:
        save_trial_artifacts(args.save_dir, res)
        print(f"Saved artifacts to: {args.save_dir}")

    print("\n--- M2 Final Verdict ---")
    print(f"Self Match Score: {res['self_score']:.4f}")
    print(f"Speech Ratio: {res['speech_ratio']:.1%}")
    print(f"Self/Both/World Frames: {res['self_frames']}/{res['both_frames']}/{res['world_frames']}")

    if res["metrics"]:
        if res["metrics"]["gated_wer"] <= res["metrics"]["raw_wer"]:
            print("\n✅ SUCCESS: Gated transcript preserved or improved reference accuracy.")
        else:
            print("\n⚠️ MIXED: Gated transcript was worse than raw against the reference text.")
    elif len(res["gated_text"]) > 5:
        print("\n✅ SUCCESS: World speech preserved after self-voice gating!")
    else:
        print("\n❌ FAILURE: World speech was lost or masked.")


if __name__ == "__main__":
    main()

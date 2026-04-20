#!/usr/bin/env python3
"""Hardware-facing M1A primitive probe playback + mic capture.

Purpose:
    Bridge synthetic M1A stimulus work into the first audible real-world path.

What it does:
    1. Generates a simple primitive probe (chirp, vowel, tone, click)
    2. Plays it through a real output device
    3. Records mic input during playback
    4. Reports timing/alignment and basic self-attribution-friendly stats

This is intentionally simpler than the full TTS loop. It is meant to be the
first hardware-facing M1A step, before ASR or lexical evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from statistics import mean, pstdev
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    print("Install with: pip install sounddevice")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cochlea.capture import SR

sys.path.insert(0, os.path.dirname(__file__))
from m1_tone_attribution import generate_stimulus_with_metadata


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        factor = max(int(orig_sr / target_sr), 1)
        return audio[::factor].astype(np.float32)


def list_devices() -> None:
    devices = sd.query_devices()
    print("\nAvailable audio devices:\n")
    print(f"{'Index':<8} {'Name':<50} {'In':<4} {'Out':<4} {'Default SR'}")
    print("-" * 84)
    for i, dev in enumerate(devices):
        print(
            f"{i:<8} {dev['name']:<50} {dev['max_input_channels']:<4} "
            f"{dev['max_output_channels']:<4} {dev['default_samplerate']:.0f}"
        )
    print()


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


def estimate_envelope_onset(audio: np.ndarray, threshold_ratio: float = 0.2, smooth_ms: float = 12.0) -> int:
    if len(audio) == 0:
        return 0
    magnitude = np.abs(audio.astype(np.float32))
    window = max(1, int(SR * smooth_ms / 1000.0))
    kernel = np.ones(window, dtype=np.float32) / window
    envelope = np.convolve(magnitude, kernel, mode="same")
    peak = float(np.max(envelope))
    if peak < 1e-9:
        return 0
    threshold = peak * threshold_ratio
    idx = np.where(envelope >= threshold)[0]
    return int(idx[0]) if len(idx) else 0


def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-10))


def classify_run_quality(capture_rms_db: float, match_score: float, onset_error_ms: float) -> tuple[str, list[str]]:
    flags: list[str] = []
    if capture_rms_db < -50:
        flags.append("quiet_capture")
    if match_score < 0.2:
        flags.append("weak_match")
    if onset_error_ms > 150.0:
        flags.append("unstable_onset")

    if not flags:
        quality = "good"
    elif capture_rms_db >= -50 and match_score >= 0.2 and onset_error_ms <= 300.0:
        quality = "borderline"
    else:
        quality = "bad"
    return quality, flags


def run_probe(
    stimulus: str,
    duration_ms: int,
    amplitude: float,
    input_device: int | None,
    output_device: int | None,
    lead_in_s: float,
    tail_s: float,
    stimulus_seed: int = 0,
    setup_label: str | None = None,
    save_path: str | None = None,
) -> dict[str, Any]:
    stimulus_meta = generate_stimulus_with_metadata(stimulus, duration_ms, amplitude, seed=stimulus_seed)
    pcm = stimulus_meta["pcm"]
    playback_duration = len(pcm) / SR

    input_name = "default"
    output_name = "default"
    if input_device is not None:
        dev_info = sd.query_devices(input_device)
        native_sr = int(dev_info["default_samplerate"] or SR)
        input_name = str(dev_info["name"])
    else:
        native_sr = SR

    if output_device is not None:
        output_name = str(sd.query_devices(output_device)["name"])

    capture_duration = lead_in_s + playback_duration + tail_s
    total_samples = int(capture_duration * native_sr)
    audio = np.zeros(total_samples, dtype=np.float32)

    print("\n" + "=" * 60)
    print("  COROLLARY — M1 Primitive Probe Capture")
    print("=" * 60)
    print(f"Stimulus: {stimulus}")
    if stimulus in {"vowel_ah_var", "vowel_ah_var_nearfield"}:
        print(
            f"Stimulus seed:     {stimulus_meta['seed']} | realized duration: {stimulus_meta['duration_ms']} ms | "
            f"f0: {stimulus_meta['f0_hz']} Hz"
        )
    print(f"Playback duration: {playback_duration:.3f}s")
    print(f"Capture duration:  {capture_duration:.3f}s")
    print(f"Input device:      {input_device if input_device is not None else 'default'} ({input_name})")
    print(f"Output device:     {output_device if output_device is not None else 'default'} ({output_name})")
    if setup_label:
        print(f"Setup label:       {setup_label}")
    if native_sr != SR:
        print(f"Mic native SR:     {native_sr} Hz (will resample to {SR} Hz)")

    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    stream = sd.InputStream(
        device=input_device,
        channels=1,
        samplerate=native_sr,
        dtype="float32",
        blocksize=1024,
    )
    stream.start()

    lead_in_samples = int(lead_in_s * native_sr)
    pos = 0
    playback_started_at: float | None = None
    capture_started_at = time.perf_counter()

    while pos < total_samples:
        if playback_started_at is None and pos >= lead_in_samples:
            sd.play(pcm, samplerate=SR, device=output_device, blocking=False)
            playback_started_at = time.perf_counter()
        data, _ = stream.read(min(1024, total_samples - pos))
        n = min(len(data), total_samples - pos)
        audio[pos:pos + n] = data[:n, 0]
        pos += n

    stream.stop()
    stream.close()
    sd.wait()
    capture_finished_at = time.perf_counter()

    if native_sr != SR:
        audio = resample_audio(audio, native_sr, SR)

    observed = audio.astype(np.float32)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.savez(save_path, known_pcm=pcm, captured_pcm=observed, sr=SR, metadata=stimulus_meta)

    score, shift = normalized_xcorr(observed, pcm)
    envelope_shift = estimate_envelope_onset(observed)
    observed_rms = float(np.sqrt(np.mean(observed ** 2))) if len(observed) else 0.0
    observed_peak = float(np.max(np.abs(observed))) if len(observed) else 0.0

    scheduled_expected_shift = int(lead_in_s * SR)
    actual_expected_shift = scheduled_expected_shift
    playback_start_delay_ms = None
    if playback_started_at is not None:
        playback_start_delay_s = max(0.0, playback_started_at - capture_started_at)
        playback_start_delay_ms = playback_start_delay_s * 1000.0
        actual_expected_shift = int(playback_start_delay_s * SR)

    onset_error_ms = abs(shift - actual_expected_shift) * 1000.0 / SR
    scheduled_onset_error_ms = abs(shift - scheduled_expected_shift) * 1000.0 / SR
    envelope_onset_error_ms = abs(envelope_shift - actual_expected_shift) * 1000.0 / SR
    quality, quality_flags = classify_run_quality(db(observed_rms), score, onset_error_ms)

    return {
        "stimulus": stimulus,
        "stimulus_family": stimulus_meta["family"],
        "stimulus_seed": stimulus_meta["seed"],
        "stimulus_duration_realized_ms": stimulus_meta["duration_ms"],
        "stimulus_amplitude_realized": stimulus_meta["amplitude"],
        "stimulus_attack_ms": stimulus_meta["attack_ms"],
        "stimulus_release_ms": stimulus_meta["release_ms"],
        "stimulus_f0_hz": stimulus_meta["f0_hz"],
        "stimulus_envelope_lfo_hz": stimulus_meta["envelope_lfo_hz"],
        "setup_label": setup_label,
        "playback_duration_s": round(playback_duration, 4),
        "input_device_name": input_name,
        "output_device_name": output_name,
        "capture_duration_s": round(capture_duration, 4),
        "capture_rms_db": round(db(observed_rms), 2),
        "capture_peak_db": round(db(observed_peak), 2),
        "match_score": round(score, 6),
        "scheduled_expected_onset_samples": scheduled_expected_shift,
        "actual_expected_onset_samples": actual_expected_shift,
        "observed_onset_samples": shift,
        "envelope_onset_samples": envelope_shift,
        "playback_start_delay_ms": None if playback_start_delay_ms is None else round(playback_start_delay_ms, 4),
        "onset_error_ms": round(onset_error_ms, 4),
        "scheduled_onset_error_ms": round(scheduled_onset_error_ms, 4),
        "envelope_onset_error_ms": round(envelope_onset_error_ms, 4),
        "capture_elapsed_s": round(capture_finished_at - capture_started_at, 4),
        "playback_started": playback_started_at is not None,
        "quality": quality,
        "quality_flags": quality_flags,
    }


def run_paired_probe(
    stimulus: str,
    duration_ms: int,
    amplitude: float,
    input_device: int | None,
    output_device: int | None,
    lead_in_s: float,
    tail_s: float,
    gap_s: float,
    stimulus_seed: int = 0,
    setup_label: str | None = None,
    save_capture: bool = False,
    output_dir: str = "data/benchmark_samples",
) -> dict[str, Any]:
    save_path_1 = None
    save_path_2 = None
    if save_capture:
        ts = int(time.time())
        save_path_1 = os.path.join(output_dir, f"{stimulus}_seed{stimulus_seed}_{ts}_1.npz")
        save_path_2 = os.path.join(output_dir, f"{stimulus}_seed{stimulus_seed + 1}_{ts}_2.npz")

    first = run_probe(
        stimulus,
        duration_ms,
        amplitude,
        input_device,
        output_device,
        lead_in_s,
        tail_s,
        stimulus_seed=stimulus_seed,
        setup_label=setup_label,
        save_path=save_path_1,
    )
    print(f"\nPaired probe gap: sleeping {gap_s:.3f}s before second run...\n")
    time.sleep(gap_s)
    second = run_probe(
        stimulus,
        duration_ms,
        amplitude,
        input_device,
        output_device,
        lead_in_s,
        tail_s,
        stimulus_seed=stimulus_seed + 1,
        setup_label=setup_label,
        save_path=save_path_2,
    )

    match_scores = [float(first["match_score"]), float(second["match_score"])]
    onset_errors = [float(first["onset_error_ms"]), float(second["onset_error_ms"])]
    good_runs = sum(1 for r in (first, second) if r.get("quality") == "good")

    envelope_onset_errors = [float(first["envelope_onset_error_ms"]), float(second["envelope_onset_error_ms"])]

    return {
        "mode": "paired",
        "stimulus": stimulus,
        "stimulus_family": first.get("stimulus_family"),
        "stimulus_seed_start": stimulus_seed,
        "setup_label": setup_label,
        "input_device_name": first.get("input_device_name"),
        "output_device_name": first.get("output_device_name"),
        "pair_gap_s": round(gap_s, 4),
        "runs": [first, second],
        "mean_match_score": round(mean(match_scores), 6),
        "match_score_std": round(pstdev(match_scores), 6) if len(match_scores) > 1 else 0.0,
        "mean_onset_error_ms": round(mean(onset_errors), 4),
        "onset_error_std_ms": round(pstdev(onset_errors), 4) if len(onset_errors) > 1 else 0.0,
        "mean_envelope_onset_error_ms": round(mean(envelope_onset_errors), 4),
        "envelope_onset_error_std_ms": round(pstdev(envelope_onset_errors), 4) if len(envelope_onset_errors) > 1 else 0.0,
        "best_match_score": round(max(match_scores), 6),
        "best_onset_error_ms": round(min(onset_errors), 4),
        "best_envelope_onset_error_ms": round(min(envelope_onset_errors), 4),
        "good_runs": good_runs,
        "paired_quality": "good" if good_runs == 2 else ("borderline" if good_runs == 1 else "bad"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware-facing primitive M1A probe capture")
    parser.add_argument("--stimulus", choices=["chirp", "vowel_ah", "vowel_ah_var", "vowel_ah_var_nearfield", "vowel_ah_var_nearfield_room2", "babble_ba_var", "babble_ba_repeat_var", "babble_ba_repeat_short_var", "babble_ba_repeat_few_var", "babble_ba_repeat_wide_var", "babble_ba_repeat_short_tight_var", "babble_da_var", "babble_ma_var", "tone", "click"], default="chirp")
    parser.add_argument("--duration-ms", type=int, default=300)
    parser.add_argument("--amplitude", type=float, default=0.7)
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.0)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--paired", action="store_true", help="Run the same probe twice with a short gap")
    parser.add_argument("--pair-gap-s", type=float, default=0.35)
    parser.add_argument("--stimulus-seed", type=int, default=0, help="Seed for variable stimulus generation")
    parser.add_argument("--setup-label", default=None, help="Short label for the physical setup / geometry")
    parser.add_argument("--save-capture", action="store_true", help="Save capture(s) as .npz for later analysis")
    parser.add_argument("--output-dir", default="data/benchmark_samples", help="Directory for saved capture .npz files")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.paired:
        result = run_paired_probe(
            stimulus=args.stimulus,
            duration_ms=args.duration_ms,
            amplitude=args.amplitude,
            input_device=args.input_device,
            output_device=args.output_device,
            lead_in_s=args.lead_in_s,
            tail_s=args.tail_s,
            gap_s=args.pair_gap_s,
            stimulus_seed=args.stimulus_seed,
            setup_label=args.setup_label,
            save_capture=args.save_capture,
            output_dir=args.output_dir,
        )
        print("\nPaired results:\n")
        print(f"  stimulus: {result['stimulus']}")
        if 'stimulus_seed_start' in result:
            print(f"  stimulus_seed_start: {result['stimulus_seed_start']}")
        print(f"  input_device_name: {result['input_device_name']}")
        print(f"  output_device_name: {result['output_device_name']}")
        print(f"  pair_gap_s: {result['pair_gap_s']}")
        if result.get('setup_label'):
            print(f"  setup_label: {result['setup_label']}")
        print(f"  mean_match_score: {result['mean_match_score']}")
        print(f"  match_score_std: {result['match_score_std']}")
        print(f"  mean_onset_error_ms: {result['mean_onset_error_ms']}")
        print(f"  onset_error_std_ms: {result['onset_error_std_ms']}")
        print(f"  mean_envelope_onset_error_ms: {result['mean_envelope_onset_error_ms']}")
        print(f"  envelope_onset_error_std_ms: {result['envelope_onset_error_std_ms']}")
        print(f"  best_match_score: {result['best_match_score']}")
        print(f"  best_onset_error_ms: {result['best_onset_error_ms']}")
        print(f"  best_envelope_onset_error_ms: {result['best_envelope_onset_error_ms']}")
        print(f"  good_runs: {result['good_runs']}")
        print(f"  paired_quality: {result['paired_quality']}")
        print("\nInterpretation:")
        print("  Paired mode is a first stab at onset stabilization via repeated measurement.")
        print("  Lower onset_error_std_ms means the same probe is landing more consistently across repeats.")
        print("  Next step: compare paired chirp vs paired vowel probes on the same hardware path.")
    else:
        save_path = None
        if args.save_capture:
            ts = int(time.time())
            save_path = os.path.join(args.output_dir, f"{args.stimulus}_seed{args.stimulus_seed}_{ts}.npz")

        result = run_probe(
            stimulus=args.stimulus,
            duration_ms=args.duration_ms,
            amplitude=args.amplitude,
            input_device=args.input_device,
            output_device=args.output_device,
            lead_in_s=args.lead_in_s,
            tail_s=args.tail_s,
            stimulus_seed=args.stimulus_seed,
            setup_label=args.setup_label,
            save_path=save_path,
        )

        print("\nResults:\n")
        for key, value in result.items():
            print(f"  {key}: {value}")

        print("\nInterpretation:")
        if result["capture_rms_db"] < -50:
            print("  ⚠️  Very quiet capture. Check speaker volume, mic gain, and placement.")
        else:
            print("  ✅ Mic captured meaningful signal.")

        if result["match_score"] >= 0.5:
            print("  ✅ Strong probe-template match.")
        elif result["match_score"] >= 0.2:
            print("  ⚠️  Weak but visible probe-template match.")
        else:
            print("  ❌ Probe-template match is poor. Check routing and room setup.")

        print(f"  Run quality: {result['quality']}")
        if result["quality_flags"]:
            print(f"  Flags: {', '.join(result['quality_flags'])}")

        print("  Next step: compare chirp vs vowel probes on the same hardware path.")


if __name__ == "__main__":
    main()

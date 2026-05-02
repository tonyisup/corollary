
#!/usr/bin/env python3
"""Calibration script for M2 benchmark timing.

Measures the true offset between known self PCM and captured audio.
This determines the effective onset_latency_ms to use in m2_benchmark.py.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from cochlea.capture import SR
from m1_tone_attribution import generate_stimulus_with_metadata


def normalized_xcorr(observed, template):
    """Normalized cross-correlation. Returns (best_score, best_shift)."""
    if len(observed) < len(template):
        return 0.0, 0
    t_norm = float(np.linalg.norm(template))
    if t_norm < 1e-9:
        return 0.0, 0
    best_score = -1.0
    best_shift = 0
    step = max(len(template) // 80, 1)
    for shift in range(0, len(observed) - len(template) + 1, step):
        window = observed[shift:shift + len(template)]
        denom = float(np.linalg.norm(window) * t_norm)
        score = 0.0 if denom < 1e-9 else float(np.dot(window, template) / denom)
        if score > best_score:
            best_score = score
            best_shift = shift
    return max(0.0, best_score), best_shift


def local_xcorr(observed, template, expected_shift, search_ms=250.0):
    """Local cross-correlation around expected shift."""
    if len(observed) < len(template):
        return 0.0, expected_shift
    t = template.astype(np.float64) - np.mean(template)
    t_std = np.std(template)
    if t_std < 1e-9:
        return 0.0, expected_shift
    t = t / t_std
    radius = int(search_ms * SR / 1000.0)
    start = max(0, expected_shift - radius)
    stop = min(len(observed) - len(template), expected_shift + radius)
    if stop <= start:
        return 0.0, expected_shift
    best_score = -1.0
    best_shift = start
    step = max(len(template) // 160, 1)
    for shift in range(start, stop + 1, step):
        w = observed[shift:shift + len(template)].astype(np.float64)
        w = w - np.mean(w)
        w_std = np.std(w)
        if w_std < 1e-9:
            score = 0.0
        else:
            w = w / w_std
            score = float(np.clip(np.dot(w, t) / len(t), -1.0, 1.0))
        if score > best_score:
            best_score = score
            best_shift = shift
    return max(0.0, best_score), best_shift


def energy_onset(observed, start_hint_samples, search_before_ms=0.0,
                 search_after_ms=800.0, frame_samples=512, threshold_ratio=0.20):
    """Find onset via energy rise."""
    fs = frame_samples
    start = max(0, start_hint_samples - int(search_before_ms * SR / 1000.0))
    stop = min(len(observed), start_hint_samples + int(search_after_ms * SR / 1000.0))
    if stop <= start + fs:
        return start_hint_samples
    frames = []
    for pos in range(start, stop - fs + 1, fs):
        frame = observed[pos:pos + fs]
        frames.append((pos, float(np.sqrt(np.mean(frame ** 2)))))
    if not frames:
        return start_hint_samples
    floor = float(np.median([r for _, r in frames[:max(1, min(5, len(frames)))]]) )
    peak = max(r for _, r in frames)
    threshold = floor + (peak - floor) * float(np.clip(threshold_ratio, 0.0, 1.0))
    for pos, rms in frames:
        if rms >= threshold:
            return pos
    return max(frames, key=lambda x: x[1])[0]


def play_array_blocking(data, device, samplerate):
    if data.ndim == 1:
        data = data[:, np.newaxis]
    with sd.OutputStream(device=device, channels=data.shape[1],
                         samplerate=samplerate, dtype="float32") as stream:
        stream.write(np.asarray(data, dtype=np.float32))


def run_capture(self_pcm, world_pcm, input_device, output_device, noise_device,
                lead_in_s=0.5, tail_s=1.5):
    """Run hardware capture. Returns (observed, self_start_s, world_start_s, total_dur)."""
    self_duration = len(self_pcm) / SR
    world_duration = len(world_pcm) / SR
    world_start_s = lead_in_s
    self_start_s = world_start_s + (world_duration / 2.0) - (self_duration / 2.0)
    total_duration = max(world_start_s + world_duration,
                         self_start_s + self_duration) + tail_s

    input_sr = int(sd.query_devices(input_device)["default_samplerate"] or SR)
    output_sr = int(sd.query_devices(output_device)["default_samplerate"] or SR)
    noise_sr = int(sd.query_devices(noise_device)["default_samplerate"] or SR)

    def get_padded(pcm, start_s, total_s, target_sr):
        padded = np.zeros(int(total_s * target_sr), dtype=np.float32)
        start_idx = int(start_s * target_sr)
        if abs(target_sr - SR) > 1:
            import librosa
            pcm = librosa.resample(pcm, orig_sr=SR, target_sr=target_sr)
        end_idx = min(len(padded), start_idx + len(pcm))
        padded[start_idx:end_idx] = pcm[:end_idx - start_idx]
        return padded

    self_padded = get_padded(self_pcm, self_start_s, total_duration, output_sr)
    world_padded = get_padded(world_pcm, world_start_s, total_duration, noise_sr)

    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    stream = sd.InputStream(device=input_device, channels=1,
                            samplerate=input_sr, dtype="float32", blocksize=1024)
    stream.start()
    import threading
    self_thread = threading.Thread(
        target=play_array_blocking, args=(self_padded, output_device, output_sr))
    world_thread = threading.Thread(
        target=play_array_blocking, args=(world_padded, noise_device, noise_sr))
    self_thread.start()
    world_thread.start()

    total_samples = int(total_duration * input_sr)
    captured = np.zeros(total_samples, dtype=np.float32)
    pos = 0
    while pos < total_samples:
        data, _ = stream.read(min(1024, total_samples - pos))
        n = min(len(data), total_samples - pos)
        captured[pos:pos + n] = data[:n, 0]
        pos += n

    self_thread.join()
    world_thread.join()
    stream.stop()
    stream.close()

    if input_sr != SR:
        import librosa
        observed = librosa.resample(captured, orig_sr=input_sr, target_sr=SR)
    else:
        observed = captured

    return observed, self_start_s, world_start_s, total_duration


def run_calibration(args):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"calibrate_self_start_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.world_speech.endswith(".npy"):
        world_pcm = np.load(args.world_speech)
    else:
        print(f"ERROR: world speech file: {args.world_speech}")
        sys.exit(1)

    trials = args.num_trials
    results = []

    print("=" * 70)
    print("M2 BENCHMARK — Self-Start Calibration")
    print(f"  Self stim:     {args.self_stim}")
    print(f"  Self duration: {args.self_duration_ms}ms")
    print(f"  Trials:        {trials}")
    print(f"  World speech:  {args.world_speech}")
    print("=" * 70)

    for t_idx in range(trials):
        print(f"\n--- Trial {t_idx + 1}/{trials} ---")

        meta = generate_stimulus_with_metadata(
            args.self_stim, args.self_duration_ms,
            args.self_amplitude, seed=t_idx)
        self_pcm = meta["pcm"]

        try:
            observed, self_start_s, world_start_s, total_dur = run_capture(
                self_pcm, world_pcm, args.input_device,
                args.output_device, args.noise_device)
        except Exception as e:
            print(f"  CAPTURE ERROR: {e}")
            results.append({"trial": t_idx, "error": str(e)})
            continue

        theoretical_shift_samples = int(self_start_s * SR)
        xcorr_score, xcorr_shift = normalized_xcorr(observed, self_pcm)
        lxcorr_score, lxcorr_shift = local_xcorr(
            observed, self_pcm, xcorr_shift, search_ms=100.0)
        energy_shift = energy_onset(
            observed, theoretical_shift_samples,
            search_before_ms=50.0, search_after_ms=500.0)
        theoretical_expected_shift = int(
            (self_start_s + args.onset_latency_ms / 1000.0) * SR)

        offset_samples = lxcorr_shift - theoretical_shift_samples
        offset_ms = offset_samples * 1000.0 / SR
        suggested_latency_ms = (lxcorr_shift / SR - self_start_s) * 1000.0

        trial_result = {
            "trial": t_idx,
            "self_start_s": self_start_s,
            "theoretical_shift_samples": int(theoretical_shift_samples),
            "lxcorr_shift_samples": int(lxcorr_shift),
            "energy_shift_samples": int(energy_shift),
            "xcorr_score": float(xcorr_score),
            "lxcorr_score": float(lxcorr_score),
            "offset_samples": int(offset_samples),
            "offset_ms": float(offset_ms),
            "suggested_onset_latency_ms": float(suggested_latency_ms),
            "current_onset_latency_ms": float(args.onset_latency_ms),
        }
        results.append(trial_result)

        print(f"  Self start (theoretical): {self_start_s * 1000:.1f} ms")
        print(f"  Xcorr shift:    {lxcorr_shift} samples ({lxcorr_shift/SR*1000:.1f} ms)")
        print(f"  Energy shift:   {energy_shift} samples ({energy_shift/SR*1000:.1f} ms)")
        print(f"  Offset:         {offset_samples} samples ({offset_ms:.2f} ms)")
        print(f"  Xcorr score:    {lxcorr_score:.4f}")
        print(f"  Suggested onset_latency_ms: {suggested_latency_ms:.2f} "
              f"(current: {args.onset_latency_ms:.2f})")

        trial_dir = save_dir / f"trial_{t_idx}"
        trial_dir.mkdir(exist_ok=True)
        np.save(trial_dir / "observed.npy", observed)
        np.save(trial_dir / "self_pcm.npy", self_pcm)
        with open(trial_dir / "result.json", "w") as f:
            json.dump(trial_result, f, indent=2)

    valid = [r for r in results if "error" not in r]
    if valid:
        suggested = [r["suggested_onset_latency_ms"] for r in valid]
        offsets = [r["offset_ms"] for r in valid]
        scores = [r["lxcorr_score"] for r in valid]
        agg = {
            "num_trials": len(valid),
            "mean_suggested_onset_latency_ms": float(np.mean(suggested)),
            "std_suggested_onset_latency_ms": float(np.std(suggested)),
            "min_suggested_onset_latency_ms": float(np.min(suggested)),
            "max_suggested_onset_latency_ms": float(np.max(suggested)),
            "mean_offset_ms": float(np.mean(offsets)),
            "mean_lxcorr_score": float(np.mean(scores)),
        }
    else:
        agg = {"error": "no valid trials"}

    summary = {
        "config": {
            "self_stim": args.self_stim,
            "self_duration_ms": args.self_duration_ms,
            "self_amplitude": args.self_amplitude,
            "world_speech": args.world_speech,
            "current_onset_latency_ms": args.onset_latency_ms,
            "input_device": args.input_device,
            "output_device": args.output_device,
            "noise_device": args.noise_device,
        },
        "aggregate": agg,
        "results": results,
    }

    summary_path = save_dir / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    if valid:
        print(f"  Trials (valid):        {len(valid)} / {trials}")
        print(f"  Mean lxcorr score:     {agg['mean_lxcorr_score']:.4f}")
        print(f"  Mean suggested latency: {agg['mean_suggested_onset_latency_ms']:.2f} ms")
        print(f"  Std suggested latency:  {agg['std_suggested_onset_latency_ms']:.2f} ms")
        print(f"  Current latency:        {args.onset_latency_ms:.2f} ms")
        print(f"  Mean offset:            {agg['mean_offset_ms']:.2f} ms")
    else:
        print("  NO VALID TRIALS")
    print(f"\nArtifacts saved to: {save_dir}")
    print(f"Summary: {summary_path}")
    print("=" * 70)

    if valid:
        print("\nTo apply calibration to M2 benchmark:")
        print(f"  --onset-latency-ms {agg['mean_suggested_onset_latency_ms']:.0f}")
        print("\nOr add to your benchmark config:")
        avg_lat = agg["mean_suggested_onset_latency_ms"]
        print(f'  "onset_latency_ms": {avg_lat:.1f}')

    return str(save_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate self-audio onset detection for M2 benchmark")
    parser.add_argument(
        "--self-stim", default="vowel_ah_var_nearfield",
        help="Self stimulus type")
    parser.add_argument(
        "--self-amplitude", type=float, default=0.15,
        help="Self stimulus amplitude")
    parser.add_argument(
        "--self-duration-ms", type=int, default=350,
        help="Self stimulus duration in ms")
    parser.add_argument(
        "--world-speech", default="data/world_speech_sample.npy",
        help="World speech npy file")
    parser.add_argument(
        "--input-device", type=int, default=3,
        help="Input (mic) device index")
    parser.add_argument(
        "--output-device", type=int, default=5,
        help="Output (speaker) device index")
    parser.add_argument(
        "--noise-device", type=int, default=4,
        help="Noise (headphones) device index")
    parser.add_argument(
        "--onset-latency-ms", type=float, default=400.0,
        help="Current onset latency used in benchmark (for reporting)")
    parser.add_argument(
        "--num-trials", type=int, default=3,
        help="Number of capture trials to average over")
    parser.add_argument(
        "--save-dir", default="data/calibration",
        help="Directory to save calibration results")
    args = parser.parse_args()
    run_calibration(args)


if __name__ == "__main__":
    import threading
    main()


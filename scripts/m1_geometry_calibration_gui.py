#!/usr/bin/env python3
"""Live geometry calibration monitor for Corollary M1 hardware setup.

Purpose:
- loop a known self stimulus
- capture the microphone continuously
- show live waveform, residual, and a few geometry-sensitive metrics
- help find a stable mic/speaker placement zone, not a razor-thin one-off optimum
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is not installed.")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cochlea.capture import SR

sys.path.insert(0, os.path.dirname(__file__))
from m1_tone_attribution import generate_stimulus_with_metadata


class SharedCapture:
    def __init__(self, max_samples: int) -> None:
        self.buffer = np.zeros(max_samples, dtype=np.float32)
        self.max_samples = max_samples
        self.write_pos = 0
        self.filled = 0
        self.lock = threading.Lock()

    def append(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        n = len(chunk)
        if n == 0:
            return
        with self.lock:
            if n >= self.max_samples:
                self.buffer[:] = chunk[-self.max_samples :]
                self.write_pos = 0
                self.filled = self.max_samples
                return
            end = self.write_pos + n
            if end <= self.max_samples:
                self.buffer[self.write_pos:end] = chunk
            else:
                first = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = chunk[:first]
                self.buffer[: n - first] = chunk[first:]
            self.write_pos = (self.write_pos + n) % self.max_samples
            self.filled = min(self.max_samples, self.filled + n)

    def latest(self, n: int) -> np.ndarray:
        with self.lock:
            n = min(n, self.filled)
            if n <= 0:
                return np.zeros(0, dtype=np.float32)
            start = (self.write_pos - n) % self.max_samples
            if start < self.write_pos:
                return self.buffer[start:self.write_pos].copy()
            return np.concatenate((self.buffer[start:], self.buffer[:self.write_pos])).copy()


def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-10))


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        factor = max(int(round(orig_sr / target_sr)), 1)
        return audio[::factor].astype(np.float32)


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
        window = observed[shift : shift + len(template)]
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


def compute_metrics(observed: np.ndarray, template: np.ndarray) -> dict[str, Any]:
    if len(observed) < len(template):
        return {
            "match_score": 0.0,
            "capture_rms_db": -120.0,
            "residual_rms_db": -120.0,
            "residual_ratio_db": 0.0,
            "onset_error_ms": 0.0,
            "stability_score": 0.0,
            "shift": 0,
            "aligned": np.zeros_like(template),
            "residual": np.zeros_like(template),
        }

    score, shift = normalized_xcorr(observed, template)
    aligned = observed[shift : shift + len(template)]
    residual = aligned - template[: len(aligned)]

    capture_rms = float(np.sqrt(np.mean(aligned ** 2))) if len(aligned) else 0.0
    residual_rms = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else 0.0

    observed_onset = estimate_envelope_onset(aligned)
    template_onset = estimate_envelope_onset(template)
    onset_error_ms = abs(observed_onset - template_onset) * 1000.0 / SR

    residual_ratio_db = db(residual_rms / max(capture_rms, 1e-10))
    stability_score = float(np.clip(score * 0.5 + max(0.0, 1.0 - onset_error_ms / 120.0) * 0.3 + max(0.0, 1.0 + residual_ratio_db / 24.0) * 0.2, 0.0, 1.0))

    return {
        "match_score": score,
        "capture_rms_db": db(capture_rms),
        "residual_rms_db": db(residual_rms),
        "residual_ratio_db": residual_ratio_db,
        "onset_error_ms": onset_error_ms,
        "stability_score": stability_score,
        "shift": shift,
        "aligned": aligned,
        "residual": residual,
    }


def playback_loop(pcm: np.ndarray, output_device: int | None, stop_event: threading.Event, gap_s: float) -> None:
    while not stop_event.is_set():
        sd.play(pcm, samplerate=SR, device=output_device, blocking=True)
        if stop_event.wait(gap_s):
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 geometry calibration GUI")
    parser.add_argument("--stimulus", default="babble_ba_var")
    parser.add_argument("--duration-ms", type=int, default=350)
    parser.add_argument("--amplitude", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-device", type=int, default=2)
    parser.add_argument("--output-device", type=int, default=4)
    parser.add_argument("--loop-gap-s", type=float, default=0.35)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    args = parser.parse_args()

    stimulus_meta = generate_stimulus_with_metadata(args.stimulus, args.duration_ms, args.amplitude, seed=args.seed)
    pcm = stimulus_meta["pcm"].astype(np.float32)

    input_sr = int(sd.query_devices(args.input_device)["default_samplerate"] or SR)
    max_samples = int(max(SR * args.window_seconds * 2.5, len(pcm) * 4))
    shared = SharedCapture(max_samples=max_samples)
    stop_event = threading.Event()

    def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, 0]
        if input_sr != SR:
            mono = resample_audio(mono, input_sr, SR)
        shared.append(mono)

    playback_thread = threading.Thread(
        target=playback_loop,
        args=(pcm, args.output_device, stop_event, args.loop_gap_s),
        daemon=True,
    )

    stream = sd.InputStream(
        device=args.input_device,
        channels=1,
        samplerate=input_sr,
        dtype="float32",
        callback=audio_callback,
        blocksize=1024,
    )

    window_samples = int(SR * args.window_seconds)
    history_len = 60
    history = {
        "match": deque(maxlen=history_len),
        "onset": deque(maxlen=history_len),
        "residual": deque(maxlen=history_len),
        "stability": deque(maxlen=history_len),
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle("Corollary M1 Geometry Calibration Monitor")
    wave_ax, residual_ax, trend_ax = axes

    try:
        stream.start()
        playback_thread.start()

        while plt.fignum_exists(fig.number):
            observed = shared.latest(max(window_samples, len(pcm) * 2))
            metrics = compute_metrics(observed, pcm)

            history["match"].append(metrics["match_score"])
            history["onset"].append(metrics["onset_error_ms"])
            history["residual"].append(metrics["residual_ratio_db"])
            history["stability"].append(metrics["stability_score"])

            aligned = metrics["aligned"]
            residual = metrics["residual"]
            x = np.arange(len(aligned)) / SR if len(aligned) else np.arange(len(pcm)) / SR

            wave_ax.clear()
            wave_ax.plot(x, pcm[: len(x)], label="template", alpha=0.8)
            if len(aligned):
                wave_ax.plot(x, aligned[: len(x)], label="observed", alpha=0.7)
            wave_ax.set_ylabel("amp")
            wave_ax.set_title("Template vs observed aligned window")
            wave_ax.legend(loc="upper right")

            residual_ax.clear()
            if len(residual):
                residual_ax.plot(x, residual[: len(x)], color="tab:red")
            residual_ax.axhline(0.0, color="black", linewidth=0.8)
            residual_ax.set_ylabel("amp")
            residual_ax.set_title("Residual (observed - template)")

            trend_ax.clear()
            steps = np.arange(len(history["match"]))
            trend_ax.plot(steps, list(history["match"]), label="match")
            trend_ax.plot(steps, list(history["stability"]), label="stability")
            trend_ax.plot(steps, [max(0.0, 1.0 - v / 120.0) for v in history["onset"]], label="onset quality")
            trend_ax.set_ylim(-0.05, 1.05)
            trend_ax.set_title("Rolling quality traces")
            trend_ax.legend(loc="upper right")
            trend_ax.set_xlabel("update")

            fig.texts = [fig.texts[0]] if fig.texts else []
            metrics_text = (
                f"stim={args.stimulus} seed={args.seed}   "
                f"match={metrics['match_score']:.3f}   "
                f"capture={metrics['capture_rms_db']:.1f} dB   "
                f"residual={metrics['residual_rms_db']:.1f} dB   "
                f"residual_ratio={metrics['residual_ratio_db']:.1f} dB   "
                f"onset_err={metrics['onset_error_ms']:.1f} ms   "
                f"stability={metrics['stability_score']:.2f}"
            )
            fig.text(0.02, 0.02, metrics_text, fontsize=10)

            plt.tight_layout(rect=(0, 0.05, 1, 0.95))
            plt.pause(0.08)
    finally:
        stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

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


def parse_device_pair(raw: str) -> tuple[int, int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two device IDs, comma-separated (e.g. 2,5)")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Device IDs must be integers") from exc


def parse_label_pair(raw: str) -> tuple[str, str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two labels, comma-separated (e.g. self,world)")
    return parts[0], parts[1]


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
    # residual_ratio_db is 20*log10(residual/capture) and is always negative.
    # More negative = better match = higher stability. Map [-40, 0] dB → [1.0, 0.0].
    residual_term = float(np.clip(abs(residual_ratio_db) / 40.0, 0.0, 1.0))
    stability_score = float(np.clip(score * 0.5 + max(0.0, 1.0 - onset_error_ms / 120.0) * 0.3 + residual_term * 0.2, 0.0, 1.0))

    return {
        "match_score": score,
        "capture_rms_db": db(capture_rms),
        "residual_rms_db": db(residual_rms),
        "residual_ratio_db": residual_ratio_db,
        "onset_error_ms": onset_error_ms,
        "onset_error_samples": abs(observed_onset - template_onset),
        "stability_score": stability_score,
        "shift": shift,
        "aligned": aligned,
        "residual": residual,
    }


def geometry_fingerprint(metrics_by_input: list[dict[str, Any]]) -> np.ndarray:
    """Build a stimulus-independent two-mic transfer fingerprint.

    The normal match margin compares one captured channel against two stimulus
    templates. That is intentionally blind when both outputs play the same
    waveform. This fingerprint instead asks how one active output is heard by
    the two input paths: relative level, relative onset, and relative match
    strength. Those ratios/deltas are the useful geometry cues.
    """
    if len(metrics_by_input) != 2:
        return np.zeros(3, dtype=np.float32)

    a, b = metrics_by_input
    rms_delta_db = float(a["capture_rms_db"] - b["capture_rms_db"])
    onset_delta_ms = float(a["shift"] - b["shift"]) * 1000.0 / SR
    match_delta = float(a["match_score"] - b["match_score"])

    # Normalize to roughly comparable ranges while preserving sign.
    return np.array(
        [
            np.clip(rms_delta_db / 24.0, -1.0, 1.0),
            np.clip(onset_delta_ms / 40.0, -1.0, 1.0),
            np.clip(match_delta, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def fingerprint_separation(history_a: deque[np.ndarray], history_b: deque[np.ndarray]) -> tuple[float, np.ndarray, np.ndarray]:
    """Return centroid distance between two output transfer fingerprints."""
    if not history_a or not history_b:
        return 0.0, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    centroid_a = np.mean(np.stack(list(history_a)), axis=0)
    centroid_b = np.mean(np.stack(list(history_b)), axis=0)
    distance = float(np.linalg.norm(centroid_a - centroid_b))
    # Max possible distance is sqrt(12) because each of 3 dims spans [-1, 1].
    score = float(np.clip(distance / np.sqrt(12.0), 0.0, 1.0))
    return score, centroid_a.astype(np.float32), centroid_b.astype(np.float32)


class LoopingOutput:
    def __init__(self, pcm: np.ndarray, gap_s: float) -> None:
        self.pcm = pcm.astype(np.float32)
        self.gap_samples = max(0, int(gap_s * SR))
        self.pos = 0
        self.in_gap = False
        self.gap_pos = 0
        self.enabled = True
        self.lock = threading.Lock()

    def set_enabled(self, enabled: bool) -> None:
        with self.lock:
            self.enabled = enabled
            if not enabled:
                self.pos = 0
                self.in_gap = False
                self.gap_pos = 0

    def callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(status, file=sys.stderr)
        out = np.zeros(frames, dtype=np.float32)
        with self.lock:
            if not self.enabled:
                outdata[:, 0] = out
                return
            i = 0
            while i < frames:
                if self.in_gap:
                    remaining_gap = self.gap_samples - self.gap_pos
                    n = min(frames - i, remaining_gap)
                    i += n
                    self.gap_pos += n
                    if self.gap_pos >= self.gap_samples:
                        self.in_gap = False
                        self.gap_pos = 0
                        self.pos = 0
                else:
                    remaining_pcm = len(self.pcm) - self.pos
                    n = min(frames - i, remaining_pcm)
                    out[i:i + n] = self.pcm[self.pos:self.pos + n]
                    i += n
                    self.pos += n
                    if self.pos >= len(self.pcm):
                        if self.gap_samples > 0:
                            self.in_gap = True
                            self.gap_pos = 0
                        else:
                            self.pos = 0
        outdata[:, 0] = out

# ============================================================
#   COROLLARY — M1 Microphone Capture Verification
# ============================================================

# Available audio INPUT devices:

# Index    Name                                               Ch   SR         Default
# --------------------------------------------------------------------------------
# 1        BL-2021USB                                         1    48000     
# 3        fifine Microphone                                  2    48000      ← DEFAULT


# Available audio OUTPUT devices:

# Index    Name                                               Ch   SR         Default
# --------------------------------------------------------------------------------
# 0        ASUS PB287Q                                        2    48000     
# 2        fifine Microphone                                  2    48000     
# 4        External Headphones                                2    48000      ← DEFAULT
# 5        Mac mini Speakers                                  2    48000     


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 geometry calibration GUI")
    parser.add_argument("--stimulus", default="babble_ba_var")
    parser.add_argument("--stimulus-a", default=None, help="Stimulus for output-device-a (defaults to --stimulus)")
    parser.add_argument("--stimulus-b", default=None, help="Stimulus for output-device-b (defaults to --stimulus)")
    parser.add_argument("--duration-ms", type=int, default=350)
    parser.add_argument("--amplitude", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-device", type=int, default=2)
    parser.add_argument("--output-device", type=int, default=3)
    parser.add_argument("--input-devices", type=parse_device_pair, default=None, help="Two input devices, comma-separated (e.g. 2,5)")
    parser.add_argument("--output-devices", type=parse_device_pair, default=None, help="Two output devices, comma-separated (e.g. 3,6)")
    parser.add_argument("--loop-gap-s", type=float, default=0.35)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--labels", type=parse_label_pair, default=("self", "world"), help="Semantic labels for the two paths, comma-separated")
    parser.add_argument("--mode", choices=["simultaneous", "alternate"], default="simultaneous", help="Simultaneous loops or one-source-at-a-time alternation")
    parser.add_argument("--alternate-period-s", type=float, default=2.0, help="How long each source stays active in alternate mode")
    args = parser.parse_args()

    input_devices = args.input_devices if args.input_devices is not None else (args.input_device, args.input_device)
    output_devices = args.output_devices if args.output_devices is not None else (args.output_device, args.output_device)
    stimuli = (args.stimulus_a or args.stimulus, args.stimulus_b or args.stimulus)
    labels = args.labels

    stream_configs: list[dict[str, Any]] = []
    for idx in range(2):
        stimulus_meta = generate_stimulus_with_metadata(stimuli[idx], args.duration_ms, args.amplitude, seed=args.seed + idx)
        pcm = stimulus_meta["pcm"].astype(np.float32)
        input_sr = int(sd.query_devices(input_devices[idx])["default_samplerate"] or SR)
        max_samples = int(max(SR * args.window_seconds * 2.5, len(pcm) * 4))
        shared = SharedCapture(max_samples=max_samples)
        looper = LoopingOutput(pcm, args.loop_gap_s)

        def make_audio_callback(target_shared: SharedCapture, target_input_sr: int):
            def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
                if status:
                    print(status, file=sys.stderr)
                mono = indata[:, 0]
                if target_input_sr != SR:
                    mono = resample_audio(mono, target_input_sr, SR)
                target_shared.append(mono)

            return audio_callback

        input_stream = sd.InputStream(
            device=input_devices[idx],
            channels=1,
            samplerate=input_sr,
            dtype="float32",
            callback=make_audio_callback(shared, input_sr),
            blocksize=1024,
        )
        output_stream = sd.OutputStream(
            device=output_devices[idx],
            channels=1,
            samplerate=SR,
            dtype="float32",
            callback=looper.callback,
            blocksize=1024,
        )
        stream_configs.append(
            {
                "label": labels[idx],
                "input_device_id": input_devices[idx],
                "output_device_id": output_devices[idx],
                "stimulus": stimuli[idx],
                "pcm": pcm,
                "shared": shared,
                "input_stream": input_stream,
                "output_stream": output_stream,
                "looper": looper,
            }
        )

    window_samples = int(SR * args.window_seconds)
    history_len = 120
    permutation_history: dict[tuple[int, int], deque[float]] = {
        (in_idx, out_idx): deque(maxlen=history_len)
        for in_idx in range(2)
        for out_idx in range(2)
    }

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.1, 1.1, 0.9, 0.9])
    fig.suptitle("Corollary M1 Geometry Calibration Monitor")
    wave_axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    residual_axes = [fig.add_subplot(gs[1, i]) for i in range(2)]
    matrix_ax = fig.add_subplot(gs[0:2, 2])
    margin_axes = [fig.add_subplot(gs[2, i]) for i in range(2)]
    summary_ax = fig.add_subplot(gs[3, :])

    # Rolling template-discrimination histories are source-active histories.
    # In alternate mode, only update a source's history while that source is
    # actively playing; otherwise the inactive source's instantaneous row is
    # just a silence/crosstalk snapshot and should not affect the verdict.
    margin_history: list[deque[float]] = [deque(maxlen=history_len), deque(maxlen=history_len)]
    own_match_history: list[deque[float]] = [deque(maxlen=history_len), deque(maxlen=history_len)]
    cross_match_history: list[deque[float]] = [deque(maxlen=history_len), deque(maxlen=history_len)]
    active_global_history: deque[float] = deque(maxlen=history_len * 2)
    geometry_history: list[deque[np.ndarray]] = [deque(maxlen=history_len), deque(maxlen=history_len)]
    geometry_score_history: deque[float] = deque(maxlen=history_len)
    same_stimulus_mode = stimuli[0] == stimuli[1]

    try:
        for cfg in stream_configs:
            cfg["input_stream"].start()
            cfg["output_stream"].start()

        while plt.fignum_exists(fig.number):
            if args.mode == "alternate":
                phase = int((time.monotonic() / max(args.alternate_period_s, 0.1))) % 2
                for idx, cfg in enumerate(stream_configs):
                    cfg["looper"].set_enabled(idx == phase)
                active_label = stream_configs[phase]["label"]
            else:
                for cfg in stream_configs:
                    cfg["looper"].set_enabled(True)
                active_label = "both"
            pair_metrics: dict[tuple[int, int], dict[str, Any]] = {}
            latest_observed: list[np.ndarray] = []

            for idx, cfg in enumerate(stream_configs):
                input_observed = cfg["shared"].latest(max(window_samples, len(cfg["pcm"]) * 2))
                latest_observed.append(input_observed)
                for out_idx, out_cfg in enumerate(stream_configs):
                    perm_metrics = compute_metrics(input_observed, out_cfg["pcm"])
                    pair_metrics[(idx, out_idx)] = perm_metrics
                    permutation_history[(idx, out_idx)].append(perm_metrics["stability_score"])

            if args.mode == "alternate":
                active_metrics = [pair_metrics[(in_idx, phase)] for in_idx in range(2)]
                geometry_history[phase].append(geometry_fingerprint(active_metrics))
            geometry_score, geometry_a, geometry_b = fingerprint_separation(geometry_history[0], geometry_history[1])
            geometry_score_history.append(geometry_score)

            matrix_ax.clear()
            score_matrix = np.array(
                [[pair_metrics[(in_idx, out_idx)]["match_score"] for out_idx in range(2)] for in_idx in range(2)],
                dtype=np.float32,
            )
            im = matrix_ax.imshow(score_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            matrix_ax.set_xticks([0, 1], labels=[f"out:{cfg['label']}" for cfg in stream_configs])
            matrix_ax.set_yticks([0, 1], labels=[f"in:{cfg['label']}" for cfg in stream_configs])
            matrix_ax.set_title("Cross-path match matrix" + (" (templates identical in same-stimulus mode)" if same_stimulus_mode else ""))
            for in_idx in range(2):
                for out_idx in range(2):
                    pm = pair_metrics[(in_idx, out_idx)]
                    matrix_ax.text(
                        out_idx,
                        in_idx,
                        f"m={pm['match_score']:.2f}\nst={pm['stability_score']:.2f}",
                        ha="center",
                        va="center",
                        color="white" if pm["match_score"] < 0.55 else "black",
                        fontsize=10,
                    )

            summary_lines: list[str] = []
            for idx, cfg in enumerate(stream_configs):
                intended = pair_metrics[(idx, idx)]
                cross = pair_metrics[(idx, 1 - idx)]
                margin = intended["match_score"] - cross["match_score"]
                rms_margin = intended["capture_rms_db"] - cross["capture_rms_db"]
                observed = latest_observed[idx]
                aligned = intended["aligned"]
                residual = intended["residual"]
                x = np.arange(len(aligned)) / SR if len(aligned) else np.arange(len(cfg["pcm"])) / SR

                wave_ax = wave_axes[idx]
                residual_ax = residual_axes[idx]

                wave_ax.clear()
                wave_ax.plot(x, cfg["pcm"][: len(x)], label=f"template:{cfg['label']}", alpha=0.8)
                if len(aligned):
                    wave_ax.plot(x, aligned[: len(x)], label="captured", alpha=0.7)
                wave_ax.set_ylabel("amp")
                wave_ax.set_title(
                    f"{cfg['label']} path  in:{cfg['input_device_id']}  out:{cfg['output_device_id']}\n"
                    f"match={intended['match_score']:.3f}  cross={cross['match_score']:.3f}  margin={margin:+.3f}"
                )
                wave_ax.legend(loc="upper right")

                residual_ax.clear()
                if len(residual):
                    residual_ax.plot(x, residual[: len(x)], color="tab:red", label="residual")
                residual_ax.axhline(0.0, color="black", linewidth=0.8)
                residual_ax.set_ylabel("amp")
                residual_ax.set_title(
                    f"{cfg['label']} residual / alignment  onset={intended['onset_error_ms']:.1f}ms  "
                    f"resid={intended['residual_ratio_db']:.1f}dB"
                )
                residual_ax.legend(loc="upper right")

                verdict = "good" if margin >= 0.15 else ("mixed" if margin >= 0.05 else "poor")
                source_is_active = args.mode != "alternate" or idx == phase
                if source_is_active:
                    margin_history[idx].append(margin)
                    own_match_history[idx].append(intended["match_score"])
                    cross_match_history[idx].append(cross["match_score"])
                    active_global_history.append(margin)

                margin_ax = margin_axes[idx]
                margin_ax.clear()
                steps = np.arange(len(margin_history[idx]))
                margin_ax.plot(steps, list(margin_history[idx]), label="match margin", color="tab:green", linewidth=2)
                margin_ax.plot(steps, list(own_match_history[idx]), label="own match", color="tab:blue", alpha=0.7)
                margin_ax.plot(steps, list(cross_match_history[idx]), label="cross match", color="tab:orange", alpha=0.7)
                margin_ax.axhline(0.0, color="black", linewidth=0.8)
                margin_ax.axhline(0.15, color="tab:green", linestyle="--", linewidth=0.8)
                margin_ax.axhline(0.05, color="tab:gray", linestyle=":", linewidth=0.8)
                margin_ax.set_ylim(-1.0, 1.0)
                margin_ax.set_title(f"{cfg['label']} rolling separation")
                margin_ax.set_xlabel("update")
                margin_ax.legend(loc="upper right", fontsize=8)

                trend_mean = float(np.mean(margin_history[idx])) if margin_history[idx] else 0.0
                trend_min = float(np.min(margin_history[idx])) if margin_history[idx] else 0.0
                if source_is_active:
                    summary_lines.append(
                        f"{cfg['label']}: ACTIVE own={intended['match_score']:.3f} cross={cross['match_score']:.3f} "
                        f"margin={margin:+.3f} mean_margin={trend_mean:+.3f} min_margin={trend_min:+.3f} "
                        f"own_rms={intended['capture_rms_db']:.1f}dB cross_rms={cross['capture_rms_db']:.1f}dB "
                        f"rms_margin={rms_margin:+.1f}dB verdict={verdict}"
                    )
                else:
                    summary_lines.append(
                        f"{cfg['label']}: inactive snapshot own={intended['match_score']:.3f} cross={cross['match_score']:.3f} "
                        f"margin={margin:+.3f} own_rms={intended['capture_rms_db']:.1f}dB "
                        f"cross_rms={cross['capture_rms_db']:.1f}dB (ignored for rolling verdict)"
                    )

            separation_a = pair_metrics[(0, 0)]["match_score"] - pair_metrics[(0, 1)]["match_score"]
            separation_b = pair_metrics[(1, 1)]["match_score"] - pair_metrics[(1, 0)]["match_score"]
            instantaneous_global_score = (separation_a + separation_b) / 2.0
            global_score = (
                float(np.mean(active_global_history))
                if args.mode == "alternate" and active_global_history
                else instantaneous_global_score
            )
            geometry_verdict = "good" if geometry_score >= 0.25 else ("mixed" if geometry_score >= 0.12 else "poor")
            geometry_summary = (
                f"Same-stimulus geometry score: {geometry_score:.3f} verdict={geometry_verdict} "
                f"{stream_configs[0]['label']}=[rmsΔ={geometry_a[0]:+.2f}, onsetΔ={geometry_a[1]:+.2f}, matchΔ={geometry_a[2]:+.2f}] "
                f"{stream_configs[1]['label']}=[rmsΔ={geometry_b[0]:+.2f}, onsetΔ={geometry_b[1]:+.2f}, matchΔ={geometry_b[2]:+.2f}]"
            )
            summary_ax.clear()
            summary_ax.axis("off")
            summary_ax.text(
                0.01,
                0.95,
                "\n".join([
                    f"Active source mode: {args.mode}  active={active_label}",
                    f"Global template separation score: {global_score:+.3f}  "
                    f"({'active-source rolling mean' if args.mode == 'alternate' else 'instantaneous mean'}; > +0.15 is a good early sign)",
                    geometry_summary,
                    *summary_lines,
                    "Interpret ACTIVE template margins as stimulus discrimination, not pure source geometry.",
                    "In alternate mode, inactive snapshots are shown for crosstalk context but ignored for rolling verdicts.",
                    "In same-stimulus + alternate mode, use the geometry score / fingerprints as the real gate.",
                    "Geometry score compares two-mic transfer fingerprints: relative RMS, onset shift, and match balance.",
                    "For dual-mic setup, prefer stable positive template margins for routing sanity plus stable geometry separation for M1c.",
                ]),
                va="top",
                fontsize=11,
                family="monospace",
            )

            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            plt.pause(0.15)
    finally:
        for cfg in stream_configs:
            try:
                cfg["output_stream"].stop()
                cfg["output_stream"].close()
            except Exception:
                pass
            try:
                cfg["input_stream"].stop()
                cfg["input_stream"].close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

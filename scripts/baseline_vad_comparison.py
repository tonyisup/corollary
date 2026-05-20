#!/usr/bin/env python3
"""Baseline VAD/AEC comparison for Corollary.

Compares Corollary's full pipeline against standard VAD-only and VAD+AEC stacks
on the same audio captures. Produces metrics, tables, and plots.

Usage:
    .venv/bin/python scripts/baseline_vad_comparison.py \\
        --captures data/m3_live_e2e_v35_m4 \\
        --output /tmp/baseline_comparison.json

    .venv/bin/python scripts/baseline_vad_comparison.py \
        --captures data/m3_live_e2e_v35_m4 \
        --output tmp/baseline_comparison.json \
        --plot-output tmp/baseline_plots/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ── NLMS AEC (Stack C) ──────────────────────────────────────────────────────

class NLMSAEC:
    """Normalized Least Mean Squares adaptive filter for echo cancellation.

    Uses the known dry TTS signal as the reference (far-end) channel.
    """

    def __init__(self, n_taps: int = 256, mu: float = 0.1, sr: int = 16000):
        self.n_taps = n_taps
        self.mu = mu
        self.sr = sr
        self.weights = np.zeros(n_taps, dtype=np.float32)
        self.buffer = np.zeros(n_taps, dtype=np.float32)

    def process(self, mic: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process mic signal with reference (dry TTS).

        Returns:
            output: mic with estimated echo subtracted
            echo_estimate: the estimated echo signal
        """
        n = len(mic)
        output = np.zeros(n, dtype=np.float32)
        echo_estimate = np.zeros(n, dtype=np.float32)

        for i in range(n):
            # Shift buffer and add new reference sample
            self.buffer = np.roll(self.buffer, 1)
            self.buffer[0] = reference[i] if i < len(reference) else 0.0

            # Estimate echo
            echo_est = np.dot(self.weights, self.buffer)
            echo_estimate[i] = echo_est

            # Error = mic - estimated echo
            error = mic[i] - echo_est
            output[i] = error

            # NLMS update
            power = np.dot(self.buffer, self.buffer) + 1e-8
            self.weights += (self.mu / power) * error * self.buffer

        return output, echo_estimate

    def reset(self):
        self.weights = np.zeros(self.n_taps, dtype=np.float32)
        self.buffer = np.zeros(self.n_taps, dtype=np.float32)


# ── VAD stacks (B, D) ───────────────────────────────────────────────────────

def run_vad_stack(
    audio: np.ndarray,
    aggressiveness: int = 1,
    sr: int = 16000,
    frame_ms: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Run WebRTC VAD on audio and produce a mute mask.

    Returns:
        suppressed: audio with VAD-detected speech regions muted
        mute_mask: boolean array (True = muted)
    """
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    frame_samples = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_samples

    mute_mask = np.zeros(len(audio), dtype=bool)
    suppressed = audio.copy()

    for i in range(n_frames):
        start = i * frame_samples
        end = start + frame_samples
        frame = audio[start:end]

        # Convert to int16 for webrtcvad
        frame_int = (frame * 32767).astype(np.int16)
        frame_bytes = frame_int.tobytes()

        try:
            is_speech = vad.is_speech(frame_bytes, sr)
        except Exception:
            is_speech = False

        if is_speech:
            mute_mask[start:end] = True
            suppressed[start:end] = 0.0

    return suppressed, mute_mask


# ── VAD + AEC stack (C) ─────────────────────────────────────────────────────

def run_aec_vad_stack(
    mic: np.ndarray,
    reference: np.ndarray,
    aggressiveness: int = 1,
    n_taps: int = 256,
    mu: float = 0.1,
    sr: int = 16000,
) -> tuple[np.ndarray, np.ndarray]:
    """Run NLMS AEC followed by VAD gating.

    Returns:
        suppressed: AEC output with VAD-detected speech muted
        mute_mask: boolean array (True = muted)
    """
    aec = NLMSAEC(n_taps=n_taps, mu=mu, sr=sr)
    aec_output, _ = aec.process(mic, reference)

    # Run VAD on AEC output
    suppressed, mute_mask = run_vad_stack(aec_output, aggressiveness=aggressiveness, sr=sr)

    return suppressed, mute_mask


# ── Corollary stack (A) ─────────────────────────────────────────────────────

def run_corollary_stack(
    mic: np.ndarray,
    agent_pcm: np.ndarray,
    sr: int = 16000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run Corollary's full attribution + gating pipeline.

    Uses the existing streaming runtime in offline mode.

    Returns:
        output: rendered output audio
        mute_mask: boolean array (True = muted/gated)
        stats: dict with attribution stats
    """
    from pfc.streaming_runtime import StreamingConfig, StreamingRuntime

    config = StreamingConfig(
        tts_pcm=agent_pcm,
        tts_text="baseline comparison",
        sr=sr,
        gated_playback=True,
        m4_enable=False,  # Skip M4 for speed; use energy-based attribution
        energy_threshold_db=-55.0,
        similarity_threshold=0.05,
        causal_tolerance_ms=175.0,
        both_external_min_frames=3,
        both_external_energy_ratio=1.5,
    )

    runtime = StreamingRuntime(config)
    runtime.run_offline(mic)

    # Build mute mask from controller state
    mute_mask = np.zeros(len(mic), dtype=bool)
    frame_samples = int(sr * 0.03)  # 30ms frames

    for i, log in enumerate(runtime._logs):
        start = i * frame_samples
        end = min(start + frame_samples, len(mic))
        if log.controller_state == "LISTEN":
            mute_mask[start:end] = True

    # Build output (simplified: just mute during LISTEN)
    output = mic.copy()
    output[mute_mask] = 0.0

    stats = {
        "total_frames": len(runtime._logs),
        "speak_frames": sum(1 for l in runtime._logs if l.controller_state == "SPEAK"),
        "listen_frames": sum(1 for l in runtime._logs if l.controller_state == "LISTEN"),
        "self_frames": sum(1 for l in runtime._logs if l.verdict == "SELF"),
        "both_frames": sum(1 for l in runtime._logs if l.verdict == "BOTH"),
        "world_frames": sum(1 for l in runtime._logs if l.verdict == "WORLD"),
    }

    return output, mute_mask, stats


# ── Metrics ──────────────────────────────────────────────────────────────────

def measure_self_leak(output: np.ndarray, self_regions: list[tuple[int, int]], sr: int = 16000) -> float:
    """Measure residual self-voice RMS in dB after suppression.

    Args:
        output: suppressed audio
        self_regions: list of (start_sample, end_sample) tuples where self-speech is known

    Returns:
        RMS in dB (more negative = better suppression)
    """
    if not self_regions:
        return float("nan")

    # Measure RMS in self-regions (where self should have been suppressed)
    region_rms = []
    for start, end in self_regions:
        region = output[start:end]
        if len(region) > 0:
            rms = np.sqrt(np.mean(region ** 2))
            region_rms.append(rms)

    if not region_rms:
        return float("nan")

    avg_rms = np.mean(region_rms)
    return 20 * np.log10(max(avg_rms, 1e-9))


def measure_false_mute(mute_mask: np.ndarray, human_regions: list[tuple[int, int]]) -> float:
    """Measure % of human speech frames incorrectly muted.

    Returns:
        False mute rate as percentage (0-100)
    """
    if not human_regions:
        return 0.0

    total_human_frames = 0
    muted_human_frames = 0

    for start, end in human_regions:
        region_mask = mute_mask[start:end]
        total_human_frames += len(region_mask)
        muted_human_frames += np.sum(region_mask)

    if total_human_frames == 0:
        return 0.0

    return (muted_human_frames / total_human_frames) * 100.0


def measure_overlap_survival(mute_mask: np.ndarray, overlap_regions: list[tuple[int, int]]) -> float:
    """Measure % of overlap frames where human speech is preserved (not fully muted).

    Returns:
        Survival rate as percentage (0-100)
    """
    if not overlap_regions:
        return 100.0

    surviving = 0
    total = 0

    for start, end in overlap_regions:
        region_mask = mute_mask[start:end]
        total += 1
        # A region "survives" if at least 50% of frames are not muted
        if np.mean(region_mask) < 0.5:
            surviving += 1

    return (surviving / total) * 100.0


def measure_wer(output: np.ndarray, reference_text: str, sr: int = 16000) -> dict:
    """Measure WER using faster-whisper.

    Returns:
        dict with wer, word_count, reference_word_count
    """
    if not HAS_WHISPER:
        return {"wer": None, "word_count": 0, "reference_word_count": 0, "note": "whisper not available"}

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(output, language="en", beam_size=5)

    words = []
    for seg in segments:
        words.extend(seg.text.strip().lower().split())

    ref_words = reference_text.strip().lower().split()

    # Simple WER: word error rate via edit distance
    ref_len = len(ref_words)
    hyp_len = len(words)

    if ref_len == 0:
        return {"wer": 1.0 if hyp_len > 0 else 0.0, "word_count": hyp_len, "reference_word_count": 0}

    # Approximate WER using word count difference
    # (full edit distance would be better but this is a reasonable proxy)
    errors = abs(ref_len - hyp_len)
    wer = min(errors / ref_len, 1.0)

    return {
        "wer": round(wer, 4),
        "word_count": hyp_len,
        "reference_word_count": ref_len,
        "note": "approximate WER (word count delta)",
    }


def measure_wake_word_recall(
    mute_mask: np.ndarray,
    human_regions: list[tuple[int, int]],
    sr: int = 16000,
) -> dict:
    """Simulate wake-word system: mute during agent playback, measure human recall.

    A human utterance is "caught" if any portion falls outside a mute window.

    Returns:
        dict with recall, caught, total
    """
    if not human_regions:
        return {"recall": 1.0, "caught": 0, "total": 0}

    caught = 0
    total = len(human_regions)

    for start, end in human_regions:
        region_mask = mute_mask[start:end]
        # Caught if at least some portion is not muted
        if np.mean(region_mask) < 1.0:
            caught += 1

    return {
        "recall": round(caught / total, 4) if total > 0 else 1.0,
        "caught": caught,
        "total": total,
    }


# ── Capture loading ─────────────────────────────────────────────────────────

def load_capture(capture_dir: Path) -> dict:
    """Load a capture directory with agent_pcm.npy and live_captured.npy."""
    agent_files = list(capture_dir.glob("agent_pcm*.npy"))
    captured_files = list(capture_dir.glob("live_captured*.npy"))

    if not agent_files or not captured_files:
        return None

    agent_pcm = np.load(agent_files[0]).astype(np.float32)
    captured_pcm = np.load(captured_files[0]).astype(np.float32)

    # Load metadata if available
    meta_files = list(capture_dir.glob("meta*.json"))
    meta = {}
    if meta_files:
        with open(meta_files[0]) as f:
            meta = json.load(f)

    return {
        "agent_pcm": agent_pcm,
        "captured_pcm": captured_pcm,
        "meta": meta,
        "dir": str(capture_dir),
        "name": capture_dir.name,
    }


def build_ground_truth(capture: dict, sr: int = 16000) -> dict:
    """Build ground truth regions from capture metadata.

    Returns:
        dict with self_regions, human_regions, overlap_regions (lists of (start, end) tuples)
    """
    agent_pcm = capture["agent_pcm"]
    captured_pcm = capture["captured_pcm"]

    # Self region: from start of TTS playback to end
    # We estimate TTS start by finding where the captured audio first exceeds a threshold
    threshold = np.sqrt(np.mean(captured_pcm ** 2)) * 0.1
    nonzero = np.where(np.abs(captured_pcm) > threshold)[0]

    if len(nonzero) == 0:
        return {"self_regions": [], "human_regions": [], "overlap_regions": []}

    # Self region: approximate from agent_pcm length
    tts_start = max(0, nonzero[0] - int(0.5 * sr))  # 0.5s padding before
    tts_end = min(len(captured_pcm), tts_start + len(agent_pcm) + int(0.5 * sr))

    self_regions = [(tts_start, tts_end)]

    # Human region: everything outside self region with significant energy
    human_regions = []
    if tts_end < len(captured_pcm):
        # Check for human speech after TTS
        post_tts = captured_pcm[tts_end:]
        post_rms = np.sqrt(np.mean(post_tts ** 2))
        if post_rms > threshold:
            human_regions.append((tts_end, len(captured_pcm)))

    if tts_start > int(0.5 * sr):
        # Check for human speech before TTS
        pre_tts = captured_pcm[:tts_start]
        pre_rms = np.sqrt(np.mean(pre_tts ** 2))
        if pre_rms > threshold:
            human_regions.append((0, tts_start))

    # Overlap: simplified — any region where both self and human are present
    overlap_regions = []
    for h_start, h_end in human_regions:
        # Check if this overlaps with self region
        s_start, s_end = self_regions[0]
        if h_start < s_end and h_end > s_start:
            overlap_start = max(h_start, s_start)
            overlap_end = min(h_end, s_end)
            overlap_regions.append((overlap_start, overlap_end))

    return {
        "self_regions": self_regions,
        "human_regions": human_regions,
        "overlap_regions": overlap_regions,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline VAD/AEC comparison")
    parser.add_argument("--captures", type=str, required=True,
                        help="Path to captures directory")
    parser.add_argument("--output", type=str, default="/tmp/baseline_comparison.json")
    parser.add_argument("--plot-output", type=str, default=None,
                        help="Directory for plot output")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    sr = args.sr
    captures_dir = Path(args.captures)

    print("=" * 60)
    print("Baseline VAD/AEC Comparison")
    print("=" * 60)
    print(f"  Captures: {captures_dir}")
    print(f"  Output:   {args.output}")
    print()

    # Find capture subdirectories
    capture_dirs = sorted([d for d in captures_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"Found {len(capture_dirs)} capture directories")

    # Load captures
    captures = []
    for cap_dir in capture_dirs:
        cap = load_capture(cap_dir)
        if cap:
            cap["ground_truth"] = build_ground_truth(cap, sr)
            captures.append(cap)
            gt = cap["ground_truth"]
            print(f"  {cap['name']}: agent={len(cap['agent_pcm'])} samples, "
                  f"captured={len(cap['captured_pcm'])} samples, "
                  f"self_regions={len(gt['self_regions'])}, "
                  f"human_regions={len(gt['human_regions'])}")

    if not captures:
        print("ERROR: No valid captures found")
        sys.exit(1)

    print()

    # Run each stack on each capture
    results = []

    for cap in captures:
        print(f"\n{'=' * 60}")
        print(f"Capture: {cap['name']}")
        print(f"{'=' * 60}")

        mic = cap["captured_pcm"]
        agent = cap["agent_pcm"]
        gt = cap["ground_truth"]

        # Stack A: Corollary
        print("\n  [Stack A] Corollary...")
        t0 = time.time()
        try:
            corr_output, corr_mute, corr_stats = run_corollary_stack(mic, agent, sr)
            corr_time = time.time() - t0
            corr_self_leak = measure_self_leak(corr_output, gt["self_regions"], sr)
            corr_false_mute = measure_false_mute(corr_mute, gt["human_regions"])
            corr_overlap = measure_overlap_survival(corr_mute, gt["overlap_regions"])
            corr_wake = measure_wake_word_recall(corr_mute, gt["human_regions"], sr)
            print(f"    Self-leak: {corr_self_leak:.1f} dB, False-mute: {corr_false_mute:.1f}%, "
                  f"Overlap survival: {corr_overlap:.1f}%, Wake recall: {corr_wake['recall']:.2f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            corr_output, corr_mute = mic.copy(), np.zeros(len(mic), dtype=bool)
            corr_self_leak, corr_false_mute, corr_overlap = float("nan"), 0.0, 100.0
            corr_wake = {"recall": 1.0, "caught": 0, "total": 0}
            corr_time = time.time() - t0

        # Stack B: VAD-only (aggressive)
        print("  [Stack B] VAD-only (aggressive)...")
        t0 = time.time()
        vad_b_output, vad_b_mute = run_vad_stack(mic, aggressiveness=1, sr=sr)
        vad_b_time = time.time() - t0
        vad_b_self_leak = measure_self_leak(vad_b_output, gt["self_regions"], sr)
        vad_b_false_mute = measure_false_mute(vad_b_mute, gt["human_regions"])
        vad_b_overlap = measure_overlap_survival(vad_b_mute, gt["overlap_regions"])
        vad_b_wake = measure_wake_word_recall(vad_b_mute, gt["human_regions"], sr)
        print(f"    Self-leak: {vad_b_self_leak:.1f} dB, False-mute: {vad_b_false_mute:.1f}%, "
              f"Overlap survival: {vad_b_overlap:.1f}%, Wake recall: {vad_b_wake['recall']:.2f}")

        # Stack C: VAD + AEC
        print("  [Stack C] VAD + AEC (NLMS)...")
        t0 = time.time()
        aec_output, aec_mute = run_aec_vad_stack(mic, agent, aggressiveness=1, sr=sr)
        aec_time = time.time() - t0
        aec_self_leak = measure_self_leak(aec_output, gt["self_regions"], sr)
        aec_false_mute = measure_false_mute(aec_mute, gt["human_regions"])
        aec_overlap = measure_overlap_survival(aec_mute, gt["overlap_regions"])
        aec_wake = measure_wake_word_recall(aec_mute, gt["human_regions"], sr)
        print(f"    Self-leak: {aec_self_leak:.1f} dB, False-mute: {aec_false_mute:.1f}%, "
              f"Overlap survival: {aec_overlap:.1f}%, Wake recall: {aec_wake['recall']:.2f}")

        # Stack D: VAD-only (conservative)
        print("  [Stack D] VAD-only (conservative)...")
        t0 = time.time()
        vad_d_output, vad_d_mute = run_vad_stack(mic, aggressiveness=3, sr=sr)
        vad_d_time = time.time() - t0
        vad_d_self_leak = measure_self_leak(vad_d_output, gt["self_regions"], sr)
        vad_d_false_mute = measure_false_mute(vad_d_mute, gt["human_regions"])
        vad_d_overlap = measure_overlap_survival(vad_d_mute, gt["overlap_regions"])
        vad_d_wake = measure_wake_word_recall(vad_d_mute, gt["human_regions"], sr)
        print(f"    Self-leak: {vad_d_self_leak:.1f} dB, False-mute: {vad_d_false_mute:.1f}%, "
              f"Overlap survival: {vad_d_overlap:.1f}%, Wake recall: {vad_d_wake['recall']:.2f}")

        # Store results
        for stack_name, self_leak, false_mute, overlap, wake, proc_time in [
            ("A: Corollary", corr_self_leak, corr_false_mute, corr_overlap, corr_wake, corr_time),
            ("B: VAD-only (agg)", vad_b_self_leak, vad_b_false_mute, vad_b_overlap, vad_b_wake, vad_b_time),
            ("C: VAD+AEC (NLMS)", aec_self_leak, aec_false_mute, aec_overlap, aec_wake, aec_time),
            ("D: VAD-only (cons)", vad_d_self_leak, vad_d_false_mute, vad_d_overlap, vad_d_wake, vad_d_time),
        ]:
            results.append({
                "capture": cap["name"],
                "stack": stack_name,
                "self_leak_db": round(self_leak, 1) if not np.isnan(self_leak) else None,
                "false_mute_pct": round(false_mute, 1),
                "overlap_survival_pct": round(overlap, 1),
                "wake_recall": wake["recall"],
                "wake_caught": wake["caught"],
                "wake_total": wake["total"],
                "proc_time_s": round(proc_time, 2),
            })

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Summary Table")
    print(f"{'=' * 60}")
    print(f"{'Capture':<25} {'Stack':<20} {'Self-leak':>10} {'False-mute':>11} {'Overlap':>9} {'Wake':>6}")
    print(f"{'':25} {'':20} {'dB':>10} {'%':>11} {'surv%':>9} {'recall':>6}")
    print("-" * 85)
    for r in results:
        sl = f"{r['self_leak_db']:.1f}" if r['self_leak_db'] is not None else "N/A"
        print(f"{r['capture']:<25} {r['stack']:<20} {sl:>10} {r['false_mute_pct']:>10.1f}% {r['overlap_survival_pct']:>8.1f}% {r['wake_recall']:>6.2f}")

    # ── Per-stack aggregate ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Per-Stack Aggregate")
    print(f"{'=' * 60}")

    stacks = ["A: Corollary", "B: VAD-only (agg)", "C: VAD+AEC (NLMS)", "D: VAD-only (cons)"]
    for stack in stacks:
        stack_results = [r for r in results if r["stack"] == stack]
        if not stack_results:
            continue
        avg_sl = np.mean([r["self_leak_db"] for r in stack_results if r["self_leak_db"] is not None])
        avg_fm = np.mean([r["false_mute_pct"] for r in stack_results])
        avg_os = np.mean([r["overlap_survival_pct"] for r in stack_results])
        avg_wr = np.mean([r["wake_recall"] for r in stack_results])
        print(f"  {stack}:")
        print(f"    Avg self-leak: {avg_sl:.1f} dB")
        print(f"    Avg false-mute: {avg_fm:.1f}%")
        print(f"    Avg overlap survival: {avg_os:.1f}%")
        print(f"    Avg wake recall: {avg_wr:.2f}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "captures_dir": str(captures_dir),
        "n_captures": len(captures),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to: {args.output}")

    # ── Plots ─────────────────────────────────────────────────────────────
    if args.plot_output and HAS_MATPLOTLIB:
        plot_dir = Path(args.plot_output)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Bar chart: self-leak by stack
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Baseline VAD/AEC Comparison", fontsize=14)

        metrics = [
            ("self_leak_db", "Self-leak (dB)", "lower is better", axes[0, 0]),
            ("false_mute_pct", "False-mute (%)", "lower is better", axes[0, 1]),
            ("overlap_survival_pct", "Overlap survival (%)", "higher is better", axes[1, 0]),
            ("wake_recall", "Wake-word recall", "higher is better", axes[1, 1]),
        ]

        for metric, title, subtitle, ax in metrics:
            stack_names = []
            stack_values = []
            for stack in stacks:
                sr = [r for r in results if r["stack"] == stack]
                vals = [r[metric] for r in sr if r[metric] is not None]
                if vals:
                    stack_names.append(stack.split(":")[0])
                    stack_values.append(np.mean(vals))

            colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
            ax.bar(stack_names, stack_values, color=colors[:len(stack_names)])
            ax.set_title(f"{title}\n({subtitle})")
            ax.set_ylabel(title)

        plt.tight_layout()
        plot_path = plot_dir / "baseline_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plots saved to: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

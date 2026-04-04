#!/usr/bin/env python3
"""M1 — Babble Parameter Autosearch.

Systematically sweeps the babbling parameter space using real TTS +
synthetic room acoustics to find optimal NLMS cerebellum settings.

Each experiment runs 10 babbling iterations against the same synthetic
room, tracking SVSR convergence.

The question isn't *if* the cerebellum can learn (it can — 40dB SVSR on
clean synthetic data). It's what parameters work best under realistic
conditions with ambient noise.

Usage:
    # Run all experiments (synthetic)
    python scripts/m1_autosearch.py

    # Limit to N experiments
    python scripts/m1_autosearch.py --limit 10

    # Resume from experiment index N
    python scripts/m1_autosearch.py --start-at 50

Results are appended to: docs/autosearch_results.tsv
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from itertools import product

import numpy as np

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from scipy.signal import fftconvolve
try:
    from scipy.signal import butter, sosfilt
    HAVE_SCIPY_FILTER = True
except ImportError:
    HAVE_SCIPY_FILTER = False

from cochlea.capture import SR, pcm_to_mels
from cerebellum.adaptive_model import BabblingCerebellum
from thalamus.gate import spectral_subtraction, compute_svsr

# ── Constants ──────────────────────────────────────────────────────────
RESULTS_FILE = "docs/autosearch_results.tsv"
RESULTS_HEADER = "experiment\tnlms_mu\tir_length_ms\tbandpass\tspeech_gate\tsvsr_db\terror_db\tir_corr\tstatus\ttime_s\tnotes"

TRUE_RT60 = 0.3       # Ground truth RT60 for synthetic room

# Parameter grid
NLMS_MU_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25]
IR_LENGTH_VALUES = [100, 150, 200, 300, 500]
BANDPASS_VALUES = [False, True]
SPEECH_GATE_VALUES = [False, True]

NOISE_LEVELS_DB = [-20, -30, -40]  # SNR added to captured signal


# ── Helpers ─────────────────────────────────────────────────────────────

def init_tsv() -> int:
    """Initialize results TSV if needed. Returns count of existing data rows."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(RESULTS_HEADER + "\n")
        return 0
    with open(RESULTS_FILE) as f:
        return max(0, len(f.readlines()) - 1)


def append_result(r: dict) -> None:
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{r['experiment']}\t"
            f"{r['nlms_mu']}\t"
            f"{r['ir_length_ms']}\t"
            f"{r['bandpass']}\t"
            f"{r['speech_gate']}\t"
            f"{r['svsr_db']:.2f}\t"
            f"{r['error_db']:.2f}\t"
            f"{r['ir_corr']:.4f}\t"
            f"{r['status']}\t"
            f"{r['time_s']:.1f}\t"
            f"{r['notes']}\n"
        )


def apply_bandpass(pcm: np.ndarray) -> np.ndarray:
    """300-3400 Hz bandpass filter."""
    if not HAVE_SCIPY_FILTER:
        return pcm
    try:
        nyq = SR / 2.0
        sos = butter(4, [300.0 / nyq, 3400.0 / nyq], btype="band", output="sos")
        return sosfilt(sos, pcm).astype(np.float32)
    except Exception:
        return pcm


def gate_silence(pcm: np.ndarray, threshold_db: float = -35.0, min_gap_ms: int = 200) -> np.ndarray:
    """Keep only signal above threshold, with padding."""
    frame_size = SR // 10
    threshold = 10 ** (threshold_db / 20.0)
    active = np.zeros(len(pcm) // frame_size, dtype=bool)
    for i in range(len(active)):
        frame = pcm[i * frame_size:(i + 1) * frame_size]
        if np.sqrt(np.mean(frame ** 2)) > threshold:
            active[i] = True
    idx = np.where(active)[0]
    if len(idx) == 0:
        return pcm
    gap = max(1, min_gap_ms // 100)
    start = max(0, idx[0] - gap) * frame_size
    end = min(len(pcm), (idx[-1] + 1 + gap) * frame_size)
    return pcm[start:end]


def generate_room_ir(rt60: float = TRUE_RT60) -> np.ndarray:
    """Generate a synthetic room impulse response."""
    length = int(rt60 * SR)
    true_ir = np.zeros(length, dtype=np.float32)
    true_ir[0] = 1.0

    # Early reflections
    true_ir[16] += 0.4
    true_ir[48] += 0.25
    true_ir[96] += 0.15
    true_ir[200] += 0.08

    # Diffuse reverb tail
    for i in range(int(0.03 * SR), len(true_ir)):
        decay = np.exp(-6.91 * (i - int(0.03 * SR)) / (rt60 * SR - int(0.03 * SR)))
        true_ir[i] += np.random.randn() * decay * 0.08

    # Normalize
    energy = np.sum(true_ir ** 2)
    if energy > 0:
        true_ir /= np.sqrt(energy)
    return true_ir


def compute_svsr_for_cerebellum(
    captured: np.ndarray,
    known_pcm: np.ndarray,
    cerebellum: BabblingCerebellum,
) -> float:
    """Compute SVSR using cerebellum's prediction, comparing only the
    overlapping portion (where the cerebellum has actual predictions).

    This is physically honest: we judge the cerebellum on what it *can*
    predict, not on room echoes beyond its model horizon.
    """
    obs_mel = pcm_to_mels(captured)
    pred_mel = cerebellum.predict_mel(known_pcm)

    # Compare only the overlapping portion
    min_frames = min(obs_mel.shape[0], pred_mel.shape[0])
    obs_trim = obs_mel[:min_frames]
    pred_trim = pred_mel[:min_frames]

    residual, mask = spectral_subtraction(obs_trim, pred_trim)
    return float(compute_svsr(obs_trim, pred_trim, residual))


def ir_correlation(learned: np.ndarray, truth: np.ndarray) -> float:
    """Compute correlation between learned and true IRs (handles length mismatch)."""
    min_len = min(len(learned), len(truth))
    if min_len == 0:
        return 0.0
    l = learned[:min_len]
    t = truth[:min_len]
    l_e = np.sqrt(np.sum(l ** 2))
    t_e = np.sqrt(np.sum(t ** 2))
    if l_e < 1e-10 or t_e < 1e-10:
        return 0.0
    return float(np.dot(l, t) / (l_e * t_e))


# ── Experiment Runner ──────────────────────────────────────────────────

async def run_single_experiment(
    exp_num: int,
    nlms_mu: float,
    ir_length_ms: int,
    use_bandpass: bool,
    use_speech_gate: bool,
    known_pcm: np.ndarray,
    true_ir: np.ndarray,
    captured_clean: np.ndarray,
    noise_db: float,
    iterations: int = 10,
) -> dict:
    """Run one parameter combination."""
    t0 = time.perf_counter()
    try:
        # Add ambient noise
        signal_rms = np.sqrt(np.mean(captured_clean ** 2))
        noise_amp = signal_rms * 10 ** (noise_db / 20.0) if noise_db > -100 else 0
        np.random.seed(exp_num)
        noise = np.random.randn(len(captured_clean)).astype(np.float32) * noise_amp
        captured = captured_clean + noise

        # Preprocessing
        bp_known = apply_bandpass(known_pcm.copy()) if use_bandpass else known_pcm.copy()
        bp_captured = apply_bandpass(captured) if use_bandpass else captured.copy()

        if use_speech_gate:
            bp_captured = gate_silence(bp_captured)

        # Babbling
        cerebellum = BabblingCerebellum(ir_length_ms=ir_length_ms, nlms_mu=nlms_mu)
        for _ in range(iterations):
            cerebellum.update(bp_known, bp_captured)

        svsr = compute_svsr_for_cerebellum(bp_captured, bp_known, cerebellum)
        conv = cerebellum.get_convergence()
        corr = ir_correlation(cerebellum.get_learned_ir(), true_ir)
        elapsed = time.perf_counter() - t0

        return {
            "experiment": exp_num,
            "nlms_mu": nlms_mu,
            "ir_length_ms": ir_length_ms,
            "bandpass": use_bandpass,
            "speech_gate": use_speech_gate,
            "noise_db": noise_db,
            "svsr_db": svsr,
            "error_db": conv["current_error_db"],
            "ir_corr": corr,
            "status": "keep" if svsr >= 20.0 else "discard",
            "time_s": elapsed,
            "notes": (f"mu={nlms_mu}, ir={ir_length_ms}ms, bp={use_bandpass}, "
                      f"gate={use_speech_gate}, noise={noise_db}dB"),
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "experiment": exp_num,
            "nlms_mu": nlms_mu,
            "ir_length_ms": ir_length_ms,
            "bandpass": use_bandpass,
            "speech_gate": use_speech_gate,
            "noise_db": noise_db,
            "svsr_db": 0.0,
            "error_db": 0.0,
            "ir_corr": 0.0,
            "status": "crash",
            "time_s": elapsed,
            "notes": f"Exception: {str(e)[:100]}",
        }


# ── Main ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="M1 Babble Parameter Autosearch")
    parser.add_argument("--limit", type=int, default=0, help="Run only N experiments (0 = all)")
    parser.add_argument("--start-at", type=int, default=0, help="Skip to experiment N")
    parser.add_argument("--iterations", type=int, default=10, help="Babble iterations per experiment")
    args = parser.parse_args()

    # Build experiment grid
    grid = list(product(NLMS_MU_VALUES, IR_LENGTH_VALUES, BANDPASS_VALUES, SPEECH_GATE_VALUES, NOISE_LEVELS_DB))
    total = len(grid)

    if args.limit > 0:
        grid = grid[:args.limit]
        total = len(grid)

    skip = args.start_at

    # Generate test data once
    np.random.seed(42)
    true_ir = generate_room_ir()

    # Get real TTS audio
    from broca.tts import synthesize
    known_pcm = await synthesize("Hello, this is Corollary testing the self-listening loop.")
    captured_clean = fftconvolve(known_pcm, true_ir, mode="full").astype(np.float32)

    init_tsv()

    width = 70
    print("\n" + "=" * width)
    print("  COROLLARY — M1 Babble Parameter Autosearch")
    print("=" * width)
    print(f"  TTS: real edge-tts audio")
    print(f"  Room: synthetic IR (RT60 = {TRUE_RT60}s)")
    print(f"  Noise levels: {NOISE_LEVELS_DB}")
    print(f"  Iterations per exp: {args.iterations}")
    print(f"  Parameter grid: {len(grid)} experiments")
    if skip:
        print(f"  Skipping first {skip} experiments")
    print("=" * width)

    results = []
    for idx, (mu, ir_ms, bp, sg, noise) in enumerate(grid):
        actual = skip + idx + 1
        if actual <= skip:
            continue

        prefix = f"[{actual}/]"
        print(f"\n  {prefix} mu={mu}, ir={ir_ms}ms, bp={bp}, gate={sg}, noise={noise}dB")

        r = await run_single_experiment(
            exp_num=actual,
            nlms_mu=mu,
            ir_length_ms=ir_ms,
            use_bandpass=bp,
            use_speech_gate=sg,
            known_pcm=known_pcm,
            true_ir=true_ir,
            captured_clean=captured_clean,
            noise_db=noise,
            iterations=args.iterations,
        )

        results.append(r)
        append_result(r)

        icon = "✅" if r["status"] == "keep" else "⚠️" if r["status"] == "discard" else "❌"
        print(f"    {icon} SVSR={r['svsr_db']:.1f} dB | error={r['error_db']:.1f} dB | ir_corr={r['ir_corr']:.3f} | {r['time_s']:.1f}s")

    # Summary
    keeps = [r for r in results if r["status"] == "keep"]
    discards = [r for r in results if r["status"] == "discard"]
    crashes = [r for r in results if r["status"] == "crash"]

    print(f"\n{'=' * width}")
    print(f"  AUTOSEARCH COMPLETE")
    print(f"{'=' * width}")
    print(f"  Total experiments: {len(results)}")
    print(f"  Kept (SVSR >= 20): {len(keeps)}")
    print(f"  Discarded:         {len(discards)}")
    print(f"  Crashed:           {len(crashes)}")

    if keeps:
        best = max(keeps, key=lambda r: r["svsr_db"])
        print(f"\n  🏆 Best result:")
        print(f"     SVSR:       {best['svsr_db']:.1f} dB")
        print(f"     Parameters: mu={best['nlms_mu']}, ir={best['ir_length_ms']}ms, "
              f"bp={best['bandpass']}, gate={best['speech_gate']}, noise={best['noise_db']}dB")
    else:
        print(f"\n  No experiments met the 20 dB SVSR target.")
        if discards:
            best_near = max(discards, key=lambda r: r["svsr_db"])
            print(f"  Closest: SVSR={best_near['svsr_db']:.1f} dB "
                  f"(mu={best_near['nlms_mu']}, ir={best_near['ir_length_ms']}ms, "
                  f"bp={best_near['bandpass']}, gate={best_near['speech_gate']})")

    print(f"\n  Results: {RESULTS_FILE}")
    print(f"{'=' * width}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""M1A → M1B bridge via repeated variable-vowel adaptation.

This script uses the hardware-facing variable vowel family (`vowel_ah_var`) as a
seed-aware babbling protocol. It repeatedly emits bounded self-like vowel probes,
captures the room response, updates the cerebellum forward model, and can then
score held-out variable-vowel seeds against the adapted model.

It now also supports trust-gated continuation. After a configurable warmup,
held-out evaluation can be used to decide whether additional adaptation cycles
are justified or should be skipped because degradation is already below a chosen
threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from statistics import mean
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    print("Install with: pip install sounddevice")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cerebellum.adaptive_model import BabblingCerebellum
from cochlea.capture import SR, pcm_to_mels
from thalamus.gate import compute_svsr, spectral_subtraction

sys.path.insert(0, os.path.dirname(__file__))
from m1_tone_attribution import generate_stimulus_with_metadata
from m1_probe_capture import resample_audio


def parse_seed_list(text: str) -> list[int]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("seed list must not be empty")
    return [int(v) for v in values]


def capture_probe(
    pcm: np.ndarray,
    input_device: int | None,
    output_device: int | None,
    lead_in_s: float,
    tail_s: float,
) -> dict[str, Any]:
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

    playback_duration = len(pcm) / SR
    capture_duration = lead_in_s + playback_duration + tail_s
    total_samples = int(capture_duration * native_sr)
    audio = np.zeros(total_samples, dtype=np.float32)

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

    return {
        "captured_pcm": audio.astype(np.float32),
        "input_device_name": input_name,
        "output_device_name": output_name,
        "capture_duration_s": round(capture_duration, 4),
        "playback_duration_s": round(playback_duration, 4),
        "capture_elapsed_s": round(capture_finished_at - capture_started_at, 4),
        "playback_started": playback_started_at is not None,
    }


def aligned_svsr(captured_pcm: np.ndarray, predicted_pcm: np.ndarray) -> float:
    observed_mel = pcm_to_mels(captured_pcm)
    predicted_mel = pcm_to_mels(predicted_pcm)

    t_obs = observed_mel.shape[0]
    t_pred = predicted_mel.shape[0]
    best_svsr = 0.0
    for shift in range(-10, 41):
        if shift >= 0:
            pred_shift = np.concatenate([
                np.tile(predicted_mel[:1], (shift, 1)),
                predicted_mel,
            ], axis=0)[:t_obs]
        else:
            pred_shift = predicted_mel[-shift:][:t_obs] if -shift < t_pred else np.zeros_like(predicted_mel[:t_obs])
            if len(pred_shift) < t_obs:
                pred_shift = np.concatenate([
                    pred_shift,
                    np.zeros((t_obs - len(pred_shift), predicted_mel.shape[1]), dtype=np.float32),
                ])
        obs_trim = observed_mel[:len(pred_shift)]
        if len(obs_trim) != len(pred_shift):
            continue
        residual, _ = spectral_subtraction(obs_trim, pred_shift)
        svsr = compute_svsr(obs_trim, pred_shift, residual)
        if svsr > best_svsr:
            best_svsr = float(svsr)
    return best_svsr


def summarize_parameter_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def top(rows_in: list[dict[str, Any]], reverse: bool) -> dict[str, Any]:
        row = sorted(rows_in, key=lambda r: float(r.get("svsr_db", 0.0)), reverse=reverse)[0]
        return {
            "seed": row.get("seed"),
            "svsr_db": row.get("svsr_db"),
            "stimulus_duration_realized_ms": row.get("stimulus_duration_realized_ms"),
            "stimulus_f0_hz": row.get("stimulus_f0_hz"),
            "stimulus_attack_ms": row.get("stimulus_attack_ms"),
            "stimulus_release_ms": row.get("stimulus_release_ms"),
        }

    durations = np.array([float(r["stimulus_duration_realized_ms"]) for r in rows], dtype=np.float32)
    f0s = np.array([float(r["stimulus_f0_hz"]) for r in rows], dtype=np.float32)
    attacks = np.array([float(r["stimulus_attack_ms"]) for r in rows], dtype=np.float32)
    releases = np.array([float(r["stimulus_release_ms"]) for r in rows], dtype=np.float32)
    svsrs = np.array([float(r["svsr_db"]) for r in rows], dtype=np.float32)

    def corr(a: np.ndarray, b: np.ndarray) -> float | None:
        if len(a) < 2 or float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
            return None
        return round(float(np.corrcoef(a, b)[0, 1]), 4)

    return {
        "row_count": len(rows),
        "best_row": top(rows, reverse=True),
        "worst_row": top(rows, reverse=False),
        "correlations": {
            "duration_ms_vs_svsr": corr(durations, svsrs),
            "f0_hz_vs_svsr": corr(f0s, svsrs),
            "attack_ms_vs_svsr": corr(attacks, svsrs),
            "release_ms_vs_svsr": corr(releases, svsrs),
        },
        "means": {
            "duration_ms": round(float(np.mean(durations)), 4),
            "f0_hz": round(float(np.mean(f0s)), 4),
            "attack_ms": round(float(np.mean(attacks)), 4),
            "release_ms": round(float(np.mean(releases)), 4),
            "svsr_db": round(float(np.mean(svsrs)), 4),
        },
    }


def evaluate_heldout(
    cerebellum: BabblingCerebellum,
    heldout_seeds: list[int],
    args: argparse.Namespace,
    phase: str,
) -> list[dict[str, Any]]:
    heldout_rows: list[dict[str, Any]] = []
    for seed in heldout_seeds:
        meta = generate_stimulus_with_metadata(args.stimulus_kind, args.duration_ms, args.amplitude, seed=seed)
        capture = capture_probe(
            meta["pcm"],
            input_device=args.input_device,
            output_device=args.output_device,
            lead_in_s=args.lead_in_s,
            tail_s=args.tail_s,
        )
        predicted_pcm = cerebellum.predict_pcm(meta["pcm"])
        svsr = aligned_svsr(capture["captured_pcm"], predicted_pcm)
        row = {
            "phase": phase,
            "seed": seed,
            "setup_label": args.setup_label,
            "stimulus_family": meta["family"],
            "stimulus_amplitude_realized": meta["amplitude"],
            "stimulus_duration_realized_ms": meta["duration_ms"],
            "stimulus_f0_hz": meta["f0_hz"],
            "stimulus_attack_ms": meta["attack_ms"],
            "stimulus_release_ms": meta["release_ms"],
            "stimulus_envelope_lfo_hz": meta["envelope_lfo_hz"],
            "svsr_db": round(svsr, 4),
            "logged_at": time.time(),
        }
        heldout_rows.append(row)
        print(f"\n{phase} seed={seed} | svsr_db={row['svsr_db']:.2f}")
    return heldout_rows


def run_adaptation(args: argparse.Namespace) -> dict[str, Any]:
    train_seeds = parse_seed_list(args.train_seeds)
    heldout_seeds = parse_seed_list(args.heldout_seeds) if args.heldout_seeds else []

    if args.dry_run:
        return {
            "mode": "dry_run",
            "setup_label": args.setup_label,
            "stimulus_kind": args.stimulus_kind,
            "train_seeds": train_seeds,
            "heldout_seeds": heldout_seeds,
            "cycles": args.cycles,
            "iterations": args.cycles * len(train_seeds),
        }

    cerebellum = BabblingCerebellum(ir_length_ms=args.ir_length_ms, nlms_mu=args.nlms_mu)
    history: list[dict[str, Any]] = []
    gating: dict[str, Any] = {
        "enabled": bool(args.trust_gate),
        "warmup_cycles": int(args.gate_warmup_cycles),
        "readapt_threshold_db": round(float(args.readapt_threshold_db), 4),
        "forced_continue": bool(args.force_continue),
        "decision": None,
    }

    print("\n" + "=" * 64)
    print("  COROLLARY — M1 Variable-Vowel Adaptation Bridge")
    print("=" * 64)
    print(f"  setup_label:   {args.setup_label}")
    print(f"  stimulus_kind: {args.stimulus_kind}")
    print(f"  train_seeds:   {train_seeds}")
    print(f"  heldout_seeds: {heldout_seeds}")
    print(f"  cycles:        {args.cycles}")
    print(f"  ir_length_ms:  {args.ir_length_ms}")
    print(f"  nlms_mu:       {args.nlms_mu}")
    if args.trust_gate:
        print(f"  trust_gate:    on (warmup_cycles={args.gate_warmup_cycles}, threshold_db={args.readapt_threshold_db:.4f})")
    print("=" * 64)

    iteration = 0
    stopped_early = False
    pre_gate_heldout_rows: list[dict[str, Any]] = []
    for cycle in range(args.cycles):
        for seed in train_seeds:
            iteration += 1
            meta = generate_stimulus_with_metadata(args.stimulus_kind, args.duration_ms, args.amplitude, seed=seed)
            print(
                f"\n--- adapt {iteration}/{args.cycles * len(train_seeds)} | cycle={cycle + 1} seed={seed} "
                f"dur={meta['duration_ms']}ms f0={meta['f0_hz']}Hz"
            )
            capture = capture_probe(
                meta["pcm"],
                input_device=args.input_device,
                output_device=args.output_device,
                lead_in_s=args.lead_in_s,
                tail_s=args.tail_s,
            )
            error = float(cerebellum.update(meta["pcm"], capture["captured_pcm"]))
            conv = cerebellum.get_convergence()
            predicted_pcm = cerebellum.predict_pcm(meta["pcm"])
            svsr = aligned_svsr(capture["captured_pcm"], predicted_pcm)
            trust = cerebellum.trust_status()
            row = {
                "iteration": iteration,
                "cycle": cycle + 1,
                "seed": seed,
                "setup_label": args.setup_label,
                "stimulus_family": meta["family"],
                "stimulus_amplitude_realized": meta["amplitude"],
                "stimulus_duration_realized_ms": meta["duration_ms"],
                "stimulus_f0_hz": meta["f0_hz"],
                "stimulus_attack_ms": meta["attack_ms"],
                "stimulus_release_ms": meta["release_ms"],
                "stimulus_envelope_lfo_hz": meta["envelope_lfo_hz"],
                "prediction_error": error,
                "prediction_error_db": conv["current_error_db"],
                "svsr_db": round(svsr, 4),
                "trust": round(float(trust["trust"]), 6),
                "capture_duration_s": capture["capture_duration_s"],
                "playback_duration_s": capture["playback_duration_s"],
                "logged_at": time.time(),
            }
            history.append(row)
            print(
                f"  error_db={row['prediction_error_db']:.2f} | svsr_db={row['svsr_db']:.2f} | trust={row['trust']:.3f}"
            )

        if args.trust_gate and heldout_seeds and (cycle + 1) == args.gate_warmup_cycles:
            pre_gate_heldout_rows = evaluate_heldout(cerebellum, heldout_seeds, args, "gate_probe_heldout")
            pre_gate_mean = round(mean(r["svsr_db"] for r in pre_gate_heldout_rows), 4)
            latest_train_mean = round(mean(r["svsr_db"] for r in history[-len(train_seeds):]), 4) if train_seeds else None
            degradation_db = None if latest_train_mean is None else round(latest_train_mean - pre_gate_mean, 4)
            should_continue = bool(args.force_continue)
            decision_reason = "force_continue" if args.force_continue else "below_threshold"
            if not args.force_continue and degradation_db is not None and degradation_db >= args.readapt_threshold_db:
                should_continue = True
                decision_reason = "threshold_exceeded"
            elif not args.force_continue:
                should_continue = False
                decision_reason = "below_threshold"

            gating.update({
                "decision": {
                    "cycle": cycle + 1,
                    "latest_train_mean_svsr_db": latest_train_mean,
                    "probe_heldout_mean_svsr_db": pre_gate_mean,
                    "degradation_db": degradation_db,
                    "should_continue": should_continue,
                    "reason": decision_reason,
                }
            })
            print(
                f"\nTrust-gated bridge decision: latest_train_mean_svsr_db={latest_train_mean} | "
                f"probe_heldout_mean_svsr_db={pre_gate_mean} | degradation_db={degradation_db} | "
                f"threshold_db={args.readapt_threshold_db:.4f} | should_continue={should_continue} | reason={decision_reason}"
            )
            if not should_continue:
                stopped_early = True
                break

    heldout_rows = evaluate_heldout(cerebellum, heldout_seeds, args, "heldout")

    result = {
        "mode": "adaptation",
        "setup_label": args.setup_label,
        "stimulus_kind": args.stimulus_kind,
        "train_seeds": train_seeds,
        "heldout_seeds": heldout_seeds,
        "cycles": args.cycles,
        "history": history,
        "heldout": heldout_rows,
        "summary": {
            "iterations": len(history),
            "svsr_initial_db": round(history[0]["svsr_db"], 4) if history else None,
            "svsr_final_db": round(history[-1]["svsr_db"], 4) if history else None,
            "prediction_error_initial_db": round(history[0]["prediction_error_db"], 4) if history else None,
            "prediction_error_final_db": round(history[-1]["prediction_error_db"], 4) if history else None,
            "trust_final": history[-1]["trust"] if history else None,
            "heldout_mean_svsr_db": round(mean(r["svsr_db"] for r in heldout_rows), 4) if heldout_rows else None,
        },
        "analysis": {
            "train_parameter_effects": summarize_parameter_effects(history),
            "heldout_parameter_effects": summarize_parameter_effects(heldout_rows),
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M1 variable-vowel adaptation bridge")
    parser.add_argument("--input-device", type=int, required=True)
    parser.add_argument("--output-device", type=int, required=True)
    parser.add_argument("--stimulus-kind", default="vowel_ah_var", choices=["vowel_ah_var", "vowel_ah_var_nearfield", "vowel_ah_var_nearfield_room2", "babble_ba_var", "babble_ba_repeat_var", "babble_ba_repeat_short_var", "babble_ba_repeat_few_var", "babble_ba_repeat_wide_var", "babble_ba_repeat_short_tight_var", "babble_da_var", "babble_ma_var"])
    parser.add_argument("--train-seeds", default="0,7,19")
    parser.add_argument("--heldout-seeds", default="23,29")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--duration-ms", type=int, default=300)
    parser.add_argument("--amplitude", type=float, default=0.7)
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.0)
    parser.add_argument("--ir-length-ms", type=int, default=300)
    parser.add_argument("--nlms-mu", type=float, default=0.25)
    parser.add_argument("--setup-label", default="baseline_close_varbridge")
    parser.add_argument("--trust-gate", action="store_true",
                        help="After the warmup cycle count, probe held-out performance and stop early when degradation stays below threshold")
    parser.add_argument("--gate-warmup-cycles", type=int, default=1,
                        help="Number of cycles to run before the trust gate probes held-out performance")
    parser.add_argument("--readapt-threshold-db", type=float, default=1.0,
                        help="Continue adaptation only when measured degradation meets or exceeds this threshold in dB")
    parser.add_argument("--force-continue", action="store_true",
                        help="Force continuation even when the trust gate would stop early")
    parser.add_argument("--output", default="-")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_adaptation(args)
    payload = json.dumps(result, sort_keys=True)
    if args.output == "-":
        print("\nJSON_RESULT=" + payload)
    else:
        with open(args.output, "w") as f:
            f.write(payload + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Explicit cross-room transfer and recovery harness for M1 variable-vowel adaptation.

Protocol:
1. Adapt in room A on a small train seed set.
2. Evaluate held-out performance in room A.
3. Pause for live room transfer without shutting down.
4. Evaluate immediate held-out performance in room B before any re-adaptation.
5. Decide whether re-adaptation is justified from the measured held-out drop.
6. Re-adapt in room B only if the drop crosses the configured threshold.
7. Evaluate held-out performance again in room B and report recovery.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cerebellum.adaptive_model import BabblingCerebellum

sys.path.insert(0, os.path.dirname(__file__))
from m1_variable_vowel_adapt import parse_seed_list, capture_probe, aligned_svsr
from m1_tone_attribution import generate_stimulus_with_metadata


def run_eval_pass(
    cerebellum: BabblingCerebellum,
    seeds: list[int],
    stimulus_kind: str,
    duration_ms: int,
    amplitude: float,
    input_device: int,
    output_device: int,
    lead_in_s: float,
    tail_s: float,
    setup_label: str,
    phase: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        meta = generate_stimulus_with_metadata(stimulus_kind, duration_ms, amplitude, seed=seed)
        capture = capture_probe(
            meta["pcm"],
            input_device=input_device,
            output_device=output_device,
            lead_in_s=lead_in_s,
            tail_s=tail_s,
        )
        predicted_pcm = cerebellum.predict_pcm(meta["pcm"])
        svsr = aligned_svsr(capture["captured_pcm"], predicted_pcm)
        rows.append({
            "phase": phase,
            "setup_label": setup_label,
            "seed": seed,
            "stimulus_duration_realized_ms": meta["duration_ms"],
            "stimulus_f0_hz": meta["f0_hz"],
            "stimulus_attack_ms": meta["attack_ms"],
            "stimulus_release_ms": meta["release_ms"],
            "svsr_db": round(float(svsr), 4),
            "logged_at": time.time(),
        })
        print(f"{phase} seed={seed} | svsr_db={svsr:.2f}")
    return rows


def run_adapt_pass(
    cerebellum: BabblingCerebellum,
    seeds: list[int],
    cycles: int,
    stimulus_kind: str,
    duration_ms: int,
    amplitude: float,
    input_device: int,
    output_device: int,
    lead_in_s: float,
    tail_s: float,
    setup_label: str,
    phase: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    iteration = 0
    for cycle in range(cycles):
        for seed in seeds:
            iteration += 1
            meta = generate_stimulus_with_metadata(stimulus_kind, duration_ms, amplitude, seed=seed)
            print(f"\n--- {phase} {iteration}/{cycles * len(seeds)} | cycle={cycle + 1} seed={seed} dur={meta['duration_ms']}ms f0={meta['f0_hz']}Hz")
            capture = capture_probe(
                meta["pcm"],
                input_device=input_device,
                output_device=output_device,
                lead_in_s=lead_in_s,
                tail_s=tail_s,
            )
            cerebellum.update(meta["pcm"], capture["captured_pcm"])
            conv = cerebellum.get_convergence()
            predicted_pcm = cerebellum.predict_pcm(meta["pcm"])
            svsr = aligned_svsr(capture["captured_pcm"], predicted_pcm)
            trust = cerebellum.trust_status()
            row = {
                "phase": phase,
                "cycle": cycle + 1,
                "iteration": iteration,
                "seed": seed,
                "setup_label": setup_label,
                "stimulus_duration_realized_ms": meta["duration_ms"],
                "stimulus_f0_hz": meta["f0_hz"],
                "stimulus_attack_ms": meta["attack_ms"],
                "stimulus_release_ms": meta["release_ms"],
                "prediction_error_db": conv["current_error_db"],
                "svsr_db": round(float(svsr), 4),
                "trust": round(float(trust["trust"]), 6),
                "logged_at": time.time(),
            }
            rows.append(row)
            print(f"  error_db={conv['current_error_db']:.2f} | svsr_db={svsr:.2f} | trust={trust['trust']:.3f}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    svsrs = [float(r["svsr_db"]) for r in rows]
    return {
        "row_count": len(rows),
        "svsr_mean_db": round(mean(svsrs), 4),
        "svsr_best_db": round(max(svsrs), 4),
        "svsr_worst_db": round(min(svsrs), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-room transfer and recovery harness")
    parser.add_argument("--input-device", type=int, required=True)
    parser.add_argument("--output-device", type=int, required=True)
    parser.add_argument("--stimulus-kind", default="vowel_ah_var_nearfield", choices=["vowel_ah_var", "vowel_ah_var_nearfield", "vowel_ah_var_nearfield_room2", "babble_ba_var", "babble_ba_repeat_var", "babble_ba_repeat_short_var", "babble_ba_repeat_few_var", "babble_ba_repeat_wide_var", "babble_ba_repeat_short_tight_var", "babble_da_var", "babble_ma_var"])
    parser.add_argument("--train-seeds", default="0,7,19")
    parser.add_argument("--heldout-seeds", default="23,29")
    parser.add_argument("--room-a-cycles", type=int, default=2)
    parser.add_argument("--room-b-cycles", type=int, default=2)
    parser.add_argument("--readapt-threshold-db", type=float, default=1.0,
                        help="Only re-adapt in room B when immediate held-out drop meets or exceeds this threshold in dB")
    parser.add_argument("--force-readapt", action="store_true",
                        help="Always run room-B re-adaptation regardless of measured drop")
    parser.add_argument("--duration-ms", type=int, default=300)
    parser.add_argument("--amplitude", type=float, default=0.7)
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.0)
    parser.add_argument("--ir-length-ms", type=int, default=300)
    parser.add_argument("--nlms-mu", type=float, default=0.25)
    parser.add_argument("--room-a-label", default="roomA")
    parser.add_argument("--room-b-label", default="roomB")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    train_seeds = parse_seed_list(args.train_seeds)
    heldout_seeds = parse_seed_list(args.heldout_seeds)

    cerebellum = BabblingCerebellum(ir_length_ms=args.ir_length_ms, nlms_mu=args.nlms_mu)

    print("\n" + "=" * 68)
    print("  COROLLARY — M1 Cross-Room Transfer and Recovery")
    print("=" * 68)
    print(f"  stimulus_kind: {args.stimulus_kind}")
    print(f"  room_a_label:  {args.room_a_label}")
    print(f"  room_b_label:  {args.room_b_label}")
    print(f"  train_seeds:   {train_seeds}")
    print(f"  heldout_seeds: {heldout_seeds}")
    print("=" * 68)

    room_a_history = run_adapt_pass(
        cerebellum, train_seeds, args.room_a_cycles, args.stimulus_kind,
        args.duration_ms, args.amplitude, args.input_device, args.output_device,
        args.lead_in_s, args.tail_s, args.room_a_label, "room_a_adapt",
    )
    room_a_heldout = run_eval_pass(
        cerebellum, heldout_seeds, args.stimulus_kind, args.duration_ms,
        args.amplitude, args.input_device, args.output_device, args.lead_in_s,
        args.tail_s, args.room_a_label, "room_a_heldout",
    )

    print("\nMOVE THE SETUP TO THE NEXT ROOM, THEN PRESS ENTER TO CONTINUE...")
    input()

    room_b_pre = run_eval_pass(
        cerebellum, heldout_seeds, args.stimulus_kind, args.duration_ms,
        args.amplitude, args.input_device, args.output_device, args.lead_in_s,
        args.tail_s, args.room_b_label, "room_b_pre_readapt_heldout",
    )

    room_a_mean = summarize(room_a_heldout).get("svsr_mean_db")
    room_b_pre_mean = summarize(room_b_pre).get("svsr_mean_db")
    immediate_transfer_drop_db = None
    if room_a_mean is not None and room_b_pre_mean is not None:
        immediate_transfer_drop_db = round(room_a_mean - room_b_pre_mean, 4)

    should_readapt = bool(args.force_readapt)
    readapt_reason = "force_readapt" if args.force_readapt else "below_threshold"
    if not args.force_readapt and immediate_transfer_drop_db is not None and immediate_transfer_drop_db >= args.readapt_threshold_db:
        should_readapt = True
        readapt_reason = "threshold_exceeded"

    print(
        f"\nTrust-gated decision: immediate_transfer_drop_db={immediate_transfer_drop_db} | "
        f"threshold_db={args.readapt_threshold_db:.4f} | should_readapt={should_readapt} | reason={readapt_reason}"
    )

    room_b_history: list[dict[str, Any]] = []
    if should_readapt:
        room_b_history = run_adapt_pass(
            cerebellum, train_seeds, args.room_b_cycles, args.stimulus_kind,
            args.duration_ms, args.amplitude, args.input_device, args.output_device,
            args.lead_in_s, args.tail_s, args.room_b_label, "room_b_readapt",
        )
        room_b_post = run_eval_pass(
            cerebellum, heldout_seeds, args.stimulus_kind, args.duration_ms,
            args.amplitude, args.input_device, args.output_device, args.lead_in_s,
            args.tail_s, args.room_b_label, "room_b_post_readapt_heldout",
        )
    else:
        room_b_post = []

    summary = {
        "room_a_heldout_mean_svsr_db": room_a_mean,
        "room_b_pre_readapt_mean_svsr_db": room_b_pre_mean,
        "room_b_post_readapt_mean_svsr_db": summarize(room_b_post).get("svsr_mean_db") if room_b_post else None,
        "immediate_transfer_drop_db": immediate_transfer_drop_db,
        "readapt_threshold_db": round(float(args.readapt_threshold_db), 4),
        "should_readapt": should_readapt,
        "readapt_reason": readapt_reason,
    }
    if summary["room_b_pre_readapt_mean_svsr_db"] is not None and summary["room_b_post_readapt_mean_svsr_db"] is not None:
        summary["readapt_recovery_db"] = round(
            summary["room_b_post_readapt_mean_svsr_db"] - summary["room_b_pre_readapt_mean_svsr_db"], 4
        )

    payload = {
        "stimulus_kind": args.stimulus_kind,
        "room_a_label": args.room_a_label,
        "room_b_label": args.room_b_label,
        "train_seeds": train_seeds,
        "heldout_seeds": heldout_seeds,
        "room_a_history": room_a_history,
        "room_a_heldout": room_a_heldout,
        "room_b_pre_readapt_heldout": room_b_pre,
        "room_b_history": room_b_history,
        "room_b_post_readapt_heldout": room_b_post,
        "summary": summary,
    }

    blob = json.dumps(payload, sort_keys=True)
    if args.output == "-":
        print("\nJSON_RESULT=" + blob)
    else:
        with open(args.output, "w") as f:
            f.write(blob + "\n")
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()

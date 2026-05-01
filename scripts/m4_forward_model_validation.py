#!/usr/bin/env python3
"""Validate whether a forward model improves self-template fidelity.

This harness is the missing evidence step between "the forward model should help"
and "the forward model actually helps on this geometry."

It can:
1. load an existing cerebellum model or train a fresh one from self-only captures
2. compare dry vs room-colored predicted self against captured self-only audio
3. report similarity and onset diagnostics per seed and in aggregate
4. emit the exact follow-up benchmark command for M2/M4 evaluation

The core question is simple:
    does predicted_self match captured_self better than dry_self?

If not, neither predicted subtraction nor room-colored M4 registration should be
expected to beat the baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from cerebellum.adaptive_model import BabblingCerebellum
from cerebellum.room_coloring import estimate_onset_samples, predict_room_colored_self
from m1_tone_attribution import generate_stimulus_with_metadata
from m1_variable_vowel_adapt import capture_probe
from m2_benchmark import local_xcorr


@dataclass
class TemplateComparison:
    seed: int
    dry_score: float
    predicted_score: float
    score_delta: float
    dry_shift_samples: int
    predicted_shift_samples: int
    captured_onset_samples: int
    dry_onset_error_samples: int
    predicted_onset_error_samples: int
    duration_ms: int
    amplitude: float


def synthetic_capture_from_model(
    dry_self_pcm: np.ndarray,
    cerebellum: BabblingCerebellum,
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    predicted = cerebellum.predict_pcm(dry_self_pcm)
    if noise_std <= 0:
        return predicted.astype(np.float32)
    rng = np.random.default_rng(seed)
    return (predicted + rng.standard_normal(len(predicted)).astype(np.float32) * noise_std).astype(np.float32)


def compare_templates_against_capture(
    captured_pcm: np.ndarray,
    dry_self_pcm: np.ndarray,
    predicted_self_pcm: np.ndarray,
    search_ms: float = 250.0,
) -> dict[str, Any]:
    captured_onset = estimate_onset_samples(captured_pcm)

    def score_template(template: np.ndarray) -> dict[str, Any]:
        if len(template) >= len(captured_pcm):
            n = min(len(template), len(captured_pcm))
            template_trim = template[:n]
            captured_trim = captured_pcm[:n]
            denom = float(np.linalg.norm(template_trim) * np.linalg.norm(captured_trim))
            score = 0.0 if denom < 1e-9 else float(np.dot(template_trim, captured_trim) / denom)
            shift = 0
        else:
            score, shift = local_xcorr(captured_pcm, template, captured_onset, search_ms=search_ms)
        template_onset = estimate_onset_samples(template)
        return {
            "score": float(score),
            "shift_samples": int(shift),
            "template_onset_samples": int(template_onset),
            "onset_error_samples": int(abs(captured_onset - (shift + template_onset))),
        }

    dry = score_template(dry_self_pcm)
    predicted = score_template(predicted_self_pcm)

    return {
        "captured_onset_samples": int(captured_onset),
        "dry": dry,
        "predicted": predicted,
    }


def build_benchmark_command(args: argparse.Namespace, cerebellum_model_path: str) -> str:
    cmd = [
        "./.venv/bin/python",
        "scripts/m2_benchmark.py",
        "--mode",
        args.benchmark_mode,
        "--self-stim",
        args.stimulus_kind,
        "--input-device",
        str(args.input_device),
        "--output-device",
        str(args.output_device),
        "--noise-device",
        str(args.noise_device),
        "--cerebellum-model",
        cerebellum_model_path,
        "--m4-enable",
        "--self-amplitude",
        str(args.benchmark_self_amplitude),
        "--self-duration-ms",
        str(args.benchmark_self_duration_ms),
        "--similarity-threshold",
        str(args.benchmark_similarity_threshold),
        "--onset-latency-ms",
        str(args.benchmark_onset_latency_ms),
        "--onset-method",
        args.benchmark_onset_method,
        "--causal-tolerance-ms",
        str(args.benchmark_causal_tolerance_ms),
    ]
    if args.m4_mock:
        cmd.append("--m4-mock")
    if args.benchmark_capture_root:
        cmd.extend(["--capture-root", args.benchmark_capture_root])
    return " ".join(shlex.quote(part) for part in cmd)


def train_cerebellum_from_hardware(args: argparse.Namespace) -> tuple[BabblingCerebellum, str, list[dict[str, Any]]]:
    cerebellum = BabblingCerebellum(ir_length_ms=args.ir_length_ms, nlms_mu=args.nlms_mu)
    train_rows: list[dict[str, Any]] = []
    seeds = parse_seed_spec(args.train_seeds)

    for cycle in range(args.train_cycles):
        for seed in seeds:
            meta = generate_stimulus_with_metadata(args.stimulus_kind, args.duration_ms, args.amplitude, seed=seed)
            capture = capture_probe(
                meta["pcm"],
                input_device=args.input_device,
                output_device=args.output_device,
                lead_in_s=args.lead_in_s,
                tail_s=args.tail_s,
            )
            error = float(cerebellum.update(meta["pcm"], capture["captured_pcm"]))
            conv = cerebellum.get_convergence()
            train_rows.append({
                "cycle": cycle + 1,
                "seed": seed,
                "duration_ms": meta["duration_ms"],
                "amplitude": meta["amplitude"],
                "prediction_error": error,
                "prediction_error_db": conv["current_error_db"],
                "trust": cerebellum.trust_status()["trust"],
            })
            print(
                f"train cycle={cycle + 1} seed={seed} | "
                f"error_db={conv['current_error_db']:.2f} | trust={cerebellum.trust_status()['trust']:.3f}"
            )

    model_dir = Path(args.output_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_dir / f"forward_model_{int(time.time())}.json")
    saved = cerebellum.save_model(path=model_path, metadata={
        "mode": "m4_forward_model_validation",
        "stimulus_kind": args.stimulus_kind,
        "train_seeds": seeds,
        "train_cycles": args.train_cycles,
    })
    return cerebellum, saved, train_rows


def parse_seed_spec(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a)
            end = int(b)
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("seed specification must not be empty")
    return values


def evaluate_hardware(args: argparse.Namespace, cerebellum: BabblingCerebellum) -> list[TemplateComparison]:
    rows: list[TemplateComparison] = []
    eval_seeds = parse_seed_spec(args.eval_seeds)

    for seed in eval_seeds:
        meta = generate_stimulus_with_metadata(args.stimulus_kind, args.duration_ms, args.amplitude, seed=seed)
        capture = capture_probe(
            meta["pcm"],
            input_device=args.input_device,
            output_device=args.output_device,
            lead_in_s=args.lead_in_s,
            tail_s=args.tail_s,
        )
        predicted_self_pcm = cerebellum.predict_pcm(meta["pcm"])
        cmp = compare_templates_against_capture(
            capture["captured_pcm"],
            meta["pcm"],
            predicted_self_pcm,
            search_ms=args.search_ms,
        )
        row = TemplateComparison(
            seed=seed,
            dry_score=cmp["dry"]["score"],
            predicted_score=cmp["predicted"]["score"],
            score_delta=cmp["predicted"]["score"] - cmp["dry"]["score"],
            dry_shift_samples=cmp["dry"]["shift_samples"],
            predicted_shift_samples=cmp["predicted"]["shift_samples"],
            captured_onset_samples=cmp["captured_onset_samples"],
            dry_onset_error_samples=cmp["dry"]["onset_error_samples"],
            predicted_onset_error_samples=cmp["predicted"]["onset_error_samples"],
            duration_ms=meta["duration_ms"],
            amplitude=meta["amplitude"],
        )
        rows.append(row)
        print(
            f"eval seed={seed} | dry={row.dry_score:.4f} | predicted={row.predicted_score:.4f} | "
            f"delta={row.score_delta:+.4f} | dry_onset_err={row.dry_onset_error_samples} | "
            f"pred_onset_err={row.predicted_onset_error_samples}"
        )
    return rows


def evaluate_synthetic(args: argparse.Namespace, cerebellum: BabblingCerebellum) -> list[TemplateComparison]:
    rows: list[TemplateComparison] = []
    eval_seeds = parse_seed_spec(args.eval_seeds)

    for seed in eval_seeds:
        meta = generate_stimulus_with_metadata(args.stimulus_kind, args.duration_ms, args.amplitude, seed=seed)
        captured = synthetic_capture_from_model(meta["pcm"], cerebellum, noise_std=args.synthetic_noise_std, seed=seed)
        predicted_self_pcm = cerebellum.predict_pcm(meta["pcm"])
        cmp = compare_templates_against_capture(captured, meta["pcm"], predicted_self_pcm, search_ms=args.search_ms)
        rows.append(TemplateComparison(
            seed=seed,
            dry_score=cmp["dry"]["score"],
            predicted_score=cmp["predicted"]["score"],
            score_delta=cmp["predicted"]["score"] - cmp["dry"]["score"],
            dry_shift_samples=cmp["dry"]["shift_samples"],
            predicted_shift_samples=cmp["predicted"]["shift_samples"],
            captured_onset_samples=cmp["captured_onset_samples"],
            dry_onset_error_samples=cmp["dry"]["onset_error_samples"],
            predicted_onset_error_samples=cmp["predicted"]["onset_error_samples"],
            duration_ms=meta["duration_ms"],
            amplitude=meta["amplitude"],
        ))
    return rows


def summarize(rows: list[TemplateComparison]) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0}
    return {
        "row_count": len(rows),
        "mean_dry_score": round(mean(r.dry_score for r in rows), 6),
        "mean_predicted_score": round(mean(r.predicted_score for r in rows), 6),
        "mean_score_delta": round(mean(r.score_delta for r in rows), 6),
        "predicted_better_count": sum(1 for r in rows if r.score_delta > 0),
        "predicted_not_worse_count": sum(1 for r in rows if r.score_delta >= 0),
        "mean_dry_onset_error_samples": round(mean(r.dry_onset_error_samples for r in rows), 3),
        "mean_predicted_onset_error_samples": round(mean(r.predicted_onset_error_samples for r in rows), 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward-model validation harness for M4/M2")
    parser.add_argument("--mode", choices=["dry_run", "synthetic", "hardware"], default="dry_run")
    parser.add_argument("--output-root", default="data/m4_forward_model_validation")
    parser.add_argument("--stimulus-kind", default="vowel_ah_var_nearfield")
    parser.add_argument("--duration-ms", type=int, default=350)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--eval-seeds", default="3,4")
    parser.add_argument("--train-cycles", type=int, default=2)
    parser.add_argument("--ir-length-ms", type=int, default=300)
    parser.add_argument("--nlms-mu", type=float, default=0.25)
    parser.add_argument("--search-ms", type=float, default=250.0)
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.0)
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--cerebellum-model", default=None)
    parser.add_argument("--synthetic-noise-std", type=float, default=0.0)
    parser.add_argument("--emit-benchmark-command", action="store_true")
    parser.add_argument("--benchmark-mode", choices=["benchmark", "offline"], default="benchmark")
    parser.add_argument("--benchmark-capture-root", default=None)
    parser.add_argument("--benchmark-self-amplitude", type=float, default=0.2)
    parser.add_argument("--benchmark-self-duration-ms", type=int, default=350)
    parser.add_argument("--benchmark-similarity-threshold", type=float, default=0.20)
    parser.add_argument("--benchmark-onset-latency-ms", type=float, default=400.0)
    parser.add_argument("--benchmark-onset-method", choices=["energy", "score", "xcorr"], default="energy")
    parser.add_argument("--benchmark-causal-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--m4-mock", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true", help="If set, run benchmark command after validation succeeds.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "dry_run":
        payload = {
            "mode": "dry_run",
            "stimulus_kind": args.stimulus_kind,
            "train_seeds": parse_seed_spec(args.train_seeds),
            "eval_seeds": parse_seed_spec(args.eval_seeds),
            "train_cycles": args.train_cycles,
            "cerebellum_model": args.cerebellum_model,
        }
        if args.emit_benchmark_command:
            model_path = args.cerebellum_model or str(output_root / "forward_model_PLACEHOLDER.json")
            payload["benchmark_command"] = build_benchmark_command(args, model_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.cerebellum_model:
        cerebellum = BabblingCerebellum.load_model(args.cerebellum_model)
        model_path = args.cerebellum_model
        train_rows: list[dict[str, Any]] = []
    elif args.mode == "hardware":
        cerebellum, model_path, train_rows = train_cerebellum_from_hardware(args)
    else:
        cerebellum = BabblingCerebellum(ir_length_ms=args.ir_length_ms, nlms_mu=args.nlms_mu)
        cerebellum.room_ir[0] = 1.0
        model_path = "synthetic_in_memory"
        train_rows = []

    if args.mode == "hardware":
        eval_rows = evaluate_hardware(args, cerebellum)
    else:
        eval_rows = evaluate_synthetic(args, cerebellum)

    summary = summarize(eval_rows)
    payload = {
        "mode": args.mode,
        "model_path": model_path,
        "train_rows": train_rows,
        "eval_rows": [asdict(row) for row in eval_rows],
        "summary": summary,
    }
    if args.emit_benchmark_command:
        payload["benchmark_command"] = build_benchmark_command(args, model_path)

    out = output_root / f"validation_{int(time.time())}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.emit_benchmark_command:
        print("\nBenchmark handoff:")
        print(payload["benchmark_command"])
    if args.run_benchmark and summary.get("mean_predicted_score", 0) > summary.get("mean_dry_score", 0):
        cmd = payload.get("benchmark_command")
        if cmd:
            print("\n=> Running benchmark handoff command:", file=sys.stderr)
            print(cmd, file=sys.stderr)
            subprocess.run(cmd, shell=True, check=False)
        else:
            print("\n=> No benchmark_command available; skipping.", file=sys.stderr)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
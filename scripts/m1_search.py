#!/usr/bin/env python3
"""Autoresearch-style config search for Corollary M1.

Initial implementation focuses on M1B synthetic config search.
Appends one JSON object per run to results/m1_results.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from itertools import product, islice
from typing import Any, Iterable

RESULTS_PATH = os.path.join("results", "m1_results.jsonl")


def ensure_results_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w"):
            pass


def load_results(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_result(path: str, row: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def generate_m1a_configs(stimulus_filter: str | None = None) -> Iterable[dict[str, Any]]:
    base = {
        "stage": "M1A",
        "scenario": "prelinguistic_self_attribution",
        "iterations": 12,
        "shift_min_frames": 0,
        "shift_max_frames": 8,
        "room_rt60": 0.3,
        "stimulus_duration_ms": 300,
    }
    stimuli = ["tone", "click", "chirp", "vowel_ah", "vowel_ah_var", "vowel_ah_var_nearfield"] if not stimulus_filter else [stimulus_filter]
    grid = product(
        stimuli,
        [0, 7, 19],
        [0.4, 0.7, 0.9],
        [-40.0, -35.0, -30.0],
        [0.45, 0.55, 0.65],
        [0.05, 0.08, 0.12],
        [4, 8, 12],
        [0.8, 0.9],
        [20.0, 40.0, 60.0, 80.0],
        [0.4, 0.6, 0.8],
    )
    for stimulus, stimulus_seed, playback_amplitude, noise_db, attribution_threshold, event_threshold, onset_jitter_ms, replay_gain, causal_tolerance_ms, causal_weight in grid:
        yield {
            **base,
            "stimulus": stimulus,
            "stimulus_seed": stimulus_seed,
            "playback_amplitude": playback_amplitude,
            "noise_db": noise_db,
            "attribution_threshold": attribution_threshold,
            "event_threshold": event_threshold,
            "onset_jitter_ms": onset_jitter_ms,
            "transformed_world_gain": 0.85,
            "replay_gain": replay_gain,
            "hard_negative_noise_db_offset": 5.0,
            "hard_negative_amplitude_jitter": 0.15,
            "causal_tolerance_ms": causal_tolerance_ms,
            "causal_weight": causal_weight,
            "similarity_weight": round(1.0 - causal_weight + 0.2, 2),
        }


def generate_m1b_configs() -> Iterable[dict[str, Any]]:
    base = {
        "stage": "M1B",
        "scenario": "fixed_room_convergence",
        "iterations": 10,
        "shift_min_frames": -10,
        "shift_max_frames": 40,
        "room_rt60": 0.3,
    }
    grid = product(
        [0.05, 0.1, 0.25, 0.4],
        [100, 200, 300, 500],
        [-40.0, -35.0, -30.0],
        [0.4, 0.7, 0.9],
        ["tone", "click", "chirp"],
    )
    for nlms_mu, ir_length_ms, noise_db, playback_amplitude, stimulus in grid:
        yield {
            **base,
            "nlms_mu": nlms_mu,
            "ir_length_ms": ir_length_ms,
            "noise_db": noise_db,
            "playback_amplitude": playback_amplitude,
            "stimulus": stimulus,
            "stimulus_duration_ms": 300,
        }


def generate_m1c_configs(stimulus_filter: str | None = None) -> Iterable[dict[str, Any]]:
    base = {
        "stage": "M1C",
        "scenario": "mixed_source_overlap",
        "iterations": 12,
        "room_rt60": 0.3,
    }
    stimuli = ["chirp", "tone", "vowel_ah"] if not stimulus_filter else [stimulus_filter]
    grid = product(
        stimuli,
        [0.55, 0.6, 0.65],
        [-40.0, -35.0, -30.0],
        [0.6, 0.8, 0.9],
        [0.7, 0.85, 0.95],
    )
    for stimulus, attribution_threshold, noise_db, transformed_world_gain, replay_gain in grid:
        yield {
            **base,
            "stimulus": stimulus,
            "stimulus_duration_ms": 300,
            "playback_amplitude": 0.7,
            "attribution_threshold": attribution_threshold,
            "noise_db": noise_db,
            "transformed_world_gain": transformed_world_gain,
            "replay_gain": replay_gain,
        }


def best_score(rows: list[dict[str, Any]], stage: str) -> float:
    scores = [float(r.get("score", 0.0)) for r in rows if r.get("stage") == stage and r.get("decision") == "keep"]
    return max(scores) if scores else 0.0


def run_eval(script_path: str, config: dict[str, Any], timeout_s: float = 120.0) -> dict[str, Any]:
    args = [sys.executable, script_path]
    cli_map = {
        "stage": "--stage",
        "scenario": "--scenario",
        "iterations": "--iterations",
        "ir_length_ms": "--ir-length-ms",
        "nlms_mu": "--nlms-mu",
        "noise_db": "--noise-db",
        "playback_amplitude": "--playback-amplitude",
        "shift_min_frames": "--shift-min-frames",
        "shift_max_frames": "--shift-max-frames",
        "stimulus": "--stimulus",
        "stimulus_duration_ms": "--stimulus-duration-ms",
        "room_rt60": "--room-rt60",
        "attribution_threshold": "--attribution-threshold",
        "event_threshold": "--event-threshold",
        "onset_jitter_ms": "--onset-jitter-ms",
        "transformed_world_gain": "--transformed-world-gain",
        "replay_gain": "--replay-gain",
        "hard_negative_noise_db_offset": "--hard-negative-noise-db-offset",
        "hard_negative_amplitude_jitter": "--hard-negative-amplitude-jitter",
        "causal_tolerance_ms": "--causal-tolerance-ms",
        "causal_weight": "--causal-weight",
        "similarity_weight": "--similarity-weight",
        "stimulus_seed": "--stimulus-seed",
    }
    for key, flag in cli_map.items():
        if key in config:
            args.extend([flag, str(config[key])])

    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"eval failed with code {proc.returncode}")
    return json.loads(proc.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Corollary M1 config search")
    parser.add_argument("--stage", default="M1B")
    parser.add_argument("--stimulus", choices=["tone", "click", "chirp", "vowel_ah", "vowel_ah_var", "vowel_ah_var_nearfield"], default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--start-at", type=int, default=0)
    parser.add_argument("--results", default=RESULTS_PATH)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = args.stage.upper()
    if stage not in {"M1A", "M1B", "M1C"}:
        raise SystemExit("Current search scaffold implements M1A, M1B, and M1C.")

    ensure_results_file(args.results)
    existing = load_results(args.results)
    seen_hashes = {row.get("config_hash") for row in existing}
    current_best = best_score(existing, stage)

    script_path = os.path.join(os.path.dirname(__file__), "m1_eval_runner.py")
    if stage == "M1A":
        config_iter = generate_m1a_configs(args.stimulus)
    elif stage == "M1B":
        config_iter = generate_m1b_configs()
    else:
        config_iter = generate_m1c_configs(args.stimulus)
    configs = islice(config_iter, args.start_at, args.start_at + args.limit)

    for idx, config in enumerate(configs, start=args.start_at):
        cfg_hash = config_hash(config)
        if cfg_hash in seen_hashes:
            continue

        started = time.time()
        row: dict[str, Any] = {
            "run_id": f"{stage.lower()}-{idx:04d}",
            "config_hash": cfg_hash,
            "search_index": idx,
            "decision": "crash",
            "stage": stage,
            "config": config,
        }
        try:
            result = run_eval(script_path, config, timeout_s=args.timeout_s)
            score = float(result.get("score", 0.0))
            guardrails_passed = bool(result.get("guardrails_passed", False))
            decision = "keep" if guardrails_passed and score >= current_best else "discard"
            if decision == "keep":
                current_best = score

            row.update(result)
            row["decision"] = decision
        except Exception as e:
            row["error"] = str(e)

        row["logged_at"] = time.time()
        row["wall_time_s"] = round(row["logged_at"] - started, 4)
        append_result(args.results, row)
        seen_hashes.add(cfg_hash)

        primary = row.get("primary_metric", "score")
        metric_value = row.get("metrics", {}).get(primary, row.get("score", 0.0))
        print(f"[{row['decision']}] {row['run_id']} score={row.get('score', 0.0):.4f} {primary}={metric_value}")


if __name__ == "__main__":
    main()

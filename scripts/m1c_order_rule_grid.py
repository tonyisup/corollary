#!/usr/bin/env python3
"""Run an unattended M1c order-rule stability grid.

This collects paired sequential world-before-self and self-before-world trials
to test whether self_evidence_db is a stable order signal.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from m1c_hardware_harness import append_jsonl_record, infer_expected_label, run_mixed_trial


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL_OUT = REPO_ROOT / "data" / "m1c_calibration" / "order_grid_20260421.jsonl"

ORDER_MODES = ("sequential_world_first", "sequential_self_first")


@dataclass(frozen=True)
class TrialSpec:
    trial_index: int
    block: str
    mode: str
    self_stim: str
    world_stim: str
    seed: int
    gap_s: float
    amplitude: float
    calibration_split: str
    setup_label: str
    expected_label: str


def split_for_seed(seed: int) -> str:
    if 0 <= seed <= 5:
        return "train"
    if 6 <= seed <= 8:
        return "validation"
    return "test"


def iter_trial_specs(args: argparse.Namespace) -> Iterable[TrialSpec]:
    base_specs: list[TrialSpec] = []
    setup_label = args.setup_label

    for seed in args.primary_seeds:
        for world_stim in args.world_stims:
            for mode in ORDER_MODES:
                base_specs.append(
                    TrialSpec(
                        trial_index=0,
                        block="primary",
                        mode=mode,
                        self_stim=args.self_stim,
                        world_stim=world_stim,
                        seed=seed,
                        gap_s=args.primary_gap_s,
                        amplitude=args.primary_amplitude,
                        calibration_split=split_for_seed(seed),
                        setup_label=setup_label,
                        expected_label=infer_expected_label(mode),
                    )
                )

    stress_specs: list[TrialSpec] = []
    for seed in args.stress_seeds:
        for world_stim in args.world_stims:
            for gap_s in args.stress_gap_s:
                for amplitude in args.stress_amplitudes:
                    for mode in ORDER_MODES:
                        stress_specs.append(
                            TrialSpec(
                                trial_index=0,
                                block="stress",
                                mode=mode,
                                self_stim=args.self_stim,
                                world_stim=world_stim,
                                seed=seed,
                                gap_s=gap_s,
                                amplitude=amplitude,
                                calibration_split="test",
                                setup_label=f"{setup_label}_stress",
                                expected_label=infer_expected_label(mode),
                            )
                        )

    specs = base_specs + stress_specs
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(specs)

    if args.max_trials is not None:
        specs = specs[: args.max_trials]

    for idx, spec in enumerate(specs, start=1):
        yield TrialSpec(**{**asdict(spec), "trial_index": idx})


def parse_csv_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M1c provisional order-rule stability grid")
    parser.add_argument("--self-stim", default="babble_ba_var")
    parser.add_argument("--world-stims", type=parse_csv_strings, default=["chirp", "vowel_ah", "tone"])
    parser.add_argument("--primary-seeds", type=parse_csv_ints, default=list(range(12)))
    parser.add_argument("--stress-seeds", type=parse_csv_ints, default=list(range(12, 18)))
    parser.add_argument("--primary-gap-s", type=float, default=0.5)
    parser.add_argument("--primary-amplitude", type=float, default=0.7)
    parser.add_argument("--stress-gap-s", type=parse_csv_floats, default=[0.15, 1.0])
    parser.add_argument("--stress-amplitudes", type=parse_csv_floats, default=[0.55, 0.85])
    parser.add_argument("--duration-ms", type=int, default=350)
    parser.add_argument("--input-device", type=int, default=2)
    parser.add_argument("--output-device", type=int, default=4)
    parser.add_argument("--noise-device", type=int, default=3)
    parser.add_argument("--lead-in-s", type=float, default=0.5)
    parser.add_argument("--tail-s", type=float, default=1.0)
    parser.add_argument("--countdown-s", type=float, default=0.5, help="Reserved for future pacing control; currently informational only")
    parser.add_argument("--inter-trial-s", type=float, default=1.0)
    parser.add_argument("--setup-label", default="m1c_order_rule_grid")
    parser.add_argument("--session-label", default=None)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_JSONL_OUT)
    parser.add_argument("--shuffle-seed", type=int, default=20260421)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the randomized trial plan without touching audio devices")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_label = args.session_label or f"m1c-order-grid-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    specs = list(iter_trial_specs(args))

    print(f"Session: {session_label}")
    print(f"Trials: {len(specs)}")
    print(f"Output: {args.jsonl_out}")
    print(f"Dry run: {args.dry_run}")

    for spec in specs:
        print(
            f"{spec.trial_index:03d} {spec.block:<7} {spec.calibration_split:<10} "
            f"{spec.mode:<24} seed={spec.seed:<2} world={spec.world_stim:<8} "
            f"gap={spec.gap_s:.2f} amp={spec.amplitude:.2f}"
        )
        if args.dry_run:
            continue

        if args.countdown_s > 0:
            time.sleep(args.countdown_s)

        result = run_mixed_trial(
            mode=spec.mode,
            self_stim=spec.self_stim,
            world_stim=spec.world_stim,
            duration_ms=args.duration_ms,
            amplitude=spec.amplitude,
            input_device=args.input_device,
            output_device=args.output_device,
            noise_device=args.noise_device,
            gap_s=spec.gap_s,
            lead_in_s=args.lead_in_s,
            tail_s=args.tail_s,
            seed=spec.seed,
            setup_label=spec.setup_label,
            expected_label=spec.expected_label,
            calibration_split=spec.calibration_split,
            session_label=session_label,
        )
        result["experiment_block"] = spec.block
        result["seed"] = spec.seed
        result["trial_index"] = spec.trial_index
        append_jsonl_record(str(args.jsonl_out), result)
        print(json.dumps(result, sort_keys=True))
        if args.inter_trial_s > 0:
            time.sleep(args.inter_trial_s)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline evaluation for the first M1c calibration model slice."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m1c_calibration.baselines import (
    evaluate_rule,
    fit_best_threshold,
    fit_stimulus_aware_rule,
    reference_provisional_rule,
    reference_stimulus_rule,
)
from m1c_calibration.dataset import load_order_examples, resolve_splits
from m1c_calibration.metrics import grouped_metrics
from m1c_calibration.runner import (
    CalibrationTrainConfig,
    evaluate_bundle,
    load_checkpoint,
    save_checkpoint,
    train_calibration_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = REPO_ROOT / "data" / "m1c_calibration" / "order_grid_20260421.jsonl"


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def evaluate_rule_bundle(rule, rows):
    metrics = evaluate_rule(rule, rows)
    predictions = metrics.pop("predictions", [])
    metrics["by_world_stim"] = grouped_metrics(rows, predictions, group_key="world_stim")
    metrics["by_split"] = grouped_metrics(rows, predictions, group_key="calibration_split")
    return metrics


def summarize_split(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"{name:<11} acc={metrics.get('accuracy', 0.0):.3f} "
        f"bal_acc={metrics.get('balanced_accuracy', 0.0):.3f} "
        f"macro_f1={metrics.get('macro_f1', 0.0):.3f} "
        f"loss={metrics.get('loss', 0.0):.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation for M1c calibration")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dims", type=parse_int_tuple, default=parse_int_tuple("64,32"))
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Load an existing checkpoint instead of training")
    parser.add_argument("--save-checkpoint", type=Path, default=None, help="Write the trained bundle to this path")
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_order_examples(args.jsonl)
    if args.max_rows is not None:
        examples = examples[: args.max_rows]
    if not examples:
        raise SystemExit(f"No order-labeled rows found in {args.jsonl}")

    splits = resolve_splits(examples, seed=args.seed)
    train_rows = splits["train"] or examples
    validation_rows = splits["validation"] or train_rows
    test_rows = splits["test"] or validation_rows

    if args.checkpoint:
        bundle = load_checkpoint(args.checkpoint, map_location=args.device)
        training_history: list[dict[str, Any]] = bundle.history
    else:
        bundle = train_calibration_model(
            train_rows,
            validation_rows,
            config=CalibrationTrainConfig(
                hidden_dims=tuple(args.hidden_dims),
                latent_dim=args.latent_dim,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                min_delta=args.min_delta,
                seed=args.seed,
                device=args.device,
            ),
        )
        training_history = bundle.history
        if args.save_checkpoint:
            save_checkpoint(bundle, args.save_checkpoint)

    reference_provisional = reference_provisional_rule()
    reference_stimulus = reference_stimulus_rule()
    fitted_provisional = fit_best_threshold(train_rows, "self_evidence_db")
    fitted_stimulus = fit_stimulus_aware_rule(train_rows, feature_key="world_evidence_db")

    reference_provisional_eval = {
        split_name: evaluate_rule_bundle(reference_provisional, split_rows)
        for split_name, split_rows in [
            ("train", train_rows),
            ("validation", validation_rows),
            ("test", test_rows),
        ]
    }
    reference_stimulus_eval = {
        split_name: evaluate_rule_bundle(reference_stimulus, split_rows)
        for split_name, split_rows in [
            ("train", train_rows),
            ("validation", validation_rows),
            ("test", test_rows),
        ]
    }
    fitted_provisional_eval = {
        split_name: evaluate_rule_bundle(fitted_provisional, split_rows)
        for split_name, split_rows in [
            ("train", train_rows),
            ("validation", validation_rows),
            ("test", test_rows),
        ]
    }
    fitted_stimulus_eval = {
        split_name: evaluate_rule_bundle(fitted_stimulus, split_rows)
        for split_name, split_rows in [
            ("train", train_rows),
            ("validation", validation_rows),
            ("test", test_rows),
        ]
    }

    model_eval = {
        split_name: evaluate_bundle(bundle, split_rows)
        for split_name, split_rows in [
            ("train", train_rows),
            ("validation", validation_rows),
            ("test", test_rows),
        ]
    }
    model_eval["train"].pop("predictions", None)
    model_eval["validation"].pop("predictions", None)
    test_predictions = model_eval["test"].pop("predictions", [])
    model_eval["test_by_world_stim"] = grouped_metrics(
        test_rows,
        test_predictions,
        group_key="world_stim",
    )
    model_eval["test_by_split"] = grouped_metrics(
        test_rows,
        test_predictions,
        group_key="calibration_split",
    )

    summary = {
        "jsonl": str(args.jsonl),
        "count": len(examples),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "feature_dim": bundle.feature_builder.output_dim,
        "best_epoch": bundle.best_epoch,
        "training_history_tail": training_history[-5:],
        "baselines": {
            "reference_provisional": reference_provisional_eval,
            "reference_stimulus": reference_stimulus_eval,
            "fitted_provisional": fitted_provisional_eval,
            "fitted_stimulus": fitted_stimulus_eval,
        },
        "model": model_eval,
    }

    if not args.json_only:
        print("M1c Calibration Eval")
        print(f"Data: {args.jsonl}")
        print(f"Rows: {len(examples)}")
        print(
            "Splits: "
            f"train={len(train_rows)} validation={len(validation_rows)} test={len(test_rows)}"
        )
        print(f"Features: {bundle.feature_builder.output_dim}")
        print(f"Best epoch: {bundle.best_epoch}")
        print("")
        print("Reference provisional rule")
        print(summarize_split("train", reference_provisional_eval["train"]))
        print(summarize_split("validation", reference_provisional_eval["validation"]))
        print(summarize_split("test", reference_provisional_eval["test"]))
        print("")
        print("Reference stimulus rule")
        print(summarize_split("train", reference_stimulus_eval["train"]))
        print(summarize_split("validation", reference_stimulus_eval["validation"]))
        print(summarize_split("test", reference_stimulus_eval["test"]))
        print("")
        print("Neural model")
        print(summarize_split("train", model_eval["train"]))
        print(summarize_split("validation", model_eval["validation"]))
        print(summarize_split("test", model_eval["test"]))
        print("")

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

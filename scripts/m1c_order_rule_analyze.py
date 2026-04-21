#!/usr/bin/env python3
"""Analyze M1c order-rule calibration rows."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from m1c_hardware_harness import PROVISIONAL_ORDER_SELF_EVIDENCE_THRESHOLD_DB


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = REPO_ROOT / "data" / "m1c_calibration" / "order_grid_20260421.jsonl"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
    return rows


def predict_order(row: dict[str, Any], threshold: float) -> str:
    return "world_before_self" if float(row["self_evidence_db"]) <= threshold else "self_before_world"


def accuracy(rows: list[dict[str, Any]], threshold: float) -> tuple[int, int, float]:
    correct = sum(1 for row in rows if predict_order(row, threshold) == row.get("expected_label"))
    total = len(rows)
    return correct, total, correct / total if total else 0.0


def balanced_accuracy(rows: list[dict[str, Any]], threshold: float) -> float:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("expected_label"))].append(row)
    recalls = []
    for label in ("world_before_self", "self_before_world"):
        label_rows = by_label.get(label, [])
        if label_rows:
            recalls.append(sum(1 for row in label_rows if predict_order(row, threshold) == label) / len(label_rows))
    return statistics.mean(recalls) if recalls else 0.0


def paired_margins(rows: list[dict[str, Any]]) -> list[float]:
    pairs: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        label = row.get("expected_label")
        if label not in {"world_before_self", "self_before_world"}:
            continue
        key = (
            row.get("experiment_block"),
            row.get("calibration_split"),
            row.get("setup_label"),
            row.get("self_stim"),
            row.get("world_stim"),
            row.get("seed"),
            row.get("gap_s"),
            row.get("amplitude"),
        )
        pairs[key][label] = float(row["self_evidence_db"])

    margins = []
    for pair in pairs.values():
        if "self_before_world" in pair and "world_before_self" in pair:
            margins.append(pair["self_before_world"] - pair["world_before_self"])
    return margins


def print_group_metrics(rows: list[dict[str, Any]], threshold: float, group_key: str) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, "unknown"))].append(row)
    print(f"\nBy {group_key}:")
    for key in sorted(groups):
        correct, total, acc = accuracy(groups[key], threshold)
        bal = balanced_accuracy(groups[key], threshold)
        margins = paired_margins(groups[key])
        median_margin = statistics.median(margins) if margins else float("nan")
        positive_rate = sum(1 for margin in margins if margin > 0) / len(margins) if margins else 0.0
        print(
            f"  {key:<12} rows={total:<4} acc={acc:.3f} bal_acc={bal:.3f} "
            f"pairs={len(margins):<3} pos_pair_rate={positive_rate:.3f} median_margin_db={median_margin:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze M1c order-rule JSONL rows")
    parser.add_argument("jsonl", nargs="?", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--threshold", type=float, default=PROVISIONAL_ORDER_SELF_EVIDENCE_THRESHOLD_DB)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        row
        for row in load_rows(args.jsonl)
        if row.get("expected_label") in {"world_before_self", "self_before_world"}
    ]

    correct, total, acc = accuracy(rows, args.threshold)
    bal = balanced_accuracy(rows, args.threshold)
    margins = paired_margins(rows)
    positive_pairs = sum(1 for margin in margins if margin > 0)
    positive_rate = positive_pairs / len(margins) if margins else 0.0
    median_margin = statistics.median(margins) if margins else float("nan")
    mean_margin = statistics.mean(margins) if margins else float("nan")

    print(f"File: {args.jsonl}")
    print(f"Rows: {total}")
    print(f"Threshold: {args.threshold:.2f} dB")
    print(f"Accuracy: {correct}/{total} = {acc:.3f}")
    print(f"Balanced accuracy: {bal:.3f}")
    print(f"Matched pairs: {len(margins)}")
    print(f"Positive paired margins: {positive_pairs}/{len(margins)} = {positive_rate:.3f}")
    print(f"Median paired margin dB: {median_margin:.3f}")
    print(f"Mean paired margin dB: {mean_margin:.3f}")

    print_group_metrics(rows, args.threshold, "calibration_split")
    print_group_metrics(rows, args.threshold, "experiment_block")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze M1c order-rule calibration rows."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PROVISIONAL_ORDER_SELF_EVIDENCE_THRESHOLD_DB = 2.62


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


def search_best_single_threshold(
    rows: list[dict[str, Any]],
    feature_key: str,
    positive_when_leq: bool | None = None,
) -> dict[str, Any]:
    values = sorted({float(row[feature_key]) for row in rows})
    if not values:
        return {"feature": feature_key, "accuracy": 0.0, "threshold": None, "direction": None}

    candidates = [values[0] - 1e-6]
    candidates.extend((a + b) / 2.0 for a, b in zip(values, values[1:]))
    candidates.append(values[-1] + 1e-6)

    directions = ["<=", ">="] if positive_when_leq is None else (["<="] if positive_when_leq else [">="])
    best: dict[str, Any] | None = None
    for threshold in candidates:
        for direction in directions:
            correct = 0
            for row in rows:
                value = float(row[feature_key])
                pred = "world_before_self" if (value <= threshold if direction == "<=" else value >= threshold) else "self_before_world"
                correct += pred == row.get("expected_label")
            acc = correct / len(rows) if rows else 0.0
            candidate = {
                "feature": feature_key,
                "accuracy": acc,
                "threshold": threshold,
                "direction": direction,
                "correct": correct,
                "total": len(rows),
            }
            if best is None or candidate["accuracy"] > best["accuracy"]:
                best = candidate
    return best or {"feature": feature_key, "accuracy": 0.0, "threshold": None, "direction": None}


def search_best_two_feature_rule(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_configs = [
        ("self_evidence_db", "<=", True),
        ("world_evidence_db", "<=", True),
        ("world_score", ">=", False),
    ]

    def candidate_thresholds(feature_key: str) -> list[float]:
        values = sorted({float(row[feature_key]) for row in rows})
        if not values:
            return []
        thresholds = [values[0] - 1e-6]
        thresholds.extend((a + b) / 2.0 for a, b in zip(values, values[1:]))
        thresholds.append(values[-1] + 1e-6)
        return thresholds

    best: dict[str, Any] | None = None
    for feature_a, dir_a, _ in feature_configs:
        for feature_b, dir_b, _ in feature_configs:
            thresholds_a = candidate_thresholds(feature_a)
            thresholds_b = candidate_thresholds(feature_b)
            for threshold_a in thresholds_a:
                for threshold_b in thresholds_b:
                    correct = 0
                    for row in rows:
                        val_a = float(row[feature_a])
                        val_b = float(row[feature_b])
                        cond_a = val_a <= threshold_a if dir_a == "<=" else val_a >= threshold_a
                        cond_b = val_b <= threshold_b if dir_b == "<=" else val_b >= threshold_b
                        pred = "world_before_self" if (cond_a and cond_b) else "self_before_world"
                        correct += pred == row.get("expected_label")
                    acc = correct / len(rows) if rows else 0.0
                    candidate = {
                        "accuracy": acc,
                        "correct": correct,
                        "total": len(rows),
                        "feature_a": feature_a,
                        "direction_a": dir_a,
                        "threshold_a": threshold_a,
                        "feature_b": feature_b,
                        "direction_b": dir_b,
                        "threshold_b": threshold_b,
                    }
                    if best is None or candidate["accuracy"] > best["accuracy"]:
                        best = candidate
    return best or {"accuracy": 0.0, "correct": 0, "total": len(rows)}


def search_best_stimulus_aware_single_feature(rows: list[dict[str, Any]], feature_key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("world_stim", "unknown"))].append(row)

    thresholds: dict[str, dict[str, Any]] = {}
    correct = 0
    total = len(rows)
    for stim, stim_rows in sorted(groups.items()):
        best = search_best_single_threshold(stim_rows, feature_key)
        thresholds[stim] = best
        correct += best.get("correct", 0)

    return {
        "feature": feature_key,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "thresholds": thresholds,
    }


def search_best_stimulus_aware_two_feature_rule(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("world_stim", "unknown"))].append(row)

    per_stim_rules: dict[str, dict[str, Any]] = {}
    correct = 0
    total = len(rows)
    for stim, stim_rows in sorted(groups.items()):
        best = search_best_two_feature_rule(stim_rows)
        per_stim_rules[stim] = best
        correct += best.get("correct", 0)

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "per_stim_rules": per_stim_rules,
    }


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

    print("\nBest single-feature threshold rules:")
    for feature in ("self_evidence_db", "world_evidence_db", "world_score"):
        best = search_best_single_threshold(rows, feature)
        print(
            f"  {feature:<18} acc={best['accuracy']:.3f} "
            f"rule: world_before_self if {feature} {best['direction']} {best['threshold']:.3f}"
        )

    best_two = search_best_two_feature_rule(rows)
    print("\nBest two-feature AND rule:")
    print(
        "  "
        f"acc={best_two['accuracy']:.3f} "
        f"rule: world_before_self if {best_two['feature_a']} {best_two['direction_a']} {best_two['threshold_a']:.3f} "
        f"AND {best_two['feature_b']} {best_two['direction_b']} {best_two['threshold_b']:.3f}"
    )

    print("\nBest stimulus-aware single-feature rules:")
    best_stim_single: dict[str, dict[str, Any]] = {}
    for feature in ("self_evidence_db", "world_evidence_db", "world_score"):
        best = search_best_stimulus_aware_single_feature(rows, feature)
        best_stim_single[feature] = best
        print(f"  {feature:<18} acc={best['accuracy']:.3f}")
        for stim, rule in best["thresholds"].items():
            print(
                f"    {stim:<8} world_before_self if {feature} {rule['direction']} {rule['threshold']:.3f} "
                f"(acc={rule['accuracy']:.3f})"
            )

    best_stim_two = search_best_stimulus_aware_two_feature_rule(rows)
    print("\nBest stimulus-aware two-feature AND rules:")
    print(f"  overall_acc={best_stim_two['accuracy']:.3f}")
    for stim, rule in best_stim_two["per_stim_rules"].items():
        print(
            f"    {stim:<8} world_before_self if {rule['feature_a']} {rule['direction_a']} {rule['threshold_a']:.3f} "
            f"AND {rule['feature_b']} {rule['direction_b']} {rule['threshold_b']:.3f} "
            f"(acc={rule['accuracy']:.3f})"
        )


if __name__ == "__main__":
    main()

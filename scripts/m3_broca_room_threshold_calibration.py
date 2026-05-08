#!/usr/bin/env python3
"""Threshold calibration for Broca room-local self-present scoring."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from m3_broca_room_reference_analysis import apply_room_filter, fit_room_filter, local_score

DEFAULT_TEXT = "I was speaking, then paused."


def load_summary_observed(path: str | Path) -> tuple[Path, dict[str, Any], np.ndarray]:
    p = Path(path)
    return p, json.loads((p / "summary.json").read_text(encoding="utf-8")), np.load(p / "observed.npy")


def expected_shift(summary: dict[str, Any], fallback: int) -> int:
    value = summary.get("self_expected_shift_samples")
    return int(value) if value is not None else fallback


def fit_filters(positives: list[tuple[Path, dict[str, Any], np.ndarray]], template: np.ndarray, *, taps: int, ridge: float, search_ms: float) -> list[dict[str, Any]]:
    filters = []
    for p, summary, observed in positives:
        shift_hint = int(summary["self_expected_shift_samples"])
        _, dry_shift = local_score(observed, template, shift_hint, search_ms=search_ms)
        h = fit_room_filter(observed, template, dry_shift, taps=taps, ridge=ridge)
        filters.append({"label": p.name, "h": h})
    return filters


def score_artifact(
    artifact: tuple[Path, dict[str, Any], np.ndarray],
    filters: list[dict[str, Any]],
    template: np.ndarray,
    *,
    fallback_shift: int,
    search_ms: float,
) -> dict[str, Any]:
    p, summary, observed = artifact
    shift = expected_shift(summary, fallback_shift)
    dry_score, _ = local_score(observed, template, shift, search_ms=search_ms)
    room_scores = []
    for f in filters:
        room_ref = apply_room_filter(template, f["h"])
        room_score, _ = local_score(observed, room_ref, shift, search_ms=search_ms)
        room_scores.append({"filter": f["label"], "score": room_score})
    max_room = max((r["score"] for r in room_scores), default=0.0)
    mean_room = float(np.mean([r["score"] for r in room_scores])) if room_scores else 0.0
    return {
        "label": p.name,
        "path": str(p),
        "dry_score": dry_score,
        "max_room_score": max_room,
        "mean_room_score": mean_room,
        "room_scores": room_scores,
        "expected_shift": shift,
    }


def recommend_threshold(pos_scores: list[dict[str, Any]], neg_scores: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    """Pick a conservative operating threshold from observed score separation."""
    min_pos = min((r[score_key] for r in pos_scores), default=0.0)
    max_neg = max((r[score_key] for r in neg_scores), default=0.0)
    if pos_scores and neg_scores and min_pos > max_neg:
        threshold = (min_pos + max_neg) / 2.0
        reason = "midpoint_between_max_negative_and_min_positive"
    elif pos_scores:
        threshold = min_pos
        reason = "minimum_positive_score_no_clear_negative_gap"
    else:
        threshold = 1.0
        reason = "no_positive_scores"
    return {
        "threshold": threshold,
        "min_positive": min_pos,
        "max_negative": max_neg,
        "reason": reason,
    }


def threshold_rows(pos_scores: list[dict[str, Any]], neg_scores: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    scores = sorted({0.0, 1.0, *[r[score_key] for r in pos_scores], *[r[score_key] for r in neg_scores]})
    thresholds = []
    for score in scores:
        thresholds.extend([max(0.0, score - 1e-6), score, min(1.0, score + 1e-6)])
    rows = []
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    for threshold in sorted(set(thresholds)):
        tp = sum(1 for r in pos_scores if r[score_key] >= threshold)
        fn = n_pos - tp
        fp = sum(1 for r in neg_scores if r[score_key] >= threshold)
        tn = n_neg - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "threshold": threshold,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "f1": f1,
        })
    return sorted(rows, key=lambda r: (r["f1"], r["precision"], -r["fpr"], r["recall"]), reverse=True)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Calibrate operational threshold for Broca room-local self-present scoring")
    parser.add_argument("--positive", action="append", required=True, help="Positive/self artifact dir; repeatable")
    parser.add_argument("--negative", action="append", default=[], help="Negative/non-self artifact dir; repeatable")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--taps", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.000112)
    parser.add_argument("--search-ms", type=float, default=256.4)
    parser.add_argument("--score-key", choices=["max_room_score", "mean_room_score", "dry_score"], default="max_room_score")
    parser.add_argument("--output", default="data/m3_broca_room_threshold_calibration_20260430.json")
    args = parser.parse_args()

    template = await synthesize(args.text)
    positives = [load_summary_observed(p) for p in args.positive]
    negatives = [load_summary_observed(p) for p in args.negative]
    fallback_shift = int(np.median([s["self_expected_shift_samples"] for _, s, _ in positives]))
    filters = fit_filters(positives, template, taps=args.taps, ridge=args.ridge, search_ms=args.search_ms)

    # Export the fitted room filter for production use
    if filters:
        # Average the per-positive filters into a single production filter
        avg_filter = np.mean([f["h"] for f in filters], axis=0)
        filter_path = Path(args.output).with_suffix(".npy")
        np.save(filter_path, avg_filter.astype(np.float64))
        print(f"Saved room filter: {filter_path}")

    pos_scores = [score_artifact(p, filters, template, fallback_shift=fallback_shift, search_ms=args.search_ms) for p in positives]
    neg_scores = [score_artifact(n, filters, template, fallback_shift=fallback_shift, search_ms=args.search_ms) for n in negatives]
    ranked_thresholds = threshold_rows(pos_scores, neg_scores, args.score_key)
    best = ranked_thresholds[0] if ranked_thresholds else None
    recommended = recommend_threshold(pos_scores, neg_scores, args.score_key)
    out = {
        "params": {
            "taps": args.taps,
            "ridge": args.ridge,
            "search_ms": args.search_ms,
            "score_key": args.score_key,
            "fallback_shift": fallback_shift,
        },
        "positive_scores": pos_scores,
        "negative_scores": neg_scores,
        "best_threshold": best,
        "recommended_threshold": recommended,
        "thresholds": ranked_thresholds,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"positives={len(pos_scores)} negatives={len(neg_scores)} score_key={args.score_key}")
    print("positive scores:", [round(r[args.score_key], 3) for r in pos_scores])
    print("negative scores:", [round(r[args.score_key], 3) for r in neg_scores])
    print("best:", best)
    print("recommended:", recommended)
    print(f"Saved: {args.output}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

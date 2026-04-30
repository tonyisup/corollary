import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from m3_broca_room_threshold_calibration import recommend_threshold, threshold_rows


def test_threshold_rows_prefers_separating_threshold() -> None:
    positives = [
        {"max_room_score": 0.35},
        {"max_room_score": 0.40},
    ]
    negatives = [
        {"max_room_score": 0.05},
    ]

    best = threshold_rows(positives, negatives, "max_room_score")[0]

    assert best["tp"] == 2
    assert best["fp"] == 0
    assert best["fn"] == 0
    assert best["precision"] == 1.0
    assert best["recall"] == 1.0
    assert 0.05 < best["threshold"] <= 0.35


def test_recommend_threshold_uses_gap_midpoint() -> None:
    positives = [{"max_room_score": 0.35}, {"max_room_score": 0.40}]
    negatives = [{"max_room_score": 0.05}, {"max_room_score": 0.10}]

    rec = recommend_threshold(positives, negatives, "max_room_score")

    assert rec["threshold"] == pytest.approx(0.225)
    assert rec["reason"] == "midpoint_between_max_negative_and_min_positive"

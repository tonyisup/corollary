import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from cerebellum.adaptive_model import BabblingCerebellum
from m2_benchmark import (
    build_m4_detector_for_self_pcm,
    summarize_offline_results,
    strategy_definitions_for_args,
)


def test_summarize_offline_results_groups_by_strategy() -> None:
    results = [
        {
            "capture": "seed_0",
            "strategies": [
                {"name": "preserve_both", "raw_wer": 0.2, "gated_wer": 0.2, "wer_delta": 0.0},
                {"name": "gain_0p5", "raw_wer": 0.2, "gated_wer": 0.1, "wer_delta": -0.1},
            ],
        },
        {
            "capture": "seed_1",
            "strategies": [
                {"name": "preserve_both", "raw_wer": 0.4, "gated_wer": 0.5, "wer_delta": 0.1},
                {"name": "gain_0p5", "raw_wer": 0.4, "gated_wer": 0.6, "wer_delta": 0.2},
            ],
        },
    ]

    summary = summarize_offline_results(results)

    assert summary["preserve_both"]["trials"] == 2
    assert summary["preserve_both"]["improved"] == 0
    assert summary["preserve_both"]["neutral"] == 1
    assert summary["preserve_both"]["regressed"] == 1
    assert summary["preserve_both"]["mean_raw_wer"] == 0.30000000000000004
    assert summary["preserve_both"]["mean_gated_wer"] == 0.35
    assert summary["gain_0p5"]["improved"] == 1
    assert summary["gain_0p5"]["regressed"] == 1
    assert summary["gain_0p5"]["mean_wer_delta"] == 0.05


def test_strategy_definitions_without_cerebellum_model() -> None:
    class Args:
        cerebellum_model = None

    names = [row["name"] for row in strategy_definitions_for_args(Args())]

    assert "preserve" in names
    assert "gain_0p5" in names
    assert "subtract_0p25" in names
    assert not any(name.startswith("predicted_subtract_") for name in names)


def test_strategy_definitions_with_cerebellum_model() -> None:
    class Args:
        cerebellum_model = "/tmp/room_ir.json"

    names = [row["name"] for row in strategy_definitions_for_args(Args())]

    assert "predicted_subtract_0p25" in names
    assert "predicted_subtract_0p5" in names


def test_cerebellum_predict_pcm_returns_room_colored_signal() -> None:
    t = np.linspace(0, 0.25, 4000, dtype=np.float32)
    dry = np.sin(2 * np.pi * 220 * t).astype(np.float32)

    cerebellum = BabblingCerebellum(ir_length_ms=100)
    cerebellum.room_ir[:] = 0.0
    cerebellum.room_ir[0] = 1.0
    cerebellum.room_ir[32] = 0.4

    predicted = cerebellum.predict_pcm(dry)

    assert predicted.ndim == 1
    assert len(predicted) > len(dry)
    assert not np.allclose(predicted[: len(dry)], dry, atol=1e-3)


def test_build_m4_detector_uses_mock_and_reports_room_colored_mode(tmp_path) -> None:
    t = np.linspace(0, 0.25, 4000, dtype=np.float32)
    dry = np.sin(2 * np.pi * 220 * t).astype(np.float32)

    cerebellum = BabblingCerebellum(ir_length_ms=100)
    cerebellum.room_ir[:] = 0.0
    cerebellum.room_ir[0] = 1.0
    cerebellum.room_ir[32] = 0.4
    model_path = tmp_path / "room_ir.json"
    cerebellum.save_model(str(model_path))

    class Args:
        m4_enable = True
        cerebellum_model = str(model_path)
        m4_mock = True
        m4_registration_threshold = 0.4
        m4_self_threshold = 0.4

    detector, info = build_m4_detector_for_self_pcm(dry, Args())

    assert detector is not None
    assert info["enabled"] is True
    assert info["registration_mode"] == "room_colored_prediction"
    assert info["mock_embeddings"] is True

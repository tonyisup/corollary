import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from cerebellum.adaptive_model import BabblingCerebellum
from m1_tone_attribution import generate_stimulus_with_metadata
from m4_forward_model_validation import (
    build_benchmark_command,
    compare_templates_against_capture,
    synthetic_capture_from_model,
)


def test_build_benchmark_command_includes_forward_model_and_m4() -> None:
    class Args:
        benchmark_mode = "benchmark"
        stimulus_kind = "vowel_ah_var_nearfield"
        input_device = 3
        output_device = 5
        noise_device = 4
        benchmark_self_amplitude = 0.2
        benchmark_self_duration_ms = 350
        benchmark_similarity_threshold = 0.20
        benchmark_onset_latency_ms = 400.0
        benchmark_onset_method = "energy"
        benchmark_causal_tolerance_ms = 75.0
        benchmark_capture_root = None
        m4_mock = True

    cmd = build_benchmark_command(Args(), "packages/data/room_ir_test.json")

    assert "scripts/m2_benchmark.py" in cmd
    assert "--cerebellum-model" in cmd
    assert "packages/data/room_ir_test.json" in cmd
    assert "--m4-enable" in cmd
    assert "--m4-mock" in cmd


def test_predicted_template_beats_dry_on_synthetic_capture() -> None:
    cerebellum = BabblingCerebellum(ir_length_ms=100)
    cerebellum.room_ir[:] = 0.0
    cerebellum.room_ir[0] = 0.9
    cerebellum.room_ir[24] = 0.35
    cerebellum.room_ir[60] = 0.15

    dry = generate_stimulus_with_metadata("vowel_ah_var_nearfield", 350, 0.2, seed=3)["pcm"]
    captured = synthetic_capture_from_model(dry, cerebellum, noise_std=0.0)
    predicted = cerebellum.predict_pcm(dry)

    cmp = compare_templates_against_capture(captured, dry, predicted, search_ms=250.0)

    assert cmp["predicted"]["score"] >= cmp["dry"]["score"]
    assert cmp["predicted"]["onset_error_samples"] <= cmp["dry"]["onset_error_samples"]

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from m2_benchmark import summarize_offline_results


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

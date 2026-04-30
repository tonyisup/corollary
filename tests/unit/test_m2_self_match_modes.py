import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from m2_speech_preservation_harness import apply_room_filter, fit_room_filter, local_xcorr


def test_local_xcorr_scores_expected_window() -> None:
    template = np.sin(np.linspace(0, np.pi * 8, 400)).astype(np.float32)
    observed = np.zeros(2000, dtype=np.float32)
    observed[800:1200] = template

    score, shift = local_xcorr(observed, template, expected_shift=790, search_ms=50)

    assert score > 0.99
    assert abs(shift - 800) < 20


def test_room_filter_can_improve_colored_template_score() -> None:
    template = np.sin(np.linspace(0, np.pi * 8, 400)).astype(np.float32)
    h_true = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    colored = np.convolve(template, h_true, mode="full").astype(np.float32)
    observed = np.zeros(2000, dtype=np.float32)
    observed[800:800 + len(colored)] = colored

    dry_score, dry_shift = local_xcorr(observed, template, expected_shift=800, search_ms=50)
    h_fit = fit_room_filter(observed, template, dry_shift, taps=8, ridge=1e-4)
    room_ref = apply_room_filter(template, h_fit)
    room_score, _ = local_xcorr(observed, room_ref, expected_shift=800, search_ms=50)

    assert room_score >= dry_score
    assert room_score > 0.99

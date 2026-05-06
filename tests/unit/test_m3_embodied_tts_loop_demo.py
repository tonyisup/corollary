import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from m3_embodied_tts_loop_demo import apply_broca_room_local_operating_defaults



def test_broca_room_local_operating_defaults_apply_recommended_values() -> None:
    args = SimpleNamespace(
        broca_room_local_operating_point=True,
        similarity_threshold=None,
        onset_latency_ms=None,
        onset_method=None,
        causal_tolerance_ms=None,
        self_match_mode=None,
        self_match_search_ms=None,
        room_filter_taps=None,
        room_filter_ridge=None,
    )

    apply_broca_room_local_operating_defaults(args)

    assert args.similarity_threshold == 0.05
    assert args.onset_latency_ms == 300.0
    assert args.onset_method == "score"
    assert args.causal_tolerance_ms == 175.0
    assert args.self_match_mode == "room_local_oracle"
    assert args.self_match_search_ms == 256.4
    assert args.room_filter_taps == 64
    assert args.room_filter_ridge == 0.000112



def test_broca_room_local_operating_defaults_preserve_explicit_overrides() -> None:
    args = SimpleNamespace(
        broca_room_local_operating_point=True,
        similarity_threshold=0.1,
        onset_latency_ms=250.0,
        onset_method="energy",
        causal_tolerance_ms=150.0,
        self_match_mode="local_xcorr",
        self_match_search_ms=111.0,
        room_filter_taps=32,
        room_filter_ridge=0.01,
    )

    apply_broca_room_local_operating_defaults(args)

    assert args.similarity_threshold == 0.1
    assert args.onset_latency_ms == 250.0
    assert args.onset_method == "energy"
    assert args.causal_tolerance_ms == 150.0
    assert args.self_match_mode == "local_xcorr"
    assert args.self_match_search_ms == 111.0
    assert args.room_filter_taps == 32
    assert args.room_filter_ridge == 0.01



def test_standard_defaults_apply_when_operating_point_disabled() -> None:
    args = SimpleNamespace(
        broca_room_local_operating_point=False,
        similarity_threshold=None,
        onset_latency_ms=None,
        onset_method=None,
        causal_tolerance_ms=None,
        self_match_mode=None,
        self_match_search_ms=None,
        room_filter_taps=None,
        room_filter_ridge=None,
    )

    apply_broca_room_local_operating_defaults(args)

    assert args.similarity_threshold == 0.20
    assert args.onset_latency_ms == 400.0
    assert args.onset_method == "energy"
    assert args.causal_tolerance_ms == 75.0
    assert args.self_match_mode == "global_xcorr"
    assert args.self_match_search_ms == 250.0
    assert args.room_filter_taps == 64
    assert args.room_filter_ridge == 1e-4

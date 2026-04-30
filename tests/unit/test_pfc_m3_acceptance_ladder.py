from scripts.m3_acceptance_ladder import run_ladder


def test_m3_acceptance_ladder_passes_current_slice() -> None:
    rows = run_ladder(
        "data/m2_baseline/m2_energy_forward_tol75_matrix_20260429_093302/self_only_seed0",
        "data/m2_policy/frame_timeline_validation_20260429_140648/vowel_seed0",
    )

    assert [row["name"] for row in rows] == [
        "defer_on_agentive_overlap",
        "speak_on_self_only",
        "interrupt_active_speech_on_overlap",
        "artifact_backed_interrupt",
    ]
    assert all(row["passed"] for row in rows)

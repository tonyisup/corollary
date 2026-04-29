from scripts.m3_artifact_ladder import replay


def test_ladder_clean_vowel_overlap_defers_and_resumes() -> None:
    row = replay(
        "clean_vowel_overlap",
        "data/m2_policy/family_policy_validation_20260429_130530/vowel_ah_var_nearfield_seed0",
        "hello",
    )
    assert row["accepted_initial_output"] is False
    assert row["resumed_output"] == "hello"
    assert "LISTEN" in row["states"]


def test_ladder_self_only_control_does_not_defer() -> None:
    row = replay(
        "self_only_control",
        "data/m2_baseline/m2_energy_forward_tol75_matrix_20260429_093302/self_only_seed0",
        "hello",
    )
    assert row["accepted_initial_output"] is True
    assert row["resumed_output"] is None
    assert "LISTEN" not in row["states"]

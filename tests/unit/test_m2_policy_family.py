from m2_policy.policy import select_family_policy, stimulus_family


def test_stimulus_family_vowel_nearfield() -> None:
    assert stimulus_family("vowel_ah_var_nearfield") == "vowel_ah_var_nearfield"
    assert stimulus_family("vowel_ah_var_nearfield_room2") == "vowel_ah_var_nearfield"


def test_stimulus_family_repeated_babble() -> None:
    assert stimulus_family("babble_ba_repeat_short_tight_var") == "repeated_babble"


def test_family_policy_promotes_vowel_gain_half() -> None:
    decision = select_family_policy("vowel_ah_var_nearfield")
    assert decision.mode == "gain"
    assert decision.gain == 0.5
    assert decision.reason == "vowel_ah_var_nearfield_promoted_gain_0p5"


def test_family_policy_preserves_repeated_babble() -> None:
    decision = select_family_policy("babble_ba_repeat_short_tight_var")
    assert decision.mode == "gain"
    assert decision.gain == 1.0
    assert decision.reason == "repeated_babble_preserve_both"


def test_family_policy_preserves_unknown() -> None:
    decision = select_family_policy("chirp")
    assert decision.mode == "gain"
    assert decision.gain == 1.0
    assert decision.reason == "unknown_preserve_both"

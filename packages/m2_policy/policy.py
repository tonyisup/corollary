"""Deterministic M2 BOTH-frame policy selection.

This encodes the first validated M2 policy split:
- vowel-family controls can safely use conservative BOTH attenuation
- repeated babble remains a stress condition and preserves BOTH
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BothPolicyDecision:
    """Concrete BOTH-frame processing settings."""

    mode: str = "gain"
    gain: float = 1.0
    subtract_gain: float = 0.0
    reason: str = "preserve_both_default"


def stimulus_family(self_stim: str) -> str:
    """Return the coarse M2 policy family for a stimulus name."""
    if self_stim.startswith("vowel_ah_var_nearfield"):
        return "vowel_ah_var_nearfield"
    if self_stim.startswith("babble_ba_repeat"):
        return "repeated_babble"
    if self_stim.startswith("babble_"):
        return "babble"
    if self_stim.startswith("vowel_"):
        return "vowel"
    return "unknown"


def select_family_policy(
    self_stim: str,
    *,
    family_gain_overrides: Mapping[str, float] | None = None,
) -> BothPolicyDecision:
    """Select a conservative M2 BOTH-frame policy from stimulus family.

    Current evidence supports `gain=0.5` for `vowel_ah_var_nearfield` only.
    Repeated babble and unknown stimuli preserve BOTH because held-out policy
    selection regressed on repeated-babble stress captures.

    `family_gain_overrides` allows controlled experiments to keep the same
    family-policy structure while swapping the concrete gain used for a family.
    """
    family = stimulus_family(self_stim)
    default_gain = 0.5 if family == "vowel_ah_var_nearfield" else 1.0
    gain = default_gain
    if family_gain_overrides and family in family_gain_overrides:
        gain = float(family_gain_overrides[family])

    if family == "vowel_ah_var_nearfield":
        reason = "vowel_ah_var_nearfield_promoted_gain_0p5"
    else:
        reason = f"{family}_preserve_both"

    if gain != default_gain:
        reason = f"{reason}_override_gain_{gain:.3f}"

    return BothPolicyDecision(
        mode="gain",
        gain=gain,
        subtract_gain=0.0,
        reason=reason,
    )

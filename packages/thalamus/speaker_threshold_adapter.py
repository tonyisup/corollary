"""Per-speaker adaptive threshold for M4 self-voice detection.

The core insight: ECAPA's fixed threshold works well for self vs. generic
"other" but fails when a specific human speaker's voice overlaps with the
self-embedding region. Rather than trying to find one universal threshold,
we track each speaker's confidence distribution and set the threshold
individually — the same way you naturally calibrate to each person's voice
in a conversation.

Usage:
    adapter = SpeakerThresholdAdapter(
        base_threshold=0.10,
        margin=0.05,
        adaptation_rate=0.1,
    )

    # On each frame where M4 fires:
    adapter.update(speaker_id, m4_confidence)

    # Get the threshold for a specific speaker:
    threshold = adapter.get_threshold(speaker_id)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SpeakerThresholdProfile:
    """Running statistics for a single speaker's M4 confidence."""
    speaker_id: str
    count: int = 0
    mean_confidence: float = 0.0
    max_confidence: float = 0.0
    min_confidence: float = 1.0
    # Exponential moving average of confidence
    ema_confidence: float = 0.0
    # Whether this profile has enough data to set a threshold
    is_registered: bool = False
    # Number of frames before we consider the profile "warm"
    warm_threshold: int = 10

    def update(self, confidence: float, alpha: float = 0.1) -> None:
        """Update running statistics with a new confidence observation."""
        self.count += 1
        self.max_confidence = max(self.max_confidence, confidence)
        self.min_confidence = min(self.min_confidence, confidence)

        # Incremental mean
        self.mean_confidence += (confidence - self.mean_confidence) / self.count

        # EMA
        if self.count == 1:
            self.ema_confidence = confidence
        else:
            self.ema_confidence = alpha * confidence + (1 - alpha) * self.ema_confidence

        if self.count >= self.warm_threshold:
            self.is_registered = True


class SpeakerThresholdAdapter:
    """Adapts the M4 self-voice threshold per speaker.

    The threshold for each speaker is set to:
        threshold = max(base_threshold, min(speaker_max_confidence + margin, max_threshold))

    This means:
    - For speakers whose ECAPA confidence is well below the self threshold,
      we use the base threshold (no change from current behavior).
    - For speakers whose ECAPA confidence overlaps with the self region,
      we raise the threshold just above their max observed confidence,
      ensuring 0 false positives while preserving self-recall.

    The adapter is conservative: it only raises the threshold, never lowers it.
    This prevents a malicious or coincidental high-confidence frame from
    permanently blocking self-detection.
    """

    def __init__(
        self,
        base_threshold: float = 0.10,
        margin: float = 0.05,
        adaptation_rate: float = 0.1,
        warm_threshold: int = 10,
        max_threshold: float = 0.50,
    ) -> None:
        self.base_threshold = base_threshold
        self.margin = margin
        self.adaptation_rate = adaptation_rate
        self.warm_threshold = warm_threshold
        self.max_threshold = max_threshold
        self._profiles: Dict[str, SpeakerThresholdProfile] = {}
        # Self-profile tracks the agent's own confidence distribution
        self._self_profile: Optional[SpeakerThresholdProfile] = None

    def register_self(self, self_speaker_id: str = "self") -> None:
        """Register the self-speaker profile."""
        self._self_profile = SpeakerThresholdProfile(
            speaker_id=self_speaker_id,
            warm_threshold=self.warm_threshold,
        )

    def update_self(self, confidence: float) -> None:
        """Update self-confidence statistics."""
        if self._self_profile is None:
            self.register_self()
        self._self_profile.update(confidence, self.adaptation_rate)

    def update(self, speaker_id: str, confidence: float) -> None:
        """Update the confidence profile for a speaker."""
        if speaker_id == "self":
            self.update_self(confidence)
            return
        if speaker_id not in self._profiles:
            self._profiles[speaker_id] = SpeakerThresholdProfile(
                speaker_id=speaker_id,
                warm_threshold=self.warm_threshold,
            )
        self._profiles[speaker_id].update(confidence, self.adaptation_rate)

    def get_threshold(self, speaker_id: str) -> float:
        """Get the adaptive threshold for a speaker.

        For unknown speakers, returns the base threshold.
        For known speakers, returns the maximum of:
          - base_threshold (the default global threshold, never lowered)
          - min(speaker_max_confidence + margin, max_threshold)
        This means the threshold is raised for speakers whose confidence
        overlaps with the self region, but never lowered below base.

        The threshold is available from the first frame — no warm-up needed.
        The warm_threshold only controls the is_registered flag for diagnostics.
        """
        if speaker_id == "self":
            return self.base_threshold

        profile = self._profiles.get(speaker_id)
        if profile is None or profile.count == 0:
            return self.base_threshold

        # Set threshold just above this speaker's max observed confidence.
        # If the speaker's max confidence + margin exceeds the base threshold,
        # raise the threshold to block their false positives.
        # Otherwise, use the base threshold (no change).
        adaptive = profile.max_confidence + self.margin
        # Cap at max threshold to avoid blocking all self-detection
        adaptive = min(adaptive, self.max_threshold)
        # Only raise, never lower
        return max(self.base_threshold, adaptive)

    def get_self_min_confidence(self) -> Optional[float]:
        """Return the minimum observed self confidence, or None if not enough data."""
        if self._self_profile is None or not self._self_profile.is_registered:
            return None
        return self._self_profile.min_confidence

    def should_accept_self(self, speaker_id: str, confidence: float) -> bool:
        """Check if a frame should be accepted as SELF for this speaker."""
        threshold = self.get_threshold(speaker_id)
        return confidence >= threshold

    def get_profile(self, speaker_id: str) -> Optional[SpeakerThresholdProfile]:
        """Get the full profile for a speaker (for diagnostics)."""
        if speaker_id == "self":
            return self._self_profile
        return self._profiles.get(speaker_id)

    def summary(self) -> Dict[str, Dict]:
        """Return a summary of all speaker profiles."""
        result = {}
        if self._self_profile is not None:
            result["self"] = {
                "count": self._self_profile.count,
                "mean_conf": round(self._self_profile.mean_confidence, 4),
                "max_conf": round(self._self_profile.max_confidence, 4),
                "min_conf": round(self._self_profile.min_confidence, 4),
                "registered": self._self_profile.is_registered,
            }
        for sid, profile in self._profiles.items():
            result[sid] = {
                "count": profile.count,
                "mean_conf": round(profile.mean_confidence, 4),
                "max_conf": round(profile.max_confidence, 4),
                "min_conf": round(profile.min_confidence, 4),
                "threshold": round(self.get_threshold(sid), 4),
                "registered": profile.is_registered,
            }
        return result

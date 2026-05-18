"""Cerebellum — Drift Alarm (M7 Phase 2).

Detects when the room forward model is degrading and signals the
cerebellum to either boost its learning rate or trigger a rebabble.

The alarm tracks a rolling window of prediction errors. When the
recent average degrades beyond a configurable threshold relative to
the baseline, it raises a drift signal. Sustained drift eventually
triggers a rebabble recommendation.

This is the "SelfModel drift alarm" from the M7 phased plan:
  Phase 1: Trust-gated NLMS (adapt on SELF only, hold on BOTH/WORLD)
  Phase 2: SelfModel drift alarm (spike adaptation rate when match degrades)
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class DriftStatus(enum.Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    STALE = "stale"


@dataclass
class DriftAlarm:
    """Monitors prediction-error trends to detect room-model drift.

    The alarm maintains a rolling window of recent prediction errors
    and compares them against a baseline (the mean error during a
    stable period). When the recent average degrades beyond
    ``spike_threshold_db``, the alarm enters DEGRADED status and
    signals that the NLMS learning rate should be boosted.

    If the alarm stays degraded for ``rebabble_after_spikes`` consecutive
    spike detections, it enters STALE status and recommends a full
    rebabble cycle.
    """

    window_frames: int = 20
    spike_threshold_db: float = 6.0
    rebabble_after_spikes: int = 5
    max_rate_multiplier: float = 4.0
    # Internal state
    _errors: list = field(default_factory=list, repr=False)
    _baseline_error_db: float = 0.0
    _baseline_set: bool = False
    _spike_count: int = 0
    _status: DriftStatus = DriftStatus.NOMINAL

    def record_error(self, error_db: float) -> None:
        """Record a new prediction-error observation (in dB)."""
        self._errors.append(float(error_db))
        if len(self._errors) > self.window_frames * 2:
            self._errors = self._errors[-self.window_frames:]

        if not self._baseline_set and len(self._errors) >= self.window_frames:
            self._baseline_error_db = float(np.mean(self._errors[-self.window_frames:]))
            self._baseline_set = True
            return

        if not self._baseline_set:
            return

        recent = self._errors[-self.window_frames:]
        recent_avg = float(np.mean(recent))
        degradation = recent_avg - self._baseline_error_db

        if degradation >= self.spike_threshold_db:
            self._spike_count += 1
            if self._spike_count >= self.rebabble_after_spikes:
                self._status = DriftStatus.STALE
            else:
                self._status = DriftStatus.DEGRADED
        else:
            self._spike_count = 0
            self._status = DriftStatus.NOMINAL
            # Slowly update baseline during stable periods
            self._baseline_error_db = 0.9 * self._baseline_error_db + 0.1 * recent_avg

    def reset_baseline(self) -> None:
        """Reset baseline tracking for a new phase (e.g., after room change).

        Clears accumulated errors and marks baseline as not set, so the
        next record_error() calls will establish a fresh baseline.
        """
        self._errors.clear()
        self._baseline_set = False
        self._baseline_error_db = 0.0
        self._spike_count = 0
        self._status = DriftStatus.NOMINAL

    @property
    def status(self) -> DriftStatus:
        return self._status

    @property
    def should_spike_adaptation(self) -> bool:
        return self._status in (DriftStatus.DEGRADED, DriftStatus.STALE)

    @property
    def should_rebabble(self) -> bool:
        return self._status == DriftStatus.STALE

    @property
    def adaptation_rate_multiplier(self) -> float:
        """Return a multiplier for the NLMS learning rate.

        1.0 = nominal, up to ``max_rate_multiplier`` when stale.
        """
        if self._status == DriftStatus.NOMINAL:
            return 1.0
        if self._status == DriftStatus.STALE:
            return self.max_rate_multiplier
        # DEGRADED: scale linearly with spike count
        fraction = self._spike_count / max(self.rebabble_after_spikes, 1)
        return 1.0 + (self.max_rate_multiplier - 1.0) * min(fraction, 1.0)

    def to_dict(self) -> dict:
        return {
            "status": self._status.value,
            "window_frames": self.window_frames,
            "spike_threshold_db": self.spike_threshold_db,
            "rebabble_after_spikes": self.rebabble_after_spikes,
            "max_rate_multiplier": self.max_rate_multiplier,
            "spike_count": self._spike_count,
            "baseline_error_db": round(self._baseline_error_db, 2),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DriftAlarm":
        alarm = cls(
            window_frames=d.get("window_frames", 20),
            spike_threshold_db=d.get("spike_threshold_db", 6.0),
            rebabble_after_spikes=d.get("rebabble_after_spikes", 5),
            max_rate_multiplier=d.get("max_rate_multiplier", 4.0),
        )
        alarm._spike_count = d.get("spike_count", 0)
        alarm._baseline_error_db = d.get("baseline_error_db", 0.0)
        alarm._baseline_set = True
        status_str = d.get("status", "nominal")
        alarm._status = DriftStatus(status_str)
        return alarm

"""Cerebellum — Mobile Device-Profile Layer (M7 Phase 4).

Models how the device's acoustic signature changes with orientation
(portrait, landscape, pocket, face-down, on-table).

Each orientation gets its own IR profile. The bank switches between
profiles based on orientation sensor input. When no sensor data is
available, falls back to a single "unknown" profile.
"""

from __future__ import annotations

import enum
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import fftconvolve

from cerebellum.trust_model import TrustTracker, TrustState, DEFAULT_IR_DIR


class DeviceOrientation(enum.Enum):
    """Canonical device orientations."""
    UNKNOWN = "unknown"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    POCKET = "pocket"
    FACE_DOWN = "face_down"
    ON_TABLE = "on_table"

    @classmethod
    def from_string(cls, s: str) -> "DeviceOrientation":
        try:
            return cls(s.lower())
        except ValueError:
            return cls.UNKNOWN


class OrientationSensor:
    """Abstract orientation sensor interface.

    In production, this would read from accelerometer/gyroscope.
    For testing and desktop use, accepts mock orientations.
    """

    def __init__(self, mock_orientation: DeviceOrientation = DeviceOrientation.UNKNOWN):
        self._mock_orientation = mock_orientation
        self._stable_count = 0

    def current_orientation(self) -> DeviceOrientation:
        """Return the current device orientation."""
        return self._mock_orientation

    def set_mock_orientation(self, orientation: DeviceOrientation) -> None:
        """Change the mock orientation (for testing)."""
        if orientation != self._mock_orientation:
            self._mock_orientation = orientation
        self._stable_count += 1

    @property
    def orientation_stable_frames(self) -> int:
        """Number of consecutive frames with the same orientation."""
        return self._stable_count


@dataclass
class DeviceProfile:
    """A single orientation-dependent acoustic profile."""

    orientation: DeviceOrientation
    ir_length: int = 1600
    nlms_mu: float = 0.05
    nlms_epsilon: float = 1e-6
    ir: np.ndarray = field(init=False)
    trust: TrustTracker = field(init=False)
    guess_count: int = 0
    _errors: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.ir = np.zeros(self.ir_length, dtype=np.float32)
        self.ir[0] = 1.0
        self.trust = TrustTracker()

    @property
    def mean_error(self) -> float:
        if not self._errors:
            return 0.0
        return float(np.mean(self._errors[-20:]))

    def guess_ir(self, **kwargs) -> None:
        """Seed the IR with an orientation-specific prior."""
        self.ir = np.zeros(self.ir_length, dtype=np.float32)
        self.ir[0] = 1.0
        self.guess_count += 1

        if self.orientation == DeviceOrientation.PORTRAIT:
            if len(self.ir) > 16:
                self.ir[16] = 0.3
            if len(self.ir) > 48:
                self.ir[48] = 0.1
        elif self.orientation == DeviceOrientation.LANDSCAPE:
            if len(self.ir) > 32:
                self.ir[32] = 0.35
            if len(self.ir) > 80:
                self.ir[80] = 0.15
        elif self.orientation == DeviceOrientation.POCKET:
            if len(self.ir) > 24:
                self.ir[24] = 0.15
            if len(self.ir) > 96:
                self.ir[96] = 0.05
        elif self.orientation == DeviceOrientation.FACE_DOWN:
            if len(self.ir) > 48:
                self.ir[48] = 0.08
        elif self.orientation == DeviceOrientation.ON_TABLE:
            if len(self.ir) > 64:
                self.ir[64] = 0.2
            if len(self.ir) > 128:
                self.ir[128] = 0.1
            if len(self.ir) > 256:
                self.ir[256] = 0.05

    def adapt(self, known: np.ndarray, captured: np.ndarray) -> float:
        """Adapt this profile's IR using gradient-based NLMS."""
        predicted = fftconvolve(known, self.ir, mode="full").astype(np.float32)
        n = min(len(predicted), len(captured))
        if n == 0:
            return 0.0
        error = captured[:n] - predicted[:n]
        rms = float(np.sqrt(np.mean(error ** 2)))
        self._errors.append(rms)

        N = self.ir_length
        mu = self.nlms_mu
        max_len = min(len(known), len(captured))
        if max_len == 0:
            return rms

        grad = np.correlate(error[:max_len], known[:max_len], mode="full")
        grad = grad[max_len - 1: max_len - 1 + N].astype(np.float32)
        power = float(np.mean(known[:max_len] ** 2)) + self.nlms_epsilon
        max_grad = 1.0 / power
        grad = np.clip(grad, -max_grad, max_grad)
        self.ir += mu * grad

        return rms

    def to_dict(self) -> dict:
        return {
            "orientation": self.orientation.value,
            "ir_length": self.ir_length,
            "nlms_mu": self.nlms_mu,
            "ir": self.ir.tolist(),
            "guess_count": self.guess_count,
            "mean_error": self.mean_error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceProfile":
        prof = cls(
            orientation=DeviceOrientation.from_string(d.get("orientation", "unknown")),
            ir_length=d.get("ir_length", 1600),
            nlms_mu=d.get("nlms_mu", 0.05),
        )
        prof.ir = np.array(d["ir"], dtype=np.float32)
        prof.guess_count = d.get("guess_count", 0)
        return prof


class DeviceProfileBank:
    """Manages per-orientation device profiles."""

    def __init__(
        self,
        ir_length: int = 1600,
        nlms_mu: float = 0.05,
        max_profiles: int = 10,
        sensor: Optional[OrientationSensor] = None,
    ):
        self.ir_length = ir_length
        self.nlms_mu = nlms_mu
        self.max_profiles = max_profiles
        self.sensor = sensor or OrientationSensor()
        self._profiles: Dict[DeviceOrientation, DeviceProfile] = {}
        self._active_orientation = DeviceOrientation.UNKNOWN

        self._profiles[DeviceOrientation.UNKNOWN] = DeviceProfile(
            DeviceOrientation.UNKNOWN, ir_length, nlms_mu
        )

    @property
    def active_orientation(self) -> DeviceOrientation:
        return self._active_orientation

    @property
    def num_profiles(self) -> int:
        return len(self._profiles)

    def get_profile(self, orientation: DeviceOrientation) -> Optional[DeviceProfile]:
        return self._profiles.get(orientation)

    def get_active_profile(self) -> DeviceProfile:
        return self._profiles[self._active_orientation]

    def get_active_ir(self) -> np.ndarray:
        return self.get_active_profile().ir

    def update_orientation(self, orientation: Optional[DeviceOrientation] = None) -> None:
        """Update the active orientation (from sensor or explicit)."""
        if orientation is None:
            orientation = self.sensor.current_orientation()

        if orientation == self._active_orientation:
            return

        self._active_orientation = orientation

        if orientation not in self._profiles:
            if len(self._profiles) >= self.max_profiles:
                self._prune_one()
            prof = DeviceProfile(orientation, self.ir_length, self.nlms_mu)
            prof.guess_ir()
            self._profiles[orientation] = prof

    def _prune_one(self) -> None:
        """Remove the worst non-unknown profile."""
        worst_orientation = None
        worst_error = float("inf")
        for orient, prof in self._profiles.items():
            if orient == DeviceOrientation.UNKNOWN:
                continue
            if prof.mean_error < worst_error:
                worst_error = prof.mean_error
                worst_orientation = orient
        if worst_orientation is not None:
            del self._profiles[worst_orientation]

    def adapt(self, known: np.ndarray, captured: np.ndarray) -> float:
        """Adapt the active profile. Returns RMS error."""
        self.update_orientation()
        return self.get_active_profile().adapt(known, captured)

    def to_dict(self) -> dict:
        return {
            "ir_length": self.ir_length,
            "nlms_mu": self.nlms_mu,
            "max_profiles": self.max_profiles,
            "active_orientation": self._active_orientation.value,
            "profiles": {o.value: p.to_dict() for o, p in self._profiles.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceProfileBank":
        bank = cls(
            ir_length=d.get("ir_length", 1600),
            nlms_mu=d.get("nlms_mu", 0.05),
            max_profiles=d.get("max_profiles", 10),
        )
        bank._active_orientation = DeviceOrientation.from_string(
            d.get("active_orientation", "unknown")
        )
        bank._profiles = {}
        for orient_str, prof_dict in d.get("profiles", {}).items():
            orient = DeviceOrientation.from_string(orient_str)
            bank._profiles[orient] = DeviceProfile.from_dict(prof_dict)
        return bank

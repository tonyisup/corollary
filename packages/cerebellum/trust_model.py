"""Cerebellum — Trust Model.

The cerebellum doesn't just predict what the cochlea will hear.  It also
tracks whether its predictions are still valid — i.e. whether the room
model it learned is still accurate.

Key insight from Tony: parents don't teach the baby *how* to hear the
room. They validate the baby's predictions by confirming them.
"I predicted X, parent responded with X" → trust grows.
"I predicted X, world gave Y" → trust decays → time to babble again.

Trust is the history of "I predicted, the world confirmed, I was right."
When that history breaks, the cerebellum knows its model is outdated
and should re-enter babbling mode.

This module implements:
- **Prediction confidence tracker** — exponential moving average of how
  well predictions match reality
- **Trust decay** — confidence slowly drops without confirmation
- **Rebabble trigger** — when trust falls below threshold
- **IR persistence** — save/load learned room IR to disk

The cerebellum uses this to decide:
- "My model is good → use it for prediction and self-voice suppression"
- "My model is questionable → babble a little to test it"
- "My model is broken → full rebabble cycle"
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# ── Trust Configuration ────────────────────────────────────────────────

# Default: trust decays by ~1% per second without confirmation
DEFAULT_DECAY_RATE = 0.001

# Below 30% trust → trigger rebabble
DEFAULT_REBABBLE_THRESHOLD = 0.30

# Above 90% trust → model is considered reliable
DEFAULT_CONFIDENCE_THRESHOLD = 0.90

# How many bad predictions in a row before marking model as stale
DEFAULT_STREAK_LIMIT = 5

# How much to update trust per confirmation (exponential moving average)
DEFAULT_ALPHA = 0.1  # 10% new, 90% old


@dataclass
class TrustState:
    """Serializable snapshot of the cerebellum's trust in its model."""

    # Current trust level (0.0 = no trust, 1.0 = full confidence)
    trust: float = 0.5

    # Running average of prediction error (lower = better predictions)
    avg_error_db: float = 0.0

    # Number of successful prediction-confirmations
    confirmations: int = 0

    # Number of prediction failures
    failures: int = 0

    # Consecutive bad predictions (resets on confirmation)
    failure_streak: int = 0

    # When the model was last confirmed as valid
    last_confirmed: Optional[float] = None

    # When the model was created / last fully relearned
    model_born: float = 0.0

    # How many times the model has been rebabbled
    rebabble_count: int = 0

    # Timestamp of last update
    last_updated: float = 0.0

    # Metadata: which IR file this trust state belongs to
    ir_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrustState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TrustTracker:
    """Tracks the cerebellum's trust in its forward model.

    Trust grows when predictions are confirmed by reality,
    and decays when predictions fail or time passes without
    confirmation.
    """

    def __init__(
        self,
        decay_rate: float = DEFAULT_DECAY_RATE,
        alpha: float = DEFAULT_ALPHA,
        rebabble_threshold: float = DEFAULT_REBABBLE_THRESHOLD,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        streak_limit: int = DEFAULT_STREAK_LIMIT,
        initial_trust: float = 0.5,
    ) -> None:
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.rebabble_threshold = rebabble_threshold
        self.confidence_threshold = confidence_threshold
        self.streak_limit = streak_limit

        self.state = TrustState(
            trust=initial_trust,
            model_born=time.perf_counter(),
            last_updated=time.perf_counter(),
        )

    def confirm(self, prediction_error_db: float) -> str:
        """Record a successful prediction (error within tolerance).

        Parameters
        ----------
        prediction_error_db : float
            The prediction error in dB for this observation.
            Lower is better.

        Returns
        -------
        str
            "confirmed" — model validated, trust grows
            "stable" — already high trust, minor update
            "recovering" — was low trust, now improving
            "warning" — model might be degrading
            "stale" — too many consecutive failures, rebabble needed
        """
        now = time.perf_counter()

        # Track error (exponential moving average)
        if self.state.avg_error_db == 0.0:
            self.state.avg_error_db = prediction_error_db
        else:
            self.state.avg_error_db = (
                self.alpha * prediction_error_db
                + (1 - self.alpha) * self.state.avg_error_db
            )

        self.state.confirmations += 1
        self.state.failure_streak = 0
        self.state.last_confirmed = now

        # Determine action
        if self.state.trust < 0.2:
            self.state.trust = min(1.0, self.state.trust + 0.15)
            action = "recovering"
        elif self.state.trust < self.confidence_threshold:
            self.state.trust = min(1.0, self.state.trust + 0.05)
            action = "confirmed"
        else:
            self.state.trust = min(1.0, self.state.trust + 0.01)
            action = "stable"

        self.state.last_updated = now
        return action

    def decay(self, seconds_elapsed: Optional[float] = None) -> None:
        """Decay trust over time without confirmation.

        Parameters
        ----------
        seconds_elapsed : float, optional
            Time since last update. If None, computes from last_updated.
        """
        if seconds_elapsed is None:
            seconds_elapsed = time.perf_counter() - self.state.last_updated

        self.state.trust *= np.exp(-self.decay_rate * seconds_elapsed)
        self.state.trust = max(0.0, self.state.trust)
        self.state.last_updated = time.perf_counter()

    def fail(self) -> str:
        """Record a prediction failure.

        Returns
        -------
        str
            "ok" — within acceptable streak
            "warning" — streak getting long
            "stale" — rebabble needed
        """
        self.state.failures += 1
        self.state.failure_streak += 1

        if self.state.failure_streak >= self.streak_limit:
            self.state.trust *= 0.5  # Halve trust on streak
            return "stale"

        self.state.trust = max(0.0, self.state.trust - 0.05)
        self.state.last_updated = time.perf_counter()

        if self.state.failure_streak >= self.streak_limit // 2:
            return "warning"

        return "ok"

    def should_rebabble(self) -> bool:
        """Check if the cerebellum should re-enter babbling mode."""
        return self.state.trust < self.rebabble_threshold

    def get_status(self) -> dict:
        """Get current trust status for logging."""
        return {
            "trust": round(self.state.trust, 3),
            "avg_error_db": round(self.state.avg_error_db, 2),
            "confirmations": self.state.confirmations,
            "failures": self.state.failures,
            "failure_streak": self.state.failure_streak,
            "status": (
                "reliable"
                if self.state.trust >= self.confidence_threshold
                else "questionable"
                if self.state.trust >= self.rebabble_threshold
                else "stale — needs rebabble"
            ),
            "decayed": True if self.state.trust < 0.3 else False,
        }


# ── IR Persistence ──────────────────────────────────────────────────────

DEFAULT_IR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
)


def save_ir(
    ir: np.ndarray,
    trust: TrustTracker,
    path: str = "",
    metadata: Optional[dict] = None,
) -> str:
    """Save learned room IR and trust state to disk.

    Parameters
    ----------
    ir : np.ndarray
        The learned room impulse response.
    trust : TrustTracker
        Current trust state.
    path : str, optional
        File path. If empty, uses default location with timestamp.
    metadata : dict, optional
        Additional metadata to save with the model.

    Returns
    -------
    str
        Path to saved file.
    """
    os.makedirs(os.path.dirname(path) or DEFAULT_IR_DIR, exist_ok=True)

    data = {
        "ir": ir.tolist(),
        "state": trust.state.to_dict(),
        "metadata": metadata or {},
    }

    if not path:
        ts = int(trust.state.model_born * 1000)
        path = os.path.join(DEFAULT_IR_DIR, f"room_ir_{ts}.json")

    with open(path, "w") as f:
        json.dump(data, f)

    trust.state.ir_file = path
    return path


def load_ir(
    path: str,
) -> tuple[np.ndarray, TrustTracker]:
    """Load a saved room IR and trust state.

    Parameters
    ----------
    path : str
        Path to saved IR file.

    Returns
    -------
    tuple
        (room_ir, trust_tracker) — loaded IR and trust state.
    """
    with open(path, "r") as f:
        data = json.load(f)

    ir = np.array(data["ir"], dtype=np.float32)
    trust = TrustTracker()
    trust.state = TrustState.from_dict(data["state"])

    return ir, trust


def list_saved_models(directory: str = DEFAULT_IR_DIR) -> list[dict]:
    """List all saved room models.

    Parameters
    ----------
    directory : str
        Directory containing saved models.

    Returns
    -------
    list[dict]
        List of model metadata dicts.
    """
    if not os.path.isdir(directory):
        return []

    models = []
    for fname in os.listdir(directory):
        if fname.endswith(".json"):
            path = os.path.join(directory, fname)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                state = data.get("state", {})
                models.append(
                    {
                        "file": fname,
                        "path": path,
                        "trust": state.get("trust", 0),
                        "confirmations": state.get("confirmations", 0),
                        "ir_length": len(data.get("ir", [])),
                        "last_updated": state.get("last_updated", 0),
                    }
                )
            except Exception:
                continue

    return sorted(models, key=lambda m: m.get("trust", 0), reverse=True)

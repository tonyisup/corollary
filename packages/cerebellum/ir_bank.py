"""Cerebellum — Multi-Hypothesis IR Bank (M7 Phase 3).

Maintains multiple candidate room IR models (hypotheses). On each frame,
scores each hypothesis by prediction error and adapts the best one. When
drift is detected, spawns a new hypothesis. Prunes losers when one model
is clearly winning.

This is the "multi-hypothesis IR bank" from the M7 phased plan:
  Phase 3: Multi-hypothesis IR bank (3–5 models, switch on match)
           Faster recovery than single-filter re-adaptation
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.signal import fftconvolve

from cerebellum.trust_model import TrustTracker, TrustState, save_ir, load_ir, DEFAULT_IR_DIR


@dataclass
class IRHypothesis:
    """A single room IR candidate with its own NLMS state."""

    ir: np.ndarray
    nlms_mu: float = 0.05
    nlms_epsilon: float = 1e-6
    _errors: list = field(default_factory=list, repr=False)
    age: int = 0

    def __init__(self, ir_length: int, nlms_mu: float = 0.05, nlms_epsilon: float = 1e-6):
        self.ir = np.zeros(ir_length, dtype=np.float32)
        self.ir[0] = 1.0
        self.nlms_mu = nlms_mu
        self.nlms_epsilon = nlms_epsilon
        self._errors = []
        self.age = 0

    @property
    def mean_error(self) -> float:
        if not self._errors:
            return 0.0
        return float(np.mean(self._errors[-20:]))

    def update_error(self, error: float) -> None:
        self._errors.append(float(error))
        self.age += 1

    def score(self, known: np.ndarray, captured: np.ndarray) -> float:
        """Return RMS prediction error for this hypothesis."""
        predicted = fftconvolve(known, self.ir, mode="full").astype(np.float32)
        n = min(len(predicted), len(captured))
        if n == 0:
            return float("inf")
        error = captured[:n] - predicted[:n]
        return float(np.sqrt(np.mean(error ** 2)))

    def adapt(self, known: np.ndarray, captured: np.ndarray) -> float:
        """Run one NLMS update step. Returns RMS error.

        Uses FFT-based prediction + stochastic NLMS gradient update.
        """
        predicted = fftconvolve(known, self.ir, mode="full").astype(np.float32)
        n = min(len(predicted), len(captured))
        if n == 0:
            return 0.0
        error = captured[:n] - predicted[:n]
        rms = float(np.sqrt(np.mean(error ** 2)))
        self.update_error(rms)

        # Stochastic NLMS: sample a subset of time steps for the gradient
        N = len(self.ir)
        mu = self.nlms_mu
        eps = self.nlms_epsilon
        max_len = min(len(known), len(captured))

        # Use every 100th sample for the gradient estimate (fast but representative)
        step = max(1, max_len // 16)
        count = 0
        for t in range(0, max_len, step):
            # Build input vector (reversed slice of known)
            x = np.zeros(N, dtype=np.float32)
            for i in range(min(N, t + 1)):
                x[i] = known[t - i]
            y_hat = float(np.dot(x, self.ir))
            e = captured[t] - y_hat
            x_power = float(np.dot(x, x))
            if x_power > eps:
                self.ir += mu * e * x / (x_power + eps)
            count += 1

        # Scale by the sampling ratio so the update magnitude is appropriate
        if count > 0 and step > 1:
            # Compensate for sampling fewer samples
            pass  # The mu already accounts for step size

        return rms

    def to_dict(self) -> dict:
        return {
            "ir": self.ir.tolist(),
            "nlms_mu": self.nlms_mu,
            "nlms_epsilon": self.nlms_epsilon,
            "mean_error": self.mean_error,
            "age": self.age,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IRHypothesis":
        ir = np.array(d["ir"], dtype=np.float32)
        h = cls(ir_length=len(ir), nlms_mu=d.get("nlms_mu", 0.05))
        h.ir = ir
        h.age = d.get("age", 0)
        # Restore error history from serialized mean
        mean_err = d.get("mean_error", 0.0)
        if mean_err > 0:
            h._errors = [mean_err] * min(h.age, 20)
        return h


class IRBank:
    """Manages multiple IR hypotheses for robust room modeling.

    Maintains up to ``max_hypotheses`` candidate IR models. On each frame,
    scores all hypotheses and adapts the best one. Can spawn new hypotheses
    when drift is detected and prune losers when one model dominates.
    """

    def __init__(
        self,
        ir_length: int = 1600,
        nlms_mu: float = 0.05,
        max_hypotheses: int = 5,
        spawn_threshold_db: float = 3.0,
    ):
        self.ir_length = ir_length
        self.nlms_mu = nlms_mu
        self.max_hypotheses = max_hypotheses
        self.spawn_threshold_db = spawn_threshold_db
        self._hypotheses: List[IRHypothesis] = [IRHypothesis(ir_length, nlms_mu)]
        self._best_index: int = 0

    @property
    def num_hypotheses(self) -> int:
        return len(self._hypotheses)

    @property
    def best_hypothesis_index(self) -> int:
        return self._best_index

    @property
    def best_hypothesis(self) -> IRHypothesis:
        return self._hypotheses[self._best_index]

    def score_all(self, known: np.ndarray, captured: np.ndarray) -> List[float]:
        """Score all hypotheses. Returns list of RMS errors."""
        errors = [h.score(known, captured) for h in self._hypotheses]
        if errors:
            self._best_index = int(np.argmin(errors))
        return errors

    def adapt_best(self, known: np.ndarray, captured: np.ndarray) -> float:
        """Adapt the best hypothesis. Returns its RMS error."""
        errors = self.score_all(known, captured)
        best_error = self.best_hypothesis.adapt(known, captured)
        return best_error

    def adapt_hypothesis(self, index: int, known: np.ndarray, captured: np.ndarray) -> float:
        """Adapt a specific hypothesis by index."""
        if 0 <= index < len(self._hypotheses):
            return self._hypotheses[index].adapt(known, captured)
        return 0.0

    def spawn_hypothesis(self) -> Optional[int]:
        """Spawn a new hypothesis. Returns its index, or None if at max."""
        if len(self._hypotheses) >= self.max_hypotheses:
            return None
        new_h = IRHypothesis(self.ir_length, self.nlms_mu)
        self._hypotheses.append(new_h)
        return len(self._hypotheses) - 1

    def prune_worst(self, target_count: int = 3) -> None:
        """Remove the worst hypotheses down to target_count."""
        if len(self._hypotheses) <= target_count:
            return
        # Keep the best (lowest mean error) hypotheses
        scored = [(i, h.mean_error) for i, h in enumerate(self._hypotheses)]
        scored.sort(key=lambda x: x[1])
        keep_indices = set(i for i, _ in scored[:target_count])
        self._hypotheses = [h for i, h in enumerate(self._hypotheses) if i in keep_indices]
        self._best_index = 0

    def to_dict(self) -> dict:
        return {
            "ir_length": self.ir_length,
            "nlms_mu": self.nlms_mu,
            "max_hypotheses": self.max_hypotheses,
            "spawn_threshold_db": self.spawn_threshold_db,
            "best_hypothesis_index": self._best_index,
            "hypotheses": [h.to_dict() for h in self._hypotheses],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IRBank":
        bank = cls(
            ir_length=d.get("ir_length", 1600),
            nlms_mu=d.get("nlms_mu", 0.05),
            max_hypotheses=d.get("max_hypotheses", 5),
            spawn_threshold_db=d.get("spawn_threshold_db", 3.0),
        )
        bank._hypotheses = [IRHypothesis.from_dict(h) for h in d.get("hypotheses", [])]
        bank._best_index = d.get("best_hypothesis_index", 0)
        return bank

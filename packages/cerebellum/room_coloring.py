"""Helpers for room-colored self prediction and alignment.

These utilities bridge the existing cerebellar forward model into downstream
M2/M4 workflows:
- generate a room-colored self template from a dry efference copy
- estimate the internal onset of that predicted template
- align the template so its onset lands on the measured mic onset
"""

from __future__ import annotations

from functools import lru_cache
from typing import Union

import numpy as np

from cerebellum.adaptive_model import BabblingCerebellum


def estimate_onset_samples(pcm: np.ndarray, threshold_ratio: float = 0.20) -> int:
    """Estimate onset as first sample crossing a fraction of the peak amplitude."""
    if pcm.size == 0:
        return 0
    peak = float(np.max(np.abs(pcm)))
    if peak < 1e-9:
        return 0
    threshold = peak * float(np.clip(threshold_ratio, 0.0, 1.0))
    indices = np.flatnonzero(np.abs(pcm) >= threshold)
    if len(indices) == 0:
        return 0
    return int(indices[0])


@lru_cache(maxsize=8)
def load_cerebellum_model(path: str) -> BabblingCerebellum:
    return BabblingCerebellum.load_model(path)


def predict_room_colored_self(
    dry_self_pcm: np.ndarray,
    cerebellum_or_path: Union[BabblingCerebellum, str],
) -> np.ndarray:
    """Generate a room-colored self prediction from dry self PCM."""
    cerebellum = (
        load_cerebellum_model(cerebellum_or_path)
        if isinstance(cerebellum_or_path, str)
        else cerebellum_or_path
    )
    return cerebellum.predict_pcm(np.asarray(dry_self_pcm, dtype=np.float32))


def align_template_to_observed_onset(
    template_pcm: np.ndarray,
    total_samples: int,
    observed_onset_samples: int,
    template_onset_samples: int | None = None,
) -> np.ndarray:
    """Place a template into a track so its internal onset matches the observation."""
    track = np.zeros(int(total_samples), dtype=np.float32)
    template = np.asarray(template_pcm, dtype=np.float32)
    if template_onset_samples is None:
        template_onset_samples = estimate_onset_samples(template)

    start = int(observed_onset_samples) - int(template_onset_samples)
    src_start = 0
    dst_start = start
    if dst_start < 0:
        src_start = -dst_start
        dst_start = 0

    n = min(len(template) - src_start, len(track) - dst_start)
    if n > 0:
        track[dst_start:dst_start + n] = template[src_start:src_start + n]
    return track

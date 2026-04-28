"""Dual-input M1C evidence fusion.

The M1C hardware result showed that single-input attribution is insufficient
for robust self/world/overlap decisions. Geometry calibration gives two
meaningful paths: a self-favored input and a world-favored input. This module
turns per-channel evidence into one source verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from .attributor import SourceVerdict


@dataclass(frozen=True)
class ChannelEvidence:
    """Evidence extracted from one input channel for one validation window."""

    self_score: float
    world_score: float
    self_evidence_db: float
    world_evidence_db: float


@dataclass(frozen=True)
class DualInputFusionConfig:
    """Thresholds for geometry-aware dual-input source fusion."""

    self_score_threshold: float = 0.20
    self_evidence_threshold_db: float = 6.0
    world_evidence_threshold_db: float = 3.0
    world_score_threshold: float = 0.12
    world_score_evidence_floor_db: float = 1.5
    world_favored_score_threshold: float = 0.44


@dataclass(frozen=True)
class DualInputFusionResult:
    verdict: SourceVerdict
    self_present: bool
    world_present: bool
    world_evidence_max_db: float
    world_score_max: float
    world_favored_score: float


def fuse_dual_input_evidence(
    channels: list[ChannelEvidence],
    *,
    mode: str = "mixed",
    config: DualInputFusionConfig | None = None,
) -> DualInputFusionResult:
    """Fuse self-favored and world-favored mic evidence into a verdict.

    Parameters
    ----------
    channels:
        Two channel records. Index 0 is expected to be the self-favored input;
        index 1 is expected to be the world-favored input.
    mode:
        Optional semantic mode. ``replay_control`` suppresses self-present even
        when replayed self-like audio matches the self template, because there
        is no motor command.
    config:
        Tunable thresholds.
    """
    if len(channels) != 2:
        raise ValueError("dual-input fusion requires exactly two channels")

    cfg = config or DualInputFusionConfig()
    self_channel, world_channel = channels

    self_present = bool(
        self_channel.self_score >= cfg.self_score_threshold
        and self_channel.self_evidence_db >= cfg.self_evidence_threshold_db
    )

    world_evidence_max = max(ch.world_evidence_db for ch in channels)
    world_score_max = max(ch.world_score for ch in channels)
    world_favored_score = world_channel.world_score
    world_present = bool(
        world_evidence_max >= cfg.world_evidence_threshold_db
        or (world_score_max >= cfg.world_score_threshold and world_evidence_max >= cfg.world_score_evidence_floor_db)
        or world_favored_score >= cfg.world_favored_score_threshold
    )

    if mode == "replay_control":
        self_present = False

    if self_present and world_present:
        verdict = SourceVerdict.BOTH
    elif self_present:
        verdict = SourceVerdict.SELF
    elif world_present:
        verdict = SourceVerdict.WORLD
    else:
        verdict = SourceVerdict.SILENCE

    return DualInputFusionResult(
        verdict=verdict,
        self_present=self_present,
        world_present=world_present,
        world_evidence_max_db=world_evidence_max,
        world_score_max=world_score_max,
        world_favored_score=world_favored_score,
    )

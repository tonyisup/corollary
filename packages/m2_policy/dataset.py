"""Dataset helpers for M2 BOTH-frame policy learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class M2PolicyRow:
    trial_dir: str
    seed: int
    strategy: str
    raw_wer: float
    gated_wer: float
    wer_delta: float
    self_score: float
    speech_ratio: float
    self_frames: int
    both_frames: int
    world_frames: int
    self_onset_error_ms: float

    @property
    def safe(self) -> bool:
        return self.wer_delta <= 0.0

    @property
    def improves(self) -> bool:
        return self.wer_delta < 0.0

    @property
    def features(self) -> dict[str, float]:
        total = max(self.self_frames + self.both_frames + self.world_frames, 1)
        return {
            "self_score": self.self_score,
            "speech_ratio": self.speech_ratio,
            "self_onset_abs_ms": abs(self.self_onset_error_ms),
            "self_frame_ratio": self.self_frames / total,
            "both_frame_ratio": self.both_frames / total,
            "world_frame_ratio": self.world_frames / total,
            "raw_wer": self.raw_wer,
        }


def _row_from_mapping(row: dict[str, Any]) -> M2PolicyRow:
    return M2PolicyRow(
        trial_dir=str(row.get("trial_dir", "")),
        seed=int(row.get("seed", 0)),
        strategy=str(row.get("strategy", "unknown")),
        raw_wer=float(row.get("raw_wer", 0.0)),
        gated_wer=float(row.get("gated_wer", 0.0)),
        wer_delta=float(row.get("wer_delta", 0.0)),
        self_score=float(row.get("self_score", 0.0)),
        speech_ratio=float(row.get("speech_ratio", 0.0)),
        self_frames=int(row.get("self_frames", 0)),
        both_frames=int(row.get("both_frames", 0)),
        world_frames=int(row.get("world_frames", 0)),
        self_onset_error_ms=float(row.get("self_onset_error_ms", 0.0)),
    )


def load_offline_summary(path: str | Path) -> list[M2PolicyRow]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_row_from_mapping(row) for row in data.get("rows", [])]


def load_many(paths: Iterable[str | Path]) -> list[M2PolicyRow]:
    rows: list[M2PolicyRow] = []
    for path in paths:
        rows.extend(load_offline_summary(path))
    return rows

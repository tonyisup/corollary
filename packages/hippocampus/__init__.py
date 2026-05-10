"""Hippocampus — episodic memory encoding, storage, and retrieval.

M5: Multi-Speaker World Model + Episodic Memory + SAG-Grounded Learning.

The hippocampus stores salient interaction turns as episodic traces.
Single events are retrievable but do *not* directly update the SAG.
Offline consolidation (separate process) extracts repeated patterns
from accumulated episodic traces and writes stable nodes/edges into the SAG.

Persistence: JSON-based event log with append-only writes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class EpisodicTrace:
    """A single interaction turn stored in episodic memory."""
    trace_id: str
    timestamp_s: float
    session_id: str

    # Attribution summary
    self_frames: int = 0
    world_frames: int = 0
    both_frames: int = 0
    silence_frames: int = 0

    # M4 outcomes
    m4_accepted_frames: int = 0
    m4_attempted_frames: int = 0
    m4_outcomes: dict = field(default_factory=dict)  # outcome -> count

    # Interaction events
    interrupt_events: int = 0
    defer_events: int = 0
    resume_events: int = 0
    collision_repairs: int = 0

    # Speaker info
    known_speakers: list = field(default_factory=list)
    attribution_confidence: float = 0.0

    # Salience
    salience_score: float = 0.0

    # Raw metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodicTrace":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ConsolidationCandidate:
    """A pattern extracted from multiple episodic traces."""
    pattern_id: str
    description: str
    trace_count: int
    first_seen_s: float
    last_seen_s: float
    confidence: float
    evidence: list = field(default_factory=list)


class Hippocampus:
    """Episodic memory store with persistence and retrieval.

    Stores interaction turns as episodic traces in a JSONL event log.
    Supports retrieval by time range, speaker, and salience.
    Provides consolidation candidates for offline SAG updates.
    """

    def __init__(
        self,
        *,
        persist_path: Optional[Path | str] = None,
        max_traces_in_memory: int = 10000,
        salience_threshold: float = 0.1,
    ) -> None:
        self._traces: list[EpisodicTrace] = []
        self._persist_path = Path(persist_path) if persist_path else None
        self._max_traces = max_traces_in_memory
        self._salience_threshold = salience_threshold
        self._trace_counter = 0
        self._session_counter = 0

        # Load existing traces if persistence path exists
        if self._persist_path and self._persist_path.exists():
            self._load()

    # -- Encoding ---------------------------------------------------------

    def encode(
        self,
        *,
        session_id: str | None = None,
        frame_log: list[dict] | None = None,
        attribution_results: list[dict] | None = None,
        m4_outcomes: dict | None = None,
        interaction_events: dict | None = None,
        known_speakers: list | None = None,
        metadata: dict | None = None,
    ) -> EpisodicTrace:
        """Encode an interaction turn into an episodic trace.

        Parameters
        ----------
        session_id : str, optional
            Session identifier. Auto-generated if not provided.
        frame_log : list[dict], optional
            Frame-level attribution results with 'verdict' keys.
        attribution_results : list[dict], optional
            Per-frame attribution metadata.
        m4_outcomes : dict, optional
            M4 outcome counts (outcome_name -> count).
        interaction_events : dict, optional
            Interaction event counts (interrupts, defers, etc.).
        known_speakers : list, optional
            List of known speaker identifiers.
        metadata : dict, optional
            Additional metadata to store.
        """
        self._trace_counter += 1
        if session_id is None:
            self._session_counter += 1
            session_id = f"session_{self._session_counter:04d}"

        trace = EpisodicTrace(
            trace_id=f"trace_{self._trace_counter:06d}",
            timestamp_s=time.time(),
            session_id=session_id,
        )

        # Summarize frame log
        if frame_log:
            for frame in frame_log:
                verdict = frame.get("verdict", "SILENCE")
                if verdict == "SELF":
                    trace.self_frames += 1
                elif verdict == "WORLD":
                    trace.world_frames += 1
                elif verdict == "BOTH":
                    trace.both_frames += 1
                else:
                    trace.silence_frames += 1

            # Compute attribution confidence
            speech_frames = trace.self_frames + trace.world_frames + trace.both_frames
            if speech_frames > 0:
                trace.attribution_confidence = trace.self_frames / speech_frames

        # M4 outcomes
        if m4_outcomes:
            trace.m4_outcomes = dict(m4_outcomes)
            trace.m4_accepted_frames = m4_outcomes.get("accepted", 0)
            trace.m4_attempted_frames = sum(
                v for k, v in m4_outcomes.items()
                if k in ("accepted", "below_threshold", "exception", "buffer_too_short")
            )

        # Interaction events
        if interaction_events:
            trace.interrupt_events = interaction_events.get("interrupts", 0)
            trace.defer_events = interaction_events.get("defers", 0)
            trace.resume_events = interaction_events.get("resumes", 0)
            trace.collision_repairs = interaction_events.get("collision_repairs", 0)

        # Known speakers
        if known_speakers:
            trace.known_speakers = list(known_speakers)

        # Metadata
        if metadata:
            trace.metadata = dict(metadata)

        # Compute salience
        trace.salience_score = self._compute_salience(trace)

        # Store
        self._traces.append(trace)
        self._persist(trace)

        # Evict old traces if over memory limit
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces:]

        return trace

    def _compute_salience(self, trace: EpisodicTrace) -> float:
        """Compute salience score for a trace.

        Higher salience for: interrupts, repairs, high M4 activity,
        multiple speakers, unusual attribution patterns.
        """
        score = 0.0

        # Interaction events are salient
        score += 0.3 * min(trace.interrupt_events, 3)
        score += 0.2 * min(trace.defer_events, 3)
        score += 0.2 * min(trace.collision_repairs, 3)

        # M4 activity
        if trace.m4_attempted_frames > 0:
            score += 0.1 * min(trace.m4_attempted_frames / 10, 1.0)

        # Multiple speakers
        if len(trace.known_speakers) > 1:
            score += 0.2

        # Unusual attribution (mostly world frames = someone else talking)
        speech = trace.self_frames + trace.world_frames + trace.both_frames
        if speech > 0 and trace.world_frames / speech > 0.8:
            score += 0.15

        return min(score, 1.0)

    # -- Retrieval --------------------------------------------------------

    def retrieve(
        self,
        *,
        since_s: float | None = None,
        until_s: float | None = None,
        min_salience: float = 0.0,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[EpisodicTrace]:
        """Retrieve episodic traces matching criteria."""
        results = []
        for trace in reversed(self._traces):  # Most recent first
            if since_s is not None and trace.timestamp_s < since_s:
                continue
            if until_s is not None and trace.timestamp_s > until_s:
                continue
            if trace.salience_score < min_salience:
                continue
            if session_id is not None and trace.session_id != session_id:
                continue
            results.append(trace)
            if len(results) >= limit:
                break
        return results

    def get_trace(self, trace_id: str) -> EpisodicTrace | None:
        """Get a specific trace by ID."""
        for trace in self._traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def get_session_traces(self, session_id: str) -> list[EpisodicTrace]:
        """Get all traces for a session."""
        return [t for t in self._traces if t.session_id == session_id]

    def get_salient_traces(self, threshold: float | None = None) -> list[EpisodicTrace]:
        """Get traces above the salience threshold."""
        threshold = threshold or self._salience_threshold
        return [t for t in self._traces if t.salience_score >= threshold]

    # -- Consolidation ---------------------------------------------------

    def get_consolidation_candidates(
        self,
        *,
        min_traces: int = 3,
        pattern_type: str = "speaker_attribution",
    ) -> list[ConsolidationCandidate]:
        """Extract consolidation candidates from accumulated traces.

        Identifies repeated patterns across episodic traces that could
        be promoted to SAG nodes/edges during offline consolidation.
        """
        candidates = []

        if len(self._traces) < min_traces:
            return candidates

        # Pattern: recurring speaker attribution
        speaker_patterns: dict[str, list[EpisodicTrace]] = {}
        for trace in self._traces:
            for speaker in trace.known_speakers:
                speaker_patterns.setdefault(speaker, []).append(trace)

        for speaker, traces in speaker_patterns.items():
            if len(traces) >= min_traces:
                avg_confidence = float(np.mean([t.attribution_confidence for t in traces]))
                candidates.append(ConsolidationCandidate(
                    pattern_id=f"speaker_{speaker}",
                    description=f"Recurring speaker: {speaker} ({len(traces)} observations)",
                    trace_count=len(traces),
                   first_seen_s=min(t.timestamp_s for t in traces),
                    last_seen_s=max(t.timestamp_s for t in traces),
                    confidence=avg_confidence,
                    evidence=[t.trace_id for t in traces[:5]],
                ))

        # Pattern: recurring interaction events
        interaction_traces = [t for t in self._traces if t.interrupt_events > 0]
        if len(interaction_traces) >= min_traces:
            candidates.append(ConsolidationCandidate(
                pattern_id="frequent_interruptions",
                description=f"Frequent interruptions ({len(interaction_traces)} traces)",
                trace_count=len(interaction_traces),
                first_seen_s=min(t.timestamp_s for t in interaction_traces),
                last_seen_s=max(t.timestamp_s for t in interaction_traces),
                confidence=float(np.mean([t.salience_score for t in interaction_traces])),
                evidence=[t.trace_id for t in interaction_traces[:5]],
            ))

        # Pattern: M4 acceptance rate trends
        m4_traces = [t for t in self._traces if t.m4_attempted_frames > 0]
        if len(m4_traces) >= min_traces:
            avg_accept_rate = float(np.mean([
                t.m4_accepted_frames / max(t.m4_attempted_frames, 1)
                for t in m4_traces
            ]))
            candidates.append(ConsolidationCandidate(
                pattern_id="m4_accept_rate",
                description=f"M4 accept rate: {avg_accept_rate:.1%} over {len(m4_traces)} traces",
                trace_count=len(m4_traces),
                first_seen_s=min(t.timestamp_s for t in m4_traces),
                last_seen_s=max(t.timestamp_s for t in m4_traces),
                confidence=avg_accept_rate,
                evidence=[t.trace_id for t in m4_traces[:5]],
            ))

        return candidates

    # -- Persistence ------------------------------------------------------

    def _persist(self, trace: EpisodicTrace) -> None:
        """Append a trace to the JSONL event log."""
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")

    def _load(self) -> None:
        """Load traces from the JSONL event log."""
        if not self._persist_path or not self._persist_path.exists():
            return
        self._traces = []
        with open(self._persist_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self._traces.append(EpisodicTrace.from_dict(d))
                except (json.JSONDecodeError, TypeError):
                    continue
        if self._traces:
            self._trace_counter = max(
                int(t.trace_id.split("_")[1]) for t in self._traces
            )

    def save(self) -> None:
        """Force-save all traces to disk."""
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "w") as f:
            for trace in self._traces:
                f.write(json.dumps(trace.to_dict()) + "\n")

    # -- Stats ------------------------------------------------------------

    @property
    def trace_count(self) -> int:
        return len(self._traces)

    @property
    def session_count(self) -> int:
        return len(set(t.session_id for t in self._traces))

    def summary(self) -> dict:
        """Return a summary of the hippocampus state."""
        if not self._traces:
            return {"trace_count": 0, "session_count": 0}

        return {
            "trace_count": len(self._traces),
            "session_count": self.session_count,
            "salient_traces": len(self.get_salient_traces()),
            "avg_salience": float(np.mean([t.salience_score for t in self._traces])),
            "total_self_frames": sum(t.self_frames for t in self._traces),
            "total_world_frames": sum(t.world_frames for t in self._traces),
            "total_interrupts": sum(t.interrupt_events for t in self._traces),
            "total_repairs": sum(t.collision_repairs for t in self._traces),
            "known_speakers": sorted(set(
                s for t in self._traces for s in t.known_speakers
            )),
            "consolidation_candidates": len(self.get_consolidation_candidates()),
        }

"""Multi-speaker world model for M5.

Tracks known speakers across sessions using ECAPA embeddings.
Provides speaker identification, dialog memory retrieval, and SAG integration.

The world model maintains a registry of known speakers, each with:
- A name/identifier
- A reference embedding (ECAPA 192-dim)
- Interaction history (from hippocampus episodic traces)
- SAG node linkage
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from hippocampus import Hippocampus
from sag.graph import SemanticAbstractionGraph
from sag.models import EdgeAssertion, EdgeType, Node
from stg.speaker import SpeakerIdentifier


@dataclass
class SpeakerProfile:
    """Profile of a known speaker."""
    speaker_id: str
    name: str
    embedding: np.ndarray | None = None
    embedding_count: int = 0  # Number of embeddings averaged
    first_seen_s: float = 0.0
    last_seen_s: float = 0.0
    interaction_count: int = 0
    total_speech_frames: int = 0
    metadata: dict = field(default_factory=dict)

    def update_embedding(self, new_embedding: np.ndarray) -> None:
        """Update running average embedding."""
        if self.embedding is None:
            self.embedding = new_embedding.copy()
            self.embedding_count = 1
        else:
            # Running average
            self.embedding = (
                self.embedding * self.embedding_count + new_embedding
            ) / (self.embedding_count + 1)
            self.embedding_count += 1

    def similarity_to(self, embedding: np.ndarray) -> float:
        """Compute cosine similarity to a given embedding."""
        if self.embedding is None:
            return 0.0
        dot = float(np.dot(self.embedding, embedding))
        norm = float(np.linalg.norm(self.embedding) * np.linalg.norm(embedding))
        if norm < 1e-9:
            return 0.0
        return dot / norm

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SpeakerProfile":
        if d.get("embedding") is not None:
            d["embedding"] = np.array(d["embedding"], dtype=np.float32)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SpeakerRegistry:
    """Registry of known speakers with embedding-based identification."""

    def __init__(
        self,
        identifier: SpeakerIdentifier | None = None,
        similarity_threshold: float = 0.5,
        self_speaker_id: str = "self",
    ) -> None:
        self._speakers: dict[str, SpeakerProfile] = {}
        self._identifier = identifier
        self._threshold = similarity_threshold
        self._self_id = self_speaker_id

    @property
    def speaker_count(self) -> int:
        return len(self._speakers)

    @property
    def known_speakers(self) -> list[str]:
        return list(self._speakers.keys())

    def register_speaker(
        self,
        speaker_id: str,
        name: str,
        embedding: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> SpeakerProfile:
        """Register a new known speaker."""
        now = time.time()
        if speaker_id in self._speakers:
            profile = self._speakers[speaker_id]
            profile.name = name
            if embedding is not None:
                profile.update_embedding(embedding)
            profile.last_seen_s = now
            if metadata:
                profile.metadata.update(metadata)
        else:
            profile = SpeakerProfile(
                speaker_id=speaker_id,
                name=name,
                embedding=embedding,
                embedding_count=1 if embedding is not None else 0,
                first_seen_s=now,
                last_seen_s=now,
                metadata=metadata or {},
            )
            self._speakers[speaker_id] = profile
        return profile

    def best_match(
        self,
        embedding: np.ndarray,
    ) -> tuple[str | None, float]:
        """Return the highest-similarity speaker regardless of threshold."""
        best_id = None
        best_sim = 0.0

        for speaker_id, profile in self._speakers.items():
            sim = profile.similarity_to(embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = speaker_id

        return best_id, best_sim

    def identify_speaker(
        self,
        embedding: np.ndarray,
        min_similarity: float | None = None,
    ) -> tuple[str | None, float]:
        """Identify a speaker from an embedding.

        Returns (speaker_id, similarity) or (None, 0.0) if no match.
        """
        threshold = min_similarity or self._threshold
        best_id, best_sim = self.best_match(embedding)
        if best_id is not None and best_sim >= threshold:
            return best_id, best_sim
        return None, 0.0

    def get_profile(self, speaker_id: str) -> SpeakerProfile | None:
        """Get a speaker profile by ID."""
        return self._speakers.get(speaker_id)

    def update_from_frame_log(
        self,
        frame_log: list[dict],
        identifier: SpeakerIdentifier | None = None,
    ) -> list[tuple[str, str, float]]:
        """Update speaker registry from a frame log.

        Extracts embeddings for WORLD/BOTH frames and either matches
        to known speakers or creates new speaker entries.

        Returns list of (speaker_id, action, similarity) tuples.
        """
        identifier = identifier or self._identifier
        if identifier is None:
            return []

        results = []
        world_frames = []

        for frame in frame_log:
            verdict = frame.get("verdict", "SILENCE")
            if verdict in ("WORLD", "BOTH"):
                # Extract embedding from frame audio if available
                frame_audio = frame.get("frame_audio")
                if frame_audio is not None:
                    world_frames.append(frame_audio)

        if not world_frames:
            return []

        # Concatenate world frames and extract embedding
        combined = np.concatenate(world_frames)
        try:
            embedding = identifier.get_embedding(combined)
        except Exception:
            return []

        # Try to match to known speaker
        speaker_id, similarity = self.identify_speaker(embedding)

        if speaker_id is not None:
            # Update existing speaker
            profile = self._speakers[speaker_id]
            profile.update_embedding(embedding)
            profile.last_seen_s = time.time()
            profile.interaction_count += 1
            profile.total_speech_frames += len(world_frames)
            results.append((speaker_id, "matched", similarity))
        else:
            # Register new speaker
            new_id = f"speaker_{len(self._speakers) + 1:03d}"
            self.register_speaker(
                new_id,
                name=new_id,
                embedding=embedding,
                metadata={"auto_registered": True},
            )
            results.append((new_id, "new", similarity))

        return results


class MultiSpeakerWorldModel:
    """M5 Multi-Speaker World Model.

    Integrates speaker identification, hippocampus episodic memory,
    and SAG knowledge representation for multi-speaker tracking.
    """

    def __init__(
        self,
        registry: SpeakerRegistry,
        hippocampus: Hippocampus | None = None,
        sag: SemanticAbstractionGraph | None = None,
    ) -> None:
        self.registry = registry
        self.hippocampus = hippocampus
        self.sag = sag

    def process_interaction(
        self,
        frame_log: list[dict],
        session_id: str | None = None,
    ) -> dict:
        """Process an interaction turn: identify speakers, update registry, encode to hippocampus."""
        results = {
            "speakers_identified": [],
            "new_speakers": [],
            "trace_id": None,
        }

        # Update speaker registry from frame log
        if self.registry._identifier is not None:
            id_results = self.registry.update_from_frame_log(frame_log)
            for speaker_id, action, similarity in id_results:
                results["speakers_identified"].append({
                    "speaker_id": speaker_id,
                    "action": action,
                    "similarity": round(similarity, 4),
                })
                if action == "new":
                    results["new_speakers"].append(speaker_id)

        # Encode to hippocampus
        if self.hippocampus is not None:
            known_speakers = [
                s["speaker_id"] for s in results["speakers_identified"]
            ]
            trace = self.hippocampus.encode(
                session_id=session_id,
                frame_log=frame_log,
                known_speakers=known_speakers,
            )
            results["trace_id"] = trace.trace_id

        # Update SAG with speaker nodes
        if self.sag is not None:
            for speaker_info in results["speakers_identified"]:
                sid = speaker_info["speaker_id"]
                profile = self.registry.get_profile(sid)
                if profile:
                    self._update_sag_speaker_node(profile)

        return results

    def _update_sag_speaker_node(self, profile: SpeakerProfile) -> None:
        """Update or create a speaker node in the SAG."""
        node_id = f"corollary.known_subjects.{profile.speaker_id}"
        self.sag.add_node(Node(
            id=node_id,
            core=(
                f"Speaker: {profile.name} "
                f"(seen {profile.interaction_count} times, "
                f"{profile.total_speech_frames} speech frames)"
            ),
            context={
                "interaction_count": profile.interaction_count,
                "total_speech_frames": profile.total_speech_frames,
                "embedding_count": profile.embedding_count,
                "first_seen_s": profile.first_seen_s,
                "last_seen_s": profile.last_seen_s,
            },
            grounding=[profile.speaker_id, profile.name, "speaker", "known"],
        ))
        self.sag.add_assertion(
            "corollary.known_subjects",
            node_id,
            EdgeType.PART_OF.value,
            EdgeAssertion(
                asserted_by="world_model",
                confidence=min(profile.interaction_count / 10.0, 1.0),
            ),
        )

    def query_speaker(self, speaker_id: str) -> dict | None:
        """Query information about a known speaker."""
        profile = self.registry.get_profile(speaker_id)
        if profile is None:
            return None

        # Get episodic traces involving this speaker
        traces = []
        if self.hippocampus is not None:
            all_traces = self.hippocampus.retrieve(limit=100)
            traces = [
                t for t in all_traces
                if speaker_id in t.known_speakers
            ]

        return {
            "speaker_id": profile.speaker_id,
            "name": profile.name,
            "interaction_count": profile.interaction_count,
            "total_speech_frames": profile.total_speech_frames,
            "first_seen_s": profile.first_seen_s,
            "last_seen_s": profile.last_seen_s,
            "recent_traces": len(traces),
        }

    def summary(self) -> dict:
        """Return world model summary."""
        return {
            "known_speakers": self.registry.speaker_count,
            "speaker_ids": self.registry.known_speakers,
            "hippocampus_traces": self.hippocampus.trace_count if self.hippocampus else 0,
            "sag_nodes": len(self.sag._nodes) if self.sag else 0,
        }

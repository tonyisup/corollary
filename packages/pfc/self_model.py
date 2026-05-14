"""SelfModel — maintains Corollary's self-concept subtree in the SAG.

M6: Self-Model and Introspection.

Listens to streaming frame logs and builds/maintains a self-concept subtree
in the Semantic Abstraction Graph:

    Corollary (root)
    ├── agent_voice       — what my voice sounds like (self-attribution evidence)
    ├── speech_patterns   — how I speak (duration, energy, repair frequency)
    └── known_subjects    — other agents I've interacted with (world speakers)

The SelfModel is a passive observer — it reads frame logs after the fact
and updates the SAG. It does NOT gate or control the audio pipeline.

Usage:
    from self_model import SelfModel

    model = SelfModel()
    model.initialize_sag(sag)

    # After a streaming run:
    model.process_frame_log(frame_log_iterator)

    # Inspect:
    print(model.summarize())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from hippocampus import Hippocampus
from sag.graph import SemanticAbstractionGraph
from sag.models import (
    ChangeKind,
    Edge,
    EdgeAssertion,
    EdgeType,
    GraphChange,
    Node,
)


# ── Node ID constants ────────────────────────────────────────────────────────

NODE_ROOT = "corollary"
NODE_AGENT_VOICE = "corollary.agent_voice"
NODE_SPEECH_PATTERNS = "corollary.speech_patterns"
NODE_KNOWN_SUBJECTS = "corollary.known_subjects"


@dataclass
class SpeechPatternStats:
    """Running statistics about Corollary's own speech."""
    total_speak_frames: int = 0
    total_interrupt_events: int = 0
    total_defer_events: int = 0
    total_resume_events: int = 0
    total_collision_repairs: int = 0
    total_self_frames: int = 0
    total_both_frames: int = 0
    energy_values: list = field(default_factory=list)
    self_match_values: list = field(default_factory=list)
    m4_accepted_frames: int = 0
    m4_attempted_frames: int = 0
    session_count: int = 0
    last_session_ts: float = 0.0

    @property
    def mean_energy_db(self) -> float:
        return float(np.mean(self.energy_values)) if self.energy_values else 0.0

    @property
    def mean_self_match(self) -> float:
        return float(np.mean(self.self_match_values)) if self.self_match_values else 0.0

    @property
    def m4_accept_rate(self) -> float:
        return self.m4_accepted_frames / max(self.m4_attempted_frames, 1)

    def to_dict(self) -> dict:
        return {
            "total_speak_frames": self.total_speak_frames,
            "total_interrupt_events": self.total_interrupt_events,
            "total_defer_events": self.total_defer_events,
            "total_resume_events": self.total_resume_events,
            "total_collision_repairs": self.total_collision_repairs,
            "total_self_frames": self.total_self_frames,
            "total_both_frames": self.total_both_frames,
            "mean_energy_db": round(self.mean_energy_db, 2),
            "mean_self_match": round(self.mean_self_match, 4),
            "m4_accept_rate": round(self.m4_accept_rate, 4),
            "m4_accepted_frames": self.m4_accepted_frames,
            "m4_attempted_frames": self.m4_attempted_frames,
            "session_count": self.session_count,
        }


class SelfModel:
    """Maintains Corollary's self-concept subtree in the SAG.

    Observes streaming frame logs and extracts:
    - Self-attribution evidence (SELF verdicts, M4 confidence)
    - Speech patterns (energy, duration, repair frequency)
    - Interaction history (interrupts, defers, resumes)
    """

    def __init__(self, hippocampus: Hippocampus | None = None) -> None:
        self.sag: Optional[SemanticAbstractionGraph] = None
        self.patterns = SpeechPatternStats()
        self._initialized = False
        self.hippocampus = hippocampus

    def initialize_sag(self, sag: SemanticAbstractionGraph) -> None:
        """Create the self-concept subtree in the SAG.

        Idempotent — safe to call multiple times.
        """
        self.sag = sag

        # Root node: Corollary
        sag.add_node(Node(
            id=NODE_ROOT,
            core="Corollary — a self-listening, closed-loop conversational AI agent",
            grounding=["corollary", "self", "agent"],
        ))

        # Agent voice node
        sag.add_node(Node(
            id=NODE_AGENT_VOICE,
            core="What Corollary's own voice sounds like — self-attribution evidence",
            grounding=["my voice", "self voice", "agent voice"],
        ))
        sag.add_assertion(NODE_ROOT, NODE_AGENT_VOICE, EdgeType.PART_OF.value,
                          EdgeAssertion(asserted_by="self_model", confidence=1.0))

        # Speech patterns node
        sag.add_node(Node(
            id=NODE_SPEECH_PATTERNS,
            core="How Corollary speaks — energy, duration, repair frequency, self-match quality",
            grounding=["speech", "speaking", "voice patterns"],
        ))
        sag.add_assertion(NODE_ROOT, NODE_SPEECH_PATTERNS, EdgeType.PART_OF.value,
                          EdgeAssertion(asserted_by="self_model", confidence=1.0))

        # Known subjects node
        sag.add_node(Node(
            id=NODE_KNOWN_SUBJECTS,
            core="Other agents Corollary has interacted with",
            grounding=["subjects", "speakers", "others"],
        ))
        sag.add_assertion(NODE_ROOT, NODE_KNOWN_SUBJECTS, EdgeType.PART_OF.value,
                          EdgeAssertion(asserted_by="self_model", confidence=1.0))

        self._initialized = True

    def process_frame_log(self, frames: list[dict]) -> dict:
        """Process a list of frame log entries from a streaming run.

        Returns a summary dict of what was observed.
        """
        if not self._initialized or self.sag is None:
            raise RuntimeError("SelfModel not initialized. Call initialize_sag() first.")

        session_frames = len(frames)
        if session_frames == 0:
            return {"frames_processed": 0}

        speak_frame_count = 0
        self_frame_count = 0
        both_frame_count = 0
        interrupt_count = 0
        defer_count = 0
        resume_count = 0
        repair_count = 0
        m4_accepted = 0
        m4_attempted = 0
        energy_vals = []
        match_vals = []

        for frame in frames:
            verdict = frame.get("verdict", "SILENCE")
            controller_state = frame.get("controller_state", "IDLE")

            if controller_state == "SPEAK":
                speak_frame_count += 1

            if verdict == "SELF":
                self_frame_count += 1
            elif verdict == "BOTH":
                both_frame_count += 1

            if frame.get("should_interrupt"):
                interrupt_count += 1
            if frame.get("deferred"):
                defer_count += 1
            if frame.get("resumed"):
                resume_count += 1
            if frame.get("collision_repair_text"):
                repair_count += 1

            m4_outcome = frame.get("m4_outcome", "not_attempted")
            if m4_outcome in ("accepted", "below_threshold", "exception", "buffer_too_short"):
                m4_attempted += 1
            if m4_outcome == "accepted":
                m4_accepted += 1

            energy = frame.get("energy_db")
            if energy is not None:
                energy_vals.append(energy)

            match = frame.get("self_match_score")
            if match is not None:
                match_vals.append(match)

        # Update running stats
        self.patterns.total_speak_frames += speak_frame_count
        self.patterns.total_self_frames += self_frame_count
        self.patterns.total_both_frames += both_frame_count
        self.patterns.total_interrupt_events += interrupt_count
        self.patterns.total_defer_events += defer_count
        self.patterns.total_resume_events += resume_count
        self.patterns.total_collision_repairs += repair_count
        self.patterns.m4_accepted_frames += m4_accepted
        self.patterns.m4_attempted_frames += m4_attempted
        self.patterns.energy_values.extend(energy_vals)
        self.patterns.self_match_values.extend(match_vals)
        self.patterns.session_count += 1
        self.patterns.last_session_ts = time.time()

        # Update SAG nodes with observed data
        self._update_agent_voice_node(m4_accepted, m4_attempted, match_vals)
        self._update_speech_patterns_node(
            speak_frame_count, self_frame_count, both_frame_count,
            interrupt_count, defer_count, resume_count, repair_count,
            energy_vals, session_frames,
        )

        # Encode episodic trace to hippocampus
        if self.hippocampus is not None:
            m4_outcomes = {}
            for frame in frames:
                outcome = frame.get("m4_outcome", "not_attempted")
                m4_outcomes[outcome] = m4_outcomes.get(outcome, 0) + 1

            self.hippocampus.encode(
                session_id=f"session_{self.patterns.session_count:04d}",
                frame_log=frames,
                m4_outcomes=m4_outcomes,
                interaction_events={
                    "interrupts": interrupt_count,
                    "defers": defer_count,
                    "resumes": resume_count,
                    "collision_repairs": repair_count,
                },
                known_speakers=self._extract_known_speakers(frames),
                metadata={
                    "speak_frames": speak_frame_count,
                    "energy_mean_db": float(np.mean(energy_vals)) if energy_vals else 0.0,
                    "self_match_mean": float(np.mean(match_vals)) if match_vals else 0.0,
                },
            )

        # Auto-add known subjects from frame logs
        known = self._extract_known_speakers(frames)
        for speaker_id in known:
            if speaker_id not in ("self", "world"):
                self.add_known_subject(
                    subject_id=speaker_id,
                    label=speaker_id,
                    context={"source": "frame_log"},
                )

        return {
            "frames_processed": session_frames,
            "speak_frames": speak_frame_count,
            "self_frames": self_frame_count,
            "both_frames": both_frame_count,
            "interrupts": interrupt_count,
            "defers": defer_count,
            "resumes": resume_count,
            "collision_repairs": repair_count,
            "m4_accepted": m4_accepted,
            "m4_attempted": m4_attempted,
        }

    def _update_agent_voice_node(
        self, m4_accepted: int, m4_attempted: int, match_vals: list
    ) -> None:
        """Update the agent_voice node with latest self-attribution evidence."""
        if not match_vals:
            return

        mean_match = float(np.mean(match_vals))
        m4_rate = m4_accepted / max(m4_attempted, 1)

        self.sag.add_node(Node(
            id=NODE_AGENT_VOICE,
            core=(
                f"Corollary's voice: mean self-match={mean_match:.3f}, "
                f"M4 accept rate={m4_rate:.1%} ({m4_accepted}/{m4_attempted} frames)"
            ),
            context={
                "mean_self_match": round(mean_match, 4),
                "m4_accept_rate": round(m4_rate, 4),
                "m4_accepted_frames": m4_accepted,
                "m4_attempted_frames": m4_attempted,
                "total_sessions": self.patterns.session_count,
            },
            grounding=["my voice", "self voice", "agent voice", "self-match"],
        ))

    def _extract_known_speakers(self, frames: list[dict]) -> list[str]:
        """Extract known speakers from frame log.

        Returns a list of speaker identifiers. Uses per-speaker IDs from
        the speaker_id field when available, otherwise falls back to
        generic categories based on attribution verdicts.
        """
        speakers = set()
        for frame in frames:
            # Prefer per-speaker ID from speaker identification
            sid = frame.get("speaker_id")
            if sid is not None:
                speakers.add(sid)
            else:
                # Fallback to generic categories
                verdict = frame.get("verdict", "SILENCE")
                if verdict == "SELF":
                    speakers.add("self")
                elif verdict in ("WORLD", "BOTH"):
                    speakers.add("world")
        return sorted(speakers)

    def _update_speech_patterns_node(
        self,
        speak_frames: int,
        self_frames: int,
        both_frames: int,
        interrupts: int,
        defers: int,
        resumes: int,
        repairs: int,
        energy_vals: list,
        total_frames: int,
    ) -> None:
        """Update the speech_patterns node with latest observations."""
        speak_ratio = speak_frames / max(total_frames, 1)
        self_ratio = self_frames / max(speak_frames, 1) if speak_frames > 0 else 0.0
        mean_energy = float(np.mean(energy_vals)) if energy_vals else 0.0

        self.sag.add_node(Node(
            id=NODE_SPEECH_PATTERNS,
            core=(
                f"Speech: {speak_ratio:.1%} speaking, {self_ratio:.1%} self-attributed, "
                f"mean energy={mean_energy:.1f}dB, "
                f"{interrupts} interrupts, {repairs} repairs over {self.patterns.session_count} sessions"
            ),
            context={
                "speak_ratio": round(speak_ratio, 4),
                "self_attribution_ratio": round(self_ratio, 4),
                "mean_energy_db": round(mean_energy, 2),
                "total_interrupts": self.patterns.total_interrupt_events,
                "total_defers": self.patterns.total_defer_events,
                "total_resumes": self.patterns.total_resume_events,
                "total_repairs": self.patterns.total_collision_repairs,
                "total_sessions": self.patterns.session_count,
            },
            grounding=["speech", "speaking", "voice patterns", "self-attribution"],
        ))

    def add_known_subject(self, subject_id: str, label: str,
                          first_seen: Optional[float] = None,
                          context: Optional[dict] = None) -> None:
        """Add a known subject (world speaker) to the SAG."""
        if not self._initialized or self.sag is None:
            raise RuntimeError("SelfModel not initialized.")

        node_id = f"corollary.known_subjects.{subject_id}"
        self.sag.add_node(Node(
            id=node_id,
            core=f"Known subject: {label}",
            context=context or {},
            grounding=[label, subject_id],
        ))
        self.sag.add_assertion(
            NODE_KNOWN_SUBJECTS, node_id, EdgeType.PART_OF.value,
            EdgeAssertion(asserted_by="self_model", confidence=0.8),
        )

    def summarize(self) -> dict:
        """Return a summary of the self-model's current state."""
        return {
            "initialized": self._initialized,
            "sag_nodes": len(self.sag._nodes) if self.sag else 0,
            "sag_edges": len(self.sag._edges) if self.sag else 0,
            "patterns": self.patterns.to_dict(),
        }

    def query_self_concept(self) -> dict:
        """Query the self-concept subtree from the SAG.

        Returns the current state of all self-model nodes.
        """
        if not self._initialized or self.sag is None:
            return {"error": "not initialized"}

        result = {}
        for node_id in [NODE_ROOT, NODE_AGENT_VOICE, NODE_SPEECH_PATTERNS, NODE_KNOWN_SUBJECTS]:
            node = self.sag._nodes.get(node_id)
            if node:
                result[node_id] = {
                    "core": node.core,
                    "context": node.context,
                }

        # Include known subjects
        subjects = []
        for (src, tgt, rel), edge in self.sag._edges.items():
            if src == NODE_KNOWN_SUBJECTS and rel == EdgeType.PART_OF.value:
                subject_node = self.sag._nodes.get(tgt)
                if subject_node:
                    subjects.append({
                        "id": tgt,
                        "core": subject_node.core,
                        "context": subject_node.context,
                    })
        result["known_subjects"] = subjects

        return result

"""Offline consolidation loop for M5 hippocampus → SAG knowledge extraction.

Runs as a background process (or on-demand) to extract repeated patterns
from accumulated episodic traces and write stable nodes/edges into the SAG.

Pattern extraction, not event transcription:
- Single events → hippocampus only
- Repeated patterns → SAG structure
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from hippocampus import ConsolidationCandidate, Hippocampus
from sag.graph import SemanticAbstractionGraph
from sag.models import EdgeAssertion, EdgeType, Node


class OfflineConsolidator:
    """Background consolidation loop: episodic traces → SAG knowledge.

    Extracts repeated patterns from hippocampus episodic memory and writes
    stable nodes/edges into the Semantic Abstraction Graph.

    The consolidator is a passive observer — it reads from the hippocampus
    and writes to the SAG. It does NOT gate or control the audio pipeline.
    """

    def __init__(
        self,
        hippocampus: Hippocampus,
        sag: SemanticAbstractionGraph,
        *,
        min_traces_for_pattern: int = 3,
        consolidation_interval_s: float = 60.0,
        persist_path: Optional[Path | str] = None,
    ) -> None:
        self.hippocampus = hippocampus
        self.sag = sag
        self._min_traces = min_traces_for_pattern
        self._interval_s = consolidation_interval_s
        self._last_consolidation_s = 0.0
        self._consolidation_count = 0
        self._persist_path = Path(persist_path) if persist_path else None

    def consolidate(self, force: bool = False) -> dict:
        """Run one consolidation pass.

        Extracts patterns from episodic traces and writes new SAG nodes/edges.
        Returns a summary of what was consolidated.

        Parameters
        ----------
        force : bool
            If True, run even if interval hasn't elapsed.
        """
        now = time.time()
        if not force and (now - self._last_consolidation_s) < self._interval_s:
            return {"consolidated": False, "reason": "interval_not_elapsed"}

        self._last_consolidation_s = now
        self._consolidation_count += 1

        candidates = self.hippocampus.get_consolidation_candidates(
            min_traces=self._min_traces,
        )

        if not candidates:
            return {"consolidated": True, "candidates_found": 0, "nodes_added": 0, "edges_added": 0}

        nodes_added = 0
        edges_added = 0

        for candidate in candidates:
            result = self._consolidate_candidate(candidate)
            nodes_added += result.get("nodes_added", 0)
            edges_added += result.get("edges_added", 0)

        summary = {
            "consolidated": True,
            "candidates_found": len(candidates),
            "nodes_added": nodes_added,
            "edges_added": edges_added,
            "consolidation_id": self._consolidation_count,
            "timestamp_s": now,
        }

        if self._persist_path:
            self._save_summary(summary)

        return summary

    def _consolidate_candidate(self, candidate: ConsolidationCandidate) -> dict:
        """Consolidate a single pattern candidate into the SAG."""
        nodes_added = 0
        edges_added = 0

        # Create a pattern node in the SAG
        pattern_node_id = f"pattern_{candidate.pattern_id}"

        # Check if this pattern already exists
        existing = self.sag._nodes.get(pattern_node_id)
        if existing:
            # Update existing node with new evidence
            self.sag.add_node(Node(
                id=pattern_node_id,
                core=candidate.description,
                context={
                    "trace_count": candidate.trace_count,
                    "confidence": round(candidate.confidence, 4),
                    "first_seen_s": candidate.first_seen_s,
                    "last_seen_s": candidate.last_seen_s,
                    "evidence_count": len(candidate.evidence),
                },
                grounding=[candidate.pattern_id, "pattern", "consolidated"],
            ))
            nodes_added = 0  # Updated, not new
        else:
            self.sag.add_node(Node(
                id=pattern_node_id,
                core=candidate.description,
                context={
                    "trace_count": candidate.trace_count,
                    "confidence": round(candidate.confidence, 4),
                    "first_seen_s": candidate.first_seen_s,
                    "last_seen_s": candidate.last_seen_s,
                    "evidence_count": len(candidate.evidence),
                },
                grounding=[candidate.pattern_id, "pattern", "consolidated"],
            ))
            nodes_added = 1

        # Link pattern to the root
        self.sag.add_assertion("corollary", pattern_node_id, EdgeType.PART_OF.value,
                               EdgeAssertion(asserted_by="offline_consolidation",
                                            confidence=candidate.confidence))
        edges_added += 1

        # For speaker patterns, link to known_subjects
        if "speaker" in candidate.pattern_id:
            speaker_name = candidate.pattern_id.replace("speaker_", "")
            subject_node_id = f"corollary.known_subjects.{speaker_name}"
            self.sag.add_node(Node(
                id=subject_node_id,
                core=f"Known speaker: {speaker_name} (observed {candidate.trace_count} times)",
                grounding=[speaker_name, "speaker", "known"],
            ))
            self.sag.add_assertion("corollary.known_subjects", subject_node_id,
                                   EdgeType.PART_OF.value,
                                   EdgeAssertion(asserted_by="offline_consolidation",
                                                confidence=candidate.confidence))
            nodes_added += 1
            edges_added += 1

        return {"nodes_added": nodes_added, "edges_added": edges_added}

    def _save_summary(self, summary: dict) -> None:
        """Persist consolidation summary to disk."""
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(self._persist_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    def summary(self) -> dict:
        """Return consolidator state summary."""
        return {
            "consolidation_count": self._consolidation_count,
            "last_consolidation_s": self._last_consolidation_s,
            "hippocampus_traces": self.hippocampus.trace_count,
            "sag_nodes": len(self.sag._nodes),
            "sag_edges": len(self.sag._edges),
        }

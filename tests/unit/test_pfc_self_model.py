"""Unit tests for the SelfModel — Corollary's self-concept in the SAG."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from pfc.self_model import SelfModel, NODE_ROOT, NODE_AGENT_VOICE, NODE_SPEECH_PATTERNS, NODE_KNOWN_SUBJECTS
from sag.graph import SemanticAbstractionGraph
from sag.models import EdgeType


def make_frame(verdict="SELF", controller_state="SPEAK", **kwargs):
    """Create a minimal frame log entry."""
    frame = {
        "frame_index": 0,
        "timestamp_s": 0.0,
        "vad_speech": True,
        "verdict": verdict,
        "controller_state": controller_state,
        "can_speak": controller_state == "SPEAK",
        "should_interrupt": False,
        "tts_playing": controller_state == "SPEAK",
        "tts_position_s": 0.0,
        "tts_total_duration_s": 3.0,
        "output_source": "tts",
        "repair_position_s": 0.0,
        "repair_total_duration_s": 0.0,
        "repair_queued": False,
        "repair_playing": False,
        "repair_completed": False,
        "m4_outcome": "not_attempted",
        "m4_confidence": None,
        "energy_db": -20.0,
        "self_match_score": 0.8,
        "self_match_score_per_frame": 0.7,
        "is_in_motor_window": True,
        "deferred": False,
        "resumed": False,
        "collision_repair_text": None,
    }
    frame.update(kwargs)
    return frame


def make_self_frames(n=50, m4_accepted=True):
    """Generate n frames of self-attributed speech."""
    frames = []
    for i in range(n):
        frames.append(make_frame(
            verdict="SELF",
            controller_state="SPEAK",
            m4_outcome="accepted" if m4_accepted else "below_threshold",
            m4_confidence=0.85 if m4_accepted else 0.05,
            energy_db=-15.0 + i * 0.1,
            self_match_score=0.8 + i * 0.001,
            frame_index=i,
            timestamp_s=i * 0.03,
        ))
    return frames


def make_mixed_frames(n_self=30, n_world=20, n_silence=10):
    """Generate mixed frames: self, world, silence."""
    frames = []
    for i in range(n_self):
        frames.append(make_frame(
            verdict="SELF", controller_state="SPEAK",
            frame_index=i, timestamp_s=i * 0.03, m4_outcome="accepted",
            m4_confidence=0.9, energy_db=-15.0, self_match_score=0.85,
        ))
    for i in range(n_world):
        frames.append(make_frame(
            verdict="WORLD", controller_state="LISTEN",
            frame_index=n_self + i, timestamp_s=(n_self + i) * 0.03,
            m4_outcome="not_attempted", energy_db=-25.0, self_match_score=0.05,
        ))
    for i in range(n_silence):
        frames.append(make_frame(
            verdict="SILENCE", controller_state="IDLE",
            frame_index=n_self + n_world + i,
            timestamp_s=(n_self + n_world + i) * 0.03,
            m4_outcome="not_attempted", energy_db=-80.0, self_match_score=0.0,
        ))
    return frames


class TestSelfModel:
    """Tests for the SelfModel class."""

    def setup_method(self):
        self.sag = SemanticAbstractionGraph()
        self.model = SelfModel()
        self.model.initialize_sag(self.sag)

    def test_initialization_creates_nodes(self):
        """initialize_sag should create the 4 self-concept nodes."""
        assert NODE_ROOT in self.sag._nodes
        assert NODE_AGENT_VOICE in self.sag._nodes
        assert NODE_SPEECH_PATTERNS in self.sag._nodes
        assert NODE_KNOWN_SUBJECTS in self.sag._nodes

    def test_initialization_creates_edges(self):
        """initialize_sag should create part-of edges from root to children."""
        assert (NODE_ROOT, NODE_AGENT_VOICE, EdgeType.PART_OF.value) in self.sag._edges
        assert (NODE_ROOT, NODE_SPEECH_PATTERNS, EdgeType.PART_OF.value) in self.sag._edges
        assert (NODE_ROOT, NODE_KNOWN_SUBJECTS, EdgeType.PART_OF.value) in self.sag._edges

    def test_initialization_idempotent(self):
        """Calling initialize_sag twice should not duplicate edges."""
        self.model.initialize_sag(self.sag)
        edge_count = len(self.sag._edges)
        assert edge_count == 3  # Exactly 3 part-of edges

    def test_process_self_frames(self):
        """Processing self-attributed frames should update stats."""
        frames = make_self_frames(n=50)
        result = self.model.process_frame_log(frames)

        assert result["frames_processed"] == 50
        assert result["self_frames"] == 50
        assert result["m4_accepted"] == 50

    def test_process_mixed_frames(self):
        """Processing mixed frames should count correctly."""
        frames = make_mixed_frames(n_self=30, n_world=20, n_silence=10)
        result = self.model.process_frame_log(frames)

        assert result["frames_processed"] == 60
        assert result["self_frames"] == 30
        assert result["speak_frames"] == 30

    def test_running_stats_accumulate(self):
        """Stats should accumulate across multiple process_frame_log calls."""
        self.model.process_frame_log(make_self_frames(n=20))
        self.model.process_frame_log(make_self_frames(n=30))

        assert self.model.patterns.session_count == 2
        assert self.model.patterns.total_self_frames == 50
        assert self.model.patterns.total_speak_frames == 50

    def test_speech_pattern_stats(self):
        """SpeechPatternStats should compute means correctly."""
        from pfc.self_model import SpeechPatternStats
        stats = SpeechPatternStats()
        stats.energy_values = [-20.0, -15.0, -10.0]
        stats.self_match_values = [0.5, 0.7, 0.9]
        stats.m4_accepted_frames = 8
        stats.m4_attempted_frames = 10

        assert abs(stats.mean_energy_db - (-15.0)) < 0.01
        assert abs(stats.mean_self_match - 0.7) < 0.01
        assert abs(stats.m4_accept_rate - 0.8) < 0.01

    def test_known_subject(self):
        """add_known_subject should create a node and edge."""
        self.model.add_known_subject("speaker_001", "Human speaker in room A",
                                     context={"room": "A", "first_interaction": "2026-05-09"})

        node_id = "corollary.known_subjects.speaker_001"
        assert node_id in self.sag._nodes
        assert (NODE_KNOWN_SUBJECTS, node_id, EdgeType.PART_OF.value) in self.sag._edges

        node = self.sag._nodes[node_id]
        assert "Human speaker in room A" in node.core
        assert node.context["room"] == "A"

    def test_query_self_concept(self):
        """query_self_concept should return the current state."""
        self.model.process_frame_log(make_self_frames(n=20))
        concept = self.model.query_self_concept()

        assert NODE_AGENT_VOICE in concept
        assert NODE_SPEECH_PATTERNS in concept
        assert "patterns" not in concept  # Not in SAG, in stats

    def test_summarize(self):
        """summarize should return current state."""
        self.model.process_frame_log(make_self_frames(n=10))
        summary = self.model.summarize()

        assert summary["initialized"] is True
        assert summary["sag_nodes"] >= 4
        assert summary["patterns"]["session_count"] == 1
        assert summary["patterns"]["total_self_frames"] == 10

    def test_uninitialized_raises(self):
        """Methods should raise if SAG is not initialized."""
        model = SelfModel()
        with pytest.raises(RuntimeError, match="not initialized"):
            model.process_frame_log([make_frame()])

    def test_empty_frame_log(self):
        """Empty frame log should return zero counts without error."""
        result = self.model.process_frame_log([])
        assert result["frames_processed"] == 0

    def test_interrupt_defer_resume_counting(self):
        """Interrupt, defer, and resume events should be counted."""
        frames = make_self_frames(n=20)
        frames[10]["should_interrupt"] = True
        frames[10]["verdict"] = "BOTH"
        frames[15]["deferred"] = True
        frames[18]["resumed"] = True
        frames[12]["collision_repair_text"] = "Sorry, go ahead."

        result = self.model.process_frame_log(frames)
        assert result["interrupts"] == 1
        assert result["defers"] == 1
        assert result["resumes"] == 1
        assert result["collision_repairs"] == 1

    def test_sag_node_updates_with_observations(self):
        """SAG node context should reflect observed data after processing."""
        frames = make_self_frames(n=50, m4_accepted=True)
        self.model.process_frame_log(frames)

        voice_node = self.sag._nodes[NODE_AGENT_VOICE]
        assert "mean_self_match" in voice_node.context
        assert voice_node.context["m4_accept_rate"] == 1.0
        assert voice_node.context["m4_accepted_frames"] == 50

        pattern_node = self.sag._nodes[NODE_SPEECH_PATTERNS]
        assert "speak_ratio" in pattern_node.context
        assert "self_attribution_ratio" in pattern_node.context
        assert pattern_node.context["total_sessions"] == 1


class TestSelfModelIntegration:
    """Integration tests: SelfModel + SAG + streaming frame log."""

    def test_full_pipeline_self_only(self):
        """End-to-end: create model, process self-only frames, query SAG."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        frames = make_self_frames(n=100, m4_accepted=True)
        result = model.process_frame_log(frames)

        assert result["self_frames"] == 100
        assert result["m4_accepted"] == 100

        # SAG should have updated nodes
        voice = sag._nodes[NODE_AGENT_VOICE]
        assert "self-match" in voice.core.lower() or "self-match" in str(voice.context)

        # Summary should be consistent
        summary = model.summarize()
        assert summary["patterns"]["total_self_frames"] == 100
        assert summary["patterns"]["m4_accept_rate"] == 1.0

    def test_full_pipeline_mixed_session(self):
        """End-to-end: mixed self/world/silence session."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        frames = make_mixed_frames(n_self=40, n_world=30, n_silence=10)
        result = model.process_frame_log(frames)

        assert result["frames_processed"] == 80
        assert result["self_frames"] == 40

        concept = model.query_self_concept()
        assert NODE_SPEECH_PATTERNS in concept
        pattern_ctx = concept[NODE_SPEECH_PATTERNS]["context"]
        assert pattern_ctx["total_sessions"] == 1

    def test_multiple_sessions_accumulate(self):
        """Multiple sessions should accumulate in SAG context."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        for _ in range(5):
            frames = make_self_frames(n=20)
            model.process_frame_log(frames)

        summary = model.summarize()
        assert summary["patterns"]["session_count"] == 5
        assert summary["patterns"]["total_self_frames"] == 100

        pattern_node = sag._nodes[NODE_SPEECH_PATTERNS]
        assert pattern_node.context["total_sessions"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

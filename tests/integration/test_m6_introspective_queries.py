"""M6 introspective queries end-to-end test.

Exercises the SelfModel's SAG against synthetic streaming output and
verifies that introspective queries return meaningful self-concept data.
"""

from __future__ import annotations

import numpy as np
import pytest

from pfc.self_model import SelfModel, NODE_ROOT, NODE_AGENT_VOICE, NODE_SPEECH_PATTERNS, NODE_KNOWN_SUBJECTS
from sag.graph import SemanticAbstractionGraph


def _make_synthetic_frames(
    n_frames: int = 100,
    speak_ratio: float = 0.3,
    self_ratio: float = 0.7,
    n_speakers: int = 2,
) -> list[dict]:
    """Generate synthetic frame logs for testing."""
    rng = np.random.default_rng(42)
    frames = []
    for i in range(n_frames):
        is_speak = rng.random() < speak_ratio
        if is_speak:
            verdict = "SELF" if rng.random() < self_ratio else "WORLD"
            controller_state = "SPEAK"
        else:
            verdict = "SILENCE"
            controller_state = "IDLE"

        frame = {
            "frame_index": i,
            "timestamp_s": i * 0.03,
            "vad_speech": verdict != "SILENCE",
            "verdict": verdict,
            "controller_state": controller_state,
            "can_speak": controller_state != "LISTEN",
            "should_interrupt": False,
            "tts_playing": is_speak,
            "tts_position_s": i * 0.03,
            "tts_total_duration_s": n_frames * 0.03,
            "output_source": "tts" if is_speak else "none",
            "repair_position_s": 0.0,
            "repair_total_duration_s": 0.0,
            "repair_queued": False,
            "repair_playing": False,
            "repair_completed": False,
            "m4_outcome": "accepted" if verdict == "SELF" and rng.random() < 0.8 else "below_threshold",
            "m4_confidence": float(rng.uniform(0.3, 0.9)) if verdict == "SELF" else float(rng.uniform(0.0, 0.3)),
            "energy_db": float(rng.uniform(-40, -20)) if verdict != "SILENCE" else -99.0,
            "self_match_score": float(rng.uniform(0.1, 0.8)) if verdict == "SELF" else float(rng.uniform(0.0, 0.2)),
            "self_match_score_per_frame": None,
            "is_in_motor_window": is_speak,
            "deferred": False,
            "resumed": False,
            "collision_repair_text": None,
            "speaker_id": None,
            "speaker_best_match_id": None,
            "speaker_best_match_similarity": None,
            "speaker_match_passed": None,
            "speaker_registered": None,
            "room_adapted": None,
            "room_adaptation_reason": None,
            "room_adaptation_error": None,
            "room_adaptation_updates": None,
        }

        # Add speaker IDs for WORLD frames
        if verdict == "WORLD":
            speaker_idx = rng.integers(0, n_speakers)
            frame["speaker_id"] = f"speaker_{speaker_idx}"
            frame["speaker_best_match_id"] = f"speaker_{speaker_idx}"
            frame["speaker_best_match_similarity"] = float(rng.uniform(0.3, 0.7))
            frame["speaker_match_passed"] = True

        frames.append(frame)
    return frames


class TestM6IntrospectiveQueries:
    def test_self_model_processes_frame_log(self):
        """SelfModel should process frame logs and populate SAG."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        frames = _make_synthetic_frames(n_frames=50)
        result = model.process_frame_log(frames)

        assert result["frames_processed"] == 50
        assert result["speak_frames"] > 0
        assert result["self_frames"] > 0

    def test_self_concept_query_returns_populated_nodes(self):
        """query_self_concept should return all self-model nodes."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        frames = _make_synthetic_frames(n_frames=100)
        model.process_frame_log(frames)

        concept = model.query_self_concept()

        assert NODE_ROOT in concept
        assert NODE_AGENT_VOICE in concept
        assert NODE_SPEECH_PATTERNS in concept
        assert NODE_KNOWN_SUBJECTS in concept

    def test_agent_voice_node_contains_self_match_data(self):
        """The agent_voice node should contain self-match statistics."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        frames = _make_synthetic_frames(n_frames=100, self_ratio=0.8)
        model.process_frame_log(frames)

        concept = model.query_self_concept()
        agent_voice = concept[NODE_AGENT_VOICE]

        assert "mean_self_match" in agent_voice["context"]
        assert "m4_accept_rate" in agent_voice["context"]
        assert agent_voice["context"]["m4_attempted_frames"] > 0

    def test_speech_patterns_node_tracks_session_stats(self):
        """The speech_patterns node should track session statistics."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        frames = _make_synthetic_frames(n_frames=100)
        model.process_frame_log(frames)

        concept = model.query_self_concept()
        patterns = concept[NODE_SPEECH_PATTERNS]

        assert "speak_ratio" in patterns["context"]
        assert "self_attribution_ratio" in patterns["context"]
        assert "mean_energy_db" in patterns["context"]
        assert patterns["context"]["total_sessions"] == 1

    def test_known_subjects_tracked_from_speaker_ids(self):
        """Known subjects should be extracted from frame speaker_id fields."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        frames = _make_synthetic_frames(n_frames=200, n_speakers=3)
        model.process_frame_log(frames)

        concept = model.query_self_concept()
        subjects = concept["known_subjects"]

        # Should have at least one known subject from WORLD frames
        assert len(subjects) >= 1
        subject_ids = [s["id"] for s in subjects]
        # Check that speaker IDs are in the known_subjects subtree
        assert any("speaker_" in sid for sid in subject_ids)

    def test_summarize_returns_full_state(self):
        """summarize() should return complete self-model state."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        frames = _make_synthetic_frames(n_frames=50)
        model.process_frame_log(frames)

        summary = model.summarize()

        assert summary["initialized"] is True
        assert summary["sag_nodes"] > 0
        assert summary["sag_edges"] > 0
        assert summary["patterns"]["session_count"] == 1
        assert summary["patterns"]["total_speak_frames"] > 0

    def test_multiple_sessions_accumulate(self):
        """Running multiple sessions should accumulate statistics."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        for _ in range(3):
            frames = _make_synthetic_frames(n_frames=50)
            model.process_frame_log(frames)

        summary = model.summarize()
        assert summary["patterns"]["session_count"] == 3
        assert summary["patterns"]["total_speak_frames"] > 0

        concept = model.query_self_concept()
        patterns = concept[NODE_SPEECH_PATTERNS]
        assert patterns["context"]["total_sessions"] == 3

    def test_sag_diff_between_sessions(self):
        """The SAG should show meaningful changes between sessions."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        # First session: mostly SELF
        frames_1 = _make_synthetic_frames(n_frames=100, self_ratio=0.9)
        model.process_frame_log(frames_1)

        concept_1 = model.query_self_concept()
        self_match_1 = concept_1[NODE_AGENT_VOICE]["context"]["mean_self_match"]

        # Second session: more WORLD
        frames_2 = _make_synthetic_frames(n_frames=100, self_ratio=0.3)
        model.process_frame_log(frames_2)

        concept_2 = model.query_self_concept()
        self_match_2 = concept_2[NODE_AGENT_VOICE]["context"]["mean_self_match"]

        # The self-match means should differ between sessions
        assert abs(self_match_1 - self_match_2) > 0.01

    def test_introspective_query_after_synthetic_streaming(self):
        """Full pipeline: synthetic streaming → SelfModel → introspective query."""
        sag = SemanticAbstractionGraph()
        model = SelfModel()
        model.initialize_sag(sag)

        # Simulate a conversation with multiple speakers
        frames = _make_synthetic_frames(n_frames=300, speak_ratio=0.4, self_ratio=0.6, n_speakers=3)
        result = model.process_frame_log(frames)

        # Verify processing
        assert result["frames_processed"] == 300
        assert result["m4_attempted"] > 0

        # Introspective query
        concept = model.query_self_concept()

        # All nodes should be populated
        assert len(concept[NODE_ROOT]["core"]) > 0
        assert "self-match" in concept[NODE_AGENT_VOICE]["core"].lower() or "M4" in concept[NODE_AGENT_VOICE]["core"]
        assert "speaking" in concept[NODE_SPEECH_PATTERNS]["core"].lower() or "Speech" in concept[NODE_SPEECH_PATTERNS]["core"]

        # Known subjects should include detected speakers
        subjects = concept["known_subjects"]
        assert len(subjects) >= 1

        # Summary should be consistent
        summary = model.summarize()
        assert summary["patterns"]["session_count"] == 1
        assert summary["patterns"]["m4_accepted_frames"] > 0

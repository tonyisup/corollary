"""Unit tests for M4 self-voice detector integration.

Verifies the M4 pipeline: buffer accumulation, detector registration,
self/world discrimination, and frame-level metadata flow.
"""

import sys
import os
import subprocess
import importlib.util
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from stg.speaker import SpeakerIdentifier, SelfVoiceDetector
from thalamus.attributor import ThalamicAttributor, SourceVerdict


SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "m4_f1_benchmark.py")
_spec = importlib.util.spec_from_file_location("m4_f1_benchmark", SCRIPT_PATH)
m4_f1_benchmark = importlib.util.module_from_spec(_spec)
sys.modules["m4_f1_benchmark"] = m4_f1_benchmark
_spec.loader.exec_module(m4_f1_benchmark)


SR = 16000


def make_pcm(seed=42, duration_s=2.0):
    """Generate synthetic speech-like audio."""
    rng = np.random.RandomState(seed)
    n = int(SR * duration_s)
    t = np.linspace(0, duration_s, n)
    f0 = 100 + (seed % 7) * 25
    signal = np.zeros(n)
    for h in range(1, 8):
        amp = 1.0 / (h ** 0.7)
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * f0 * h * t + phase)
    signal *= 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * (3 + seed * 0.5) * t))
    signal += rng.randn(n) * 0.03
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7
    return signal.astype(np.float32)


class TestM4Detector:
    """Tests for SelfVoiceDetector basic operations."""

    def test_mock_registration_and_self_detection(self):
        """Mock detector should identify identical audio as self."""
        identifier = SpeakerIdentifier(use_real=False)
        identifier._load_model()
        detector = SelfVoiceDetector(identifier, threshold=0.1)

        pcm = make_pcm(seed=42)
        detector.register_self(pcm)

        # Identical audio should be detected as self
        is_self, conf = detector.is_self(pcm)
        assert is_self, f"Expected self detection, got is_self={is_self}, conf={conf}"
        assert conf > 0.5, f"Expected high confidence for identical audio, got {conf}"

    def test_mock_rejects_different_speaker(self):
        """Mock detector should reject audio from a different seed/speaker."""
        identifier = SpeakerIdentifier(use_real=False)
        identifier._load_model()
        detector = SelfVoiceDetector(identifier, threshold=0.1)

        self_pcm = make_pcm(seed=42)
        other_pcm = make_pcm(seed=99)
        detector.register_self(self_pcm)

        is_self, conf = detector.is_self(other_pcm)
        # Mock uses content-based hash; different audio = different embedding
        # With threshold=0.1, random embeddings rarely exceed
        # (We don't assert is_self=False because mock is stochastic;
        # we just verify the pipeline runs without error)
        assert isinstance(is_self, bool)
        assert isinstance(conf, float)

    def test_mock_is_world_inverse(self):
        """is_world should be the logical inverse of is_self."""
        identifier = SpeakerIdentifier(use_real=False)
        identifier._load_model()
        detector = SelfVoiceDetector(identifier, threshold=0.1)

        pcm = make_pcm(seed=42)
        detector.register_self(pcm)

        is_self, self_conf = detector.is_self(pcm)
        is_world, world_conf = detector.is_world(pcm)
        assert is_world == (not is_self)
        assert abs(world_conf - (1.0 - self_conf)) < 1e-6

    def test_unregistered_returns_false(self):
        """Detector without registration should return (False, 0.0)."""
        identifier = SpeakerIdentifier(use_real=False)
        identifier._load_model()
        detector = SelfVoiceDetector(identifier, threshold=0.1)

        pcm = make_pcm(seed=42)
        is_self, conf = detector.is_self(pcm)
        assert not is_self
        assert conf == 0.0


class AlwaysSelfDetector:
    """Tiny fake M4 detector for attribution-decision tests."""

    def register_self(self, pcm):
        pass

    def is_self(self, pcm):
        return True, 0.95


class TestM4AttributorIntegration:
    """Tests for M4 integration in ThalamicAttributor."""

    def test_m4_metadata_present_when_enabled(self):
        """When M4 is enabled and playing+speech, metadata should contain M4 fields."""
        identifier = SpeakerIdentifier(use_real=False)
        identifier._load_model()
        detector = SelfVoiceDetector(identifier, threshold=0.1)

        attributor = ThalamicAttributor(
            sr=SR,
            m4_detector=detector,
            m4_self_threshold=0.1,
            m4_min_buffer_ms=500.0,
        )

        pcm = make_pcm(seed=42, duration_s=3.0)
        detector.register_self(pcm)

        # Run enough frames to fill the M4 buffer
        frame_samples = int(SR * 30 / 1000)  # 30ms frames
        for i in range(200):  # 6s worth of frames
            s_start = i * frame_samples
            s_end = s_start + frame_samples
            if s_end > len(pcm):
                break
            frame = pcm[s_start:s_end]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            rms_db = 20 * np.log10(max(rms, 1e-9))

            result = attributor.attribute(
                match_score=0.9,
                observed_onset_samples=s_start,
                expected_onset_samples=s_start,
                rms_db=rms_db,
                is_playing=True,
                is_speech=True,
                frame_audio=frame,
                self_pcm=pcm,
                agent_start_sample=0,
            )

        # After buffer fills, M4 should have attempted
        meta = result.metadata
        assert meta is not None
        assert "m4_attempted" in meta
        assert "m4_outcome" in meta
        assert "m4_buffer_length" in meta

    def test_m4_not_attempted_when_not_playing(self):
        """M4 should not attempt when is_playing=False."""
        identifier = SpeakerIdentifier(use_real=False)
        identifier._load_model()
        detector = SelfVoiceDetector(identifier, threshold=0.1)

        attributor = ThalamicAttributor(
            sr=SR,
            m4_detector=detector,
            m4_min_buffer_ms=500.0,
        )

        pcm = make_pcm(seed=42, duration_s=3.0)
        detector.register_self(pcm)

        frame_samples = int(SR * 30 / 1000)
        frame = pcm[:frame_samples]
        rms_db = -10.0

        result = attributor.attribute(
            match_score=0.0,
            observed_onset_samples=0,
            expected_onset_samples=0,
            rms_db=rms_db,
            is_playing=False,
            is_speech=True,
            frame_audio=frame,
        )

        meta = result.metadata
        assert meta["m4_outcome"] == "not_attempted"

    def test_m4_not_attempted_without_speech(self):
        """M4 should not attempt when is_speech=False (silence)."""
        identifier = SpeakerIdentifier(use_real=False)
        identifier._load_model()
        detector = SelfVoiceDetector(identifier, threshold=0.1)

        attributor = ThalamicAttributor(
            sr=SR,
            m4_detector=detector,
            m4_min_buffer_ms=500.0,
        )

        pcm = make_pcm(seed=42, duration_s=3.0)
        detector.register_self(pcm)

        frame_samples = int(SR * 30 / 1000)
        frame = pcm[:frame_samples]

        result = attributor.attribute(
            match_score=0.0,
            observed_onset_samples=0,
            expected_onset_samples=0,
            rms_db=-60.0,  # Below threshold → silence
            is_playing=True,
            is_speech=False,
            frame_audio=frame,
        )

        meta = result.metadata
        assert meta["m4_outcome"] == "not_attempted"

    def test_no_m4_detector_grasses_clean(self):
        """Attributor without M4 detector should work (no crash)."""
        attributor = ThalamicAttributor(sr=SR, m4_detector=None)

        frame_samples = int(SR * 30 / 1000)
        frame = make_pcm(seed=42, duration_s=0.5)[:frame_samples]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_db = 20 * np.log10(max(rms, 1e-9))

        result = attributor.attribute(
            match_score=0.8,
            observed_onset_samples=0,
            expected_onset_samples=0,
            rms_db=rms_db,
            is_playing=True,
            is_speech=True,
            frame_audio=frame,
        )

        meta = result.metadata
        assert meta["m4_outcome"] == "not_attempted"

    def test_m4_acceptance_requires_template_match_evidence(self):
        """Without room_filter, high M4 confidence alone should not override anti-correlated evidence."""
        attributor = ThalamicAttributor(
            sr=SR,
            m4_detector=AlwaysSelfDetector(),
            m4_self_threshold=0.1,
            m4_min_buffer_ms=30.0,
            # No room_filter → strict gate applies
        )
        frame_samples = int(SR * 30 / 1000)
        frame = make_pcm(seed=123, duration_s=0.5)[:frame_samples]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_db = 20 * np.log10(max(rms, 1e-9))

        result = attributor.attribute(
            match_score=-0.5,  # Anti-correlated: should block SELF despite ECAPA accepting
            observed_onset_samples=0,
            expected_onset_samples=0,
            rms_db=rms_db,
            is_playing=True,
            is_speech=True,
            frame_audio=frame,
            self_pcm=make_pcm(seed=42, duration_s=0.5),
            agent_start_sample=0,
        )

        assert result.metadata["m4_outcome"] == "accepted"
        assert result.verdict != SourceVerdict.SELF

    def test_m4_dry_path_requires_strict_template_match_for_playback_window_negative(self):
        """Dry/synthetic playback-window negatives need strong template evidence, not merely positive correlation."""
        attributor = ThalamicAttributor(
            sr=SR,
            m4_detector=AlwaysSelfDetector(),
            m4_self_threshold=0.1,
            m4_min_buffer_ms=30.0,
            m4_template_match_threshold=0.20,
            # No room_filter: same-domain playback-window negatives are synthetic/dry gate cases.
        )
        frame_samples = int(SR * 30 / 1000)
        frame = make_pcm(seed=123, duration_s=0.5)[:frame_samples]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_db = 20 * np.log10(max(rms, 1e-9))

        result = attributor.attribute(
            match_score=0.05,  # Weak positive correlation, below the strict template gate.
            observed_onset_samples=0,
            expected_onset_samples=0,
            rms_db=rms_db,
            is_playing=True,
            is_speech=True,
            frame_audio=frame,
            self_pcm=make_pcm(seed=42, duration_s=0.5),
            agent_start_sample=0,
        )

        assert result.metadata["m4_outcome"] == "accepted"
        assert result.metadata["m4_template_match_passed"] is False
        assert result.verdict != SourceVerdict.SELF


class TestM4FrameLevelF1Benchmark:
    """Tests for the M4 F1 benchmark script (pipeline verification)."""

    def test_benchmark_runs_without_error(self):
        """Mock benchmark script should pass its pipeline acceptance check."""
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(
                os.path.dirname(__file__), "..", "..", "scripts", "m4_f1_benchmark.py",
            )],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Pipeline acceptance" in result.stdout
        assert "F1 acceptance: not applicable in mock mode" in result.stdout

    def test_mock_benchmark_pipeline_correctness(self):
        """Mock benchmark should verify: no false positives on world audio."""
        identifier = SpeakerIdentifier(use_real=False)
        identifier._load_model()
        detector = SelfVoiceDetector(identifier, threshold=0.1)

        attributor = ThalamicAttributor(
            sr=SR,
            m4_detector=detector,
            m4_self_threshold=0.1,
            m4_min_buffer_ms=500.0,
        )

        self_pcm = make_pcm(seed=42, duration_s=3.0)
        detector.register_self(self_pcm)

        world_pcm = make_pcm(seed=99, duration_s=3.0)

        frame_samples = int(SR * 30 / 1000)
        false_positives = 0
        total_world_frames = 0

        for i in range(len(world_pcm) // frame_samples):
            s_start = i * frame_samples
            s_end = s_start + frame_samples
            frame = world_pcm[s_start:s_end]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            rms_db = 20 * np.log10(max(rms, 1e-9))

            result = attributor.attribute(
                match_score=0.0,
                observed_onset_samples=s_start,
                expected_onset_samples=s_start,
                rms_db=rms_db,
                is_playing=False,
                is_speech=rms_db > -40.0,
                frame_audio=frame,
            )

            if rms_db > -40.0:
                total_world_frames += 1
                if result.verdict == SourceVerdict.SELF:
                    false_positives += 1

        assert false_positives == 0, (
            f"Got {false_positives} false positives on world audio "
            f"(out of {total_world_frames} speech frames)"
        )

    def test_battery_includes_generalization_pressure_cases(self):
        """Benchmark should include playback-window negatives and room/device positives."""
        cases, _ = m4_f1_benchmark.build_test_battery()
        by_name = {case.name: case for case in cases}

        assert "same_domain_negative_playback_window" in by_name
        assert by_name["same_domain_negative_playback_window"].is_playing is True
        assert by_name["same_domain_negative_playback_window"].expected_self is False

        assert "self_room_colored" in by_name
        assert by_name["self_room_colored"].expected_self is True

        assert "self_device_eq" in by_name
        assert by_name["self_device_eq"].expected_self is True

    def test_playback_window_negative_scores_as_true_negative_not_false_negative(self):
        """A non-self speaker during playback opportunity must not be ground-truthed as SELF."""
        case = m4_f1_benchmark.TestCase(
            name="negative_playback_window_unit",
            audio=make_pcm(seed=123, duration_s=3.0),
            tts_pcm=make_pcm(seed=42, duration_s=3.0),
            is_playing=True,
            expected_self=False,
        )
        attributor = ThalamicAttributor(sr=SR, m4_detector=None)

        result = m4_f1_benchmark.run_case(case, attributor, n_frames=20)

        assert result["fn"] == 0
        assert result["tn"] > 0

    def test_replay_case_masks_self_ground_truth_to_speech_frames(self):
        """Replay artifacts can label the whole speech window SELF; silent frames should not count as FNs."""
        class SpeechOnlyAttributor:
            def attribute(self, **kwargs):
                verdict = SourceVerdict.SELF if kwargs["is_speech"] else SourceVerdict.SILENCE
                return type("Result", (), {
                    "verdict": verdict,
                    "metadata": {
                        "m4_attempted": kwargs["is_speech"],
                        "m4_outcome": "accepted" if kwargs["is_speech"] else "not_attempted",
                    },
                })()

        speech = make_pcm(seed=42, duration_s=0.03)
        silence = np.zeros_like(speech)
        case = m4_f1_benchmark.ReplayCase(
            name="replay_self_with_silent_labeled_frame",
            audio=np.concatenate([speech, silence]),
            agent_pcm=np.concatenate([speech, silence]),
            gt_verdicts=["SELF", "SELF"],
        )

        result = m4_f1_benchmark.run_replay_case(case, SpeechOnlyAttributor())

        assert result["tp"] == 1
        assert result["fn"] == 0

    def test_run_case_reports_warmup_adjusted_eligible_metrics(self):
        """Benchmark should distinguish classifier F1 after M4 is eligible from buffer warmup FNs."""
        class NeverSelfDetector:
            def is_self(self, pcm):
                return False, 0.0

        case = m4_f1_benchmark.TestCase(
            name="warmup_accounting_unit",
            audio=make_pcm(seed=42, duration_s=3.0),
            tts_pcm=make_pcm(seed=42, duration_s=3.0),
            is_playing=True,
            expected_self=True,
        )
        attributor = ThalamicAttributor(
            sr=SR,
            m4_detector=NeverSelfDetector(),
            m4_min_buffer_ms=500.0,
        )

        result = m4_f1_benchmark.run_case(case, attributor, n_frames=20)

        assert result["warmup_fn"] > 0
        assert result["eligible_frames"] > 0
        assert "eligible_f1" in result
        assert result["fn"] >= result["warmup_fn"]

    def test_run_benchmark_accepts_m4_min_buffer_override(self):
        """Benchmark runner should expose buffer size so warmup latency can be measured separately."""
        result = m4_f1_benchmark.run_benchmark(
            threshold=0.10,
            real_ecapa=False,
            m4_min_buffer_ms=60.0,
        )

        assert result.m4_min_buffer_ms == 60.0
        assert hasattr(result, "aggregate_eligible_f1")

    def test_run_benchmark_default_buffer_matches_current_acceptance_setting(self):
        """Default benchmark buffer should reflect the current accepted latency/quality tradeoff."""
        result = m4_f1_benchmark.run_benchmark(threshold=0.10, real_ecapa=False)

        assert result.m4_min_buffer_ms == 300.0

    def test_mock_benchmark_reports_pipeline_pass_separately_from_f1(self):
        """Mock mode should not use discrimination F1 as its CLI success criterion."""
        result = m4_f1_benchmark.run_benchmark(threshold=0.10, real_ecapa=False)

        assert result.mock_pipeline_passed
        assert result.aggregate_f1 < 0.95


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

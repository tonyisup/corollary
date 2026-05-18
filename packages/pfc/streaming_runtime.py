"""PFC — Streaming Runtime for real-time M3 turn-taking.

Production streaming callback loop that gates TTS output in real-time
based on VAD → attribution → controller decisions.

Unlike the offline harness (which pre-plans TTS segments from known scenario
timing), the streaming runtime makes gating decisions online in the audio
callback. This is the bridge between artifact-backed validation and a
production streaming system.

Usage:
    from pfc.streaming_runtime import StreamingRuntime, StreamingConfig

    config = StreamingConfig(
        tts_pcm=agent_pcm,
        tts_text="Hello, I'm Corollary.",
        input_device=3,
        output_device=5,
        m4_enable=True,
        gated_playback=True,
        save_path=Path("/tmp/streaming_log.jsonl"),
    )
    runtime = StreamingRuntime(config)
    runtime.run()
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from thalamus.attributor import AttributionResult, ThalamicAttributor, SourceVerdict
from cochlea.vad import SmoothedVAD, VAD_FRAME_SAMPLES
from pfc.agentive_controller import AgentiveController, AgentState


@dataclass
class StreamingConfig:
    """Configuration for the streaming runtime."""

    # TTS audio (pre-synthesized)
    tts_pcm: np.ndarray = None
    tts_text: str = ""

    # Optional pre-synthesized repair cue audio. This is queued separately from
    # the main utterance so it can play once after a collision without consuming
    # or clearing the deferred original utterance.
    repair_pcm: Optional[np.ndarray] = None

    # Audio devices
    input_device: int = 3
    output_device: int = 5
    sr: int = 16000

    # TTS chunking
    tts_chunk_ms: float = 200.0  # TTS consumption unit; internally split into VAD frames
    # Hardware callback blocksize. Smaller = lower interruption latency but more
    # callback overhead. Must be evenly divisible by VAD_FRAME_SAMPLES (480).
    # Default 120 ms (1920 samples = 4 × 480). Previous default was 390 ms (6240).
    callback_blocksize_ms: float = 120.0
    output_request_s: float = 0.0  # when the agent first intends to speak

    # Feature flags
    m4_enable: bool = False
    gated_playback: bool = True  # if False, always play TTS (no gating)

    # M4 params
    m4_self_threshold: float = 0.50
    m4_min_buffer_ms: float = 60.0
    m4_scorer_path: Optional[str] = None  # Path to fine-tuned scorer.pt (feature-based)

    # Room filter
    room_filter_path: Optional[str] = None
    room_local_threshold: Optional[float] = None

    # M7 trust-gated room filter adaptation. Disabled by default; when enabled,
    # room_adapter should expose update_trust_gated(...) and get_learned_ir().
    enable_room_filter_adaptation: bool = False
    room_adapter: Optional[Any] = None
    room_adapt_confidence_threshold: float = 0.80
    room_adapt_match_threshold: float = 0.20

    # Controller params
    listen_threshold_ms: float = 100.0
    cooldown_ms: float = 200.0
    both_external_min_frames: int = 7
    both_external_energy_ratio: float = 2.0
    collision_repair_text: Optional[str] = "Sorry, go ahead."

    # Attributor params
    energy_threshold_db: float = -55.0
    similarity_threshold: float = 0.05
    causal_tolerance_ms: float = 175.0

    # Logging
    save_path: Optional[Path] = None  # JSONL log file

    # Self-model (M6)
    self_model: Optional[Any] = None  # SelfModel instance — observes frame logs after run

    # Hippocampus (M5)
    hippocampus: Optional[Any] = None  # Hippocampus instance — episodic memory encoding

    # Per-speaker adaptive threshold (M4+)
    speaker_threshold_adapter: Optional[Any] = None  # SpeakerThresholdAdapter instance

    # Conversational registration
    enable_conversational_registration: bool = False  # Auto-register new speakers from WORLD audio
    registration_speaker_name: str = "User"  # Default name for auto-registered speakers


@dataclass
class StreamingFrameLog:
    """Per-frame log entry for post-hoc analysis."""
    frame_index: int
    timestamp_s: float
    vad_speech: bool
    verdict: str
    controller_state: str
    can_speak: bool
    should_interrupt: bool
    tts_playing: bool
    tts_position_s: float
    tts_total_duration_s: float
    output_source: str
    repair_position_s: float
    repair_total_duration_s: float
    repair_queued: bool
    repair_playing: bool
    repair_completed: bool
    m4_outcome: str
    m4_confidence: Optional[float]
    energy_db: float
    self_match_score: float
    self_match_score_per_frame: Optional[float]
    is_in_motor_window: bool
    deferred: bool
    resumed: bool
    collision_repair_text: Optional[str] = None
    speaker_id: Optional[str] = None  # Identified speaker for this frame
    speaker_best_match_id: Optional[str] = None
    speaker_best_match_similarity: Optional[float] = None
    speaker_match_passed: Optional[bool] = None
    speaker_registered: Optional[str] = None  # Set to speaker_id when a new speaker is auto-registered
    room_adapted: Optional[bool] = None  # M7: whether this frame updated the room filter
    room_adaptation_reason: Optional[str] = None
    room_adaptation_error: Optional[float] = None
    room_adaptation_updates: Optional[int] = None


@dataclass(frozen=True)
class SyntheticAttributionFrame:
    """Synthetic attribution frame for deterministic streaming validation.

    This bypasses room acoustics and VAD so tests can validate online
    controller/TTS-gate behavior directly. It complements offline audio replay,
    which validates callback plumbing but can under-exercise SPEAK detection when
    room-colored captures are compared against dry TTS templates.
    """

    time_s: float
    verdict: SourceVerdict
    is_speech: bool
    energy_db: float = -30.0
    self_match_score: float = 0.0
    self_match_score_per_frame: Optional[float] = None
    is_in_motor_window: bool = False
    causal_score: float = 0.0
    confidence: float = 0.9
    m4_outcome: str = "not_attempted"
    m4_confidence: Optional[float] = None
    speaker_id: Optional[str] = None  # Per-speaker adaptive threshold ID


class StreamingRuntime:
    """Real-time streaming callback loop for M3 turn-taking.

    Opens an sd.Stream and in each callback:
    1. Captures mic frame
    2. Runs VAD
    3. Runs attribution (with optional M4)
    4. Feeds attribution to controller
    5. Gates TTS output based on controller decision
    6. Logs structured metadata
    """

    def __init__(self, config: StreamingConfig) -> None:
        self.config = config
        self.sr = config.sr

        # TTS state
        self._tts_pcm = config.tts_pcm.astype(np.float32) if config.tts_pcm is not None else None
        self._tts_text = config.tts_text
        self._tts_position = 0  # current sample position in TTS PCM
        self._tts_chunk_samples = int(config.tts_chunk_ms * self.sr / 1000.0)
        self._tts_total_samples = len(self._tts_pcm) if self._tts_pcm is not None else 0

        # Collision repair audio state. Repair audio is a separate suppressible
        # source from the original TTS utterance: it can play once after a
        # collision, while the original utterance remains deferred for resume.
        self._repair_pcm = config.repair_pcm.astype(np.float32) if config.repair_pcm is not None else None
        self._repair_position = 0
        self._repair_total_samples = len(self._repair_pcm) if self._repair_pcm is not None else 0
        self._repair_pending = False
        self._repair_started = False
        self._repair_completed = False

        # Pipeline components
        self._vad = SmoothedVAD()
        self._controller = AgentiveController(
            listen_threshold_ms=config.listen_threshold_ms,
            cooldown_ms=config.cooldown_ms,
            both_external_min_frames=config.both_external_min_frames,
            both_external_energy_ratio_threshold=config.both_external_energy_ratio,
            collision_repair_text=config.collision_repair_text,
        )

        # Room filter
        room_filter = None
        room_local_threshold = config.room_local_threshold
        if config.room_filter_path:
            room_filter = np.load(config.room_filter_path)
            if room_local_threshold is None:
                room_local_threshold = 0.04

        # M4 detector
        m4_detector = None
        m4_identifier = None
        if config.m4_enable:
            try:
                if config.m4_scorer_path is not None:
                    # Use fine-tuned scorer (real ECAPA + trained projection head)
                    from stg.speaker import SpeakerIdentifier, FineTunedSelfVoiceDetector
                    m4_identifier = SpeakerIdentifier(use_real=True, backend="ecapa")
                    m4_detector = FineTunedSelfVoiceDetector(m4_identifier, threshold=config.m4_self_threshold)
                    m4_detector.load_scorer(config.m4_scorer_path)
                    print(f"  M4: loaded fine-tuned scorer from {config.m4_scorer_path}")
                else:
                    # Fallback to mock detector (hash-based, no real ECAPA)
                    from stg.speaker import SpeakerIdentifier, SelfVoiceDetector
                    m4_identifier = SpeakerIdentifier(use_real=False)
                    m4_detector = SelfVoiceDetector(m4_identifier, threshold=config.m4_self_threshold)
                if self._tts_pcm is not None:
                    m4_detector.register_self(self._tts_pcm)
            except Exception as e:
                print(f"  WARNING: M4 detector failed to initialize: {e}")
                m4_detector = None
                m4_identifier = None

        # Speaker identification for per-speaker adaptive threshold
        speaker_identifier = None
        speaker_world_model = None
        if config.speaker_threshold_adapter is not None:
            try:
                from stg.speaker import SpeakerIdentifier
                from pfc.world_model import SpeakerRegistry, MultiSpeakerWorldModel
                speaker_identifier = SpeakerIdentifier(use_real=True)
                registry = SpeakerRegistry(
                    identifier=speaker_identifier,
                    self_speaker_id="self",
                )
                speaker_world_model = MultiSpeakerWorldModel(registry=registry)
                if self._tts_pcm is not None:
                    speaker_world_model.registry.register_speaker(
                        "self", "self", embedding=None,
                    )
            except Exception as e:
                print(f"  WARNING: Speaker identification failed to initialize: {e}")
                speaker_identifier = None
                speaker_world_model = None

        self._attributor = ThalamicAttributor(
            energy_threshold_db=config.energy_threshold_db,
            similarity_threshold=config.similarity_threshold,
            causal_tolerance_ms=config.causal_tolerance_ms,
            sr=self.sr,
            m4_detector=m4_detector,
            m4_self_threshold=config.m4_self_threshold,
            m4_min_buffer_ms=config.m4_min_buffer_ms,
            room_filter=room_filter,
            room_local_threshold=room_local_threshold,
            speaker_threshold_adapter=config.speaker_threshold_adapter,
        )

        # M7 trust-gated room filter adaptation
        self._room_adapter = config.room_adapter
        self._enable_room_filter_adaptation = config.enable_room_filter_adaptation
        self._room_adapt_confidence_threshold = config.room_adapt_confidence_threshold
        self._room_adapt_match_threshold = config.room_adapt_match_threshold
        self._last_room_adaptation_result: Optional[dict] = None

        # Per-speaker adaptive threshold adapter (also stored for post-run inspection)
        self._speaker_threshold_adapter = config.speaker_threshold_adapter

        # Speaker identification state for per-speaker adaptive threshold
        self._speaker_identifier = speaker_identifier
        self._speaker_world_model = speaker_world_model
        self._current_speaker_id: Optional[str] = None
        self._last_speaker_best_match_id: Optional[str] = None
        self._last_speaker_best_match_similarity: Optional[float] = None
        self._last_speaker_match_passed: Optional[bool] = None
        # WORLD audio buffer: accumulates non-self speech for speaker identification
        self._speaker_id_buffer: Optional[np.ndarray] = None
        self._speaker_id_interval_frames = 20  # Run speaker ID every 20 VAD frames (~200ms)
        self._speaker_id_frame_counter = 0

        # Conversational registration state
        self._conversational_registration = config.enable_conversational_registration
        self._registration_speaker_name = config.registration_speaker_name
        self._registration_complete = False
        self._registration_min_samples = int(0.3 * self.sr)  # 0.3s of WORLD speech
        self._registration_min_similarity = 0.0

        # Logging
        self._logs: list[StreamingFrameLog] = []
        self._frame_index = 0
        self._start_time = 0.0

        # Deferred output tracking
        self._output_requested = False
        self._was_deferred = False
        self._was_resumed = False

        # Live hardware runtime state. The CoreAudio callback must stay
        # realtime-safe, so run() uses this lock only for short controller/TTS
        # state transitions; heavy VAD/M4/attribution runs on a worker thread.
        self._state_lock = threading.RLock()
        self._reset_live_diagnostics()

    def _reset_live_diagnostics(self) -> None:
        """Reset live-run timing and async-analysis diagnostics."""
        self._live_sample_cursor = 0
        self._live_callback_count = 0
        self._live_analysis_blocks_enqueued = 0
        self._live_analysis_blocks_dropped = 0
        self._live_analysis_queue_max_depth = 0
        self._live_analysis_worker_errors: list[str] = []

    def _claim_live_callback_timestamp(self, frame_samples: int) -> float:
        """Return callback start time from a monotonic sample clock.

        Wall-clock callback cadence can jitter; frame logs should reflect the
        audio stream timeline, not scheduler timing. The callback owns this
        method, so no heavyweight locking is required.
        """
        timestamp_s = self._live_sample_cursor / self.sr
        self._live_sample_cursor += int(frame_samples)
        self._live_callback_count += 1
        return timestamp_s

    def _update_room_filter_adaptation(
        self,
        *,
        known_pcm: np.ndarray,
        captured_pcm: np.ndarray,
        attr_result: AttributionResult,
        self_match_score: float,
    ) -> Optional[dict]:
        """M7 phase 1: trust-gated room filter adaptation.

        The room adapter may update only when explicitly enabled and when the
        attribution evidence says this frame is trusted SELF. The adapter owns
        the detailed gate; the runtime provides the current attribution verdict,
        M4 confidence, and room-local/self-match score. On successful updates,
        the learned IR is published back into the attributor for subsequent
        room-local scoring.
        """
        if not self._enable_room_filter_adaptation or self._room_adapter is None:
            return None

        if known_pcm.size == 0 or float(np.sqrt(np.mean(known_pcm ** 2))) < 1e-8:
            result = {
                "adapted": False,
                "reason": "no_efference",
                "error": None,
                "updates": getattr(self._room_adapter, "total_updates", None),
            }
            self._last_room_adaptation_result = result
            return result

        metadata = attr_result.metadata or {}
        match_candidates = [
            attr_result.room_local_score,
            attr_result.self_match_score_per_frame,
            self_match_score,
        ]
        finite_matches = [
            float(score)
            for score in match_candidates
            if score is not None and np.isfinite(float(score))
        ]
        # Live playback often has stale/low room-local scores while the
        # per-frame self evidence and M4 confidence are strong. Use the best
        # available finite match signal for the trust gate, while still requiring
        # SELF verdict + M4 confidence in the adapter itself.
        match_score = max(finite_matches) if finite_matches else None

        result = self._room_adapter.update_trust_gated(
            known_pcm.astype(np.float32),
            captured_pcm.astype(np.float32),
            verdict=attr_result.verdict.name,
            m4_confidence=metadata.get("m4_confidence"),
            self_match_score=match_score,
            adapt_confidence_threshold=self._room_adapt_confidence_threshold,
            adapt_match_threshold=self._room_adapt_match_threshold,
        )
        self._last_room_adaptation_result = result

        if result.get("adapted") and hasattr(self._room_adapter, "get_learned_ir"):
            self._attributor.room_filter = self._room_adapter.get_learned_ir()
            if self._attributor.room_local_threshold is None:
                self._attributor.room_local_threshold = 0.04

        return result

    def _update_speaker_id(
        self,
        vad_frame: np.ndarray,
        attr_result,
    ) -> Optional[str]:
        """Run speaker identification for per-speaker adaptive threshold.

        Accumulates audio frames for speaker embeddings and periodically runs
        the speaker identification pipeline to determine the current speaker.
        Cold-start conversational registration uses clean WORLD frames only;
        after a non-self speaker is known, BOTH frames are also eligible so
        speaker ID can persist through overlap. The result is stored in
        `_current_speaker_id` and passed to the attributor on subsequent frames.

        When conversational registration is enabled and no human speaker is
        yet registered, auto-registers a new speaker from accumulated WORLD
        audio once enough clean speech has been collected.

        Returns the speaker_id if a new speaker was registered this cycle,
        or None otherwise.
        """
        if self._speaker_world_model is None or self._speaker_identifier is None:
            return

        self._speaker_id_frame_counter += 1

        # Accumulate speaker-ID audio on every eligible frame so the buffer can
        # actually reach the minimum duration needed for embeddings. Cold-start
        # conversational registration must stay WORLD-only to avoid registering
        # a self+human overlap mixture. Once a non-self speaker is known, BOTH
        # frames are useful for ongoing identification during overlap.
        verdict = attr_result.verdict
        registry = self._speaker_world_model.registry
        known_non_self = any(s != "self" for s in registry.known_speakers)
        eligible_for_speaker_id = verdict == SourceVerdict.WORLD or (
            verdict == SourceVerdict.BOTH and known_non_self
        )
        if eligible_for_speaker_id:
            if self._speaker_id_buffer is None:
                self._speaker_id_buffer = vad_frame.copy()
            else:
                self._speaker_id_buffer = np.concatenate(
                    [self._speaker_id_buffer, vad_frame]
                )
            # Keep buffer from growing unbounded (max ~2s of audio)
            max_buffer_samples = int(2.0 * self.sr)
            if len(self._speaker_id_buffer) > max_buffer_samples:
                self._speaker_id_buffer = self._speaker_id_buffer[-max_buffer_samples:]

        # Only run speaker identification periodically
        if self._speaker_id_frame_counter < self._speaker_id_interval_frames:
            return
        self._speaker_id_frame_counter = 0

        # Need enough audio for a reliable embedding (at least 0.5s)
        min_samples = int(0.3 * self.sr)
        if self._speaker_id_buffer is None or len(self._speaker_id_buffer) < min_samples:
            return None

        try:
            # Compute embedding and inspect registry match quality. We keep the
            # best-match diagnostics even when similarity falls below threshold
            # so live runs can distinguish "no resemblance" from
            # "candidate exists but is too weak".
            embedding = self._speaker_identifier.get_embedding(self._speaker_id_buffer)
            registry = self._speaker_world_model.registry
            best_id, best_similarity = registry.best_match(embedding)
            self._last_speaker_best_match_id = best_id
            self._last_speaker_best_match_similarity = best_similarity

            speaker_id, similarity = registry.identify_speaker(embedding)
            self._last_speaker_match_passed = speaker_id is not None and speaker_id != "self"
            if speaker_id is not None and speaker_id != "self":
                self._current_speaker_id = speaker_id
            elif (
                self._conversational_registration
                and not self._registration_complete
                and speaker_id is None
            ):
                # No existing speaker matched — auto-register from accumulated WORLD audio.
                # Only register once per conversation (first time we get enough clean speech).
                if (
                    self._speaker_id_buffer is not None
                    and len(self._speaker_id_buffer) >= self._registration_min_samples
                ):
                    new_id = (
                        f"human_{self._registration_speaker_name.lower().replace(' ', '_')}"
                    )
                    # Don't register if it would collide with "self"
                    if new_id == "self":
                        new_id = "human_user"
                    # Check if already registered
                    if new_id not in registry.known_speakers:
                        registry.register_speaker(
                            new_id, self._registration_speaker_name, embedding=embedding
                        )
                        self._registration_complete = True
                        self._current_speaker_id = new_id
                        self._last_speaker_match_passed = True
                        return new_id
        except Exception:
            # Speaker identification is best-effort; don't crash the audio callback
            pass
        return None

    def _update_adapter_from_clean_world_evidence(
        self,
        attr_result: AttributionResult,
        speaker_id: Optional[str],
    ) -> None:
        """Update non-self adaptive-threshold profiles from clean WORLD evidence.

        This is deliberately separated from the attributor's M4 frame path.
        Non-self profile learning should only happen after verdict resolution on
        frames that the system already considers clean WORLD, avoiding overlap-
        contaminated BOTH frames from ratcheting non-self thresholds upward.
        """
        if self._speaker_threshold_adapter is None or speaker_id in (None, "", "self"):
            return
        if attr_result.verdict != SourceVerdict.WORLD:
            return
        m4_metadata = attr_result.metadata or {}
        m4_conf = m4_metadata.get("m4_confidence")
        if m4_conf is None:
            return
        self._speaker_threshold_adapter.update(speaker_id, float(m4_conf))

    def _get_tts_chunk(self) -> np.ndarray:
        """Return the next TTS chunk to emit, or zeros if done/gated."""
        if self._tts_pcm is None or self._tts_position >= self._tts_total_samples:
            return np.zeros(self._tts_chunk_samples, dtype=np.float32)

        end = min(self._tts_position + self._tts_chunk_samples, self._tts_total_samples)
        chunk = self._tts_pcm[self._tts_position:end]
        self._tts_position = end

        # Pad last chunk if needed
        if len(chunk) < self._tts_chunk_samples:
            padded = np.zeros(self._tts_chunk_samples, dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded

        return chunk

    def _compute_normalized_xcorr(self, observed: np.ndarray, reference: np.ndarray) -> float:
        """Small local normalized cross-correlation helper.

        Kept local so streaming runtime does not import validation scripts with
        heavyweight/side-effect dependencies.
        """
        n = min(len(observed), len(reference))
        if n <= 0:
            return 0.0
        obs = observed[:n].astype(np.float32)
        ref = reference[:n].astype(np.float32)
        obs_norm = float(np.linalg.norm(obs))
        ref_norm = float(np.linalg.norm(ref))
        if obs_norm < 1e-9 or ref_norm < 1e-9:
            return 0.0
        return float(np.dot(obs, ref) / (obs_norm * ref_norm))

    def _advance_tts_position(self, n_samples: int) -> None:
        """Advance TTS position by n_samples (for explicitly dropped chunks)."""
        self._tts_position = min(self._tts_position + n_samples, self._tts_total_samples)

    def _maybe_request_or_resume_output(self, timestamp_s: float) -> tuple[bool, bool]:
        """Issue initial output request or resume queued output if allowed.

        Returns:
            (deferred_now, resumed_now)
        """
        deferred_now = False
        resumed_now = False

        if (
            not self._output_requested
            and self._tts_pcm is not None
            and timestamp_s >= self.config.output_request_s
        ):
            accepted = self._controller.request_output(self._tts_text, current_time=timestamp_s)
            self._output_requested = True
            if not accepted:
                deferred_now = True
                self._was_deferred = True

        # Repair audio has priority over the deferred original utterance. If a
        # collision repair cue is queued, hold the original text in the
        # controller until the repair clip has finished.
        if self._repair_pending or self._repair_started:
            return deferred_now, resumed_now

        if (
            self._output_requested
            and self._controller.has_deferred_output
            and self._controller.state != AgentState.LISTEN
        ):
            text = self._controller.pop_deferred_output()
            if text is not None:
                self._controller.request_output(text, current_time=timestamp_s)
                resumed_now = True
                self._was_resumed = True
                self._was_deferred = False

        return deferred_now, resumed_now

    def _get_repair_vad_chunk(self) -> np.ndarray:
        """Return one repair-cue VAD chunk, or zeros when unavailable/done."""
        if self._repair_pcm is None or self._repair_position >= self._repair_total_samples:
            return np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
        end = min(self._repair_position + VAD_FRAME_SAMPLES, self._repair_total_samples)
        chunk = self._repair_pcm[self._repair_position:end]
        self._repair_position = end
        if len(chunk) < VAD_FRAME_SAMPLES:
            padded = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded
        if self._repair_position >= self._repair_total_samples:
            self._repair_pending = False
            self._repair_started = False
            self._repair_completed = True
        return chunk

    def _gate_tts_chunk(self, can_speak: bool) -> tuple[np.ndarray, bool, str, bool, bool]:
        """Return one VAD-sized output chunk according to current gate state.

        Gated silence holds positions; it does not consume repair or original
        TTS audio. This preserves defer/repair/resume semantics.

        Returns:
            (chunk, tts_playing, output_source, repair_playing, repair_completed_now)
        """
        repair_completed_before = self._repair_completed
        if self.config.gated_playback and not can_speak:
            return np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32), False, "none", False, False

        if self._repair_pending or self._repair_started:
            if self._repair_pcm is None or self._repair_total_samples == 0:
                self._repair_pending = False
                self._repair_started = False
                self._repair_completed = True
            else:
                self._repair_started = True
                chunk = self._get_repair_vad_chunk()
                completed_now = self._repair_completed and not repair_completed_before
                return chunk, True, "repair", True, completed_now

        if self._tts_pcm is None or self._tts_position >= self._tts_total_samples:
            return np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32), False, "none", False, False
        if not self._output_requested:
            return np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32), False, "none", False, False
        if self.config.gated_playback and self._controller.state != AgentState.SPEAK:
            return np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32), False, "none", False, False
        return self._get_tts_vad_chunk(), True, "tts", False, False

    def _log_attribution_decision(
        self,
        attr_result: AttributionResult,
        timestamp_s: float,
        vad_speech: bool,
        tts_playing: bool,
        output_source: str,
        repair_playing: bool,
        repair_completed: bool,
        deferred: bool,
        resumed: bool,
        self_match_score: float,
        collision_repair_text: Optional[str] = None,
        speaker_id: Optional[str] = None,
    ) -> StreamingFrameLog:
        m4_metadata = attr_result.metadata or {}
        room_adaptation = self._last_room_adaptation_result or {}
        log = StreamingFrameLog(
            frame_index=self._frame_index,
            timestamp_s=round(timestamp_s, 6),
            vad_speech=vad_speech,
            verdict=attr_result.verdict.name,
            controller_state=self._controller.state.name,
            can_speak=self._controller.state != AgentState.LISTEN,
            should_interrupt=False,
            tts_playing=tts_playing,
            tts_position_s=round(self._tts_position / self.sr, 6),
            tts_total_duration_s=round(self._tts_total_samples / self.sr, 6) if self._tts_total_samples else 0.0,
            output_source=output_source,
            repair_position_s=round(self._repair_position / self.sr, 6),
            repair_total_duration_s=round(self._repair_total_samples / self.sr, 6) if self._repair_total_samples else 0.0,
            repair_queued=self._repair_pending,
            repair_playing=repair_playing,
            repair_completed=repair_completed,
            m4_outcome=m4_metadata.get("m4_outcome", "not_attempted"),
            m4_confidence=m4_metadata.get("m4_confidence"),
            energy_db=round(attr_result.energy_db, 2),
            self_match_score=round(self_match_score, 4),
            self_match_score_per_frame=(
                round(attr_result.self_match_score_per_frame, 4)
                if attr_result.self_match_score_per_frame is not None
                else None
            ),
            is_in_motor_window=attr_result.is_in_motor_window,
            deferred=deferred,
            resumed=resumed,
            collision_repair_text=collision_repair_text,
            speaker_id=self._current_speaker_id,
            speaker_best_match_id=self._last_speaker_best_match_id,
            speaker_best_match_similarity=(
                round(self._last_speaker_best_match_similarity, 4)
                if self._last_speaker_best_match_similarity is not None
                else None
            ),
            speaker_match_passed=self._last_speaker_match_passed,
            room_adapted=room_adaptation.get("adapted"),
            room_adaptation_reason=room_adaptation.get("reason"),
            room_adaptation_error=(
                round(float(room_adaptation["error"]), 6)
                if room_adaptation.get("error") is not None
                else None
            ),
            room_adaptation_updates=room_adaptation.get("updates"),
        )
        self._logs.append(log)
        self._frame_index += 1
        return log

    def _apply_controller_and_gate(
        self,
        attr_result: AttributionResult,
        timestamp_s: float,
        vad_speech: bool,
        self_match_score: float = 0.0,
    ) -> tuple[np.ndarray, StreamingFrameLog]:
        request_deferred, request_resumed = self._maybe_request_or_resume_output(timestamp_s)
        decision = self._controller.process_attribution(attr_result, current_time=timestamp_s)
        if decision.collision_repair_text and self._repair_pcm is not None:
            self._repair_pending = True
            self._repair_started = False
            self._repair_completed = False
            self._repair_position = 0
        resume_deferred, resume_resumed = self._maybe_request_or_resume_output(timestamp_s)
        tts_chunk, tts_playing, output_source, repair_playing, repair_completed = self._gate_tts_chunk(decision.can_speak)
        log = self._log_attribution_decision(
            attr_result=attr_result,
            timestamp_s=timestamp_s,
            vad_speech=vad_speech,
            tts_playing=tts_playing,
            output_source=output_source,
            repair_playing=repair_playing,
            repair_completed=repair_completed,
            deferred=request_deferred or resume_deferred,
            resumed=request_resumed or resume_resumed,
            self_match_score=self_match_score,
            collision_repair_text=decision.collision_repair_text,
            speaker_id=self._current_speaker_id,
        )
        log.should_interrupt = decision.should_interrupt
        log.can_speak = decision.can_speak
        log.controller_state = self._controller.state.name
        return tts_chunk, log

    def run_synthetic(self, frames: list[SyntheticAttributionFrame]) -> list[StreamingFrameLog]:
        """Run deterministic streaming validation from synthetic attribution frames."""
        self._start_time = time.time()
        self._logs = []
        self._frame_index = 0
        self._tts_position = 0
        self._repair_position = 0
        self._repair_pending = False
        self._repair_started = False
        self._repair_completed = False
        self._output_requested = False
        self._was_deferred = False
        self._was_resumed = False

        for frame in frames:
            attr = AttributionResult(
                verdict=frame.verdict,
                confidence=frame.confidence,
                self_match_score=frame.self_match_score,
                is_in_motor_window=frame.is_in_motor_window,
                causal_score=frame.causal_score,
                energy_db=frame.energy_db,
                is_speech=frame.is_speech,
                metadata={
                    "m4_outcome": frame.m4_outcome,
                    "m4_confidence": frame.m4_confidence,
                },
                self_match_score_per_frame=frame.self_match_score_per_frame,
            )

            # Update speaker threshold adapter for synthetic/replay validation.
            # Synthetic frames already carry clean verdict + speaker labels, so
            # preserving explicit WORLD/self learning here keeps offline tests
            # representative even though the live path now learns non-self
            # profiles only from post-verdict WORLD evidence.
            if self._speaker_threshold_adapter is not None:
                m4_conf = frame.m4_confidence if frame.m4_confidence is not None else 0.0
                sid = frame.speaker_id if frame.speaker_id is not None else "unknown"
                if sid == "self":
                    self._speaker_threshold_adapter.update("self", m4_conf)
                    self._current_speaker_id = None
                elif frame.verdict == SourceVerdict.WORLD:
                    self._speaker_threshold_adapter.update(sid, m4_conf)
                    self._current_speaker_id = sid
                else:
                    self._current_speaker_id = sid if sid != "self" else None

            self._apply_controller_and_gate(
                attr,
                timestamp_s=frame.time_s,
                vad_speech=frame.is_speech,
                self_match_score=frame.self_match_score,
            )

        if self.config.save_path:
            self._save_logs(self.config.save_path)

        # Feed frame logs to SelfModel (M6)
        if self.config.self_model is not None and self._logs:
            frame_dicts = [self._log_to_dict(l) for l in self._logs]
            self.config.self_model.process_frame_log(frame_dicts)

        return self._logs

    def _process_frame(self, mic_frame: np.ndarray) -> tuple[np.ndarray, StreamingFrameLog]:
        """Process one audio frame through the full pipeline.

        The mic_frame can be any length. It is split into VAD_FRAME_SAMPLES-sized
        sub-frames, each of which is processed through VAD → attribution → controller.
        The TTS output is emitted in VAD-frame-sized chunks.

        Returns (output_chunk, log_entry) where output_chunk is the same length
        as mic_frame, and log_entry is for the *last* sub-frame.
        """
        timestamp = time.time() - self._start_time

        # Current speaker ID for per-speaker adaptive threshold.
        # _update_speaker_id() maintains self._current_speaker_id based on
        # accumulated WORLD audio. Use it here so the adaptive threshold
        # from previous frames applies to the current frame.
        current_speaker_id: Optional[str] = self._current_speaker_id

        # Split mic_frame into VAD-sized sub-frames
        num_vad_frames = max(1, len(mic_frame) // VAD_FRAME_SAMPLES)
        output = np.zeros_like(mic_frame)
        last_log = None

        for vad_i in range(num_vad_frames):
            vad_start = vad_i * VAD_FRAME_SAMPLES
            vad_end = vad_start + VAD_FRAME_SAMPLES
            vad_frame = mic_frame[vad_start:vad_end]

            if len(vad_frame) < VAD_FRAME_SAMPLES:
                # Pad last frame
                padded = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
                padded[:len(vad_frame)] = vad_frame
                vad_frame = padded

            # 1. VAD
            vad_speech = self._vad.process_frame(vad_frame)

            # 2. Compute RMS
            rms = float(np.sqrt(np.mean(vad_frame ** 2)))
            rms_db = 20.0 * np.log10(max(rms, 1e-10))

            # 3. Determine if TTS is "playing"
            is_playing = (
                self._controller.state == AgentState.SPEAK
                and self._tts_position < self._tts_total_samples
            )

            # 4. Compute self-match score
            tts_position_s = self._tts_position / self.sr
            agent_start_sample = 0
            observed_onset_samples = self._tts_position
            expected_onset_samples = 0

            self_match_score = 0.0
            if self._tts_pcm is not None and is_playing:
                ref_start = self._tts_position
                ref_end = min(ref_start + VAD_FRAME_SAMPLES, self._tts_total_samples)
                if ref_end > ref_start:
                    ref_chunk = self._tts_pcm[ref_start:ref_end]
                    mic_chunk = vad_frame[:len(ref_chunk)]
                    if len(mic_chunk) == len(ref_chunk):
                        self_match_score = self._compute_normalized_xcorr(mic_chunk, ref_chunk)

            # 5. Attribution
            attr_result = self._attributor.attribute(
                match_score=self_match_score,
                observed_onset_samples=observed_onset_samples,
                expected_onset_samples=expected_onset_samples,
                rms_db=rms_db,
                is_playing=is_playing,
                is_speech=vad_speech,
                frame_audio=vad_frame,
                self_pcm=self._tts_pcm,
                agent_start_sample=agent_start_sample,
                speaker_id=current_speaker_id,
            )

            # 5b. Speaker identification for per-speaker adaptive threshold
            registered_id = self._update_speaker_id(vad_frame, attr_result)

            # 5c. Update non-self adaptive-threshold profiles only from clean
            # WORLD evidence, after verdict resolution and speaker ID update.
            self._update_adapter_from_clean_world_evidence(
                attr_result,
                self._current_speaker_id,
            )

            # 6. Controller + real-time TTS gate
            sub_timestamp = timestamp + (vad_i * VAD_FRAME_SAMPLES / self.sr)
            tts_chunk, last_log = self._apply_controller_and_gate(
                attr_result,
                timestamp_s=sub_timestamp,
                vad_speech=vad_speech,
                self_match_score=self_match_score,
            )

            # 6b. M7 trust-gated room filter adaptation. Use the actual output
            # chunk emitted for this frame as the efference copy; if output is
            # gated/silent, the adapter holds with no_efference.
            room_adaptation = self._update_room_filter_adaptation(
                known_pcm=tts_chunk,
                captured_pcm=vad_frame,
                attr_result=attr_result,
                self_match_score=self_match_score,
            )
            if room_adaptation is not None:
                last_log.room_adapted = room_adaptation.get("adapted")
                last_log.room_adaptation_reason = room_adaptation.get("reason")
                err = room_adaptation.get("error")
                last_log.room_adaptation_error = round(float(err), 6) if err is not None else None
                last_log.room_adaptation_updates = room_adaptation.get("updates")

            # Record conversational registration event on the frame log
            if registered_id is not None:
                last_log.speaker_registered = registered_id

            output[vad_start:vad_end] = tts_chunk

        return output, last_log  # type: ignore[return-value]

    def _render_output_frame(self, frame_samples: int, timestamp_s: float) -> np.ndarray:
        """Render one hardware callback worth of output without mic analysis.

        This method is intentionally realtime-safe: it only advances controller
        request/resume state and copies queued TTS/repair audio. It must not run
        VAD, M4, speaker-ID, room adaptation, or attribution.
        """
        output = np.zeros(frame_samples, dtype=np.float32)
        num_vad_frames = max(1, frame_samples // VAD_FRAME_SAMPLES)
        with self._state_lock:
            for vad_i in range(num_vad_frames):
                vad_start = vad_i * VAD_FRAME_SAMPLES
                vad_end = vad_start + VAD_FRAME_SAMPLES
                if vad_end > frame_samples:
                    break
                sub_timestamp = timestamp_s + (vad_i * VAD_FRAME_SAMPLES / self.sr)
                self._maybe_request_or_resume_output(sub_timestamp)
                can_speak = (not self.config.gated_playback) or self._controller.state != AgentState.LISTEN
                tts_chunk, _, _, _, _ = self._gate_tts_chunk(can_speak)
                output[vad_start:vad_end] = tts_chunk
        return output

    def _analyze_mic_frame(
        self,
        mic_frame: np.ndarray,
        output_frame: np.ndarray,
        timestamp_s: float,
    ) -> None:
        """Analyze mic input off the audio callback and append frame logs.

        This mirrors the attribution side of _process_frame but does not render
        or advance playback. Controller decisions affect future callback blocks.
        """
        num_vad_frames = max(1, len(mic_frame) // VAD_FRAME_SAMPLES)
        current_speaker_id: Optional[str] = self._current_speaker_id
        for vad_i in range(num_vad_frames):
            vad_start = vad_i * VAD_FRAME_SAMPLES
            vad_end = vad_start + VAD_FRAME_SAMPLES
            vad_frame = mic_frame[vad_start:vad_end]
            known_chunk = output_frame[vad_start:vad_end]
            if len(vad_frame) < VAD_FRAME_SAMPLES:
                padded = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
                padded[:len(vad_frame)] = vad_frame
                vad_frame = padded
            if len(known_chunk) < VAD_FRAME_SAMPLES:
                padded = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
                padded[:len(known_chunk)] = known_chunk
                known_chunk = padded

            vad_speech = self._vad.process_frame(vad_frame)
            rms = float(np.sqrt(np.mean(vad_frame ** 2)))
            rms_db = 20.0 * np.log10(max(rms, 1e-10))
            tts_playing = bool(np.sqrt(np.mean(known_chunk ** 2)) > 1e-8)

            self_match_score = 0.0
            if tts_playing:
                self_match_score = self._compute_normalized_xcorr(vad_frame, known_chunk)

            sub_timestamp = timestamp_s + (vad_i * VAD_FRAME_SAMPLES / self.sr)
            with self._state_lock:
                tts_position_snapshot = self._tts_position
                speaker_id_snapshot = current_speaker_id

            attr_result = self._attributor.attribute(
                match_score=self_match_score,
                observed_onset_samples=tts_position_snapshot,
                expected_onset_samples=0,
                rms_db=rms_db,
                is_playing=tts_playing,
                is_speech=vad_speech,
                frame_audio=vad_frame,
                self_pcm=self._tts_pcm,
                agent_start_sample=0,
                speaker_id=speaker_id_snapshot,
            )

            registered_id = self._update_speaker_id(vad_frame, attr_result)
            self._update_adapter_from_clean_world_evidence(
                attr_result,
                self._current_speaker_id,
            )
            room_adaptation = self._update_room_filter_adaptation(
                known_pcm=known_chunk,
                captured_pcm=vad_frame,
                attr_result=attr_result,
                self_match_score=self_match_score,
            )

            with self._state_lock:
                decision = self._controller.process_attribution(attr_result, current_time=sub_timestamp)
                if decision.collision_repair_text and self._repair_pcm is not None:
                    self._repair_pending = True
                    self._repair_started = False
                    self._repair_completed = False
                    self._repair_position = 0

                log = self._log_attribution_decision(
                    attr_result=attr_result,
                    timestamp_s=sub_timestamp,
                    vad_speech=vad_speech,
                    tts_playing=tts_playing,
                    output_source="tts" if tts_playing else "none",
                    repair_playing=False,
                    repair_completed=False,
                    deferred=False,
                    resumed=False,
                    self_match_score=self_match_score,
                    collision_repair_text=decision.collision_repair_text,
                    speaker_id=self._current_speaker_id,
                )
                log.should_interrupt = decision.should_interrupt
                log.can_speak = decision.can_speak
                log.controller_state = self._controller.state.name
                if registered_id is not None:
                    log.speaker_registered = registered_id
                if room_adaptation is not None:
                    log.room_adapted = room_adaptation.get("adapted")
                    log.room_adaptation_reason = room_adaptation.get("reason")
                    err = room_adaptation.get("error")
                    log.room_adaptation_error = round(float(err), 6) if err is not None else None
                    log.room_adaptation_updates = room_adaptation.get("updates")

    def _get_tts_vad_chunk(self) -> np.ndarray:
        """Return one VAD_FRAME_SAMPLES-sized TTS chunk, or zeros."""
        if self._tts_pcm is None or self._tts_position >= self._tts_total_samples:
            return np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
        end = min(self._tts_position + VAD_FRAME_SAMPLES, self._tts_total_samples)
        chunk = self._tts_pcm[self._tts_position:end]
        self._tts_position = end
        if len(chunk) < VAD_FRAME_SAMPLES:
            padded = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded
        return chunk

    def run(self, duration_s: float = 10.0) -> list[StreamingFrameLog]:
        """Run the streaming loop for a fixed duration.

        This is the main entry point for offline/replay mode (no hardware needed).
        For hardware mode, use run_with_hardware().
        """
        import sounddevice as sd

        self._start_time = time.time()
        self._logs = []
        self._reset_live_diagnostics()
        all_output: list[np.ndarray] = []
        all_input: list[np.ndarray] = []

        # Keep everything at 16kHz — the OS will resample to hardware rate.
        # Blocksize must be evenly divisible by VAD_FRAME_SAMPLES to avoid
        # zero-padding gaps that cause audible clicks.
        vad_frame = VAD_FRAME_SAMPLES  # 480 samples
        blocksize = int(self.config.callback_blocksize_ms * self.sr / 1000.0)
        n_vad_frames = max(1, blocksize // vad_frame)
        blocksize = n_vad_frames * vad_frame  # VAD-aligned

        analysis_queue: queue.Queue[tuple[np.ndarray, np.ndarray, float] | None] = queue.Queue(maxsize=8)
        stop_worker = threading.Event()

        def analysis_worker() -> None:
            while not stop_worker.is_set() or not analysis_queue.empty():
                try:
                    item = analysis_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    analysis_queue.task_done()
                    break
                mic_frame, output_frame, timestamp_s = item
                try:
                    self._analyze_mic_frame(mic_frame, output_frame, timestamp_s)
                except Exception as exc:  # best-effort analysis must not break playback
                    self._live_analysis_worker_errors.append(repr(exc))
                finally:
                    analysis_queue.task_done()

        worker = threading.Thread(target=analysis_worker, name="corollary-audio-analysis", daemon=True)
        worker.start()

        def callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"  Audio status: {status}")

            timestamp_s = self._claim_live_callback_timestamp(frames)
            mic_frame = indata[:, 0].copy()
            chunk = self._render_output_frame(frames, timestamp_s=timestamp_s)

            outdata[:len(chunk), 0] = chunk[:frames]
            if len(chunk) < frames:
                outdata[len(chunk):, 0] = 0

            try:
                analysis_queue.put_nowait((mic_frame.copy(), chunk[:frames].copy(), timestamp_s))
                self._live_analysis_blocks_enqueued += 1
                self._live_analysis_queue_max_depth = max(
                    self._live_analysis_queue_max_depth,
                    analysis_queue.qsize(),
                )
            except queue.Full:
                # Drop oldest analysis block rather than ever blocking the realtime callback.
                try:
                    analysis_queue.get_nowait()
                    analysis_queue.task_done()
                except queue.Empty:
                    pass
                self._live_analysis_blocks_dropped += 1
                try:
                    analysis_queue.put_nowait((mic_frame.copy(), chunk[:frames].copy(), timestamp_s))
                    self._live_analysis_blocks_enqueued += 1
                    self._live_analysis_queue_max_depth = max(
                        self._live_analysis_queue_max_depth,
                        analysis_queue.qsize(),
                    )
                except queue.Full:
                    self._live_analysis_blocks_dropped += 1

            all_output.append(chunk[:frames].copy())
            all_input.append(mic_frame.copy())

        print(f"Starting streaming runtime ({duration_s}s)...")
        print(f"  Sample rate: {self.sr} Hz")
        print(f"  Gated playback: {self.config.gated_playback}")
        print(f"  M4 enabled: {self.config.m4_enable}")
        print(f"  TTS duration: {self._tts_total_samples / self.sr:.2f}s")

        with sd.Stream(
            samplerate=self.sr,
            blocksize=blocksize,
            dtype="float32",
            channels=1,
            callback=callback,
            device=(self.config.input_device, self.config.output_device),
        ):
            time.sleep(duration_s)

        stop_worker.set()
        try:
            analysis_queue.put_nowait(None)
        except queue.Full:
            try:
                analysis_queue.get_nowait()
                analysis_queue.task_done()
            except queue.Empty:
                pass
            analysis_queue.put_nowait(None)
        analysis_queue.join()
        worker.join(timeout=2.0)

        elapsed = time.time() - self._start_time
        print(f"  Completed: {self._frames} frames in {elapsed:.2f}s")
        if self._live_analysis_blocks_dropped:
            print(f"  Dropped analysis blocks: {self._live_analysis_blocks_dropped}")
        if self._live_analysis_worker_errors:
            print(f"  Analysis worker errors: {self._live_analysis_worker_errors[:3]}")

        # Save logs
        if self.config.save_path:
            self._save_logs(self.config.save_path)

        # Feed frame logs to SelfModel (M6)
        if self.config.self_model is not None and self._logs:
            frame_dicts = [self._log_to_dict(l) for l in self._logs]
            self.config.self_model.process_frame_log(frame_dicts)

        return self._logs

    def run_offline(
        self,
        input_audio: np.ndarray,
        input_sr: int = 16000,
    ) -> list[StreamingFrameLog]:
        """Run the streaming pipeline on pre-recorded audio (no hardware).

        This is the deterministic replay mode — same pipeline as run() but
        driven by a pre-recorded input buffer instead of a live mic.

        Returns the list of frame logs for analysis.
        """
        self._start_time = time.time()
        self._logs = []
        self._frame_index = 0
        self._tts_position = 0
        self._repair_position = 0
        self._repair_pending = False
        self._repair_started = False
        self._repair_completed = False
        self._output_requested = False
        self._was_deferred = False
        self._was_resumed = False

        # Resample if needed
        if input_sr != self.sr:
            from scipy.signal import resample as scipy_resample
            input_audio = scipy_resample(
                input_audio,
                int(len(input_audio) * self.sr / input_sr)
            ).astype(np.float32)

        # Process frame-by-frame
        chunk_samples = self._tts_chunk_samples
        num_chunks = (len(input_audio) + chunk_samples - 1) // chunk_samples

        print(f"Offline streaming replay: {len(input_audio)} samples, {num_chunks} chunks")

        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, len(input_audio))
            mic_frame = input_audio[start:end]

            # Pad last frame if needed
            if len(mic_frame) < chunk_samples:
                padded = np.zeros(chunk_samples, dtype=np.float32)
                padded[:len(mic_frame)] = mic_frame
                mic_frame = padded

            chunk, _ = self._process_frame(mic_frame)
            # Logs are already appended inside _process_frame

        elapsed = time.time() - self._start_time
        print(f"  Completed: {self._frame_index} frames in {elapsed:.2f}s")

        # Save logs
        if self.config.save_path:
            self._save_logs(self.config.save_path)

        # Feed frame logs to SelfModel (M6)
        if self.config.self_model is not None and self._logs:
            frame_dicts = [self._log_to_dict(l) for l in self._logs]
            self.config.self_model.process_frame_log(frame_dicts)

        return self._logs

    def _log_to_dict(self, log: StreamingFrameLog) -> dict:
        """Convert a StreamingFrameLog to a dict for SelfModel consumption."""
        return {
            "frame_index": log.frame_index,
            "timestamp_s": log.timestamp_s,
            "vad_speech": log.vad_speech,
            "verdict": log.verdict,
            "controller_state": log.controller_state,
            "can_speak": log.can_speak,
            "should_interrupt": log.should_interrupt,
            "tts_playing": log.tts_playing,
            "tts_position_s": log.tts_position_s,
            "tts_total_duration_s": log.tts_total_duration_s,
            "output_source": log.output_source,
            "repair_position_s": log.repair_position_s,
            "repair_total_duration_s": log.repair_total_duration_s,
            "repair_queued": log.repair_queued,
            "repair_playing": log.repair_playing,
            "repair_completed": log.repair_completed,
            "m4_outcome": log.m4_outcome,
            "m4_confidence": log.m4_confidence,
            "energy_db": log.energy_db,
            "self_match_score": log.self_match_score,
            "self_match_score_per_frame": log.self_match_score_per_frame,
            "is_in_motor_window": log.is_in_motor_window,
            "deferred": log.deferred,
            "resumed": log.resumed,
            "collision_repair_text": log.collision_repair_text,
            "speaker_id": log.speaker_id,
            "speaker_best_match_id": log.speaker_best_match_id,
            "speaker_best_match_similarity": log.speaker_best_match_similarity,
            "speaker_match_passed": log.speaker_match_passed,
            "speaker_registered": log.speaker_registered,
            "room_adapted": log.room_adapted,
            "room_adaptation_reason": log.room_adaptation_reason,
            "room_adaptation_error": log.room_adaptation_error,
            "room_adaptation_updates": log.room_adaptation_updates,
        }

    def _save_logs(self, path: Path) -> None:
        """Save frame logs to JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for log in self._logs:
                f.write(json.dumps(self._log_to_dict(log)) + "\n")
        print(f"  Saved {len(self._logs)} log entries to {path}")

    @property
    def _frames(self) -> int:
        return self._frame_index

    def get_summary(self) -> dict:
        """Return a summary of the streaming run."""
        if not self._logs:
            return {"error": "no logs"}

        total = len(self._logs)
        speech_frames = sum(1 for l in self._logs if l.vad_speech)
        tts_played_frames = sum(1 for l in self._logs if l.tts_playing)
        repair_played_frames = sum(1 for l in self._logs if l.repair_playing)
        deferred_frames = sum(1 for l in self._logs if l.deferred)
        resumed_frames = sum(1 for l in self._logs if l.resumed)
        interrupt_frames = sum(1 for l in self._logs if l.should_interrupt)
        collision_repair_events = sum(1 for l in self._logs if l.collision_repair_text)

        verdict_counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        for l in self._logs:
            verdict_counts[l.verdict] = verdict_counts.get(l.verdict, 0) + 1
            state_counts[l.controller_state] = state_counts.get(l.controller_state, 0) + 1

        return {
            "total_frames": total,
            "speech_frames": speech_frames,
            "tts_played_frames": tts_played_frames,
            "repair_played_frames": repair_played_frames,
            "tts_total_duration_s": self._tts_total_samples / self.sr if self._tts_total_samples else 0,
            "repair_total_duration_s": self._repair_total_samples / self.sr if self._repair_total_samples else 0,
            "deferred_events": deferred_frames,
            "resumed_events": resumed_frames,
            "interrupt_events": interrupt_frames,
            "collision_repair_events": collision_repair_events,
            "repair_audio_completed": self._repair_completed,
            "verdict_counts": verdict_counts,
            "controller_state_counts": state_counts,
            "final_controller_state": self._logs[-1].controller_state if self._logs else None,
            "tts_completed": self._tts_position >= self._tts_total_samples,
            "gated_playback": self.config.gated_playback,
            "m4_enabled": self.config.m4_enable,
            "live_callback_count": self._live_callback_count,
            "analysis_blocks_enqueued": self._live_analysis_blocks_enqueued,
            "analysis_blocks_dropped": self._live_analysis_blocks_dropped,
            "analysis_queue_max_depth": self._live_analysis_queue_max_depth,
            "analysis_worker_errors": list(self._live_analysis_worker_errors),
        }

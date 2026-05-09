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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from thalamus.attributor import ThalamicAttributor, SourceVerdict
from cochlea.vad import SmoothedVAD, VAD_FRAME_SAMPLES
from pfc.agentive_controller import AgentiveController, AgentState


@dataclass
class StreamingConfig:
    """Configuration for the streaming runtime."""

    # TTS audio (pre-synthesized)
    tts_pcm: np.ndarray = None
    tts_text: str = ""

    # Audio devices
    input_device: int = 3
    output_device: int = 5
    sr: int = 16000

    # TTS chunking
    tts_chunk_ms: float = 200.0  # emit TTS in this-size chunks

    # Feature flags
    m4_enable: bool = False
    gated_playback: bool = True  # if False, always play TTS (no gating)

    # M4 params
    m4_self_threshold: float = 0.50
    m4_min_buffer_ms: float = 60.0

    # Room filter
    room_filter_path: Optional[str] = None
    room_local_threshold: Optional[float] = None

    # Controller params
    listen_threshold_ms: float = 100.0
    cooldown_ms: float = 200.0
    both_external_min_frames: int = 7
    both_external_energy_ratio: float = 2.0

    # Attributor params
    energy_threshold_db: float = -55.0
    similarity_threshold: float = 0.05
    causal_tolerance_ms: float = 175.0

    # Logging
    save_path: Optional[Path] = None  # JSONL log file


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
    m4_outcome: str
    m4_confidence: Optional[float]
    energy_db: float
    self_match_score: float
    self_match_score_per_frame: Optional[float]
    is_in_motor_window: bool
    deferred: bool
    resumed: bool


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

        # Pipeline components
        self._vad = SmoothedVAD()
        self._controller = AgentiveController(
            listen_threshold_ms=config.listen_threshold_ms,
            cooldown_ms=config.cooldown_ms,
            both_external_min_frames=config.both_external_min_frames,
            both_external_energy_ratio_threshold=config.both_external_energy_ratio,
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
        if config.m4_enable:
            try:
                from stg.speaker import SpeakerIdentifier, SelfVoiceDetector
                identifier = SpeakerIdentifier(use_real=False)
                m4_detector = SelfVoiceDetector(identifier, threshold=config.m4_self_threshold)
                if self._tts_pcm is not None:
                    m4_detector.register_self(self._tts_pcm)
            except Exception as e:
                print(f"  WARNING: M4 detector failed to initialize: {e}")
                m4_detector = None

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
        )

        # Logging
        self._logs: list[StreamingFrameLog] = []
        self._frame_index = 0
        self._start_time = 0.0

        # Deferred output tracking
        self._was_deferred = False
        self._was_resumed = False

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

    def _advance_tts_position(self, n_samples: int) -> None:
        """Advance TTS position by n_samples (for unplayed muted chunks)."""
        self._tts_position = min(self._tts_position + n_samples, self._tts_total_samples)

    def _process_frame(self, mic_frame: np.ndarray) -> tuple[np.ndarray, StreamingFrameLog]:
        """Process one audio frame through the full pipeline.

        The mic_frame can be any length. It is split into VAD_FRAME_SAMPLES-sized
        sub-frames, each of which is processed through VAD → attribution → controller.
        The TTS output is emitted in VAD-frame-sized chunks.

        Returns (output_chunk, log_entry) where output_chunk is the same length
        as mic_frame, and log_entry is for the *last* sub-frame.
        """
        timestamp = time.time() - self._start_time

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
                from scripts.m2_speech_preservation_harness import normalized_xcorr
                ref_start = self._tts_position
                ref_end = min(ref_start + VAD_FRAME_SAMPLES, self._tts_total_samples)
                if ref_end > ref_start:
                    ref_chunk = self._tts_pcm[ref_start:ref_end]
                    mic_chunk = vad_frame[:len(ref_chunk)]
                    if len(mic_chunk) == len(ref_chunk):
                        self_match_score, _ = normalized_xcorr(mic_chunk, ref_chunk)

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
            )

            # 6. Controller
            sub_timestamp = timestamp + (vad_i * VAD_FRAME_SAMPLES / self.sr)
            decision = self._controller.process_attribution(
                attr_result, current_time=sub_timestamp
            )

            # 7. TTS gating — emit one VAD-sized chunk
            tts_chunk = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
            tts_playing = False
            deferred = False
            resumed = False

            if self._tts_position < self._tts_total_samples:
                if self.config.gated_playback:
                    if decision.can_speak:
                        if self._was_deferred and self._controller.state == AgentState.SPEAK:
                            resumed = True
                            self._was_resumed = True
                            self._was_deferred = False
                        tts_chunk = self._get_tts_vad_chunk()
                        tts_playing = True
                    else:
                        self._advance_tts_position(VAD_FRAME_SAMPLES)
                        if not self._was_deferred:
                            deferred = True
                            self._was_deferred = True
                else:
                    tts_chunk = self._get_tts_vad_chunk()
                    tts_playing = True

            output[vad_start:vad_end] = tts_chunk

            # 8. Build log entry
            m4_metadata = attr_result.metadata or {}
            last_log = StreamingFrameLog(
                frame_index=self._frame_index,
                timestamp_s=round(sub_timestamp, 6),
                vad_speech=vad_speech,
                verdict=attr_result.verdict.name,
                controller_state=decision.state.name,
                can_speak=decision.can_speak,
                should_interrupt=decision.should_interrupt,
                tts_playing=tts_playing,
                tts_position_s=round(tts_position_s, 6),
                tts_total_duration_s=round(self._tts_total_samples / self.sr, 6) if self._tts_total_samples else 0.0,
                m4_outcome=m4_metadata.get("m4_outcome", "not_attempted"),
                m4_confidence=m4_metadata.get("m4_confidence"),
                energy_db=round(rms_db, 2),
                self_match_score=round(self_match_score, 4),
                self_match_score_per_frame=(
                    round(attr_result.self_match_score_per_frame, 4)
                    if attr_result.self_match_score_per_frame is not None
                    else None
                ),
                is_in_motor_window=attr_result.is_in_motor_window,
                deferred=deferred,
                resumed=resumed,
            )
            self._logs.append(last_log)
            self._frame_index += 1

        return output, last_log  # type: ignore[return-value]

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
        all_output: list[np.ndarray] = []
        all_input: list[np.ndarray] = []

        def callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"  Audio status: {status}")

            mic_frame = indata[:, 0].copy()
            chunk, log = self._process_frame(mic_frame)

            # Output TTS (or zeros)
            outdata[:, 0] = chunk[:frames]
            if frames < len(chunk):
                outdata[frames:, 0] = 0

            self._logs.append(log)
            all_output.append(chunk[:frames].copy())
            all_input.append(mic_frame.copy())

        print(f"Starting streaming runtime ({duration_s}s)...")
        print(f"  Gated playback: {self.config.gated_playback}")
        print(f"  M4 enabled: {self.config.m4_enable}")
        print(f"  TTS duration: {self._tts_total_samples / self.sr:.2f}s")

        blocksize = self._tts_chunk_samples
        with sd.Stream(
            samplerate=self.sr,
            blocksize=blocksize,
            dtype="float32",
            channels=1,
            callback=callback,
            device=(self.config.input_device, self.config.output_device),
        ):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                asyncio.sleep(duration_s)
            )

        elapsed = time.time() - self._start_time
        print(f"  Completed: {self._frames} frames in {elapsed:.2f}s")

        # Save logs
        if self.config.save_path:
            self._save_logs(self.config.save_path)

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

        return self._logs

    def _save_logs(self, path: Path) -> None:
        """Save frame logs to JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for log in self._logs:
                entry = {
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
                    "m4_outcome": log.m4_outcome,
                    "m4_confidence": log.m4_confidence,
                    "energy_db": log.energy_db,
                    "self_match_score": log.self_match_score,
                    "self_match_score_per_frame": log.self_match_score_per_frame,
                    "is_in_motor_window": log.is_in_motor_window,
                    "deferred": log.deferred,
                    "resumed": log.resumed,
                }
                f.write(json.dumps(entry) + "\n")
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
        deferred_frames = sum(1 for l in self._logs if l.deferred)
        resumed_frames = sum(1 for l in self._logs if l.resumed)
        interrupt_frames = sum(1 for l in self._logs if l.should_interrupt)

        verdict_counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        for l in self._logs:
            verdict_counts[l.verdict] = verdict_counts.get(l.verdict, 0) + 1
            state_counts[l.controller_state] = state_counts.get(l.controller_state, 0) + 1

        return {
            "total_frames": total,
            "speech_frames": speech_frames,
            "tts_played_frames": tts_played_frames,
            "tts_total_duration_s": self._tts_total_samples / self.sr if self._tts_total_samples else 0,
            "deferred_events": deferred_frames,
            "resumed_events": resumed_frames,
            "interrupt_events": interrupt_frames,
            "verdict_counts": verdict_counts,
            "controller_state_counts": state_counts,
            "final_controller_state": self._logs[-1].controller_state if self._logs else None,
            "tts_completed": self._tts_position >= self._tts_total_samples,
            "gated_playback": self.config.gated_playback,
            "m4_enabled": self.config.m4_enable,
        }

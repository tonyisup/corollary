# M3 Streaming Runtime — Config Schema

> **Source of truth:** `packages/pfc/streaming_runtime.py` `StreamingConfig` dataclass.
> This document is derived from that source. If they diverge, the source wins.

## `StreamingConfig` Reference

### TTS audio

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tts_pcm` | `np.ndarray` | `None` | Pre-synthesized agent TTS audio (float32, mono, 16kHz). Always pre-render before opening the callback. |
| `tts_text` | `str` | `""` | Text label for the TTS utterance. Used in logs and controller deferred-output tracking. |
| `repair_pcm` | `Optional[np.ndarray]` | `None` | Pre-synthesized repair cue audio. Queued separately from the main utterance; plays once after collision without consuming the deferred original. |

### Audio devices

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_device` | `int` | `3` | Mic input device index. Current hardware: 3 = fifine Microphone. |
| `output_device` | `int` | `5` | Agent TTS output device index. Current hardware: 5 = Mac mini Speakers. |
| `sr` | `int` | `16000` | Sample rate in Hz. All audio must match. |

### TTS chunking

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tts_chunk_ms` | `float` | `200.0` | Callback buffer size in ms. Internally split into VAD-frame-sized sub-frames (30ms at 16kHz = 480 samples). |
| `output_request_s` | `float` | `0.0` | Timestamp (s) when the agent first intends to speak. Frames before this are WORLD/SILENCE; TTS is requested at this time. |

### Feature flags

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `m4_enable` | `bool` | `False` | Enable M4 speaker-embedding self-voice detector. Builds `SelfVoiceDetector` from agent PCM. |
| `gated_playback` | `bool` | `True` | Enable real-time TTS gating. When `False`, TTS always plays (no suppression). When `True`, TTS is muted while controller state is LISTEN. |

### M4 params

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `m4_self_threshold` | `float` | `0.50` | Confidence threshold for M4 self-voice classification. Higher = more conservative. |
| `m4_min_buffer_ms` | `float` | `60.0` | Minimum audio buffer (ms) before M4 can trigger. M4 needs >=700ms stimulus duration and a buffer <=300ms for reliable triggering. |

### Room filter

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `room_filter_path` | `Optional[str]` | `None` | Path to pre-fitted room filter `.npy` (64-tap FIR). If set, enables room-local scoring in the attributor. |
| `room_local_threshold` | `Optional[float]` | `None` | Threshold for room-local self-match override. If `None` and `room_filter_path` is set, defaults to `0.04`. Production auto-calibrated value: `0.052`. |
| `enable_room_filter_adaptation` | `bool` | `False` | M7 phase-1 trust-gated NLMS. When enabled, runtime calls `room_adapter.update_trust_gated(...)` using emitted TTS as efference copy and mic audio as capture. |
| `room_adapter` | `Optional[Any]` | `None` | Adapter object, normally `BabblingCerebellum`, exposing `update_trust_gated(...)` and `get_learned_ir()`. Disabled unless `enable_room_filter_adaptation=True`. |
| `room_adapt_confidence_threshold` | `float` | `0.80` | Minimum M4 confidence for room-filter adaptation. SELF verdict alone is insufficient. |
| `room_adapt_match_threshold` | `float` | `0.20` | Minimum room-local/self-match score for room-filter adaptation. BOTH/WORLD/SILENCE always hold. |

### Controller params

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `listen_threshold_ms` | `float` | `100.0` | Minimum sustained external speech (ms) before WORLD→LISTEN transition. Hysteresis to avoid flapping. |
| `cooldown_ms` | `float` | `200.0` | Silence duration (ms) after speech ends before controller resets speech timing. |
| `both_external_min_frames` | `int` | `7` | Consecutive BOTH frames with elevated energy ratio required to classify overlap as external speech. Prevents transient ratio spikes from triggering interrupt. |
| `both_external_energy_ratio` | `float` | `2.0` | Energy ratio threshold (observed / expected self RMS) for classifying BOTH frames as external speech. |
| `collision_repair_text` | `Optional[str]` | `"Sorry, go ahead."` | Social repair cue emitted once per real SPEAK→LISTEN interruption. Set to `None` to disable. |

### Attributor params

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `energy_threshold_db` | `float` | `-55.0` | Energy threshold (dB) below which frames are classified as SILENCE. |
| `similarity_threshold` | `float` | `0.05` | Normalized cross-correlation threshold for SELF attribution. Production auto-calibrated default. |
| `causal_tolerance_ms` | `float` | `175.0` | Time window (ms) for causal matching between expected and observed self-onset. Calibrated to room acoustics and hardware latency. |

### Logging

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_path` | `Optional[Path]` | `None` | Path for structured JSONL frame logs. Each line is one frame with VAD, verdict, controller state, TTS position, repair state, M4 metadata, energy, and self-match scores. |

---

## Tuned Production Values

These values were calibrated on the Mac mini + fifine Microphone path and validated across multiple M3 runs:

| Parameter | Calibrated | Notes |
|-----------|------------|-------|
| `tts_chunk_ms` | 200.0 | 200ms blocks; internally split into 30ms VAD frames |
| `similarity_threshold` | 0.05 | From auto-calibration; room-local scoring reduces BOTH false negatives |
| `causal_tolerance_ms` | 175.0 | Covers hardware + room onset latency |
| `cooldown_ms` | 200.0 | Long enough to bridge pauses in natural speech |
| `both_external_min_frames` | 7 | ~210ms at 30ms/frame; prevents transient spikes |
| `both_external_energy_ratio` | 2.0 | Observed/expected RMS ratio; 2.0 = 6dB above expected self |
| `m4_min_buffer_ms` | 60.0 | Smallest buffer with reliable M4 triggering on >=700ms stimuli |
| `m4_self_threshold` | 0.50 | Conservative; real ECAPA-TDNN is tighter than mock |
| `room_local_threshold` | 0.04 | Companion to production filter; auto-cal gives ~0.052 |

---

## Validation Ladder

| Gate | Command | Expects |
|------|---------|---------|
| Deterministic acceptance | `scripts/m3_streaming_acceptance.py` | exit 0, all 5 synthetic/smoke checks pass |
| Gated playback acceptance | `scripts/m3_gated_playback_acceptance.py` | exit 0, `all_suppression_passed=true` |
| Hardware repair validation | `scripts/m3_streaming_repair_hardware_acceptance.py` | exit 0, all 9 repair criteria pass |
| Fast regression | `pytest tests/unit/test_streaming_runtime.py -q` | 18 passed |

---

## Design Invariants

1. **No synthesis in callback.** `tts_pcm` and `repair_pcm` must be fully rendered before `StreamingRuntime.run()` opens the audio stream.
2. **Repair queue is independent.** Repair PCM is a separate output source. It never advances `tts_pcm` position.
3. **Gated silence holds position.** When `can_speak=False`, TTS and repair positions are not consumed. The runtime mutes output without advancing.
4. **Synthetic frames are the behavior gate.** Offline replay validates plumbing. Synthetic injection validates controller/TTS-gating logic.
5. **Gated metrics differ from reflex metrics.** In `--gated-playback` mode, use analyzer fields (`segment_overlap_passed`, `defer_passed`, `all_suppression_passed`), not legacy MI/TI/resume counters.

# Corollary Data Dictionary

> Single source of truth for all shared data types, constants, and pipeline conventions.
> Every symbol below should have a one-to-one mapping into the codebase.

---

## 1. Audio Constants

| Symbol | Value | Meaning |
|---|---|---|
| `SR` | **16 000** | Target sample rate for all Corollary audio (Hz). The system resamples everything to 16 kHz internally. |
| `HOP` | **160** | Hop length for Mel spectrogram frames, in samples (10 ms at 16 kHz). |
| `WIN` | **400** | STFT window length, in samples (25 ms at 16 kHz). |
| `N_MELS` | **80** | Number of Mel frequency bands in the spectrogram. |
| `N_FFT` | **512** | FFT size used by `librosa.feature.melspectrogram`. |

**Derived constants:**

- **Frame interval:** `HOP / SR = 10 ms` — each row in a Mel spectrogram covers 10 ms of audio.
- **1 second of audio** = `SR` samples = **100 Mel frames**.

### Audio Data Conventions

| Term | Format | DType | Shape | Notes |
|---|---|---|---|---|
| PCM audio | Raw waveform | `np.float32` | `(N,)` 1-D | Normalized to `[-1.0, +1.0]`. Mono. |
| Mel spectrogram | Log-power mel | `np.float32` | `(T, N_MELS)` | In dB, produced by `librosa.power_to_db`. `T ≈ N * SR / HOP`. |

---

## 2. Room Acoustics Constants

| Symbol | Default | Meaning |
|---|---|---|
| `DEFAULT_RT60` | **0.3 s** | Reverberation time (T60) for a small room. Time for 60 dB decay after impulse. |
| `DEFAULT_FS` | **16 000** | Sample rate for room IR generation. Should match `SR`. |
| `DEFAULT_NOISE_DB` | **-60 dB** | Noise floor added after room convolution, relative to audio peak. |

**RT60 reference values:**

| RT60 | Description | Example |
|---|---|---|
| 0.1 s | Dead / anechoic | Recording booth |
| 0.3 s | Small dry room | Default in Corollary |
| 0.6 s | Normal room | Living room, moderate |
| 1.0 s+ | Live space | Hall, gym |

---

## 3. Thalamus / Gating Constants

| Symbol | Value | Meaning |
|---|---|---|
| `DEFAULT_SVSR_THRESHOLD` | **20.0 dB** | Self-Voice Suppression Ratio threshold. Below this = gate classifies as "world voice" (residual is significant). Target from M0 spec. |
| `SUBTRACTION_FLOOR` | **0.01** | Minimum fraction of observed energy to keep during spectral subtraction. Prevents over-subtraction from negative results. |

---

## 4. STG (ASR) Constants

| Symbol | Default | Meaning |
|---|---|---|
| `DEFAULT_MODEL_SIZE` | `"base"` | faster-whisper model size. Override via `WHISPER_MODEL` env var. Options: `tiny`, `base`, `small`, `medium`, `large`. |
| `DEFAULT_DEVICE` | `"cpu"` | Compute device. Apple Silicon uses `"cpu"` (CTranslate2). |
| `DEFAULT_COMPUTE_TYPE` | `"int8"` | CTranslate2 compute type. Override via `WHISPER_COMPUTE` env var. |

**faster-whisper Segment fields (v1.2.1):**
`id`, `seek`, `start`, `end`, `text`, `tokens`, `avg_logprob`, `compression_ratio`, `no_speech_prob`, `words`, `temperature`

> **⚠️ There is NO `confidence` field.** Confidence must be derived: `confidence = max(0.0, 1.0 - no_speech_prob)`. (Memory: confirmed in production testing, MEMORY.md.)

---

## 5. Data Structures (Pydantic Models)

### 5.1 Bus Events (`packages/bus/events.py`)

#### `AudioFrame`
```
pcm: np.ndarray          # float32 PCM, shape (N,)
timestamp: float         # wall-clock time (perf_counter)
sample_rate: int = 16000
channels: int = 1
```
Raw audio frame entering the cochlea.

#### `EffCopyFrame`
```
pcm: np.ndarray                          # float32 PCM, shape (N,)
timestamp: float                         # production time (perf_counter)
source: str = "broca"                    # origin: "broca" | "cerebellum" ..."
```

#### `MelBlock`
```
mel: np.ndarray                          # float32, shape (T, N_MELS)
timestamp: float                         # production time
source: str = "cochlea"                  # "cochlea" | "cerebellum"
```

#### `ASRToken`
```
token: str                               # transcribed word/token
start: float                             # start time (seconds from recording start)
end: float                               # end time
confidence: float                        # derived from no_speech_prob
is_world: bool = True                    # True if survived thalamic gate
```

#### `Utterance`
```
text: str                                # full transcribed text
start: float                             # utterance-level start time
end: float                               # utterance-level end time
tokens: List[ASRToken]                   # constituent tokens
language: str                            # detected language code (e.g. "en")
```

#### `DiarizationInfo`
> *Placeholder for future speaker separation.*

#### `MemoryEvent`
```
event_type: str                          # "utterance" | "self_voice" | ...
timestamp: float
payload: dict
```
What gets persisted to long-term memory when the Hippocampus commits.

#### `MemoryPayload`
```
utterance_text: str
self_voice_suppressed: bool
svsr_db: float
timestamp: float
```

#### `LoopIndex`
```
iteration: int                           # 0-based loop counter start
self_voice_count: int = 0
world_voice_count: int = 0
last_self_voice_ms: int = 0
last_world_voice_ms: int = 0
last_asr_text: str = ""
```
Tracks cumulative loop statistics over time.

### 5.2 SAG Models (`packages/sag/models.py`)

#### `Node`
```
id: str                                  # unique identifier (word/concept)
core: str                                # core meaning / definition
grounding: list[str]                     # lexical groundings (synonyms, surface forms)
category: str = "concept"                # semantic category
confidence: float = 1.0
last_updated: float = 0.0
```

#### `EdgeType` (enum)
| Value | Meaning |
|---|---|
| `HYPONYM` | is-a relationship (dog → animal) |
| `HYPERNYM` | category relationship (animal ← dog) |
| `MERONYM` | part-of relationship (wheel → car) |
| `HOLONYM` | whole-of relationship (car ← wheel) |
| `ASSOCIATIVE` | loose semantic link |
| `DISAMBIGUATION` | resolves ambiguity between concepts |
| `GROUNDING` | links node to surface form |
| `TEMPORAL` | time-based relationship |
| `CAUSAL` | cause-effect relationship |
| `CONTRADICTION` | incompatible concepts |

#### `EdgeAssertion`
```
source_id: str
target_id: str
edge_type: EdgeType
confidence: float = 0.5
evidence: str = ""
```

#### `Edge`
```
assertion: EdgeAssertion
created_at: float
updated_at: float
weight: float = 1.0
```

#### `GraphChange`
> Tracks mutations to the SAG for audit and replay.

#### `ChangeKind` (enum)
`NODE_ADD`, `NODE_UPDATE`, `NODE_REMOVE`, `EDGE_ADD`, `EDGE_UPDATE`, `EDGE_REMOVE`

### 5.3 STG Data (`packages/stg/asr.py`)

#### `Segment`
```
id: int                                  # segment index
text: str                                # transcribed text
start: float                             # start time (seconds)
end: float                               # end time
confidence: float                        # derived: 1.0 - no_speech_prob
words: Optional[List[Word]]              # per-word timestamps
```

#### `Word`
```
word: str
start: float
end: float
probability: float                       # model's word-level probability
```

#### `TranscriptionResult`
```
segments: List[Segment]
language: str
```
Convenience: `.text` property concatenates all segment text.

### 5.4 Reasoning (`packages/reasoning/`)

#### `ReasoningResult`
```
text: str                                # generated reasoning output
confidence: float                        # model's confidence
latency_ms: float                        # time to generate
provider: str                            # which model produced this
```

#### `ReasoningProvider` (ABC)
Interface that all reasoning providers must implement.
- `.generate(prompt: str) -> ReasoningResult`

### 5.5 PFC (`packages/pfc/simulated_loop.py`)

#### `LoopResult`
```
original_text: str                       # the utterance that was synthesized
asr_result: Optional[TranscriptionResult]
svsr_db: float                           # self-voice suppression ratio
is_world: bool                           # True if gate classified as world voice
gated_text: str                          # what survived the gate
sag_nodes: List[str]                     # SAG node IDs resolved
roundtrip_seconds: float                 # wall-clock total
steps: Dict[str, Any]                    # intermediate diagnostics
```
- `.wer` property: word error rate (symmetric difference of word sets).

### 5.6 Cochlea (`packages/cochlea/capture.py`)

#### `EfferenceBuffer`
Rolling ring-buffer for efference-copy PCM.
- `max_ms`: max duration to retain (default **3000 ms**)
- `append(pcm: np.ndarray)` — appends samples, wraps around when full
- `.audio` — returns all `_filled` samples in correct temporal order
- `.is_full` — whether buffer has wrapped at least once

### 5.7 Cerebellum — Adaptive Model (`packages/cerebellum/adaptive_model.py`)

#### `BabblingCerebellum`
Forward model that learns the room impulse response through babbling (NLMS adaptive filter).

```python
# Creation
cerebellum = BabblingCerebellum(
    ir_length_ms=300,        # length of room IR to learn
    sample_rate=16000,       # audio sample rate
    nlms_mu=0.05,            # NLMS step size (0 < mu < 2)
    nlms_epsilon=1e-6,       # numerical stability
)

# Learning: one babbling pair
error = cerebellum.update(known_pcm, captured_pcm)

# Prediction with learned room model
predicted_mel = cerebellum.predict_mel(efference_pcm)
predicted_pcm = cerebellum.predict_pcm(efference_pcm)

# Diagnostics
conv = cerebellum.get_convergence()  # {updates, current_error, trend, ...}
is_ready = cerebellum.is_ready(min_updates=3)
room_ir = cerebellum.get_learned_ir()  # copy of learned IR
```

Key parameters:
| Param | Default | Meaning |
|---|---|---|
| `ir_length_ms` | 300 | Max room IR length to learn (ms). 300ms = 4800 samples at 16kHz. |
| `nlms_mu` | 0.05 | Normalized LMS step size. Larger = faster learning but less stable. |
| `nlms_epsilon` | 1e-6 | NLMS numerical stability constant. |

Convergence states:
| State | Meaning |
|---|---|
| `"improving"` | Latest error < previous error |
| `"stable"` | Error not decreasing |

---

## 6. Pipeline Flow & Data Contracts

```
  ┌─────────┐   PCM    ┌──────────┐  PCM   ┌───────────┐  Mel   ┌──────────┐
  │ Broca   │ ────────► │ Cochlea  │ ──────► │ Cerebellum│ ─────►│ Thalamus │
  │ (TTS)   │           │ (DSP)    │          │(fwd model)│      │ (gate)   │
  └─────────┘           └────┬─────┘          └───────────┘      └────┬─────┘
                             │                                        │
                             │ PCM (room audio)                       │ residual mel
                             ▼                                        ▼
                        ┌─────────┐                              ┌─────────┐
                        │  STG    │                              │  SAG    │
                        │  (ASR)  │ ──── tokens ──────────────►  │(resolve)│
                        └─────────┘                              └─────────┘
                              │                                        │
                              ▼                                        ▼
                        ┌─────────┐                              ┌─────────────┐
                        │ PFC     │ ◄──────── feedback ───────── │ hippocampus │
                        │(coord)  │                              │ (memory)   │
                        └─────────┘                              └─────────────┘
```

### Data types at each boundary:

| Stage Out → Stage In | Data | Shape | Units |
|---|---|---|---|
| Broca → Room/Cochlea | PCM | `(N,)` float32 | [-1.0, +1.0] |
| Room → Cochlea | PCM | `(N+M-1,)` float32 | convolved audio |
| Cochlea → Cerebellum | Mel | `(T, 80)` float32 | dB |
| Cerebellum → Thalamus | Predicted Mel | `(T, 80)` float32 | dB |
| Thalamus → STG/SAG | Residual Mel + Mask | `(T, 80)` + `(T, 80)` bool | dB |
| STG → PFC | TranscriptionResult | segments | text + timestamps |
| PFC → Reasoning | Prompt | text | natural language |
| Hippocampus → PFC | MemoryPayload | structured | context |

---

## 7. Glossary

| Term | Abbreviation | Meaning |
|---|---|---|
| PCM | Pulse-Code Modulation | Raw digital audio samples |
| Mel Spectrogram | — | Frequency representation mimicking human ear response |
| RT60 | — | Time for sound to decay 60 dB in a room |
| IR | Impulse Response | Acoustic fingerprint of a room |
| SVSR | Self-Voice Suppression Ratio | How well efference copy cancels self-voice (dB) |
| AEC | Acoustic Echo Cancellation | Removing speaker output from mic input |
| IR Convolution | — | Simulating room acoustics by convolving audio with IR |
| Efference Copy | — | Internal copy of motor command sent to sensory areas |
| Forward Model | — | Predicts sensory consequences of motor commands |
| Spectral Subtraction | — | Removing predicted spectrum from observed spectrum |
| WER | Word Error Rate | (S + D + I) / N — substitution + deletion + insertion errors |

---

## 8. Environment Variables

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `"base"` | faster-whisper model size |
| `WHISPER_COMPUTE` | `"int8"` | CTranslate2 compute precision |

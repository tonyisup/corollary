# Design Overview — Corollary M0 (Simulated Loop)

> **Status:** Design (M0)  
> **Created:** 2026-04-03  
> **Scope:** Simulated self-listening loop — no microphone, no speakers

## M0 Objective

Build a **simulated closed loop** where Corollary speaks, the audio passes through a simulated room, and the ASR transcribes the result. The cerebellum's efference copy is used to distinguish self-voice from world-voice *before* it reaches the ASR.

**No real audio in M0.** Everything runs through a synthetic loop:
```
TTS text → [simulated room IR] → [corrupted audio] → ASR → transcription
```

## Architecture at M0

```
┌─────────────────────────────────────────────┐
│  Broca (TTS)                                │
│  input:  "Hello, world"                     │
│  output: audio samples (numpy array)        │
│  efference: predicted phoneme/time sequence  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Cerebellum (Forward Model)                 │
│  input:  efference copy from Broca          │
│  output: predicted STG output               │
│  purpose: "I said this, so expect to hear it"│
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Simulated Room                             │
│  input:  Broca audio                        │
│  output: audio with delay, reverb, noise    │
│  purpose: test AEC under realistic conditions│
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STG (ASR)                                  │
│  input:  simulated room output              │
│  output: transcription with timestamps      │
│  efference-aware: compares vs prediction    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  SAG (Semantic Abstraction Graph)           │
│  input:  ASR transcription                  │
│  output: resolved semantic nodes            │
│  purpose: ground words in stable concepts   │
└─────────────────────────────────────────────┘
```

## Key Components

### Cochlea (Audio Preprocessor)
- **Role:** Resample, filterbank, feature extraction
- **M0 scope:** Stub — accepts numpy audio arrays directly

### Cerebellum (Forward Model)
- **Role:** Predicts what the STG should hear when Broca speaks
- **M0 scope:** 
  - Takes TTS text → generates expected phoneme sequence
  - Compares predicted vs actual ASR output
  - Returns residual: `1.0` = pure self, `0.0` = pure world

### Thalamus (Gating/Masking)
- **Role:** Decides what reaches conscious processing
- **M0 scope:** Pass-through. If cerebellum residual > threshold, forward to PFC.

### STG (ASR)
- **Role:** Transcription + diarization
- **M0 scope:** 
  - Use `faster-whisper-large-v3-turbo` 
  - Simulated room adds noise/reverb to test robustness
  - Timestamps aligned with efference copy for comparison

### Broca (TTS)
- **Role:** Text-to-speech generation
- **M0 scope:** Any TTS that produces numpy audio (pyttsx3, edge-tts, or ElevenLabs)

### Salience Network
- **Role:** Detects novelty, decides memory writes
- **M0 scope:** Stub. Always returns salience=0.5 (no concept of "new" yet)

### Hippocampus (Episodic Memory)
- **Role:** Stores conversation turns for later retrieval
- **M0 scope:** In-memory list of `(timestamp, speaker, transcript, sag_nodes)`

### PFC (Controller)
- **Role:** Orchestrates the loop, maintains working memory
- **M0 scope:** The main loop driver. Uses SAG for entity tracking.

### SAG (Semantic Abstraction Graph)
- **Role:** Grounded knowledge representation
- **M0 scope:** 
  - Nodes created from ASR output
  - Resolve ASR tokens to SAG nodes for disambiguation
  - Track entity co-occurrence via edge assertions

## Efference Copy Flow

The core mechanism that makes Corollary "self-listening":

1. **Broca** generates speech command ("say hello")
2. **Cerebellum** receives a copy of the motor plan
3. **Cerebellum** predicts what the STG will hear
4. **STG** receives actual audio (self + world)
5. **Cerebellum** compares prediction vs reality
6. If prediction matches → this is **self** (attenuate or tag)
7. If prediction differs → this contains **world** (forward to PFC)

In M0, the "world" is simulated silence (only self-voice exists), so the cerebellum should achieve near-zero residual.

## M0 Success Criteria

| Criterion | Target |
|-----------|--------|
| TTS → simulated room → ASR round-trip | Transcription matches original with ≥90% WER |
| Efference copy detects self-voice | Residual < 0.2 for pure self-voice input |
| SAG resolves entities | ≥80% of unique words resolved to SAG nodes |
| End-to-end latency | <200 ms for simulated (non-realtime) processing |

## Next Steps (M1)

The next hardware stage should be treated as developmental and pre-linguistic.

### M1a: Self-sound attribution
- Real microphone input
- Simple emitted sounds first: tones, clicks, vowels, babble, nonsense syllables
- Efference-copy alignment against observed acoustic events
- Primary question: "Did I just make that sound?"

### M1b: Babbling-driven room learning
- Repeated self-generated sounds to adapt the forward model to room acoustics
- Measure convergence and improved attribution after adaptation

### M1c: Mixed-source attribution
- Add external/world sounds
- Distinguish self-only, world-only, and overlapping segments

### M2: Speech-aware self/world separation
- Detect speech vs non-speech under overlap
- Preserve external speech while modeling self-generated sound

### M3: Agentive listening and proto-turn-taking
- Distinguish vocal agents from non-agentive environment noise
- Detect when to inhibit self-output and listen
- Introduce contingent timing and proto-turn-taking before full conversation

Only after these stages should Corollary push hard on explicit self-voice classification, multi-speaker memory, SAG-informed lexical processing under overlap, and full conversational persistence.

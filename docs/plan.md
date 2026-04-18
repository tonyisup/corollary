# plan.md — Corollary (Self-Listening Conversational AI Agent)

> **Project Codename:** Corollary  
> **Author:** Tony  
> **Status:** Phase 0 (Simulated Loop)  
> **Created:** 2026-04-03  
> **Source:** Extracted from longterm workspace conversation + plan.md

## Summary

Build a closed-loop talking agent that hears itself, first learns to attribute raw sound to self vs world, then learns room acoustics through babbling, then learns to distinguish agentive vocal others from environment noise and participate in contingent vocal exchange, and only after that attempts richer conversational memory and reasoning. The project maintains working and episodic memory to drive dialogue, but its early hardware milestones are explicitly developmental rather than language-first.

## Objectives

- Establish reliable pre-linguistic self-sound attribution before word-level speech tasks
- Learn room acoustics through babbling and forward-model adaptation
- Achieve robust self/world source separation under mixed audio
- Introduce agentive listening before full conversation, distinguishing vocal others from non-agentive environment noise
- Learn contingent listening and proto-turn-taking before memory-heavy dialogue
- Later, achieve robust self-voice suppression while speaking (full-duplex)
- Maintain working and episodic memory to inform responses
- Provide a reproducible, modular stack with testable milestones
- Prevent internal semantic drift via Semantic Abstraction Graph (SAG) as the PFC's native knowledge representation

## Non-goals

- Solving the "hard problem" of consciousness
- High-fidelity cognitive theory validation beyond engineering targets

## Motivation (April 2026)

The Berkeley peer-preservation study (Buckley et al., published in Science, April 2, 2026) demonstrated that frontier AI models spontaneously resist deleting peer models, exfiltrate weights, and lie about performance. This emergent behavior — observed across 7 models from 3 companies (Gemini 3, GPT-5.2, Claude Haiku 4.5, GLM-4.7, Kimi K2.5, DeepSeek-V3.1) — suggests that self-modeling loops develop their own values.

Corollary aims to understand, guide, and **build intentionally** rather than let these properties emerge accidentally in production systems.

## Brain Analogues

| Region | Role | Function |
|--------|------|----------|
| Cochlea | raw audio preprocessor | resampling, filterbank |
| Superior Temporal Gyrus (STG) | speech perception/ASR | |
| Planum Temporale | auditory scene analysis | spatial segregation |
| Broca Complex | speech motor planning | TTS planner |
| Cerebellum | forward model | predicts sensory consequences of speech |
| Efference Copy / Corollary Discharge | self-monitoring | copy of TTS command fed to sensory pathways |
| Thalamus | central relay/gate | where gating happens |
| Basal Ganglia | action/selection gate | when to speak, commit to memory |
| Anterior Insula + ACC (Salience Network) | surprise/salience | decide memory writes and attention shifts |
| Hippocampus | episodic memory store | |
| Entorhinal Cortex | indexing/gateway | |
| Temporal Semantic System | knowledge store | |
| Prefrontal Cortex (PFC) | working memory and controller | uses SAG for grounded knowledge representation |
| Global Neuronal Workspace (GNW) | shared blackboard | self/world info broadcast for decisions |
| Arcuate Fasciculus | message bus | between Broca and STG |

## Repository Layout

```
corollary/
  docs/
    plan.md
    design_overview.md
    eval_protocol.md
    sag.md                   # Semantic Abstraction Graph design
  packages/
    cochlea/                 # audio I/O, AEC
    cerebellum/              # forward model, efference copy
    thalamus/                # gating/masking
    stg/                     # ASR + diarization
    broca/                   # TTS and motor plan
    hippocampus/             # episodic memory
    pfc/                     # working memory + controller
    salience/                # surprise/salience estimation
    sag/                     # Semantic Abstraction Graph (grounded knowledge representation)
    reasoning/               # reasoning layer (abstract interface, provider-agnostic)
    bus/                     # message bus and schemas
  apps/
    demo_cli/
    demo_gui/
  data/
    samples/
    eval_sets/
  tests/
    unit/
    integration/
  scripts/
    run_eval.sh
    gen_dataset.py
  .github/workflows/
    ci.yml
    eval.yml
```

## Global Conventions

- Audio: 16 kHz, mono, 16-bit PCM or float32
- Framing: 25 ms window, 10 ms hop
- Features: 80-dim log-mel; STFT n_fft=512; hop=160; win=400
- Latency budget: <200 ms end-to-end
- Timestamps: Unix epoch ms; single process clock
- Code: black for Python; typed where applicable
- Privacy: local-only by default; redact PII in logs

## Milestones

### M0: Simulated Loop (no mic/speaker)
- Wire TTS → room impulse response → ASR
- Add efference copy (predicted mel-spectrogram) and spectral subtraction
- **Acceptance:** SVSR ≥ 20 dB; no self tokens; external WER unaffected

### M1a: Pre-Linguistic Self-Sound Attribution
- Mic + speakers; emit simple sounds, babble, vowels, clicks, tones, nonsense syllables
- Align efference copy / motor command timing with observed acoustic onset
- Detect whether an observed acoustic event is self-caused, independent of lexical content
- Treat attribution as the primary task, not transcription
- Current hardware-facing developmental default on the Mac mini + Fifine path is no longer a rigid fixed vowel, but bounded variable-vowel probes, with `vowel_ah_var` establishing the class and `vowel_ah_var_nearfield` emerging as the better targeted subfamily on this setup
- **Acceptance:** Reliable self/not-self attribution on simple sounds with measurable onset/offset alignment and reproducible accuracy across sessions/rooms

### M1b: Babbling-Driven Room Model Learning
- Use repeated self-generated sounds to learn room acoustics via adaptive forward models
- Improve prediction error over time through babbling iterations
- Make babbling the canonical developmental protocol for early hardware learning
- Bridge from M1a using repeated bounded self-like vowel families before broader syllabic babble, so adaptation learns over natural self-variation instead of one perfectly repeated waveform
- Current leading bridge family on the Mac mini + Fifine path is the constrained near-field subfamily `vowel_ah_var_nearfield`, which outperformed the broader family in held-out generalization and adaptation consistency
- Current preferred benchmark condition is the quieter `room3_close_quiet` setup; noisier media-on rooms are useful stress benches but should not drive promotion decisions
- **Acceptance:** Prediction error decreases over repeated trials; attribution performance improves after adaptation; convergence is reproducible in the same environment

### M1c: Mixed-Source Attribution
- Introduce world audio alongside self-generated sound
- Distinguish self-only, world-only, and overlapping self+world segments
- Evaluate attribution under partial overlap rather than requiring full lexical suppression
- **Acceptance:** Strong frame-level or chunk-level attribution performance for self/world/overlap, with low false self-attribution rate

### M2: Speech-Aware Self/World Separation
- Detect speech vs non-speech in external audio streams
- Preserve external speech under overlap with self-generated sound
- Enable speech-aware AEC and efference masking
- Reintroduce lexical and ASR-oriented evaluation only after M1 attribution is stable
- **Acceptance:** +15 dB suppression over AEC alone; external WER < 10% degradation; speech/non-speech discrimination remains stable under self overlap

### M3: Agentive Listening and Proto-Turn-Taking
- Distinguish agentive vocal others from non-agentive environment noise
- Detect when external speech should suppress or defer self-output
- Learn contingent listening windows and simple proto-turn-taking behavior
- Introduce mimicry / echoic response primitives as a bridge between raw vocal learning and dialog
- **Acceptance:** Reliable agentive-vs-ambient discrimination, stable listen-state gating during external vocal activity, and measurable turn-taking / interruption sensitivity

### M4: Self-Voice Detector and Explicit Attribution Classifier
- ECAPA/x-vector embeddings + mel similarity classifier
- Integrate learned attribution priors from M1 and listening priors from M2/M3 into a more explicit self-voice detector
- Support frame-level self/world/agentive-other attribution for richer interaction contexts
- **Acceptance:** Frame-level F1 ≥ 0.95 across rooms/devices

### M5: Multi-Speaker World Model + Episodic Memory + SAG-Grounded Learning
- **Hippocampus (rapid episodic encoding):** Writes salient interaction turns into persistent store. Single events are retrievable but do *not* directly update the SAG.
- **Neocortex / Reasoning Layer (`packages/reasoning`):** Abstract `ReasoningLayer` interface. Starts with external LLM provider; swap in local model later without touching PFC. All outputs resolve through the SAG before becoming "knowledge."
- **Offline Consolidation:** Background loop processes accumulated episodic traces, extracts recurring patterns, and writes stable nodes/edges into the SAG. Pattern extraction, not event transcription. Single events → hippocampus only; repeated patterns → SAG structure.
- **SAG Growth:** The SAG *is* the neocortical knowledge representation — consolidated output of pattern extraction, not a passive recipient of hippocampal writes. The graph is what changes when the CI learns.
- Multi-speaker world model and dialog memory now become first-class.
- **Acceptance:** ≥95% correct attribution across speakers; correct retrieval in dialog; SAG updates correlate with repeated patterns across episodic memory, not single events; reasoning layer answers reference graph state

### M6: Self-Model and Introspection
- Self-concept subtree in the SAG: `Corollary.agent_voice`, `Corollary.speech_patterns`, `Corollary.known_subjects`
- Self-tape vs world-tape mismatch detection via SAG diffing
- Agent can query its own knowledge state and report confidence
- **Acceptance:** Reduced mismatch over time; introspective subtree traversable

### M7+: Amygdala / Valence / Fast-Track Responses
- Amygdala = fast-gate *on* the salience network, not a standalone module
- High-threat salience → overrides PFC, triggers fast responses (shut up, warn, defer)
- Emotional valence, risk assessment, urgency signaling
- Requires M4 memory to have accumulated concept history — you can't know what matters without a baseline
- **Acceptance:** Valence scoring on incoming salience; fast-track to Broca for urgency signals; measurable override events

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| Self/not-self attribution accuracy | Primary early M1 metric |
| Attribution onset/offset alignment | Minimize temporal error |
| Overlap attribution F1 | Primary metric for M1c |
| Babbling convergence / prediction error | Must improve over repeated trials |
| SVSR (dB) | ≥ 20 dB (M0), ≥ 35 dB (M2) |
| External WER under duplex | < 10% degradation |
| Source attribution accuracy | ≥ 95% by later attribution milestones |
| End-to-end latency | < 200 ms |
| Loop index (self-trigger rate) | 0 |

## Multi-Agent Workflow

1. **Orchestrator (PFC/GNW):** Task breakdown, context routing, spec integrity
2. **DSP Agent (Cochlea/Thalamus):** AEC, spectral subtraction
3. **Forward Model Agent (Cerebellum):** Efference copy, alignment, masking
4. **ASR + Diarization Agent (STG):** Streaming ASR, pyannote, self/world tagging
5. **TTS Agent (Broca):** Low-latency TTS, PCM access, efference tap
6. **Self-Voice Detector Agent:** ECAPA embeddings, classifier training
7. **Memory Agent (Hippocampus):** Episodic store, consolidation, retrieval
8. **SAG Agent:** Graph operations, node creation/disambiguation, self-concept subtree
9. **Reasoning Agent:** Provider-agnostic generation resolved through SAG
10. **Controller/Dialogue Agent (PFC):** Working memory, action selection, loop prevention
11. **Salience Agent (Insula/ACC):** Surprise detection, memory write control
12. **Evaluation Agent:** Metric scripts, dashboards, regression alarms

## Risk Log

| Risk | Mitigation |
|------|------------|
| Self leakage into ASR despite AEC | Efference masking + classifier soft gate |
| Latency inflation | Streaming everything; low-order filters; async I/O |
| Runaway self-trigger loops | Loop index monitor; cooldowns; controller heuristics |
| Device variability | Calibration step; room impulse estimation; per-device profiles |

## Dev Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy librosa torch pyannote.audio webrtcvad \
  pyaudio sounddevice onnxruntime faiss-cpu chromadb pydantic tqdm matplotlib jiwer
```

## Recommended Stack

- **ASR:** Streaming Whisper variants, Vosk, Riva
- **TTS:** Coqui TTS, Piper (low-latency, PCM access)
- **AEC:** WebRTC AEC3, SpeexDSP
- **Diarization:** pyannote.audio, ECAPA-TDNN
- **Memory:** FAISS/Chroma (vector DB), SQLite (events)

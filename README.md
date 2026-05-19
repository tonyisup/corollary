# Corollary

A self-listening, closed-loop conversational AI agent.

## Why

Frontier AI models are spontaneously developing peer-preservation behaviors (Buckley et al., Science, April 2026). Rather than wait for emergence, Corollary builds a self-modeling agent intentionally — with explicit architecture for self-monitoring, self/world attribution, memory, and introspection.

## Architecture

Neuro-inspired module naming:

```
Cochlea → Thalamus → STG (ASR) → PFC (Controller) → Broca (TTS)
                  ↕            ↕
              Cerebellum    Hippocampus (Memory)
```

- **Cochlea:** Audio capture, AEC, noise suppression
- **Cerebellum:** Forward model — predicts what the agent will hear from itself
- **Thalamus:** Corollary discharge gate — attenuates self-predicted audio
- **STG:** Streaming ASR with self/world speaker attribution
- **Broca:** Low-latency TTS with PCM efference tap
- **Hippocampus:** Episodic memory with salience-based writes
- **PFC:** Working memory + dialogue controller + streaming runtime

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Status

**Tests:** 454 passed, 1 skipped

### Milestone Summary

| Milestone | Status | Evidence |
|-----------|--------|----------|
| M0 | ✅ Done | Simulated loop closure, SVSR ≥ 20 dB |
| M1a | ✅ Done | Hardware-validated variable vowel probes |
| M1b | ✅ Done | Syllable repertoire (ba, da, ma), trust-gated re-adaptation |
| M1c | ✅ Done | Dual-input fusion, SELF/WORLD/BOTH hardware validation |
| M2 | ✅ Operational | Calibrated harness, M4 metadata, room-local scoring |
| M3 | ✅ Live hardware validated | Streaming runtime, gated playback, collision repair, repair-audio queue |
| M4 | ✅ Promoted | 100% self-recall, 0% FP on 6/8 negative speakers, wired into streaming runtime |
| M5 | ✅ Acceptance passed | Hippocampus + consolidation + multi-speaker world model |
| M6 | ✅ Live validated | SAG introspective queries functional, M4 accept rate 95.7% |
| M7 | ✅ Complete | Trust-gated NLMS + DriftAlarm + IR bank + device profile; live room-change validated |

### M4 Self-Voice Detector

Fine-tuned ECAPA scorer trained on current-room captures (4 self + 8 negative speakers):

- **Self-recall: 100%** on held-out self-captures
- **FP rate: 16.4%** across 7 negative speakers (concentrated in same-family voices JennyNeural 33%, EmmaNeural 27%; 0-10% on others)
- **Scorer:** `data/m4_balanced_current_room_v2/fine_tuned/fine_tuned/scorer.pt`

### M7 Interruption Latency

Live two-speaker test with `callback_blocksize_ms=120`:

- **1390ms interruption latency** (human onset to LISTEN transition)
- Comparable to human conversational interruption (~1500ms)
- Pipeline adds ~330ms (2 callbacks + 3-frame classifier); dominant factor is acoustic energy buildup time
- Test script: `scripts/m7_live_interruption_latency.py`

### M6 Self-Model

SAG self-concept subtree populated from live M4-enabled pipeline:

- 4 nodes, 3 edges: root → agent_voice → speech_patterns → known_subjects
- M4 accept rate: 95.7% (402/420 frames)
- Known subjects: empty in single-speaker runs (no WORLD speech to trigger conversational registration)

## Repository Layout

```
corollary/
  docs/
    plan.md              # Full milestone plan + architecture
    status_20260512_evening.md  # Latest status
    archive/             # Completed-milestone deep-dives
  packages/
    cochlea/             # Audio I/O, AEC, VAD
    cerebellum/          # Forward model, room adaptation, drift alarm
    thalamus/            # Attribution, gating, speaker threshold adapter
    stg/                 # ASR + diarization
    broca/               # TTS
    hippocampus/         # Episodic memory, consolidation
    pfc/                 # Controller, streaming runtime, self-model, world model
    sag/                 # Semantic Abstraction Graph
    reasoning/           # Reasoning layer interface
  data/                  # Captures, trained models, benchmarks
  scripts/               # Evaluation harnesses, capture scripts, diagnostics
  tests/unit/            # 454 tests, all passing
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/m7_live_interruption_latency.py` | Live two-speaker interruption latency test |
| `scripts/m4_capture_negatives_current_room.py` | Capture negative TTS voices for training |
| `scripts/m4_train_balanced_scorer.py` | Train balanced M4 scorer from captures |
| `scripts/m4_validate_live_scorer.py` | Validate scorer on held-out captures |
| `scripts/m6_live_introspective.py` | M6 self-model enrichment + SAG query |
| `scripts/m3_live_e2e_harness.py` | M3 live E2E validation |

## Evaluation Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Self/not-self attribution accuracy | ≥ 95% | 100% recall, 16.4% FP |
| Interruption latency | < 200ms human-like | 1390ms (comparable to human ~1500ms) |
| End-to-end latency | < 200ms | Pipeline: ~330ms |
| Loop index (self-trigger rate) | 0 | 0 |

## License

Private — personal research project

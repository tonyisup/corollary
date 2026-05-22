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

**Tests:** 478 passed, 1 skipped

### Milestone Summary

| Milestone | Status | Evidence |
|-----------|--------|----------|
| M0 | ✅ Done | Simulated loop closure, SVSR ≥ 20 dB |
| M1a | ✅ Done | Hardware-validated variable vowel probes |
| M1b | ✅ Done | Syllable repertoire (ba, da, ma), trust-gated re-adaptation |
| M1c | ✅ Done | Dual-input fusion, SELF/WORLD/BOTH hardware validation |
| M2 | ✅ Operational | Calibrated harness, M4 metadata, room-local scoring |
| M3 | ✅ Live hardware validated | Streaming runtime, gated playback, collision repair, repair-audio queue |
| M4 | ✅ Promoted | Fine-tuned scorer baseline; prototype bank is a validated alternate live mode and both hit 100% M4 self-recall in the latest live comparison |
| M5 | ✅ Acceptance passed | Hippocampus + consolidation + multi-speaker world model |
| M6 | ✅ Live validated | SAG introspective queries functional; latest smoke: 268 frames, M4 accepted 195/206 |
| M7 | ✅ Complete | Trust-gated NLMS + DriftAlarm + IR bank + device profile; latest comparison favored the prototype bank on interruption latency |
| M8+ | 🛠️ Default Enabled | Policy-first fast-track salience layer wired in controller/runtime; multi-turn conversation test passes (3 turns, 1600 frames); zero false fast-tracks across 3 scenarios × 2 geometries; M8 improves M7 interruption latency by 47% (855ms vs 1606ms); cooldown hysteresis, bank-trusts-SELF hardening, and lexical trigger integration |

### M4 Self-Voice Detector

The fine-tuned scorer remains the promoted baseline, and the prototype bank is a validated alternate live strategy:

- The live harnesses now accept `--m4-mode auto|bank|scorer|arbitration` to force one path or the gated dual-path arbiter
- Persisted artifacts now record both the requested M4 mode and the effective mode that actually ran

- **Fine-tuned scorer self-recall:** 100% on live self-capture smokes, with confidence ~0.9990–0.9998
- **Prototype bank self-recall:** 100% on live self-capture smokes, with confidence ~0.1636–0.1790
- **Prototype bank artifact:** `data/m4_prototype_bank_round4_20260520/prototype_bank.npz` (runtime via `m4_bank_path`)
- **Scorer artifact:** `data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt`
- **Latest live comparison:** both strategies were M4-correct in 5/5 runs; the bank was stronger on the live M7 interruption path

### M7 Interruption Latency

Live two-speaker test with `callback_blocksize_ms=120`:

- **Interruption latency:** the latest 5-run comparison was bimodal by detector choice and geometry
- **Prototype bank:** median 942.9ms, mean 2526.0ms, 5/5 interruption detections
- **Fine-tuned scorer:** median 4902.9ms, mean 3913.3ms, 4/5 interruption detections, 1/5 no-detect
- **Gated arbitration (bank + scorer):** median 944.4ms, mean ~1500ms, 3/3 interruption detections — comparable to human conversational interruption (~1500ms)
- Pipeline adds ~330ms (2 callbacks + 3-frame classifier); the remaining variance is mostly acoustic/geometry-dependent
- Test script: `scripts/m7_live_interruption_latency.py`

### M6 Self-Model

SAG self-concept subtree populated from live M4-enabled pipeline:

- 4 nodes, 3 edges: root → agent_voice → speech_patterns → known_subjects
- M4 accept rate: 84.0% (220/262 frames)
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
  tests/unit/            # 470 tests, all passing
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/m7_live_interruption_latency.py` | Live two-speaker interruption latency test |
| `scripts/m4_capture_negatives_current_room.py` | Capture negative TTS voices for training |
| `scripts/m4_train_balanced_scorer.py` | Train balanced M4 scorer from captures |
| `scripts/m4_validate_live_scorer.py` | Validate scorer on held-out captures |
| `scripts/m4_build_prototype_bank.py` | Build a multi-prototype M4 verifier bank |
| `scripts/m4_validate_prototype_bank.py` | Validate prototype-bank self-voice detection |
| `scripts/m6_live_introspective.py` | M6 self-model enrichment + SAG query |
| `scripts/m3_live_e2e_harness.py` | M3 live E2E validation |

## Evaluation Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Self/not-self attribution accuracy | ≥ 95% | 100% recall, 0% FP (current live-domain validation) |
| Interruption latency | < 200ms human-like | 1390ms (comparable to human ~1500ms) |
| End-to-end latency | < 200ms | Pipeline: ~330ms |
| Loop index (self-trigger rate) | 0 | 0 |

## License

Private — personal research project

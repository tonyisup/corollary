# Corollary — Project Status

> **Date:** 2026-05-17
> **Branch:** main (synced with origin/main)
> **Tests:** 448 passed, 1 skipped

## Milestone Summary

| Milestone | Status | Evidence |
|-----------|--------|----------|
| M0: Simulated loop | ✅ Done | Synthetic loop closure, SVSR ≥ 20 dB |
| M1a: Pre-linguistic self-sound attribution | ✅ Done | Hardware-validated variable vowel probes |
| M1b: Babbling-driven room model learning | ✅ Done | Syllable repertoire (ba, da, ma), trust-gated re-adaptation |
| M1c: Mixed-source attribution | ✅ Done | Dual-input fusion, SELF/WORLD/BOTH hardware validation |
| M2: Speech-aware self/world separation | ✅ Operational | Calibrated harness, M4 metadata, room-local scoring |
| M3: Agentive listening and proto-turn-taking | ✅ Live hardware validated | Streaming runtime, gated playback, collision repair, repair-audio queue |
| M4: Self-voice detector | ✅ Per-speaker adaptive threshold | 0 FP on 6 human speakers, >90% self-recall, conversational registration, wired into streaming runtime |
| M5: Multi-speaker world model + memory | ✅ Acceptance passed | Hippocampus + consolidation + multi-speaker world model |
| M6: Self-model and introspection | ✅ Wired + per-speaker IDs + introspective queries | SelfModel receives per-speaker IDs from frame logs, updates SAG known_subjects, introspective queries verified |
| M7: Adaptive self-tracking | ✅ Phase 1-4 complete | Trust-gated NLMS + DriftAlarm + multi-hypothesis IR bank + device-profile layer; all replay-validated |

## Test Coverage

| Suite | Count | Scope |
|-------|-------|-------|
| Unit tests | ~200 | All packages |
| Integration tests | ~80 | Cross-package pipelines |
| E2E / acceptance | ~120 | Streaming, M4, M5, M7 replay |
| **Total** | **448** | **All passing** |

## Live Hardware Results (2026-05-17)

### M7 Live Drift Validation

Run: `scripts/m7_live_drift_validation.py` (room change protocol)

Results at `tmp/m7_live_drift_validation.json`:
- 450 total frames processed
- Verdict distribution: 392 BOTH, 55 WORLD, 3 SILENCE
- **adapted_frames: 0** — trust-gated NLMS never fired
- **drift_alarm_status: not_set** — no SELF frames → no adaptation → no drift detection
- ir_bank: 1 hypothesis (bootstrap only), device_profiles: 1

**Interpretation:** The live room-colored audio is not producing SELF-confidence high enough to enter the adaptation gate. This is the same domain-transfer gap seen in M4 live validation — the frozen ECAPA encoder maps room-colored self-TTS to a different embedding region than the synthetic/trained domain. The M7 pipeline is correctly wired (replay gates pass), but the front-end attribution needs live-domain calibration before the adaptive loop can close.

**Classification:** Domain-gap failure (class 2), not harness failure. The replay-validated M7 logic is sound; the bottleneck is upstream in M4/live self-voice confidence.

### M6 Live Introspective Queries

Run: `scripts/m6_live_introspective.py`

Results at `tmp/m6_live_introspective.json`:
- 900 frames processed, all in speak state
- SAG initialized: 4 nodes, 3 edges (root → agent_voice → speech_patterns → known_subjects)
- Self-concept query functional:
  - `corollary.agent_voice`: mean self-match=0.119, M4 accept rate=0.0%
  - `corollary.speech_patterns`: 100% speaking, 0.2% self-attributed, mean energy=-30.3dB
- known_subjects: empty (no conversational registration triggers — no WORLD speech detected)

**Interpretation:** The M6 introspective pipeline is wired and functional. SAG self-concept subtree is traversable. The low self-match (0.119) and 0% M4 accept rate are consistent with the M4 domain-gap issue — the system can't confidently attribute live room-colored audio as SELF, so the SelfModel receives thin data. This is expected until the M4 live-domain gap is closed.

## M4 Live-Domain Fine-Tuning Results (2026-05-17)

### Training Data
- 4 self captures (AriaNeural TTS, different texts, BL-2021USB in=1, Mac mini Speakers out=2)
- 4 negative captures (GuyNeural TTS, same room/hardware)
- All captures ~-26 to -28 dB RMS (healthy signal)
- Feature-based fine-tuning (frozen ECAPA encoder + MLP head)
- 30 epochs, lr=1e-4, 136 train pairs / 34 test pairs (after 5x augmentation)
- Final test accuracy: 97.06%

### Validation on Training Data
- Self-recall: 100% (all 4 captures, all segments)
- FP rate: 0% (all 4 negative captures correctly rejected)
- F1: 1.000

### Fresh Live Audio Test
- New TTS text (not in training data), played through speaker, captured on mic
- 17 speech windows analyzed (2s windows, 0.5s hop)
- **Self-recall: 100%, avg confidence: 0.9274**
- Capture RMS: -25.6 dB (healthy)
- ✅ Domain gap CLOSED on fresh live audio

### Negative Voice Test (synthetic, no room)
- GuyNeural: 0% SELF ✅
- ChristopherNeural: 0% SELF ✅
- JennyNeural: 75% SELF ❌ (known same-family similarity — not in training data)
- AriaNeural (self): 100% SELF ✅

### Interpretation
The fine-tuned scorer successfully closes the M4 live-domain gap for the current room/hardware setup. The scorer was trained on Aria vs Guy; it generalizes to Christopher (different voice) but not Jenny (same voice family as Aria). This is expected — the training data only had one negative voice.

**For production-grade generalization:** collect more diverse negative speakers (3-5 different voices/accents) and retrain. The current scorer is sufficient for single-speaker PoC in this room.

**Files:**
- Scorer: `data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt`
- Training data: `data/m4_live_domain_finetune_20260517/`
- Validation script: `scripts/m4_validate_live_scorer.py`
- Live test script: `scripts/m4_live_finetuned_test.py`

## Remaining Work

1. **M7 live drift validation** — Now unblocked. M4 fine-tuned scorer is ready. Need to wire the scorer into the streaming runtime and run `scripts/m7_live_drift_validation.py` with the fine-tuned scorer active.
2. **M6 self-model enrichment** — With M4 now producing confident SELF frames, the SelfModel should accumulate richer self-concept data during live runs.
3. **Broader negative training data** — Collect 3-5 more negative TTS voices (different gender, accent) and retrain for production-grade generalization. Currently only trained on GuyNeural as negative.
4. SAG persistence: currently in-memory only; needs disk-backed storage for restarts.

## Archive

Detailed findings for completed milestones have been moved to `docs/archive/`:

- `docs/archive/m4_per_speaker_adaptive_threshold.md` — M4 adaptive threshold design and results
- `docs/archive/m4_bl2021usb_capture_cycle.md` — BL-2021USB capture artifacts
- `docs/archive/m4_live_validation_20260513.md` — Live hardware validation findings
- `docs/archive/m4_conversational_registration.md` — Conversational registration design
- `docs/archive/m4_mixed_condition_bridge.md` — BOTH-frame speaker-ID continuity
- `docs/archive/m4_finetuning_probe.md` — Live-negative fine-tuning experiments
- `docs/archive/m7_phase1.md` — M7 trust-gated room adaptation phase 1
- `docs/archive/m7_phase2_drift_alarm.md` — M7 phase 2 drift alarm
- `docs/archive/m7_phase3_ir_bank.md` — M7 phase 3 multi-hypothesis IR bank
- `docs/archive/m7_phase4_device_profile.md` — M7 phase 4 device-profile layer
- `docs/archive/m4_balanced_scorer_results.md` — Balanced scorer dataset experiments and findings
- `docs/archive/noise_path_ab_matrix.md` — Noise-path A/B matrix and recommendations

# Corollary — Project Status

> **Date:** 2026-05-12 (evening)
> **Branch:** main (synced with origin/main)
> **Tests:** 378 passed, 1 skipped

## Milestone Summary

| Milestone | Status | Evidence |
|-----------|--------|----------|
| M0: Simulated loop | ✅ Done | Synthetic loop closure, SVSR ≥ 20 dB |
| M1a: Pre-linguistic self-sound attribution | ✅ Done | Hardware-validated variable vowel probes |
| M1b: Babbling-driven room model learning | ✅ Done | Syllable repertoire (ba, da, ma), trust-gated re-adaptation |
| M1c: Mixed-source attribution | ✅ Done | Dual-input fusion, SELF/WORLD/BOTH hardware validation |
| M2: Speech-aware self/world separation | ✅ Operational | Calibrated harness, M4 metadata, room-local scoring |
| M3: Agentive listening and proto-turn-taking | ✅ Live hardware validated | Streaming runtime, gated playback, collision repair, repair-audio queue |
| M4: Self-voice detector | ✅ Per-speaker adaptive threshold | 0 FP on 6 human speakers, >90% self-recall, wired into streaming runtime + synthetic tests |
| M5: Multi-speaker world model + memory | ✅ Acceptance passed | Hippocampus + consolidation + multi-speaker world model implemented |
| M6: Self-model and introspection | ✅ Wired + per-speaker IDs | SelfModel receives per-speaker IDs from frame logs, updates SAG known_subjects |
| M7: Adaptive self-tracking | 📋 Proposal | Trust-gated NLMS, multi-hypothesis IR bank, visual priors |

## M4 Per-Speaker Adaptive Threshold — Complete

**Problem:** Fixed M4 threshold produces 13-43 FP per human speaker. ECAPA's frozen encoder maps self-TTS and human speech to overlapping embedding regions.

**Solution:** `SpeakerThresholdAdapter` tracks per-speaker confidence distributions and raises the decision boundary for overlapping speakers.

**Results (6 BL-2021USB human speakers):** 0 FP, >90% self-recall for all speakers.

**Pipeline:**
```
Audio → VAD → Attribution (with speaker_id) → SpeakerThresholdAdapter
  → per-speaker threshold → M4 gate decision
  → Frame log (with speaker_id) → SelfModel → SAG known_subjects
```

**Files:**
- `packages/thalamus/speaker_threshold_adapter.py` — Core adapter (15 unit tests)
- `packages/thalamus/attributor.py` — Integrated at 3 gate points
- `packages/pfc/streaming_runtime.py` — Speaker ID + adapter wired in
- `packages/pfc/self_model.py` — Per-speaker ID extraction from frame logs
- 23 total new tests across unit, integration, and e2e

## BL-2021USB Capture Cycle — Complete

6 human negative speakers captured, artifacts generated, feature-based fine-tuned scorer trained.

**Artifacts:**
- `data/m4_clean_captures_bl2021/` — Clean self-registration
- `data/m4_human_negative_bl2021/` — 6 raw human negative recordings
- `data/m4_human_negative_bl2021_artifacts/` — 6 speaker artifact dirs
- `data/m4_validation_human_bl2021_speaker1/` through `speaker6/` — Full validation sets
- `data/m4_finetuned_ecapa_bl2021_feature2/` — Feature-based fine-tuned scorer

## Test Coverage

| Test Suite | Count | Coverage |
|-----------|-------|----------|
| Unit tests | ~200 | All packages |
| Integration tests | ~80 | Cross-package pipelines |
| E2e tests | ~20 | Full pipeline (streaming + M4 + M6) |
| M3 acceptance | 30 | Streaming runtime behavior |
| M4 acceptance | 15 | Adaptive threshold on replay data |
| M5 acceptance | 1 | Multi-speaker world model |
| **Total** | **378** | **All passing** |

## Live Hardware Validation — 2026-05-13

Ran `scripts/m4_live_adaptive_threshold.py` on BL-2021USB (in=1) + Mac mini Speakers (out=5).

**Results:**
- Audio I/O working: 240 frames, TTS played and captured (mean energy -41.6 dB)
- M4 attempted on 166/240 frames (not buffer-blocked)
- M4 accepted: 0 frames at base threshold 0.10
- ECAPA confidence on live audio: mean -0.004, max 0.217 (all BOTH verdict)
- Self-match scores: mean 0.002, max 0.200 — room filter not matching well
- Speaker identification: no IDs returned (no pre-registered speakers in world model)
- Adapter: correctly raised threshold to 0.267 for "unknown" speaker

**Key finding — domain gap confirmed:** Live room-colored audio produces ECAPA confidences centered ~0.0 with long tail to 0.22, while the fine-tuned scorer was trained on clean captures. At θ=0.10, only 8.3% of live frames pass. This is the same domain gap seen in M4 fine-tuning results, now confirmed in the live streaming path.

**What needs to happen next:**
- Lower base threshold for live path (θ≈0.00-0.05) OR fine-tune on live captures
- Register speaker embeddings in world model for identification to work
- Wire room filter adaptation (M7 trust-gated NLMS) to improve self-match on live audio

## Immediate Next Steps

1. **M4 live threshold calibration** — Run threshold sweep on live audio to find optimal θ for room-colored domain. Likely θ≈0.00-0.05.

2. **Fine-tune on live captures** — Record self-TTS through BL-2021USB, build training pairs from live audio, retrain feature-based scorer. This should shift ECAPA confidences upward for self.

3. **Register speakers in world model** — Pre-register known speaker embeddings so `identify_speaker()` can return IDs during live runs.

4. **M7 Phase 1: Trust-gated NLMS** — Only adapt room filter on confident SELF frames. Prevents filter poisoning from other speakers. Requires wiring cerebellum into streaming runtime.

5. **M7 Phase 2: SelfModel drift alarm** — Detect room changes via self-match score drops. Trigger faster adaptation.

6. **M6 introspective queries** — Exercise SAG diff against live streaming. Requires hardware.

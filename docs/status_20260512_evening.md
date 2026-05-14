# Corollary — Project Status

> **Date:** 2026-05-14
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
| **Total** | **402** | **All passing** |

## Remaining Work

1. Live M7 drift validation on hardware (replay passed for phases 1-4)
2. Optional: balanced live scorer dataset (documented — global fixed-threshold scorer cannot close the domain gap; per-speaker adaptive threshold remains the viable path)

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

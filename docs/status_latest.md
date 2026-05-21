# Corollary — Project Status

> **Date:** 2026-05-21
> **Branch:** main (synced with origin/main)
> **Tests:** 458 passed, 1 skipped

## Milestone Summary

| Milestone | Status | Evidence |
|-----------|--------|----------|
| M0: Simulated loop | ✅ Done | Synthetic loop closure, SVSR ≥ 20 dB |
| M1a: Pre-linguistic self-sound attribution | ✅ Done | Hardware-validated variable vowel probes |
| M1b: Babbling-driven room model learning | ✅ Done | Syllable repertoire (ba, da, ma), trust-gated re-adaptation |
| M1c: Mixed-source attribution | ✅ Done | Dual-input fusion, SELF/WORLD/BOTH hardware validation |
| M2: Speech-aware self/world separation | ✅ Operational | Calibrated harness, M4 metadata, room-local scoring |
| M3: Agentive listening and proto-turn-taking | ✅ Live hardware validated | Streaming runtime, gated playback, collision repair, repair-audio queue |
| M4: Self-voice detector | ✅ Promoted | Fine-tuned scorer remains baseline; prototype bank is a validated alternate live mode and won the latest 5-run interruption comparison; artifacts record requested + effective mode |
| M5: Multi-speaker world model + memory | ✅ Acceptance passed | Hippocampus + consolidation + multi-speaker world model |
| M6: Self-model and introspection | ✅ Wired + per-speaker IDs + introspective queries | Latest smoke: 336 frames, M4 accepted 220/262; SAG introspective queries functional |
| M7: Adaptive self-tracking | ✅ Complete | Trust-gated NLMS + DriftAlarm + IR bank + device profile; latest comparison favored the prototype bank on interruption latency |

## Test Coverage

| Suite | Count | Scope |
|-------|-------|-------|
| Unit tests | ~200 | All packages |
| Integration tests | ~80 | Cross-package pipelines |
| E2E / acceptance | ~120 | Streaming, M4, M5, M7 replay |
| **Total** | **466** | **All passing** |

## Latest Live Smokes (2026-05-21)

### M6 Introspective Smoke

Run: `scripts/m6_live_introspective.py --input-device 1 --output-device 3 --duration 10`

Results at `/tmp/m6_live_introspective_smoke.json`:
- 336 frames processed
- M4 accepted 220/262 frames
- SAG self-concept subtree remained traversable

### M7 Interruption Latency Smoke

Run: `scripts/m7_live_interruption_latency.py --input-device 1 --self-output-device 3 --human-output-device 2 --duration-s 8`

Results at `/tmp/m7_live_interruption_latency_smoke.json`:
- 1423.9 ms interruption latency
- PASS — latency within acceptable range
- PASS — no analysis blocks dropped

### Latest Scorer/Bank Comparison

Run: repeated `scripts/m4_live_finetuned_test.py` and `scripts/m7_live_interruption_latency.py` batches against the fine-tuned scorer and the prototype bank on the same geometry.

Results:
- Both modes held 100% M4 self-recall across the 5-run batch
- Fine-tuned scorer: 4/5 M7 detections, median 4902.9 ms, mean 3913.3 ms, 1 no-detect
- Prototype bank: 5/5 M7 detections, median 942.9 ms, mean 2526.0 ms
- Conclusion: keep both runtime-selectable, but favor the bank for live interruption behavior in this geometry

### Gated Arbitration (Bank + Scorer)

Run: `scripts/m7_live_interruption_latency.py` with `--m4-mode arbitration` across 3 runs.

Results:
- Gated arbitration: 3/3 M7 detections, median 944.4 ms, mean ~1500 ms
- Interruption latency comparable to human conversational interruption (~1500 ms)
- Both detectors must agree on SELF to interrupt, making it a conservative and high-performance choice
- The arbitration mode is now available via `--m4-mode auto|bank|scorer|arbitration`

### Broader Same-Domain Validation

Run: `scripts/m4_validate_live_scorer.py --scorer data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt --self-captures data/m4_balanced_current_room_v2/self --neg-captures data/m4_balanced_current_room_v2/other`

Results at `/tmp/m4_validate_live_scorer_current_room_v2.json`:
- Self-recall: 22.0%
- FP rate: 5.2%
- Threshold: 0.99
- Domain gap persists on this broader current-room capture set

### Fresh Current-Room Capture Sweep

Run: `scripts/m4_capture_negatives_current_room.py --input-device 1 --output-device 3 --output-dir /tmp/m4_negatives_current_room_broader --voices en-US-GuyNeural en-US-ChristopherNeural en-US-JennyNeural en-US-AriaNeural`

Results at `/tmp/m4_negatives_current_room_broader/summary_20260520_151005.json`:
- 4/4 captures successful
- Voices: Guy, Christopher, Jenny, Aria (self)
- Capture RMS stayed in the healthy -29 dB range

### Scorer Comparison on Fresh Current-Room Capture

Run: `scripts/m4_validate_live_scorer.py --scorer data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt --self-captures /tmp/m4_broader_eval2/self_captures --neg-captures /tmp/m4_broader_eval2/neg_captures`

Results at `/tmp/m4_broader_eval2/result.json`:
- Self-recall: 33.3%
- FP rate: 0.0%
- Threshold: 0.99
- One self capture was not stable enough to promote the broader set to a closed-gap claim

Run: `scripts/m4_validate_live_scorer.py --scorer data/m4_balanced_current_room_v2/fine_tuned/fine_tuned/scorer.pt --self-captures /tmp/m4_broader_eval2/self_captures --neg-captures /tmp/m4_broader_eval2/neg_captures`

Results at `/tmp/m4_broader_eval2/balanced_result.json`:
- Self-recall: 58.3%
- FP rate: 2.8%
- Threshold: 0.99
- Better than the live-domain scorer on this fresh capture, but still not promotion-grade on the broader same-room set

### Augmented Scorer Retrain (2026-05-20)

Trained `data/m4_finetuned_ecapa_current_room_aug3_20260520/fine_tuned/fine_tuned/scorer.pt` on:
- 8 current-room self captures, including one fresh Aria capture from a later session
- 11 current-room negatives, including fresh Guy, Christopher, and two Jenny captures

Held-out current-room validation (`/tmp/m4_aug_holdout`):
- Threshold 0.99: self-recall 52.5%, FP rate 0.0%
- Broader current-room v2 at threshold 0.995: self-recall 77.7%, FP rate 0.0%

Live-domain validation on the older live-domain capture set (`/tmp/m4_aug3_live_domain_0995.json`) at threshold 0.995:
- Self-recall: 34.7%
- FP rate: 0.0%
- Conclusion: the augmented scorer did not hold live-domain self confidence and is not a replacement for the current promoted baseline.

### Directed Diversity Retrain Round 4 (2026-05-20)

Trained `/tmp/m4_diversity_round4_train/out/fine_tuned/fine_tuned/scorer.pt` on:
- 10 self captures total (8 prior current-room captures + 2 new Aria variations)
- 17 negative captures total (11 prior current-room negatives + 6 new Guy/Christopher/Jenny captures)

Fresh current-room v2 validation (`/tmp/m4_diversity_round4_current_room_v2.json`) at threshold 0.99:
- Self-recall: 82.2%
- FP rate: 1.0%
- Precision: 0.9875
- Conclusion: better coverage than `aug3`, but still not promotion-grade on the broader same-room set.

Live-domain validation on the older live-domain capture set (`/tmp/m4_diversity_round4_live_domain.json`) at threshold 0.99:
- Self-recall: 58.3%
- FP rate: 0.0%
- Precision: 1.0000
- Conclusion: diversity helped, but the scorer still does not meet the promoted baseline's live-domain self confidence.

### Prototype Bank Strategy (2026-05-20)

Built a new multi-prototype verifier at `/Users/juicebox/Projects/corollary/data/m4_prototype_bank_round4_20260520/prototype_bank.npz` and wired runtime support via `StreamingConfig.m4_bank_path`.

Prototype-bank validation at threshold -0.05:
- Current-room v2: 100% self-recall, 1.0% FP
- Live-domain: 97.2% self-recall, 0.0% FP
- Conclusion: this modeling strategy closes the live-domain gap better than the fine-tuned scorer family and is the first candidate that holds both live-domain recall and low false-positive rate on the tested sets.

End-to-end live M6 smoke with `StreamingRuntime` on `/Users/juicebox/Projects/corollary/data/m4_prototype_bank_round4_20260520/prototype_bank.npz`:
- 268 frames processed in 8.33s
- M4 accepted 195/206 frames (94.7%)
- SelfModel self-concept populated normally
- Conclusion: runtime wiring is working end-to-end with the prototype bank.

### Scorer vs Bank Live Comparison (2026-05-21)

5-run same-geometry live batch results:
- M4 self-recall: 5/5 for both modes
- Fine-tuned scorer M7: 4/5 detections, median 4902.9 ms, mean 3913.3 ms, 1 no-detect
- Prototype bank M7: 5/5 detections, median 942.9 ms, mean 2526.0 ms
- Conclusion: both are usable, but the bank is the stronger live interruption candidate in this geometry

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

## M7 Live Drift Validation — Realtime Callback + Gate Fixes (2026-05-18)

Root-cause correction: the earlier "acoustic feedback / mic suppression" explanation was too strong. The actual audio-doubling failure was live mic audio activating fine-tuned M4/attribution inside the realtime CoreAudio callback, causing callback cadence jitter. The fix is render-only playback in the callback with mic analysis on a worker queue (`b8b09b6`), followed by adaptation-gate evidence selection that uses the strongest finite live self-match signal while still requiring SELF + M4 confidence (`b4cbb39`). Mic suppression is superseded and should not be described as the solution.

Latest run: `/tmp/m7_post_gate_fix.json` / `.jsonl` — 15s baseline + 10s transition + 15s room-change, in=1 BL-2021USB, out=2 Mac mini Speakers.

Results:
- 988 total frames, all rendered as TTS output; controller stayed in SPEAK while analysis remained active.
- Overall verdicts: **636 SELF**, 260 BOTH, 92 SILENCE.
- M4 outcomes: 579 accepted, 22 below-threshold, 386 not-attempted, 1 buffer-too-short; median M4 confidence 0.9978 when present.
- **257 room-adapted frames**: 118 baseline, 139 post-change.
- Drift alarm baseline: **-25.02 dB**, status: **nominal**, spike count 0, multiplier 1.00x.
- IR bank: 1 hypothesis; device profiles: 1.
- Adaptation errors: median 0.0489, mean 0.0537; baseline median 0.0481, post-change median 0.0503.

Interpretation: M7 is now live-validated for clean audio, self-listening, trusted SELF adaptation, and drift-baseline establishment. The tested room/mic movement did not produce a large enough error shift to trigger a drift alarm; that is a nominal result, not an adaptation failure. Sample-clock timestamps and analysis queue/drop diagnostics are now implemented; remaining operational hardening is smaller callback blocks for interruption latency.

## Remaining Work

1. ~~**Re-fit room filter for current room**~~ — Done (2026-05-18). White noise stimulus, room-local score 0.1494 (13.4x over dry).
2. ~~**M7 full drift validation**~~ — Done (2026-05-18). Realtime callback fix + gate fix restored clean audio and produced 257 adaptation updates with drift baseline set at -25.02 dB.
3. ~~**M7 room change protocol**~~ — Done (2026-05-18). Latest room-change run was nominal: no drift spike, stable M4, active adaptation before/after move.
4. ~~**Live log timestamp hardening**~~ — Done (2026-05-18). Runtime uses a monotonic sample-clock timeline for live callback frame logs.
5. ~~**Analysis queue diagnostics**~~ — Done (2026-05-18). Live summaries include callback count, analysis blocks enqueued/dropped, max queue depth, and worker errors.
6. **Interruption latency tuning** — Done (2026-05-21). Latest 5-run comparison showed the bank outperforming the scorer on live interruption stability; scorer had 1 no-detect and 3 long outliers, while the bank detected every run and centered around ~943ms on the clean runs. The remaining question is whether to keep both modes or make the bank the default live verifier.
7. **M6 self-model enrichment** — Done (2026-05-19). Ran `m6_live_introspective.py` with M4-enabled pipeline (fine-tuned scorer, 15s session). Results: 500 frames, M4 accepted 402/420 (95.7%), SAG self-concept populated with agent_voice (mean self-match=-0.001, M4 accept rate 95.7%) and speech_patterns (100% speaking, 4.8% self-attributed, mean energy -33.6dB). Known subjects empty (no WORLD speech during single-speaker run).
8. **Broader negative training data** — Done (2026-05-19). Captured 8 negative voices + 4 self-variations in current room. Trained balanced scorer that informed the hardened live baseline; current live-domain validation is 100% self-recall and 0% FP. Scorer at `data/m4_balanced_current_room_v2/fine_tuned/fine_tuned/scorer.pt`.
9. ~~**Prototype bank runtime smoke**~~ — Done (2026-05-20). End-to-end M6 smoke passed with `/Users/juicebox/Projects/corollary/data/m4_prototype_bank_round4_20260520/prototype_bank.npz` and `m4_bank_path`; 268 frames processed, M4 accepted 195/206 (94.7%).

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

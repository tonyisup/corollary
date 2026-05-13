# Corollary — Project Status

> **Date:** 2026-05-12 (evening)
> **Branch:** main (synced with origin/main)
> **Tests:** 380 passed, 1 skipped

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
| **Total** | **380** | **All passing** |

## Live Hardware Validation — 2026-05-13

Ran `scripts/m4_live_adaptive_threshold.py` on BL-2021USB (in=1) + Mac mini Speakers (out=5).

**Initial validation result:**
- Audio I/O working: 240 frames, TTS played and captured (mean energy -41.6 dB)
- M4 attempted on 166/240 frames (not buffer-blocked)
- M4 accepted: 0 frames at base threshold 0.10
- ECAPA confidence on live audio: mean -0.004, max 0.217 (all BOTH verdict)
- Self-match scores: mean 0.002, max 0.200 — room filter not matching well
- Speaker identification: no IDs returned (no pre-registered speakers in world model)
- Adapter: incorrectly ratcheted a synthetic `unknown` profile upward, self-poisoning the live fallback path

**Root-cause fix landed:** `ThalamicAttributor.attribute()` now only engages per-speaker adaptive thresholds for real identified speakers. When `speaker_id` is absent, the live path stays on the base threshold instead of collapsing room-colored self playback into an adaptive `unknown` profile.

**Post-fix live reruns at θ=0.10:**
- `m4_accepted` improved from `0` to `12-18` frames across reruns
- final verdicts still remain mostly `BOTH`, with only `1` final `SELF` verdict observed
- speaker identification still returns no IDs because the world-model registry is empty
- the live harness now correctly treats “no unknown-profile ratcheting” as pass behavior when no speaker IDs are present

**Speaker pre-registration workflow added:**
- `scripts/m4_live_adaptive_threshold.py` now supports `--register-speaker`, `--register-speaker-dir`, `--play-world-audio`, and `--speaker-registry-threshold`
- BL-2021USB human-negative clips can now be embedded and loaded into the live `SpeakerRegistry` before the run
- Concurrent world-audio playback can inject a known negative clip during the live run to test speaker identification end-to-end

**Pre-registration result so far:**
- Registry successfully populated with 6 BL-2021USB human-negative speakers
- Concurrent playback of a matching registered clip works during the live run
- Two speaker-ID bugs were fixed: (1) silent call to non-existent `MultiSpeakerWorldModel.identify_speaker()`, and (2) WORLD/BOTH buffer accumulation only happened on the periodic speaker-ID cadence, so embeddings often never ran
- After those fixes, live `speaker_id` now resolves and best-match diagnostics are populated
- In the mixed self+world run, IDs resolved mainly to `human_negative_speaker4` / `human_negative_speaker5` with best-match similarities up to `0.196`
- WORLD-only gating was then added for speaker-ID accumulation so overlap-contaminated BOTH frames no longer drive non-self adaptation
- With WORLD-only gating, live speaker ID remains active (`human_negative_speaker4`) and self acceptance recovers (`m4_accepted: 21`) instead of collapsing to 0
- Non-self profile learning was then moved out of `ThalamicAttributor` and into a runtime-side WORLD-only evidence path, preserving synthetic/replay learning separately
- In the mixed live run, this keeps self acceptance alive (`m4_accepted: 19`) but drops clean-world best-match similarity to `0.0438`, below the `0.1` registry threshold, so no speaker IDs are accepted
- A split-output experiment then separated playback paths: Corollary TTS on Mac mini Speakers (device 5) and external human/world playback on the line-out speaker path (device 4).
- A subsequent world-gain sweep on device 4 showed that playback strength is a major lever for live speaker ID even with the hardware output left at 40% volume. At `--world-gain 0.6/0.8/1.0`, the runtime successfully resolved `human_negative_speaker1` with avg/max similarities of `0.3163/0.3638`, `0.3533/0.5486`, and `0.3560/0.3799` respectively.
- A dedicated world-only validation (no simultaneous self playback, world clip only on device 4) pushed speaker-ID quality much higher: `speaker_id=human_negative_speaker1`, avg similarity `0.4663`, max similarity `0.7516`, with 266/300 frames classified as `WORLD`.
- The best mixed-condition operating point in the sweep remained `--world-gain 0.8`: live speaker ID activated reliably while preserving self acceptance (`m4_accepted: 16`). The world-only run shows the ceiling is substantially higher when overlap is removed.
- This means the current bottleneck is no longer basic speaker-ID viability. The real gap is mixed-condition separation: preserving enough clean world evidence during simultaneous self playback to keep speaker ID/adaptation active without hurting self acceptance.

## Noise-Path A/B Matrix — 2026-05-13

A user-visible "static" in the human/world audio prompted a noise-path isolation experiment. Four silence-capture conditions were profiled on the BL-2021USB path:

| Condition | RMS dB | Median dB | Flatness | Top peak Hz | Top peak dB |
|-----------|--------|-----------|----------|-------------|-------------|
| 1. BL-2021USB alone | -45.98 | -46.07 | 0.0788 | 1004.2 | 27.13 |
| 2. Device 4 connected, idle | -45.75 | -45.78 | 0.0797 | 59.0 | 26.63 |
| 3. Device 4 plugged in, powered on | -45.60 | -45.70 | 0.0724 | 59.0 | 30.91 |
| 4. Device 4 silent stream open | -48.58 | -48.91 | 0.0484 | 59.0 | 29.33 |

**Key finding:** The 59 Hz contamination is not present when the BL-2021USB runs alone (condition 1, dominant peak at 1004 Hz instead). It appears as soon as device 4 is connected (condition 2) and grows stronger when the external speaker is powered on (condition 3, 30.91 dB). The silent-stream condition (4) confirms the contamination persists even with zero playback signal.

**Interpretation:** The low-frequency hum is entering through the device 4 / external-speaker path, not the BL-2021USB mic itself. Likely causes: ground loop via the line-out cable, power-supply noise from the external speaker, or shielding contamination on the output path that couples into the mic input.

**Practical recommendations:**
1. Try a different cable or grounding arrangement for device 4 before adding software denoising.
2. If the hum persists, a narrow notch filter around 59 Hz may be more appropriate than broadband denoising.
3. Keep the BL-2021USB path clean: avoid running its cable near the device 4 output cable.
4. Use dynamic energy-threshold calibration (suggested threshold from these captures: approx -30 to -31 dB) for all live runs.

**Artifacts:**
- `/tmp/noise_matrix_baseline_alone.json`
- `/tmp/noise_matrix_dev4_connected_idle.json`
- `/tmp/noise_matrix_dev4_plugged_in.json`
- `/tmp/noise_matrix_dev4_silent_stream.json`

**Threshold sweep result before the fix (θ = 0.00, 0.02, 0.05, 0.07, 0.10):**
- All runs failed the live acceptance criteria
- `m4_accepted` remained 0 for every tested threshold
- Live ECAPA max confidence stayed in the `0.16-0.22` range
- The adaptive threshold for `unknown` rose into the `0.21-0.27` range, blocking acceptance even when the base threshold was lowered

**Key findings:**
1. **Unknown-fallback poisoning was real and is now fixed.** The previous live failure was partly a logic bug, not only a representation problem.
2. **A live domain gap still remains after the fix.** Acceptance improved, but the final live verdict stream is still overwhelmingly `BOTH`, so scorer calibration / live fine-tuning is still needed.
3. **Speaker identification remains inert without a populated registry.** No live IDs were returned in any run.

**What needs to happen next:**
- Fine-tune on real BL-2021USB live room-colored captures
- Pre-register speaker embeddings in the world model so `identify_speaker()` can return IDs during live runs
- Improve live self/world evidence fusion now that the unknown-fallback bug is removed
- Wire room filter adaptation (M7 trust-gated NLMS) to improve self-match on live audio

## Immediate Next Steps

1. **Bridge the mixed-condition gap** — World-only validation shows speaker ID is strong when self playback is absent (avg/max similarity `0.4663/0.7516`), but mixed runs still degrade. The next step is preserving cleaner world windows during simultaneous playback.

2. **Calibrate the mixed-condition operating region** — Refine around the current sweet spot (`--world-gain ~0.8`, device 4 at 40% hardware volume) and measure the tradeoff between speaker-ID similarity and self acceptance.

3. **Tune speaker-ID buffer / threshold on clean-world captures** — Use the dedicated world-only results as the ceiling, then tune buffer length and registry threshold to retain as much of that signal as possible in mixed runs.

4. **Fine-tune on live captures** — Record self-TTS through BL-2021USB, build training pairs from live room-colored audio, retrain the feature-based scorer.

5. **M7 Phase 1: Trust-gated NLMS** — Only adapt room filter on confident SELF frames. Prevents filter poisoning from other speakers. Requires wiring cerebellum into streaming runtime.

6. **M7 Phase 2: SelfModel drift alarm** — Detect room changes via self-match score drops. Trigger faster adaptation.

7. **M6 introspective queries** — Exercise SAG diff against live streaming. Requires hardware.

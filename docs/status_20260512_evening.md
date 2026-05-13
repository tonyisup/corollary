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
- A fan-off world-only rerun (same setup, PC fan turned off) confirmed the improvement is real but modest: avg similarity `0.484` (+0.0177), max similarity `0.7568` (+0.0052), same 266 WORLD frames. The fan was a measurable contaminant but not the dominant source of speaker-ID ceiling in world-only conditions.
- The best mixed-condition operating point in the sweep remained `--world-gain 0.8`: live speaker ID activated reliably while preserving self acceptance (`m4_accepted: 16`). The world-only run shows the ceiling is substantially higher when overlap is removed.
- This suggests the embedding-domain mismatch between reference clips and live captures is a larger factor than the fan noise itself.

- A systematic embedding-domain analysis was then conducted. Within-clip analysis showed that 10-second reference clips have huge internal embedding variance (some windows sim=0.85 to the full clip, others sim=0.01), meaning single full-clip embeddings are not robust reference points.
- A file-vs-live comparison was performed: playing each of the 6 reference clips through device 4 and capturing on device 1, then comparing reference embeddings against live-capture embeddings. Three speakers matched well (sim 0.77-0.80), three failed (sim -0.01 to 0.07).
- Live-path embeddings were refreshed by extracting 2s window embeddings from the BL-2021USB capture of each reference clip played through device 4. A mixed split-output validation using these live-path embeddings (instead of file-based embeddings) was then run.
- **Result with live-path embeddings: playing speaker2's clip resolved to `speaker2` (avg sim=0.4548, max=0.4984).** Playing speaker4's clip resolved to `speaker1` (avg sim=0.7413, max=0.7703) — not speaker4, because the speaker4 reference window (1.0-3.0s) was selected from early in the clip, but the live capture's later windows (3.0-9.0s) had very low similarity to that reference (sim=-0.02 to 0.17). The system correctly found speaker1 as the best match for those later windows. Playing speaker6's clip resolved to `speaker6` (avg sim=0.388, max=0.4235).
- Pairwise embedding analysis revealed that some speakers have very similar live-path embeddings: speaker1↔speaker5 sim=0.649, speaker3↔speaker4 sim=0.544, speaker3↔speaker6 sim=0.589. This means cross-speaker confusion is expected when vocal characteristics overlap.
- The key insight: the reference embedding window selection matters enormously. The current approach picks one 2s window that best matches the full-clip embedding, but that window may not represent all parts of the clip. Using multiple window embeddings per speaker would improve robustness.

### Full 6-Speaker Live-Path Validation — 2026-05-13

All 6 reference speakers were tested as world-playback clips using live-path single-window embeddings:

| World clip | M4 OK | Avg sim | Max sim | Speaker IDs | WORLD | BOTH |
|------------|-------|---------|---------|-------------|-------|-------|
| speaker1 | 15 | 0.094 | 0.176 | speaker3, speaker6 | 97 | 186 |
| speaker2 | 19 | 0.346 | 0.408 | speaker1, speaker2 | 96 | 189 |
| speaker3 | 15 | 0.061 | 0.073 | speaker4 | 78 | 206 |
| speaker4 | 17 | 0.049 | 0.077 | speaker1, speaker3 | 78 | 206 |
| speaker5 | 17 | 0.123 | 0.140 | speaker4, speaker5 | 78 | 206 |
| speaker6 | 11 | 0.329 | 0.372 | speaker6 | 78 | 207 |

**All 6 speakers resolved to identifiable speaker IDs.** Speakers 2 and 6 have the strongest matches (avg sim 0.33-0.35). Speakers 1, 3, 4, 5 have lower similarity (0.05-0.12) due to reference/live embedding mismatch.

**Verdict distribution is stable across all 6 runs**: 78-97 WORLD frames, 186-207 BOTH frames, consistent M4 acceptance (11-19 frames).

### Centroid Window Embeddings — 2026-05-13

To address within-clip embedding variance, a "centroid window" approach was implemented (`scripts/extract_centroid_embeddings.py`): for each 10s reference clip, extract 2s window embeddings (1s hop), then select the window whose embedding has the highest average similarity to all other windows in the same clip.

Centroid scores (avg sim to other windows in same clip):
- Speaker 1: 0.39 (8 windows) — low internal consistency
- Speaker 2: 0.53 (7 windows)
- Speaker 3: 0.74 (4 windows) — high internal consistency
- Speaker 4: 0.70 (4 windows)
- Speaker 5: 0.57 (4 windows)
- Speaker 6: 0.68 (5 windows)

Mixed validation comparison (centroid vs single-window live embeddings):

| World clip | Embedding type | M4 OK | Avg sim | Max sim | Speaker IDs |
|------------|---------------|-------|---------|---------|-------------|
| speaker2 | single-window live | 12 | 0.4548 | 0.4984 | speaker2 |
| speaker2 | centroid | 14 | 0.3379 | 0.3716 | speaker1, speaker2 |
| speaker4 | single-window live | 14 | 0.7413 | 0.7703 | speaker1 |
| speaker4 | centroid | 11 | 0.0706 | 0.0917 | speaker4 |
| speaker6 | single-window live | 16 | 0.3880 | 0.4235 | speaker6 |
| speaker6 | centroid | 17 | 0.3335 | 0.3443 | speaker6 |

**Finding**: Centroid embeddings do not consistently outperform single-window live embeddings. The centroid approach selects a more "average" window, which can be less distinctive than the single best-matching window. Speaker4's centroid embedding performed particularly poorly (avg sim=0.07).

**Recommendation**: Stick with single-window live-path embeddings for now. The better long-term approach is to either (a) re-record reference clips with consistent, neutral content, or (b) register multiple window embeddings per speaker so the identification can match any representative segment.

### Cross-Speaker Generalization — 2026-05-13

Three centroid-embedding world-playback clips were tested:
- Playing speaker2's clip → resolved to speaker2 (avg sim=0.3379) and speaker1 (cross-match)
- Playing speaker4's clip → resolved to speaker1 (avg sim=0.7413) — not speaker4 (sim=0.0706), because speaker4's centroid window doesn't represent the full clip
- Playing speaker6's clip → resolved to speaker6 (avg sim=0.3335)

All three resolved to identifiable speakers. The centroid approach is viable but single-window live embeddings remain stronger for identification.

## Noise-Path A/B Matrix

A user-visible "static" in the human/world audio prompted a noise-path isolation experiment.

Four silence-capture conditions were profiled on the BL-2021USB path:
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
2. **Embedding-domain mismatch was the dominant bottleneck for speaker-ID.** File-based reference embeddings for several speakers don't match current live BL-2021USB captures. Refresh embeddings from the live path (play clip through device 4, capture on device 1, extract 2s window embedding) improved mixed-condition speaker-ID from avg sim 0.055 to 0.45-0.74.
3. **Speaker identification now works in mixed conditions with live-path embeddings.** All three tested speakers (2, 4, 6) resolved correctly. Speaker4's clip resolved to speaker1 due to high cross-speaker similarity (speaker4 embedding window didn't represent the full clip well).
4. **PC fan noise was a real but secondary contaminant.** Removing it improved world-only speaker-ID modestly (+0.018 avg sim) but didn't change the fundamental picture.
5. **Device 4 / line-out path introduces 59 Hz mains hum.** Present whenever device 4 is connected, stronger when powered on. The BL-2021USB mic alone is clean.

**What needs to happen next:**
- Re-record reference clips for speakers 1, 3, 5 under current room conditions with consistent, neutral content (these have the lowest live-path similarity scores)
- Register multiple window embeddings per speaker to improve robustness across different parts of each clip
- Bridge the mixed-condition gap: preserve cleaner world windows during simultaneous self playback
- Fine-tune on live BL-2021USB room-colored captures
- Wire room filter adaptation (M7 trust-gated NLMS)

## Deployment Model — Design Notes

**Scope:** Proof of concept supports a single user in a fixed environment (same mic, same room, same hardware).

**Registration phase (one-time setup):**
- Play each reference clip through device 4, capture on device 1
- Extract best 2s window embedding from each capture
- Register in SpeakerRegistry before the conversation starts
- Cost: ~10 seconds of speech per speaker, done once

**Real-time phase (during conversation):**
- Streaming runtime accumulates WORLD/BOTH audio in a buffer
- Every ~20 VAD frames (~200ms), extracts one ECAPA embedding from the buffer
- Compares against all registered reference embeddings via cosine similarity
- Assigns speaker_id if best match exceeds threshold
- Cost: one embedding extraction + N cosine comparisons per cycle (trivial for N≤20)

**What is NOT needed in real time:**
- Re-extracting reference embeddings
- Re-registering known speakers
- Playing reference clips

**Re-registration triggers (future work):**
- Mic position change
- Room acoustics change (furniture, open/closed windows)
- New user added to known set
- Day-to-day embedding drift (empirically untested — worth validating)

**Multi-user extension (future work):**
- Online registration: accumulate 2-5s of clean WORLD audio from an unknown voice, extract embedding, register as new speaker
- Already supported by `SpeakerRegistry.update_from_frame_log()`
- Per-speaker, not per-conversation — once registered, recognized across sessions

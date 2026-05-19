# Spec: Corollary vs. Standard VAD/AEC Baseline Comparison

**Date:** 2026-05-19
**Type:** Benchmark experiment specification
**Goal:** Produce empirical, side-by-side data showing what corollary-discharge architecture buys over a standard VAD+AEC stack.

---

## 1. Why This Exists

Grant reviewers and skeptical engineers will ask: "Why not just use WebRTC VAD + a standard AEC + a wake-word model?" This experiment produces a fair, apples-to-apples comparison on the same audio artifacts, so we can answer that with data rather than architecture diagrams.

## 2. Baselines Compared

| Stack | Components | What it represents |
|-------|-----------|-------------------|
| **A: Corollary (full)** | Forward model (room-colored FIR) + M4 speaker-embedding scorer + thalamic gating + AgentiveController | The hypothesis — physical self-prediction + speaker attribution drives better full-duplex behavior |
| **B: VAD-only** | WebRTC VAD (aggressiveness=1, frames=30) on the raw mic input, applied as a mute gate during agent playback | The cheapest common baseline — "just turn off the mic when I'm talking" |
| **C: VAD + adaptive-filter AEC** | Custom NLMS adaptive filter (256 taps, μ=0.1) applied before WebRTC VAD. Uses the known TTS dry signal as the reference channel. | A plausible open-source stack — what a careful hobbyist would build. NLMS is chosen because we already have an implementation in the cerebellum package, ensuring reproducibility without introducing a new dependency. |
| **D: VAD-only (conservative)** | WebRTC VAD (aggressiveness=3, high sensitivity) — mutes less aggressively, higher self-false-positive tolerance | The "never lose a word" baseline — maximal false positives for self, minimal human loss |

Stack B and D bracket the practical VAD space: aggressive vs. conservative. If Corollary beats both, it beats the whole range.

## 3. Test Artifacts (inputs)

We use existing captures from the repo — no new recording required. Different artifact families serve different purposes:

- **Live TTS-to-Mic captures** (`data/m3_streaming_repair_live_*`, `data/m6_introspective_*`): Primary overlap test bed. These contain actual agent playback captured on the room mic, with known TTS timing (we know exactly when the agent spoke). Ground truth for world speech (human interruptions) is reconstructed from ASR timestamps on the captured audio minus the known TTS regions.
- **Stress tests** (`data/stress_test/`): Noise and interference conditions. Captures with phone music, room noise, severe interference. Ground truth is known for the injected noise but human speech labels may be absent — these artifacts are used for **self-leak** and **noise rejection** metrics only.
- **M4 balanced captures** (`data/m4_balanced_current_room/`): Single-source VAD characterization. These are captures of different TTS voices individually or human negatives individually — not mixed. Used to calibrate per-stack VAD sensitivity thresholds, not for overlap metrics.

### Ground truth note
Not all captures have frame-level ground truth. The classification is:

| Artifact family | Self speech timing | World speech timing | Overlap timing |
|----------------|-------------------|--------------------|----------------|
| Live TTS-to-Mic | ✅ Known (TTS playback timestamps) | ⚠️ Reconstructed via ASR on non-TTS regions | ✅ Derived from the two above |
| Stress tests | ❌ Unknown | ❌ Unknown (interference, not speech) | ❌ N/A |
| M4 balanced | ✅ Known | ✅ Known (separate files) | ❌ No overlap |

The live TTS-to-Mic captures are the primary dataset. Stress tests measure self-leak only. M4 balanced data calibrates VAD thresholds.

## 4. Metrics

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Self-leak dB** | Residual self-voice RMS after suppression, measured on known self-regions | The raw "did I silence myself" question |
| **Human WER loss** | WER of human transcripts under each stack vs. a clean reference transcript. Reference = the original TTS text for self-speech; for human speech, we use the transcript from a clean, non-overlapping capture of the same utterance (or ASR on the post-suppression audio as a proxy, with the caveat noted below) | The real cost: how many human words get lost |
| **False-mute rate** | % of confirmed human speech frames incorrectly suppressed by the stack's mute gate | How often the system accidentally silences a person |
| **Self-attribution F1** | Frame-level SELF/WORLD classification accuracy (Corollary only; N/A for VAD baselines) | The internal mechanism's precision |
| **Overlap survival rate** | % of overlap frames where at least partial human speech is preserved (not fully muted) | The full-duplex test |
| **Wake-word-free recall** | See Section 5b for precise definition | Demonstrates "always listening, never self-triggered" advantage |
| **Processing latency** | End-to-end delay from input sample availability to output sample emission per stack | VAD-only: ~0ms; VAD+AEC: filter delay (~16ms for 256 taps at 16kHz); Corollary: callback + analysis queue delay (measured from M7 harness, currently ~200-500ms) |

### WER measurement caveat
WER on room-colored audio with self-leak will be noisy. We measure:
- For self-speech: ASR output vs. the known TTS text (deterministic reference).
- For human speech during overlap: ASR on suppressed audio vs. a clean reference transcript of the same utterance (when available). When no clean reference exists, we report ASR word count delta (expected words vs. recognized words) as a proxy, with the caveat that this conflates WER with word count loss.

## 5. Test Protocol

### 5a. Per-capture comparison
For each live TTS-to-Mic capture:
1. Feed the raw mic WAV into each stack (A, B, C, D).
2. Each stack produces an output WAV (suppressed/enhanced) and a mute mask (per-frame suppression decision).
3. Run Whisper base ASR on each output.
4. Compare ASR transcripts, mute masks, and per-stack metrics against ground truth labels.

### 5b. Wake-word simulation
**Precise definition:** We simulate a push-to-talk/wake-word system that fully mutes the mic during known agent playback windows. A human utterance is "caught" if **any portion** of the utterance (defined as a continuous stretch of human ASR-confirmed speech outside the agent's TTS windows) falls outside a mute window and is therefore transcribed. An utterance is "missed" if the entire utterance falls within agent playback and is fully muted.

**Recall = caught utterances / total human utterances.**

This is utterance-level recall, not frame-level, because wake-word systems operate on utterance boundaries, not frames.

### 5c. Latency measurement
For each stack, measure the processing pipeline delay:
- **VAD-only (B, D):** Single frame classification → mute decision. Effectively zero added latency beyond the VAD frame size (~30ms).
- **VAD+AEC (C):** NLMS filter adds processing delay equal to the filter length in samples (256 taps at 16kHz = 16ms block delay).
- **Corollary (A):** Measured from the M7 harness callback → analysis queue → gating decision. Currently ~200-500ms depending on configuration.

Report per-stack latency in the summary table so reviewers can see the tradeoff: Corollary buys quality but costs latency.

## 6. Expected Outputs

### 1. Summary table
```
Capture          | Stack | Self-leak dB | Human WER loss | False-mute % | Overlap survival % | Latency ms
-----------------+-------+--------------+----------------+--------------+-------------------+----------
m3_live_001      | A     | -28.3        | 2.1%           | 0            | 97.2              | 350
m3_live_001      | B     | -12.1        | 18.7%          | 15.3         | 68.4              | 30
m3_live_001      | C     | -18.5        | 8.4%           | 6.1          | 82.7              | 46
m3_live_001      | D     | -3.2         | 1.8%           | 0            | 98.1              | 30
```

### 2. Per-stack aggregate
| Stack | Avg self-leak dB | Avg human WER loss | Avg false-mute % | Avg latency ms | Notes |
|-------|-----------------|-------------------|------------------|---------------|-------|
| A: Corollary | TBD | TBD | TBD | TBD | Full pipeline |
| B: VAD-only (aggressive) | TBD | TBD | TBD | ~30 | Cheap but lossy |
| C: VAD+AEC (NLMS) | TBD | TBD | TBD | ~46 | Middle ground |
| D: VAD-only (conservative) | TBD | TBD | TBD | ~30 | Never mutes, self-leaks |

### 3. Plots
- Bar chart: human WER loss by stack (lower is better)
- Bar chart: self-leak dB by stack (more negative = less leak)
- Overlap survival rate by stack
- Latency vs. quality scatter: x-axis = avg WER loss, y-axis = latency ms

## 7. Script Structure

```
scripts/baseline_vad_comparison.py
├── load_capture(path) → raw audio waveform
├── load_ground_truth(capture_id) → self_regions, world_regions, overlap_regions (when available)
├── run_vad_stack(audio, config) → suppressed audio, mute mask
├── run_aec_vad_stack(audio, reference_tts, config) → NLMS-filtered + VAD, suppressed audio, mute mask
├── run_corollary_stack(audio, ground_truth) → suppressed audio, attribution mask
├── run_wake_word_simulation(audio, self_regions, human_regions) → utterance-level recall
├── measure_wer(output_wav, reference_text) → WER or word-count delta (when reference unavailable)
├── measure_self_leak(output_wav, self_regions) → dB
├── measure_false_mute(mute_mask, human_regions) → %
├── measure_overlap_survival(mute_mask, overlap_regions) → %
├── measure_latency(stack_config) → ms
└── produce_table(captures, stacks) → markdown + CSV + plots
```

## 8. Dependencies

- `webrtcvad` — already in the venv, used for Stacks B and D
- `numpy` — for NLMS filter implementation (Stack C)
- `faster-whisper` or `openai-whisper` — for ASR-based WER measurement
- `matplotlib` — for plots
- No new external dependencies beyond what's already in the corollary venv

## 9. What's Not in Scope

- Commercial AEC systems (NVIDIA Broadcast, Krisp) — no public API for fair comparison.
- End-to-end commercial wake-word detectors (Hey Siri, OK Google) — same reason.
- Subjective listening tests — this is an objective benchmark.

## 10. Success Criteria

Corollary wins the grant argument if:
- Self-leak is competitive with or better than AEC (Stack C), while VAD-only (Stacks B/D) either leaks badly or mutes too aggressively.
- Human WER loss is strictly better than VAD-only aggressive (Stack B), which is the most common cheap baseline.
- Overlap survival rate is >90% for Corollary, showing it doesn't just mute the whole channel.

If Corollary loses to a simple VAD on any single metric, we report the full tradeoff table honestly — the latency vs. quality scatter plot contextualizes that. That's publication-worthy either way.

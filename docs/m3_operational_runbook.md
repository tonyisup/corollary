# M3 Operational Runbook

> **Last updated:** 2026-05-09
> **Status:** M3 core behavior validated. This runbook documents how to reproduce and verify it.

## Purpose

This is the single reference for running, analyzing, and validating the M3 streaming turn-taking pipeline. It covers three modes:

1. **Deterministic acceptance** — no hardware, no saved artifacts required beyond the repo. This is the default regression gate.
2. **Offline replay** — deterministic replay of saved hardware captures through the streaming callback. Validates VAD/attribution/controller/log plumbing on real captured audio.
3. **Live hardware** — full live streaming with mic input, TTS output, and optional human stimulus on a separate output device. This is the promotion-grade validation, but requires specific hardware.

---

## Prerequisites

```bash
cd ~/Projects/corollary
source .venv/bin/activate
```

All commands below assume the venv is active and CWD is the repo root.

### Hardware assumptions (live mode only)

| Role | Device | Index |
|------|--------|-------|
| Mic input | fifine Microphone | 3 |
| Agent TTS output | Mac mini Speakers | 5 |
| Human stimulus output | External Headphones | 4 |

If your device indices differ, override with `--input-device`, `--output-device`, `--human-output-device`.

---

## 1. Deterministic Acceptance (regression gate)

This is the **default validation command**. It requires no hardware and no saved artifacts beyond what's tracked in the repo.

```bash
.venv/bin/python scripts/m3_streaming_acceptance.py \
  --run-dir data/m3_gated_playback_test \
  --output /tmp/m3_acceptance_verdict.json
```

**Exit codes:**
- `0` — all acceptance criteria met
- `1` — one or more criteria failed
- `2` — setup/input missing

**What it checks:**

| Check | Method | What it validates |
|-------|--------|-------------------|
| `synthetic_pre_agent_defer` | Synthetic WORLD frames before agent request | TTS deferred, position held at 0, later resumes and completes |
| `synthetic_overlap_interrupt_resume` | Synthetic BOTH frames during agent SPEAK | Interrupt fires, TTS position held during LISTEN, repair cue emitted exactly once, repair audio plays after LISTEN, original TTS resumes and completes |
| `offline_audio_replay_smoke` | Replays all 5 saved hardware captures from `data/m3_gated_playback_test` | VAD/attribution/controller/log plumbing works on real captured audio |

**Interpretation:**
- Synthetic checks are the **behavior gate** — they validate online controller/TTS-gating logic directly.
- Offline replay is a **smoke test** — it validates callback plumbing on real audio but is intentionally not used as the overlap behavior gate (room-colored captures can under-match dry TTS templates in local xcorr).

**Expected verdict excerpt:**
```json
{
  "pass": true,
  "acceptance_criteria": {
    "synthetic_pre_agent_defer_passed": true,
    "synthetic_overlap_interrupt_resume_passed": true,
    "synthetic_collision_repair_passed": true,
    "synthetic_repair_audio_passed": true,
    "offline_audio_replay_smoke_passed": true
  }
}
```

**When to run:** after any change to `packages/pfc/streaming_runtime.py`, `packages/pfc/agentive_controller.py`, `packages/thalamus/attributor.py`, or `packages/cochlea/vad.py`.

---

## 2. Offline Replay (saved hardware captures)

Replay a specific scenario's captured audio through the streaming pipeline. No hardware needed.

```bash
# Replay a specific scenario (use explicit paths)
.venv/bin/python scripts/m3_streaming_demo.py \
  --mode offline \
  --input-audio data/m3_gated_playback_test/human_early/live_captured.npy \
  --agent-pcm data/m3_gated_playback_test/human_early/agent_pcm.npy \
  --gated-playback \
  --m4-enable \
  --output /tmp/streaming_replay_log.jsonl \
  --summary /tmp/streaming_replay_summary.json
```

**Scenarios:** Replace `human_early` with `human_late`, `human_before`, `human_after`, or `overlap_self_first` in both paths.

**Custom audio files:**
```bash
.venv/bin/python scripts/m3_streaming_demo.py \
  --mode offline \
  --input-audio /path/to/live_captured.npy \
  --agent-pcm /path/to/agent_pcm.npy \
  --gated-playback \
  --output /tmp/streaming_replay_log.jsonl
```

**Note:** The `--scenario` flag references `data/m3_interrupt_v3-mic_adjusted/` which is not tracked. Use explicit `--input-audio` and `--agent-pcm` paths pointing to `data/m3_gated_playback_test/<scenario>/`.

### Post-hoc analysis

```bash
# Summary view
.venv/bin/python scripts/m3_streaming_analyze.py /tmp/streaming_replay_log.jsonl

# Full analysis JSON
.venv/bin/python scripts/m3_streaming_analyze.py /tmp/streaming_replay_log.jsonl \
  --full \
  --output /tmp/streaming_analysis.json
```

**Key fields in analysis output:**
- `tts_completed: true` — original utterance was not lost
- `interrupt_events` — number of SPEAK→LISTEN transitions
- `resumed_events` — number of LISTEN→SPEAK resumptions
- `state_transitions` — controller state machine trace
- `verdict_counts` — SELF/WORLD/BOTH/SILENCE distribution

---

## 3. Gated Playback Acceptance (saved artifacts)

Re-run the gated playback analyzer on a saved hardware run directory. No hardware needed.

```bash
.venv/bin/python scripts/m3_gated_playback_acceptance.py \
  --run-dir data/m3_gated_playback_test \
  --output /tmp/m3_gated_acceptance_verdict.json
```

**Exit codes:**
- `0` — all suppression checks passed
- `1` — one or more checks failed

**What it validates:** physical TTS segment scheduling avoided overlap with human windows. Uses `segment_overlap_passed`, `defer_passed`, and `all_suppression_passed` analyzer fields — **not** legacy reflex MI/TI metrics.

**Important:** In gated playback mode, legacy reflex metrics (MI, TI, resume) are diagnostic only. The analyzer fields are the primary pass criteria. See `docs/m3_gated_playback_validation_20260508.md`.

---

## 4. Live Hardware (promotion-grade validation)

> **Only run this when the hardware is physically set up and you're ready to validate a new behavior or regression.**

### Standard live run (no human stimulus)

```bash
.venv/bin/python scripts/m3_streaming_demo.py \
  --mode live \
  --input-device 3 \
  --output-device 5 \
  --duration 10.0 \
  --gated-playback \
  --m4-enable \
  --agent-text "I was speaking, then paused, and then I continued my thought." \
  --output /tmp/streaming_live_log.jsonl \
  --summary /tmp/streaming_live_summary.json
```

### Live run with human stimulus (repair-audio validation)

```bash
.venv/bin/python scripts/m3_streaming_demo.py \
  --mode live \
  --input-device 3 \
  --output-device 5 \
  --human-output-device 4 \
  --play-human-stimulus \
  --human-delay-s 0.45 \
  --duration 12.0 \
  --gated-playback \
  --m4-enable \
  --agent-text "I was speaking, then paused, and then I continued my thought." \
  --repair-text "Sorry, go ahead." \
  --human-text "Excuse me, I need to say something here." \
  --output data/m3_streaming_repair_live_$(date +%Y%m%d_%H%M%S)/streaming_log.jsonl \
  --summary data/m3_streaming_repair_live_$(date +%Y%m%d_%H%M%S)/summary.json
```

### Live run validation criteria

After a live run with human stimulus, check the summary JSON for:

| Criterion | Expected | Meaning |
|-----------|----------|---------|
| `interrupt_events` | ≥1 | Controller detected overlap and interrupted |
| `collision_repair_events` | 1 | Repair cue emitted exactly once |
| `repair_played_frames` | >0 | Repair audio was emitted |
| `repair_audio_completed` | true | Repair audio played to completion |
| `resumed_events` | ≥1 | Original utterance resumed after repair |
| `tts_completed` | true | Original utterance played to completion |
| `final_controller_state` | `IDLE` | Controller returned to idle |

### Validated live artifact

The last validated live run is tracked at:
```
data/m3_streaming_repair_live_20260509_072237/
  streaming_log.jsonl
  summary.json
  hardware_repair_validation.json  ← pass=true, all 9 criteria met
```

---

## 5. Full Test Suite

```bash
.venv/bin/python -m pytest -q
```

**Expected:** 238 passed, 1 skipped, 3 warnings (deprecation noise from webrtcvad/SWIG).

---

## Decision Matrix

| Situation | Command | Hardware? |
|-----------|---------|-----------|
| After code changes to streaming/controller/attributor/VAD | Deterministic acceptance (§1) | No |
| After changes to audio capture or device config | Offline replay (§2) | No |
| After changes to gated playback segment logic | Gated playback acceptance (§3) | No |
| New hardware, new room, or promotion validation | Live hardware (§4) | Yes |
| Before any commit that touches M3 code | Full test suite (§5) | No |

---

## Key Architecture Notes

- **No synthesis inside the callback.** All TTS/repair/human audio is pre-rendered before opening the realtime stream. This is an intentional boundary: live synthesis in the callback would add blocking/network risk.
- **Repair audio is a separate queue.** It plays after LISTEN releases, before the deferred original utterance resumes. The original TTS position is held during repair playback.
- **Synthetic frames are the behavior gate.** Offline replay validates plumbing; synthetic injection validates logic.
- **Gated playback metrics differ from reflex metrics.** In `--gated-playback` mode, use analyzer fields (`segment_overlap_passed`, `defer_passed`, `all_suppression_passed`), not legacy MI/TI/resume counters.

---

## References

- `docs/m3_status_20260508.md` — full M3 status and commit history
- `docs/m3_collision_repair_20260508.md` — collision repair policy and repair-audio design
- `docs/m3_streaming_repair_hardware_validation_20260509.md` — live hardware validation details
- `docs/m3_gated_playback_validation_20260508.md` — gated playback metric interpretation
- `corollary-audio-eval-workflows` skill — detailed pitfalls and calibration procedures

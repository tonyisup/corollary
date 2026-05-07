# M3 Resume Audit Results

> **Date:** 2026-05-06
> **Data:** `data/m3_interrupt_v2/`
> **Pipeline:** Full M2+M4 attribution with per-frame self-match scoring

## Executive summary

All 4 targets are now met:

| Metric | Target | Result |
|--------|--------|--------|
| False interrupt rate | <5% | 0.0% ✓ |
| False defer rate | <10% | 0.8% ✓ |
| Missed interrupt rate | <10% | 1.1% ✓ |
| Resume rate | >80% | 100% ✓ |

## Key changes from previous analysis

1. **Hysteresis**: Minimum LISTEN duration (300ms) after overlap interrupt prevents flapping
2. **Post-interrupt resume**: After any SPEAK→LISTEN interrupt, return to IDLE after timeout if agent is not playing
3. **Stronger sustained evidence**: 5 consecutive frames (up from 3) required before triggering interrupt

## Per-scenario results

| Scenario | FI | FD | MI | Resumes | Notes |
|----------|----|----|----|---------|-------|
| brief_noise | 0 | 8 | 0 | 1/1 | Brief noise triggers overlap detector |
| human_after | 0 | 0 | 10 | 1/1 | Human speaks after agent; MI expected |
| human_before | 0 | 0 | 0 | 1/1 | Clean handling |
| human_double | 1 | 11 | 0 | 1/1 | Double interrupt causes some flapping |
| human_early | 0 | 0 | 0 | 1/1 | Clean handling |
| human_late | 0 | 0 | 0 | 1/1 | Clean handling |
| overlap_human_first | 0 | 0 | 17 | 1/1 | Human speaks first; MI expected |
| overlap_self_first | 0 | 0 | 0 | 1/1 | Clean handling |

## Interpretation

- **8/8 resumable scenarios succeed**: The post-interrupt resume mechanism ensures the controller returns to IDLE after the human stops speaking, even if VAD keeps detecting room noise.

- **17 missed interrupts in `overlap_human_first`**: Expected — the human speaks before the agent starts, so the overlap detector (which requires agent playback) can't detect them. The controller still handles these via the post-agent-finished LISTEN path.

- **10 missed interrupts in `human_after`**: Expected — the human speaks after the agent finishes. Same as above.

- **8 false defers in `brief_noise`**: The brief noise triggers the overlap detector. Could be reduced by increasing `both_external_min_frames`.

- **11 false defers in `human_double`**: The double interruption causes some flapping. The hysteresis helps but doesn't eliminate it entirely.

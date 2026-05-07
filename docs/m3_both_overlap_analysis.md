# M3 BOTH-Overlap Discrimination Analysis

> **Date:** 2026-05-06
> **Status:** Per-frame scoring implemented, hysteresis + resume added

## Problem

When the agent is speaking (`is_in_motor_window=True`) and the attributor classifies a frame as `BOTH`, the controller needs to distinguish:
1. **Self-dominant BOTH** (room reflections, TTS bleed) → sustain SPEAK
2. **Human-overlap BOTH** (human speaking over agent) → yield (SPEAK→LISTEN)

## Architecture

### Per-frame self-match scoring (IMPLEMENTED)

The attributor now computes a per-frame energy ratio (observed / expected self RMS) by comparing each frame's audio against the corresponding slice of the agent PCM template.

**Files changed:**
- `AttributionResult`: added `self_match_score_per_frame` field
- `ThalamicAttributor.attribute()`: added `self_pcm` and `agent_start_sample` params
- `ThalamicAttributor._compute_per_frame_energy_ratio()`: new method
- `AgentiveController`: uses per-frame energy ratio for overlap detection

### Overlap detection policy

The controller uses a multi-signal approach:

1. **Energy ratio threshold**: `both_external_energy_ratio_threshold` (default 2.0)
2. **Sustained evidence**: `both_external_min_frames` (default 5) consecutive frames
3. **Hysteresis**: `both_external_listen_min_ms` (default 300ms) minimum LISTEN duration after interrupt
4. **Post-interrupt resume**: After any SPEAK→LISTEN interrupt, return to IDLE after 2x min_listen_duration if agent is not playing

### Signals evaluated

| Signal | Discriminates? | Notes |
|--------|---------------|-------|
| Match score (xcorr) | No | Constant within scenario |
| Excess RMS | Partial | Works when human louder, fails when quieter |
| Absolute RMS threshold | No | TTS variability (std ~10 dB) |
| M4 below-threshold | No | Fires too often during self-playback |
| Per-frame energy ratio | Yes | Best signal; requires self_pcm from attributor |
| Combined (sustained + ratio) | Yes | Current approach |

## Results

### Offline analysis (data/m3_interrupt_v2/)

| Metric | Target | Result |
|--------|--------|--------|
| False interrupt rate | <5% | 0.0% ✓ |
| False defer rate | <10% | 0.8% ✓ |
| Missed interrupt rate | <10% | 1.1% ✓ |
| Resume rate | >80% | 100% ✓ |

### Per-scenario breakdown

| Scenario | FI | FD | MI | Resumes |
|----------|----|----|----|---------|
| brief_noise | 0 | 8 | 0 | 1/1 |
| human_after | 0 | 0 | 10 | 1/1 |
| human_before | 0 | 0 | 0 | 1/1 |
| human_double | 1 | 11 | 0 | 1/1 |
| human_early | 0 | 0 | 0 | 1/1 |
| human_late | 0 | 0 | 0 | 1/1 |
| overlap_human_first | 0 | 0 | 17 | 1/1 |
| overlap_self_first | 0 | 0 | 0 | 1/1 |

### Known limitations

1. **brief_noise (8 FD)**: Brief noise triggers overlap detector. Could be reduced by increasing `both_external_min_frames` or `both_external_energy_ratio_threshold`.

2. **human_double (1 FI, 11 FD)**: Double interruption causes some flapping. The hysteresis helps but doesn't eliminate it entirely.

3. **human_after (10 MI)**: Human speaks after agent finishes. Overlap detector only fires during agent playback, so these are detected as "missed interrupts" (the controller enters LISTEN when agent finishes, not from overlap detection). The post-interrupt resume still handles them correctly.

4. **overlap_human_first (17 MI)**: Human speaks first, before agent starts. Same as above — the overlap detector can't detect human speech that starts before the agent.

## Controller parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `both_external_energy_ratio_threshold` | 2.0 | Energy ratio above which BOTH is classified as external speech |
| `both_external_min_frames` | 5 | Consecutive frames required before triggering interrupt |
| `both_external_listen_min_ms` | 300 | Minimum LISTEN duration after overlap interrupt (hysteresis) |

## Remaining work

1. **Tune `brief_noise` false defers**: Increase `both_external_min_frames` to 8-10, or add a "brief noise" filter that ignores short energy spikes.

2. **Handle pre-agent human speech**: The overlap detector only works during agent playback. For scenarios where the human speaks first, a different approach is needed (e.g., detect human speech before agent starts and defer initial output).

3. **Live testing**: All results are from offline analysis. Live testing with real human speech is needed to validate the approach.

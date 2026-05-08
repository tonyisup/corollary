# M3 Gated Playback Validation — 2026-05-08

## Context

Controller-gated physical playback was added after the project audit found that the live harness could score controller decisions while still playing TTS unconditionally. That made reflex decisions valid, but did not prove physical suppression/defer.

The new mode uses `--gated-playback` to render physical TTS output segments that avoid scheduled human speech windows.

## Commands

Smoke validation:

```bash
.venv/bin/python scripts/m3_live_e2e_harness.py \
  --scenario human_before \
  --gated-playback \
  --calibrate \
  --input-device 3 \
  --output-device 5 \
  --human-output-device 4 \
  --room-filter data/m3_broca_room_filter_production.npy \
  --m4-enable \
  --save-dir data/m3_gated_smoke_human_before_v2
```

Full gated validation:

```bash
.venv/bin/python scripts/m3_live_e2e_harness.py \
  --all \
  --gated-playback \
  --calibrate \
  --input-device 3 \
  --output-device 5 \
  --human-output-device 4 \
  --room-filter data/m3_broca_room_filter_production.npy \
  --m4-enable \
  --save-dir data/m3_gated_playback_test
```

Analyze:

```bash
.venv/bin/python scripts/m3_gated_playback_analyze.py \
  data/m3_gated_playback_test \
  --output data/m3_gated_playback_test/gated_analysis.json
```

## Implementation Details

Key helpers in `scripts/m3_live_e2e_harness.py`:

- `plan_controller_gated_tts_segments()` — converts scenario timing into physical TTS output segments.
- `build_controller_gated_tts_buffer()` — renders segments into the actual output buffer.
- `controller_request_time_s()` — keeps controller request timing separate from physical playback timing.
- `annotate_metric_mode()` — labels metrics as either reflex interrupt/resume mode or gated suppression mode.

Analyzer:

- `scripts/m3_gated_playback_analyze.py`

Tests:

- `tests/unit/test_m3_gated_playback.py`
- `tests/unit/test_m3_gated_playback_analyze.py`
- `tests/unit/test_m3_live_e2e_metrics.py`

## Bug Found During Hardware Smoke

First gated `human_before` smoke physically delayed TTS correctly but reported `false_defers=1`.

Root cause:

- Gated mode changed `agent_start_s` to the delayed physical TTS segment start.
- Controller scoring then used this delayed physical start as the output request time.
- For defer behavior, the controller request should occur at the original agent intent time, not when delayed audio finally plays.

Fix:

- Preserve `original_agent_start_s` separately.
- Use delayed `agent_start_s` only for attribution/motor-window timing.
- Use `controller_request_time_s(expected_behavior, original_agent_start_s)` for controller scoring.

Smoke result after fix:

- Run: `data/m3_gated_smoke_human_before_v2`
- `false_defers=0`
- TTS segment starts at `3.800s`, after human window ends at `3.630s`.

## Full Hardware Result

Full gated hardware run:

- Run: `data/m3_gated_playback_test`
- Analyzer: `data/m3_gated_playback_test/gated_analysis.json`
- `all_suppression_passed=true`
- Full test suite after changes: `217 passed, 1 skipped`

Scenario summary:

| Scenario | Gated Result | Evidence |
|---|---:|---|
| human_before | pass | `defer_passed=true`; TTS starts after human window |
| human_after | pass | normal single TTS segment; no gated overlap expected |
| human_early | pass | two TTS segments; `segment_overlap_passed=true` |
| human_late | pass | two TTS segments; `segment_overlap_passed=true` |
| overlap_self_first | pass | two TTS segments; `segment_overlap_passed=true` |

## Metrics Interpretation

Two different validation modes now exist.

### 1. Reflex interrupt/resume mode

TTS is allowed to overlap with human speech.

Primary metrics:

- `true_interrupts`
- `missed_interrupts`
- `resume_rate`
- `false_interrupts`
- `false_defers`

A successful reflex run should show true interrupts and resumes on overlap scenarios.

### 2. Gated suppression mode

Physical TTS is delayed/split around known scheduled human windows. Because overlap is prevented, old reflex metrics can show `MI=3`, `TI=0`, and `resume=0/3` on overlap scenarios even when gated playback is working.

Primary metrics:

- `gated_playback=true`
- `tts_segments` present
- `segment_overlap_passed=true` for overlap scenarios
- `defer_passed=true` for `human_before`
- `all_suppression_passed=true`

Legacy interrupt/resume metrics are diagnostic in this mode, not pass/fail criteria.

## Remaining Boundary

This is scheduled harness gating, not true production streaming gating. Production gating would compute attribution/controller decisions in the live callback and gate TTS dynamically based on online evidence rather than known scenario windows.

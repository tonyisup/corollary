# M3 Collision Repair — Policy Output

> Date: 2026-05-08
> Status: Implemented as deterministic controller policy/logging; synthesized repair audio remains a future integration.

## Problem

M3 already interrupts and later resumes when a human overlaps with agent speech. The missing social layer was an explicit repair act: when the agent realizes it collided with the human, it should surface a cue such as "Sorry, go ahead." instead of only muting itself.

## Implemented behavior

`AgentiveController` now includes `collision_repair_text` in `ControllerDecision`.

On a real SPEAK → LISTEN interruption:

1. `should_interrupt=True`
2. the active utterance is moved to `deferred_utterance`
3. controller state becomes `LISTEN`
4. `collision_repair_text` is emitted exactly once for that interruption
5. after the human finishes/cooldown expires, the original utterance can resume normally

Default repair text:

```text
Sorry, go ahead.
```

The text is configurable via `AgentiveController(collision_repair_text=...)` and can be disabled with `collision_repair_text=None`.

## Streaming integration

`StreamingConfig` exposes the same `collision_repair_text` field.

`StreamingRuntime` carries the repair cue into each `StreamingFrameLog` and JSONL output as:

```json
{"collision_repair_text": "Sorry, go ahead."}
```

`get_summary()` reports `collision_repair_events`.

## Acceptance criteria

The deterministic streaming acceptance gate now requires the synthetic overlap-interrupt scenario to show:

- at least one interrupt event
- exactly one collision repair event
- repair text equals `Sorry, go ahead.`
- TTS position is held during LISTEN
- interrupted TTS resumes and completes

Command:

```bash
.venv/bin/python scripts/m3_streaming_acceptance.py --run-dir data/m3_gated_playback_test
```

## Evidence

Targeted tests:

```bash
.venv/bin/python -m pytest tests/unit/test_pfc_m2_turn_bridge.py tests/unit/test_streaming_runtime.py -q
```

Result after implementation: 25 passed.

The acceptance script now includes:

```json
"synthetic_collision_repair_passed": true
```

## Important boundary

This commit does not synthesize or play a separate repair audio clip. It exposes the repair act as structured controller policy output. That is the right boundary for this pass because:

- the repo already has validated TTS gating/resume semantics
- repair phrasing should be orchestration-visible before being attached to a second audio path
- adding synthesized repair audio would require a priority/queue policy so the repair cue does not consume or corrupt the deferred original utterance

## Next integration step

Add a repair-audio path that can optionally render the repair cue immediately after interruption, then return to LISTEN without advancing the deferred original utterance. Acceptance should verify:

1. repair clip is emitted once per collision
2. original utterance remains deferred while repair plays
3. repair playback is itself suppressible if the human is still speaking
4. original utterance resumes only after human cooldown

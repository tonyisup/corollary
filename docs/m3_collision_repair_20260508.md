# M3 Collision Repair — Policy and Repair Audio

> Date: 2026-05-08
> Status: Implemented as deterministic controller policy output plus optional suppressible repair-audio playback in the streaming runtime. Live hardware validation passed in `data/m3_streaming_repair_live_20260509_072237`.

## Problem

M3 already interrupts and later resumes when a human overlaps with agent speech. The missing social layer was an explicit repair act: when the agent realizes it collided with the human, it should surface a cue such as "Sorry, go ahead." instead of only muting itself.

## Controller behavior

`AgentiveController` includes `collision_repair_text` in `ControllerDecision`.

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

## Streaming repair-audio behavior

`StreamingConfig` exposes:

- `collision_repair_text`: the repair cue text/policy payload
- `repair_pcm`: optional pre-synthesized repair audio for that cue

`StreamingRuntime` treats repair audio as a separate output source from the original utterance:

1. Collision interrupt queues repair audio if `repair_pcm` is present.
2. While the controller is in `LISTEN`, the repair queue is suppressed and does not advance.
3. When the controller leaves `LISTEN`, repair audio plays before the deferred original utterance resumes.
4. The original TTS position is held during repair playback.
5. Once repair audio completes, the deferred original utterance resumes through the normal controller path.

This separation is intentional. Calling `request_output()` for the repair cue would clear/overwrite the deferred original utterance; the runtime-level repair queue avoids that corruption.

## Streaming log schema

`StreamingFrameLog` and JSONL now include:

```json
{
  "collision_repair_text": "Sorry, go ahead.",
  "output_source": "repair",
  "repair_position_s": 0.03,
  "repair_total_duration_s": 0.12,
  "repair_queued": false,
  "repair_playing": true,
  "repair_completed": false
}
```

`get_summary()` reports:

- `collision_repair_events`
- `repair_played_frames`
- `repair_total_duration_s`
- `repair_audio_completed`

## Acceptance criteria

The deterministic streaming acceptance gate now requires the synthetic overlap-interrupt scenario to show:

- at least one interrupt event
- exactly one collision repair event
- repair text equals `Sorry, go ahead.`
- TTS position is held during LISTEN
- repair audio plays after LISTEN, not during LISTEN
- original TTS position remains held while repair audio plays
- original TTS resumes after repair audio completes
- interrupted TTS eventually completes

Command:

```bash
.venv/bin/python scripts/m3_streaming_acceptance.py --run-dir data/m3_gated_playback_test
```

The acceptance script includes:

```json
{
  "synthetic_collision_repair_passed": true,
  "synthetic_repair_audio_passed": true
}
```

## Evidence

Deterministic streaming tests:

```bash
.venv/bin/python -m pytest tests/unit/test_streaming_runtime.py -q
```

Result after repair-audio implementation: 17 passed.

Deterministic acceptance:

```json
{
  "synthetic_collision_repair_passed": true,
  "synthetic_repair_audio_passed": true
}
```

Live hardware validation:

```text
data/m3_streaming_repair_live_20260509_072237/hardware_repair_validation.json
pass=true
interrupt_once=true
collision_repair_once=true
repair_audio_completed=true
repair_audio_played=true
repair_not_during_listen=true
original_tts_held_during_repair=true
tts_resumed_after_repair=true
tts_completed=true
final_idle=true
```

## Important boundary

This implementation supports a pre-synthesized repair clip (`repair_pcm`) and proves suppressible playback/queue semantics. It does not yet call Broca/edge-tts live from inside the streaming callback. That is the right boundary because live synthesis in the callback path would add blocking/network risk and should be handled by an upstream orchestration layer that pre-renders repair assets.

## Next optional integration step

Wire `scripts/m3_streaming_demo.py` or the upstream orchestrator to pre-synthesize `collision_repair_text` via Broca and pass the resulting PCM into `StreamingConfig.repair_pcm` for live hardware validation.

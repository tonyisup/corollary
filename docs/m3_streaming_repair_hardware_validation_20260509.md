# M3 Streaming Repair Audio — Live Hardware Validation

> Date: 2026-05-09
> Status: Passed
> Run: `data/m3_streaming_repair_live_20260509_072237`

## Goal

Validate the production streaming callback path with real hardware after adding suppressible repair-audio queue semantics.

This is stronger than deterministic synthetic acceptance because it exercises:

- live `sounddevice.Stream` callback
- mic input from device 3
- agent TTS output through device 5
- delayed human stimulus through device 4
- pre-rendered Broca repair cue passed as `StreamingConfig.repair_pcm`
- online VAD → attribution → controller → output gate loop

## Command

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
  --output data/m3_streaming_repair_live_20260509_072237/streaming_log.jsonl \
  --summary data/m3_streaming_repair_live_20260509_072237/summary.json
```

## Code fixes made before passing run

1. `scripts/m3_streaming_demo.py` now pre-renders live-mode agent, repair, and optional human stimulus audio before opening the realtime stream. Previously live mode set `agent_pcm=None`, so no TTS could play.
2. `StreamingRuntime.run()` now uses `time.sleep(duration_s)` inside the stream context instead of `asyncio.get_event_loop().run_until_complete(...)`, which fails on Python 3.14 when no default event loop exists.
3. `AgentiveController` now preserves a freshly resumed SPEAK state during initial motor holdoff, so post-repair SILENCE cannot immediately collapse SPEAK→IDLE before the first resumed TTS chunk emits.

## Summary

```json
{
  "total_frames": 360,
  "speech_frames": 168,
  "tts_played_frames": 216,
  "repair_played_frames": 74,
  "tts_total_duration_s": 4.2535,
  "repair_total_duration_s": 2.198375,
  "deferred_events": 0,
  "resumed_events": 1,
  "interrupt_events": 1,
  "collision_repair_events": 1,
  "repair_audio_completed": true,
  "final_controller_state": "IDLE",
  "tts_completed": true,
  "gated_playback": true,
  "m4_enabled": true
}
```

## Acceptance criteria

Saved validation: `data/m3_streaming_repair_live_20260509_072237/hardware_repair_validation.json`

```json
{
  "pass": true,
  "criteria": {
    "interrupt_once": true,
    "collision_repair_once": true,
    "repair_audio_completed": true,
    "repair_audio_played": true,
    "repair_not_during_listen": true,
    "original_tts_held_during_repair": true,
    "tts_resumed_after_repair": true,
    "tts_completed": true,
    "final_idle": true
  }
}
```

## Timing

```json
{
  "interrupt_s": [2.158359],
  "repair_text_s": [2.158359],
  "listen_range_s": [2.158359, 5.388309],
  "repair_audio_range_s": [2.788211, 6.44833],
  "resume_s": [6.498264],
  "tts_after_repair_first_s": 6.498264
}
```

## Interpretation

Live streaming M3 now demonstrates the full embodied repair loop:

1. Agent begins speaking.
2. Human stimulus overlaps through separate output hardware.
3. Controller interrupts once.
4. Collision repair text fires once.
5. Repair audio plays from its separate queue.
6. Original utterance remains held during repair audio.
7. Original utterance resumes after repair audio completes.
8. Original utterance completes and controller returns to IDLE.

This closes the live hardware validation gap for repair-audio queue semantics. Remaining work is productization/cleanup rather than core M3 behavior validation.

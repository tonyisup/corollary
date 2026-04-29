# M3 Start — Agentive Listening and Proto-Turn-Taking

> Status: first M2→M3 integration slice  
> Date: 2026-04-29

## Purpose

M3 begins once M2 can provide a conservative enough auditory gate. M3 does not
require perfect speech enhancement; it needs a reliable signal for when the
agent should defer speaking because external/overlapping speech is active.

## First M3 rule

M2 attribution feeds the PFC `AgentiveController`:

| M2 verdict | Speech? | M3 interpretation |
| --- | --- | --- |
| `WORLD` | yes | external agentive speech; listen/defer |
| `BOTH` | yes | overlapping speech; preserve and listen/defer |
| `SELF` | no/yes | self activity; do not treat as external turn |
| `SILENCE` | no | cooldown toward idle; resume if deferred |

The important bridge decision is that speech-like `BOTH` frames are treated as
agentive external speech for turn-taking. This is conservative: when another
voice may be present, the agent should avoid speaking over it.

## Implementation

- `packages/pfc/agentive_controller.py`
  - new `treat_both_speech_as_agentive` option, default `True`
- `packages/pfc/m2_turn_bridge.py`
  - timestamped M2 frame bridge into the controller
- `scripts/m3_turn_taking_demo.py`
  - synthetic demo: overlapping speech causes deferral; silence/cooldown resumes
- `tests/unit/test_pfc_m2_turn_bridge.py`
  - validates BOTH-speech deferral and resume behavior

## Demo

```bash
./.venv/bin/python scripts/m3_turn_taking_demo.py
```

Expected behavior:

1. M2 emits speech-like `BOTH` frames.
2. PFC transitions into `LISTEN` after the listen threshold.
3. Requested output is deferred.
4. Silence/cooldown returns PFC to `IDLE`.
5. Deferred output can resume.

## Current boundary

This is an early controller integration, not a full social model. It validates
turn-taking mechanics using M2 attribution events. Later M3 work should add
richer agentive-other detection, timing policy, and live hardware demos.

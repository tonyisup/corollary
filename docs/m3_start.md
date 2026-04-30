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
  - synthetic demo: overlapping speech causes deferral, and overlap during active speech causes interrupt/defer/resume
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
6. In the second scenario, accepted speech is interrupted by speech-bearing `BOTH`, queued, and resumed after cooldown.

## Current boundary

This is an early controller integration, not a full social model. It validates
turn-taking mechanics using M2 attribution events. Later M3 work should add
richer agentive-other detection, timing policy, and live hardware demos.

## SPEAK lifecycle slice

M3 now treats `SPEAK` as an actual controller phase rather than only a transient
accepted-output marker:

1. `request_output(...)` from `IDLE` enters `SPEAK` and records the active
   utterance.
2. M2 `SELF` frames sustain the speaking phase.
3. Speech-bearing `WORLD` or conservative speech-bearing `BOTH` frames during
   `SPEAK` interrupt the active utterance, queue it as deferred output, and
   transition to `LISTEN`.
4. Silence/cooldown returns the controller to `IDLE`, where deferred output can
   resume.

Focused coverage:

```bash
./.venv/bin/python -m pytest \
  tests/unit/test_pfc_agentive_controller.py \
  tests/unit/test_pfc_m2_turn_bridge.py \
  tests/unit/test_pfc_m2_artifact_replay.py
```

Observed: `12 passed`.

The synthetic demo now prints two scenarios: pre-speech deferral and active-speech
interruption.

## M2 artifact replay

The next bridge from synthetic M3 frames to hardware-facing behavior is replaying
saved M2 artifacts:

- `packages/pfc/m2_artifact_replay.py`
- `scripts/m3_replay_m2_artifact.py`
- `tests/unit/test_pfc_m2_artifact_replay.py`

Example:

```bash
./.venv/bin/python scripts/m3_replay_m2_artifact.py \
  data/m2_policy/family_policy_validation_20260429_130530/vowel_ah_var_nearfield_seed0
```

Observed replay from that artifact:

```text
summary=self/both/world=0/27/100 speech_ratio=0.38125
t=1.00s verdict=BOTH speech=True state=IDLE can_speak=True
t=1.15s verdict=BOTH speech=True state=LISTEN can_speak=False
t=1.50s verdict=SILENCE speech=False state=IDLE can_speak=True
accepted_initial_output=False
resumed_output='I can respond now.'
```

This proves saved M2 summaries can drive the M3 defer/resume controller path.
The replay is intentionally coarse because M2 summaries currently store frame
counts rather than the full per-frame verdict timeline. A later M3 hardware demo
should save the full M2 frame sequence.

## M3 artifact ladder

Added `scripts/m3_artifact_ladder.py` to run a graded replay ladder:

1. clean vowel overlap
2. repeated-babble overlap stress
3. world-only control
4. self-only control

Example:

```bash
./.venv/bin/python scripts/m3_artifact_ladder.py \
  --output data/m3_artifact_ladder_20260429_140211.json
```

Observed ladder result:

| Case | Verdict replay | M3 states | Output behavior |
| --- | --- | --- | --- |
| clean vowel overlap | `BOTH → BOTH → SILENCE` | `IDLE → LISTEN → IDLE` | defer, then resume |
| babble overlap stress | `BOTH → BOTH → SILENCE` | `IDLE → LISTEN → IDLE` | defer, then resume |
| world-only control | `BOTH → BOTH → SILENCE` | `IDLE → LISTEN → IDLE` | defer, then resume |
| self-only control | `SELF → SILENCE` | `IDLE → IDLE` | speak immediately |

Note: the world-only artifact replays as `BOTH` because existing M2 summary
artifacts store aggregate frame counts and that older world-only control still
reported motor-window `BOTH` frames. This is acceptable for the current M3
replay ladder because `BOTH+speech` is conservatively treated as listen/defer,
but future M2 artifacts should save full per-frame verdict timelines for more
faithful M3 replay.

## Per-frame M2 replay timeline

M2 artifacts now save a per-frame verdict timeline:

- `frame_verdicts.jsonl`
- each row includes frame index, time span, verdict, speech flag, RMS, match score,
  motor-window flag, causal score, and onset error

`frames_from_artifact(...)` now prefers `frame_verdicts.jsonl` when present and
falls back to aggregate `summary.json` only for older artifacts.

Fresh validation artifacts:
- `data/m2_policy/frame_timeline_validation_20260429_140648/vowel_seed0`
- `data/m2_policy/frame_timeline_validation_20260429_140648/babble_seed0`

The vowel timeline contains:
- `WORLD`: 130 frames
- `BOTH`: 29 frames
- `SILENCE`: 1 frame

Replay now uses real frame verdict order rather than synthesized aggregate
counts. Example vowel replay entered `LISTEN` after sustained world speech and
resumed after silence:

```text
t=0.00s verdict=WORLD speech=True state=IDLE can_speak=True
...
t=0.15s verdict=WORLD speech=True state=LISTEN can_speak=False
...
t=0.68s verdict=SILENCE speech=False state=IDLE can_speak=True
accepted_initial_output=False
resumed_output='I can respond now.'
```

## Live-ish M2 to M3 demo

Added `scripts/m3_live_m2_turn_demo.py`, which runs one M2 speech-preservation
trial and immediately feeds the in-memory `frame_records` into M3 turn-taking.
This avoids relying on saved artifacts for the main controller path.

Example:

```bash
./.venv/bin/python scripts/m3_live_m2_turn_demo.py \
  --save-dir data/m3_live_m2_turn_demo_20260429_142110
```

Observed run:

```text
M2 policy: vowel_ah_var_nearfield_promoted_gain_0p5 mode=gain gain=0.5
M2 frames self/both/world: 0/29/129
M2 WER raw/gated: 0.375/0.375
M3 accepted_initial_output=False
M3 resumed_output='I can respond now.'
M3 states: IDLE -> IDLE -> IDLE -> IDLE -> IDLE -> LISTEN -> LISTEN -> ... -> IDLE
```

During this work, an important replay-selector issue surfaced: selecting the
first interesting frames could choose early non-speech `WORLD` frames and miss
the actual speech-bearing segment. Timeline replay now prioritizes speech-bearing
`WORLD`/`BOTH` frames before falling back to controls such as `SELF`.

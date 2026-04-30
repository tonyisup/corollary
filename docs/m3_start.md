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

## Artifact-backed SPEAK interruption replay

Added `scripts/m3_speak_interrupt_replay.py` to compose saved M2 evidence into
the minimal M3 SPEAK-interruption path:

1. a self-only M2 artifact supplies a `SELF` frame that represents active agent
   speech,
2. an overlap artifact supplies speech-bearing `WORLD`/`BOTH` evidence,
3. synthetic silence closes the cooldown window.

Default command:

```bash
./.venv/bin/python scripts/m3_speak_interrupt_replay.py
```

Observed behavior:

```text
t=1.00s verdict=SELF speech=False state=IDLE can_speak=True interrupt=False
t=1.15s verdict=WORLD speech=True state=LISTEN can_speak=False interrupt=True
t=1.50s verdict=SILENCE speech=False state=IDLE can_speak=True interrupt=False
accepted_initial_output=True
resumed_output='I was speaking, then paused.'
```

This is still a composed replay, not a fully live simultaneous speaking demo,
but it proves the new M3 SPEAK lifecycle works against saved M2 artifact frames
rather than purely synthetic frame fixtures.

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

The demo now accepts explicit M2 calibration flags plus M3 controller flags such
as `--request-at-s`, `--listen-threshold-ms`, and `--cooldown-ms`. Passing
`--request-at-s 0` asks M3 to begin speaking at the first replayed M2 frame,
which can exercise the active-SPEAK interruption path if later M2 frames contain
speech-bearing `WORLD`/`BOTH` evidence.

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

### Live-ish SPEAK interruption smoke run (2026-04-30)

Command:

```bash
./.venv/bin/python scripts/m3_live_m2_turn_demo.py \
  --request-at-s 0 \
  --save-dir data/m3_live_m2_turn_interrupt_20260430_0945
```

Observed M2/M3 behavior:

```text
M2 frames self/both/world: 0/29/131
M3 request_at_s=0.0
M3 accepted_initial_output=True
M3 resumed_output='I can respond now.'
M3 interrupts: [False, True, False, ...]
```

This shows the live-ish path can now accept output, encounter M2 agentive speech
evidence, interrupt, and resume. The run remains “live-ish” rather than fully
embodied because the agent is not yet emitting real TTS while the M2 capture is
running.

## M3 acceptance ladder

Added `scripts/m3_acceptance_ladder.py` as the current regression/acceptance
slice for M3 turn-taking behavior. It checks four cases:

1. defer on agentive overlap,
2. speak on self-only frames,
3. interrupt active synthetic speech on overlap,
4. interrupt active speech using artifact-backed M2 frames.

Command:

```bash
./.venv/bin/python scripts/m3_acceptance_ladder.py \
  --output data/m3_acceptance_ladder_20260430.json
```

Observed:

```text
PASS defer_on_agentive_overlap
PASS speak_on_self_only
PASS interrupt_active_speech_on_overlap
PASS artifact_backed_interrupt
```

This gives M3 a concrete acceptance gate before moving from replay/live-ish
controller demos toward a fully embodied TTS/mic loop.

## Embodied Broca → M2 → M3 TTS loop

Added `scripts/m3_embodied_tts_loop_demo.py` as the first embodied bridge:

1. Broca synthesizes the actual requested agent utterance.
2. The generated PCM is passed into the M2 harness as the self-reference and
   played through the output device during mic capture.
3. M2 emits frame records from the real capture.
4. M3 consumes those frames with `request_at_s=0` so accepted speech can be
   interrupted by later speech-bearing `WORLD`/`BOTH` evidence.

Command:

```bash
./.venv/bin/python scripts/m3_embodied_tts_loop_demo.py \
  --save-dir data/m3_embodied_tts_loop_20260430_1005
```

Observed:

```text
Raw ASR: "I was speaking, then paused."
M2 frames self/both/world: 0/86/74
M2 self_score=0.031 onset_error_ms=150.0
M3 accepted_initial_output=True
M3 resumed_output='I was speaking, then paused.'
M3 interrupts: [False, True, False, ...]
```

Important boundary: the embodied path now uses real Broca audio and real mic
capture, but the first M2 attribution of TTS self-voice was weak
(`self_score=0.031`, no hard `SELF` frames). The turn-taking behavior still
works conservatively through speech-bearing `BOTH`/`WORLD`, but improving
TTS-self attribution is the next M2/M3 integration target.

## Broca self-only attribution calibration

Added `scripts/m3_broca_self_attribution_calibration.py` to remove world speech
and sweep a small set of attribution parameters against embodied Broca TTS.

Quick command:

```bash
./.venv/bin/python scripts/m3_broca_self_attribution_calibration.py \
  --latencies-ms 300,400 \
  --tolerances-ms 75,175 \
  --similarity-thresholds 0.02 \
  --onset-methods energy \
  --save-root data/m3_broca_self_calibration_20260430_quick
```

Observed top configs:

```text
trial001 energy latency=300 tolerance=175 threshold=0.02
  self/both/world=27/59/74 self_score=0.094 onset_error=0.0ms
trial000 energy latency=300 tolerance=75 threshold=0.02
  self/both/world=29/57/74 self_score=0.030 onset_error=0.0ms
```

Key result: Broca TTS self-only can produce hard `SELF` frames once onset
latency is corrected from `400ms` to about `300ms` and the similarity threshold
is lowered for room-colored TTS. The remaining low self-score confirms that a
room-colored/learned self reference is still the likely next improvement, but
basic timing calibration already recovers a meaningful self-present signal.

## Room-colored Broca self-reference analysis

Added `scripts/m3_broca_room_reference_analysis.py` to compare dry Broca PCM
matching against a small learned FIR room-colored reference on saved self-only
calibration artifacts.

Command:

```bash
./.venv/bin/python scripts/m3_broca_room_reference_analysis.py \
  data/m3_broca_self_calibration_20260430_quick/trial000_energy_lat300_tol75_thr0p02 \
  data/m3_broca_self_calibration_20260430_quick/trial001_energy_lat300_tol175_thr0p02 \
  data/m3_broca_self_calibration_20260430_quick/trial002_energy_lat400_tol75_thr0p02 \
  data/m3_broca_self_calibration_20260430_quick/trial003_energy_lat400_tol175_thr0p02 \
  --output data/m3_broca_self_calibration_20260430_quick/room_reference_analysis.json
```

Observed proof-of-concept:

```text
trial002 ... dry_local=0.143 room_local=0.381 improvement=0.238
trial003 ... dry_local=0.083 room_local=0.134 improvement=0.051
```

This is an offline same-artifact fit, so it is not held-out evidence yet. It
still strongly supports the architectural direction: embodied Broca self
attribution should compare mic capture against a room-colored forward reference,
not dry TTS PCM.

## Held-out room-reference validation

Added `scripts/m3_broca_room_reference_holdout.py` to fit a room filter on one
self-only Broca artifact and score it on different held-out artifacts.

Command:

```bash
./.venv/bin/python scripts/m3_broca_room_reference_holdout.py \
  data/m3_broca_self_calibration_20260430_quick/trial000_energy_lat300_tol75_thr0p02 \
  data/m3_broca_self_calibration_20260430_quick/trial001_energy_lat300_tol175_thr0p02 \
  data/m3_broca_self_calibration_20260430_quick/trial002_energy_lat400_tol75_thr0p02 \
  data/m3_broca_self_calibration_20260430_quick/trial003_energy_lat400_tol175_thr0p02 \
  --output data/m3_broca_self_calibration_20260430_quick/room_reference_holdout.json
```

Observed:

```text
pairs=12 improved=7 mean_improvement=0.031
best: dry=0.143 room=0.354 improvement=0.210
```

This is modest but real held-out evidence. The next implementation step is to
turn the room-colored score into an optional M2 attribution mode, probably as a
local expected-window score rather than replacing global dry xcorr everywhere.

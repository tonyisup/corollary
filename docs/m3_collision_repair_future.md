# Future Milestone: Overlap Collision Repair / Floor Negotiation

> Status: Future milestone — do not mix into current M3 reflex validation.

## Motivation

In phone-like human conversation, overlap collisions are normal. Without body-language cues, both speakers sometimes start at once and then perform a quick floor-negotiation repair:

- "Sorry, go ahead."
- "No, you first."
- "I think we talked over each other — go ahead."
- "Let me finish this one sentence, then I want to hear you."

This is a real conversational behavior Corollary should eventually support, but it should not be added before the lower-level acoustic/controller loop is trustworthy.

## Layering

Keep these as separate layers:

### M3: Reflexive turn-taking substrate

M3 should answer:

> Who is speaking, and should I stop emitting audio?

Responsibilities:

- detect agentive human speech
- detect overlap while the agent is speaking
- pause/defer safely
- avoid false interrupts
- resume when the human overlap has ended
- produce reliable state transitions: `SPEAK -> LISTEN -> IDLE`

### Future collision-repair policy

A future M3.5/M5 policy should answer:

> Given that we collided, what social move should I make next?

Responsibilities:

- decide whether to silently yield, apologize/yield, continue briefly, or explicitly negotiate the floor
- distinguish substantive interruptions from backchannels (`yeah`, `right`, `mm-hmm`)
- use conversation context and user preference
- avoid repeated over-yielding or repeated talking-over

## Why not now

Adding repair utterances now would blur failure modes:

- acoustic attribution failure
- overlap detector failure
- controller state-machine failure
- resume failure
- conversational policy failure

Current M3 validation should remain focused on the primitive reflex:

> Detect overlap, yield, and resume correctly.

## Passive instrumentation to add now

It is reasonable to log collision metadata now without changing behavior.

Suggested future event fields:

- `overlap_detected`
- `agent_was_speaking`
- `human_overlap_duration_ms`
- `agent_overlap_duration_ms`
- `agent_yielded`
- `human_continued_after_yield`
- `collision_repair_needed`
- `repair_policy`
  - `silent_yield`
  - `apology_yield`
  - `continue_briefly`
  - `explicit_floor_negotiation`

## Suggested milestone gate

Only promote collision repair after:

1. dual-speaker E2E metrics are semantically clean
2. overlap interrupt/resume is stable across repeated live runs
3. human-before and human-after cases are not counted as ordinary overlap interrupts
4. backchannel vs interruption examples are represented in data

## Candidate future scenarios

- Both start at the same time; agent yields silently
- Both start at the same time; user stops first; agent continues
- User gives a backchannel during agent speech; agent should not yield
- User gives substantive interruption during agent speech; agent should yield
- Repeated collisions; agent explicitly negotiates: "Sorry, go ahead."
- Critical instruction in progress; agent says: "One second — I’ll finish this sentence." then yields

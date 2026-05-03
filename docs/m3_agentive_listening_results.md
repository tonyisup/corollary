# M3 Agentive Listening — Decision Benchmark Results

> **Status:** Controller validated, attribution still fragile
> **Date:** 2026-05-03
> **Scope:** M3 turn-taking controller with M2+M4 attribution

## Controller: Proven

The `AgentiveController` is now validated across 8 synthetic scenarios with 33
decision-point checks. All pass.

### Scenarios tested

| Scenario | States | Result |
|----------|--------|--------|
| clean_self | SPEAK→IDLE | PASS — stays in SPEAK, never defers |
| clean_world | IDLE→LISTEN | PASS — defers after listen_threshold |
| self_then_world | SPEAK→IDLE→LISTEN | PASS — completes, then defers |
| world_interrupts_self | SPEAK→LISTEN→IDLE→LISTEN→IDLE | PASS — immediate interrupt |
| self_resumes_after_interrupt | SPEAK→LISTEN→IDLE→LISTEN→IDLE | PASS — resumes after cooldown |
| brief_both_during_self | SPEAK→IDLE | PASS — BOTH during motor window = no interrupt |
| sustained_world_during_self | SPEAK→LISTEN→IDLE→LISTEN→IDLE | PASS — immediate interrupt on first WORLD |
| world_during_silence | IDLE→LISTEN→IDLE | PASS — defers after listen_threshold |

### Key design decisions validated

1. **listen_threshold_ms (100ms)**: Only applies to IDLE→LISTEN transition. SPEAK→LISTEN is immediate on first WORLD frame.
2. **BOTH during motor window = self-activity**: Uses `is_in_motor_window` from the attribution result (ground truth), not controller state.
3. **Cooldown-based resume**: After world speech ends, cooldown_ms (200ms) must pass before deferred output runs.
4. **request_output() required**: Agent must explicitly request to speak. SELF attribution frames alone don't trigger SPEAK state.

## Attribution: Still Fragile

The controller is correct. The problem is getting it the right input.

### Known issues

| Issue | Impact | Root cause |
|-------|--------|------------|
| Ambient WORLD before playback | False defer (2/5 real turns) | Room reflections classified as speech |
| BOTH during playback | Conservative (not wrong) | M4 confidence < 0.50 → BOTH instead of SELF |
| No real interruption data | Recovery unvalidated | No recorded human interruption events |

### What would make M3 "proven"

1. **Real interruption recordings**: Human speech overlapping Corollary's output, with ground truth labels
2. **End-to-end metric**: Does M4 reduce false interrupts / false defers compared to M2-only?
3. **Recovery validation**: After interruption, does Corollary resume its deferred utterance?

## Next Steps

1. **Record interruption scenarios**: 5-10 recordings of human speech interrupting Corollary at various points during playback
2. **Label ground truth**: For each frame, label whether Corollary should be speaking, listening, or resuming
3. **Compare M2-only vs M2+M4**: Run both attribution pipelines through the same scenarios and measure decision quality
4. **Target**: <5% false interrupt rate, >80% successful resume rate

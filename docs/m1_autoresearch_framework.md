# M1 Autoresearch Framework for Corollary

> **Status:** Draft
> **Created:** 2026-04-17
> **Inspiration:** `~/src/autoresearch`
> **Scope:** Applying bounded autonomous experiment loops to Corollary M1 source-attribution work

## Why this document exists

The `autoresearch` repo demonstrates a very strong pattern:

1. fix the evaluation harness
2. bound each experiment tightly
3. let the agent change a constrained surface area
4. score the result
5. keep or discard
6. repeat

That pattern is highly relevant to Corollary.

What does **not** transfer directly is the exact setup:
- Corollary is not a single scalar-loss model-training loop
- Corollary is not a single-file system
- Corollary has real-world audio variability, device effects, and room dependence

So the right move is not to copy `autoresearch` literally.
The right move is to adapt its **research-operating-system pattern**.

## What is good about autoresearch

From `~/src/autoresearch`, the key useful ideas are:

### 1. Fixed evaluation ground truth
In autoresearch, `prepare.py` is frozen and the eval metric is not negotiable.

**Corollary analogue:**
- freeze benchmark scenarios for each M1 stage
- freeze the metric computation per scenario
- do not let experiment code change the scoring rules casually

This prevents metric gaming.

### 2. Bounded experiment time
Autoresearch uses a fixed 5-minute budget.

**Corollary analogue:**
- fixed number of trials or fixed wall-clock budget per experiment
- fixed babbling iteration counts for a benchmark tier
- fixed mixed-source test set per evaluation pass

This keeps experiments comparable.

### 3. Constrained change surface
Autoresearch intentionally limits changes to `train.py`.

**Corollary analogue:**
We likely should not allow the agent to freely mutate the whole repo during autonomous sweeps.
Instead, define a constrained mutable surface:
- experiment config
- tuning parameters
- maybe one attribution strategy module

Not everything at once.

### 4. Keep/discard branching logic
Autoresearch advances only on improvement.

**Corollary analogue:**
- keep config/code changes only if they improve stage-specific metrics
- otherwise revert
- preserve logs for both successes and failures

### 5. Lightweight research memory
Autoresearch uses `results.tsv` as the running experiment ledger.

**Corollary analogue:**
- store each run as a row in a structured ledger
- include config, metrics, room, devices, and notes
- support ranking and filtering by stage

This is especially important because Corollary has environment-specific effects.

## What should be different for Corollary

## 1. Multi-metric evaluation, not single-metric optimization
Autoresearch optimizes a single scalar, `val_bpb`.

Corollary M1 cannot safely collapse to one metric too early.

For example:
- M1a cares about self/not-self accuracy, onset alignment, false self rate
- M1b cares about convergence slope and attribution improvement
- M1c cares about overlap F1 and world→self confusion

So Corollary needs either:
- a stage-specific score function, or
- Pareto-style ranking with guardrails

### Recommendation
Use a **primary metric plus hard constraints**.

Examples:
- **M1a primary:** self attribution F1
  - constraints: false self rate below threshold, onset error below threshold
- **M1b primary:** attribution improvement after adaptation
  - constraints: convergence stability, no blow-up in prediction error
- **M1c primary:** overlap F1
  - constraints: low world→self confusion, stable label transitions

## 2. Environment-aware logging
Unlike autoresearch, Corollary experiments depend heavily on:
- room
- mic
- speaker
- volume
- ambient noise

A result without environmental metadata is nearly useless.

### Recommendation
Every run record should include:
- run id
- stage
- scenario id
- room tag
- input device
- output device
- playback level
- interference type
- parameter config hash
- metrics
- keep/discard decision

## 3. Separate config search from code mutation
In autoresearch, code mutation is the main action.

For Corollary M1, the first big win is likely **parameter search**, not open-ended code mutation.

That means the progression should be:

### Phase A
Config-level autoresearch
- sweep thresholds, IR length, `nlms_mu`, shift window, etc.
- keep code fixed

### Phase B
Small strategy mutations
- alternative alignment rules
- alternative gating heuristics
- different overlap logic

### Phase C
Broader code evolution
- only after benchmarks and logging are stable

This is safer and more interpretable.

## Recommended Corollary adaptation

## Core pattern

Define a bounded research loop:

1. choose stage and benchmark set
2. choose candidate config or small code change
3. run fixed evaluation harness
4. collect metrics and metadata
5. compute stage-specific score + guardrail checks
6. mark result as keep/discard
7. save to experiment ledger
8. iterate

## Minimal architecture

### Frozen pieces
- benchmark definitions
- scoring functions
- result schema
- experiment runner interface

### Mutable pieces
- parameter configs
- optional attribution strategy implementation
- optional search policy

## Suggested files

```text
docs/
  m1_autoresearch_framework.md
  m1_experiment_matrix.md
  m1_parameter_tuning.md
scripts/
  m1_eval_runner.py
  m1_search.py
  m1_score.py
  m1_manifest.py
results/
  m1_results.jsonl
  m1_best_configs.json
configs/
  m1/
    baseline.yaml
    sweeps/
```

## Proposed experiment loop for Corollary

## Stage 0: Baseline lock-in
Before autonomous search begins:
- define benchmark subsets for M1a, M1b, M1c
- define canonical metrics
- record baseline config and baseline results

No search before baseline.

## Stage 1: Config sweeps
Run bounded parameter search over known knobs:
- playback amplitude
- onset threshold
- shift window
- `nlms_mu`
- IR length
- hysteresis
- overlap threshold

This is the cleanest first application of autoresearch principles.

## Stage 2: Guided local search
After initial sweeps:
- hill-climb around promising configs
- stage-specific narrowing
- cross-room validation on top candidates

## Stage 3: Strategy search
Only after config search stabilizes:
- try alternative alignment algorithms
- alternative confidence fusion
- alternative overlap decision logic

## Stage 4: Optional code-evolving loop
If desired later, allow the agent to propose small code diffs, run eval, and keep/discard based on score.

But this should come **after** the harness is mature.

## Suggested scoring approach

## M1a scoring
Primary objective:
- maximize self attribution F1

Guardrails:
- false self-attribution rate must stay below threshold
- onset alignment error must stay below threshold

## M1b scoring
Primary objective:
- maximize improvement in attribution after adaptation

Guardrails:
- convergence must be monotonic enough or at least net-positive
- prediction error must not explode

## M1c scoring
Primary objective:
- maximize overlap F1

Guardrails:
- world→self confusion below threshold
- replayed-self negative control must pass

## Ledger format recommendation

A TSV is probably too thin for Corollary.
Use JSONL.

Each line should include fields like:

```json
{
  "run_id": "m1a-2026-04-17-001",
  "stage": "M1a",
  "scenario": "tone_baseline",
  "room": "office",
  "input_device": "macmini-mic",
  "output_device": "desk-speaker",
  "config": {
    "playback_amplitude": 0.5,
    "onset_threshold": 0.12,
    "shift_window_ms": 120
  },
  "metrics": {
    "self_f1": 0.93,
    "false_self_rate": 0.02,
    "onset_error_ms": 18
  },
  "score": 0.93,
  "guardrails_passed": true,
  "decision": "keep"
}
```

This is much easier to analyze later.

## Search policy options

## Simple grid/random search first
Best first move.

Pros:
- easy to reason about
- reproducible
- enough for current parameter count

## Successive narrowing
After initial results:
- narrow search around promising regions
- test top configs across more rooms/interference

## Bayesian optimization later
Possible, but probably premature.
Corollary is still at the stage where benchmark design matters more than fancy search.

## Recommended first implementation

If we want the smallest useful autoresearch-style layer for Corollary, I would build:

### 1. `scripts/m1_eval_runner.py`
Runs one experiment config against one benchmark subset and emits JSON.

### 2. `scripts/m1_search.py`
Loops over configs, calls the eval runner, appends JSONL rows, ranks results.

### 3. `scripts/m1_score.py`
Contains stage-specific scoring and guardrail logic.

### 4. `results/m1_results.jsonl`
The experiment ledger.

That is enough to create a real autoresearch loop without overengineering.

## My recommendation

Yes, Corollary should adopt an autoresearch-style framework, but in this order:

1. **freeze M1 benchmarks**
2. **add structured run logging**
3. **run config-level search first**
4. **only later allow code-level autonomous mutation**

That preserves rigor.

If we jump too early to open-ended code mutation, the agent will optimize noise, room quirks, or benchmark artifacts instead of the actual developmental problem.

## Closing

The transferable insight from `autoresearch` is not “let the agent rewrite everything.”

It is:

**Make research a tight loop with fixed evaluation, bounded experiments, explicit memory, and ruthless keep/discard discipline.**

That is exactly what Corollary needs next, especially for M1.

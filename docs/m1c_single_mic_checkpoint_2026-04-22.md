# M1c Single-Mic Checkpoint (2026-04-22)

> Status: checkpoint note before expected dual-mic follow-up. This is a calibration/status document, not a milestone-completion claim.

## Executive summary

The single-mic M1c work established that:
- the setup can produce real self/world structure
- naive global scalar thresholding is too weak
- stimulus-aware calibration is materially better than global thresholding
- external speaker volume materially affects world-path evidence
- the current single-mic setup is useful for learning, but probably not the place to expect the main breakthrough in self/world differentiation

Current recommendation:
- stop grinding single-mic threshold tweaks indefinitely
- move next toward **dual-mic experiments with simple sounds** once the new microphone is available

## What was learned

### 1. The hardware path can produce meaningful signal

Across multiple M1c runs, the harness consistently produced nontrivial values for:
- `self_score`
- `world_score`
- `self_evidence_db`
- `world_evidence_db`

This means the problem is not lack of any signal at all.

### 2. Whole-run RMS was misleading and was fixed

Early M1c hardware runs returned `SILENCE` verdicts even when the logs showed strong match structure.

Cause:
- whole-capture RMS averaged over mostly silent windows and drove attribution toward `SILENCE`

Fix made:
- `scripts/m1c_hardware_harness.py` was updated to compute local window metrics:
  - `baseline_rms_db`
  - `self_rms_db`
  - `world_rms_db`
  - `expected_self_rms_db`
  - `expected_world_rms_db`
  - `self_evidence_db`
  - `world_evidence_db`

Result:
- the harness became much more honest about local evidence structure even when the verdict itself remained limited.

### 3. The first tiny 6-row scalar hypothesis did not generalize

Initial tiny-batch observation:
- `self_evidence_db <= 2.62` separated `world_before_self` vs `self_before_world` on a 6-row batch

Larger grid result:
- this did **not** hold up on the larger structured run set
- accuracy on the larger grid was near chance / too weak for promotion

Conclusion:
- do not use a single global scalar threshold as the operating rule

### 4. Stimulus-aware calibration is better than global thresholding

From the larger order-grid analysis:
- best global single-feature rule was weak/moderate
- best global two-feature rule was only slightly better
- best **stimulus-aware** rules were materially better

Strongest honest takeaway:
- the order/calibration signal appears at least partly **stimulus-dependent**
- `world_evidence_db` is the strongest current simple feature
- adding `world_score` helps somewhat, but the gain is modest

Operational recommendation from this phase:
- keep the simpler **stimulus-aware `world_evidence_db`** rule as the current analysis baseline
- treat richer per-stimulus two-feature rules as analysis-only for now

## Tooling added during this phase

### Scripts / docs added or updated
- `scripts/m1c_hardware_harness.py`
  - local evidence metrics
  - calibration-ready JSONL row output
  - provisional order-prediction fields
- `scripts/m1_scratch.sh`
  - world-first / self-first structured collection into JSONL
- `scripts/m1c_order_rule_grid.py`
  - larger unattended order grid
- `scripts/m1c_order_rule_analyze.py`
  - global and stimulus-aware threshold search
- `scripts/m1c_stimulus_calibration_extended.sh`
  - longer unattended calibration sweep
- `scripts/m1c_world_volume_ladder.sh`
  - controlled external-speaker volume ladder
- `docs/adaptive_attribution_calibration.md`
- `docs/m1c_provisional_order_rule_2026-04-21.md`
- `docs/m1c_order_rule_grid_results_2026-04-21.md`

## What went wrong / what to watch out for

### 1. `verdict == SILENCE` is not by itself the deciding signal

The thalamic verdict remained limited / often unhelpful for this phase.

Important practice:
- prioritize raw/local evidence metrics over the verdict when evaluating M1c calibration quality

### 2. Post-restart runs exposed bad operating regimes

Some unattended runs after restart looked clearly invalid because:
- `baseline_rms_db` became much hotter than the expected event windows
- `self_evidence_db` and `world_evidence_db` became strongly negative (roughly `-25 dB` to `-30 dB` mean in the worst cases)

Conclusion:
- those runs should not be mixed into calibration fitting
- a short sanity gate is needed before committing to long unattended runs

### 3. External speaker volume is a real variable

The volume ladder showed a clear trend:
- low volume gave strongly negative `world_evidence_db`
- increasing volume improved world-path evidence substantially
- but higher world volume also reduced self-side cleanliness / separation

Observed rough pattern:
- `V1`: too low
- `V2`: transitional / near crossover
- `V3`: first clearly useful world level
- `V4`: strongest world level, but with more risk of over-dominating the balance

Caveat:
- the speaker wheel has no reliable detents, so reproducing an exact analog `V3` by hand is fuzzy

Conclusion:
- speaker volume matters, but it is a poor long-term precision control if it cannot be reset repeatably

## Current honest status of single-mic work

What the single-mic phase accomplished:
- showed that structured local evidence is measurable
- exposed which naive assumptions failed
- established stimulus-aware calibration as the better path
- identified external speaker volume as a genuine control knob
- recovered a usable operating regime after bad post-restart runs

What it did **not** accomplish:
- robust, repeatable single-mic order calibration across conditions
- a trustworthy promoted threshold rule
- a convincing reason to keep investing heavily in single-mic threshold tuning alone

## Recommended next step

Once the new microphone arrives:

### Move to dual-mic with simple sounds first

Use the dual-mic phase to improve:
- directional asymmetry
- self vs world differentiation
- repeatability of setup
- simpler geometric interpretation of separation

Do **not** jump immediately to richer speech/semantic tasks.

First dual-mic target should still use simple stimuli such as:
- `chirp`
- `tone`
- optionally `vowel_ah`

The goal is to get a visibly more distinct self/world separation before increasing task complexity.

## Suggested handoff question for the next session

When the new mic is available, the next concrete question should be:

> Can a dual-mic simple-sound setup produce a cleaner and more repeatable separation between self and world than the current single-mic setup?

That is the right bridge from this checkpoint into the next M1c push.

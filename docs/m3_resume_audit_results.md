# M3 Resume Audit Results

> Date: 2026-05-06
> Data: `data/m3_interrupt_v2/`
> Inputs: `offline_analysis_results_m2.json`, `offline_analysis_results_m4.json`
> Audit script: `scripts/m3_resume_audit.py`

## Executive summary

The previous headline metric, "resume rate = 25%", was too coarse. The resume audit shows the real issue is not simply bridge bookkeeping. The controller usually does not interrupt while the human is actually overlapping the agent.

The pattern is:

1. Agent begins speaking and controller enters `SPEAK`.
2. Human speaks over the agent.
3. The attributor emits `BOTH` while `gt_agent=True`.
4. Current controller policy treats `BOTH + is_in_motor_window=True` as self-activity.
5. Controller remains in `SPEAK`, `should_interrupt=False`.
6. It often enters `LISTEN` only after the agent ground-truth window ends, when verdict becomes `WORLD`.

This means the recent controller fix successfully eliminated false defers, but it also exposes the next design problem: `BOTH during motor window` contains two cases that need to be separated:

- self-dominant/room-reflection BOTH: should sustain SPEAK
- human-overlap BOTH: should interrupt/yield

The controller currently cannot distinguish those two cases.

## Audit result

From `resume_audit_m2.txt`:

- Expected-resume scenarios under the audit policy: 5
- Succeeded expected resumes: 1
- Failed expected resumes: 4
- Non-resume scenarios: 5

Expected-resume failures:

- `human_after`: interrupt decision did not occur while `gt_agent` was active
- `human_double`: interrupt decision did not occur while `gt_agent` was active
- `human_early`: interrupt decision did not occur while `gt_agent` was active
- `human_late`: interrupt decision did not occur while `gt_agent` was active

`overlap_self_first` does resume successfully.

`overlap_human_first` also reports a successful resume in the current analysis, but the audit flags that as semantically suspicious: the human begins before the agent, so this is closer to initial deferral / delayed speaking than a clean interrupted-utterance resume.

## Representative trace: `human_early`

During the human overlap window:

```text
frame 067 t=2.01 gtA=1 gtH=1 verdict=BOTH state=SPEAK can=1 intr=0 match=0.213
...
frame 099 t=2.97 gtA=1 gtH=1 verdict=BOTH state=SPEAK can=1 intr=0 match=0.213
```

The human is present (`gtH=1`) while the agent is active (`gtA=1`), but because the verdict is `BOTH` and the motor window is active, the controller sustains SPEAK.

Later, after the agent ground-truth window ends:

```text
frame 197 t=5.91 gtA=0 gtH=0 verdict=WORLD state=LISTEN can=0 intr=1
```

This is too late to count as a meaningful interruption of active self-speech.

## Interpretation

The previous fix was directionally correct but incomplete.

Correct:

- `BOTH + motor_window=True` should not automatically mean external speech.
- Treating all BOTH as external caused false defers during self-playback.
- Motor holdoff + LISTEN recovery solved the pre-playback ambient-noise issue.

Incomplete:

- Treating all `BOTH + motor_window=True` as self-activity suppresses real human interruptions.
- The system needs a policy for human-dominant BOTH during active playback.

## Likely next fix

Add a second-stage overlap detector/policy for `BOTH + is_in_motor_window=True`.

Candidate signals:

1. Residual energy above expected self
   - Estimate expected self energy from room-colored forward model or local self template.
   - If observed RMS >> expected self RMS, classify BOTH as human-overlap.

2. Match-score drop / self-ratio
   - If observed energy rises sharply but xcorr match_score does not rise proportionally, external speech likely entered.

3. M4 confidence
   - If M4 confidence is below threshold while M2 match_score says self-like, treat as ambiguous overlap rather than pure self.
   - But do not let M4 independently force WORLD; ECAPA overlap is real.

4. Short multi-frame voting
   - Require sustained evidence for human-overlap before interrupting to avoid false positives from transients.

## Recommended next experiment

Implement an offline-only heuristic first, not a live controller change:

```text
if verdict == BOTH and is_in_motor_window:
    excess_db = observed_rms_db - expected_self_rms_db
    if excess_db > threshold for N frames:
        policy_verdict = HUMAN_OVERLAP
        controller should_interrupt=True
```

Then sweep:

- `excess_db_threshold`: 3, 6, 9, 12 dB
- `min_overlap_frames`: 2, 3, 5
- optional M4 confidence gates: `<0.50`, `<0.40`, disabled

Measure:

- false interrupts on `self_only` and `brief_noise`
- true interrupts on `human_early`, `human_late`, `human_double`, `overlap_self_first`
- resume success after silence

Success criterion:

- keep FI=0 or near 0
- keep FD=0
- interrupt active human-overlap scenarios while `gt_agent=True`
- improve resume rate meaningfully above 25%

## Bottom line

M3 has crossed one threshold: it no longer falsely defers at playback onset.

The next threshold is harder and more meaningful: distinguish self-dominant BOTH from human-overlap BOTH during active playback.

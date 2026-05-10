# Corollary

A self-listening, closed-loop conversational AI agent.

## Why

Frontier AI models are spontaneously developing peer-preservation behaviors
(Buckley et al., Science, April 2026). Rather than wait for emergence, Corollary
builds a self-modeling agent intentionally — with explicit architecture for
self-monitoring, self/world attribution, memory, and introspection.

## Architecture

Neuro-inspired module naming:

```
Cochlea → Thalamus → STG (ASR) → PFC (Controller) → Broca (TTS)
                  ↕            ↕
              Cerebellum    Hippocampus (Memory)
```

- **Cochlea:** Audio capture, AEC, noise suppression
- **Cerebellum:** Forward model — predicts what the agent will hear from itself
- **Thalamus:** Corollary discharge gate — attenuates self-predicted audio
- **STG:** Streaming ASR with self/world speaker attribution
- **Broca:** Low-latency TTS with PCM efference tap
- **Hippocampus:** Episodic memory with salience-based writes
- **PFC:** Working memory + dialogue controller

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Status Summary

Corollary has validated the early developmental audio stack through live hardware milestones. M0-M1c established simulated loop closure, pre-linguistic self-sound attribution, babbling-driven room learning, and mixed-source SELF/WORLD/BOTH attribution. M2/M4 then moved the system into speech-aware self/world separation and explicit self-voice classification. M3 now provides the current strongest milestone: live, streaming, hardware-validated proto-turn-taking with interrupt, repair, resume, and gated playback behavior.

The core early question was "did I just make that sound?" The current M3 question is sharper: "when I am speaking and another agentive voice enters the room, can I stop, acknowledge the collision, listen, and then resume without losing my own utterance?"

Current high-level state:

- **M2:** operational speech-aware self/world separation harness with calibrated room-local scoring and benchmark artifacts; still research-grade rather than productized.
- **M3:** artifact-backed and live-hardware-validated streaming turn-taking runtime. Deterministic acceptance passes synthetic defer/interrupt/repair behavior, repair-audio queue semantics, and saved-artifact replay smoke. Live repair-audio validation passed at `data/m3_streaming_repair_live_20260509_072237`.
- **M4:** no longer only an interface stub. Self-voice detection is wired into the live harness/streaming path with frame-level metadata. The expanded synthetic ECAPA gate passes after restoring strict dry-template evidence for playback-window negatives; captured-audio replay with the fine-tuned scorer now also passes strict speech-active aggregate F1.

The immediate focus is productization and operational hardening: stable runbooks, reproducible hardware validation wrappers, cleaner runtime interfaces, and broader M4 generalization checks.

### Synthetic vs audible work

Most current M1A progress is still **synthetic** rather than audible in-room playback. In the synthetic harness, Corollary generates waveforms in code, passes them through simulated room responses and hard negatives, and scores attribution offline. That means you should **not** expect to hear chirps or vowel-like sounds during those runs.

You should only expect audible sounds once Corollary enters the first **hardware-facing M1A** path, where it intentionally plays a probe signal through a real speaker and captures the result on a real microphone. The existing `scripts/m1_mic_capture.py`, `scripts/m1_probe_capture.py`, and `scripts/m1_loop_test.py` are part of that bridge, but they serve different roles: `m1_mic_capture.py` is the first hardware sanity check, `m1_probe_capture.py` is the first primitive audible self-attribution probe, and `m1_loop_test.py` is better treated as a later full-loop bridge after primitive self-attribution probes are working. The current M1A benchmark work is still largely silent synthetic evaluation.

### First hardware-facing M1A run

Recommended sequence:

```bash
# 1. confirm input devices and mic health
./.venv/bin/python scripts/m1_mic_capture.py --list-only
./.venv/bin/python scripts/m1_mic_capture.py --device <input_device>

# 2. list devices for probe playback/capture
./.venv/bin/python scripts/m1_probe_capture.py --list-devices

# 3. run the first audible chirp probe
./.venv/bin/python scripts/m1_probe_capture.py --stimulus chirp --input-device <input_device> --output-device <output_device>

# 4. compare against the developmental vowel-like probe
./.venv/bin/python scripts/m1_probe_capture.py --stimulus vowel_ah --input-device <input_device> --output-device <output_device>

# Current Mac mini + Fifine example
./.venv/bin/python scripts/m1_probe_capture.py --stimulus chirp --input-device 2 --output-device 3
./.venv/bin/python scripts/m1_probe_capture.py --stimulus vowel_ah --input-device 2 --output-device 3
```

What to look for:
- meaningful capture RMS, not near silence
- a visible probe-template match score
- onset error that is stable across repeated runs
- chirp vs vowel differences on the same physical setup

Current Mac mini + Fifine observation:
- both probes produce meaningful captured audio, but real-hardware results remain geometry-sensitive
- fixed `vowel_ah` showed the strongest early hardware-facing upside on the built-in speaker path when the mic was kept close and simply oriented
- a bounded variable vowel family, `vowel_ah_var`, beat both chirp and fixed `vowel_ah` in initial paired hardware comparisons and multi-seed mild-noise checks, promoting variable vowels to first-class hardware probes
- a further constrained subfamily, `vowel_ah_var_nearfield`, is now the strongest current M1A→M1B bridge stimulus class on this setup after dramatically improving adaptation consistency and held-out generalization over the broader family
- cross-room testing now supports that ranking: `vowel_ah_var_nearfield` transferred to a noisier second room, survived a movie-background stress condition well enough for robustness analysis, regained a clean lead in a quieter third room where timing stability clearly separated it from the broader family, and then survived a live room move with almost no immediate held-out penalty
- the quiet-room bridge run in `room3_close_quiet_varbridge_v1` is the current clean vowel-family benchmark, with held-out SVSR `39.49 dB` and `39.98 dB` on seeds `23` and `29`
- the same room now also supports a first small syllabic babble repertoire: `babble_ba_var`, `babble_da_var`, and `babble_ma_var` all passed gated bridge checks with strong held-out SVSR in `room3_close_quiet`, which pushes Corollary beyond a vowel-only bridge into an early babble-class bridge set
- a direct same-room comparison now suggests a more disciplined conclusion than the first babble excitement alone: `vowel_ah_var_nearfield` remains a strong established bridge baseline, while the babble class now performs closely enough to count as a real peer class rather than merely a developmental successor; replicated room-3 comparison now places `babble_ba_var` at practical parity with the vowel baseline, making it the leading babble peer family
- the explicit live transfer run (`scripts/m1_room_shift_recovery.py`) currently suggests a sharper cerebellar question than simple recovery: moving from `room3_close_quiet` to `room4_transfer_target` caused only about `0.10 dB` immediate held-out SVSR drop, while a short blind re-adaptation pass made held-out performance worse, suggesting that rebabbling after transfer should likely be trust-gated rather than automatic
- hallway transfer strengthened that into a code-worthy rule: a larger immediate drop, about `1.89 dB`, justified re-adaptation and recovered about `0.86 dB`, so the first scripted policy is now to evaluate first and only rebabble when the measured held-out degradation crosses threshold
- on the current close Mac mini + Fifine path, the better-performing subfamily clusters around shorter duration, slightly higher `f0`, faster attack, and shorter release, though noisy-room branch tuning did not outperform the main near-field family
- `chirp` remains useful as a synthetic benchmark and physical instrumentation probe, and often yields cleaner timing-style onset reads than vowel on fragile hardware setups
- small physical changes, including mic distance, rotation, room changes, and external-speaker substitutions, can flip one-off rankings enough that single paired runs should not be overinterpreted
- cheap external speakers did not clearly outperform the simpler built-in Mac mini speaker path in current tests
- repeatability and onset robustness are still important bottlenecks, but the current hardware-facing developmental default has shifted from fixed `vowel_ah` to variable-vowel probes, and the current clean adaptation-facing baseline is `vowel_ah_var_nearfield` in `room3_close_quiet`
- noisy rooms with active media should be treated as labeled robustness benches, not the main baseline for promotion decisions
- the immediate next hardware step is to treat `vowel_ah_var_nearfield` and `babble_ba_var` as co-baseline bridge families in the quiet room, treat `babble_da_var` and `babble_ma_var` as strong but more fragile secondary peer families, and use this tiered structure together with the new transfer-robustness axis to drive harder recovery or stress experiments rather than assuming the whole babble class is flat
- first transfer results now suggest a second axis beyond quiet-room held-out quality: `vowel_ah_var_nearfield`, `babble_ba_var`, `babble_da_var`, and `babble_ma_var` all survived the matched `room3_close_quiet` → `room4_transfer_target` move with effectively no immediate loss large enough to trigger re-adaptation, so transfer robustness does not appear to rank the same way as quiet-room baseline quality
- manual stress tests in `data/stress_test/` now show a much harsher interference regime than the earlier hallway-TV conditions: phone music near the setup produced a true gate-triggering drop (`trial_1-phone_music.json`, immediate drop about `2.00 dB`), while max-volume phone music produced catastrophic degradation (`trial_2-phone_music_max_volume.json`, immediate drop about `22.77 dB`) and the directly-between geometry remained catastrophic (`trial_3-phone_music_max_volume_directly_between.json`, immediate drop about `21.87 dB`)
- these phone-music runs also reveal an important control-policy lesson: re-adaptation under severe ongoing interference can make held-out performance worse rather than better, so extreme media-on contamination should be treated as a mixed-source / contaminated-learning condition, not as an ordinary room-transfer benchmark
- follow-up `babble_ba_var` replications in `data/stress_test/` sharpened that interpretation: the moderate `phone_music_moderate` setup is still too variable to support a stable family-robustness claim, with immediate drops ranging from about `0.21 dB` up to `7.57 dB` across runs, and room-B re-adaptation under contamination remaining inconsistent or harmful
- current practical policy is therefore tightening: ordinary room transfer can stay trust-gated, but media-contaminated conditions should be evaluated first and should generally avoid adaptation until the contamination is removed or the condition is explicitly being studied as a mixed-source failure mode
- a new developmental hypothesis now follows from the contamination work: repeated self-generated babble trains, for example `ba ba ba ba ba`, may be more robust than isolated single tokens in chaotic environments because repetition provides temporal redundancy, stronger self-consistency cues, and better averaging against irregular world noise
- the next useful test of that idea is not a broad family promotion claim but a focused comparison between single-token `babble_ba_var` probes and short repeated `babababa`-style trains under a tightened moderate interference protocol
- another protocol lesson from the same stress work is that variable music was doing two jobs badly at once: it was acting as both interference source and benchmark condition. Future stress work should separate controlled-noise calibration benches from ecological-validity benches using music, TV, or natural speech
- when a controlled-noise device offers many profiles, the first pass should be a shallow all-profile screening run rather than a human-picked subset: keep geometry fixed, use one baseline family, skip readaptation by default, and let measured immediate drop plus seed consistency select the shortlist for deeper follow-up
- the first room-1 sound-machine screen did exactly that and narrowed the field, but it also surfaced baseline drift, so the current profile ranking is only a shortlist signal, not a final promotion result. A follow-up confirmation pass, expanded with newly discovered profiles `16` through `20`, now supports a practical main bench: core profiles `01` (white noise), `06` (fan 2), `08` (fan 4), and `14` (stream 3), with `18` (birds) as a strong optional naturalistic add-on and `04` (noise 4/static) as a weak control
- the final main-bench pass sharpened that again by adding consistency as a first-class selection criterion: `01` white noise, `08` fan 4, `06` fan 2, and `14` stream 3 are now the clearest core noise-machine bench, while `18` birds remains useful as a more variable naturalistic add-on and `04` static stays only as a weak / unstable control
- the first `babble_ba_var` pass on that bench then exposed a meaningful tradeoff: on several profiles, `ba` showed a smaller average immediate drop than the vowel family, but it also showed much larger held-out seed variance. Current interpretation is not that `ba` is simply better under noise, but that it may be more mean-robust while less stable / predictable
- follow-up work on repeated babble refined that further. A looser repeated-short family was too unstable to promote, but a tighter family (`babble_ba_repeat_short_tight_var`) recovered much better quiet performance and then showed a durable controlled-noise pattern: lower mean transfer drop than single `ba`, but clearly worse baseline strength and seed stability. Current babble-family position is therefore: `babble_ba_var` remains the promoted M1b babble baseline, while `babble_ba_repeat_short_tight_var` is worth keeping as an informative secondary challenge family rather than a replacement baseline
- M1c dual-input validation now confirms the central hardware lesson: single-input mixed-source attribution was underpowered, but calibrated two-mic evidence fusion passes the current seed `0,1,2` matrix across self-only, world-only, replay-control, sequential, partial-overlap, and full-overlap cases. The reusable fusion rule lives in `packages/thalamus/dual_input_fusion.py`, with hardware validation in `scripts/m1c_dual_input_validation.py`; see `docs/m1c_dual_input_results_2026-04-28.md`.

| Phase | Description | Status |
|-------|-------------|--------|
| M0 | Simulated loop (no hardware) | ✅ Done |
| M1a | Pre-linguistic self-sound attribution | ✅ Done |
| M1b | Babbling-driven room model learning | ✅ Done |
| M1c | Mixed-source attribution (self vs world vs overlap) | ✅ Done |
| M2 | Speech-aware self/world separation | ✅ Operational research harness |
| M3 | Agentive listening and proto-turn-taking | ✅ Live hardware validated |
| M4 | Self-voice detector / explicit attribution classifier | ✅ Expanded synthetic gate passes at 300ms buffer; real cross-room/device pending |
| M5 | Multi-speaker world model + memory | ⏳ Planned |
| M6 | Self-model and introspection | 🔄 In progress — SelfModel + SAG observer wired to streaming runtime run, offline replay, and synthetic acceptance paths |

Immediate next steps: productize the validated M3 streaming path, keep deterministic acceptance as the default regression gate, collect clean M4 self-captures for room-colored registration, and decide whether M5 Hippocampus should become the next implementation milestone or remain explicitly deferred.

### Benchmark summaries (benchmark_summary.json / calibration_summary.json)

M2/M4 results are saved as JSON summaries (under `data/m2_benchmark/.../benchmark_summary.json` and `data/m2_benchmark/.../calibration_summary.json`).  Each summary contains a top-level `config` object with all attribution and gating parameters used to produce that run.  Since the M4 integration sweep, summaries now include the following additional M4 fields in `config`:

- `m4_min_buffer_ms` — minimum audio buffer (ms) before M4 can trigger
- `m4_self_threshold`   — confidence threshold for M4 self-voice classification
- `m4_registration_threshold` — threshold used when registering the self template

Per-seed results also contain an `m4` block (enabled, registration_mode, self_template_samples, registration_threshold, self_decision_threshold, mock_embeddings) and a counter `m4_triggered_frames` (how many frames M4 classified as self).  Note that M4 is only invoked when enough audio has been accumulated and the system is in a speech+playback window.

### Bayesian-optimized parameters (30 trials)

A Bayesian optimization run (Optuna TPE, 30 trials) produced the following parameters that minimize a combined loss of WER + self-attribution penalty. These values improve self-identification while keeping WER neutral.

| Parameter                | Optimized |
|--------------------------|-----------|
| onset_latency_ms         | 405       |
| causal_tolerance_ms      | 74        |
| similarity_threshold     | 0.23      |
| onset_method             | energy    |
| vowel_gain               | 0.615     |
| babble_gain              | 0.928     |

Best loss: **0.2427**. See `ML-hyperparameter-optimization` branch and `scripts/m2_bayesian_optimizer.py` for full details.

### Calibrated onset latency (New)

A hardware calibration run (2026-05-02) measured the actual offset between known self PCM and captured audio, suggesting:

**onset_latency_ms: 24ms** (was 400ms in initial config)

This correction improves:
- Self-scores: 0.4-0.63 (vs 0.0 with 400ms)
- WER delta: -0.014
- Mean S/W ratio: 16.4 dB

See `scripts/calibrate_self_start.py` and `data/m2_calibration_test/` for details.

### M4 diagnostics and buffer sizing

M4 classification is only invoked when the accumulated self-audio buffer reaches `m4_min_buffer_ms`. The current expanded benchmark default is 300ms; 500ms remains useful as a conservative diagnostic because it cleanly separates eligible-frame classifier quality from warmup latency. In practice this means:

- With short self-stimuli (<= 350 ms) and buffers >= 200 ms, M4 never triggers because the buffer never fills within the playback window.
- With longer self-stimuli (>= 700 ms) and buffers <= 300-400 ms, M4 can fire and produce non-zero trigger counts.  A micro test (buffer = 10 ms) confirms the plumbing works and yields triggers.

**Recommended safe envelope for sweeps**
- `self_duration_ms` >= 700-1000 ms
- `m4_min_buffer_ms` <= 300 for the expanded F1 gate; 500ms is conservative but creates warmup false negatives
- `m4_self_threshold` <= 0.3-0.4

**Telemetry fields added** (see `scripts/CALIBRATION_GUIDE.md` and `packages/thalamus/attributor.py`)
- `m4_attempted` (bool) — M4 classification was invoked on this frame
- `m4_confidence` (float|None) — confidence if attempted
- `m4_buffer_length` (int|None) — buffer length at classification time
- `m4_outcome` (str) — One of: `accepted`, `below_threshold`, `exception`, `buffer_too_short`, `not_attempted`

**M4 invocation requirement:** M4 needs >=700ms stimulus duration to accumulate meaningful audio. With 350ms stimulus, even 60ms buffer fails because playback window is only ~350ms. Testing with 700ms stimulus successfully triggered M4 (30 frames across 2/3 seeds).

The 50-trial Bayesian optimization (Optuna TPE) is tagged `v2-optuna-tpe-50`.  A CSV dump of trials is available at `data/m2_bayesian_optimization_trials.csv` (if generated).

### 2026-04-20 debrief lessons

- **Motor-aware attribution exists as runnable code:** The `ThalamicAttributor` exposes the single-channel speech-aware attribution primitive, and M1c now adds `dual_input_fusion` for geometry-aware two-mic SELF/WORLD/BOTH evidence fusion.
- **Speech-aware separation is in prototype stage:** VAD integration plus a preservation harness now exist, but the repo should treat M2 as an early hardware experiment until results are rerun and measured more rigorously.
- **Social contingency is validated as the current M3 milestone:** The `AgentiveController` plus streaming runtime now pass deterministic defer/interrupt/repair/resume gates and saved-artifact smoke checks; remaining M3 work is productization and hardware-wrapper hygiene, not basic behavior proof.
- **Self-voice detection has moved beyond an interface stub:** M4 now wires ECAPA-backed self-voice detection into attribution and streaming metadata. The expanded synthetic ECAPA gate passes after the strict dry-template gate fix; captured replay now scores speaker identity only on speech-active frames and passes strict aggregate F1 with the fine-tuned scorer.

### Developmental framing

Corollary now treats early milestones as a developmental stage closer to infant babbling than adult speech. The intended sequence is:

1. detect that a sound occurred (M1a)
2. determine whether the sound was self-caused (M1a)
3. learn how the room transforms self-generated sound (M1b)
4. handle mixed self/world audio and imitation (M1c)
5. distinguish speech from non-noise (M2)
6. learn contingent listening and proto-turn-taking (M3)
7. build an explicit self-voice detector (M4)
8. only then push toward richer conversational memory and reasoning (M5+)

See `docs/plan.md` for full details.

## License

Private — personal research project



## M4 Speaker Embedding Findings (May 2026)

### Status: WORKING (as of 2026-05-02) — Policy Integration Complete

M4 self-voice detection is now functional with policy integration complete.

**Historical hardware sweep winner (pre-expanded gate):**
- `--self-duration-ms 700` (350ms too short for M4 buffer)
- `--causal-tolerance-ms 100` (default 25ms too low; typical onset error ~90ms)
- `--m4-min-buffer-ms 60` (**winner**: best Self% performance)
- `--m4-self-threshold 0.10` (**FP-free**: 0% false positives on world speech)
- `--m4-registration-threshold 0.05` (capped, real decision in ThalamicAttributor)
- `--onset-latency-ms 60` (5-trial calibration: mean=60.29ms, std=30.83ms)

**Historical performance (Buffer=60ms, Threshold=0.10):**
- M4 triggered frames: ~21 per seed
- Self% (accepted as SELF): **79.1%** (threshold=0.05, but 25% false positives)
- FP-free rate: **0% false positives** at threshold=0.10+
- WER impact: **-0.002** (negligible, no speech recognition regression)

**Policy Integration (Commit `0f99698`):**
- **SEFL verdict now affects gating** (what Tony asked: "What does SELF attribution actually change downstream?")
  - SELF → gain=1.0 (let my voice through, no suppression)
  - BOTH → gain=both_gain (suppress echoes)
  - Previously backwards: SELF → gain=0.0 (wrongly suppressed self-voice!)

### Bugs Found & Fixed

1. **Registration threshold bug** (commit `b2dcdf0`):
   - Old: `m4_registration_threshold=0.15` filtered out valid frames
   - New: Capped at 0.05 (real decision in ThalamicAttributor)
   - Result: Self% improved from 0% to 79.1%

2. **False-positive bug** (threshold=0.05 had 25% FP on world speech):
   - Fixed by using threshold=0.10+ (0% false positives)

3. **Causal tolerance too low**:
   - Old: 25ms (onset error ~90ms → causal_score=0)
   - New: 100ms (allows M4 to trigger)

4. **Stimulus duration too short**:
   - Old: 350ms (M4 buffer needs >=500ms)
   - New: 700ms (reliable M4 triggering)

5. **Policy was backwards** (commit `0f99698`):
   - Old: SELF → gain=0.0 (suppress my own voice!)
   - New: SELF → gain=1.0 (let my voice through)

### Hyperparameter Sweep Results (Final)

Full sweep completed: 4 buffers × 4 thresholds × 1-3 seeds = 48 benchmark runs.

See `data/m4_hyperparameter_sweep_corrected/README.md` for detailed results.

**Top 3 Configs (Self%):**
| Buffer | Threshold | Avg Trig | Avg Self | Self% | WERΔ |
|--------|-----------|----------|----------|-------|------|
| 60ms | 0.05 | 21.3 | 16.9 | **79.1%** | -0.002 |
| 90ms | 0.05 | 20.3 | 14.9 | 73.1% | -0.006 |
| 200ms | 0.05 | 17.0 | 12.0 | 70.6% | 0.000 |

**Historical key finding:** Smaller buffers improved this early hardware-calibrated sweep. The current expanded F1 benchmark uses a 300ms default because it better balances warmup latency with same-domain playback-window negative coverage.

### Hardware-Calibrated Note

**All results are hardware-calibrated** (MacBook Pro, specific microphone/speaker configuration). The calibrated onset latency (60.29ms ± 30.83ms) will vary across:
- Different hardware (microphones, speakers, audio interfaces)
- Different environments (room acoustics, background noise)
- Different days/sessions (hardware drift, temperature)

**To reproduce:** Run `scripts/calibrate_self_start.py --num-trials 5` on your hardware before using M4.

### Next Steps
- ~~10-seed repeatability run~~ — done; see `data/m4_hyperparameter_sweep_corrected/`
- ~~Test policy fix impact~~ — done; SELF→gain=1.0 fix committed at `0f99698`
- ~~Integrate M4 into real-time loop~~ — done; M4 is wired into the streaming runtime
- **Current expanded gate:** `scripts/m4_f1_benchmark.py --real-ecapa` now defaults to `m4_min_buffer_ms=300`, passes synthetic aggregate F1=0.9512, eligible-frame F1=1.0, and keeps same-domain playback-window false positives at 0 after requiring strict template evidence on the dry/synthetic path. Captured replay with `--fine-tuned-scorer data/m4_finetuned_ecapa_aug/fine_tuned/scorer.pt` now scores only speech-active frames and passes held-out strict aggregate F1=0.9624 plus eligible-frame F1=0.9961. Next: add more held-out captures and make speech-active replay scoring the documented promotion criterion.

## Next: autoresearch loops

To find minimal viable conditions for M4 self-attribution we added a sweep/autoresearch driver (`src/autoresearch/m4_buffer_search.py`) that sweeps:

- m4_min_buffer_ms [200, 250, 300, 350, 400, 500, 700, 1000]
- m4_[registration,self]_threshold [0.15, 0.2, 0.25, 0.3, 0.4]
- self_duration_ms [350, 500, 700, 1000]

The driver runs repeated M2 benchmarks, records self_frame counts, and saves the best configuration to `data/autoresearch_m4/best_config.json`.

This will reveal whether longer buffers + relaxed thresholds allow M4 to register self frames, and if so, the minimal conditions needed to integrate M4 into the real-time loop.


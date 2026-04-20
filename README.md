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

The initial simulated loop (M0) has been successfully validated, and the repo now contains active M1-era experiments around babbling, adaptive room learning, and forward-model convergence. The immediate focus is no longer word-level self-suppression. Instead, the next milestone is the more fundamental developmental problem: pre-linguistic source separation and self-attribution.

The core question for early hardware work is not yet "what word did I say?" but "did I just make that sound?"

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

| Phase | Description | Status |
|-------|-------------|--------|
| M0 | Simulated loop (no hardware) | ✅ Done |
| M1a | Pre-linguistic self-sound attribution | 🎯 Current focus |
| M1b | Babbling-driven room model learning | 🧪 In progress |
| M1c | Mixed-source attribution (self vs world vs overlap) | ⏳ Planned |
| M2 | Speech-aware self/world separation | ⏳ Planned |
| M3 | Agentive listening and proto-turn-taking | ⏳ Planned |
| M4 | Self-voice detector / explicit attribution classifier | ⏳ Planned |
| M5 | Multi-speaker world model + memory | ⏳ Planned |
| M6 | Self-model and introspection | ⏳ Planned |

### Developmental framing

Corollary now treats early M1 as a developmental stage closer to infant babbling than adult speech. The intended sequence is:

1. detect that a sound occurred
2. determine whether the sound was self-caused
3. learn how the room transforms self-generated sound
4. handle mixed self/world audio
5. distinguish speech from non-speech and vocal agents from environment noise
6. learn contingent listening and proto-turn-taking
7. only then push toward richer conversational memory and reasoning

See `docs/plan.md` for full details.

## License

Private — personal research project

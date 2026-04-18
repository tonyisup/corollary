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
- on the current close Mac mini + Fifine path, the better-performing subfamily clusters around shorter duration, slightly higher `f0`, faster attack, and shorter release
- `chirp` remains useful as a synthetic benchmark and physical instrumentation probe, and often yields cleaner timing-style onset reads than vowel on fragile hardware setups
- small physical changes, including mic distance, rotation, and external-speaker substitutions, can flip one-off rankings enough that single paired runs should not be overinterpreted
- cheap external speakers did not clearly outperform the simpler built-in Mac mini speaker path in current tests
- repeatability and onset robustness are still important bottlenecks, but the current hardware-facing developmental default has shifted from fixed `vowel_ah` to variable-vowel probes, and the current adaptation-facing default candidate is `vowel_ah_var_nearfield`
- the immediate next hardware step is to use the constrained near-field family for further M1B adaptation work and then test whether its generalization survives broader room changes

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

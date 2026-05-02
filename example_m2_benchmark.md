
# M2 Benchmark Skill - Quick Start Example

This demonstrates the `m2-benchmark` Hermes skill for running M2 speech-preservation benchmarks.

Also see: `~/.hermes/skills/mlops/m2-benchmark/SKILL.md`

## Prerequisites

- Audio devices configured on 16 kHz (check with `sox --i -r` or `python -c "import sounddevice as sd; print(sd.query_devices())"`)
- World speech file at `data/world_speech_sample.npy` (already present)
- Virtual environment activated (`.venv/bin/activate`)

## Minimal Example

### 1. Run Calibration (Find Operating Amplitude)

```bash
cd ~/Projects/corollary

.venv/bin/python scripts/m2_benchmark.py \
  --mode calibrate \
  --self-stim vowel_ah_var_nearfield \
  --self-duration-ms 350 \
  --input-device 3 \
  --output-device 5 \
  --noise-device 4 \
  --save-root data/m2_benchmark \
  --similarity-threshold 0.20 \
  --onset-method energy \
  --causal-tolerance-ms 75.0 \
  --speech-override vad
```

**What this does:**
- Sweeps self-amplitude through [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0]
- For each amplitude, captures audio with self + world speech mix
- Runs ASR on raw and gated audio, computes WER
- Saves results to `data/m2_benchmark/calibration_YYYYMMDD_HHMMSS/`

**Expected output:** A table showing per-amplitude raw WER, gated WER, WER delta, and S/W ratio.

**Choose amplitude** where raw WER is ~0.2-0.4 (e.g., 0.15).

---

### 2. Run Full Benchmark (Multi-Seed)

```bash
cd ~/Projects/corollary

.venv/bin/python scripts/m2_benchmark.py \
  --mode benchmark \
  --self-stim vowel_ah_var_nearfield \
  --self-amplitude 0.15 \
  --self-duration-ms 350 \
  --input-device 3 \
  --output-device 5 \
  --noise-device 4 \
  --save-root data/m2_benchmark \
  --num-seeds 5 \
  --both-gain 0.5 \
  --both-mode gain \
  --similarity-threshold 0.20 \
  --speech-override vad \
  --causal-tolerance-ms 75.0
```

**What this does:**
- Runs 5 independent trials (seeds 0-4) at amplitude 0.15
- Same gating configuration (both-gain=0.5, gain mode)
- Saves per-seed results + aggregate summary

**View results:**
```bash
cat data/m2_benchmark/benchmark_*/benchmark_summary.json | python -m json.tool
```

---

### 3. Offline Replay (Compare Gating Strategies)

```bash
cd ~/Projects/corollary

# Replay against existing captures (no new hardware runs needed)
.venv/bin/python scripts/m2_benchmark.py \
  --mode offline \
  --capture-root data/m2_benchmark/calibration_*/captures \
  --num-seeds 1
```

**What this does:**
- Evaluates 7 gating strategies against the same saved captures:
  - `preserve`: Pass BOTH frames at gain 1.0
  - `gain_0p5`: Pass BOTH frames at gain 0.5
  - `subtract_0p1`: Subtract 0.1× self template from BOTH
  - `subtract_0p25`: Subtract 0.25× self template from BOTH
  - `subtract_0p5`: Subtract 0.5× self template from BOTH
  - `subtract_0p75`: Subtract 0.75× self template from BOTH
  - `subtract_1p0`: Subtract 1.0× self template from BOTH

**View strategy comparison:**
```bash
jq '.strategies' data/m2_benchmark/offline_*/offline_summary.json
```

---

## Using the Hermes Skill Wrapper

You can also invoke this through Hermes agent:

```bash
# Ask Hermes to run the skill
"run m2-benchmark calibration with defaults"
```

The skill definition is at:
- `~/.hermes/skills/mlops/m2-benchmark/SKILL.md`

---

## Key Parameters Cheat Sheet

| Task | Command | Critical Flags |
|------|---------|---------------|
| **Calibration** | `--mode calibrate` | `--self-stim`, `--save-root` |
| **Benchmark** | `--mode benchmark` | `--self-amplitude`, `--num-seeds` |
| **Offline** | `--mode offline` | `--capture-root` |
| **M4 Override** | Any mode | `--m4-enable`, `--m4-self-threshold` |
| **Subtract Mode** | Any mode | `--both-mode subtract`, `--both-subtract-gain` |

---

## Troubleshooting

### Issue: Raw WER = 1.0
- **Cause:** World speech not captured correctly (sample rate or device issue)
- **Fix:** Verify 16 kHz devices; check world speech file exists

### Issue: Gated WER = 1.0
- **Cause:** All speech frames suppressed (VAD or override)
- **Fix:** Add `--speech-override true` temporarily for testing

### Issue: WER delta ≈ 0.0
- **Cause:** Gating already optimal or broken
- **Fix:** Try `--both-mode subtract` or adjust `--both-gain`

### Issue: No captures saved
- **Cause:** Directory permissions
- **Fix:** Ensure `--save-root` exists and is writable

---

## Full Parameter Reference

See `~/.hermes/skills/mlops/m2-benchmark/SKILL.md` for complete documentation of all CLI parameters.


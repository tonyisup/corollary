# Corollary Babble Autosearch

> Find optimal babbling parameters to achieve SVSR ≥ 20 dB.
> Runs systematically through parameter combinations, records results,
> keeps the best, discards the rest.

## Setup

Fixed baseline:
- **Input:** Synthetic room IR (0.3s RT60) + real TTS audio (`broca.tts.synthesize`)
- **TTS phrase:** "Hello, this is Corollary testing the self-listening loop."
- **Metric:** SVSR after 10 babbling iterations (higher is better)
- **Target:** SVSR ≥ 20 dB (M0 spec)

The test uses synthetic room data because:
1. We already proved NLMS converges with 40dB SVSR on clean data
2. The question is *which parameter combination* converges fastest
3. The best synthetic params will transfer to real hardware

## Parameter Space

| Parameter | Values | Description |
|---|---|---|
| `nlms_mu` | 0.001, 0.005, 0.01, 0.05, 0.1, 0.25 | NLMS step size |
| `ir_length_ms` | 100, 150, 200, 300, 500 | Room IR length to learn |
| `bandpass` | false, true | Apply 300-3400 Hz bandpass filter |
| `speech_gate` | false, true | Gate captured audio to active TTS portion |

Total combinations: 6 × 5 × 2 × 2 = **120 experiments**

## Running the Search

```bash
# Run all experiments (slow, will take hours)
.venv/bin/python3 scripts/m1_autosearch.py

# Run just 10 experiments for a quick sanity check
.venv/bin/python3 scripts/m1_autosearch.py --limit 10

# Resume from a specific starting index
.venv/bin/python3 scripts/m1_autosearch.py --start-at 50
```

## Results

Each experiment records:
- **commit:** Current git hash
- **nlms_mu, ir_length_ms, bandpass, speech_gate:** Parameter values
- **svsr_db:** Final SVSR after 10 iterations
- **error_db:** NLMS error after 10 iterations
- **ir_corr:** Correlation between learned IR and true room IR
- **status:** `keep` (SVSR ≥ 20), `discard` (SVSR < 20), or `crash`
- **time_s:** Wall-clock duration
- **notes:** Brief description

Results are logged in `docs/autosearch_results.tsv`.

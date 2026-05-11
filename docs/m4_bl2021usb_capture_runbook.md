# M4 BL-2021USB Capture Runbook

> **Last updated:** 2026-05-11
> **Status:** Draft — production M4 capture cycle for BL-2021USB mic
> **Hardware target:** BL-2021USB (input device 1), Mac mini Speakers (output device 5)

## Purpose

Re-baseline the entire M4 self-voice detection pipeline on the production
microphone (BL-2021USB). The existing M4 scorer (`m4_finetuned_ecapa_v4`) and
all capture artifacts were recorded on the Fifine mic (input device 3) and do
not transfer to a different microphone.

This runbook covers the full cycle:
1. Clean self-registration capture
2. Validation capture (self + negative)
3. Human-voice negative capture
4. Fine-tuning
5. Replay acceptance verification

---

## Prerequisites

```bash
cd ~/Projects/corollary
source .venv/bin/activate
```

### Hardware assumptions

| Role | Device | Index |
|------|--------|-------|
| Mic input (production) | BL-2021USB | 1 |
| Agent TTS output | Mac mini Speakers | 5 |

Verify device indices before starting:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

If the BL-2021USB is not device 1, override `--input-device` in every command
below.

### Physical setup

- Place the BL-2021USB in its **permanent production position** before
  capturing. Moving the mic after registration invalidates the room filter.
- Use the same speaker output device and position as the intended production
  setup.
- Minimize background noise during captures (close windows, pause media, etc.).
- Do not move or reorient the mic between registration and validation captures.

---

## Phase 1: Clean Self-Registration Capture

Play a known TTS phrase through the speaker and capture it through the BL-2021USB.
This produces the room-colored self-embedding used by all subsequent stages.

```bash
.venv/bin/python scripts/m4_clean_capture.py \
  --input-device 1 \
  --output-device 5 \
  --output-dir data/m4_clean_captures_bl2021 \
  --text "Hello, I am Corollary. This is a clean self-recording for room-colored embedding registration."
```

**Expected output:**
- `data/m4_clean_captures_bl2021/agent_pcm_<ts>.npy` — dry TTS PCM
- `data/m4_clean_captures_bl2021/captured_<ts>.npy` — mic capture
- `data/m4_clean_captures_bl2021/meta_<ts>.json` — capture metadata

**Sanity checks:**
- Capture RMS should be between -25 dB and -40 dB (typical room level)
- Duration should match TTS + ~1s padding
- If RMS < -50 dB: check mic connection, input device index, and system gain
- If capture sounds distorted: reduce TTS amplitude or increase mic distance

**Playback verification (recommended):**
```bash
.venv/bin/python scripts/m4_playback_check.py \
  --capture data/m4_clean_captures_bl2021/captured_<ts>.npy \
  --agent-pcm data/m4_clean_captures_bl2021/agent_pcm_<ts>.npy \
  --output-device 5
```
This plays back both the captured audio and the dry agent TTS so you can
audibly verify the mic is picking up clean speech. It also prints RMS, peak,
speech region, and warnings. Use `--dry-run` to skip playback and just print
metadata.

**Troubleshooting:**
- `PortAudioError: Invalid device`: re-run the device query above; the
  BL-2021USB index may differ from the M1c dual-input setup if other audio
  devices are connected
- Near-silent capture: check that the BL-2021USB is not muted in macOS
  Sound preferences and that input volume is reasonable (50-75%)

---

## Phase 2: Validation Capture (Self + Negative)

Capture two scenarios: self-only TTS and a different-voice negative. These are
the held-out validation set for the fine-tuned scorer.

```bash
.venv/bin/python scripts/m4_capture_validation.py \
  --input-device 1 \
  --output-device 5 \
  --output-dir data/m4_capture_validation_bl2021_$(date +%Y%m%d_%H%M%S) \
  --self-text "Hello, I am Corollary. This is my voice speaking through the room." \
  --negative-text "I am a different speaker. This is a same-domain negative validation sample." \
  --self-voice "default" \
  --negative-voice "default"
```

**Expected output:**
```
data/m4_capture_validation_bl2021_<ts>/
  self_only/
    agent_pcm.npy
    live_captured.npy
    frame_records.json
    metadata.json
  same_domain_negative_playback_window/
    agent_pcm.npy
    live_captured.npy
    frame_records.json
    metadata.json
  summary.json
```

**Sanity checks:**
- Both scenarios should have capture RMS in the same ballpark (-25 to -40 dB)
- `self_only` metadata: `speech_verdict: SELF`, `dry_run: false`
- `same_domain_negative_playback_window` metadata: `speech_verdict: WORLD`,
  `dry_run: false`

**Playback verification (recommended):**
```bash
.venv/bin/python scripts/m4_playback_check.py \
  --capture data/m4_capture_validation_bl2021_<ts>/self_only/live_captured.npy \
  --agent-pcm data/m4_capture_validation_bl2021_<ts>/self_only/agent_pcm.npy \
  --output-device 5
```

**Note:** The `--self-voice` and `--negative-voice` flags depend on your Broca
TTS configuration. If you have multiple voices available, use a clearly
different voice for the negative to maximize embedding separation. If only one
voice is available, the same voice with different text content still provides
a valid negative (the room-colored embedding will differ from the registered
self template).

---

## Phase 3: Human-Voice Negative Capture

This is the critical gap from the PoC promotion. Record 3-5 different human
speakers reading the negative validation text through the BL-2021USB setup.

For each speaker:

```bash
# Quick ad-hoc recording script
.venv/bin/python -c "
import sounddevice as sd
import numpy as np
from pathlib import Path
import time

sr = 16000
duration = 10
out = Path('data/m4_human_negative_bl2021')
out.mkdir(parents=True, exist_ok=True)

print('Recording human negative sample...')
print('Please speak: I am a different speaker. This is a same-domain negative validation sample.')
time.sleep(2)
print('...speak now')

sd.default.device = 1
rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()

ts = time.strftime('%Y%m%d_%H%M%S')
path = out / f'human_negative_speaker1_{ts}.npy'
np.save(path, rec.flatten())
rms = float(np.sqrt(np.mean(rec ** 2)))
print(f'Saved: {path}')
print(f'RMS: {20*np.log10(max(rms, 1e-9)):.1f} dB')
"
```

Repeat for each speaker, incrementing the speaker label. Then build the formal
artifact:

```bash
.venv/bin/python scripts/m4_build_artifact_from_recording.py \
  --negative-audio data/m4_human_negative_bl2021/human_negative_speaker1_<ts>.npy \
  --output-dir data/m4_validation_human_bl2021 \
  --input-device 1 \
  --output-device 5
```

**Target:** At least 3 different human speakers, ideally 5+. Diversity matters
more than quantity — different genders, pitches, and speaking styles.

**Sanity checks:**
- Each recording should have clear speech energy (RMS > -35 dB)
- No clipping (check max absolute value < 1.0)
- Each speaker should be audibly distinct

**Playback verification (recommended):**
```bash
.venv/bin/python scripts/m4_playback_check.py \
  --capture data/m4_human_negative_bl2021/human_negative_speaker1_<ts>.npy \
  --output-device 5
```

---

## Phase 4: Fine-Tune ECAPA on BL-2021USB Captures

Train a new scorer on the production mic's captures.

```bash
.venv/bin/python scripts/m4_finetune_ecapa.py \
  --clean-captures data/m4_clean_captures_bl2021 \
  --other-speaker-dirs \
    data/m4_capture_validation_bl2021_<ts>/same_domain_negative_playback_window \
    data/m4_validation_human_bl2021/same_domain_negative_playback_window \
  --output-dir data/m4_finetuned_ecapa_bl2021 \
  --num-epochs 20 \
  --lr 1e-4 \
  --batch-size 16 \
  --output /tmp/m4_finetune_bl2021_log.json
```

**Expected output:**
- `data/m4_finetuned_ecapa_bl2021/fine_tuned/scorer.pt` — production scorer
  checkpoint
- `data/m4_finetuned_ecapa_bl2021/fine_tuned/config.json` — training config
- Training log at `/tmp/m4_finetune_bl2021_log.json`

**Sanity checks:**
- Final test accuracy should be > 85% (the PoC run achieved 87.5%)
- If accuracy is < 80%: check that self and negative captures are genuinely
  different in embedding space; may need more diverse negatives
- Training should complete in < 5 minutes on CPU, < 1 minute on MPS

**Note:** The `--other-speaker-dirs` argument accepts multiple directories. Include
both the synthetic negative (Phase 2) and all human negatives (Phase 3). The more
diverse the negatives, the better the human-voice generalization.

---

## Phase 5: Replay Acceptance Verification

Verify the new scorer passes all gates on the BL-2021USB captures.

### 5a. Synthetic expanded gate (no hardware)

```bash
.venv/bin/python scripts/m4_f1_benchmark.py \
  --real-ecapa \
  --threshold 0.70 \
  --m4-min-buffer-ms 300 \
  --fine-tuned-scorer data/m4_finetuned_ecapa_bl2021/fine_tuned/scorer.pt \
  --output /tmp/m4_bl2021_synthetic.json
```

**Pass criteria:** aggregate F1 >= 0.95, eligible F1 >= 0.95, 0 negative FP

### 5b. Held-out replay gate (no hardware)

```bash
.venv/bin/python scripts/m4_capture_replay_acceptance.py \
  --replay-dir data/m4_capture_validation_bl2021_<ts> \
  --register-from-clean data/m4_clean_captures_bl2021 \
  --fine-tuned-scorer data/m4_finetuned_ecapa_bl2021/fine_tuned/scorer.pt \
  --threshold 0.70 \
  --output /tmp/m4_bl2021_replay_acceptance.json
```

**Exit code 0** = pass. Check the JSON for per-scenario F1 and FP counts.

### 5c. Human-voice negative gate (no hardware)

```bash
.venv/bin/python scripts/m4_multi_artifact_acceptance.py \
  --artifact-dirs data/m4_validation_human_bl2021 \
  --register-from-clean data/m4_clean_captures_bl2021 \
  --fine-tuned-scorer data/m4_finetuned_ecapa_bl2021/fine_tuned/scorer.pt \
  --threshold 0.70 \
  --output /tmp/m4_bl2021_human_negative_acceptance.json
```

**Pass criteria:** 0 false positives on human speech at the chosen threshold.

If this gate fails (expected with < 5 speakers), iterate:
- Add more human negative speakers (Phase 3)
- Retrain (Phase 4)
- Re-verify (Phase 5c)

### 5d. Full suite

```bash
.venv/bin/python -m pytest -q
```

**Expected:** 341+ passed, 1 skipped. If new tests were added for BL-2021USB
artifacts, the count will be higher.

---

## Decision Matrix

| Situation | Command | Hardware? |
|-----------|---------|-----------|
| First-time BL-2021USB setup | Phase 1 → 2 → 3 → 4 → 5 (full cycle) | Yes (1,2,3) |
| New human negative speaker | Phase 3 → 4 → 5c | Yes (3 only) |
| Retrain after adding speakers | Phase 4 → 5 | No |
| Regression check after code changes | Phase 5a → 5b | No |
| Before commit touching M4 code | Phase 5d | No |

---

## Artifact Registry

After a successful full cycle, the production artifacts are:

| Artifact | Path | Purpose |
|----------|------|---------|
| Clean registration | `data/m4_clean_captures_bl2021/` | Room-colored self-embedding |
| Validation capture | `data/m4_capture_validation_bl2021_<ts>/` | Held-out self + negative |
| Human negatives | `data/m4_validation_human_bl2021/` | Human-voice rejection |
| Fine-tuned scorer | `data/m4_finetuned_ecapa_bl2021/fine_tuned/scorer.pt` | Production scorer |
| Acceptance verdicts | `/tmp/m4_bl2021_*.json` | Gate evidence |

**Important:** The Fifine-based artifacts (`m4_clean_captures_fixed`,
`m4_finetuned_ecapa_v4`, `m4_capture_validation_heldout_20260509`) remain as
development history but should not be used for production validation.

---

## Known Pitfalls

1. **Mic position matters.** The room filter is fitted to a specific mic/speaker
   geometry. If you move the BL-2021USB after registration, re-run Phase 1.

2. **Sample rate mismatch.** The clean capture script had a bug (fixed 2026-05-09)
   where 48k PCM was fed into a 16k stream, causing 1/3-speed playback. The
   current script handles this correctly, but if you hear slowed-down audio
   during capture, check that `broca.tts.synthesize()` returns 16k PCM.

3. **Threshold portability.** The θ=0.70 threshold was tuned on Fifine data. It
   may need adjustment for the BL-2021USB. Run Phase 5a with `--threshold`
   sweeps (0.60–0.80 in 0.05 steps) and pick the value that maximizes
   eligible F1 while keeping negative FP at 0.

4. **Human-voice gap may persist.** Even with 5+ speakers, ECAPA's frozen
   encoder may not fully separate self-TTS from human speech in the same
   embedding space. If Phase 5c fails after exhausting the data path, the next
   step is unfreezing the last 2–3 ECAPA layers for full encoder fine-tuning
   (not yet scripted).

5. **Device index drift.** The BL-2021USB may not always be device 1 (e.g., if
   a USB hub is disconnected and reconnected). Always verify with
   `sd.query_devices()` before running captures.

---

## References

- `docs/m3_operational_runbook.md` — M3 streaming validation (precedent for
  this runbook's structure)
- `docs/status_20260509.md` — M4 promotion decision and known limitations
- `data/m4_promotion_record_20260511.json` — PoC promotion gate results
  (Fifine baseline, for comparison)
- `corollary-audio-eval-workflows` skill — detailed pitfalls and calibration
  procedures

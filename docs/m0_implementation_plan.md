# M0 Implementation Plan: Simulated Loop

> **Goal:** TTS → simulated room IR → ASR with efference copy, proving the self-suppression architecture works  
> **Acceptance:** SVSR ≥ 20 dB, no self tokens in output, external WER unaffected

## Architecture

```
┌──────────┐    PCM frames     ┌──────────┐   features    ┌──────────┐
│  Broca   │ ──efference──►│ Cerebellum │───predict──►│ Thalamus  │
│  (TTS)   │   copy (PCM)   │ (fwd model)│   mel-spec  │  (gate)   │
└────┬─────┘                 └────┬─────┘              └────┬─────┘
     │                            │                         │
     │ self-audio                 │ room-audio              │
     ▼                            ▼                         ▼
┌──────────┐              ┌──────────┐              ┌──────────┐
│ Sim Room │──room audio──►│  STG    │──transcript──►│   PFC    │
│  (conv)  │              │  (ASR)  │               │(controller)
└──────────┘              └──────────┘              └──────────┘
```

The core idea: we generate speech, run it through a simulated room, transcribe it, and use the efference copy to predict what we should hear back. The thalamus gates based on prediction error.

## Dependencies

| Component | M0 Implementation | External Dep |
|-----------|-------------------|-------------|
| Broca | `edge-tts` or `pyttsx3` CLI | None |
| Sim Room | `scipy.signal.fftconvolve` with room IR | scipy |
| Cerebellum | `pcm_to_mels` (already done) | librosa |
| Thalamus | spectral subtraction gate (stub) | numpy |
| STG | `faster-whisper-large-v3-turbo` | faster-whisper |
| PFC | simple async coordinator | asyncio |
| SAG | resolve() ASR tokens → nodes | ✅ done |
| EfferenceBuffer | ring buffer for TTS PCM | ✅ done |

## Implementation Order

### Step 1: Broca M0 — TTS Generation
```python
# packages/broca/tts.py
async def synthesize(text: str, sample_rate: int = 16000) -> np.ndarray:
    """Generate speech using edge-tts. Returns float32 PCM."""
    ...
```
- Use `edge-tts` (free, no API key) with `asyncio`
- Resample to 16kHz mono
- **Test:** 1k samples of known text in TTS, output shape matches

### Step 2: Simulated Room Acoustics
```python
# packages/cochlea/room.py  
def apply_room_ir(audio: np.ndarray, ir: np.ndarray = None) -> np.ndarray:
    """Convolve audio with room impulse response.
    
    Default IR: synthetic small room reverb (RT60=0.3s).
    """
    ...
```
- Generate synthetic IR (exponential decay + comb filters)
- Use `scipy.signal.fftconvolve`
- Add optional noise floor
- **Test:** impulse response preserves energy, room reverb adds expected decay

### Step 3: STG M0 — ASR Integration
```python
# packages/stg/asr.py
async def transcribe(audio: np.ndarray) -> list[dict]:
    """Transcribe audio with faster-whisper. Returns segments."""
    ...
```
- Wrap faster-whisper (CTranslate2 backend, works on Apple Silicon)
- Parse segment output into `{text, start, end, confidence}`
- **Test:** known audio file → expected transcript, WER < 10%

### Step 4: Thalamus M0 — Spectral Subtraction Gate
```python
# packages/thalamus/gate.py (extend existing)
def compute_svsr(predicted_mel: np.ndarray, observed_mel: np.ndarray) -> float:
    """Compute Self-Voice Suppression Ratio.
    
    Returns dB of attenuation achievable with efference copy. 
    SVSR = 10 * log10(sum(observed^2) / sum(residual^2))
    """
    ...
```
- Spectral subtraction: `residual = observed_mel - predicted_mel`
- SVSR = how much self-voice is suppressed
- Gate: if SVSR > threshold, residual is "world voice"
- **Test:** identical predicted/observed → infinite SVSR (complete suppression)

### Step 5: PFC M0 — Simulated Loop Coordinator
```python
# packages/pfc/simulated_loop.py
async def run_simulated_loop(utterance: str) -> dict:
    """Full M0 loop: TTS → room → cerebellum predict → ASR → SAG resolve → gate."""
    return {
        "original_text": utterance,
        "asr_transcript": segments,
        "svsr_db": svsr,
        "sag_nodes": resolved_nodes,
        "gated_output": gated_segments,
    }
```
- Orchestrates all components
- Runs synchronously first (no real-time requirements in M0)
- **Test:** "Hello world" → transcript matches, SVSR ≥ 20 dB

### Step 6: Eval Harness
```bash
# scripts/run_m0_eval.sh
# Runs the loop over a test corpus and reports metrics
```
- Test corpus: 20 sentences (mix of lengths, with numbers/acronyms)
- Report: WER, SVSR (dB), self-token rate (should be 0), SAG resolve rate

## Success Criteria (M0 Gate)

| Metric | Target | How Measured |
|--------|--------|-------------|
| TTS output quality | WER < 5% on clean (no room) audio | ASR on direct TTS output |
| Room IR realism | RT60 ≈ 0.3s | Energy decay analysis |
| Efference copy accuracy | SVSR ≥ 20 dB | Predicted vs observed mel diff |
| Self-token rate | 0 tokens in gated output | Gated transcript has no self-voice |
| SAG resolve rate | ≥ 80% of unique words | SAG.resolve() success rate |
| Round-trip time | < 5s per utterance (no real-time req) | Wall-clock timing |

## File Plan

```
packages/
  broca/
    tts.py                 # edge-tts wrapper [new]
  cochlea/
    room.py                # room IR simulation [new]
    capture.py             # ✅ already has EfferenceBuffer + pcm_to_mels
  stg/
    asr.py                 # faster-whisper wrapper [new]
  thalamus/
    gate.py                # ✅ already has basic gate class, extend with SVSR
  pfc/
    simulated_loop.py      # M0 coordinator [new]
tests/
  unit/
    test_broca.py          # [new]
    test_room.py           # [new]
    test_stg.py            # [new]
    test_thalamus_gate.py  # [new]
    test_simulated_loop.py # [new]
  eval/
    test_roundtrip.py      # WER measurement [new]
    test_efference.py      # SVSR measurement [new]
scripts/
  run_m0_eval.sh           # [new]
docs/
  m0_implementation_plan.md # ← this file
```

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `faster-whisper` install issues on Apple Silicon | Test with CPU fallback or whisper.cpp as backup |
| edge-tts network dependency | Include a pre-recorded 3-second sample as test fixture |
| Room IR convolution memory blow-up for long audio | Block-process with overlap-add if needed |
| SVSR threshold calibration | Start with synthetic test (predict == observe); tune with real room IR samples later |

## Timeline Estimate

- Steps 1-3 (Broca + Room + STG): ~2 hours, parallelizable
- Step 4 (Thalamus extension): ~30 min
- Step 5 (Loop coordinator): ~1 hour
- Step 6 (Eval harness): ~1 hour
- Integration + cleanup: ~1 hour

**Total: ~5-6 hours of focused work**

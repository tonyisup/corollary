# Corollary Geometry Sensitivity + Future Work

> Status: Living document — updated 2026-05-03 after M4 proven (197 SELF frames!)
> Context: Night session where M4 became fully operational in M3 embodied loop.

## The Core Insight (2026-05-03)

**Corollary is currently sensitive to geometry/environment changes.**

Tonight's session proved M4 works amazingly (197 SELF frames vs 0 without M4!), but this success is **tied to the current room setup**:

- Your home office
- Current mic/speaker positions
- Your voice as the registered speaker
- Current reverb profile

**Change any of these → performance degrades.**

---

## 1. Current Sensitivities

### A. M4 (Speaker Embeddings — ECAPA-TDNN)

| Factor | Sensitivity | Impact |
|--------|-------------|--------|
| **Room change** | High | ECAPA-TDNN trained on your voice in *this* room. New room = different reverb → embedding drift → false rejections |
| **Mic position** | High | Moving mic 2 feet changes the acoustic path → embedding changes |
| **New speaker** | Critical | `SpeakerIdentifier.register_self()` only knows YOU. Someone else speaks → M4 says "not self" → correct BOTH verdict → agent stays in `LISTEN` |
| **Clothing/health** | Medium | Cold, mask, different microphone → voice timbre shifts slightly |

**Root cause**: M4 uses **fixed registration** (`SpeakerIdentifier.register_self()` called once on your TTS voice). No adaptation.

---

### B. M2 (Timing + Similarity)

| Factor | Sensitivity | Impact |
|--------|-------------|--------|
| **Room reverb** | High | `causal_score` depends on onset alignment. More reverb → longer onset delay → `onset_error_ms` increases → `is_timely = False` → M4 might not even be checked! |
| **similarity_threshold** | Medium | Current `0.05` (room-local oracle) is tuned to YOUR room's cross-correlation profile. New room = different room color → `match_score` drops → `is_self_like = False` |
| **causal_tolerance_ms** | Medium | Set to `175ms` for current room. New room might need `300ms+` |

**Root cause**: M2 uses **fixed thresholds** tuned to current geometry.

---

### C. M3 (Turn-Taking Controller)

| Factor | Sensitivity | Impact |
|--------|-------------|--------|
| **M2 verdict quality** | High | M3's `LISTEN → SPEAK` transitions depend entirely on M2. If M2 drifts → M3 gets wonky |
| **listen_threshold_ms** | Low | Conservative `300ms` works across rooms (agent waits longer before speaking) |
| **cooldown_ms** | Low | `200ms` is safe for most scenarios |

**Root cause**: M3 is only as good as its upstream (M2 + M4).

---

## 2. Concrete Failure Scenario

**Scenario**: You move from home office to living room (more reverb, different layout).

**What happens:**

1. **M4 fails first**:
   - ECAPA-TDNN sees unfamiliar reverb → embedding doesn't match registration
   - `m4_confidence` drops from `0.427` → `0.15` (below `m4_self_threshold=0.10`? Wait, 0.15 > 0.10... but might be unreliable)
   - **Result**: Some SELF frames become BOTH (false world-speech detection)

2. **M2 timing drifts**:
   - Living room has 50ms longer reverb tail
   - `onset_error_ms` = `240ms` (was `188.5ms` in office)
   - `causal_tolerance_ms=175ms` → `is_timely = False`
   - **Result**: M4 might not even be checked (our current logic checks M4 FIRST now, but timing still affects `match_score`)

3. **M3 gets confused**:
   - More BOTH frames → agent thinks "someone else is speaking"
   - Stays in `LISTEN` longer
   - **Result**: Higher latency, missed turns

**Measurable degradation:**
- Office: 197 SELF frames, 200ms latency
- Living room (prediction): ~50 SELF frames, 400ms+ latency

---

## 3. Mitigations + Future Work

### Short-Term (1-2 weeks)

#### A. Re-Registration Script
```bash
# New script: scripts/m4_reregister.py
# Usage: cd ~/Projects/corollary && .venv/bin/python scripts/m4_reregister.py --room "living_room"
```
- Captures your voice in new room
- Re-registers ECAPA-TDNN embedding
- Saves room profile to `data/room_profiles/living_room/`
```

#### B. Auto-Calibration for M2
```bash
# Extend existing: scripts/m2_benchmark.py
# New flag: --auto-calibrate
```
- Plays test tones in new room
- Measures actual onset delays
- Adjusts `causal_tolerance_ms` automatically
- Tunes `similarity_threshold` for new room color
```

#### C. "Unsure" State for M3
```python
# In packages/pfc/agentive_controller.py
class AgentState(Enum):
    IDLE = 0
    LISTEN = 1
    SPEAK = 2
    UNSURE = 3  # NEW: low-confidence zone
```
- When M2/M4 confidence is low → enter `UNSURE`
- Conservative: stay in `LISTEN`, don't speak
- Log the uncertainty for later analysis
```

---

### Medium-Term (1-2 months)

#### D. Room-Invariant Models
- Train M4 on multi-room data (your voice in 5+ rooms)
- Data augmentation: synthetic reverb, mic position jitter
- Goal: Embedding that works across geometries

#### E. Online Adaptation
- M4 periodically re-registers based on its own high-confidence SELF frames
- "I'm 90% sure this is me → update my embedding"
- Requires careful guardrails (don't adapt to world speech!)

#### F. Geometry-Aware Thresholds
- Store room profiles: `{room_id: {causal_tolerance_ms, similarity_threshold, ...}}`
- Auto-detect room (via initial impulse response measurement?)
- Load appropriate thresholds
---

### Long-Term (3+ months)

#### G. Multi-Speaker Support
- M4 maintains a registry of *known speakers* (you, family, frequent visitors)
- Unknown speaker → special verdict `UNKNOWN_SPEAKER`
- M3 can have different policies per speaker ("family can interrupt me, strangers can't")

#### H. Self-Supervised Calibration
- Corollary runs "calibration mode" for 30 seconds on startup
- Plays test tones, measures room response
- Auto-tunes all parameters
- "I'm ready to talk in this room!"
---

## 4. "Corollary Goes Mobile" — Stress-Test Project

**Idea**: Take the current working system (office setup) and test it in 3-5 different environments.

### Test Matrix

| Environment | Reverb (RT60) | Mic Position | Your Voice | Expected Degradation |
|-------------|-----------------|---------------|------------|--------------------------|
| Home office (baseline) | ~0.3s | Fixed | Yes | 0% (reference) |
| Living room | ~0.6s | Moved 3ft | Yes | Medium |
| Kitchen | ~0.2s | Different device | Yes | Low |
| Outdoor balcony | ~0.1s | Laptop mic | Yes | High (wind noise!) |
| Friend's house | ~0.5s | Unknown setup | Yes | Critical |

### Procedure

```bash
# For each environment:
cd ~/Projects/corollary

# 1. Re-register M4 (if needed)
.venv/bin/python scripts/m4_reregister.py --room "living_room"

# 2. Run M3 embodied loop
.venv/bin/python scripts/m3_back_and_forth_demo.py \
  --turns 5 \
  --m4-enable --m4-min-buffer-ms 60 \
  --broca-room-local-operating-point \
  --save-dir data/mobile_test_living_room/

# 3. Compare results
.venv/bin/python scripts/m4_comparison.py \
  --baseline-dir data/m3_baseline_05 \
  --m4-dir data/mobile_test_living_room/ \
  --output data/mobile_comparison.json
```

### Success Criteria

| Metric | Office (baseline) | Living Room (target) | Kitchen (target) |
|--------|-------------------|----------------------|------------------|
| SELF frames | 197 | >150 (80% retention) | >170 (90% retention) |
| Avg latency | 200ms | <300ms | <250ms |
| Recovery rate | 0%* | >0% (with interruptions) | >0% |

*No interruptions in current demo.

---

## 5. Current Status (2026-05-03 Night)

### ✅ What's Working
- **M4**: 197 SELF frames (proven massive impact!)
- **M3**: Embodied loop with ASR-contingent responses + interruption recovery
- **Docs**: All 14 commits pushed to origin/main

### ⚠️ What's Fragile
- Tied to current geometry (your home office)
- No re-registration script (M4 breaks in new rooms)
- Fixed thresholds (M2 timing tuned to one setup)
- No "unsure" state (M3 can't handle low-confidence scenarios)

### 🚀 Next Session Suggestion
**Option A (Safe)**: Build `m4_reregister.py` + test in 1 new room
**Option B (Ambitious)**: Start "Corollary Goes Mobile" project (3-room test matrix)
**Option C (Different)**: Pivot to multi-speaker support (register family members)

---

## 6. Key Quote to Remember

> *"Corollary works amazingly in this one room. That's not a product — that's a prototype. Let's make it robust."*

---

**Session reset note**: User plans to reset session after this doc is written. Future sessions should:
1. Read this doc (`docs/corollary_geometry_sensitivity.md`)
2. Decide which mitigation to tackle first
3. Continue from commit `cbe8d74` (all tonight's work pushed)

---

**End of doc. Session reset ready! 🚀**

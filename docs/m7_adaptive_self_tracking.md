# M7 — Adaptive Self-Tracking & Multi-Sensor Attribution

> **Date:** 2026-05-09  
> **Status:** Proposal for consideration  
> **Prerequisites:** M4 promotion complete, M3 productization stable

## Motivation

The current self-voice detection path fits the room filter once during calibration and assumes it stays valid. In reality:

- Users move between rooms (impulse response changes)
- On mobile, device orientation and occlusion change the speaker/mic transfer function
- Background noise profiles shift

A single static filter works in a controlled environment but breaks as soon as the acoustic context changes. This doc proposes two complementary paths:

1. **Continuous room adaptation** (audio-only, builds on existing NLMS)
2. **Multi-sensor attribution** (visual priors as cross-modal evidence)

---

## Part 1: Continuous Room Adaptation (Audio-Only)

### Core Insight

The NLMS adaptive filter in `BabblingCerebellum` already does sample-by-sample convergence. The question isn't *how* to adapt — it's *when* to trust the adaptation. If the filter adapts on frames that contain another person's voice, it learns the wrong voice and poisons the self-model.

### Architectural Design

#### Trust-Gated NLMS

Add a gate to the NLMS update rule:

```python
def update_filter(self, observed, predicted, verdict, m4_confidence):
    if verdict == "SELF" and m4_confidence > self.adapt_threshold:
        self.nlms.step(observed, predicted)  # adapt
    elif verdict in ("BOTH", "WORLD"):
        pass  # hold filter steady
```

Key properties:
- **Only adapts on confident SELF frames.** Never on BOTH or WORLD verdicts.
- **Self-match score as the gate.** M4 confidence + thalamic self-match jointly determine "trust this frame."
- **SelfModel detects drift.** A sudden drop in `mean_self_match` or M4 acceptance rate triggers a faster adaptation window — the system recognizes "something changed, I should learn faster."

#### Multi-Hypothesis Room Models

Keep a bank of N recent IR estimates (e.g., 3–5). When the self-match score drops:

1. Score the observed audio against each model in the bank
2. If one model matches significantly better, switch to it
3. If none match, start adapting a new one

This is roughly what humans do — you don't relearn the room impulse response when you walk into your kitchen, you recognize "oh, this sounds like the kitchen" and swap priors.

#### Risk: Adaptation Eating Valid World Speech

This is the core failure mode. If a trusted other speaker's voice sounds enough like self-voice, and the verdict system classifies it as SELF, the filter adapts on their speech and learns to suppress them. Mitigations:

- M4 must pass same-domain negative validation before enabling continuous adaptation
- Cross-modal evidence (Part 2) provides additional guards
- Slow worst-case adaptation rate with exponential decay of old coefficients

**Milestone:** M3.5 / early M7 — fits between M3 productization and the broader M7 amygdala/valence work.

### Mobile-Specific Challenge

On a phone, the device's own transfer function changes with orientation (speaker to face vs. speaker to table vs. phone in pocket). This is a **device-characterization layer** on top of the room-adaptation problem. I'd treat it as a separate sub-problem:

1. First: solve continuous room tracking on a static device
2. Then: add a device-profile layer that models how the phone's acoustic signature changes with orientation

### Phased Plan

| Phase | What | Success Criteria |
|-------|------|-----------------|
| 1 | Trust-gated NLMS (adapt on SELF only, hold on BOTH/WORLD) | ✅ Replay-validated: `scripts/m7_trust_gated_replay.py` shows 46.8% room-B error reduction; WORLD/BOTH produce 0 updates; live drift validation remains next |
| 2 | SelfModel drift alarm (spike adaptation rate when match degrades) | ✅ Implemented: `DriftAlarm` modulates NLMS learning rate; sustained drift triggers rebabble |
| 3 | Multi-hypothesis IR bank (3–5 models, switch on match) | Faster recovery than single-filter re-adaptation |
| 4 | Mobile device-profile layer | Stable self-attribution across phone orientations |

---

## Part 2: Multi-Sensor Attribution (Visual Priors)

### Core Insight

Audio-only attribution is fundamentally ambiguous when two voices overlap or when the self-voice detector is unsure. A visual channel breaks this ambiguity: when the camera confirms "Tony is present and facing the device," the audio attribution prior should shift. When it detects "unknown face speaking," that's an agentive other, not WORLD noise.

### Architectural Design

#### Visual Evidence Interface

Add a lightweight sensor channel that runs on the same frame timeline as audio attribution:

```python
@dataclass
class VisualEvidence:
    known_faces: list[str]       # ["tony", "sarah"]
    face_count: int              # number of faces detected
    mouth_motion: bool           # is the closest face's mouth moving?
    device_orientation: str      # portrait, landscape, pocket, face-down
    head_pose_angle: float       # degrees from camera axis

    def evidence_for(self, speaker_id: str) -> float
    """Return [0, 1] confidence that speaker_id is the active speaker."""
```

#### Integration Points

1. **M4 SelfVoiceDetector with visual priors.** When `VisualEvidence` is available, the detector incorporates it as an additional feature or gating signal. Example: known face present + mouth motion detected → boost M4 confidence for that speaker's embedding match.

2. **SelfModel learns cross-modal patterns.** SelfModel already tracks running statistics per session. With visual evidence, it accumulates:
   - "When Tony's face is detected, M4 accepts at 0.97"
   - "When no face is detected, M4 accepts at 0.72"
   
   This reveals *when* the self-voice detector is fragile — structural information about the system's own vulnerability.

3. **Tiebreaker mode (privacy-efficient).** Run face recognition only when audio attribution is uncertain (both score is high, M4 confidence is low). This keeps processing budget manageable and addresses consent — the camera doesn't stream continuously, it activates on ambiguity.

4. **Walking-around scenario.** On mobile:
   - Face detected + mouth motion + speaker attribution = "user is speaking"
   - No face + voice from device speaker direction = "someone else is in the room"
   - Pocket/orientation detected = "trust audio more aggressively since there's no competing visual evidence"

#### Practical Concerns

- **Privacy / battery.** Continuous face recognition burns power and raises consent questions. Tiebreaker mode (activate only on audio uncertainty) is the right default.

- **"Who is talking when multiple faces present."** Face detection tells you faces exist, not which one produces the audio stream. Solutions:
  - Lip-sync alignment (detect mouth motion + correlate with audio onset)
  - Spatial audio (voice direction relative to camera position)
  - Both — combine for robustness

- **Lighting / occlusion.** Mobile cameras deal with partial faces, bad lighting, hands covering the frame. The system needs graceful degradation when visual evidence is absent or unreliable.

**Milestone:** M5 territory — you need the multi-speaker world model to already distinguish "agentive vocal others" from environment noise before face recognition adds value.

### Sensible First Step

Add the `VisualEvidence` stub to the `SelfVoiceDetector` interface. Wire it in mock mode. Run the M4 F1 benchmark with `visual_evidence` passed through. If mock-mode shows it moving the needle on confusing cases (particularly the same-domain negative cases), invest in the real implementation. If not, you've saved the integration cost.

---

## Combined Architecture

When both paths are active, the attribution flow looks like:

```
                    ┌─────────────────┐
Audio ──────────────│ Thalamic        │─── verdict (SELF/WORLD/BOTH)
                    │ Attributor      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
Visual ─────────────│ SelfVoice       │─── M4 outcome + confidence
                    │ Detector        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
NLMS Filter ────────│ Trust Gate      │─── update filter iff verdict=SELF
(IR estimation)     │                 │     and M4 confidence > threshold
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ SelfModel       │─── accumulate statistics, detect drift,
                    │ (SAG observer)  │     trigger multi-hypothesis switch
                    └─────────────────┘
```

The critical path through this chain is the **trust gate** — it's what prevents the system from learning the wrong thing. Every other component either feeds into it or monitors its output.

## Open Questions

1. **What's the right adaptation decay rate?** Too slow and the system can't keep up with room changes. Too fast and it incorporates noise. This is an empirical question — the M4 F1 benchmark should sweep it.

2. **How many room models in the bank?** 3–5 is a guess. More models give coverage but increase scoring cost and risk of false switches.

3. **Should visual evidence be optional or required?** On desktop, it's always available (webcam on). On mobile, it may not be. The interface should degrade gracefully.

4. **When does the system know it's walking vs. standing still?** Gyroscope + accelerometer could inform the device-profile layer — "orientation is changing rapidly, expect transfer function drift." This is mobile-specific and probably needs dedicated profiling.

## References

- `docs/plan.md` — M7: Amygdala / Valence / Fast-Track Responses
- `packages/pfc/self_model.py` — M6 SelfModel (passive observer, drift signal source)
- `scripts/m4_f1_benchmark.py` — F1 benchmark harness (evaluation substrate for both paths)
- `packages/cerebellum/` — BabblingCerebellum with NLMS adaptive filter

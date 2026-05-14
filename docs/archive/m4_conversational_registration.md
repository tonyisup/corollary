# M4 Conversational Registration

> **Date:** 2026-05-13
> **Status:** Complete

## Insight

Instead of a separate training/registration phase, registration happens organically through the first few turns of conversation — the "small talk" when strangers meet.

## Concept

1. Corollary opens with canned elicitation questions: "How are you?", "What's your name?", "Tell me about your day"
2. These are known TTS utterances (self-attributed), so the system knows it's Corollary speaking
3. Human responses accumulate as WORLD/BOTH audio
4. Once enough clean human speech is collected (~2-5 seconds), extract an embedding and register as a new speaker
5. The conversation continues with the human now identified — per-speaker adaptive thresholding, SAG updates, etc. all work naturally

## Advantages

- No separate "training mode" — registration is part of the conversation flow
- Mirrors how humans learn who they're talking to in the first few seconds
- More natural UX: the system "gets to know" the user through interaction

## Implementation

Integrated into `StreamingRuntime._update_speaker_id()`. Set `enable_conversational_registration=True` in `StreamingConfig`.

**How it works:**
1. During the streaming loop, WORLD audio is accumulated in `_speaker_id_buffer`
2. Every 20 VAD frames, the system extracts an embedding and checks for matches
3. If no existing speaker matches and >=0.3s of WORLD speech is accumulated, auto-registers as new speaker
4. Registration happens once per conversation (`_registration_complete` flag)
5. The `speaker_registered` field on `StreamingFrameLog` records when registration occurred

**Test results (3 turns, ~6s each):**
- Turn 1: 14 WORLD frames → registered on frame 179 (end of turn)
- Turn 2: 46 WORLD frames → 21 matched (speaker identified)
- Turn 3: 50 WORLD frames → 41 matched (consistent identification)

**Files:**
- `packages/pfc/streaming_runtime.py` — config fields, registration state, auto-registration logic
- `tests/integration/test_conversational_registration.py` — 6 tests

## Deployment Model

**Scope:** Single user, fixed environment (same mic, same room, same hardware).

**Registration:** Conversational (on-the-fly). System plays elicitation openers, accumulates WORLD audio, auto-registers speaker after >=0.3s of clean speech.

**Real-time:** Every ~20 VAD frames, extracts ECAPA embedding from WORLD/BOTH buffer, compares against registered embeddings via cosine similarity.

**Re-registration triggers (future work):** mic position change, room acoustics change, new user, day-to-day drift.

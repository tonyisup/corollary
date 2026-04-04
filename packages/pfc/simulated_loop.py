"""PFC — M0 Simulated Loop Coordinator.

Orchestrates: TTS → simulated room → cerebellum predict → ASR → thalamus gate → SAG resolve.
No real microphone or speakers — everything runs through synthetic audio.

Usage:
    from pfc.simulated_loop import run_simulated_loop
    result = await run_simulated_loop("Hello, this is a test utterance.")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import re
import numpy as np

from broca.tts import synthesize
from cochlea.capture import pcm_to_mels
from cochlea.room import apply_room_ir, generate_room_ir, estimate_rt60
from cerebellum.forward_model import predict_mel
from stg.asr import transcribe, TranscriptionResult
from thalamus.gate import ThalamicGate, DEFAULT_SVSR_THRESHOLD
try:
    from sag.graph import SemanticAbstractionGraph
    from sag.models import Node
    SAG_AVAILABLE = True
except ImportError:
    SAG_AVAILABLE = False


@dataclass
class LoopResult:
    """Output of one simulated loop iteration."""

    original_text: str
    asr_result: Optional[TranscriptionResult]
    svsr_db: float
    is_world: bool
    gated_text: str
    sag_nodes: List[str]
    roundtrip_seconds: float
    steps: Dict[str, Any] = field(default_factory=dict)

    @property
    def wer(self) -> float:
        """Word Error Rate using sequence matching.
        
        Normalizes punctuation before computing token-level Levenshteins.
        """
        if self.asr_result is None:
            return 1.0

        def _tokens(text: str) -> List[str]:
            return [
                t for t in re.split(r"[\W_]+", text.lower()) if t
            ]

        a = _tokens(self.original_text)
        b = _tokens(self.asr_result.text)

        if not a and not b:
            return 0.0

        # Levenshteins distance on words
        n, m = len(a), len(b)
        # DP table
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                tmp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = tmp

        return dp[m] / max(len(a), len(b))


async def run_simulated_loop(
    utterance: str,
    room_rt60: float = 0.3,
    room_length: float = 1.0,
    svsr_threshold: float = DEFAULT_SVSR_THRESHOLD,
    build_sag: bool = True,
    whisper_model: str = "tiny",
) -> LoopResult:
    """Run a complete M0 simulated self-listening loop.

    Pipeline:
        1. Broca: Synthesize audio from text
        2. Room: Apply simulated room IR
        3. Cerebellum: Predict mel spectrogram from efference copy
        4. STG: Transcribe room audio via ASR
        5. Thalamus: Gate via spectral subtraction (self vs world)
        6. SAG: Resolve ASR tokens to semantic nodes (optional)

    Args:
        utterance: Text to speak.
        room_rt60: Reverberation time in seconds.
        room_length: IR length in seconds.
        svsr_threshold: dB threshold for self-suppression detection.
        build_sag: Whether to build and resolve against a SAG.
        whisper_model: Whisper model size for ASR.

    Returns:
        LoopResult with all metrics and intermediate outputs.
    """
    t_start = time.perf_counter()
    result_steps: Dict[str, Any] = {}

    # ── Step 1: Broca (TTS) ─────────────────────────────────────────
    pcm_tts = await synthesize(utterance)
    result_steps["tts_samples"] = len(pcm_tts)
    result_steps["tts_duration_s"] = len(pcm_tts) / 16000.0

    # ── Step 2: Simulated Room ──────────────────────────────────────
    ir = generate_room_ir(rt60=room_rt60, length=room_length)
    result_steps["room_ir_length"] = len(ir)
    result_steps["room_rt60_estimate"] = estimate_rt60(ir)

    pcm_room = apply_room_ir(pcm_tts, ir)
    result_steps["room_samples"] = len(pcm_room)

    # ── Step 3: Cerebellum (Efference Copy Prediction) ─────────────
    # In M0, the cerebellum should account for the room IR in its prediction.
    # The forward model predicts what the cochlea will hear, which includes
    # room acoustics. Phase 2+ will learn this automatically.
    predicted_pcm = apply_room_ir(pcm_tts, ir)  # Predict what room hears
    predicted_mel = predict_mel(predicted_pcm)
    result_steps["predicted_mel_shape"] = predicted_mel.shape

    # ── Step 4: STG (ASR on Room Audio) ────────────────────────────
    t_before_asr = time.perf_counter()
    asr_result = transcribe(pcm_room, model_size=whisper_model)
    result_steps["asr_duration_s"] = time.perf_counter() - t_before_asr
    result_steps["asr_transcript"] = asr_result.text

    # ── Step 5: Thalamus (Spectral Subtraction Gate) ────────────────
    observed_mel = pcm_to_mels(pcm_room)
    # Match time dimensions: observed mel may be longer than predicted
    # Pad predicted to match
    t_obs, t_pred = observed_mel.shape[0], predicted_mel.shape[0]
    if t_obs > t_pred:
        padded = np.zeros((t_obs, predicted_mel.shape[1]), dtype=np.float32)
        padded[:t_pred] = predicted_mel
        predicted_mel = padded
    elif t_obs < t_pred:
        predicted_mel = predicted_mel[:t_obs]

    gate = ThalamicGate(svsr_threshold=svsr_threshold)
    gate_result = gate.gate(observed_mel, predicted_mel)

    svsr_db = gate_result["svsr"]
    is_world = gate_result["is_world"]
    residual_mel = gate_result["residual"]

    # Gated text: what survived the gate (world voice, if any)
    # For self-voice, SVSR should be high → is_world=False → gated text is empty
    if is_world:
        gated_text = asr_result.text  # Transcribe from residual (todo: inverse mel→text)
    else:
        gated_text = ""  # Self-voice suppressed

    result_steps["svsr_db"] = svsr_db
    result_steps["is_world"] = is_world

    # ── Step 6: SAG (Semantic Resolution) ───────────────────────────
    sag_nodes: List[str] = []
    if build_sag and SAG_AVAILABLE:
        graph = SemanticAbstractionGraph()
        # Populate SAG with words from the utterance as initial nodes
        for word in utterance.lower().split():
            word = word.strip(".,!?;:\"'()[]{}")
            if not word:
                continue
            graph.add_node(Node(
                id=word,
                core=f"Concept: {word}",
                grounding=[word],
            ))

        # Resolve ASR transcript tokens against SAG
        for token in asr_result.text.lower().split():
            token = token.strip(".,!?;:\"'()[]{}")
            if not token:
                continue
            node = graph.resolve(token)
            if node is not None:
                sag_nodes.append(node.id)

        result_steps["sag_node_count"] = len(graph.nodes())
        result_steps["sag_edge_count"] = len(graph.edges())

    roundtrip = time.perf_counter() - t_start

    return LoopResult(
        original_text=utterance,
        asr_result=asr_result,
        svsr_db=svsr_db,
        is_world=is_world,
        gated_text=gated_text,
        sag_nodes=sag_nodes,
        roundtrip_seconds=roundtrip,
        steps=result_steps,
    )

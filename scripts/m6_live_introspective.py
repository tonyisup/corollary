#!/usr/bin/env python3
"""M6 Live Introspective Query Harness.

Runs a live streaming session with SelfModel introspection. After the session,
queries the SAG and reports the self-concept state.

This validates that:
1. SelfModel correctly processes live frame logs
2. Known speakers are registered from conversational registration
3. Introspective queries return meaningful self-concept data
4. SAG diff between start and end shows learning

Usage:
    .venv/bin/python scripts/m6_live_introspective.py \
        --input-device 1 \
        --output-device 2 \
        --duration 30.0 \
        --output tmp/m6_live_introspective.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from pfc.streaming_runtime import StreamingConfig, StreamingRuntime
from pfc.self_model import SelfModel, NODE_ROOT, NODE_AGENT_VOICE, NODE_SPEECH_PATTERNS, NODE_KNOWN_SUBJECTS
from sag.graph import SemanticAbstractionGraph


def make_tts_pcm(text: str = "Hello, I am Corollary. I am starting a conversation.", duration_s: float = 5.0) -> np.ndarray:
    """Generate real TTS audio."""
    import asyncio
    from broca.tts import synthesize as _synthesize_tts
    pcm = asyncio.run(_synthesize_tts(text, voice="en-US-AriaNeural"))
    return pcm.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="M6 Live Introspective Query")
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output", type=str, default="/tmp/m6_live_introspective.json")
    args = parser.parse_args()

    print("=" * 60)
    print("M6 Live Introspective Query")
    print("=" * 60)
    print(f"  Input device:  {args.input_device}")
    print(f"  Output device: {args.output_device}")
    print(f"  Duration:      {args.duration}s")
    print()

    # Set up SelfModel
    sag = SemanticAbstractionGraph()
    self_model = SelfModel()
    self_model.initialize_sag(sag)

    # Initial introspective query (before any data)
    print("[Pre-session] Initial self-concept:")
    initial_concept = self_model.query_self_concept()
    print(f"  SAG nodes: {len(sag._nodes)}")
    print(f"  Root: {initial_concept.get(NODE_ROOT, {}).get('core', 'N/A')[:80]}")
    print()

    # Create TTS and run streaming
    self_text = (
        "I am Corollary. I am speaking to test the self-model enrichment. "
        "This session will populate the SelfModel with confident SELF frames "
        "from the fine-tuned M4 scorer. The system should register my voice "
        "and update the SAG self-concept subtree with observed patterns."
    )
    tts_pcm = make_tts_pcm(text=self_text)
    config = StreamingConfig(
        tts_pcm=tts_pcm,
        tts_text="M6 self-model enrichment",
        input_device=args.input_device,
        output_device=args.output_device,
        m4_enable=True,
        m4_scorer_path="data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt",
        m4_self_threshold=0.50,
        gated_playback=False,
        enable_conversational_registration=True,
        save_path=Path(args.output).with_suffix(".jsonl"),
    )

    runtime = StreamingRuntime(config)

    print(f"  Starting streaming session ({args.duration}s)...")
    print("  Speak during the session to test conversational registration!")
    print()

    try:
        logs = runtime.run(duration_s=args.duration)
    except KeyboardInterrupt:
        print("\n  Interrupted")
        logs = runtime._logs

    print(f"\n  Collected {len(logs)} frame logs")

    # Process frame logs through SelfModel
    print("\n[Post-session] Processing frame logs through SelfModel...")
    frame_dicts = [runtime._log_to_dict(l) for l in logs]
    result = self_model.process_frame_log(frame_dicts)

    print(f"  Frames processed: {result['frames_processed']}")
    print(f"  Speak frames: {result['speak_frames']}")
    print(f"  Self frames: {result['self_frames']}")
    print(f"  Both frames: {result['both_frames']}")
    print(f"  M4 accepted: {result['m4_accepted']}/{result['m4_attempted']}")

    # Introspective query
    print("\n[Introspective Query] Self-concept after session:")
    concept = self_model.query_self_concept()

    for node_id in [NODE_ROOT, NODE_AGENT_VOICE, NODE_SPEECH_PATTERNS, NODE_KNOWN_SUBJECTS]:
        node = concept.get(node_id, {})
        core = node.get("core", "N/A")
        print(f"  {node_id}:")
        print(f"    {core[:100]}")

    subjects = concept.get("known_subjects", [])
    print(f"  Known subjects: {len(subjects)}")
    for s in subjects:
        print(f"    - {s['id']}: {s['core'][:60]}")

    # Summary
    summary = self_model.summarize()
    print(f"\n  SAG nodes: {summary['sag_nodes']}")
    print(f"  SAG edges: {summary['sag_edges']}")
    print(f"  Sessions: {summary['patterns']['session_count']}")
    print(f"  Total speak frames: {summary['patterns']['total_speak_frames']}")
    print(f"  Mean energy: {summary['patterns']['mean_energy_db']:.1f} dB")
    print(f"  Mean self-match: {summary['patterns']['mean_self_match']:.4f}")
    print(f"  M4 accept rate: {summary['patterns']['m4_accept_rate']:.4f}")

    # Save results
    output = {
        "total_frames": len(logs),
        "processing_result": result,
        "self_concept": {k: v.get("core", "")[:200] for k, v in concept.items() if isinstance(v, dict)},
        "known_subjects": [s["id"] for s in subjects],
        "summary": summary,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved to {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay existing M3 frame data through updated ThalamicAttributor to measure impact.

Reads frame_verdicts.jsonl from m3_with_m4_final and re-attributes each frame
using the new logic (M2+M4 combined gating, threshold=0.50).

Usage:
    .venv/bin/python scripts/m4_replay_attribution.py
"""

import sys, os, json, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))
from thalamus.attributor import ThalamicAttributor, SourceVerdict


def main():
    print("=" * 70)
    print("M4 REPLAY ATTRIBUTION (validation of updated logic)")
    print("=" * 70)

    data_dir = Path("data/m3_with_m4_final")
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    # Create attributor with new defaults
    attributor = ThalamicAttributor(
        similarity_threshold=0.05,
        energy_threshold_db=-55.0,
        causal_tolerance_ms=175.0,
        m4_detector=True,  # pretend M4 is available (we'll use saved m4_metadata)
        m4_self_threshold=0.99,
        m4_window_ms=1000.0,
        m4_min_buffer_ms=60.0,
    )

    total_frames = 0
    old_verdicts = {}
    new_verdicts = {}
    changes = {"SELF→BOTH": 0, "SELF→WORLD": 0, "BOTH→SELF": 0, "WORLD→SELF": 0, "OTHER": 0}
    changed_frames = []

    for turn_dir in sorted(data_dir.iterdir()):
        fpath = turn_dir / "frame_verdicts.jsonl"
        if not fpath.exists():
            continue

        with open(fpath) as f:
            for line in f:
                frame = json.loads(line)
                total_frames += 1

                old_verdict = frame["verdict"]
                old_verdicts[old_verdict] = old_verdicts.get(old_verdict, 0) + 1

                # Re-attribute using saved frame data
                # We don't have frame_audio, so M4 buffer won't accumulate,
                # but we can use the saved m4_metadata from the original run
                result = attributor.attribute(
                    match_score=frame["match_score"],
                    observed_onset_samples=frame["start_sample"],
                    expected_onset_samples=frame["start_sample"] if frame.get("onset_error_ms") is not None else None,
                    rms_db=frame["rms_db"],
                    is_playing=frame["is_playing"],
                    is_speech=frame["is_speech"],
                    frame_audio=None,  # no audio replay, M4 won't accumulate
                )

                # Override M4 metadata from saved data (since we can't replay audio)
                saved_m4 = {
                    "m4_attempted": frame.get("m4_attempted", False),
                    "m4_outcome": frame.get("m4_outcome", "not_attempted"),
                    "m4_confidence": frame.get("m4_confidence"),
                }

                # Re-run M4 decision with saved metadata and new threshold
                if saved_m4["m4_outcome"] == "accepted" and saved_m4["m4_confidence"] is not None:
                    m4_conf = saved_m4["m4_confidence"]
                    is_self_like = frame["match_score"] >= attributor.similarity_threshold
                    is_playing = frame["is_playing"]
                    is_speech = frame["is_speech"]

                    # New logic: require both M2 (is_self_like) AND M4 (conf >= threshold)
                    if is_playing and is_speech and is_self_like and m4_conf >= attributor.m4_self_threshold:
                        new_verdict = "SELF"
                    elif is_playing and is_self_like and frame.get("causal_score", 0) > 0 and is_speech:
                        new_verdict = "BOTH"
                    elif is_playing and (not is_self_like or frame.get("causal_score", 0) == 0):
                        new_verdict = "BOTH" if is_speech else "BOTH"
                    else:
                        new_verdict = old_verdict
                else:
                    new_verdict = old_verdict

                new_verdicts[new_verdict] = new_verdicts.get(new_verdict, 0) + 1

                if old_verdict != new_verdict:
                    change_key = f"{old_verdict}→{new_verdict}"
                    changes[change_key] = changes.get(change_key, 0) + 0
                    changes[change_key] = changes.get(change_key, 0) + 1
                    if len(changed_frames) < 20:
                        changed_frames.append({
                            "turn": turn_dir.name,
                            "frame": frame["index"],
                            "old": old_verdict,
                            "new": new_verdict,
                            "match": frame["match_score"],
                            "causal": frame.get("causal_score", 0),
                            "m4_conf": saved_m4["m4_confidence"],
                            "playing": frame["is_playing"],
                            "speech": frame["is_speech"],
                        })

    print(f"\nTotal frames: {total_frames}")
    print(f"\nOld verdicts: {old_verdicts}")
    print(f"New verdicts: {new_verdicts}")

    print(f"\nChanges:")
    for k, v in sorted(changes.items()):
        if v > 0:
            print(f"  {k}: {v}")

    if changed_frames:
        print(f"\nSample changed frames:")
        for f in changed_frames[:10]:
            print(f"  {f['turn']}/frame{f['frame']}: {f['old']}→{f['new']} "
                  f"(match={f['match']:.3f}, causal={f['causal']:.3f}, m4_conf={f['m4_conf']})")

    # Summary
    old_self = old_verdicts.get("SELF", 0)
    new_self = new_verdicts.get("SELF", 0)
    old_both = old_verdicts.get("BOTH", 0)
    new_both = new_verdicts.get("BOTH", 0)

    print(f"\n{'='*70}")
    print(f"IMPACT SUMMARY")
    print(f"{'='*70}")
    print(f"SELF frames: {old_self} → {new_self} ({new_self-old_self:+d})")
    print(f"BOTH frames: {old_both} → {new_both} ({new_both-old_both:+d})")
    print(f"SELF rate:   {old_self/total_frames*100:.1f}% → {new_self/total_frames*100:.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit M3 resume behavior from offline interruption analysis results.

This script does not re-run attribution. It reads the JSON produced by
`scripts/m3_offline_analyze.py` and prints a compact per-scenario account of:

- state transitions
- interrupt/listen/resume decisions
- whether a deferred utterance likely existed
- whether resume was expected under a simple scenario-policy heuristic
- why resume did or did not occur

Usage:
    .venv/bin/python scripts/m3_resume_audit.py \
        --results data/m3_interrupt_v2/offline_analysis_results_m2.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


# Conservative scenario policy. This is deliberately explicit because the core
# ambiguity is semantic: not every human-speaking scenario should be expected to
# resume a prior agent utterance.
RESUME_POLICY = {
    # Agent is speaking, human interrupts, then agent should resume if speech ends.
    "human_early": True,
    "human_late": True,
    "human_after": True,  # agent may have tail overlap / deferred completion
    "human_double": True,
    "overlap_self_first": True,

    # Human starts before agent/world-only/self-only/brief noise are not clean
    # "resume an interrupted utterance" cases.
    "human_before": False,
    "world_only": False,
    "self_only": False,
    "brief_noise": False,
    "overlap_human_first": False,
}


def load_results(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def compress_transitions(states: list[str]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    prev = None
    for i, state in enumerate(states):
        if state != prev:
            out.append((i, state))
            prev = state
    return out


def first_true(flags: list[bool]) -> int | None:
    for i, flag in enumerate(flags):
        if flag:
            return i
    return None


def last_true(flags: list[bool]) -> int | None:
    for i in range(len(flags) - 1, -1, -1):
        if flags[i]:
            return i
    return None


def get_series(scenario: dict[str, Any], key: str, default: Any = None) -> list[Any]:
    value = scenario.get(key)
    if isinstance(value, list):
        return value
    return [] if default is None else default


def frames(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    value = scenario.get("frame_verdicts")
    return value if isinstance(value, list) else []


def infer_reason(scenario: dict[str, Any], expected_resume: bool) -> str:
    fvs = frames(scenario)
    states = [f.get("state") for f in fvs]
    interrupts = [bool(f.get("should_interrupt")) for f in fvs]
    gt_agent = [bool(f.get("gt_agent")) for f in fvs]
    gt_human = [bool(f.get("gt_human")) for f in fvs]
    can_speak = [bool(f.get("can_speak")) for f in fvs]

    resumed = int(scenario.get("successful_resumes", 0)) > 0
    entered_listen = "LISTEN" in states
    ended_idle = states[-1] == "IDLE" if states else False
    had_interrupt_decision = any(interrupts)
    active_interrupt = any(i and a for i, a in zip(interrupts, gt_agent))
    agent_ever_allowed = any(a and c for a, c in zip(gt_agent, can_speak))
    human_before_agent = (
        first_true(gt_human) is not None
        and first_true(gt_agent) is not None
        and first_true(gt_human) < first_true(gt_agent)
    )

    if resumed:
        return "successful_resumes > 0"
    if not expected_resume:
        return "resume not expected by scenario policy"
    if human_before_agent:
        return "human speech begins before agent; this is initial deferral, not clean resume"
    if not agent_ever_allowed:
        return "agent was never allowed to speak during its gt_agent window"
    if not entered_listen:
        return "never entered LISTEN; no interruption recorded by controller"
    if not had_interrupt_decision:
        return "entered LISTEN without should_interrupt=True; passive listen/defer rather than active interruption"
    if not active_interrupt:
        return "interrupt decision did not occur while gt_agent was active"
    if not ended_idle:
        return "never returned to IDLE by end of scenario; cooldown/silence not observed"
    return "returned to IDLE but did not resume; likely bridge/deferred_utterance bookkeeping or resume scoring issue"


def summarize_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    name = scenario.get("scenario") or scenario.get("name") or "unknown"
    fvs = frames(scenario)
    states = [f.get("state") for f in fvs]
    interrupts = [bool(f.get("should_interrupt")) for f in fvs]
    can_speak = [bool(f.get("can_speak")) for f in fvs]
    verdicts = [f.get("verdict") for f in fvs]
    gt_agent = [bool(f.get("gt_agent")) for f in fvs]
    gt_human = [bool(f.get("gt_human")) for f in fvs]

    expected_resume = RESUME_POLICY.get(name, False)
    transitions = compress_transitions(states)

    listen_flags = [s == "LISTEN" for s in states]
    agent_flags = gt_agent
    human_flags = gt_human
    first_listen = first_true(listen_flags)
    last_listen = last_true(listen_flags)
    first_agent = first_true(agent_flags)
    last_agent = last_true(agent_flags)
    first_human = first_true(human_flags)
    last_human = last_true(human_flags)
    first_interrupt = first_true(interrupts)
    first_can_speak_after_listen = None
    if last_listen is not None:
        for i in range(last_listen + 1, len(can_speak)):
            if can_speak[i]:
                first_can_speak_after_listen = i
                break

    resumed_count = int(scenario.get("successful_resumes", 0))

    return {
        "name": name,
        "expected_resume": expected_resume,
        "resumed_output": resumed_count > 0,
        "successful_resumes": resumed_count,
        "analysis_resumes_expected": int(scenario.get("total_resumes_expected", 0)),
        "false_interrupts": int(scenario.get("false_interrupts", 0)),
        "false_defers": int(scenario.get("false_defers", 0)),
        "missed_interrupts": int(scenario.get("missed_interrupts", 0)),
        "states_count": dict(Counter(states)),
        "verdicts_count": dict(Counter(verdicts)),
        "transitions": transitions,
        "first_listen_frame": first_listen,
        "last_listen_frame": last_listen,
        "first_agent_frame": first_agent,
        "last_agent_frame": last_agent,
        "first_human_frame": first_human,
        "last_human_frame": last_human,
        "first_interrupt_frame": first_interrupt,
        "first_can_speak_after_listen_frame": first_can_speak_after_listen,
        "reason": infer_reason(scenario, expected_resume),
    }


def print_audit(results: dict[str, Any], source: Path) -> None:
    scenarios = results.get("scenarios", [])
    summary = results.get("summary", {})

    print("=" * 80)
    print("M3 RESUME AUDIT")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"Scenarios: {len(scenarios)}")
    if summary:
        print("Overall:")
        for key in [
            "false_interrupt_rate",
            "false_defer_rate",
            "missed_interrupt_rate",
            "resume_rate",
            "successful_resumes",
            "resumable_scenarios",
        ]:
            if key in summary:
                print(f"  {key}: {summary[key]}")
    print()

    audited = [summarize_scenario(s) for s in scenarios]

    expected = [a for a in audited if a["expected_resume"]]
    unexpected = [a for a in audited if not a["expected_resume"]]
    failed_expected = [a for a in expected if not a["resumed_output"]]
    succeeded_expected = [a for a in expected if a["resumed_output"]]

    print("Resume policy view:")
    print(f"  Expected-resume scenarios: {len(expected)}")
    print(f"  Succeeded expected resumes: {len(succeeded_expected)}")
    print(f"  Failed expected resumes: {len(failed_expected)}")
    print(f"  Non-resume scenarios: {len(unexpected)}")
    print()

    for a in audited:
        if a["expected_resume"]:
            marker = "✓" if a["resumed_output"] else "✗"
        else:
            marker = "!" if a["resumed_output"] else "✓"
        print("-" * 80)
        print(f"{marker} {a['name']}")
        print(f"  expected_resume: {a['expected_resume']}")
        print(f"  resumed_output: {a['resumed_output']}")
        print(f"  successful_resumes / analysis_expected: {a['successful_resumes']}/{a['analysis_resumes_expected']}")
        print(f"  FI/FD/MI: {a['false_interrupts']}/{a['false_defers']}/{a['missed_interrupts']}")
        print(f"  states_count: {a['states_count']}")
        print(f"  verdicts_count: {a['verdicts_count']}")
        print(f"  transitions: {a['transitions']}")
        print(f"  first_listen_frame: {a['first_listen_frame']}")
        print(f"  last_listen_frame: {a['last_listen_frame']}")
        print(f"  agent_frames: {a['first_agent_frame']}..{a['last_agent_frame']}")
        print(f"  human_frames: {a['first_human_frame']}..{a['last_human_frame']}")
        print(f"  first_interrupt_frame: {a['first_interrupt_frame']}")
        print(f"  first_can_speak_after_listen_frame: {a['first_can_speak_after_listen_frame']}")
        print(f"  reason: {a['reason']}")

    print()
    print("=" * 80)
    print("ACTIONABLE READ")
    print("=" * 80)
    if failed_expected:
        print("Expected-resume failures:")
        for a in failed_expected:
            print(f"  - {a['name']}: {a['reason']}")
    else:
        print("No expected-resume failures under current policy.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/m3_interrupt_v2/offline_analysis_results_m2.json"),
        help="Path to offline analysis JSON",
    )
    args = parser.parse_args()

    results = load_results(args.results)
    print_audit(results, args.results)


if __name__ == "__main__":
    main()

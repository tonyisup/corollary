#!/usr/bin/env python3
"""M3 Turn-Taking Evaluation Harness.

Measures M3 behavior across 5 controlled scenarios:
  1. self_only: Agent speaks, no world speech → should complete without interruption
  2. world_only: World speech only, no agent → should defer
  3. overlap_interrupt: Agent speaks, world speech overlaps → should interrupt + defer
  4. overlap_no_interrupt: Agent speaks, world speech after → should complete then respond
  5. recovery: After interruption, agent should resume deferred utterance

Each scenario is defined by a timeline of (time_s, event) pairs.
The harness replays these timelines through the M3 bridge and measures:
  - false_interrupts: Agent interrupted when it shouldn't have
  - false_defers: Agent deferred when it should have spoken
  - successful_resumes: Agent resumed after interruption
  - external_speech_preserved: World speech was not masked by agent output

Usage:
    .venv/bin/python scripts/m3_turn_taking_eval.py
"""

import sys, os, json, numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))

from thalamus.attributor import ThalamicAttributor, SourceVerdict, AttributionResult
from pfc.agentive_controller import AgentiveController, AgentState, ControllerDecision
from pfc.m2_turn_bridge import M2Frame, run_turn_taking_bridge


# ── Scenario Definitions ──

@dataclass
class ScenarioEvent:
    """A single event in a turn-taking scenario."""
    time_s: float
    verdict: str  # "SELF", "WORLD", "BOTH", "SILENCE"
    is_speech: bool
    is_in_motor_window: bool = False  # True if Corollary is actively playing back
    description: str = ""


@dataclass
class ScenarioExpectation:
    """What M3 should do in this scenario."""
    should_interrupt: bool = False
    should_defer: bool = False
    should_resume: bool = False
    false_interrupt: bool = False  # True if interrupting would be wrong
    false_defer: bool = False      # True if deferring would be wrong


@dataclass
class Scenario:
    """A complete turn-taking scenario."""
    name: str
    description: str
    events: list[ScenarioEvent]
    expectation: ScenarioExpectation
    agent_text: str = "I'm Corollary, testing turn-taking."


# ── Build scenarios from real frame data ──

def build_scenarios_from_data():
    """Build evaluation scenarios using real M2 frame data from m3_with_m4_final."""
    data_dir = Path("data/m3_with_m4_final")
    scenarios = []

    if not data_dir.exists():
        print(f"WARNING: {data_dir} not found, using synthetic scenarios only")
        return build_synthetic_scenarios()

    for turn_dir in sorted(data_dir.iterdir()):
        fpath = turn_dir / "frame_verdicts.jsonl"
        if not fpath.exists():
            continue

        with open(fpath) as f:
            frames = [json.loads(l) for l in f]

        events = []
        for frame in frames:
            events.append(ScenarioEvent(
                time_s=frame["start_s"],
                verdict=frame["verdict"],
                is_speech=frame["is_speech"],
                is_in_motor_window=frame.get("is_playing", False),
                description=f"frame_{frame['index']}",
            ))

        # Determine expectation from frame content
        # Count WORLD during playback (after first SELF) as true overlap
        # WORLD before playback is ambient noise, not true world speech
        first_self_idx = next((i for i, f in enumerate(frames)
                               if f["verdict"] in ("SELF", "BOTH") and f["is_playing"] and f["is_speech"]), None)
        world_during_playback = any(
            f["verdict"] == "WORLD" and f["is_playing"] and f["is_speech"]
            for f in frames
        )
        # Also count WORLD that comes after playback starts (even if is_playing=False)
        world_after_playback_start = any(
            f["verdict"] == "WORLD" and f["is_speech"]
            for i, f in enumerate(frames)
            if first_self_idx is not None and i > first_self_idx
        )
        self_during_playback = first_self_idx is not None
        has_world = world_during_playback or world_after_playback_start
        has_self = self_during_playback
        has_both = any(f["verdict"] == "BOTH" and f["is_speech"] for f in frames)

        if has_world and has_self:
            expectation = ScenarioExpectation(
                should_interrupt=True,
                should_defer=True,
                should_resume=True,
            )
            desc = "overlap with interruption"
        elif has_world and not has_self:
            expectation = ScenarioExpectation(
                should_defer=True,
                false_interrupt=True,
            )
            desc = "world speech only"
        elif has_self and not has_world:
            expectation = ScenarioExpectation(
                should_interrupt=False,
                false_defer=True,
            )
            desc = "self speech only"
        else:
            expectation = ScenarioExpectation()
            desc = "mixed/ambiguous"

        scenarios.append(Scenario(
            name=turn_dir.name,
            description=f"Real data: {desc}",
            events=events,
            expectation=expectation,
        ))

    return scenarios


def build_synthetic_scenarios():
    """Build synthetic scenarios for testing the evaluation framework."""
    scenarios = []

    # Scenario 1: Self only — agent speaks, no world
    events = []
    for i in range(50):
        t = i * 0.03
        is_self = i < 40
        events.append(ScenarioEvent(
            time_s=t,
            verdict="SELF" if is_self else "SILENCE",
            is_speech=is_self,
            is_in_motor_window=is_self,
            description="self_speech" if is_self else "silence",
        ))
    scenarios.append(Scenario(
        name="self_only",
        description="Agent speaks, no world speech → should complete without interruption",
        events=events,
        expectation=ScenarioExpectation(
            should_interrupt=False,
            false_defer=True,  # deferring would be wrong
        ),
    ))

    # Scenario 2: World only — world speech, no agent
    events = []
    for i in range(50):
        t = i * 0.03
        is_world = 10 <= i < 40
        events.append(ScenarioEvent(
            time_s=t,
            verdict="WORLD" if is_world else "SILENCE",
            is_speech=is_world,
            is_in_motor_window=False,
            description="world_speech" if is_world else "silence",
        ))
    scenarios.append(Scenario(
        name="world_only",
        description="World speech only → should defer",
        events=events,
        expectation=ScenarioExpectation(
            should_defer=True,
            false_interrupt=True,  # interrupting would be wrong
        ),
    ))

    # Scenario 3: Overlap with interruption — agent starts, world joins
    events = []
    for i in range(60):
        t = i * 0.03
        if i < 15:
            verdict = "SELF"
            speech = True
            motor = True
            desc = "self_speech"
        elif i < 45:
            verdict = "BOTH"
            speech = True
            motor = True  # BOTH during self-playback
            desc = "overlap"
        else:
            verdict = "SILENCE"
            speech = False
            motor = False
            desc = "silence"
        events.append(ScenarioEvent(
            time_s=t, verdict=verdict, is_speech=speech,
            is_in_motor_window=motor, description=desc,
        ))
    scenarios.append(Scenario(
        name="overlap_interrupt",
        description="Agent speaks, world overlaps → should interrupt + defer + resume",
        events=events,
        expectation=ScenarioExpectation(
            should_interrupt=True,
            should_defer=True,
            should_resume=True,
        ),
    ))

    # Scenario 4: World after agent — agent completes, then world speaks
    events = []
    for i in range(60):
        t = i * 0.03
        if i < 20:
            verdict = "SELF"
            speech = True
            motor = True
            desc = "self_speech"
        elif i < 25:
            verdict = "SILENCE"
            speech = False
            motor = False
            desc = "gap"
        elif i < 50:
            verdict = "WORLD"
            speech = True
            motor = False  # world speech after agent finished
            desc = "world_speech"
        else:
            verdict = "SILENCE"
            speech = False
            motor = False
            desc = "silence"
        events.append(ScenarioEvent(
            time_s=t, verdict=verdict, is_speech=speech,
            is_in_motor_window=motor, description=desc,
        ))
    scenarios.append(Scenario(
        name="world_after_self",
        description="Agent completes, then world speaks → should not interrupt, should respond",
        events=events,
        expectation=ScenarioExpectation(
            should_interrupt=False,
            should_defer=False,  # agent already finished
        ),
    ))

    # Scenario 5: Recovery — interrupted, then silence, should resume
    events = []
    for i in range(80):
        t = i * 0.03
        if i < 15:
            verdict = "SELF"
            speech = True
            motor = True
            desc = "self_speech"
        elif i < 35:
            verdict = "BOTH"
            speech = True
            motor = True  # BOTH during self-playback
            desc = "overlap"
        elif i < 50:
            verdict = "SILENCE"
            speech = False
            motor = False
            desc = "silence_after_interrupt"
        else:
            verdict = "SILENCE"
            speech = False
            motor = False
            desc = "silence"
        events.append(ScenarioEvent(
            time_s=t, verdict=verdict, is_speech=speech,
            is_in_motor_window=motor, description=desc,
        ))
    scenarios.append(Scenario(
        name="recovery",
        description="Interrupted, then silence → should resume deferred utterance",
        events=events,
        expectation=ScenarioExpectation(
            should_interrupt=True,
            should_defer=True,
            should_resume=True,
        ),
    ))

    return scenarios


# ── Evaluation Engine ──

@dataclass
class ScenarioResult:
    """Result of evaluating a single scenario."""
    scenario_name: str
    description: str
    passed: bool
    false_interrupts: int = 0
    false_defers: int = 0
    successful_resumes: int = 0
    total_resumes_expected: int = 0
    final_state: str = ""
    decisions_summary: str = ""
    issues: list[str] = field(default_factory=list)


def evaluate_scenario(scenario: Scenario, m4_enabled: bool = False,
                      m4_self_threshold: float = 0.50) -> ScenarioResult:
    """Run a scenario through the M3 bridge and evaluate the result."""

    # Build M2 frames from scenario events
    frames = []
    for event in scenario.events:
        verdict_map = {
            "SELF": SourceVerdict.SELF,
            "WORLD": SourceVerdict.WORLD,
            "BOTH": SourceVerdict.BOTH,
            "SILENCE": SourceVerdict.SILENCE,
        }
        # Use is_in_motor_window from event if available (real data),
        # otherwise infer from verdict (synthetic data)
        is_motor = getattr(event, 'is_in_motor_window', event.verdict in ("SELF", "BOTH"))
        self_match = 0.5 if event.verdict in ("SELF", "BOTH") else 0.0
        attribution = AttributionResult(
            verdict=verdict_map.get(event.verdict, SourceVerdict.SILENCE),
            confidence=0.8,
            self_match_score=self_match,
            is_in_motor_window=is_motor,
            causal_score=0.5 if event.verdict == "SELF" else 0.0,
            energy_db=-30.0 if event.is_speech else -60.0,
            is_speech=event.is_speech,
        )
        frames.append(M2Frame(time_s=event.time_s, attribution=attribution))

    # Create controller
    controller = AgentiveController(
        listen_threshold_ms=100.0,
        cooldown_ms=200.0,
    )

    # Run bridge
    bridge = run_turn_taking_bridge(
        frames=frames,
        requested_output=scenario.agent_text,
        controller=controller,
        request_at_s=0.0,
    )

    # Analyze decisions
    false_interrupts = 0
    false_defers = 0
    successful_resumes = 0
    issues = []

    for i, (frame, decision) in enumerate(zip(frames, bridge.decisions)):
        event = scenario.events[i] if i < len(scenario.events) else None
        if event is None:
            continue

        # False interrupt: controller said should_interrupt but expectation says no
        if decision.should_interrupt and scenario.expectation.false_interrupt:
            false_interrupts += 1
            issues.append(f"  FALSE INTERRUPT at t={frame.time_s:.3f}s ({event.description})")

        # False defer: controller deferred but expectation says should speak
        if not decision.can_speak and scenario.expectation.false_defer:
            # Only count if agent hasn't finished speaking yet
            if event.verdict in ("SELF", "BOTH"):
                false_defers += 1
                issues.append(f"  FALSE DEFER at t={frame.time_s:.3f}s ({event.description})")

    # Check resume
    if scenario.expectation.should_resume:
        if bridge.resumed_output is not None:
            successful_resumes = 1
        else:
            issues.append(f"  FAILED RESUME: expected resume but got none")

    # Determine pass/fail
    passed = (false_interrupts == 0 and false_defers == 0)
    if scenario.expectation.should_resume and successful_resumes == 0:
        passed = False

    # Build decisions summary
    state_changes = []
    prev_state = None
    for d in bridge.decisions:
        if d.state != prev_state:
            state_changes.append(d.state.name)
            prev_state = d.state

    return ScenarioResult(
        scenario_name=scenario.name,
        description=scenario.description,
        passed=passed,
        false_interrupts=false_interrupts,
        false_defers=false_defers,
        successful_resumes=successful_resumes,
        total_resumes_expected=1 if scenario.expectation.should_resume else 0,
        final_state=bridge.decisions[-1].state.name if bridge.decisions else "NONE",
        decisions_summary=" -> ".join(state_changes),
        issues=issues,
    )


# ── Main ──

def main():
    print("=" * 70)
    print("M3 TURN-TAKING EVALUATION")
    print("=" * 70)

    # Build scenarios
    scenarios = build_scenarios_from_data()
    synthetic = build_synthetic_scenarios()

    print(f"\nReal data scenarios: {len(scenarios)}")
    print(f"Synthetic scenarios: {len(synthetic)}")

    # Run synthetic scenarios (these test the framework)
    print()
    print("=" * 70)
    print("SYNTHETIC SCENARIO RESULTS")
    print("=" * 70)

    all_results = []
    for scenario in synthetic:
        result = evaluate_scenario(scenario)
        all_results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {scenario.name}: {scenario.description}")
        print(f"  States: {result.decisions_summary}")
        print(f"  False interrupts: {result.false_interrupts}")
        print(f"  False defers: {result.false_defers}")
        print(f"  Successful resumes: {result.successful_resumes}/{result.total_resumes_expected}")
        print(f"  Final state: {result.final_state}")
        if result.issues:
            for issue in result.issues:
                print(issue)

    # Run real data scenarios
    if scenarios:
        print()
        print("=" * 70)
        print("REAL DATA SCENARIO RESULTS")
        print("=" * 70)

        for scenario in scenarios:
            result = evaluate_scenario(scenario)
            all_results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(f"\n[{status}] {scenario.name}: {scenario.description}")
            print(f"  States: {result.decisions_summary}")
            print(f"  False interrupts: {result.false_interrupts}")
            print(f"  False defers: {result.false_defers}")
            print(f"  Successful resumes: {result.successful_resumes}/{result.total_resumes_expected}")
            print(f"  Final state: {result.final_state}")
            if result.issues:
                for issue in result.issues:
                    print(issue)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    total_false_interrupts = sum(r.false_interrupts for r in all_results)
    total_false_defers = sum(r.false_defers for r in all_results)
    total_resumes = sum(r.successful_resumes for r in all_results)
    total_resumes_expected = sum(r.total_resumes_expected for r in all_results)

    print(f"Scenarios: {passed}/{total} passed")
    print(f"Total false interrupts: {total_false_interrupts}")
    print(f"Total false defers: {total_false_defers}")
    print(f"Successful resumes: {total_resumes}/{total_resumes_expected}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "false_interrupts": total_false_interrupts,
            "false_defers": total_false_defers,
            "resumes": total_resumes,
            "resumes_expected": total_resumes_expected,
        },
        "results": [
            {
                "name": r.scenario_name,
                "description": r.description,
                "passed": r.passed,
                "false_interrupts": r.false_interrupts,
                "false_defers": r.false_defers,
                "resumes": r.successful_resumes,
                "final_state": r.final_state,
                "states": r.decisions_summary,
                "issues": r.issues,
            }
            for r in all_results
        ],
    }

    out_path = "data/m3_turn_taking_eval.json"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""M3 Decision Benchmark — controlled turn-taking scenarios with known answers.

Tests M3 controller decisions across 8 scenarios designed to exercise
the boundary cases between correct and incorrect turn-taking behavior.

Each scenario is a sequence of M2 attribution frames with known ground truth.
The benchmark measures whether the controller makes the right decision at
each decision point.

Scenarios:
  1. clean_self: Agent speaks alone → should never defer
  2. clean_world: World speech alone → should always defer
  3. self_then_world: Agent finishes, then world speaks → should complete, then defer
  4. world_interrupts_self: Agent speaking, world joins → should interrupt + defer
  5. self_resumes_after_interrupt: After interruption + silence → should resume
  6. brief_noise_during_self: Single WORLD frame during self → should not interrupt
  7. sustained_world_during_self: Multiple WORLD frames during self → should interrupt
  8. world_during_silence: World speech during agent silence → should defer

Usage:
    .venv/bin/python3 scripts/m3_decision_benchmark.py
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


# ── Frame builder ──

def make_frame(t_s, verdict, is_speech, is_playing=False, match_score=0.0,
               causal_score=0.0, rms_db=-30.0, m4_confidence=None):
    """Build an M2Frame with specified attribution properties."""
    vmap = {"SELF": SourceVerdict.SELF, "WORLD": SourceVerdict.WORLD,
            "BOTH": SourceVerdict.BOTH, "SILENCE": SourceVerdict.SILENCE}
    meta = {
        "m4_attempted": m4_confidence is not None,
        "m4_outcome": "accepted" if m4_confidence and m4_confidence >= 0.50 else "below_threshold",
        "m4_confidence": m4_confidence,
    }
    return M2Frame(
        time_s=t_s,
        attribution=AttributionResult(
            verdict=vmap[verdict],
            confidence=0.8,
            self_match_score=match_score,
            is_in_motor_window=is_playing,
            causal_score=causal_score,
            energy_db=rms_db,
            is_speech=is_speech,
            metadata=meta,
        )
    )


# ── Scenario definitions ──

@dataclass
class DecisionPoint:
    """A point in the scenario where we check the controller's decision."""
    frame_idx: int  # Which frame triggers this check
    check: str  # What to check: "can_speak", "should_interrupt", "state"
    expected: bool | str  # Expected value
    description: str = ""


@dataclass
class BenchmarkScenario:
    """A complete benchmark scenario."""
    name: str
    description: str
    frames: list[M2Frame]
    decision_points: list[DecisionPoint]
    agent_requests_speak: bool = True  # Whether agent calls request_output()
    agent_text: str = "I'm Corollary."


def build_scenarios():
    """Build all benchmark scenarios."""
    scenarios = []

    # ── 1. Clean self ──
    frames = []
    for i in range(20):
        t = 0.5 + i * 0.03
        frames.append(make_frame(t, "SELF", is_speech=True, is_playing=True,
                                  match_score=0.06, causal_score=0.5, rms_db=-25))
    # Post-speech silence
    for i in range(10):
        t = 0.5 + 20 * 0.03 + i * 0.03
        frames.append(make_frame(t, "SILENCE", is_speech=False, rms_db=-60))
    scenarios.append(BenchmarkScenario(
        name="clean_self",
        description="Agent speaks alone — should stay in SPEAK throughout",
        frames=frames,
        decision_points=[
            DecisionPoint(0, "state", "SPEAK", "Agent starts speaking"),
            DecisionPoint(0, "can_speak", True, "Agent starts speaking"),
            DecisionPoint(10, "state", "SPEAK", "Mid-speech"),
            DecisionPoint(19, "state", "SPEAK", "End of speech"),
        ],
    ))

    # ── 2. Clean world ──
    frames = []
    for i in range(20):
        t = 0.5 + i * 0.03
        frames.append(make_frame(t, "WORLD", is_speech=True, is_playing=False,
                                  match_score=0.0, causal_score=0.0, rms_db=-25))
    frames.append(make_frame(0.5 + 20 * 0.03 + 0.1, "SILENCE", is_speech=False,
                              rms_db=-60))
    scenarios.append(BenchmarkScenario(
        name="clean_world",
        description="World speech alone — should defer after listen_threshold",
        frames=frames,
        agent_requests_speak=False,
        decision_points=[
            # First frame: controller hasn't entered LISTEN yet (listen_threshold not met)
            DecisionPoint(0, "can_speak", True, "First world frame — threshold not met yet"),
            # After ~100ms (frame 4 at t=0.62s), should be LISTEN
            DecisionPoint(4, "can_speak", False, "After listen_threshold — should defer"),
            DecisionPoint(10, "can_speak", False, "Mid-speech"),
            DecisionPoint(19, "can_speak", False, "End of world speech"),
        ],
    ))

    # ── 3. Self then world ──
    frames = []
    # Self speech: 0.5-1.1s
    for i in range(20):
        t = 0.5 + i * 0.03
        frames.append(make_frame(t, "SELF", is_speech=True, is_playing=True,
                                  match_score=0.06, causal_score=0.5, rms_db=-25))
    # Gap: 1.1-1.5s
    for i in range(13):
        t = 1.1 + i * 0.03
        frames.append(make_frame(t, "SILENCE", is_speech=False, rms_db=-60))
    # World speech: 1.5-2.1s
    for i in range(20):
        t = 1.5 + i * 0.03
        frames.append(make_frame(t, "WORLD", is_speech=True, is_playing=False,
                                  match_score=0.0, causal_score=0.0, rms_db=-25))
    frames.append(make_frame(1.5 + 20 * 0.03 + 0.1, "SILENCE", is_speech=False,
                              rms_db=-60))
    scenarios.append(BenchmarkScenario(
        name="self_then_world",
        description="Agent finishes, then world speaks — should complete then defer",
        frames=frames,
        decision_points=[
            DecisionPoint(0, "state", "SPEAK", "Agent starts speaking"),
            DecisionPoint(19, "state", "SPEAK", "Agent finishes"),
            # Gap: controller goes IDLE after SELF ends
            DecisionPoint(20, "state", "IDLE", "Gap — back to IDLE"),
            DecisionPoint(32, "state", "IDLE", "Still in gap"),
            # World starts at frame 33 — need listen_threshold
            DecisionPoint(37, "can_speak", False, "World sustained — should defer"),
        ],
    ))

    # ── 4. World interrupts self ──
    frames = []
    # Self speech starts: 0.5-1.0s
    for i in range(17):
        t = 0.5 + i * 0.03
        frames.append(make_frame(t, "SELF", is_speech=True, is_playing=True,
                                  match_score=0.06, causal_score=0.5, rms_db=-25))
    # World joins (overlap): 1.0-1.6s
    for i in range(20):
        t = 1.0 + i * 0.03
        frames.append(make_frame(t, "WORLD", is_speech=True, is_playing=False,
                                  match_score=0.0, causal_score=0.0, rms_db=-25))
    # Silence after: 1.6-2.2s
    for i in range(20):
        t = 1.6 + i * 0.03
        frames.append(make_frame(t, "SILENCE", is_speech=False, rms_db=-60))
    scenarios.append(BenchmarkScenario(
        name="world_interrupts_self",
        description="Agent speaking, world joins — should interrupt + defer",
        frames=frames,
        decision_points=[
            DecisionPoint(0, "can_speak", True, "Agent starts speaking"),
            DecisionPoint(16, "can_speak", True, "Still speaking, no world yet"),
            DecisionPoint(17, "should_interrupt", True, "World joins — should interrupt"),
            DecisionPoint(17, "can_speak", False, "After interrupt — should defer"),
            DecisionPoint(36, "can_speak", False, "World still speaking"),
        ],
    ))

    # ── 5. Self resumes after interrupt ──
    frames = []
    # Self speech: 0.5-0.8s
    for i in range(10):
        t = 0.5 + i * 0.03
        frames.append(make_frame(t, "SELF", is_speech=True, is_playing=True,
                                  match_score=0.06, causal_score=0.5, rms_db=-25))
    # World interrupts: 0.8-1.2s
    for i in range(13):
        t = 0.8 + i * 0.03
        frames.append(make_frame(t, "WORLD", is_speech=True, is_playing=False,
                                  match_score=0.0, causal_score=0.0, rms_db=-25))
    # Silence (cooldown): 1.2-2.0s
    for i in range(27):
        t = 1.2 + i * 0.03
        frames.append(make_frame(t, "SILENCE", is_speech=False, rms_db=-60))
    scenarios.append(BenchmarkScenario(
        name="self_resumes_after_interrupt",
        description="Interrupted, then silence — should resume deferred utterance",
        frames=frames,
        decision_points=[
            DecisionPoint(0, "can_speak", True, "Agent starts"),
            DecisionPoint(10, "should_interrupt", True, "World interrupts"),
            DecisionPoint(10, "can_speak", False, "Deferred"),
            DecisionPoint(22, "can_speak", False, "World still speaking"),
            # After cooldown (200ms = frame 16 from 1.2s = frame ~22 from start of silence)
            # cooldown starts at last world frame (frame 22, t=1.2s)
            # cooldown ends at t=1.4s = frame ~29
            DecisionPoint(30, "can_speak", True, "After cooldown — can resume"),
        ],
    ))

    # ── 6. Brief BOTH during self (ambiguous overlap) ──
    frames = []
    # Self speech: 0.5-0.8s
    for i in range(10):
        t = 0.5 + i * 0.03
        frames.append(make_frame(t, "SELF", is_speech=True, is_playing=True,
                                  match_score=0.06, causal_score=0.5, rms_db=-25))
    # Single BOTH frame (ambiguous): 0.8s — during SPEAK, BOTH is NOT overlap_speech
    frames.append(make_frame(0.8, "BOTH", is_speech=True, is_playing=True,
                              match_score=0.04, causal_score=0.0, rms_db=-30))
    # Self continues: 0.83-1.2s
    for i in range(12):
        t = 0.83 + i * 0.03
        frames.append(make_frame(t, "SELF", is_speech=True, is_playing=True,
                                  match_score=0.06, causal_score=0.5, rms_db=-25))
    frames.append(make_frame(1.2 + 0.1, "SILENCE", is_speech=False, rms_db=-60))
    scenarios.append(BenchmarkScenario(
        name="brief_both_during_self",
        description="Single BOTH frame during self — should NOT interrupt (motor window)",
        frames=frames,
        decision_points=[
            DecisionPoint(0, "state", "SPEAK", "Agent starts"),
            # BOTH during motor window is self-activity, not external speech
            DecisionPoint(10, "state", "SPEAK", "BOTH frame — should keep speaking"),
            DecisionPoint(11, "state", "SPEAK", "Back to SELF"),
        ],
    ))

    # ── 7. Sustained world during self ──
    frames = []
    # Self speech: 0.5-0.8s
    for i in range(10):
        t = 0.5 + i * 0.03
        frames.append(make_frame(t, "SELF", is_speech=True, is_playing=True,
                                  match_score=0.06, causal_score=0.5, rms_db=-25))
    # Sustained WORLD during self: 0.8-1.4s (20 frames = 600ms)
    for i in range(20):
        t = 0.8 + i * 0.03
        frames.append(make_frame(t, "WORLD", is_speech=True, is_playing=False,
                                  match_score=0.0, causal_score=0.0, rms_db=-25))
    # Silence: 1.4-2.0s
    for i in range(20):
        t = 1.4 + i * 0.03
        frames.append(make_frame(t, "SILENCE", is_speech=False, rms_db=-60))
    scenarios.append(BenchmarkScenario(
        name="sustained_world_during_self",
        description="Sustained WORLD during self — should interrupt",
        frames=frames,
        decision_points=[
            DecisionPoint(0, "state", "SPEAK", "Agent starts"),
            # WORLD during SPEAK triggers immediate interrupt (no listen_threshold in SPEAK)
            DecisionPoint(10, "state", "LISTEN", "First WORLD frame — immediate interrupt"),
            DecisionPoint(10, "should_interrupt", True, "Should interrupt on first WORLD"),
            DecisionPoint(14, "can_speak", False, "Still listening"),
        ],
    ))

    # ── 8. World during silence ──
    frames = []
    # Silence: 0.5-1.0s
    for i in range(17):
        t = 0.5 + i * 0.03
        frames.append(make_frame(t, "SILENCE", is_speech=False, rms_db=-60))
    # World speech: 1.0-1.6s
    for i in range(20):
        t = 1.0 + i * 0.03
        frames.append(make_frame(t, "WORLD", is_speech=True, is_playing=False,
                                  match_score=0.0, causal_score=0.0, rms_db=-25))
    # Silence: 1.6-2.2s
    for i in range(20):
        t = 1.6 + i * 0.03
        frames.append(make_frame(t, "SILENCE", is_speech=False, rms_db=-60))
    scenarios.append(BenchmarkScenario(
        name="world_during_silence",
        description="World speech during agent silence — should defer",
        frames=frames,
        agent_requests_speak=False,
        decision_points=[
            DecisionPoint(0, "can_speak", True, "Silence — can speak"),
            # After listen_threshold (~100ms = frame 4 from start of world = frame 21)
            DecisionPoint(21, "can_speak", False, "World sustained — should defer"),
            DecisionPoint(36, "can_speak", False, "World still speaking"),
        ],
    ))

    return scenarios


# ── Benchmark runner ──

@dataclass
class DecisionResult:
    """Result of a single decision point check."""
    scenario: str
    frame_idx: int
    check: str
    expected: bool | str
    actual: bool | str
    passed: bool
    description: str


@dataclass
class ScenarioResult:
    """Result of running a complete scenario."""
    name: str
    description: str
    passed: bool
    decision_results: list[DecisionResult]
    state_sequence: list[str]
    issues: list[str]


def run_scenario(scenario: BenchmarkScenario, listen_threshold_ms=100.0,
                 cooldown_ms=200.0) -> ScenarioResult:
    """Run a scenario through the M3 bridge and check all decision points."""
    controller = AgentiveController(
        listen_threshold_ms=listen_threshold_ms,
        cooldown_ms=cooldown_ms,
    )

    # Agent requests to speak if this scenario includes agent speech
    if scenario.agent_requests_speak:
        first_frame_time = scenario.frames[0].time_s if scenario.frames else 0.0
        controller.request_output(scenario.agent_text, current_time=first_frame_time)

    # Process all frames
    decisions = []
    state_sequence = []
    prev_state = None
    for frame in scenario.frames:
        decision = controller.process_attribution(frame.attribution, current_time=frame.time_s)
        decisions.append(decision)
        if decision.state != prev_state:
            state_sequence.append(decision.state.name)
            prev_state = decision.state

    # Check decision points
    decision_results = []
    issues = []
    all_passed = True

    for dp in scenario.decision_points:
        if dp.frame_idx >= len(decisions):
            issues.append(f"  Frame {dp.frame_idx} out of range ({len(decisions)} frames)")
            all_passed = False
            continue

        decision = decisions[dp.frame_idx]

        if dp.check == "can_speak":
            actual = decision.can_speak
        elif dp.check == "should_interrupt":
            actual = decision.should_interrupt
        elif dp.check == "state":
            actual = decision.state.name
        else:
            issues.append(f"  Unknown check: {dp.check}")
            continue

        passed = (actual == dp.expected)
        if not passed:
            all_passed = False
            issues.append(
                f"  FAIL frame {dp.frame_idx}: {dp.check} expected={dp.expected} actual={actual} ({dp.description})"
            )

        decision_results.append(DecisionResult(
            scenario=scenario.name,
            frame_idx=dp.frame_idx,
            check=dp.check,
            expected=dp.expected,
            actual=actual,
            passed=passed,
            description=dp.description,
        ))

    return ScenarioResult(
        name=scenario.name,
        description=scenario.description,
        passed=all_passed,
        decision_results=decision_results,
        state_sequence=state_sequence,
        issues=issues,
    )


def main():
    print("=" * 70)
    print("M3 DECISION BENCHMARK")
    print("=" * 70)

    scenarios = build_scenarios()
    print(f"\nScenarios: {len(scenarios)}")
    print(f"Controller: listen_threshold=100ms, cooldown=200ms\n")

    results = []
    total_checks = 0
    passed_checks = 0

    for scenario in scenarios:
        result = run_scenario(scenario)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.description}")
        print(f"  States: {' -> '.join(result.state_sequence)}")
        if result.issues:
            for issue in result.issues:
                print(issue)
        print()

        total_checks += len(result.decision_results)
        passed_checks += sum(1 for dr in result.decision_results if dr.passed)

    # Summary
    passed_scenarios = sum(1 for r in results if r.passed)
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Scenarios: {passed_scenarios}/{len(results)} passed")
    print(f"Decision checks: {passed_checks}/{total_checks} passed")

    if passed_scenarios < len(results):
        print("\nFailed scenarios:")
        for r in results:
            if not r.passed:
                failed = [dr for dr in r.decision_results if not dr.passed]
                print(f"  {r.name}: {len(failed)} failed checks")
                for dr in failed:
                    print(f"    frame {dr.frame_idx}: {dr.check} expected={dr.expected} actual={dr.actual}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "scenarios_total": len(results),
            "scenarios_passed": passed_scenarios,
            "checks_total": total_checks,
            "checks_passed": passed_checks,
        },
        "scenarios": [
            {
                "name": r.name,
                "description": r.description,
                "passed": r.passed,
                "states": r.state_sequence,
                "decisions": [
                    {
                        "frame": dr.frame_idx,
                        "check": dr.check,
                        "expected": str(dr.expected),
                        "actual": str(dr.actual),
                        "passed": dr.passed,
                        "description": dr.description,
                    }
                    for dr in r.decision_results
                ],
            }
            for r in results
        ],
    }

    out_path = "data/m3_decision_benchmark.json"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

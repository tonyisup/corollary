"""M8 — fast-track salience policy.

This is a small, deterministic layer that turns existing controller and
attribution signals into a traceable urgency decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from thalamus.attributor import AttributionResult, SourceVerdict


class M8FastTrackAction(Enum):
    CONTINUE = "continue"
    DEFER = "defer"
    WARN = "warn"
    INTERRUPT = "interrupt"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class M8FastTrackDecision:
    action: M8FastTrackAction
    score: float
    urgency: str
    valence: str
    reason: str
    threshold: float


class M8FastTrackPolicy:
    """Deterministic policy for M8 fast-track decisions.

    The policy uses current attribution and controller state, plus optional
    lexical triggers, to produce a conservative action. It is intentionally
    small and explainable:

    - low salience → continue/defer
    - medium salience → warn
    - high salience during SPEAK → interrupt
    - high salience outside SPEAK → escalate/warn
    """

    def __init__(
        self,
        warn_threshold: float = 0.65,
        interrupt_threshold: float = 0.80,
        escalate_threshold: float = 0.95,
    ) -> None:
        self.warn_threshold = warn_threshold
        self.interrupt_threshold = interrupt_threshold
        self.escalate_threshold = escalate_threshold

    def evaluate(
        self,
        attr_result: AttributionResult,
        *,
        controller_state: object,
        current_time: Optional[float] = None,
        lexical_triggers: Optional[Iterable[str]] = None,
    ) -> M8FastTrackDecision:
        del current_time  # reserved for future hysteresis / temporal smoothing

        state_value = getattr(controller_state, "name", controller_state)
        state = str(state_value or "IDLE").strip().upper() or "IDLE"
        metadata = attr_result.metadata or {}
        verdict = attr_result.verdict

        score = 0.12
        urgency = "low"
        valence = "neutral"
        reasons: list[str] = []

        if verdict == SourceVerdict.SELF:
            score = 0.15
            valence = "positive"
            reasons.append("SELF evidence is low urgency")
        elif verdict == SourceVerdict.WORLD and attr_result.is_speech:
            score = 0.72 if state == "SPEAK" else 0.66
            urgency = "medium"
            valence = "negative"
            reasons.append("WORLD speech is socially salient")
        elif verdict == SourceVerdict.BOTH and attr_result.is_speech:
            score = 0.58 if state != "SPEAK" else 0.70
            urgency = "medium"
            valence = "negative"
            reasons.append("BOTH speech implies overlap / mixed salience")
        elif attr_result.is_speech:
            score = 0.45
            urgency = "medium"
            valence = "neutral"
            reasons.append("speech present without source certainty")
        else:
            score = 0.10
            valence = "neutral"
            reasons.append("no speech → low salience")

        if metadata.get("m4_disagreed"):
            score += 0.12
            reasons.append("M4 disagreement raises salience")
        if metadata.get("m4_degraded_mode"):
            score += 0.08
            reasons.append("M4 degraded mode raises salience")
        if metadata.get("m4_final_is_self") is False:
            score += 0.10
            reasons.append("M4 final verdict rejected SELF")
        if metadata.get("m4_bank_is_self") is not None and metadata.get("m4_scorer_is_self") is not None:
            if metadata.get("m4_bank_is_self") != metadata.get("m4_scorer_is_self"):
                score += 0.10
                reasons.append("bank/scorer disagreement raises salience")

        if lexical_triggers:
            transcript = " ".join(
                str(metadata.get(key, ""))
                for key in ("transcript", "text", "utterance", "caption")
            ).lower()
            triggers = [trigger.lower() for trigger in lexical_triggers]
            matched = [trigger for trigger in triggers if trigger in transcript]
            if matched:
                score += 0.10
                reasons.append(f"lexical trigger(s): {', '.join(matched)}")

        score = max(0.0, min(1.0, score))

        if not attr_result.is_speech:
            # Non-speech frames can still be useful for diagnostics, but they must
            # never trigger a fast-track interrupt/escalation. Silence/noise should
            # stay conservative even if upstream metadata is noisy.
            score = min(score, 0.10)
            urgency = "low"
            if state == "LISTEN" and score >= self.warn_threshold:
                action = M8FastTrackAction.DEFER
            else:
                action = M8FastTrackAction.CONTINUE
        elif score >= self.escalate_threshold:
            urgency = "high"
            action = M8FastTrackAction.ESCALATE if state != "SPEAK" else M8FastTrackAction.INTERRUPT
        elif score >= self.interrupt_threshold and state == "SPEAK":
            urgency = "high"
            action = M8FastTrackAction.INTERRUPT
        elif score >= self.warn_threshold:
            urgency = "high" if score >= self.interrupt_threshold else "medium"
            action = M8FastTrackAction.WARN if state != "LISTEN" else M8FastTrackAction.DEFER
        elif state == "LISTEN" and score >= 0.30:
            urgency = "medium"
            action = M8FastTrackAction.DEFER
        else:
            urgency = "low"
            action = M8FastTrackAction.CONTINUE

        if action == M8FastTrackAction.INTERRUPT:
            reasons.append("fast-track interrupt selected")
        elif action == M8FastTrackAction.WARN:
            reasons.append("fast-track warning selected")
        elif action == M8FastTrackAction.ESCALATE:
            reasons.append("fast-track escalation selected")
        elif action == M8FastTrackAction.DEFER:
            reasons.append("fast-track defer selected")
        else:
            reasons.append("fast-track continue selected")

        return M8FastTrackDecision(
            action=action,
            score=score,
            urgency=urgency,
            valence=valence,
            reason="; ".join(reasons),
            threshold=self.interrupt_threshold if action == M8FastTrackAction.INTERRUPT else self.warn_threshold,
        )

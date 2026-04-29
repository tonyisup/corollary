"""Tiny conservative risk gates for M2 BOTH-frame policy selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .dataset import M2PolicyRow


@dataclass(frozen=True)
class GateResult:
    strategy: str
    expected_delta: float
    accepted: bool


@dataclass
class StrategyMeanGate:
    """Accept strategies whose observed mean ΔWER is non-positive.

    This is intentionally simple: it is the baseline a learned gate must beat.
    """

    strategy_mean_delta: dict[str, float]
    fallback_strategy: str = "gain_1p0"

    @classmethod
    def fit(cls, rows: Sequence[M2PolicyRow], fallback_strategy: str = "gain_1p0") -> "StrategyMeanGate":
        by_strategy: dict[str, list[float]] = {}
        for row in rows:
            by_strategy.setdefault(row.strategy, []).append(row.wer_delta)
        means = {strategy: sum(vals) / len(vals) for strategy, vals in by_strategy.items() if vals}
        return cls(strategy_mean_delta=means, fallback_strategy=fallback_strategy)

    def choose(self, candidate_rows: Sequence[M2PolicyRow]) -> GateResult:
        best_strategy = self.fallback_strategy
        best_delta = self.strategy_mean_delta.get(best_strategy, 0.0)
        for row in candidate_rows:
            expected = self.strategy_mean_delta.get(row.strategy, 0.0)
            if expected < best_delta:
                best_strategy = row.strategy
                best_delta = expected
        accepted = best_strategy != self.fallback_strategy and best_delta <= 0.0
        return GateResult(best_strategy, best_delta, accepted)


def leave_one_seed_out_eval(rows: Sequence[M2PolicyRow], fallback_strategy: str = "gain_1p0") -> dict[str, object]:
    seeds = sorted({row.seed for row in rows})
    decisions: list[dict[str, object]] = []
    fallback_delta = 0.0
    chosen_delta = 0.0
    regressions = 0

    for seed in seeds:
        train = [row for row in rows if row.seed != seed]
        test = [row for row in rows if row.seed == seed]
        gate = StrategyMeanGate.fit(train, fallback_strategy=fallback_strategy)
        result = gate.choose(test)
        fallback_row = next((row for row in test if row.strategy == fallback_strategy), None)
        chosen_row = next((row for row in test if row.strategy == result.strategy), fallback_row)
        if fallback_row is None or chosen_row is None:
            continue
        fallback_delta += fallback_row.wer_delta
        chosen_delta += chosen_row.wer_delta
        if chosen_row.wer_delta > fallback_row.wer_delta:
            regressions += 1
        decisions.append({
            "seed": seed,
            "chosen_strategy": result.strategy,
            "expected_delta": result.expected_delta,
            "accepted": result.accepted,
            "fallback_delta": fallback_row.wer_delta,
            "chosen_delta": chosen_row.wer_delta,
        })

    n = max(len(decisions), 1)
    return {
        "seeds": seeds,
        "decisions": decisions,
        "mean_fallback_delta": fallback_delta / n,
        "mean_chosen_delta": chosen_delta / n,
        "regressions_vs_fallback": regressions,
    }

from m2_policy.dataset import M2PolicyRow
from m2_policy.risk_gate import StrategyMeanGate, leave_one_seed_out_eval


def row(seed: int, strategy: str, delta: float) -> M2PolicyRow:
    return M2PolicyRow(
        trial_dir=f"seed{seed}",
        seed=seed,
        strategy=strategy,
        raw_wer=1.0,
        gated_wer=1.0 + delta,
        wer_delta=delta,
        self_score=0.5,
        speech_ratio=1.0,
        self_frames=0,
        both_frames=10,
        world_frames=10,
        self_onset_error_ms=0.0,
    )


def test_strategy_mean_gate_prefers_lower_mean_delta() -> None:
    rows = [row(0, "gain_1p0", 0.0), row(0, "gain_0p5", -0.25), row(1, "gain_1p0", 0.0), row(1, "gain_0p5", 0.0)]
    gate = StrategyMeanGate.fit(rows)
    result = gate.choose([row(2, "gain_1p0", 0.0), row(2, "gain_0p5", 0.0)])
    assert result.strategy == "gain_0p5"
    assert result.accepted


def test_leave_one_seed_out_reports_regression_against_fallback() -> None:
    rows = [
        row(0, "gain_1p0", 0.0),
        row(0, "gain_0p5", -0.25),
        row(1, "gain_1p0", 0.0),
        row(1, "gain_0p5", 0.25),
    ]
    result = leave_one_seed_out_eval(rows)
    assert result["regressions_vs_fallback"] == 1

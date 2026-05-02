#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

def build_arg_string(trial, args):
    params = {
        "nlms_mu": trial.suggest_float("nlms_mu", args.nlms_mu_range[0], args.nlms_mu_range[1]),
        "train_cycles": trial.suggest_int("train_cycles", args.train_cycles_range[0], args.train_cycles_range[1]),
        "search_ms": trial.suggest_float("search_ms", args.search_ms_range[0], args.search_ms_range[1]),
    }
    cmd = [
        sys.executable, str(SCRIPT_DIR / "m4_forward_model_validation.py"),
        "--mode", args.mode,
        "--output-root", args.output_root,
        "--stimulus-kind", args.stimulus_kind,
        "--duration-ms", str(args.duration_ms),
        "--amplitude", str(args.amplitude),
        "--train-seeds", args.train_seeds,
        "--eval-seeds", args.eval_seeds,
        "--nlms-mu", str(params["nlms_mu"]),
        "--train-cycles", str(params["train_cycles"]),
        "--search-ms", str(params["search_ms"]),
        "--input-device", str(args.input_device),
        "--output-device", str(args.output_device),
        "--noise-device", str(args.noise_device),
    ]
    if args.cerebellum_model:
        cmd += ["--cerebellum-model", args.cerebellum_model]
    return cmd, params



def run_trial(cmd, output_root):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    # validator saves a file: output_root/validation_<ts>.json
    out_dir = Path(output_root)
    json_files = sorted(out_dir.glob("validation_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if json_files:
        try:
            return json.loads(json_files[0].read_text())
        except Exception:
            pass
    # fallback: try stdout
    out = result.stdout.strip()
    start = out.find("{")
    end = out.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(out[start:end])
        except json.JSONDecodeError:
            pass
    # print errors for debugging
    if result.stderr:
        import sys
        print("STDERR:", result.stderr, file=sys.stderr)
    return None
def compute_loss(summary):
    dry = summary.get("mean_dry_score", 0.0)
    pred = summary.get("mean_predicted_score", 0.0)
    od = summary.get("mean_dry_onset_error_samples", 0.0)
    op = summary.get("mean_predicted_onset_error_samples", 0.0)
    delta = op - od
    score_gain = pred - dry
    loss = -score_gain + 0.01 * max(0.0, delta)
    return loss, score_gain, delta

def objective(trial, args):
    cmd, params = build_arg_string(trial, args)
    print("\nTrial " + str(trial.number) + ": " + str(params))
    summary_obj = run_trial(cmd, args.output_root)
    if summary_obj is None:
        print("Failed to parse summary")
        return 1e6
    # summary_obj may be full validator output
    summ = summary_obj["summary"] if "summary" in summary_obj else summary_obj
    loss, gain, delta = compute_loss(summ)
    trial.set_user_attr("summary", summary_obj)
    trial.set_user_attr("params", params)
    trial.set_user_attr("score_gain", gain)
    trial.set_user_attr("onset_delta", delta)
    print("  loss=" + str(round(loss,4)) + "  gain=" + str(round(gain,4)) + "  onset_delta=" + str(round(delta,1)))
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="hardware")
    parser.add_argument("--output-root", default="data/m4_forward_model_optimizer")
    parser.add_argument("--stimulus-kind", default="vowel_ah_var_nearfield")
    parser.add_argument("--duration-ms", type=int, default=350)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--eval-seeds", default="3,4")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=5)
    parser.add_argument("--noise-device", type=int, default=4)
    parser.add_argument("--cerebellum-model", default=None)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--nlms-mu-range", nargs=2, type=float, default=[0.05, 0.25])
    parser.add_argument("--train-cycles-range", nargs=2, type=int, default=[2, 8])
    parser.add_argument("--search-ms-range", nargs=2, type=float, default=[50.0, 150.0])
    parser.add_argument("--study-name", default="m4_forward_model_study")
    parser.add_argument("--storage", default="sqlite:///data/m4_forward_model_optimizer_study.db")
    args = parser.parse_args()

    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials)

    print("\n=== Best trial ===")
    best = study.best_trial
    print("  loss=" + str(round(best.value,4)))
    print("  params=" + str(best.params))
    print("  score_gain=" + str(best.user_attrs.get("score_gain", "N/A")))
    print("  onset_delta=" + str(best.user_attrs.get("onset_delta", "N/A")))
    best_summary = best.user_attrs.get("summary", {})
    out_path = Path(args.output_root) / ("best_trial_" + str(best.number) + ".json")
    out_path.write_text(json.dumps(best_summary, indent=2))
    print("Saved best summary to " + str(out_path))

if __name__ == "__main__":
    main()
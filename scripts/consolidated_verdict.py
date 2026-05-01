#!/usr/bin/env python3
import argparse, glob, json, os, sys

def latest(pattern):
    m = sorted(glob.glob(pattern))
    return m[-1] if m else None

def load(p):
    if not p or not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def extract_summary(vobj):
    # If already canonical top-level keys
    canonical = ["mean_dry_score", "mean_predicted_score",
                 "mean_dry_onset_error_samples", "mean_predicted_onset_error_samples",
                 "row_count", "predicted_better_count"]
    if all(k in vobj for k in canonical):
        return {k: vobj[k] for k in canonical}
    # If summary subkey
    if "summary" in vobj:
        s = vobj["summary"]
        if all(k in s for k in canonical):
            return {k: s[k] for k in canonical}
    # If eval_rows present, compute
    if "eval_rows" in vobj:
        rows = vobj["eval_rows"]
        if rows:
            drs = [r.get("dry_score", 0) for r in rows]
            prs = [r.get("predicted_score", 0) for r in rows]
            od = [r.get("dry_onset_error_samples", 0) for r in rows]
            op = [r.get("predicted_onset_error_samples", 0) for r in rows]
            wins = sum(1 for r in rows if r.get("predicted_score", 0) > r.get("dry_score", 0))
            n = len(rows)
            return {
                "mean_dry_score": sum(drs)/n,
                "mean_predicted_score": sum(prs)/n,
                "mean_dry_onset_error_samples": sum(od)/n,
                "mean_predicted_onset_error_samples": sum(op)/n,
                "row_count": n,
                "predicted_better_count": wins,
            }
    return None

def verdict(validator, benchmark, threshold):
    out = {"promoted": False, "gates": {}, "validator": None, "benchmark": None}
    if validator is None:
        out["error"] = "No validator file found."
        return out
    v = validator
    summ = extract_summary(v)
    if summ is None:
        out["error"] = "Could not extract required summary fields from validator."
        out["raw_validator_keys"] = list(v.keys())
        return out

    rc = summ["row_count"]
    pc = summ["predicted_better_count"]
    sd = summ["mean_dry_score"]
    sp = summ["mean_predicted_score"]
    od = summ["mean_dry_onset_error_samples"]
    op = summ["mean_predicted_onset_error_samples"]

    g1 = sp >= sd + threshold
    g2 = op <= od
    g3 = (pc / rc >= 0.5) if rc > 0 else False

    out["gates"] = {
        "score_delta_ok": g1,
        "onset_ok": g2,
        "majority_win_ok": g3,
        "score_delta": sp - sd,
        "onset_delta": op - od,
        "win_ratio": pc / rc if rc > 0 else 0.0,
    }

    base = g1 and g2 and g3
    bench_ok = True
    if benchmark is not None:
        b = benchmark
        dkeys = [k for k in b if "delta" in k.lower() or "improvement" in k.lower()]
        vals = [b[k] for k in dkeys if isinstance(b[k], (int, float))]
        if vals:
            bench_ok = min(vals) >= 0.0
        out["benchmark"] = {"delta_keys": dkeys, "bench_ok": bench_ok}

    out["promoted"] = base and bench_ok
    out["validator"] = summ
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validator", default=None)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--threshold", type=float, default=0.01)
    args = ap.parse_args()

    vp = args.validator or latest("data/m4_forward_model_validation/validation_*.json")
    bp = args.benchmark or latest("data/m2_benchmark/benchmark_*.json")
    vo = load(vp)
    bo = load(bp)
    v = verdict(vo, bo, args.threshold)
    print(json.dumps(v, indent=2))
    if v.get("error"):
        print("", file=sys.stderr)
        print("ERR:", v["error"], file=sys.stderr)
        sys.exit(2)
    if v["promoted"]:
        print("")
        print("=> PROMOTED: model ready for benchmark deployment")
        sys.exit(0)
    else:
        print("")
        print("=> NOT PROMOTED: fix model/geometry or retrain")
        sys.exit(1)

if __name__ == "__main__":
    main()

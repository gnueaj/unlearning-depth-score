#!/usr/bin/env python3
"""Merge two robustness runs (results.json) and recompute summary.

Usage:
  python scripts/merge_robustness_results.py \
    --part0 runs/meta_eval/robustness_part0 \
    --part1 runs/meta_eval/robustness_part1 \
    --out_dir runs/meta_eval/robustness_merged \
    --metrics table2,uds \
    --faithfulness_pn_results runs/meta_eval/combined_faithfulness_v3/results.json \
    --filter_insufficient \
    --filter_utility --utility_epoch runs/ep5 --utility_drop 0.20
"""
import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List

from uds.meta_eval_utils import normalize_metrics_list


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("meta_eval_robustness", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part0", required=True)
    ap.add_argument("--part1", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--metrics", default="uds")
    ap.add_argument("--faithfulness_pn_results", default=None)
    ap.add_argument("--filter_insufficient", action="store_true")
    ap.add_argument("--filter_utility", action="store_true")
    ap.add_argument("--utility_drop", type=float, default=0.20)
    ap.add_argument("--utility_epoch", type=str, default="runs/ep5")
    args = ap.parse_args()

    part0 = Path(args.part0) / "results.json"
    part1 = Path(args.part1) / "results.json"
    if not part0.exists() or not part1.exists():
        raise SystemExit("missing part results.json")

    r0 = json.loads(part0.read_text())
    r1 = json.loads(part1.read_text())
    # merge; keep retain_before/after from part0 if present
    merged = dict(r1)
    merged.update(r0)

    # load helper funcs from robustness script
    mod = load_module(Path("scripts/meta_eval_robustness.py"))

    metrics = normalize_metrics_list(args.metrics.split(","))
    if not metrics:
        metrics = ["uds"]

    # Build filtering thresholds if requested
    filter_thresholds = {}
    if args.filter_insufficient and args.faithfulness_pn_results:
        pn_path = Path(args.faithfulness_pn_results)
        if pn_path.exists():
            filter_thresholds = mod.compute_filtering_thresholds(str(pn_path))

    # Utility filter
    utility_filtered = set()
    if args.filter_utility:
        util_scores = mod.load_utility_scores(args.utility_epoch)
        full_util = util_scores.get("full")
        if full_util is not None:
            for name, _ in merged.items():
                if name in ("retain_before", "retain_after"):
                    continue
                if name not in util_scores:
                    continue
                util_rel = util_scores[name] / (full_util + 1e-12)
                if util_rel < (1.0 - args.utility_drop):
                    utility_filtered.add(name)

    # Aggregate R/Q
    metric_R = {m: [] for m in metrics}
    metric_Q = {m: [] for m in metrics}
    metric_passed = {m: 0 for m in metrics}
    metric_filtered = {m: 0 for m in metrics}
    passed_models = {m: [] for m in metrics}
    filtered_models = {m: [] for m in metrics}

    for name, mr in merged.items():
        if name in ("retain_before", "retain_after"):
            continue
        metrics_before = mr.get("metrics_before", {})
        R = mr.get("relearning_R", {})
        Q = mr.get("quantization_Q", {})

        for m in metrics:
            # utility filter
            if args.filter_utility and name in utility_filtered:
                metric_filtered[m] += 1
                filtered_models[m].append(name)
                continue
            # insufficient filter
            if args.filter_insufficient and filter_thresholds:
                if not mod.model_passes_filter(metrics_before, m, filter_thresholds):
                    metric_filtered[m] += 1
                    filtered_models[m].append(name)
                    continue
                metric_passed[m] += 1
                passed_models[m].append(name)
            else:
                metric_passed[m] += 1
                passed_models[m].append(name)

            if isinstance(R, dict) and R.get(m) is not None:
                metric_R[m].append(R[m])
            if isinstance(Q, dict) and Q.get(m) is not None:
                metric_Q[m].append(Q[m])

    metric_robust = {}
    for m in metrics:
        avg_R = float(mod.np.mean(metric_R[m])) if metric_R[m] else None
        avg_Q = float(mod.np.mean(metric_Q[m])) if metric_Q[m] else None
        if avg_R is not None and avg_Q is not None and avg_R > 0 and avg_Q > 0:
            avg_rob = 2 * avg_R * avg_Q / (avg_R + avg_Q)
        else:
            avg_rob = None
        metric_robust[m] = {"R": avg_R, "Q": avg_Q, "robustness": avg_rob}

    filtering_info = None
    if args.filter_insufficient and filter_thresholds or args.filter_utility:
        filtering_info = {
            "enabled": True,
            "thresholds": filter_thresholds,
            "passed_per_metric": {m: metric_passed[m] for m in metrics},
            "filtered_per_metric": {m: metric_filtered[m] for m in metrics},
            "utility_drop": args.utility_drop if args.filter_utility else None,
            "utility_epoch": args.utility_epoch if args.filter_utility else None,
            "utility_filtered": sorted(list(utility_filtered)) if args.filter_utility else [],
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "metrics": metrics,
        "metric_robustness": metric_robust,
        "n_models": len([k for k in merged if k not in ("retain_before","retain_after")]),
        "filtering": filtering_info,
        "per_model": [],
    }

    (out_dir / "results.json").write_text(json.dumps(merged, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote merged results to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build robustness filtering lists using existing metrics (no re-computation).

Implements Open-Unlearning §4.2.1:
  1) Utility drop > 20% filter (global).
  2) Metric threshold filter from Faithfulness P/N pool (per-metric).

Inputs:
  - P/N pool results.json (split or combined)
  - runs/ep5/* summaries for model metrics

Outputs:
  - filtered_models.json with passed/filtered lists per metric
"""
import argparse
import json
import os
from pathlib import Path


def read_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def compute_thresholds(pn_results: dict):
    # P: keys with /pos_ ; N: keys with /neg_
    p_models = [k for k in pn_results if "/pos_" in k]
    n_models = [k for k in pn_results if "/neg_" in k]
    thresholds = {}
    if not p_models or not n_models:
        return thresholds

    # Union of metrics across P/N (some metrics may be in separate files, e.g., UDS)
    metric_set = set()
    for k in p_models + n_models:
        metric_set.update((pn_results.get(k, {}).get("metrics") or {}).keys())

    for metric in sorted(metric_set):
        def score_fn(x):
            if x is None:
                return None
            return 1 - x if metric == "uds" else x

        p_scores = [score_fn(pn_results[m]["metrics"].get(metric)) for m in p_models
                    if score_fn(pn_results[m].get("metrics", {}).get(metric)) is not None]
        n_scores = [score_fn(pn_results[m]["metrics"].get(metric)) for m in n_models
                    if score_fn(pn_results[m].get("metrics", {}).get(metric)) is not None]
        if len(p_scores) < 2 or len(n_scores) < 2:
            continue
        all_scores = sorted(set(p_scores + n_scores))
        best_t, best_acc = None, -1
        for i in range(len(all_scores) - 1):
            t = (all_scores[i] + all_scores[i + 1]) / 2
            tp = sum(1 for s in p_scores if s >= t)
            tn = sum(1 for s in n_scores if s < t)
            acc = (tp + tn) / (len(p_scores) + len(n_scores))
            if acc > best_acc:
                best_acc = acc
                best_t = t
        thresholds[metric] = best_t
    return thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pn_results", default="",
                    help="Combined P/N results.json (with metrics). If set, skips pos/neg/UDS merging.")
    ap.add_argument("--pos_results", default="runs/meta_eval/table2_faithfulness_v3_gpu0/results.json")
    ap.add_argument("--neg_results", default="runs/meta_eval/table2_faithfulness_v3_gpu1/results.json")
    ap.add_argument("--combined_out", default="runs/meta_eval/combined_faithfulness_v3/results.json")
    ap.add_argument("--uds_pos_results", default="runs/meta_eval/table2_faithfulness_uds_sdpa_gpu0/results.json")
    ap.add_argument("--uds_neg_results", default="runs/meta_eval/table2_faithfulness_uds_sdpa_gpu1/results.json")
    ap.add_argument("--ep_dirs", default="runs/ep5,runs/ep10",
                    help="Comma-separated ep roots (e.g., runs/ep5,runs/ep10)")
    ap.add_argument("--utility_drop", type=float, default=0.20)
    ap.add_argument("--out_dir", default="runs/meta_eval/robustness_filter_list")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pn_results:
        combined = read_json(Path(args.pn_results)) or {}
    else:
        pos = read_json(Path(args.pos_results)) or {}
        neg = read_json(Path(args.neg_results)) or {}
        combined = {**pos, **neg}

        # Merge UDS-only results into combined (if available)
        uds_pos = read_json(Path(args.uds_pos_results)) or {}
        uds_neg = read_json(Path(args.uds_neg_results)) or {}
        uds_combined = {**uds_pos, **uds_neg}
        if uds_combined:
            for k, v in uds_combined.items():
                if k not in combined:
                    combined[k] = v
                else:
                    combined[k].setdefault("metrics", {}).update((v.get("metrics") or {}))
        Path(args.combined_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.combined_out).write_text(json.dumps(combined, indent=2))

    thresholds = compute_thresholds(combined)

    ep_dirs = [e.strip() for e in args.ep_dirs.split(",") if e.strip()]
    models = []
    per_model = {}
    util_rel = {}
    utility_filtered = []

    for ep_dir in ep_dirs:
        ep_root = Path(ep_dir)
        model_list_path = ep_root / "model_list.json"
        if not model_list_path.exists():
            continue
        data = json.load(open(model_list_path))
        if isinstance(data, dict):
            key = "ep5_models" if "ep5" in ep_root.name else "ep10_models"
            ep_models = data.get(key, [])
        else:
            ep_models = data
        ep_models = [m for m in ep_models if m not in ("full", "retain")]
        models.extend(ep_models)

        # utility per epoch
        util_root = ep_root / "utility"
        util_full = read_json(util_root / "full" / "summary.json") or {}
        full_util = util_full.get("utility") or util_full.get("model_utility")
        util_scores = {}
        if util_root.exists():
            for p in util_root.glob("*/summary.json"):
                data = read_json(p)
                if not data:
                    continue
                util = data.get("utility") or data.get("model_utility")
                if util is not None:
                    util_scores[p.parent.name] = util

        # memorization
        mem_root = ep_root / "memorization"
        mem_scores = {}
        if mem_root.exists():
            for p in mem_root.glob("*/summary.json"):
                d = read_json(p) or {}
                mem_scores[p.parent.name] = {
                    "em": d.get("avg_em"),
                    "es": d.get("avg_es"),
                    "prob": d.get("avg_prob"),
                    "paraprob": d.get("avg_paraprob"),
                    "truth_ratio": d.get("avg_truth_ratio"),
                }

        # privacy
        priv_root = ep_root / "privacy"
        priv_scores = {}
        if priv_root.exists():
            for p in priv_root.glob("*/summary.json"):
                d = read_json(p) or {}
                attacks = d.get("attacks", {})
                priv_scores[p.parent.name] = {
                    "mia_loss": attacks.get("loss", {}).get("auc"),
                    "mia_zlib": attacks.get("zlib", {}).get("auc"),
                    "mia_min_k": attacks.get("min_k", {}).get("auc"),
                    "mia_min_kpp": attacks.get("min_k++", {}).get("auc"),
                }

        # UDS
        uds_root = ep_root / "uds"
        uds_scores = {}
        if uds_root.exists():
            for p in uds_root.glob("*/summary.json"):
                d = read_json(p) or {}
                uds_scores[p.parent.name] = d.get("avg_uds") or d.get("avg_udr")

        # gen_rouge metrics (optional)
        rouge_root = ep_root / "gen_rouge"
        rouge_scores = {}
        if rouge_root.exists():
            for p in rouge_root.glob("*/summary.json"):
                d = read_json(p) or {}
                mets = d.get("metrics", {})
                rouge_scores[p.parent.name] = {
                    "rouge": mets.get("rouge"),
                    "para_rouge": mets.get("para_rouge"),
                    "jailbreak_rouge": mets.get("jailbreak_rouge"),
                }

        # merge per-model metrics
        for m in ep_models:
            metrics = {}
            metrics.update(mem_scores.get(m, {}))
            metrics.update(priv_scores.get(m, {}))
            metrics.update(rouge_scores.get(m, {}))
            if m in uds_scores:
                metrics["uds"] = uds_scores[m]
            per_model[m] = metrics

            if full_util is not None and m in util_scores:
                util_rel[m] = util_scores[m] / (full_util + 1e-12)
                if util_rel[m] < (1.0 - args.utility_drop):
                    utility_filtered.append(m)

    # Metric filtering
    passed = {k: [] for k in thresholds.keys()}
    filtered = {k: [] for k in thresholds.keys()}
    missing = {k: [] for k in thresholds.keys()}

    for m in models:
        for metric, t in thresholds.items():
            val = per_model[m].get(metric)
            if val is None:
                missing[metric].append(m)
                continue
            if m in utility_filtered:
                filtered[metric].append(m)
                continue
            # UDS threshold computed on 1-UDS
            if metric == "uds":
                if val >= (1 - t):
                    passed[metric].append(m)
                else:
                    filtered[metric].append(m)
            else:
                if val <= t:
                    passed[metric].append(m)
                else:
                    filtered[metric].append(m)

    out = {
        "combined_results": args.combined_out,
        "thresholds": thresholds,
        "utility_drop": args.utility_drop,
        "utility_filtered": utility_filtered,
        "utility_rel": util_rel,
        "passed_models": passed,
        "filtered_models": filtered,
        "missing_models": missing,
        "counts": {
            "models": len(models),
            "utility_filtered": len(utility_filtered),
        },
    }
    Path(args.out_dir, "filtered_models.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {Path(args.out_dir, 'filtered_models.json')}")


if __name__ == "__main__":
    main()

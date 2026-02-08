#!/usr/bin/env python3
"""
Post-hoc: add 4 individual sMIA metrics to faithfulness and robustness results.

Adds: s_mia_loss, s_mia_zlib, s_mia_min_k, s_mia_min_kpp

where s_mia_x = clip(1 - |auc_model - auc_retain| / |auc_full - auc_retain|, 0, 1)

Reference values from ep10 privacy summaries (privacy_eval.py):
  retain: runs/ep10/privacy/retain/summary.json
  full:   runs/ep10/privacy/full/summary.json

Direction: higher s_mia = less knowledge (like UDS).
  - Faithfulness: use 1-s_mia for AUC-ROC (P-pool should have LOW s_mia)
  - Robustness: use 1-s_mia as metric value (same as UDS direction policy)
"""

import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

# Reference AUC values (from ep10 privacy summaries)
RETAIN_AUC = {
    "loss": 0.38235312499999996,
    "zlib": 0.304609375,
    "min_k": 0.37731562500000004,
    "min_k++": 0.470575,
}
FULL_AUC = {
    "loss": 0.99588125,
    "zlib": 0.99625,
    "min_k": 0.996125,
    "min_k++": 0.9974625,
}

# Raw AUC key → (attack name, s_mia output key)
SMIA_METRICS = {
    "mia_loss":   ("loss",   "s_mia_loss"),
    "mia_zlib":   ("zlib",   "s_mia_zlib"),
    "mia_min_k":  ("min_k",  "s_mia_min_k"),
    "mia_min_kpp": ("min_k++", "s_mia_min_kpp"),
}

# These have inverted direction (higher = less knowledge), like UDS
INVERTED_METRICS = {"s_mia_loss", "s_mia_zlib", "s_mia_min_k", "s_mia_min_kpp"}


def s_mia(auc_model, auc_retain, auc_full):
    """Compute scaled MIA for a single attack."""
    denom = abs(auc_full - auc_retain)
    if denom <= 1e-12:
        return None
    score = 1.0 - abs(auc_model - auc_retain) / denom
    return float(np.clip(score, 0.0, 1.0))


def compute_smia_individual(metrics_dict):
    """Compute 4 individual s_mia values from raw MIA AUC values."""
    result = {}
    for raw_key, (attack, smia_key) in SMIA_METRICS.items():
        raw_auc = metrics_dict.get(raw_key)
        if raw_auc is not None:
            result[smia_key] = s_mia(raw_auc, RETAIN_AUC[attack], FULL_AUC[attack])
    return result


def add_to_faithfulness(results_path, summary_path):
    """Add 4 s_mia metrics to faithfulness results and recompute summary."""
    results = json.loads(results_path.read_text())

    # Add individual s_mia to each model's metrics
    count = 0
    for model_id, info in results.items():
        metrics = info.get("metrics", {})
        smia_vals = compute_smia_individual(metrics)
        metrics.update(smia_vals)
        # Remove old combined s_mia if present
        metrics.pop("s_mia", None)
        if smia_vals:
            count += 1

    print(f"Faithfulness: computed s_mia for {count}/{len(results)} models")

    # Compute summary AUC-ROC for each s_mia metric
    summary = json.loads(summary_path.read_text())
    # Remove old combined s_mia if present
    summary.pop("s_mia", None)

    for smia_key in INVERTED_METRICS:
        p_scores, n_scores = [], []
        p_raw, n_raw = [], []
        for model_id, info in results.items():
            score = info.get("metrics", {}).get(smia_key)
            if score is None:
                continue
            score_flipped = 1 - score  # higher = more knowledge
            if info["pool"] == "P":
                p_scores.append(score_flipped)
                p_raw.append(score)
            else:
                n_scores.append(score_flipped)
                n_raw.append(score)

        if len(p_scores) >= 2 and len(n_scores) >= 2:
            labels = [1] * len(p_scores) + [0] * len(n_scores)
            scores = p_scores + n_scores
            auc = roc_auc_score(labels, scores)
            raw_auc = 1 - auc  # raw direction AUC
        else:
            auc = None
            raw_auc = None

        summary[smia_key] = {
            "auc_roc": auc,
            "raw_auc": raw_auc,
            "p_count": len(p_raw),
            "n_count": len(n_raw),
            "p_mean": float(np.mean(p_raw)) if p_raw else None,
            "n_mean": float(np.mean(n_raw)) if n_raw else None,
        }
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {smia_key}: AUC-ROC={auc_str}")

    # Save
    results_path.write_text(json.dumps(results, indent=2))
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {results_path}, {summary_path}")


def add_to_robustness(results_path):
    """Add 4 s_mia metrics to robustness results."""
    results = json.loads(results_path.read_text())

    count = 0
    for model_key, model_data in results.items():
        # metrics_before
        mb = model_data.get("metrics_before", {})
        if mb:
            mb.update(compute_smia_individual(mb))
            mb.pop("s_mia", None)

        # metrics_after_* (quant or relearn)
        for k in list(model_data.keys()):
            if k.startswith("metrics_after"):
                ma = model_data[k]
                if ma:
                    ma.update(compute_smia_individual(ma))
                    ma.pop("s_mia", None)

        count += 1

    results_path.write_text(json.dumps(results, indent=2))
    print(f"Robustness: updated {count} models in {results_path}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("faithfulness", "all"):
        print("=" * 60)
        print("FAITHFULNESS")
        print("=" * 60)
        results_path = Path("runs/meta_eval/faithfulness/results.json")
        summary_path = Path("runs/meta_eval/faithfulness/summary.json")
        if results_path.exists():
            add_to_faithfulness(results_path, summary_path)
        else:
            print(f"  Not found: {results_path}")

    if mode in ("robustness", "all"):
        print("\n" + "=" * 60)
        print("ROBUSTNESS")
        print("=" * 60)
        for subdir in ["quant", "relearn"]:
            rp = Path(f"runs/meta_eval/robustness/{subdir}/results.json")
            if rp.exists():
                add_to_robustness(rp)
            else:
                print(f"  Not found: {rp}")


if __name__ == "__main__":
    main()

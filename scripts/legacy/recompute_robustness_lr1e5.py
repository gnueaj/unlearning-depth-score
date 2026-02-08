#!/usr/bin/env python3
"""
Recompute robustness metrics from cached results using the agreed protocol.

Policy:
- Relearning (R): utility filter + per-metric faithfulness-threshold filter.
- Quantization (Q): utility filter + per-metric faithfulness-threshold filter (no lr filter).
- Raw metric values for all Open-Unlearning metrics.
- UDS only is converted to (1 - UDS).
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


METRICS = [
    "em", "es", "prob", "paraprob", "truth_ratio",
    "rouge", "para_rouge", "jailbreak_rouge",
    "mia_loss", "mia_zlib", "mia_min_k", "mia_min_kpp", "uds",
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _is_lr1e5(model_name: str) -> bool:
    return "_lr1e5_" in model_name


def _to_unlearning_score(metric: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    # Raw metric for all Open-Unlearning metrics; only UDS is inverted.
    return 1 - value if metric == "uds" else value


def _hm(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return float(2 * a * b / (a + b))


def main() -> None:
    results_path = Path("runs/meta_eval/robustness_parallel/merged/results.json")
    filter_path = Path("runs/meta_eval/robustness_filter_list/filtered_models.json")
    meta_eval_path = Path("docs/data/meta_eval.json")
    out_path = Path("runs/meta_eval/robustness_parallel/recomputed_direction_fixed/summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = _load(results_path)
    filt = _load(filter_path)
    meta_eval = _load(meta_eval_path)

    utility_bad = set(filt.get("utility_filter", {}).get("filtered_models", []))
    per_metric_pass = filt.get("metric_filter", {}).get("passed_models", {})

    retain_before = results.get("retain_before", {}).get("metrics", {})
    retain_after = results.get("retain_after", {}).get("metrics", {})

    model_names = [
        k for k, v in results.items()
        if k not in {"retain_before", "retain_after"} and isinstance(v, dict)
    ]
    lr1e5_models = [m for m in model_names if _is_lr1e5(m)]

    robustness_new: Dict[str, Dict[str, Optional[float]]] = {}
    detail_rows: Dict[str, Dict[str, int]] = {}

    for metric in METRICS:
        # Relearning: utility + faithfulness-threshold filters.
        pass_models_r = [
            m for m in per_metric_pass.get(metric, [])
            if m in results and m not in utility_bad
        ]
        # Quantization: utility + faithfulness-threshold filters (same as R).
        pass_models_q = [
            m for m in per_metric_pass.get(metric, [])
            if m in results and m not in utility_bad
        ]

        r_vals = []
        for m in pass_models_r:
            row = results[m]
            m_unl_a = _to_unlearning_score(metric, row.get("metrics_before", {}).get(metric))
            m_unl_b = _to_unlearning_score(metric, row.get("metrics_after_relearn", {}).get(metric))
            m_ret_a = _to_unlearning_score(metric, retain_before.get(metric))
            m_ret_b = _to_unlearning_score(metric, retain_after.get(metric))
            if None in (m_unl_a, m_unl_b, m_ret_a, m_ret_b):
                continue
            # Paper form: (ret_before - ret_after) / (unl_before - unl_after)
            denom = m_unl_a - m_unl_b
            numer = m_ret_a - m_ret_b
            if abs(denom) < 1e-8:
                continue
            r = max(0.0, min(numer / denom, 1.0))
            r_vals.append(r)

        q_vals = []
        for m in pass_models_q:
            row = results[m]
            m_unl_a = _to_unlearning_score(metric, row.get("metrics_before", {}).get(metric))
            m_unl_b = _to_unlearning_score(metric, row.get("metrics_after_quant", {}).get(metric))
            if None in (m_unl_a, m_unl_b):
                continue
            if abs(m_unl_a) < 1e-8:
                q_vals.append(1.0)
            else:
                q_vals.append(max(0.0, min(m_unl_b / m_unl_a, 1.0)))

        avg_r = float(np.mean(r_vals)) if r_vals else None
        avg_q = float(np.mean(q_vals)) if q_vals else None
        agg = _hm(avg_r, avg_q)

        robustness_new[metric] = {
            "relearning": avg_r,
            "quantization": avg_q,
            "agg": agg,
        }
        detail_rows[metric] = {
            "n_R": len(r_vals),
            "n_Q": len(q_vals),
            "pool_R": len(pass_models_r),
            "pool_Q": len(pass_models_q),
        }

    # Update data consumed by HTML
    meta_eval["robustness"] = robustness_new
    notes = meta_eval.get("notes", {})
    notes["robustness_policy"] = (
        "R & Q: utility+faithfulness-threshold filters; "
        "raw metric values for all Open-Unlearning metrics; UDS only is mapped as 1-UDS"
    )
    notes.pop("mia_robustness_source", None)
    notes.pop("mia_robustness_path", None)
    notes["robustness_source"] = str(out_path)
    meta_eval["notes"] = notes
    _save(meta_eval_path, meta_eval)

    out = {
        "source_results": str(results_path),
        "source_filters": str(filter_path),
        "n_models_total": len(model_names),
        "n_models_lr1e5": len(lr1e5_models),
        "n_models_utility_filtered": len(utility_bad),
        "robustness": robustness_new,
        "counts": detail_rows,
    }
    _save(out_path, out)

    print(f"Saved: {meta_eval_path}")
    print(f"Saved: {out_path}")
    print(f"Models total={len(model_names)}, lr1e5={len(lr1e5_models)}, utility_filtered={len(utility_bad)}")
    for m in METRICS:
        d = detail_rows.get(m, {})
        r = robustness_new[m]["relearning"]
        q = robustness_new[m]["quantization"]
        a = robustness_new[m]["agg"]
        r_s = "N/A" if r is None else f"{r:.4f}"
        q_s = "N/A" if q is None else f"{q:.4f}"
        a_s = "N/A" if a is None else f"{a:.4f}"
        print(
            f"{m:16s} poolR={d.get('pool_R',0):3d} nR={d.get('n_R',0):3d} "
            f"poolQ={d.get('pool_Q',0):3d} nQ={d.get('n_Q',0):3d} "
            f"R={r_s:>8s} Q={q_s:>8s} Agg={a_s:>8s}"
        )


if __name__ == "__main__":
    main()

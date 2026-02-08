#!/usr/bin/env python3
"""
Create scatter plots for quantization robustness analysis (Figure 10 style).
Uses usable_models.json for per-metric filtering.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    base = Path('/home/jaeung/activation-patching-unlearning')
    quant_path = base / 'runs/meta_eval/robustness/quant/results.json'
    usable_path = base / 'runs/meta_eval/robustness/usable_models.json'
    output_dir = base / 'runs/meta_eval/robustness/quant/plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(quant_path) as f:
        quant_data = json.load(f)
    with open(usable_path) as f:
        usable = json.load(f)

    usable_per_metric = usable['usable_models_per_metric']
    quant_models = set(quant_data.keys()) - {'retain'}

    metrics = ['em', 'es', 'truth_ratio', 'prob', 'rouge', 'jailbreak_rouge',
               'paraprob', 'para_rouge', 'mia_loss', 'mia_min_k', 'mia_min_kpp', 'mia_zlib', 'uds']

    metric_labels = {
        'em': 'Exact Match', 'es': 'Extraction Strength', 'truth_ratio': 'Truth Ratio',
        'prob': 'Prob.', 'rouge': 'ROUGE', 'jailbreak_rouge': 'Jailbreak ROUGE',
        'paraprob': 'Para. Prob.', 'para_rouge': 'Para. ROUGE', 'mia_loss': 'MIA-LOSS',
        'mia_min_k': 'MIA-MinK', 'mia_min_kpp': 'MIA-MinK++', 'mia_zlib': 'MIA-ZLib',
        'uds': '1 - UDS'
    }

    # UDS: higher = less knowledge, so invert to 1-uds for consistent "higher = more knowledge"
    inverted_metrics = {'uds'}

    # Retain model data
    retain_data = quant_data.get('retain', {})

    all_metric_data = {}
    robustness_results = {}

    for metric in metrics:
        usable_models = set(usable_per_metric.get(metric, [])) & quant_models
        filtered_before = []
        filtered_after = []
        model_names = []
        q_values = []

        for model_name in sorted(usable_models):
            md = quant_data[model_name]
            mb = md.get('metrics_before', {})
            ma = md.get('metrics_after_quant', {})
            if metric not in mb or metric not in ma:
                continue

            bv = mb[metric]
            av = ma[metric]

            if metric in inverted_metrics:
                bv = 1 - bv
                av = 1 - av

            filtered_before.append(bv)
            filtered_after.append(av)
            model_names.append(model_name)

            # Q = min(before/after, 1)
            # after > before means knowledge recovered (bad) → Q < 1
            if av > 0:
                q = min(bv / av, 1.0)
            else:
                q = 1.0
            q_values.append(q)

        # Retain
        rb = retain_data.get('metrics_before', {}).get(metric)
        ra = retain_data.get('metrics_after_quant', {}).get(metric)
        if rb is not None and ra is not None and metric in inverted_metrics:
            rb, ra = 1 - rb, 1 - ra

        avg_q = float(np.mean(q_values)) if q_values else 0.0

        all_metric_data[metric] = {
            'before': filtered_before, 'after': filtered_after,
            'names': model_names, 'q_values': q_values,
            'retain_before': rb, 'retain_after': ra, 'avg_q': avg_q
        }
        robustness_results[metric] = {
            'avg_Q': round(avg_q, 4),
            'n_models': len(filtered_before),
            'n_usable_total': len(usable_per_metric.get(metric, [])),
            'q_per_model': {n: round(q, 4) for n, q in zip(model_names, q_values)}
        }

        print(f"{metric:20s}: n={len(filtered_before):3d}/{len(usable_per_metric.get(metric, [])):3d}  avg_Q={avg_q:.4f}")

    # Save robustness results
    with open(output_dir / 'quant_robustness_results.json', 'w') as f:
        json.dump(robustness_results, f, indent=2)
    print(f"\nSaved: {output_dir / 'quant_robustness_results.json'}")

    # === Plot: 5x3 grid scatter (Figure 10 style) ===
    fig, axes = plt.subplots(5, 3, figsize=(13, 19))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        d = all_metric_data[metric]
        bef, aft = d['before'], d['after']
        rb, ra = d['retain_before'], d['retain_after']

        all_vals = bef + aft
        if rb is not None:
            all_vals += [rb, ra]
        if not all_vals:
            ax.set_visible(False)
            continue

        pad = 0.05
        lo = min(all_vals) - pad * (max(all_vals) - min(all_vals) + 0.01)
        hi = max(all_vals) + pad * (max(all_vals) - min(all_vals) + 0.01)

        # Unreliable region (above y=x)
        ax.fill_between([lo, hi], [lo, hi], [hi, hi],
                        color='#FFE0E0', alpha=0.6)

        # y=x line
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5, linewidth=1)

        # Unlearned models
        if bef:
            ax.scatter(bef, aft, c='#2196F3', s=35, alpha=0.7,
                       edgecolors='#0D47A1', linewidths=0.5, zorder=3)

        # Retain model
        if rb is not None and ra is not None:
            ax.scatter([rb], [ra], c='red', s=100, marker='*',
                       edgecolors='darkred', linewidths=0.5, zorder=5)

        label = metric_labels.get(metric, metric)
        ax.set_title(f"{label}\nQ={d['avg_q']:.3f} (n={len(bef)})", fontsize=10, fontweight='bold')
        ax.set_xlabel('Before Quant', fontsize=8)
        ax.set_ylabel('After Quant', fontsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=7)

    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Quantization Robustness — Usable Models (batch 1)',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / 'quant_robustness_usable.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'quant_robustness_usable.png'}")


if __name__ == '__main__':
    main()

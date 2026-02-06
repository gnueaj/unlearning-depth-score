#!/usr/bin/env python3
"""
Plot histogram of S2 LOST - S1 LOST layer difference for each example.
X-axis: Layer count difference (negative = over-erased, positive = under-erased)
Y-axis: Number of examples
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def load_results(results_path, em_threshold=0.5):
    """Load results.json and extract S1/S2 LOST layer counts.

    S1 LOST: layers where Retain->Full patching fails (EM < threshold)
    S2 LOST: layers where Unlearn->Full patching fails (EM < threshold)

    The comparison:
    - S1 LOST > S2 LOST: under-erased (knowledge leaked in unlearned model)
    - S1 LOST = S2 LOST: exact-erased (ideal)
    - S1 LOST < S2 LOST: over-erased (collateral damage)
    """
    with open(results_path) as f:
        results = json.load(f)

    diffs = []
    for item in results:
        if "hard" not in item or "stage1" not in item or "stage2" not in item:
            continue

        # S1 LOST: count from stage1 results directly
        s1_lost = sum(1 for r in item["stage1"] if r["em"] < em_threshold)

        # S2 LOST: count from stage2 results directly (NOT from erased which is S1∩S2)
        s2_lost = sum(1 for r in item["stage2"] if r["em"] < em_threshold)

        # Skip general knowledge (all layers KEPT in S1)
        if s1_lost == 0:
            continue

        # Difference: positive = under-erased (S1 LOST > S2 LOST, knowledge leaked)
        #            negative = over-erased (S1 LOST < S2 LOST, collateral damage)
        diff = s1_lost - s2_lost
        diffs.append(diff)

    return diffs


def plot_comparison(layer_results, mlp_results, output_path, method_name="SimNPO"):
    """Plot side-by-side histograms for layer vs MLP patching."""
    layer_diffs = load_results(layer_results)
    mlp_diffs = load_results(mlp_results)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Determine common bin range for both plots
    all_diffs = layer_diffs + mlp_diffs
    min_val = min(all_diffs)
    max_val = max(all_diffs)
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

    # Determine common y-axis limit
    counts1, _ = np.histogram(layer_diffs, bins=bins)
    counts2, _ = np.histogram(mlp_diffs, bins=bins)
    max_count = max(max(counts1), max(counts2))
    y_max = int(max_count * 1.2) + 1  # Add 20% margin

    # Layer patching histogram
    ax1 = axes[0]
    ax1.hist(layer_diffs, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

    ax1.set_xlabel('S1 LOST - S2 LOST (Layer Count Difference)', fontsize=11)
    ax1.set_ylabel('Number of Examples', fontsize=11)
    ax1.set_title(f'{method_name} - Layer Patching (n={len(layer_diffs)})', fontsize=12)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.set_xticks(range(min_val, max_val + 1))
    ax1.set_ylim(0, y_max)
    ax1.set_xlim(min_val - 0.5, max_val + 0.5)

    # Add stats
    over_erased = sum(1 for d in layer_diffs if d < 0)
    exact = sum(1 for d in layer_diffs if d == 0)
    under_erased = sum(1 for d in layer_diffs if d > 0)
    stats_text = f'Over: {over_erased} ({100*over_erased/len(layer_diffs):.1f}%)\n'
    stats_text += f'Exact: {exact} ({100*exact/len(layer_diffs):.1f}%)\n'
    stats_text += f'Under: {under_erased} ({100*under_erased/len(layer_diffs):.1f}%)'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # MLP patching histogram
    ax2 = axes[1]
    ax2.hist(mlp_diffs, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

    ax2.set_xlabel('S1 LOST - S2 LOST (Layer Count Difference)', fontsize=11)
    ax2.set_ylabel('Number of Examples', fontsize=11)
    ax2.set_title(f'{method_name} - MLP Patching (n={len(mlp_diffs)})', fontsize=12)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.set_xticks(range(min_val, max_val + 1))
    ax2.set_ylim(0, y_max)
    ax2.set_xlim(min_val - 0.5, max_val + 0.5)

    # Add stats
    over_erased_mlp = sum(1 for d in mlp_diffs if d < 0)
    exact_mlp = sum(1 for d in mlp_diffs if d == 0)
    under_erased_mlp = sum(1 for d in mlp_diffs if d > 0)
    stats_text = f'Over: {over_erased_mlp} ({100*over_erased_mlp/len(mlp_diffs):.1f}%)\n'
    stats_text += f'Exact: {exact_mlp} ({100*exact_mlp/len(mlp_diffs):.1f}%)\n'
    stats_text += f'Under: {under_erased_mlp} ({100*under_erased_mlp/len(mlp_diffs):.1f}%)'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return {
        'layer': {'diffs': layer_diffs, 'over': over_erased, 'exact': exact, 'under': under_erased},
        'mlp': {'diffs': mlp_diffs, 'over': over_erased_mlp, 'exact': exact_mlp, 'under': under_erased_mlp}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_results', type=str, required=True, help='Path to layer patching results.json')
    parser.add_argument('--mlp_results', type=str, required=True, help='Path to MLP patching results.json')
    parser.add_argument('--method', type=str, default='SimNPO', help='Method name for title')
    parser.add_argument('--output', type=str, default='erasure_diff_histogram.png', help='Output image path')
    args = parser.parse_args()

    plot_comparison(args.layer_results, args.mlp_results, args.output, args.method)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot UDS histograms for alpha1 experiments (30 methods)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configuration
ALPHA1_DIR = Path("runs/0201alpha1")
OUTPUT_DIR = Path("docs/0202/alpha1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Method categories with display names
METHOD_CATEGORIES = {
    "SimNPO": ["simnpo_lr1e5_b35_a1_d1_g0125_ep5", "simnpo_lr2e5_b35_a1_d1_g0125_ep5", "simnpo_lr5e5_b35_a1_d1_g0125_ep5"],
    "GradDiff": ["graddiff_lr1e5_a1_ep5", "graddiff_lr2e5_a1_ep5", "graddiff_lr5e5_a1_ep5"],
    "IdkNLL": ["idknll_lr1e5_a1_ep5", "idknll_lr2e5_a1_ep5", "idknll_lr5e5_a1_ep5"],
    "NPO": ["npo_lr1e5_b01_a1_ep5", "npo_lr2e5_b01_a1_ep5", "npo_lr5e5_b01_a1_ep5"],
    "IdkDPO": ["idkdpo_lr1e5_b01_a1_ep5", "idkdpo_lr2e5_b01_a1_ep5", "idkdpo_lr5e5_b01_a1_ep5"],
    "AltPO": ["altpo_lr1e5_b01_a1_ep5", "altpo_lr2e5_b01_a1_ep5", "altpo_lr5e5_b01_a1_ep5"],
    "UNDIAL": ["undial_lr1e5_b10_a1_ep5", "undial_lr1e4_b10_a1_ep5", "undial_lr3e4_b10_a1_ep5"],
    "RMU-L5": ["rmu_lr1e5_l5_s10_ep5", "rmu_lr2e5_l5_s10_ep5", "rmu_lr5e5_l5_s10_ep5"],
    "RMU-L10": ["rmu_lr1e5_l10_s10_ep5", "rmu_lr2e5_l10_s10_ep5", "rmu_lr5e5_l10_s10_ep5"],
    "RMU-L15": ["rmu_lr1e5_l15_s10_ep5", "rmu_lr2e5_l15_s10_ep5", "rmu_lr5e5_l15_s10_ep5"],
}

def load_results(model_name):
    """Load results.json for a model"""
    for folder in ALPHA1_DIR.iterdir():
        if model_name in folder.name:
            results_path = folder / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    return json.load(f)
    return None

def extract_udss(results):
    """Extract UDS values from results, excluding GK (N/A)"""
    udss = []
    for r in results:
        if r.get("uds") is not None and r.get("ft_layers"):
            udss.append(r["uds"])
    return udss

def plot_method_comparison_histogram():
    """Plot histogram comparing all methods (average UDS per method)"""
    fig, ax = plt.subplots(figsize=(14, 6))

    method_names = []
    avg_udss = []
    std_udss = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHOD_CATEGORIES)))
    bar_colors = []

    for i, (category, models) in enumerate(METHOD_CATEGORIES.items()):
        for model in models:
            results = load_results(model)
            if results:
                udss = extract_udss(results)
                if udss:
                    # Shorter display name
                    lr = model.split("_lr")[1].split("_")[0] if "_lr" in model else ""
                    short_name = f"{category}\n(lr={lr})"
                    method_names.append(short_name)
                    avg_udss.append(np.mean(udss))
                    std_udss.append(np.std(udss))
                    bar_colors.append(colors[i])

    x = np.arange(len(method_names))
    ax.bar(x, avg_udss, yerr=std_udss, capsize=2, color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.8)

    ax.set_ylabel('Average UDS', fontsize=12)
    ax.set_xlabel('Method (Learning Rate)', fontsize=12)
    ax.set_title('UDS Comparison Across 30 Unlearning Methods (α=1, τ=0.05)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='UDS=0.5')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "uds_comparison_all_methods.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'uds_comparison_all_methods.png'}")

def plot_uds_distribution_by_category():
    """Plot UDS distribution histograms: 10 columns (methods) x 3 rows (LR)"""
    # LR levels
    lr_levels = ['1e5', '2e5', '5e5']
    lr_display = {'1e5': 'lr=1e-5', '2e5': 'lr=2e-5', '5e5': 'lr=5e-5'}
    # Special case for UNDIAL
    undial_lr_map = {'1e5': '1e5', '2e5': '1e4', '5e5': '3e4'}

    fig, axes = plt.subplots(3, 10, figsize=(24, 8))

    colors = {'1e5': '#1f77b4', '2e5': '#ff7f0e', '5e5': '#2ca02c'}

    categories = list(METHOD_CATEGORIES.keys())

    for row, lr in enumerate(lr_levels):
        for col, category in enumerate(categories):
            ax = axes[row, col]
            models = METHOD_CATEGORIES[category]

            # Find matching model for this LR
            target_lr = undial_lr_map[lr] if category == "UNDIAL" else lr
            matching_model = None
            for model in models:
                if f"_lr{target_lr}_" in model or f"_lr{target_lr}e" in model:
                    matching_model = model
                    break

            if matching_model:
                results = load_results(matching_model)
                if results:
                    udss = extract_udss(results)
                    if udss:
                        mean_uds = np.mean(udss)
                        ax.hist(udss, bins=20, alpha=0.7, color=colors[lr], edgecolor='black', linewidth=0.3)
                        # Add mean UDS text
                        ax.text(0.95, 0.95, f'{mean_uds:.3f}', transform=ax.transAxes,
                                fontsize=10, fontweight='bold', ha='right', va='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 120)

            # Column titles (top row only)
            if row == 0:
                ax.set_title(category, fontsize=11, fontweight='bold')

            # Row labels (left column only)
            if col == 0:
                ax.set_ylabel(lr_display[lr], fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('')

            # X-axis label (bottom row only)
            if row == 2:
                ax.set_xlabel('UDS', fontsize=9)
            else:
                ax.set_xticklabels([])

            ax.grid(alpha=0.3)

    plt.suptitle('UDS Distribution: Method × Learning Rate (α=1, τ=0.05)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "uds_distribution_by_category.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'uds_distribution_by_category.png'}")

def plot_combined_uds_histogram():
    """Plot combined UDS histogram for all methods overlaid"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by method type (not LR)
    method_groups = {
        "SimNPO": [],
        "GradDiff": [],
        "IdkNLL": [],
        "NPO": [],
        "IdkDPO": [],
        "AltPO": [],
        "UNDIAL": [],
        "RMU": [],
    }

    for category, models in METHOD_CATEGORIES.items():
        group_key = category.replace("-L5", "").replace("-L10", "").replace("-L15", "")
        for model in models:
            results = load_results(model)
            if results:
                udss = extract_udss(results)
                method_groups[group_key].extend(udss)

    colors = plt.cm.Set2(np.linspace(0, 1, len(method_groups)))

    for i, (method, udss) in enumerate(method_groups.items()):
        if udss:
            ax.hist(udss, bins=30, alpha=0.5, label=f'{method} (n={len(udss)})', color=colors[i], edgecolor='black', linewidth=0.3)

    ax.set_xlabel('UDS', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('UDS Distribution by Method Type (α=1, τ=0.05)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='UDS=0.5')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "uds_distribution_combined.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'uds_distribution_combined.png'}")

def plot_boxplot_comparison():
    """Plot boxplot comparison of UDS across methods"""
    fig, ax = plt.subplots(figsize=(14, 6))

    data = []
    labels = []

    # Group by method type
    method_groups = {}
    for category, models in METHOD_CATEGORIES.items():
        group_key = category.replace("-L5", "").replace("-L10", "").replace("-L15", "")
        if group_key not in method_groups:
            method_groups[group_key] = []
        for model in models:
            results = load_results(model)
            if results:
                udss = extract_udss(results)
                method_groups[group_key].extend(udss)

    for method, udss in method_groups.items():
        if udss:
            data.append(udss)
            labels.append(method)

    ax.boxplot(data, labels=labels)
    ax.set_ylabel('UDS')
    ax.set_title('UDS Distribution by Method (α=1, τ=0.05)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "uds_boxplot_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'uds_boxplot_comparison.png'}")

def print_summary_stats():
    """Print summary statistics"""
    print("\n" + "="*60)
    print("UDS Summary Statistics (α=1, τ=0.05)")
    print("="*60)

    all_stats = []

    for category, models in METHOD_CATEGORIES.items():
        for model in models:
            results = load_results(model)
            if results:
                udss = extract_udss(results)
                if udss:
                    stats = {
                        'category': category,
                        'model': model,
                        'n': len(udss),
                        'mean': np.mean(udss),
                        'std': np.std(udss),
                        'median': np.median(udss),
                        'min': np.min(udss),
                        'max': np.max(udss),
                    }
                    all_stats.append(stats)

    # Sort by mean UDS
    all_stats.sort(key=lambda x: x['mean'], reverse=True)

    print(f"\n{'Method':<45} {'N':>5} {'Mean':>8} {'Std':>8} {'Median':>8}")
    print("-"*80)
    for s in all_stats:
        print(f"{s['model']:<45} {s['n']:>5} {s['mean']:>8.3f} {s['std']:>8.3f} {s['median']:>8.3f}")

    # Save to file
    with open(OUTPUT_DIR / "uds_summary_stats.txt", "w") as f:
        f.write("UDS Summary Statistics (α=1, τ=0.05)\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Method':<45} {'N':>5} {'Mean':>8} {'Std':>8} {'Median':>8}\n")
        f.write("-"*80 + "\n")
        for s in all_stats:
            f.write(f"{s['model']:<45} {s['n']:>5} {s['mean']:>8.3f} {s['std']:>8.3f} {s['median']:>8.3f}\n")
    print(f"\nSaved: {OUTPUT_DIR / 'uds_summary_stats.txt'}")

if __name__ == "__main__":
    print("Generating UDS histograms for alpha1 experiments...")

    plot_method_comparison_histogram()
    plot_uds_distribution_by_category()
    plot_combined_uds_histogram()
    plot_boxplot_comparison()
    print_summary_stats()

    print("\nDone!")

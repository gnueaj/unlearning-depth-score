#!/usr/bin/env python3
"""
Advanced UDS visualizations for alpha5 experiments
- Parallel coordinate plot (weak→mid→strong UDS movement per instance)
- Layer-wise delta heatmap
- Instance-wise UDS CDF comparison
- Layer-wise UDS contribution
- Method ranking scatter (mean vs std)
- Fixed distribution chart with overflow indicator
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

# Configuration
ALPHA5_DIR = Path("runs/0201alpha5")
OUTPUT_DIR = Path("docs/0202/alpha5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Method categories
METHOD_CATEGORIES = {
    "SimNPO": ["simnpo_lr1e5_b35_a1_d1_g0125_ep5", "simnpo_lr2e5_b35_a1_d1_g0125_ep5", "simnpo_lr5e5_b35_a1_d1_g0125_ep5"],
    "GradDiff": ["graddiff_lr1e5_a5_ep5", "graddiff_lr2e5_a5_ep5", "graddiff_lr5e5_a5_ep5"],
    "IdkNLL": ["idknll_lr1e5_a5_ep5", "idknll_lr2e5_a5_ep5", "idknll_lr5e5_a5_ep5"],
    "NPO": ["npo_lr1e5_b01_a5_ep5", "npo_lr2e5_b01_a5_ep5", "npo_lr5e5_b01_a5_ep5"],
    "IdkDPO": ["idkdpo_lr1e5_b01_a5_ep5", "idkdpo_lr2e5_b01_a5_ep5", "idkdpo_lr5e5_b01_a5_ep5"],
    "AltPO": ["altpo_lr1e5_b01_a5_ep5", "altpo_lr2e5_b01_a5_ep5", "altpo_lr5e5_b01_a5_ep5"],
    "UNDIAL": ["undial_lr1e5_b10_a5_ep5", "undial_lr1e4_b10_a5_ep5", "undial_lr3e4_b10_a5_ep5"],
    "RMU-L5": ["rmu_lr1e5_l5_s10_ep5", "rmu_lr2e5_l5_s10_ep5", "rmu_lr5e5_l5_s10_ep5"],
    "RMU-L10": ["rmu_lr1e5_l10_s10_ep5", "rmu_lr2e5_l10_s10_ep5", "rmu_lr5e5_l10_s10_ep5"],
    "RMU-L15": ["rmu_lr1e5_l15_s10_ep5", "rmu_lr2e5_l15_s10_ep5", "rmu_lr5e5_l15_s10_ep5"],
}

# LR mapping for UNDIAL
UNDIAL_LR_MAP = {'1e5': '1e5', '2e5': '1e4', '5e5': '3e4'}


def load_results(model_name):
    """Load results.json for a model"""
    for folder in ALPHA5_DIR.iterdir():
        if model_name in folder.name:
            results_path = folder / "results.json"
            if results_path.exists() and results_path.stat().st_size > 0:
                try:
                    with open(results_path) as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse {results_path}")
                    return None
    return None


def get_model_for_lr(category, lr_level):
    """Get model name for a category and LR level"""
    models = METHOD_CATEGORIES[category]
    target_lr = UNDIAL_LR_MAP[lr_level] if category == "UNDIAL" else lr_level
    for model in models:
        if f"_lr{target_lr}_" in model or f"_lr{target_lr}e" in model:
            return model
    return None


def extract_udss_with_idx(results):
    """Extract (idx, uds) pairs from results"""
    pairs = []
    for r in results:
        if r.get("uds") is not None and r.get("ft_layers"):
            pairs.append((r["idx"], r["uds"]))
    return pairs


def extract_layer_deltas(results):
    """Extract layer-wise S1 and S2 deltas"""
    layer_data = defaultdict(lambda: {"s1": [], "s2": []})
    for r in results:
        if r.get("layer_results"):
            for lr in r["layer_results"]:
                layer = lr["layer"]
                if lr.get("s1_delta") is not None:
                    layer_data[layer]["s1"].append(lr["s1_delta"])
                if lr.get("s2_delta") is not None:
                    layer_data[layer]["s2"].append(lr["s2_delta"])
    return layer_data


# ============================================================================
# 1. Fixed distribution chart with broken axis (wave mark)
# ============================================================================
def plot_uds_distribution_fixed():
    """Plot UDS distribution with broken axis for bars exceeding y-limit"""
    lr_levels = ['1e5', '2e5', '5e5']
    lr_display = {'1e5': 'lr=1e-5', '2e5': 'lr=2e-5', '5e5': 'lr=5e-5'}
    Y_MAX = 120
    Y_BREAK_BOTTOM = 100  # Where to break
    Y_BREAK_TOP = 110     # Resume from here (visual gap)

    fig, axes = plt.subplots(3, 10, figsize=(24, 8))
    colors = {'1e5': '#1f77b4', '2e5': '#ff7f0e', '5e5': '#2ca02c'}
    categories = list(METHOD_CATEGORIES.keys())

    for row, lr in enumerate(lr_levels):
        for col, category in enumerate(categories):
            ax = axes[row, col]
            model = get_model_for_lr(category, lr)
            has_overflow = False
            max_overflow_count = 0

            if model:
                results = load_results(model)
                if results:
                    udss = [r["uds"] for r in results if r.get("uds") is not None and r.get("ft_layers")]
                    if udss:
                        mean_uds = np.mean(udss)

                        # Compute histogram
                        counts, bins = np.histogram(udss, bins=20, range=(0, 1))

                        # Plot bars with overflow handling
                        for i, (count, left_edge) in enumerate(zip(counts, bins[:-1])):
                            width = bins[i+1] - left_edge
                            if count > Y_MAX:
                                has_overflow = True
                                max_overflow_count = max(max_overflow_count, count)
                                # Draw bar up to Y_MAX with diagonal hatch to indicate overflow
                                ax.bar(left_edge, Y_MAX, width=width, alpha=0.7,
                                       color=colors[lr], edgecolor='black', linewidth=0.3, align='edge')
                                # Add count label on top
                                ax.text(left_edge + width/2, Y_MAX + 2, f'{count}',
                                       ha='center', va='bottom', fontsize=6, fontweight='bold',
                                       color='darkred')
                            else:
                                ax.bar(left_edge, count, width=width, alpha=0.7,
                                       color=colors[lr], edgecolor='black', linewidth=0.3, align='edge')

                        # Add mean UDS text
                        ax.text(0.95, 0.95, f'{mean_uds:.3f}', transform=ax.transAxes,
                                fontsize=10, fontweight='bold', ha='right', va='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Add broken axis indicator if overflow exists
            if has_overflow:
                # Draw wave/break marks on both sides of y-axis at Y_MAX
                d = 0.015  # Size of diagonal lines
                kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1)
                # Left side waves
                ax.plot((-d, +d), (0.97-d, 0.97+d), **kwargs)
                ax.plot((-d, +d), (0.94-d, 0.94+d), **kwargs)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, Y_MAX + 15 if has_overflow else Y_MAX)

            if row == 0:
                ax.set_title(category, fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(lr_display[lr], fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('')
            if row == 2:
                ax.set_xlabel('UDS', fontsize=9)
            else:
                ax.set_xticklabels([])
            ax.grid(alpha=0.3)

    plt.suptitle('UDS Distribution: Method × Learning Rate (α=5, τ=0.05)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "uds_distribution_by_category_fixed.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'uds_distribution_by_category_fixed.png'}")


# ============================================================================
# 2. Parallel coordinate plot (weak→mid→strong per instance)
# ============================================================================
def plot_parallel_coordinates():
    """Plot parallel coordinates showing UDS movement from weak to strong LR per instance"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Exclude RMU (has layer variants)
    methods = ["SimNPO", "GradDiff", "IdkNLL", "NPO", "IdkDPO", "AltPO", "UNDIAL"]

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]

        # Load data for weak, mid, strong
        models = METHOD_CATEGORIES[method]
        data_by_lr = {}

        for i, lr_label in enumerate(['weak', 'mid', 'strong']):
            lr_level = ['1e5', '2e5', '5e5'][i]
            model = get_model_for_lr(method, lr_level)
            if model:
                results = load_results(model)
                if results:
                    data_by_lr[lr_label] = dict(extract_udss_with_idx(results))

        if len(data_by_lr) == 3:
            # Find common indices
            common_idx = set(data_by_lr['weak'].keys()) & set(data_by_lr['mid'].keys()) & set(data_by_lr['strong'].keys())

            # Plot lines for each instance
            x = [0, 1, 2]
            for idx in common_idx:
                y = [data_by_lr['weak'][idx], data_by_lr['mid'][idx], data_by_lr['strong'][idx]]
                # Color by trend: increasing=red, decreasing=blue, stable=gray
                if y[2] > y[0] + 0.1:
                    color = 'red'
                    alpha = 0.3
                elif y[2] < y[0] - 0.1:
                    color = 'blue'
                    alpha = 0.3
                else:
                    color = 'gray'
                    alpha = 0.1
                ax.plot(x, y, color=color, alpha=alpha, linewidth=0.5)

            # Plot mean line
            means = [np.mean(list(data_by_lr[lr].values())) for lr in ['weak', 'mid', 'strong']]
            ax.plot(x, means, color='black', linewidth=3, marker='o', markersize=8, label='Mean')

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['weak\n(lr=1e-5)', 'mid\n(lr=2e-5)', 'strong\n(lr=5e-5)'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('UDS')
            ax.set_title(f'{method}\n(n={len(common_idx)})', fontweight='bold')
            ax.grid(alpha=0.3)

            # Add legend
            red_patch = mpatches.Patch(color='red', alpha=0.5, label='Increasing')
            blue_patch = mpatches.Patch(color='blue', alpha=0.5, label='Decreasing')
            gray_patch = mpatches.Patch(color='gray', alpha=0.3, label='Stable')
            ax.legend(handles=[red_patch, blue_patch, gray_patch], loc='lower right', fontsize=8)

    # Hide unused subplot
    axes[7].axis('off')

    plt.suptitle('Instance-wise UDS Movement: Weak → Mid → Strong LR (α=5, τ=0.05)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "uds_parallel_coordinates.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'uds_parallel_coordinates.png'}")


# ============================================================================
# 3. Layer-wise delta heatmap
# ============================================================================
def plot_layerwise_delta_heatmap():
    """Plot heatmap of mean S2 delta by layer for each method"""
    methods = list(METHOD_CATEGORIES.keys())
    lr_levels = ['1e5', '2e5', '5e5']
    n_layers = 16

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for ax_idx, lr in enumerate(lr_levels):
        ax = axes[ax_idx]

        # Collect data: methods x layers
        data = np.zeros((len(methods), n_layers))

        for m_idx, method in enumerate(methods):
            model = get_model_for_lr(method, lr)
            if model:
                results = load_results(model)
                if results:
                    layer_data = extract_layer_deltas(results)
                    for layer in range(n_layers):
                        if layer in layer_data and layer_data[layer]["s2"]:
                            data[m_idx, layer] = np.mean(layer_data[layer]["s2"])

        # Plot heatmap
        im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=0.3)

        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(range(n_layers))
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xlabel('Layer')
        ax.set_title(f'lr={lr.replace("e5", "e-5")}', fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Mean Δ_S2')

    plt.suptitle('Layer-wise Mean S2 Delta by Method (α=5, τ=0.05)\nHigher = More knowledge loss when patching',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "layerwise_s2_delta_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'layerwise_s2_delta_heatmap.png'}")


# ============================================================================
# 4. Instance-wise UDS CDF comparison
# ============================================================================
def plot_uds_cdf():
    """Plot CDF of UDS for each method type"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Group methods
    method_groups = {
        "SimNPO": [], "GradDiff": [], "IdkNLL": [], "NPO": [],
        "IdkDPO": [], "AltPO": [], "UNDIAL": [], "RMU": [],
    }

    for category, models in METHOD_CATEGORIES.items():
        group_key = category.replace("-L5", "").replace("-L10", "").replace("-L15", "")
        for model in models:
            results = load_results(model)
            if results:
                udss = [r["uds"] for r in results if r.get("uds") is not None and r.get("ft_layers")]
                method_groups[group_key].extend(udss)

    colors = plt.cm.Set1(np.linspace(0, 1, len(method_groups)))

    for i, (method, udss) in enumerate(method_groups.items()):
        if udss:
            sorted_udss = np.sort(udss)
            cdf = np.arange(1, len(sorted_udss) + 1) / len(sorted_udss)
            ax.plot(sorted_udss, cdf, label=f'{method} (n={len(udss)})',
                   color=colors[i], linewidth=2)

    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='UDS=0.5')
    ax.set_xlabel('UDS', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('CDF of Instance-wise UDS by Method Type (α=5, τ=0.05)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "uds_cdf_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'uds_cdf_comparison.png'}")


# ============================================================================
# 5. Layer-wise UDS contribution (stacked bar)
# ============================================================================
def plot_layerwise_contribution():
    """Plot stacked bar showing which layers contribute most to UDS"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    methods = ["SimNPO", "GradDiff", "IdkNLL", "NPO", "IdkDPO", "AltPO", "UNDIAL", "RMU-L10"]
    n_layers = 16

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]

        # Get mid LR model
        model = get_model_for_lr(method, '2e5')
        if model:
            results = load_results(model)
            if results:
                # Calculate per-layer contribution to UDS
                layer_contributions = np.zeros(n_layers)
                total_s1 = np.zeros(n_layers)
                total_s2 = np.zeros(n_layers)

                for r in results:
                    if r.get("layer_results") and r.get("ft_layers"):
                        for lr in r["layer_results"]:
                            layer = lr["layer"]
                            s1_d = lr.get("s1_delta", 0) or 0
                            s2_d = lr.get("s2_delta", 0) or 0
                            if s1_d > 0.05:  # FT condition
                                total_s1[layer] += s1_d
                                total_s2[layer] += min(s2_d, s1_d)  # clip

                # Calculate contribution ratio
                for layer in range(n_layers):
                    if total_s1[layer] > 0:
                        layer_contributions[layer] = total_s2[layer] / total_s1[layer]

                # Plot bar
                colors_layer = plt.cm.viridis(np.linspace(0.2, 0.8, n_layers))
                ax.bar(range(n_layers), layer_contributions, color=colors_layer, edgecolor='black', linewidth=0.5)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
                ax.set_xlabel('Layer')
                ax.set_ylabel('Layer UDS Contribution')
                ax.set_title(f'{method} (lr=2e-5)', fontweight='bold')
                ax.set_ylim(0, 1.2)
                ax.set_xticks(range(n_layers))
                ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Layer-wise UDS Contribution (mid LR, α=5, τ=0.05)\nHigher = layer contributes more to erasure',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "layerwise_uds_contribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'layerwise_uds_contribution.png'}")


# ============================================================================
# 6. Method ranking scatter (mean vs std)
# ============================================================================
def plot_method_ranking_scatter():
    """Scatter plot of mean UDS vs std UDS for all 30 methods"""
    fig, ax = plt.subplots(figsize=(12, 8))

    category_colors = plt.cm.tab10(np.linspace(0, 1, len(METHOD_CATEGORIES)))

    for i, (category, models) in enumerate(METHOD_CATEGORIES.items()):
        for model in models:
            results = load_results(model)
            if results:
                udss = [r["uds"] for r in results if r.get("uds") is not None and r.get("ft_layers")]
                if udss:
                    mean_uds = np.mean(udss)
                    std_uds = np.std(udss)

                    # Extract LR for marker
                    if "_lr1e5_" in model or "_lr1e5e" in model:
                        marker = 'o'  # weak
                    elif "_lr2e5_" in model or "_lr1e4_" in model:
                        marker = 's'  # mid
                    else:
                        marker = '^'  # strong

                    ax.scatter(mean_uds, std_uds, c=[category_colors[i]], marker=marker,
                              s=100, edgecolors='black', linewidth=0.5, alpha=0.8)

    # Legend for categories
    legend_handles = [mpatches.Patch(color=category_colors[i], label=cat)
                     for i, cat in enumerate(METHOD_CATEGORIES.keys())]
    legend1 = ax.legend(handles=legend_handles, loc='upper left', title='Method', fontsize=8)
    ax.add_artist(legend1)

    # Legend for LR markers
    marker_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='weak (lr=1e-5)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='mid (lr=2e-5)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='strong (lr=5e-5)'),
    ]
    ax.legend(handles=marker_handles, loc='upper right', title='LR', fontsize=8)

    ax.set_xlabel('Mean UDS', fontsize=12)
    ax.set_ylabel('Std UDS', fontsize=12)
    ax.set_title('Method Ranking: Mean vs Std of UDS (α=5, τ=0.05)\nIdeal: High mean, Low std',
                fontsize=14, fontweight='bold')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.4)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "method_ranking_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'method_ranking_scatter.png'}")


# ============================================================================
# 7. RMU Layer comparison
# ============================================================================
def plot_rmu_layer_comparison():
    """Compare RMU performance across different target layers"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rmu_methods = ["RMU-L5", "RMU-L10", "RMU-L15"]
    lr_levels = ['1e5', '2e5', '5e5']
    lr_display = {'1e5': '1e-5', '2e5': '2e-5', '5e5': '5e-5'}

    for ax_idx, lr in enumerate(lr_levels):
        ax = axes[ax_idx]

        data = []
        labels = []

        for method in rmu_methods:
            model = get_model_for_lr(method, lr)
            if model:
                results = load_results(model)
                if results:
                    udss = [r["uds"] for r in results if r.get("uds") is not None and r.get("ft_layers")]
                    if udss:
                        data.append(udss)
                        labels.append(method.replace("RMU-", ""))

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors = ['#ff9999', '#99ff99', '#9999ff']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel('UDS')
            ax.set_xlabel('Target Layer')
            ax.set_title(f'lr={lr_display[lr]}', fontweight='bold')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle('RMU: Effect of Target Layer on UDS (α=5, τ=0.05)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rmu_layer_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'rmu_layer_comparison.png'}")


if __name__ == "__main__":
    print("Generating advanced UDS visualizations for alpha5 experiments...")
    print("=" * 60)

    plot_uds_distribution_fixed()
    plot_parallel_coordinates()
    plot_layerwise_delta_heatmap()
    plot_uds_cdf()
    plot_layerwise_contribution()
    plot_method_ranking_scatter()
    plot_rmu_layer_comparison()

    print("=" * 60)
    print("Done!")

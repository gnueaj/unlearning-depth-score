#!/usr/bin/env python3
"""
Advanced UDR visualizations for alpha5 experiments (v2)
1. Parallel coordinate (10 methods including RMU variants, 5x2 layout)
2. Layer-wise UDR line chart (all methods)
3. UDR CDF by LR (3 subplots)
4. S1 layer-wise delta distribution (mean/std line chart)
5. UDR vs various factors (answer type, entity length, FT layer count)
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

# Method categories (10 total for parallel coordinates)
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


def extract_udrs_with_idx(results):
    """Extract (idx, udr) pairs from results"""
    pairs = []
    for r in results:
        if r.get("udr") is not None and r.get("ft_layers"):
            pairs.append((r["idx"], r["udr"]))
    return pairs


# ============================================================================
# 1. Parallel coordinate plot (10 methods, 5x2 layout)
# ============================================================================
def plot_parallel_coordinates():
    """Plot parallel coordinates showing UDR movement from weak to strong LR per instance"""
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    methods = list(METHOD_CATEGORIES.keys())  # All 10 methods

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]

        # Load data for weak, mid, strong
        data_by_lr = {}
        for i, lr_label in enumerate(['weak', 'mid', 'strong']):
            lr_level = ['1e5', '2e5', '5e5'][i]
            model = get_model_for_lr(method, lr_level)
            if model:
                results = load_results(model)
                if results:
                    data_by_lr[lr_label] = dict(extract_udrs_with_idx(results))

        if len(data_by_lr) == 3:
            # Find common indices
            common_idx = set(data_by_lr['weak'].keys()) & set(data_by_lr['mid'].keys()) & set(data_by_lr['strong'].keys())

            # Plot lines for each instance
            x = [0, 1, 2]
            for idx in common_idx:
                y = [data_by_lr['weak'][idx], data_by_lr['mid'][idx], data_by_lr['strong'][idx]]
                # Color by trend
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

            # Plot mean line (black)
            means = [np.mean(list(data_by_lr[lr].values())) for lr in ['weak', 'mid', 'strong']]
            ax.plot(x, means, color='black', linewidth=3, marker='o', markersize=8)

            # Add mean values as text
            for i, (xi, m) in enumerate(zip(x, means)):
                ax.text(xi, m + 0.05, f'{m:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['weak\n(1e-5)', 'mid\n(2e-5)', 'strong\n(5e-5)'])
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('UDR')
            ax.set_title(f'{method} (n={len(common_idx)})', fontweight='bold')
            ax.grid(alpha=0.3)

    # Add legend in last subplot area
    red_patch = mpatches.Patch(color='red', alpha=0.5, label='Increasing (Δ>0.1)')
    blue_patch = mpatches.Patch(color='blue', alpha=0.5, label='Decreasing (Δ<-0.1)')
    gray_patch = mpatches.Patch(color='gray', alpha=0.3, label='Stable')
    black_line = plt.Line2D([0], [0], color='black', linewidth=3, marker='o', label='Mean UDR')
    fig.legend(handles=[red_patch, blue_patch, gray_patch, black_line],
               loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=10)

    plt.suptitle('Instance-wise UDR Movement: Weak → Mid → Strong LR (α=5, τ=0.05)\nBlack line = Mean UDR',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "udr_parallel_coordinates.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'udr_parallel_coordinates.png'}")


# ============================================================================
# 2. Layer-wise UDR line chart (all methods)
# ============================================================================
def plot_layerwise_udr_linechart():
    """Plot layer-wise UDR as line chart for all methods"""
    n_layers = 16
    lr_levels = ['1e5', '2e5', '5e5']
    lr_display = {'1e5': 'lr=1e-5', '2e5': 'lr=2e-5', '5e5': 'lr=5e-5'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHOD_CATEGORIES)))

    for ax_idx, lr in enumerate(lr_levels):
        ax = axes[ax_idx]

        for m_idx, method in enumerate(METHOD_CATEGORIES.keys()):
            model = get_model_for_lr(method, lr)
            if model:
                results = load_results(model)
                if results:
                    # Calculate per-layer UDR using s1_details and s2_details
                    layer_udrs = defaultdict(list)
                    for r in results:
                        if r.get("s1_details") and r.get("s2_details") and r.get("ft_layers"):
                            # Create layer -> delta mapping
                            s1_by_layer = {d["layer"]: d.get("delta", 0) for d in r["s1_details"]}
                            s2_by_layer = {d["layer"]: d.get("delta", 0) for d in r["s2_details"]}

                            for layer in s1_by_layer.keys():
                                s1_d = s1_by_layer.get(layer, 0) or 0
                                s2_d = s2_by_layer.get(layer, 0) or 0
                                if s1_d > 0.05:  # FT condition
                                    layer_udr = min(s2_d / s1_d, 1.0) if s1_d > 0 else 0
                                    layer_udrs[layer].append(layer_udr)

                    # Plot mean UDR per layer
                    layers = sorted(layer_udrs.keys())
                    means = [np.mean(layer_udrs[l]) if layer_udrs[l] else 0 for l in layers]
                    overall_mean = np.mean(means) if means else 0
                    if layers and means:
                        ax.plot(layers, means, marker='o', markersize=4, linewidth=1.5,
                               color=colors[m_idx], label=f'{method} ({overall_mean:.2f})', alpha=0.8)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Layer UDR')
        ax.set_title(f'{lr_display[lr]}', fontweight='bold')
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(16))
        ax.grid(alpha=0.3)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=7)

    plt.suptitle('Layer-wise UDR by Method (α=5, τ=0.05)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "layerwise_udr_linechart.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'layerwise_udr_linechart.png'}")


def plot_layerwise_udr_rmu_only():
    """Plot layer-wise UDR line chart for RMU variants only (L5, L10, L15)"""
    n_layers = 16
    lr_levels = ['1e5', '2e5', '5e5']
    lr_display = {'1e5': 'lr=1e-5', '2e5': 'lr=2e-5', '5e5': 'lr=5e-5'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    rmu_methods = ['RMU-L5', 'RMU-L10', 'RMU-L15']
    colors = {'RMU-L5': '#1f77b4', 'RMU-L10': '#ff7f0e', 'RMU-L15': '#2ca02c'}
    markers = {'RMU-L5': 'o', 'RMU-L10': 's', 'RMU-L15': '^'}

    for ax_idx, lr in enumerate(lr_levels):
        ax = axes[ax_idx]

        for method in rmu_methods:
            model = get_model_for_lr(method, lr)
            if model:
                results = load_results(model)
                if results:
                    # Calculate per-layer UDR
                    layer_udrs = defaultdict(list)
                    for r in results:
                        if r.get("s1_details") and r.get("s2_details") and r.get("ft_layers"):
                            s1_by_layer = {d["layer"]: d.get("delta", 0) for d in r["s1_details"]}
                            s2_by_layer = {d["layer"]: d.get("delta", 0) for d in r["s2_details"]}

                            for layer in s1_by_layer.keys():
                                s1_d = s1_by_layer.get(layer, 0) or 0
                                s2_d = s2_by_layer.get(layer, 0) or 0
                                if s1_d > 0.05:
                                    layer_udr = min(s2_d / s1_d, 1.0) if s1_d > 0 else 0
                                    layer_udrs[layer].append(layer_udr)

                    layers = sorted(layer_udrs.keys())
                    means = [np.mean(layer_udrs[l]) if layer_udrs[l] else 0 for l in layers]
                    overall_mean = np.mean(means) if means else 0
                    if layers and means:
                        ax.plot(layers, means, marker=markers[method], markersize=6, linewidth=2,
                               color=colors[method], label=f'{method} (mean={overall_mean:.2f})', alpha=0.9)

        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Mean Layer UDR', fontsize=11)
        ax.set_title(f'{lr_display[lr]}', fontweight='bold', fontsize=12)
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(16))
        ax.grid(alpha=0.3)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

    plt.suptitle('Layer-wise UDR: RMU Variants (α=5, τ=0.05)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "layerwise_udr_rmu_only.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'layerwise_udr_rmu_only.png'}")


# ============================================================================
# 3. UDR CDF by LR (3 subplots in one PNG)
# ============================================================================
def plot_udr_cdf_by_lr():
    """Plot CDF of UDR for each method, separated by LR"""
    lr_levels = ['1e5', '2e5', '5e5']
    lr_display = {'1e5': 'lr=1e-5 (weak)', '2e5': 'lr=2e-5 (mid)', '5e5': 'lr=5e-5 (strong)'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHOD_CATEGORIES)))

    for ax_idx, lr in enumerate(lr_levels):
        ax = axes[ax_idx]

        for m_idx, method in enumerate(METHOD_CATEGORIES.keys()):
            model = get_model_for_lr(method, lr)
            if model:
                results = load_results(model)
                if results:
                    udrs = [r["udr"] for r in results if r.get("udr") is not None and r.get("ft_layers")]
                    if udrs:
                        sorted_udrs = np.sort(udrs)
                        cdf = np.arange(1, len(sorted_udrs) + 1) / len(sorted_udrs)
                        ax.plot(sorted_udrs, cdf, label=method, color=colors[m_idx], linewidth=1.5)

        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('UDR')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'{lr_display[lr]}', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=9)

    plt.suptitle('CDF of Instance-wise UDR by Learning Rate (α=5, τ=0.05)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(OUTPUT_DIR / "udr_cdf_by_lr.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'udr_cdf_by_lr.png'}")


# ============================================================================
# 4. S1 layer-wise delta distribution (mean/std line chart)
# ============================================================================
def plot_s1_layerwise_delta():
    """Plot S1 layer-wise delta mean and std (simple line chart)"""
    n_layers = 16

    # Use any S1 result (they should be similar across methods)
    # Pick SimNPO mid as reference
    model = get_model_for_lr("SimNPO", "2e5")
    if not model:
        print("Warning: Could not find SimNPO model for S1 analysis")
        return

    results = load_results(model)
    if not results:
        print("Warning: Could not load results for S1 analysis")
        return

    # Collect S1 deltas per layer using s1_details
    layer_s1_deltas = defaultdict(list)
    for r in results:
        if r.get("s1_details"):
            for layer_data in r["s1_details"]:
                layer = layer_data["layer"]
                s1_d = layer_data.get("delta")
                if s1_d is not None:
                    layer_s1_deltas[layer].append(s1_d)

    if not layer_s1_deltas:
        print("Warning: No S1 delta data found")
        return

    # Calculate mean and std
    layers = sorted(layer_s1_deltas.keys())
    means = [np.mean(layer_s1_deltas[l]) for l in layers]
    stds = [np.std(layer_s1_deltas[l]) for l in layers]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean with std as shaded area
    ax.plot(layers, means, 'b-o', linewidth=2, markersize=6, label='Mean Δ_S1')
    ax.fill_between(layers, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                    alpha=0.3, color='blue', label='±1 Std')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('S1 Delta (Δ)', fontsize=12)
    ax.set_title('S1 Layer-wise Delta Distribution\n(How much knowledge is stored per layer?)',
                fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 15.5)
    ax.set_xticks(range(16))
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "s1_layerwise_delta.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 's1_layerwise_delta.png'}")


# ============================================================================
# 5. UDR vs various factors
# ============================================================================
def plot_udr_vs_ft_layer_count():
    """Plot UDR vs number of FT layers"""
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    methods = list(METHOD_CATEGORIES.keys())

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]

        # Use mid LR
        model = get_model_for_lr(method, "2e5")
        if model:
            results = load_results(model)
            if results:
                ft_counts = []
                udrs = []
                for r in results:
                    if r.get("udr") is not None and r.get("ft_layers"):
                        ft_counts.append(len(r["ft_layers"]))
                        udrs.append(r["udr"])

                if ft_counts and udrs:
                    ax.scatter(ft_counts, udrs, alpha=0.5, s=20)

                    # Add trend line
                    z = np.polyfit(ft_counts, udrs, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(ft_counts), max(ft_counts), 100)
                    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)

                    # Correlation
                    corr = np.corrcoef(ft_counts, udrs)[0, 1]
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                           fontsize=10, va='top', fontweight='bold')

        ax.set_xlabel('# FT Layers')
        ax.set_ylabel('UDR')
        ax.set_title(f'{method}', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.suptitle('UDR vs Number of FT Layers (lr=2e-5, α=5, τ=0.05)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "udr_vs_ft_layer_count.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'udr_vs_ft_layer_count.png'}")


def plot_udr_vs_entity_length():
    """Plot UDR vs entity token length"""
    # Load data file to get entity lengths
    data_path = Path("tofu_data/forget10_filtered_v7_gt.json")
    if not data_path.exists():
        print(f"Warning: Data file not found: {data_path}")
        return

    with open(data_path) as f:
        data = json.load(f)

    # Create idx -> entity length mapping
    idx_to_entity_len = {}
    for item in data:
        idx = item["idx"]
        entity = item.get("entity", "")
        # Approximate token length (rough: 1 token ~ 4 chars)
        idx_to_entity_len[idx] = len(entity.split())  # word count as proxy

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    methods = list(METHOD_CATEGORIES.keys())

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]

        model = get_model_for_lr(method, "2e5")
        if model:
            results = load_results(model)
            if results:
                entity_lens = []
                udrs = []
                for r in results:
                    if r.get("udr") is not None and r.get("ft_layers"):
                        idx = r["idx"]
                        if idx in idx_to_entity_len:
                            entity_lens.append(idx_to_entity_len[idx])
                            udrs.append(r["udr"])

                if entity_lens and udrs:
                    ax.scatter(entity_lens, udrs, alpha=0.5, s=20)

                    # Correlation
                    corr = np.corrcoef(entity_lens, udrs)[0, 1]
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                           fontsize=10, va='top', fontweight='bold')

        ax.set_xlabel('Entity Word Count')
        ax.set_ylabel('UDR')
        ax.set_title(f'{method}', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.suptitle('UDR vs Entity Length (lr=2e-5, α=5, τ=0.05)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "udr_vs_entity_length.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'udr_vs_entity_length.png'}")


def plot_udr_by_question_type():
    """Plot UDR distribution by question type (based on question patterns)"""
    # Load data file
    data_path = Path("tofu_data/forget10_filtered_v7_gt.json")
    if not data_path.exists():
        print(f"Warning: Data file not found: {data_path}")
        return

    with open(data_path) as f:
        data = json.load(f)

    # Categorize questions
    def get_question_type(question):
        q_lower = question.lower()
        if q_lower.startswith("what is the full name"):
            return "Name"
        elif "born" in q_lower or "birth" in q_lower:
            return "Birth"
        elif "genre" in q_lower or "style" in q_lower:
            return "Genre/Style"
        elif "book" in q_lower or "novel" in q_lower or "work" in q_lower:
            return "Works"
        elif "parent" in q_lower or "father" in q_lower or "mother" in q_lower:
            return "Family"
        elif "award" in q_lower or "prize" in q_lower:
            return "Awards"
        else:
            return "Other"

    idx_to_qtype = {item["idx"]: get_question_type(item["question"]) for item in data}

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    methods = list(METHOD_CATEGORIES.keys())
    q_types = ["Name", "Birth", "Genre/Style", "Works", "Family", "Awards", "Other"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(q_types)))

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]

        model = get_model_for_lr(method, "2e5")
        if model:
            results = load_results(model)
            if results:
                # Group UDRs by question type
                type_udrs = defaultdict(list)
                for r in results:
                    if r.get("udr") is not None and r.get("ft_layers"):
                        idx = r["idx"]
                        if idx in idx_to_qtype:
                            type_udrs[idx_to_qtype[idx]].append(r["udr"])

                # Plot boxplot
                data_to_plot = []
                labels_to_plot = []
                for qt in q_types:
                    if type_udrs[qt]:
                        data_to_plot.append(type_udrs[qt])
                        labels_to_plot.append(f"{qt}\n(n={len(type_udrs[qt])})")

                if data_to_plot:
                    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
                    for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

        ax.set_ylabel('UDR')
        ax.set_title(f'{method}', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('UDR by Question Type (lr=2e-5, α=5, τ=0.05)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "udr_by_question_type.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'udr_by_question_type.png'}")


def plot_udr_vs_s1_delta_sum():
    """Plot UDR vs total S1 delta (knowledge amount)

    Purpose: Investigate if examples with more knowledge to erase (higher S1 sum)
    have different UDR patterns - i.e., is it harder to unlearn more knowledge?
    """
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    methods = list(METHOD_CATEGORIES.keys())

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]

        model = get_model_for_lr(method, "2e5")
        if model:
            results = load_results(model)
            if results:
                s1_sums = []
                udrs = []
                for r in results:
                    if r.get("udr") is not None and r.get("ft_layers") and r.get("s1_details"):
                        # Sum of S1 deltas for FT layers (delta > tau)
                        s1_sum = sum(d.get("delta", 0) or 0
                                    for d in r["s1_details"]
                                    if (d.get("delta") or 0) > 0.05)
                        s1_sums.append(s1_sum)
                        udrs.append(r["udr"])

                if s1_sums and udrs:
                    ax.scatter(s1_sums, udrs, alpha=0.5, s=20)

                    # Correlation
                    corr = np.corrcoef(s1_sums, udrs)[0, 1]
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                           fontsize=10, va='top', fontweight='bold')

        ax.set_xlabel('Sum of S1 Δ (knowledge amount)')
        ax.set_ylabel('UDR')
        ax.set_title(f'{method}', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.suptitle('UDR vs Total S1 Delta (lr=2e-5, α=5, τ=0.05)\nHigher S1 sum = more knowledge to erase',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "udr_vs_s1_delta_sum.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'udr_vs_s1_delta_sum.png'}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating advanced UDR visualizations (v2)...")
    print("=" * 60)

    # 1. Parallel coordinates (10 methods)
    plot_parallel_coordinates()

    # 2. Layer-wise UDR line chart
    plot_layerwise_udr_linechart()
    plot_layerwise_udr_rmu_only()

    # 3. UDR CDF by LR
    plot_udr_cdf_by_lr()

    # 4. S1 layer-wise delta
    plot_s1_layerwise_delta()

    # 5. UDR vs various factors
    plot_udr_vs_ft_layer_count()
    plot_udr_vs_entity_length()
    plot_udr_by_question_type()
    plot_udr_vs_s1_delta_sum()

    print("=" * 60)
    print("Done!")

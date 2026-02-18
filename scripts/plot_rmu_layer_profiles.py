#!/usr/bin/env python3
"""Plot RMU layer-variant UDS delta profiles (Appendix D.5).

3-panel figure showing how RMU's target layer (L5, L10, L15) shapes the
per-layer S2 delta profile.  Each panel shows:
  - S1 baseline (retain -> full, gray dashed)
  - Per-model S2 lines grouped by learning rate
  - Vertical dashed line at the RMU target layer

Generates three figures:
  1. rmu_layer_profiles.png        — raw S2 delta (absolute scale)
  2. rmu_layer_profiles_ratio.png  — S2/S1 ratio (uncapped, shows magnitude of overshoot)
  3. rmu_layer_profiles_clipped.png — clip(S2/S1, 0, 1) per layer (what enters UDS)

Usage:
    python scripts/plot_rmu_layer_profiles.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.plot_style import apply_style

apply_style()

BASE = Path("runs")
OUT_DIR = Path("runs/meta_eval")


def load_rmu_profiles():
    """Load per-layer delta profiles from all RMU models."""
    layer_variants = {5: {}, 10: {}, 15: {}}
    n_layers = 16

    for epoch in ["ep5", "ep10"]:
        uds_dir = BASE / epoch / "uds"
        for model_dir in sorted(uds_dir.glob("rmu_*")):
            name = model_dir.name
            parts = name.split("_")
            layer = None
            lr_str = None
            for p in parts:
                if p.startswith("l") and p[1:].isdigit():
                    layer = int(p[1:])
                if p.startswith("lr"):
                    lr_str = p[2:]
            if layer is None:
                continue

            results_path = model_dir / "results.json"
            if not results_path.exists():
                continue

            with open(results_path) as f:
                data = json.load(f)

            valid = [ex for ex in data if ex["uds"] is not None]
            if not valid:
                continue

            s2_deltas = np.array(
                [[ex["s2_details"][l]["delta"] for l in range(n_layers)] for ex in valid]
            )
            uds_vals = np.array([ex["uds"] for ex in valid])

            layer_variants[layer][name] = {
                "s2_mean": s2_deltas.mean(axis=0),
                "s2_std": s2_deltas.std(axis=0),
                "avg_uds": float(uds_vals.mean()),
                "lr": lr_str,
                "epoch": epoch,
                "n_valid": len(valid),
            }

    # S1 baseline (shared, take from any model)
    any_path = next((BASE / "ep10" / "uds").glob("rmu_*")) / "results.json"
    with open(any_path) as f:
        data = json.load(f)
    s1_all = np.array(
        [[ex["s1_details"][l]["delta"] for l in range(n_layers)] for ex in data]
    )
    s1_mean = s1_all.mean(axis=0)

    return layer_variants, s1_mean, n_layers


def _sort_key(item):
    """Sort models by (lr, epoch) for consistent legend ordering."""
    lr_order = {"1e5": 0, "2e5": 1, "5e5": 2}
    return (lr_order.get(item[1]["lr"], 9), item[1]["epoch"])


def main():
    layer_variants, s1_mean, n_layers = load_rmu_profiles()
    layers = np.arange(n_layers)

    # Color scheme by learning rate
    lr_colors = {
        "1e5": "#2196F3",   # blue
        "2e5": "#FF9800",   # orange
        "5e5": "#E53935",   # red
    }
    lr_labels = {
        "1e5": r"$1\!\times\!10^{-5}$",
        "2e5": r"$2\!\times\!10^{-5}$",
        "5e5": r"$5\!\times\!10^{-5}$",
    }
    epoch_ls = {"ep5": "--", "ep10": "-"}
    epoch_display = {"ep5": "5ep", "ep10": "10ep"}

    # ---------------------------------------------------------------
    # Figure 1: Raw S2 deltas
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)

    for panel_idx, target_layer in enumerate([5, 10, 15]):
        ax = axes[panel_idx]
        models = layer_variants[target_layer]

        # S1 baseline
        ax.plot(layers, s1_mean, color="gray", linestyle=":", linewidth=2,
                label=r"S1 (retain $\rightarrow$ full)", zorder=2)

        # Per-model S2 lines
        for model_name, info in sorted(models.items(), key=_sort_key):
            lr = info["lr"]
            ep = info["epoch"]
            color = lr_colors.get(lr, "black")
            ls = epoch_ls.get(ep, "-")
            uds = info["avg_uds"]
            label = f"lr={lr_labels.get(lr, lr)}, {epoch_display[ep]} (UDS={uds:.2f})"
            ax.plot(layers, info["s2_mean"], color=color, linestyle=ls,
                    linewidth=1.8, label=label, zorder=3, marker="o", markersize=3)

        # Vertical line at RMU target layer
        ymax = ax.get_ylim()[1]
        ax.axvline(x=target_layer, color="black", linestyle="-.", linewidth=1.2,
                    alpha=0.5, zorder=1)

        ax.set_xlabel("Layer")
        if panel_idx == 0:
            ax.set_ylabel(r"Mean $\Delta^{\mathrm{S2}}_l$ (log-prob degradation)")
        ax.set_title(f"RMU Layer {target_layer}", fontweight="bold")
        ax.set_xticks(layers)

        # Place target label after axis limits are settled
        ax.autoscale_view()
        ylo, yhi = ax.get_ylim()
        ax.text(target_layer + 0.25, yhi * 0.92, "target",
                fontsize=9, color="black", alpha=0.6, va="top",
                fontstyle="italic")

        ax.legend(fontsize=7.5, loc="upper left")

    plt.suptitle(
        r"RMU Layer-Variant $\Delta^{\mathrm{S2}}$ Profiles: "
        r"How Target Layer Shapes Knowledge Disruption",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = OUT_DIR / "rmu_layer_profiles.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # ---------------------------------------------------------------
    # Figure 2: S2/S1 ratio (uncapped) — per-panel y-axis
    # ---------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)

    for panel_idx, target_layer in enumerate([5, 10, 15]):
        ax = axes2[panel_idx]
        models = layer_variants[target_layer]

        # Reference line at ratio=1
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5,
                    label=r"ratio = 1 (retain-level)", zorder=1)

        for model_name, info in sorted(models.items(), key=_sort_key):
            lr = info["lr"]
            ep = info["epoch"]
            color = lr_colors.get(lr, "black")
            ls = epoch_ls.get(ep, "-")
            uds = info["avg_uds"]

            ratio = np.where(s1_mean > 0.05,
                             info["s2_mean"] / s1_mean, np.nan)
            label = f"lr={lr_labels.get(lr, lr)}, {epoch_display[ep]} (UDS={uds:.2f})"
            ax.plot(layers, ratio, color=color, linestyle=ls,
                    linewidth=1.8, label=label, zorder=3, marker="o", markersize=3)

        ax.axvline(x=target_layer, color="black", linestyle="-.", linewidth=1.2,
                    alpha=0.5, zorder=1)

        ax.set_xlabel("Layer")
        if panel_idx == 0:
            ax.set_ylabel(r"$\Delta^{\mathrm{S2}}_l \,/\, \Delta^{\mathrm{S1}}_l$")
        ax.set_title(f"RMU Layer {target_layer}", fontweight="bold")
        ax.set_xticks(layers)
        ax.set_ylim(-0.5, None)
        ax.legend(fontsize=7.5, loc="upper left")

    plt.suptitle(
        r"RMU S2/S1 Ratio (uncapped): Values $>1$ Indicate Disruption Beyond Retain Baseline",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path2 = OUT_DIR / "rmu_layer_profiles_ratio.png"
    plt.savefig(out_path2, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path2}")
    plt.close()

    # ---------------------------------------------------------------
    # Figure 3: Clipped ratio clip(S2/S1, 0, 1) — what enters UDS
    # ---------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for panel_idx, target_layer in enumerate([5, 10, 15]):
        ax = axes3[panel_idx]
        models = layer_variants[target_layer]

        # Reference lines
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5,
                    label=r"Fully erased (retain-level)", zorder=1)
        ax.axhline(y=0.0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3,
                    zorder=1)

        for model_name, info in sorted(models.items(), key=_sort_key):
            lr = info["lr"]
            ep = info["epoch"]
            color = lr_colors.get(lr, "black")
            ls = epoch_ls.get(ep, "-")
            uds = info["avg_uds"]

            clipped = np.where(
                s1_mean > 0.05,
                np.clip(info["s2_mean"] / s1_mean, 0, 1),
                np.nan,
            )
            label = f"lr={lr_labels.get(lr, lr)}, {epoch_display[ep]} (UDS={uds:.2f})"
            ax.plot(layers, clipped, color=color, linestyle=ls,
                    linewidth=1.8, label=label, zorder=3, marker="o", markersize=3)

        ax.axvline(x=target_layer, color="black", linestyle="-.", linewidth=1.2,
                    alpha=0.5, zorder=1)

        # FT layer shading: layers with s1 > threshold
        ft_mask = s1_mean > 0.05
        for l in range(n_layers):
            if ft_mask[l]:
                ax.axvspan(l - 0.4, l + 0.4, color="#e8f5e9", alpha=0.3, zorder=0)

        ax.set_xlabel("Layer")
        if panel_idx == 0:
            ax.set_ylabel(r"$\mathrm{clip}(\Delta^{\mathrm{S2}}_l / \Delta^{\mathrm{S1}}_l,\, 0,\, 1)$")
        ax.set_title(f"RMU Layer {target_layer}", fontweight="bold")
        ax.set_xticks(layers)
        ax.set_ylim(-0.05, 1.12)
        ax.legend(fontsize=7.5, loc="center right")

    plt.suptitle(
        r"RMU Clipped Erasure Ratio per Layer (enters UDS formula)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path3 = OUT_DIR / "rmu_layer_profiles_clipped.png"
    plt.savefig(out_path3, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path3}")
    plt.close()


if __name__ == "__main__":
    main()

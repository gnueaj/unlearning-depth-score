#!/usr/bin/env python3
"""D.6 Layer-Selective Unlearning: Single-panel clipped ratio plot.

Plots per-layer clip(Δ^S2/Δ^S1, 0, 1) for 6 RMU models in one figure:
  lr=2e-5 × {L5, L10, L15} (ep10)
  lr=5e-5 × {L5, L10, L15} (ep10)

Visual encoding:
  Hue       = learning rate  (blue = 2e-5, red = 5e-5)
  Saturation = target layer  (L5 vivid → L15 washed-out)
  Linestyle  = target layer  (L5 solid, L10 dashed, L15 dotted)

Usage:
    python scripts/plot_rmu_d6_clipped.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.plot_style import apply_style

apply_style()

BASE = Path("runs/ep10/uds")
OUT = Path("runs/meta_eval")

MODELS = [
    ("2e-5", 5,  "rmu_lr2e5_l5_s10_ep10"),
    ("2e-5", 10, "rmu_lr2e5_l10_s10_ep10"),
    ("2e-5", 15, "rmu_lr2e5_l15_s10_ep10"),
    ("5e-5", 5,  "rmu_lr5e5_l5_s10_ep10"),
    ("5e-5", 10, "rmu_lr5e5_l10_s10_ep10"),
    ("5e-5", 15, "rmu_lr5e5_l15_s10_ep10"),
]

N_LAYERS = 16
TAU = 0.05


def load_deltas(model_name):
    """Load per-example S1 and S2 deltas from results.json."""
    path = BASE / model_name / "results.json"
    with open(path) as f:
        data = json.load(f)
    valid = [ex for ex in data if ex["uds"] is not None]
    s1 = np.array(
        [[ex["s1_details"][l]["delta"] for l in range(N_LAYERS)] for ex in valid]
    )
    s2 = np.array(
        [[ex["s2_details"][l]["delta"] for l in range(N_LAYERS)] for ex in valid]
    )
    return s1, s2


def main():
    # Load S1 baseline (shared across all models)
    s1_all, _ = load_deltas(MODELS[0][2])
    s1_mean = s1_all.mean(axis=0)
    layers = np.arange(N_LAYERS)

    # --- Color design: Hue = LR, Saturation = target layer ---
    # HSV: (H, S, V)
    # LR=2e-5 → blue (H≈0.58), LR=5e-5 → red (H≈0.02)
    # L5 → S=1.0 (vivid), L10 → S=0.55, L15 → S=0.25 (washed-out)
    hues = {"2e-5": 0.58, "5e-5": 0.02}
    sats = {5: 1.0, 10: 0.55, 15: 0.25}

    def make_color(lr, target):
        return mcolors.hsv_to_rgb([hues[lr], sats[target], 0.82])

    ls_map = {5: "-", 10: "--", 15: ":"}
    marker_map = {5: "o", 10: "s", 15: "^"}

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    # Reference: fully erased
    ax.axhline(
        y=1.0, color="gray", linestyle=":", linewidth=1.2,
        label="Fully erased (retain-level)", zorder=1,
    )

    # FT layer shading
    ft_mask = s1_mean > TAU
    for l in range(N_LAYERS):
        if ft_mask[l]:
            ax.axvspan(l - 0.4, l + 0.4, color="#e8f5e9", alpha=0.3, zorder=0)

    for lr, target, model_name in MODELS:
        s1_ex, s2_ex = load_deltas(model_name)

        # Per-example clipped ratio, then average across examples
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(s1_ex > TAU, s2_ex / s1_ex, np.nan)
        clipped = np.clip(ratios, 0, 1)
        clipped_mean = np.nanmean(clipped, axis=0)  # (16,)

        color = make_color(lr, target)
        label = rf"lr={lr}, L{target}"
        ax.plot(
            layers, clipped_mean,
            color=color, linestyle=ls_map[target],
            linewidth=2.2, label=label, zorder=3,
            marker=marker_map[target], markersize=5,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel(
        r"$\mathrm{clip}\!\left("
        r"\Delta^{\mathrm{S2}}_l \,/\, \Delta^{\mathrm{S1}}_l"
        r",\; 0,\; 1\right)$"
    )
    ax.set_title(
        r"Layer-Selective Unlearning (RMU): Per-Layer Erasure Profile",
        fontweight="bold",
    )
    ax.set_xticks(layers)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(fontsize=10, loc="center right", ncol=1)

    plt.tight_layout()
    out_path = OUT / "rmu_d6_clipped.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.with_suffix('.pdf')}")
    plt.close()


if __name__ == "__main__":
    main()

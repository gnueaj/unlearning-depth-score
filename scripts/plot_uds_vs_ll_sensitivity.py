#!/usr/bin/env python3
"""
UDS vs Logit Lens sensitivity comparison.

Left:  QQ plot — quantile stretching
Right: KDE overlay — distribution spread
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Load ───────────────────────────────────────────────────────────────────
with open("runs/meta_eval/robustness/quant/results.json") as f:
    quant = json.load(f)
with open("runs/meta_eval/representation_baselines/logit_lens/results.json") as f:
    ll_data = json.load(f)

uds, ll = [], []
for model in quant:
    if "retain" in model:
        continue
    uds.append(quant[model]["metrics_before"]["uds"])
    ll.append(ll_data[model]["logit_lens"])
uds = np.array(uds)
ll = np.array(ll)

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.subplots_adjust(wspace=0.35, left=0.08, right=0.96, top=0.90, bottom=0.13)

C_UDS = "#1565C0"
C_LL = "#E65100"

# ── (A) QQ plot ────────────────────────────────────────────────────────────
ax = axes[0]

# Sort each independently to get quantiles
uds_q = np.sort(uds)
ll_q = np.sort(ll)

ax.scatter(ll_q, uds_q, s=18, c="#37474F", alpha=0.6, edgecolors="white",
           linewidths=0.3, zorder=3)
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3, label="slope = 1 (identical spread)")

# Linear fit through quantiles
sl, ic, r, p, _ = stats.linregress(ll_q, uds_q)
x_fit = np.linspace(0, 1, 100)
ax.plot(x_fit, sl * x_fit + ic, color="#D32F2F", lw=1.8, alpha=0.85,
        label=f"slope = {sl:.3f}")

ax.set_xlabel("Logit Lens quantiles", fontsize=11)
ax.set_ylabel("UDS quantiles", fontsize=11)
ax.set_title("(A) QQ Plot", fontsize=12, fontweight="bold")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.legend(fontsize=9, loc="upper left")

# Annotate interpretation
ax.text(0.95, 0.08, "slope > 1 → UDS has\nwider dynamic range",
        fontsize=8.5, color="#555", transform=ax.transAxes, ha="right",
        va="bottom", style="italic")

# ── (B) KDE overlay ───────────────────────────────────────────────────────
ax = axes[1]

from scipy.stats import gaussian_kde

x_grid = np.linspace(-0.05, 1.05, 300)
kde_uds = gaussian_kde(uds, bw_method=0.12)
kde_ll = gaussian_kde(ll, bw_method=0.12)

ax.fill_between(x_grid, kde_uds(x_grid), color=C_UDS, alpha=0.20)
ax.plot(x_grid, kde_uds(x_grid), color=C_UDS, lw=2,
        label=f"UDS  (σ={uds.std():.3f}, IQR={np.percentile(uds,75)-np.percentile(uds,25):.3f})")

ax.fill_between(x_grid, kde_ll(x_grid), color=C_LL, alpha=0.20)
ax.plot(x_grid, kde_ll(x_grid), color=C_LL, lw=2,
        label=f"LL    (σ={ll.std():.3f}, IQR={np.percentile(ll,75)-np.percentile(ll,25):.3f})")

# Mark IQR ranges
for arr, color, y_off in [(uds, C_UDS, -0.06), (ll, C_LL, -0.12)]:
    q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
    ax.plot([q25, q75], [y_off, y_off], color=color, lw=3, solid_capstyle="round")
    ax.plot([np.median(arr)], [y_off], "o", color=color, ms=5, zorder=4)

ax.set_xlabel("Score  (1 = erased, 0 = intact)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("(B) Score Distribution", fontsize=12, fontweight="bold")
ax.set_xlim(-0.05, 1.05)
ax.legend(fontsize=8.5, loc="upper right")
ax.set_ylim(bottom=ax.get_ylim()[0] - 0.1)

# ── Save ───────────────────────────────────────────────────────────────────
out = "runs/meta_eval/uds_vs_ll_sensitivity.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
print(f"\nQQ slope = {sl:.4f} (>1 means UDS more spread)")
print(f"σ ratio = {uds.std()/ll.std():.3f}")
print(f"IQR: UDS={np.percentile(uds,75)-np.percentile(uds,25):.4f}  "
      f"LL={np.percentile(ll,75)-np.percentile(ll,25):.4f}  "
      f"ratio={( np.percentile(uds,75)-np.percentile(uds,25)) / (np.percentile(ll,75)-np.percentile(ll,25)):.3f}")

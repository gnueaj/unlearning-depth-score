#!/usr/bin/env python3
"""Visualize rank-ordered spread for UDS vs Logit Lens over 150 models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_style import apply_style


def main() -> None:
    apply_style()

    repo = Path(__file__).resolve().parents[1]
    relearn_path = repo / "runs/meta_eval/robustness/relearn/results.json"
    rep_path = repo / "runs/meta_eval/robustness/relearn/rep_baselines_results.json"

    with open(relearn_path) as f:
        relearn = json.load(f)
    with open(rep_path) as f:
        rep = json.load(f)

    mids = sorted((set(relearn.keys()) - {"retain"}) & (set(rep.keys()) - {"retain"}))

    uds_vals = []
    ll_vals = []
    used_models = []
    for mid in mids:
        u_raw = relearn[mid].get("metrics_before", {}).get("uds")
        l_raw = rep[mid].get("logit_lens_before")
        if u_raw is None or l_raw is None:
            continue
        # knowledge direction (higher = more retained knowledge)
        uds_vals.append(1.0 - float(u_raw))
        ll_vals.append(1.0 - float(l_raw))
        used_models.append(mid)

    uds = np.array(uds_vals, dtype=float)
    ll = np.array(ll_vals, dtype=float)

    if len(uds) == 0:
        raise RuntimeError("No overlapping model data found for UDS and Logit Lens.")

    colors = ["#1f77b4", "#ff7f0e"]  # blue / orange
    fig, ax = plt.subplots(1, 1, figsize=(7.9, 4.8))

    # Single panel: rank-ordered curves (spread)
    uds_sorted = np.sort(uds)
    ll_sorted = np.sort(ll)
    rank = np.arange(1, len(uds_sorted) + 1)
    ax.plot(rank, uds_sorted, color=colors[0], lw=2.3, label="UDS (Ours)")
    ax.plot(rank, ll_sorted, color=colors[1], lw=2.0, label="Logit Lens")
    ax.set_xlim(1, len(rank))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model Rank (sorted)")
    ax.set_ylabel("Score")
    ax.set_title("Rank-Ordered Spread (150 Models)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", frameon=True)
    ax.text(
        0.98,
        0.06,
        (
            f"Std Dev.: UDS={uds.std(ddof=1):.3f}, LL={ll.std(ddof=1):.3f}\n"
            f"IQR: UDS={np.subtract(*np.quantile(uds, [0.75, 0.25])):.3f}, "
            f"LL={np.subtract(*np.quantile(ll, [0.75, 0.25])):.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )
    fig.tight_layout()

    out_dir = repo / "runs/meta_eval/representation_baselines/plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "uds_vs_logitlens_model_sensitivity.png"
    pdf_path = out_dir / "uds_vs_logitlens_model_sensitivity.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    docs_figs = repo / "docs/figs"
    docs_figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(docs_figs / "uds_vs_logitlens_model_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(docs_figs / "uds_vs_logitlens_model_sensitivity.pdf", bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

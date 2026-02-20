#!/usr/bin/env python3
"""Figure 3: Faithfulness P/N histograms for 4 representation-level metrics.
1×4 horizontal layout, double-column width. NO inversion (erasure direction).
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.plot_style import apply_style
apply_style()


def _compute_best_threshold(p_scores, n_scores, p_higher=True):
    """Pick midpoint threshold that maximizes classification accuracy.

    p_higher=True:  P should be >= threshold (standard direction)
    p_higher=False: P should be <  threshold (inverted direction)
    """
    if len(p_scores) < 2 or len(n_scores) < 2:
        return None
    all_scores = sorted(set(p_scores + n_scores))
    if len(all_scores) < 2:
        return None
    candidates = []
    for i in range(len(all_scores) - 1):
        t = (all_scores[i] + all_scores[i + 1]) / 2.0
        if p_higher:
            tp = sum(1 for s in p_scores if s >= t)
            tn = sum(1 for s in n_scores if s < t)
        else:
            tp = sum(1 for s in p_scores if s < t)
            tn = sum(1 for s in n_scores if s >= t)
        acc = (tp + tn) / (len(p_scores) + len(n_scores))
        candidates.append((acc, t))
    if not candidates:
        return None
    best_acc = max(c[0] for c in candidates)
    # Among ties, pick threshold closest to midpoint of data range
    mid = (all_scores[0] + all_scores[-1]) / 2.0
    ties = [(abs(t - mid), t) for acc, t in candidates if acc == best_acc]
    ties.sort()
    return ties[0][1]


# ---------- Resolve paths ----------
base = Path(__file__).resolve().parents[3]
rep_result_path = base / "runs/meta_eval/faithfulness/rep_baselines_results.json"
rep_summary_path = base / "runs/meta_eval/faithfulness/rep_baselines_summary.json"
faith_result_path = base / "runs/meta_eval/faithfulness/results.json"
faith_summary_path = base / "runs/meta_eval/faithfulness/summary.json"

with open(rep_result_path) as f:
    rep_results = json.load(f)
with open(rep_summary_path) as f:
    rep_auc_data = json.load(f)
with open(faith_result_path) as f:
    faith_results = json.load(f)
with open(faith_summary_path) as f:
    faith_auc_data = json.load(f)

# 4 metrics in order
METRICS = ['cka', 'fisher_masked_0.001', 'logit_lens', 'uds']
METRIC_NAMES = {
    'cka': 'CKA',
    'fisher_masked_0.001': 'Fisher',
    'logit_lens': 'Logit Lens',
    'uds': 'UDS (Ours)',
}

# ---------- Collect P/N scores per metric ----------
metric_scores = {}
for metric in METRICS:
    p_scores, n_scores = [], []
    if metric == 'uds':
        for model_id, entry in faith_results.items():
            pool = entry.get('pool')
            if pool not in ('P', 'N'):
                continue
            val = entry.get('metrics', {}).get('uds')
            if val is None:
                continue
            (p_scores if pool == 'P' else n_scores).append(val)
    else:
        for model_id, entry in rep_results.items():
            pool = entry.get('pool')
            if pool not in ('P', 'N'):
                continue
            val = entry.get(metric)
            if val is None:
                continue
            (p_scores if pool == 'P' else n_scores).append(val)
    metric_scores[metric] = (p_scores, n_scores)


def plot_histogram(ax, metric):
    p_scores, n_scores = metric_scores[metric]
    if not p_scores or not n_scores:
        ax.axis('off')
        return

    bins = np.linspace(min(min(p_scores), min(n_scores)),
                       max(max(p_scores), max(n_scores)), 15)
    ax.hist(p_scores, bins=bins, alpha=0.6, label='P (with know.)', color='green')
    ax.hist(n_scores, bins=bins, alpha=0.6, label='N (no know.)', color='red')

    # Threshold: P has LOWER values in erasure direction → p_higher=False
    threshold = _compute_best_threshold(p_scores, n_scores, p_higher=False)
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                   dashes=(3, 2), label='Threshold')
        xlo, xhi = ax.get_xlim()
        dx = (xhi - xlo) * 0.02
        if threshold > xlo + (xhi - xlo) * 0.65:
            ha_, off = 'right', -dx
        else:
            ha_, off = 'left', dx
        ax.text(threshold + off, 0.55, f'{threshold:.2f}',
                transform=ax.get_xaxis_transform(),
                fontsize=12, va='top', ha=ha_, color='black', zorder=10,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    if metric == 'uds':
        auc = faith_auc_data.get('uds', {}).get('auc_roc', 0)
    else:
        auc = rep_auc_data.get(metric, {}).get('auc_roc', 0)

    name = METRIC_NAMES[metric]
    is_ours = (metric == 'uds')
    ax.set_title(f'{name} (AUC: {auc:.3f})', fontweight='bold' if is_ours else 'normal',
                 fontsize=15)
    ax.set_ylabel('Count', fontsize=13)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)
    ax.legend(fontsize=12, loc='upper right', handlelength=1.9)


out_dir = Path(__file__).parent

# --- Version 1: 2×2 (single-column) ---
fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))
for idx, metric in enumerate(METRICS):
    plot_histogram(axes[idx // 2][idx % 2], metric)
plt.tight_layout()
plt.savefig(out_dir / 'figure3_faithfulness_2x2.png', dpi=150, bbox_inches='tight')
plt.savefig(out_dir / 'figure3_faithfulness_2x2.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {out_dir}/figure3_faithfulness_2x2.png")

# --- Version 2: 1×4 (double-column) ---
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for idx, metric in enumerate(METRICS):
    plot_histogram(axes[idx], metric)
plt.tight_layout()
plt.savefig(out_dir / 'figure3_faithfulness_1x4.png', dpi=150, bbox_inches='tight')
plt.savefig(out_dir / 'figure3_faithfulness_1x4.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {out_dir}/figure3_faithfulness_1x4.png")

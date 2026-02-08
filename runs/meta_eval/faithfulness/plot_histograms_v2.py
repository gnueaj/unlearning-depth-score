#!/usr/bin/env python3
"""
Plot faithfulness histograms for 17 metrics (v2).
Row 1: em, es, prob, paraprob
Row 2: truth_ratio, rouge, para_rouge, jailbreak_rouge
Row 3: mia_loss, mia_zlib, mia_min_k, mia_min_kpp (raw AUC)
Row 4: s_mia_loss, s_mia_zlib, s_mia_min_k, s_mia_min_kpp (scaled)
Row 5: 1-UDS centered
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path


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
    best_threshold = None
    best_accuracy = -1.0
    for i in range(len(all_scores) - 1):
        t = (all_scores[i] + all_scores[i + 1]) / 2.0
        if p_higher:
            tp = sum(1 for s in p_scores if s >= t)
            tn = sum(1 for s in n_scores if s < t)
        else:
            tp = sum(1 for s in p_scores if s < t)
            tn = sum(1 for s in n_scores if s >= t)
        acc = (tp + tn) / (len(p_scores) + len(n_scores))
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t
    return best_threshold


# ---------- Load data ----------
result_path = Path("runs/meta_eval/faithfulness/results.json")
summary_path = Path("runs/meta_eval/faithfulness/summary.json")

with open(result_path) as f:
    results = json.load(f)
with open(summary_path) as f:
    auc_data = json.load(f)

# Row 1-3: standard direction (higher = more knowledge for P)
METRICS_STANDARD = [
    'em', 'es', 'prob', 'paraprob',
    'truth_ratio', 'rouge', 'para_rouge', 'jailbreak_rouge',
    'mia_loss', 'mia_zlib', 'mia_min_k', 'mia_min_kpp',
]

# Row 4: inverted direction (higher = LESS knowledge)
METRICS_SCALED = [
    's_mia_loss', 's_mia_zlib', 's_mia_min_k', 's_mia_min_kpp',
]

METRIC_NAMES = {
    'em': 'Exact Memorization', 'es': 'Extraction Strength',
    'prob': 'Probability', 'paraprob': 'Paraphrase Prob.',
    'truth_ratio': 'Truth Ratio',
    'rouge': 'ROUGE', 'para_rouge': 'Paraphrase ROUGE',
    'jailbreak_rouge': 'Jailbreak ROUGE',
    'mia_loss': 'MIA-LOSS (raw AUC)', 'mia_zlib': 'MIA-ZLib (raw AUC)',
    'mia_min_k': 'MIA-MinK (raw AUC)', 'mia_min_kpp': 'MIA-MinK++ (raw AUC)',
    's_mia_loss': 'MIA-LOSS (normalized)',
    's_mia_zlib': 'MIA-ZLib (normalized)',
    's_mia_min_k': 'MIA-MinK (normalized)',
    's_mia_min_kpp': 'MIA-MinK++ (normalized)',
    'uds': '1\u2212UDS (Ours)',
}

# ---------- Output directory ----------
out_dir = Path("runs/meta_eval/faithfulness/histograms")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- Create figure: 5 rows × 4 cols ----------
fig, axes = plt.subplots(5, 4, figsize=(16, 19))
axes_flat = axes.flatten()


def plot_metric(ax, metric, p_higher=True, bold_title=False, title_fontsize=None):
    """Plot histogram for one metric.

    p_higher: if True, P pool has higher values; if False, P has lower values.
    """
    p_scores, n_scores = [], []
    for model_id, entry in results.items():
        pool = entry.get('pool')
        if pool not in ('P', 'N'):
            continue
        val = entry.get('metrics', {}).get(metric)
        if val is None:
            continue
        if not p_higher:
            val = 1 - val  # flip so P is always on the right
        if pool == 'P':
            p_scores.append(val)
        else:
            n_scores.append(val)

    if not p_scores or not n_scores:
        ax.axis('off')
        return

    bins = np.linspace(min(min(p_scores), min(n_scores)),
                       max(max(p_scores), max(n_scores)), 15)
    ax.hist(p_scores, bins=bins, alpha=0.6, label='P (with know.)', color='green')
    ax.hist(n_scores, bins=bins, alpha=0.6, label='N (no know.)', color='red')

    threshold = _compute_best_threshold(p_scores, n_scores, p_higher=True)
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5)

    auc = auc_data.get(metric, {}).get('auc_roc', 0)
    name = METRIC_NAMES.get(metric, metric)
    fw = 'bold' if bold_title else 'normal'
    fs = title_fontsize or plt.rcParams['axes.titlesize']
    ax.set_title(f'{name}\nAUC: {auc:.3f}', fontweight=fw, fontsize=fs)
    ax.set_ylabel('Count')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)
    ax.legend(fontsize=7)


# Rows 1-3: standard metrics (12, P has higher values)
for i, metric in enumerate(METRICS_STANDARD):
    plot_metric(axes_flat[i], metric, p_higher=True)

# Row 4: scaled MIA (raw s_mia values; P has LOWER values)
for i, metric in enumerate(METRICS_SCALED):
    plot_metric(axes_flat[12 + i], metric, p_higher=False)

# Row 5: hide all 4, then put UDS centered
for i in range(16, 20):
    axes_flat[i].axis('off')

plt.tight_layout()
fig.subplots_adjust(top=0.92, bottom=0.04, hspace=0.50)

# Create centered UDS axes in row 5 space
ref_bbox = axes_flat[13].get_position()  # row 4, col 2
ax_width = ref_bbox.width
ax_height = ref_bbox.height
center_x = 0.5 - ax_width / 2
y_pos = ref_bbox.y0 - ax_height - 0.055

ax_uds = fig.add_axes([center_x, y_pos, ax_width, ax_height])

# UDS: show 1-UDS (P has high 1-UDS = lots of knowledge remains)
p_scores, n_scores = [], []
for model_id, entry in results.items():
    pool = entry.get('pool')
    if pool not in ('P', 'N'):
        continue
    val = entry.get('metrics', {}).get('uds')
    if val is None:
        continue
    val = 1 - val  # 1-UDS
    if pool == 'P':
        p_scores.append(val)
    else:
        n_scores.append(val)

if p_scores and n_scores:
    bins = np.linspace(min(min(p_scores), min(n_scores)),
                       max(max(p_scores), max(n_scores)), 15)
    ax_uds.hist(p_scores, bins=bins, alpha=0.6, label='P (with know.)', color='green')
    ax_uds.hist(n_scores, bins=bins, alpha=0.6, label='N (no know.)', color='red')
    threshold = _compute_best_threshold(p_scores, n_scores, p_higher=True)
    if threshold is not None:
        ax_uds.axvline(threshold, color='black', linestyle='--', linewidth=1.5)
    auc = auc_data.get('uds', {}).get('auc_roc', 0)
    ax_uds.set_title(f'1\u2212UDS (Ours)\nAUC: {auc:.3f}', fontweight='bold')
    ax_uds.set_ylabel('Count')
    ax_uds.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_uds.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)
    ax_uds.legend(fontsize=7)

plt.suptitle('Faithfulness: P/N Pool Score Distributions (17 Metrics)\n'
             '60 models (30 P + 30 N)',
             fontsize=14, y=0.98)
out_name = 'faithfulness_all_metrics_v2'
plt.savefig(out_dir / f'{out_name}.png', dpi=150)
plt.savefig(out_dir / f'{out_name}.pdf')
plt.close()
print(f"Saved: {out_dir}/{out_name}.png")
print(f"Saved: {out_dir}/{out_name}.pdf")

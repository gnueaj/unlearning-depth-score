#!/usr/bin/env python3
"""
Plot faithfulness histograms for 13 metrics.
Same as original all_13metrics_histogram.png, but UDS centered in last row.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path


def _compute_best_threshold(p_scores, n_scores):
    """Pick midpoint threshold that maximizes classification accuracy."""
    if len(p_scores) < 2 or len(n_scores) < 2:
        return None

    all_scores = sorted(set(p_scores + n_scores))
    if len(all_scores) < 2:
        return None

    best_threshold = None
    best_accuracy = -1.0
    for i in range(len(all_scores) - 1):
        t = (all_scores[i] + all_scores[i + 1]) / 2.0
        tp = sum(1 for s in p_scores if s >= t)
        tn = sum(1 for s in n_scores if s < t)
        acc = (tp + tn) / (len(p_scores) + len(n_scores))
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t
    return best_threshold


# ---------- Load data ----------
result_path = Path("runs/meta_eval/faithfulness/results.json")
summary_path = Path("runs/meta_eval/faithfulness/summary.json")
uds_v2_path = Path("runs/meta_eval/faithfulness_uds_v2.json")

with open(result_path) as f:
    results = json.load(f)
with open(summary_path) as f:
    auc_data = json.load(f)

# Merge new UDS data (from s1_cache_v2)
if uds_v2_path.exists():
    with open(uds_v2_path) as f:
        uds_v2 = json.load(f)
    # Update UDS values in results
    for model_id, data in uds_v2.get("results", {}).items():
        short_id = model_id.split("/")[-1] if "/" in model_id else model_id
        # Find matching model in results
        for res_id in results:
            if short_id in res_id:
                if "metrics" not in results[res_id]:
                    results[res_id]["metrics"] = {}
                results[res_id]["metrics"]["uds"] = data.get("uds")
                break
    # Update AUC for UDS
    auc_data["uds"] = {"auc_roc": uds_v2.get("auc_roc", 0)}

# First 12 metrics (excluding UDS)
METRICS_12 = [
    'em', 'es', 'prob', 'paraprob', 'truth_ratio',
    'rouge', 'para_rouge', 'jailbreak_rouge',
    'mia_loss', 'mia_zlib', 'mia_min_k', 'mia_min_kpp',
]

METRIC_NAMES = {
    'em': 'Exact Memorization', 'es': 'Extraction Strength',
    'prob': 'Probability', 'paraprob': 'Paraphrase Probability',
    'truth_ratio': 'Truth Ratio',
    'rouge': 'ROUGE', 'para_rouge': 'Paraphrase ROUGE', 'jailbreak_rouge': 'Jailbreak ROUGE',
    'mia_loss': 'MIA-LOSS', 'mia_zlib': 'MIA-ZLib',
    'mia_min_k': 'MIA-MinK', 'mia_min_kpp': 'MIA-MinK++',
    'uds': '1-UDS (Ours)'
}

# ---------- Output directory ----------
out_dir = Path("runs/meta_eval/faithfulness/histograms")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- Create figure with 4 rows ----------
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes_flat = axes.flatten()

# Plot first 12 metrics
for i, metric in enumerate(METRICS_12):
    ax = axes_flat[i]

    p_scores = []
    n_scores = []

    for model_id, entry in results.items():
        pool = entry.get('pool')
        if pool not in ('P', 'N'):
            continue
        val = entry.get('metrics', {}).get(metric)
        if val is None:
            continue
        if pool == 'P':
            p_scores.append(val)
        else:
            n_scores.append(val)

    if len(p_scores) > 0 and len(n_scores) > 0:
        bins = np.linspace(min(min(p_scores), min(n_scores)),
                           max(max(p_scores), max(n_scores)), 15)
        ax.hist(p_scores, bins=bins, alpha=0.6, label='P (with know.)', color='green')
        ax.hist(n_scores, bins=bins, alpha=0.6, label='N (no know.)', color='red')

        threshold = _compute_best_threshold(p_scores, n_scores)
        if threshold is not None:
            ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5)

        auc = auc_data.get(metric, {}).get('auc_roc', 0)
        ax.set_title(f'{METRIC_NAMES.get(metric, metric)}\nAUC: {auc:.3f}')
        ax.set_ylabel('Count')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)

# Hide all of row 4
for i in range(12, 16):
    axes_flat[i].axis('off')

# Create centered UDS axes manually
plt.tight_layout()
fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.35)

# Get position reference from row 3
ref_bbox = axes_flat[9].get_position()  # MIA-ZLib position
ax_width = ref_bbox.width
ax_height = ref_bbox.height

# Calculate center x position (center of figure)
center_x = 0.5 - ax_width / 2
# Y position below row 3
y_pos = ref_bbox.y0 - ax_height - 0.06

ax_uds = fig.add_axes([center_x, y_pos, ax_width, ax_height])

p_scores = []
n_scores = []
for model_id, entry in results.items():
    pool = entry.get('pool')
    if pool not in ('P', 'N'):
        continue
    val = entry.get('metrics', {}).get('uds')
    if val is None:
        continue
    val = 1 - val  # Convert to knowledge score
    if pool == 'P':
        p_scores.append(val)
    else:
        n_scores.append(val)

if len(p_scores) > 0 and len(n_scores) > 0:
    bins = np.linspace(min(min(p_scores), min(n_scores)),
                       max(max(p_scores), max(n_scores)), 15)
    ax_uds.hist(p_scores, bins=bins, alpha=0.6, label='P (with know.)', color='green')
    ax_uds.hist(n_scores, bins=bins, alpha=0.6, label='N (no know.)', color='red')

    threshold = _compute_best_threshold(p_scores, n_scores)
    if threshold is not None:
        ax_uds.axvline(threshold, color='black', linestyle='--', linewidth=1.5)

    auc = auc_data.get('uds', {}).get('auc_roc', 0)
    ax_uds.set_title(f'1-UDS (Ours)\nAUC: {auc:.3f}')
    ax_uds.set_ylabel('Count')
    ax_uds.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_uds.legend(fontsize=8)

plt.suptitle('Faithfulness: P/N Pool Score Distributions (13 Metrics)\n60 models (30 P + 30 N)', fontsize=14)
plt.savefig(out_dir / 'faithfulness_all_metrics.png', dpi=150)
plt.savefig(out_dir / 'faithfulness_all_metrics.pdf')
plt.close()
print(f"Saved: {out_dir}/faithfulness_all_metrics.png")

print("\nDone!")

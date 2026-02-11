#!/usr/bin/env python3
"""
Plot faithfulness histograms for 13 metrics + 4 normalized MIA + 4 representation baselines (v2).
Row 1: em, es, prob, paraprob
Row 2: truth_ratio, rouge, para_rouge, jailbreak_rouge
Row 3: mia_loss, mia_zlib, mia_min_k, mia_min_kpp (raw AUC)
Row 4: s_mia_loss, s_mia_zlib, s_mia_min_k, s_mia_min_kpp (scaled)
Row 5: CKA, Logit Lens, Fisher Masked (0.1%), 1-UDS (Ours)
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


# ---------- Load data ----------
result_path = Path("runs/meta_eval/faithfulness/results.json")
summary_path = Path("runs/meta_eval/faithfulness/summary.json")

with open(result_path) as f:
    results = json.load(f)
with open(summary_path) as f:
    auc_data = json.load(f)

# Load representation baselines
rep_result_path = Path("runs/meta_eval/faithfulness/rep_baselines_results.json")
rep_summary_path = Path("runs/meta_eval/faithfulness/rep_baselines_summary.json")

with open(rep_result_path) as f:
    rep_results = json.load(f)
with open(rep_summary_path) as f:
    rep_auc_data = json.load(f)

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
    'uds': '$1-$UDS (Ours)',
}

# Row 5: representation baselines + UDS
REP_METHODS = ['cka', 'fisher_masked_0.001', 'logit_lens']
REP_NAMES = {
    'cka': '$1-$CKA',
    'fisher_masked_0.001': '$1-$Fisher Masked (0.1\\%)',
    'logit_lens': '$1-$Logit Lens',
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
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                   dashes=(5, 3), label='Threshold')
        xlo, xhi = ax.get_xlim()
        dx = (xhi - xlo) * 0.02
        if threshold > xlo + (xhi - xlo) * 0.65:
            ha_, off = 'right', -dx
        else:
            ha_, off = 'left', dx
        ax.text(threshold + off, 0.65, f'{threshold:.2f}',
                transform=ax.get_xaxis_transform(),
                fontsize=14, va='top', ha=ha_, color='black', zorder=10,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    auc = auc_data.get(metric, {}).get('auc_roc', 0)
    name = METRIC_NAMES.get(metric, metric)
    fw = 'bold' if bold_title else 'normal'
    fs = title_fontsize or plt.rcParams['axes.titlesize']
    ax.set_title(f'{name}\nAUC: {auc:.3f}', fontweight=fw, fontsize=fs)
    ax.set_ylabel('Count')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)
    ax.legend(fontsize=11, loc='upper right', handlelength=1.9)



# Threshold overrides (in raw/erasure space; will be converted to 1-score space)
THRESHOLD_OVERRIDES = {
    'fisher_masked_0.001': 0.6420,
}


def plot_rep_baseline(ax, method, bold_title=False):
    """Plot histogram for a representation baseline metric.

    Scores are erasure-direction (higher = more erasure = less knowledge).
    Flip to 1-score so P is on the right (more knowledge).
    """
    p_scores, n_scores = [], []
    for model_id, entry in rep_results.items():
        pool = entry.get('pool')
        if pool not in ('P', 'N'):
            continue
        val = entry.get(method)
        if val is None:
            continue
        val = 1 - val  # flip: higher = more knowledge
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

    if method in THRESHOLD_OVERRIDES:
        threshold = 1.0 - THRESHOLD_OVERRIDES[method]  # convert raw→1-score space
    else:
        threshold = _compute_best_threshold(p_scores, n_scores, p_higher=True)
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                   dashes=(5, 3), label='Threshold')
        xlo, xhi = ax.get_xlim()
        dx = (xhi - xlo) * 0.02
        if threshold > xlo + (xhi - xlo) * 0.65:
            ha_, off = 'right', -dx
        else:
            ha_, off = 'left', dx
        ax.text(threshold + off, 0.65, f'{threshold:.2f}',
                transform=ax.get_xaxis_transform(),
                fontsize=14, va='top', ha=ha_, color='black', zorder=10,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    auc = rep_auc_data.get(method, {}).get('auc_roc', 0)
    name = REP_NAMES.get(method, method)
    fw = 'bold' if bold_title else 'normal'
    ax.set_title(f'{name}\nAUC: {auc:.3f}', fontweight=fw)
    ax.set_ylabel('Count')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)
    ax.legend(fontsize=11, loc='upper right', handlelength=1.9)


# Rows 1-3: standard metrics (12, P has higher values)
for i, metric in enumerate(METRICS_STANDARD):
    plot_metric(axes_flat[i], metric, p_higher=True)

# Row 4: scaled MIA (raw s_mia values; P has LOWER values)
for i, metric in enumerate(METRICS_SCALED):
    plot_metric(axes_flat[12 + i], metric, p_higher=False)

# Row 5: CKA, Fisher, Logit Lens, UDS
# Positions 16, 17, 18 = representation baselines
for i, method in enumerate(REP_METHODS):
    plot_rep_baseline(axes_flat[16 + i], method)

# Position 19 = UDS (1-UDS, P has higher values)
ax_uds = axes_flat[19]
p_scores_uds, n_scores_uds = [], []
for model_id, entry in results.items():
    pool = entry.get('pool')
    if pool not in ('P', 'N'):
        continue
    val = entry.get('metrics', {}).get('uds')
    if val is None:
        continue
    val = 1 - val  # 1-UDS: higher = more knowledge
    if pool == 'P':
        p_scores_uds.append(val)
    else:
        n_scores_uds.append(val)

if p_scores_uds and n_scores_uds:
    bins = np.linspace(min(min(p_scores_uds), min(n_scores_uds)),
                       max(max(p_scores_uds), max(n_scores_uds)), 15)
    ax_uds.hist(p_scores_uds, bins=bins, alpha=0.6, label='P (with know.)', color='green')
    ax_uds.hist(n_scores_uds, bins=bins, alpha=0.6, label='N (no know.)', color='red')
    threshold = _compute_best_threshold(p_scores_uds, n_scores_uds, p_higher=True)
    if threshold is not None:
        ax_uds.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                       dashes=(5, 3), label='Threshold')
        xlo, xhi = ax_uds.get_xlim()
        dx = (xhi - xlo) * 0.02
        if threshold > xlo + (xhi - xlo) * 0.65:
            ha_, off = 'right', -dx
        else:
            ha_, off = 'left', dx
        ax_uds.text(threshold + off, 0.65, f'{threshold:.2f}',
                    transform=ax_uds.get_xaxis_transform(),
                    fontsize=14, va='top', ha=ha_, color='black', zorder=10,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    auc = auc_data.get('uds', {}).get('auc_roc', 0)
    ax_uds.set_title(f'$1-$UDS (Ours)\nAUC: {auc:.3f}', fontweight='bold')
    ax_uds.set_ylabel('Count')
    ax_uds.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_uds.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)
    ax_uds.legend(fontsize=11, loc='upper right', handlelength=1.9)

plt.suptitle('Faithfulness: P/N Pool Score Distributions (60 models: 30 P + 30 N)\n'
             '13 Metrics + 4 Normalized MIA + 3 Representation Baselines',
             fontsize=15, y=0.99)
plt.tight_layout()
fig.subplots_adjust(top=0.93, bottom=0.04, hspace=0.38, wspace=0.28)
out_name = 'faithfulness_all_metrics_v2'
plt.savefig(out_dir / f'{out_name}.png', dpi=150, bbox_inches='tight')
plt.savefig(out_dir / f'{out_name}.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {out_dir}/{out_name}.png")
print(f"Saved: {out_dir}/{out_name}.pdf")

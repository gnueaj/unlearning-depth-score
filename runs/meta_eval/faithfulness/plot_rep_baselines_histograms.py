#!/usr/bin/env python3
"""
Plot faithfulness histograms for representation baselines (CKA, Logit Lens, Fisher).
Also includes threshold sensitivity analysis comparing with UDS.
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.plot_style import apply_style
apply_style()


def _compute_best_threshold(p_scores, n_scores, p_higher=True):
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
base_dir = Path("runs/meta_eval/faithfulness")
result_path = base_dir / "results.json"
summary_path = base_dir / "summary.json"

with open(result_path) as f:
    results = json.load(f)
with open(summary_path) as f:
    auc_data = json.load(f)

# Also load UDS faithfulness for comparison
uds_summary_path = Path("runs/meta_eval/faithfulness/summary.json")
with open(uds_summary_path) as f:
    uds_auc_data = json.load(f)
uds_results_path = Path("runs/meta_eval/faithfulness/results.json")
with open(uds_results_path) as f:
    uds_results = json.load(f)

METHODS = ['cka', 'logit_lens', 'fisher']
METHOD_NAMES = {
    'cka': 'CKA (Geometry)',
    'logit_lens': 'Logit Lens (Fixed Decoder)',
    'fisher': 'Fisher Information',
}

# ---------- Figure 1: Histograms (1 row × 4 cols: CKA, LL, Fisher, UDS) ----------
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

for i, method in enumerate(METHODS):
    ax = axes[i]
    # Score direction: higher = more erasure (retain-like = no knowledge)
    # P pool should have LOWER scores (has knowledge → 0)
    # N pool should have HIGHER scores (no knowledge → 1)
    p_scores = [v[method] for v in results.values()
                if v['pool'] == 'P' and v.get(method) is not None]
    n_scores = [v[method] for v in results.values()
                if v['pool'] == 'N' and v.get(method) is not None]

    if not p_scores or not n_scores:
        ax.axis('off')
        continue

    # Flip to 1-score so P is on the right (higher = more knowledge)
    p_flip = [1 - s for s in p_scores]
    n_flip = [1 - s for s in n_scores]

    bins = np.linspace(min(min(p_flip), min(n_flip)),
                       max(max(p_flip), max(n_flip)), 15)
    ax.hist(p_flip, bins=bins, alpha=0.6, label='P (with know.)', color='green')
    ax.hist(n_flip, bins=bins, alpha=0.6, label='N (no know.)', color='red')

    threshold = _compute_best_threshold(p_flip, n_flip, p_higher=True)
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5)

    auc = auc_data.get(method, {}).get('auc_roc', 0)
    name = METHOD_NAMES.get(method, method)
    ax.set_title(f'{name}\nAUC: {auc:.3f}')
    ax.set_ylabel('Count')
    ax.set_xlabel(f'$1 - $ {method} score')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)
    ax.legend(fontsize=9)

# 4th panel: UDS for comparison
ax = axes[3]
p_scores_uds, n_scores_uds = [], []
for model_id, entry in uds_results.items():
    pool = entry.get('pool')
    if pool not in ('P', 'N'):
        continue
    val = entry.get('metrics', {}).get('uds')
    if val is None:
        continue
    val = 1 - val  # 1-UDS (P has high 1-UDS)
    if pool == 'P':
        p_scores_uds.append(val)
    else:
        n_scores_uds.append(val)

if p_scores_uds and n_scores_uds:
    bins = np.linspace(min(min(p_scores_uds), min(n_scores_uds)),
                       max(max(p_scores_uds), max(n_scores_uds)), 15)
    ax.hist(p_scores_uds, bins=bins, alpha=0.6, label='P (with know.)', color='green')
    ax.hist(n_scores_uds, bins=bins, alpha=0.6, label='N (no know.)', color='red')
    threshold = _compute_best_threshold(p_scores_uds, n_scores_uds, p_higher=True)
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5)
    auc = uds_auc_data.get('uds', {}).get('auc_roc', 0)
    ax.set_title(f'$1 -$ UDS (Ours)\nAUC: {auc:.3f}', fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_xlabel('$1 -$ UDS')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)
    ax.legend(fontsize=9)

plt.suptitle('Faithfulness: Representation Baselines vs UDS\n'
             '60 models (30 P + 30 N)', fontsize=14, y=1.02)
plt.tight_layout()
out_path = base_dir / "faithfulness_histograms.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

# ---------- Figure 2: Threshold sensitivity analysis ----------
# Compare delta/FT layer distributions: UDS vs Logit Lens
# Load anchor cache for LL d_ret distribution
anchor_path = Path("runs/meta_eval/representation_baselines/anchor/anchor_cache.json")
with open(anchor_path) as f:
    anchor = json.load(f)

# LL: d_ret per layer per example
ll_full = {int(k): v for k, v in anchor['ll_full'].items()}
ll_ret = {int(k): v for k, v in anchor['ll_ret'].items()}
n_layers = len(ll_full)
n_examples = len(ll_full[0])

# Collect all d_ret values (LL)
ll_d_rets = []
for l in range(n_layers):
    for i in range(n_examples):
        d_ret = ll_full[l][i] - ll_ret[l][i]
        ll_d_rets.append(d_ret)

# Load UDS S1 cache for UDS delta distribution
s1_cache_path = Path("runs/meta_eval/s1_cache_eager.json")
uds_d_rets = []
if s1_cache_path.exists():
    with open(s1_cache_path) as f:
        s1_cache = json.load(f)
    for idx, entry in s1_cache.items():
        for delta_val in entry.get('s1_deltas', []):
            uds_d_rets.append(delta_val)

# CKA: weight = 1 - CKA(full, ret)
cka_weights = []
if 'cka_full_ret' in anchor:
    for l in sorted(anchor['cka_full_ret'].keys(), key=int):
        w = 1.0 - anchor['cka_full_ret'][l]
        cka_weights.append(w)

# Fisher: excess = F_ret - F_full (use 'mean' aggregation)
fisher_excess = []
if 'fisher_full' in anchor and 'fisher_ret' in anchor:
    for l in sorted(anchor['fisher_full'].keys(), key=int):
        f_full = anchor['fisher_full'][l]
        f_ret = anchor['fisher_ret'][l]
        # Handle both scalar and dict formats
        if isinstance(f_full, dict):
            f_full = f_full.get('mean', 0)
        if isinstance(f_ret, dict):
            f_ret = f_ret.get('mean', 0)
        excess = max(f_ret - f_full, 0)
        fisher_excess.append(excess)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: UDS delta distribution
ax = axes[0, 0]
if uds_d_rets:
    ax.hist(uds_d_rets, bins=50, alpha=0.7, color='blue')
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label=r'$\tau$ = 0.05')
    n_above = sum(1 for d in uds_d_rets if d > 0.05)
    n_total = len(uds_d_rets)
    ax.set_title(f'UDS: S1 Delta Distribution\n'
                 f'{n_above}/{n_total} ({100*n_above/n_total:.1f}\%) above $\\tau$=0.05')
    ax.set_xlabel(r'$\Delta^{S1}_l$ (full logprob $-$ patched logprob)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)

# Panel 2: LL d_ret distribution
ax = axes[0, 1]
ax.hist(ll_d_rets, bins=50, alpha=0.7, color='orange')
ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label=r'$\tau$ = 0.05')
n_above = sum(1 for d in ll_d_rets if d > 0.05)
n_total = len(ll_d_rets)
ax.set_title(f'Logit Lens: $d_{{ret}}$ Distribution\n'
             f'{n_above}/{n_total} ({100*n_above/n_total:.1f}\%) above $\\tau$=0.05')
ax.set_xlabel(r'$d_{\mathrm{ret}} = k_{\mathrm{full}} - k_{\mathrm{ret}}$ (fixed decoder logprob gap)')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)

# Panel 3: CKA weight per layer
ax = axes[1, 0]
if cka_weights:
    layers = list(range(len(cka_weights)))
    ax.bar(layers, cka_weights, color='green', alpha=0.7)
    ax.set_title(f'CKA: Layer Weight = $1 -$ CKA(full, retain)\n'
                 f'All {len(cka_weights)} layers are FT (weight $>$ 0)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Weight')
    ax.set_xticks(layers)
    ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)

# Panel 4: Fisher excess per layer
ax = axes[1, 1]
if fisher_excess:
    layers = list(range(len(fisher_excess)))
    ax.bar(layers, fisher_excess, color='purple', alpha=0.7)
    ax.set_title(f'Fisher: Excess = $F_{{ret}} - F_{{full}}$ (per-param mean, log1p)\n'
                 f'All {len(fisher_excess)} layers are FT (excess $>$ 0)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Excess')
    ax.set_xticks(layers)
    ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)

plt.suptitle(r'Threshold \& FT Layer Sensitivity Analysis' '\n'
             r'UDS $\tau$=0.05 vs Logit Lens $\tau$=0.05 vs CKA/Fisher weight-based',
             fontsize=14, y=1.02)
plt.tight_layout()
out_path2 = base_dir / "threshold_sensitivity.png"
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path2}")

# ---------- Summary table ----------
print("\n=== Faithfulness AUC-ROC Comparison ===")
print(f"{'Method':20s} | {'AUC-ROC':>8s} | {'P mean':>8s} | {'N mean':>8s} | {'Separation':>10s}")
print("-" * 70)
for m in METHODS:
    d = auc_data[m]
    sep = d['N_mean'] - d['P_mean']
    print(f"{METHOD_NAMES[m]:20s} | {d['auc_roc']:8.4f} | {d['P_mean']:8.4f} | {d['N_mean']:8.4f} | {sep:10.4f}")
uds_auc = uds_auc_data.get('uds', {}).get('auc_roc', 0)
print(f"{'UDS (Ours)':20s} | {uds_auc:8.4f} |      -   |      -   |          -")

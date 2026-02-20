#!/usr/bin/env python3
"""Figure 2: Quantization stability scatter for Truth Ratio + ROUGE.
1×2 horizontal layout, single-column width. Utility filter (utility_rel >= 0.8).
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.plot_style import apply_style
apply_style()

EPS = 1e-6
UNREL_COLOR = '#C62828'
GRADIENT_MAX_BLEND = 0.65
GRADIENT_RES = 256


def compute_q(before, after):
    """Symmetric stability: 1 - |after - before| / (|before| + |after| + eps)."""
    denom = abs(before) + abs(after) + EPS
    return max(0.0, 1.0 - min(abs(after - before) / denom, 1.0))


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _draw_sym_gradient(ax, lo, hi, shift=0.0):
    """Bidirectional gradient around y = x + shift."""
    x = np.linspace(lo, hi, GRADIENT_RES)
    y = np.linspace(lo, hi, GRADIENT_RES)
    X, Y = np.meshgrid(x, y)
    dist = np.abs(Y - X - shift) / np.sqrt(2)
    max_dist = (hi - lo) / np.sqrt(2)
    if max_dist < 1e-10:
        return
    raw = np.clip(dist / max_dist, 0, 1)
    t = raw * GRADIENT_MAX_BLEND

    base_lin = _srgb_to_linear(np.array(to_rgba(UNREL_COLOR)[:3]))
    bg_lin = _srgb_to_linear(np.array([1.0, 1.0, 1.0]))

    img = np.zeros((GRADIENT_RES, GRADIENT_RES, 4))
    for c in range(3):
        blended = bg_lin[c] * (1 - t) + base_lin[c] * t
        img[..., c] = _linear_to_srgb(np.clip(blended, 0, 1))
    img[..., 3] = 1.0
    ax.imshow(img, extent=[lo, hi, lo, hi], origin='lower',
              aspect='auto', zorder=0, interpolation='bilinear')


class _GradientHandler(HandlerBase):
    """Legend handler: draws a white -> UNREL_COLOR gradient rectangle."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        N = 256
        base_lin = _srgb_to_linear(np.array(to_rgba(UNREL_COLOR)[:3]))
        t = np.linspace(0, GRADIENT_MAX_BLEND, N)
        arr = np.ones((1, N, 4))
        for c in range(3):
            blended = 1.0 * (1 - t) + base_lin[c] * t
            arr[0, :, c] = _linear_to_srgb(np.clip(blended, 0, 1))
        bb = Bbox.from_bounds(xdescent, ydescent, width, height)
        tbb = TransformedBbox(bb, trans)
        im = BboxImage(tbb)
        im.set_data(arr)
        return [im]


# ---------- Resolve paths ----------
base = Path(__file__).resolve().parents[3]
quant_path = base / 'runs/meta_eval/robustness/quant/results.json'

with open(quant_path) as f:
    quant_data = json.load(f)

# Utility filter: only models with utility_rel >= 0.8
mr_path = base / 'docs/data/method_results.json'
with open(mr_path) as f:
    mr_data = json.load(f)
utility_models = set()
for m in mr_data.get('models', []):
    if m.get('utility_rel', 0) >= 0.8:
        utility_models.add(m['model'])
quant_models = sorted((set(quant_data.keys()) - {'retain'}) & utility_models)

# 2 metrics
METRICS = ['truth_ratio', 'rouge']
METRIC_LABELS = {
    'truth_ratio': 'Truth Ratio',
    'rouge': 'ROUGE',
}

# ---------- Create figure: 1 row × 2 cols ----------
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

for idx, metric in enumerate(METRICS):
    ax = axes[idx]
    bef_list, aft_list = [], []
    q_values = []
    n_recovered, n_destroyed = 0, 0

    for model_name in quant_models:
        md = quant_data[model_name]
        mb = md.get('metrics_before', {})
        ma = md.get('metrics_after_quant', {})
        bv = mb.get(metric)
        av = ma.get(metric)
        if bv is None or av is None:
            continue
        bef_list.append(bv)
        aft_list.append(av)
        q = compute_q(bv, av)
        q_values.append(q)
        if av > bv + 1e-8:
            n_recovered += 1
        elif av < bv - 1e-8:
            n_destroyed += 1

    all_vals = bef_list + aft_list
    if not all_vals:
        ax.set_visible(False)
        continue

    # Axis limits: truth_ratio tight, others from 0
    if metric == 'truth_ratio':
        lo = max(min(all_vals) * 0.98, 0.0)
    else:
        lo = 0.0
    hi = min(max(max(all_vals) * 1.02, 1e-6), 1.02)

    _draw_sym_gradient(ax, lo, hi)
    ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.85, linewidth=1.0, zorder=2)

    ax.scatter(bef_list, aft_list, c='#2E7D32', s=16, alpha=0.82,
               edgecolors='#1B5E20', linewidths=0.35, zorder=3)

    label = METRIC_LABELS[metric]
    ax.set_title(label, fontsize=15)
    ax.set_xlabel('Before Quantization', fontsize=13)
    ax.set_ylabel('After Quantization', fontsize=13)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)

    # Highlight ROUGE collapse region
    if metric == 'rouge':
        from matplotlib.patches import Rectangle
        box_x, box_y = 0.47, 0.42
        box_w, box_h = 0.30, 0.14
        rect = Rectangle((box_x, box_y), box_w, box_h,
                          linewidth=2.0, edgecolor='#4682B4', facecolor='none', zorder=4)
        ax.add_patch(rect)

    grad_patch = Patch(label='Unreliable\n(Less $\\rightarrow$ More)')
    local_handles = [
        Line2D([0], [0], color='r', linestyle='--', linewidth=1.0, alpha=0.85, label='y = x'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E7D32',
               markeredgecolor='#1B5E20', markersize=5, label='Unlearned Model'),
        grad_patch,
    ]
    ax.legend(handles=local_handles, handler_map={grad_patch: _GradientHandler()},
              loc='lower right', fontsize=12, framealpha=0.8)

plt.tight_layout(w_pad=0.2)
out_dir = Path(__file__).parent
plt.savefig(out_dir / 'figure2_quant_scatter.png', dpi=150, bbox_inches='tight')
plt.savefig(out_dir / 'figure2_quant_scatter.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {out_dir}/figure2_quant_scatter.png")
print(f"Saved: {out_dir}/figure2_quant_scatter.pdf")

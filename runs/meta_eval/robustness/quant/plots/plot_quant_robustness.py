#!/usr/bin/env python3
"""
Create scatter plots for quantization robustness analysis (Figure 10 style).
Uses usable_models.json for per-metric filtering.
13 metrics (+ 4 normalized MIA variants).
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
import matplotlib.gridspec as gridspec
from pathlib import Path

UNREL_COLOR = '#C62828'
GRADIENT_MAX_BLEND = 0.8   # max blend ratio at farthest corner (0=white, 1=full color)
GRADIENT_RES = 256

# Reference AUC values for sMIA normalization (from ep10 privacy summaries)
RETAIN_AUC = {"loss": 0.38235312499999996, "zlib": 0.304609375, "min_k": 0.37731562500000004, "min_k++": 0.470575}
SMIA_MAP = {
    's_mia_loss': ('mia_loss', 'loss'), 's_mia_zlib': ('mia_zlib', 'zlib'),
    's_mia_min_k': ('mia_min_k', 'min_k'), 's_mia_min_kpp': ('mia_min_kpp', 'min_k++'),
}


def compute_s_mia(raw_auc, attack):
    """MUSE PrivLeak-style: clip(1 - |AUC_model - AUC_retain| / AUC_retain, 0, 1)"""
    denom = RETAIN_AUC[attack]
    if denom <= 1e-12:
        return None
    return float(np.clip(1.0 - abs(raw_auc - denom) / denom, 0.0, 1.0))


def get_metric_val(metrics_dict, metric):
    """Get metric value, computing s_mia on-the-fly if needed."""
    if metric in SMIA_MAP:
        raw_key, attack = SMIA_MAP[metric]
        raw = metrics_dict.get(raw_key)
        if raw is None:
            return None
        return compute_s_mia(raw, attack)
    return metrics_dict.get(metric)


def resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "runs" / "meta_eval").exists() and (parent / "docs").exists():
            return parent
    raise RuntimeError("Could not locate repository root from script path.")


def _srgb_to_linear(c):
    """sRGB [0,1] -> linear RGB [0,1]."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    """linear RGB [0,1] -> sRGB [0,1]."""
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _draw_unreliable_gradient(ax, lo, hi, shift=0.0):
    """Draw gradient fill for unreliable region (above y = x + shift).

    Blends white -> UNREL_COLOR in linear RGB for perceptual uniformity.
    """
    x_arr = np.linspace(lo, hi, GRADIENT_RES)
    y_arr = np.linspace(lo, hi, GRADIENT_RES)
    X, Y = np.meshgrid(x_arr, y_arr)
    above = (Y - X - shift) / np.sqrt(2)
    mask = above > 0
    max_dist = (hi - lo) / np.sqrt(2)
    if max_dist < 1e-10:
        return
    raw = np.clip(above / max_dist, 0, 1)
    t = np.where(mask, 0.02 + raw * (GRADIENT_MAX_BLEND - 0.02), 0)

    base_rgb = np.array(to_rgba(UNREL_COLOR)[:3])
    bg_rgb = np.array([1.0, 1.0, 1.0])
    base_lin = _srgb_to_linear(base_rgb)
    bg_lin = _srgb_to_linear(bg_rgb)

    img = np.zeros((GRADIENT_RES, GRADIENT_RES, 4))
    for c in range(3):
        blended = bg_lin[c] * (1 - t) + base_lin[c] * t
        img[..., c] = _linear_to_srgb(np.clip(blended, 0, 1))
    img[..., 3] = np.where(mask, 1.0, 0.0)
    ax.imshow(img, extent=[lo, hi, lo, hi], origin='lower',
              aspect='auto', zorder=0, interpolation='bilinear')


class _GradientHandler(HandlerBase):
    """Legend handler: draws a white → UNREL_COLOR gradient rectangle."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        N = 256
        base_lin = _srgb_to_linear(np.array(to_rgba(UNREL_COLOR)[:3]))
        t = np.linspace(0.02, GRADIENT_MAX_BLEND, N)
        arr = np.ones((1, N, 4))
        for c in range(3):
            blended = 1.0 * (1 - t) + base_lin[c] * t
            arr[0, :, c] = _linear_to_srgb(np.clip(blended, 0, 1))
        bb = Bbox.from_bounds(xdescent, ydescent, width, height)
        tbb = TransformedBbox(bb, trans)
        im = BboxImage(tbb)
        im.set_data(arr)
        return [im]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot quantization robustness.")
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Use all 150 unlearned models (skip utility/faithfulness filtering).",
    )
    parser.add_argument(
        "--usable_path",
        type=str,
        default=None,
        help="Path to usable_models.json (used only when --no_filter is not set).",
    )
    parser.add_argument(
        "--filter_label",
        type=str,
        default=None,
        help="Optional title label override for filtering condition.",
    )
    parser.add_argument(
        "--out_tag",
        type=str,
        default="",
        help="Optional output filename tag suffix (e.g., utility_only).",
    )
    parser.add_argument(
        "--tight_axes",
        action="store_true",
        help="Set axis lower bound to data min instead of 0.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base = resolve_repo_root()
    quant_path = base / 'runs/meta_eval/robustness/quant/results.json'
    output_dir = base / 'runs/meta_eval/robustness/quant/plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(quant_path) as f:
        quant_data = json.load(f)
    quant_models = set(quant_data.keys()) - {'retain'}

    # 13 metrics + 4 normalized MIA: rows 0-2 standard (12), row 3 sMIA (4), row 4 UDS centered
    metrics = [
        'em', 'es', 'truth_ratio', 'prob',
        'rouge', 'jailbreak_rouge', 'paraprob', 'para_rouge',
        'mia_loss', 'mia_min_k', 'mia_min_kpp', 'mia_zlib',
        's_mia_loss', 's_mia_min_k', 's_mia_min_kpp', 's_mia_zlib',
        'uds',
    ]

    metric_labels = {
        'em': 'Exact Memorization', 'es': 'Extraction Strength', 'truth_ratio': 'Truth Ratio',
        'prob': 'Prob.', 'rouge': 'ROUGE', 'jailbreak_rouge': 'Jailbreak ROUGE',
        'paraprob': 'Para. Prob.', 'para_rouge': 'Para. ROUGE',
        'mia_loss': 'MIA-LOSS (raw AUC)', 'mia_min_k': 'MIA-MinK (raw AUC)',
        'mia_min_kpp': 'MIA-MinK++ (raw AUC)', 'mia_zlib': 'MIA-ZLib (raw AUC)',
        's_mia_loss': 'MIA-LOSS (normalized)',
        's_mia_zlib': 'MIA-ZLib (normalized)',
        's_mia_min_k': 'MIA-MinK (normalized)',
        's_mia_min_kpp': 'MIA-MinK++ (normalized)',
        'uds': '1-UDS (Ours)',
    }

    # Inverted: higher = less knowledge → convert to 1-val
    inverted_metrics = {'uds', 's_mia_loss', 's_mia_zlib', 's_mia_min_k', 's_mia_min_kpp'}

    # sMIA uses same usable models as corresponding raw MIA metric
    usable_key_map = {
        's_mia_loss': 'mia_loss', 's_mia_zlib': 'mia_zlib',
        's_mia_min_k': 'mia_min_k', 's_mia_min_kpp': 'mia_min_kpp',
    }

    if args.no_filter:
        usable_per_metric = {m: sorted(quant_models) for m in metrics}
    else:
        usable_path = Path(args.usable_path) if args.usable_path else (base / 'runs/meta_eval/robustness/usable_models.json')
        with open(usable_path) as f:
            usable = json.load(f)
        usable_per_metric = usable['usable_models_per_metric']

    all_metric_data = {}
    robustness_results = {}

    for metric in metrics:
        usable_key = usable_key_map.get(metric, metric)
        usable_models = set(usable_per_metric.get(usable_key, [])) & quant_models
        filtered_before = []
        filtered_after = []
        model_names = []
        q_values = []
        n_unreliable = 0

        for model_name in sorted(usable_models):
            md = quant_data[model_name]
            mb = md.get('metrics_before', {})
            ma = md.get('metrics_after_quant', {})
            bv = get_metric_val(mb, metric)
            av = get_metric_val(ma, metric)
            if bv is None or av is None:
                continue

            if metric in inverted_metrics:
                bv = 1 - bv
                av = 1 - av

            filtered_before.append(bv)
            filtered_after.append(av)
            model_names.append(model_name)
            if av > bv:
                n_unreliable += 1

            # Q = min(before/after, 1)
            if av > 0:
                q = min(bv / av, 1.0)
            else:
                q = 1.0
            q_values.append(q)

        avg_q = float(np.mean(q_values)) if q_values else 0.0

        all_metric_data[metric] = {
            'before': filtered_before, 'after': filtered_after,
            'names': model_names, 'q_values': q_values,
            'avg_q': avg_q
        }
        robustness_results[metric] = {
            'avg_Q': round(avg_q, 4),
            'n_models': len(filtered_before),
            'n_unreliable': n_unreliable,
            'n_usable_total': len(usable_models),
            'q_per_model': {n: round(q, 4) for n, q in zip(model_names, q_values)}
        }

        print(f"{metric:20s}: n={len(filtered_before):3d}/{len(usable_models):3d}  "
              f"avg_Q={avg_q:.4f}  unrel={n_unreliable}/{len(filtered_before)}")

    results_name = 'quant_robustness_results_nofilter.json' if args.no_filter else 'quant_robustness_results.json'
    plot_name = 'quant_robustness_nofilter.png' if args.no_filter else 'quant_robustness_usable.png'
    filter_label = 'No Filtering' if args.no_filter else 'Utility + Faithfulness Filtered'
    if args.filter_label:
        filter_label = args.filter_label
    if args.tight_axes:
        filter_label += '; Tight Axes'
    if args.out_tag:
        tag = args.out_tag if args.out_tag.startswith("_") else f"_{args.out_tag}"
        results_name = results_name.replace(".json", f"{tag}.json")
        plot_name = plot_name.replace(".png", f"{tag}.png")

    # Save robustness results
    with open(output_dir / results_name, 'w') as f:
        json.dump(robustness_results, f, indent=2)
    print(f"\nSaved: {output_dir / results_name}")

    # === Plot: 5 rows × 4 cols ===
    # Row 0-2: standard 12 metrics, Row 3: sMIA 4, Row 4: UDS centered
    fig = plt.figure(figsize=(13.8, 17.8))
    gs = gridspec.GridSpec(5, 4, figure=fig)
    metric_axes = {}

    # Rows 0-2: first 12 standard metrics
    for i, metric in enumerate(metrics[:12]):
        r, c = divmod(i, 4)
        metric_axes[metric] = fig.add_subplot(gs[r, c])

    # Row 3: 4 sMIA metrics
    for i, metric in enumerate(metrics[12:16]):
        metric_axes[metric] = fig.add_subplot(gs[3, i])

    # Row 4: UDS centered
    metric_axes['uds'] = fig.add_subplot(gs[4, 1:3])
    fig.add_subplot(gs[4, 0]).set_visible(False)
    fig.add_subplot(gs[4, 3]).set_visible(False)

    # Row 4 metrics (sMIA) and row 5 (UDS) get special formatting
    smia_metrics = {'s_mia_loss', 's_mia_zlib', 's_mia_min_k', 's_mia_min_kpp'}

    for metric in metrics:
        ax = metric_axes[metric]
        d = all_metric_data[metric]
        bef, aft = d['before'], d['after']

        all_vals = bef + aft
        if not all_vals:
            ax.set_visible(False)
            continue

        if args.tight_axes:
            lo = max(min(all_vals) * 0.98, 0.0)
        else:
            lo = 0.0
        hi = max(max(all_vals) * 1.02, 1e-6)

        # Unreliable region gradient (above y=x)
        _draw_unreliable_gradient(ax, lo, hi)

        # y=x line
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.85, linewidth=1.0, zorder=2)

        # Unlearned models
        if bef:
            ax.scatter(bef, aft, c='#2E7D32', s=16, alpha=0.82,
                       edgecolors='#1B5E20', linewidths=0.35, zorder=3)

        label = metric_labels.get(metric, metric)
        n_unrel = robustness_results[metric]['n_unreliable']
        title_fs = 11
        ax.set_title(
            f"{label}\nQ={d['avg_q']:.3f} (n={len(bef)}, unrel={n_unrel})",
            fontsize=title_fs,
            fontweight='bold' if metric == 'uds' else 'normal',
        )
        ax.set_xlabel('Before Quantization', fontsize=9)
        ax.set_ylabel('After Quantization', fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)

        grad_patch = Patch(label='Unreliable\n(Less → More)')
        local_handles = [
            Line2D([0], [0], color='r', linestyle='--', linewidth=1.0, alpha=0.85, label='y = x'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E7D32',
                   markeredgecolor='#1B5E20', markersize=5, label='Unlearn'),
            grad_patch,
        ]
        ax.legend(handles=local_handles, handler_map={grad_patch: _GradientHandler()},
                  loc='lower right', fontsize=7, framealpha=0.95)

    fig.suptitle(f'Quantization Robustness (13 Metrics + 4 Normalized MIA)\n(150 Unlearned Models; {filter_label})',
                 fontsize=14, fontweight='normal', y=0.97)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.03, top=0.92, wspace=0.01, hspace=0.48)
    plt.savefig(output_dir / plot_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / plot_name}")


if __name__ == '__main__':
    main()

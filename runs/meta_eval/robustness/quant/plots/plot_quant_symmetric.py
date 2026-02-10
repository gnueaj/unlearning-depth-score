#!/usr/bin/env python3
"""
Symmetric quantization stability: Q = 1 - clip(|m_after - m_before| / (|m_before| + |m_after| + eps), 0, 1)

Bidirectional — penalizes both knowledge recovery AND destruction after 4-bit NF4 quantization.
Canberra-like denominator normalizes by scale so near-zero and large metrics are treated comparably.
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.plot_style import apply_style
apply_style()
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
import matplotlib.gridspec as gridspec

EPS = 1e-6
UNREL_COLOR = '#C62828'
GRADIENT_MAX_BLEND = 0.65
GRADIENT_RES = 256

# Reference AUC values for sMIA normalization
RETAIN_AUC = {"loss": 0.38235312499999996, "zlib": 0.304609375,
              "min_k": 0.37731562500000004, "min_k++": 0.470575}
SMIA_MAP = {
    's_mia_loss': ('mia_loss', 'loss'), 's_mia_zlib': ('mia_zlib', 'zlib'),
    's_mia_min_k': ('mia_min_k', 'min_k'), 's_mia_min_kpp': ('mia_min_kpp', 'min_k++'),
}


def compute_s_mia(raw_auc, attack):
    denom = RETAIN_AUC[attack]
    if denom <= 1e-12:
        return None
    return float(np.clip(1.0 - abs(raw_auc - denom) / denom, 0.0, 1.0))


def get_metric_val(metrics_dict, metric):
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
    raise RuntimeError("Could not locate repository root.")


def compute_q(before, after):
    """Symmetric stability: 1 - |after - before| / (|before| + |after| + eps)."""
    denom = abs(before) + abs(after) + EPS
    return max(0.0, 1.0 - min(abs(after - before) / denom, 1.0))


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _draw_sym_gradient(ax, lo, hi, shift=0.0):
    """Bidirectional gradient around y = x + shift.
    Same unreliable color on both sides — any deviation is penalized.
    """
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


def parse_args():
    parser = argparse.ArgumentParser(description="Plot symmetric quantization stability.")
    parser.add_argument("--no_filter", action="store_true",
                        help="Use all 150 unlearned models.")
    parser.add_argument("--usable_path", type=str, default=None)
    parser.add_argument("--filter_label", type=str, default=None)
    parser.add_argument("--out_tag", type=str, default="")
    parser.add_argument("--tight_axes", action="store_true")
    parser.add_argument("--utility_only", action="store_true",
                        help="Filter by utility_rel >= 0.8 only.")
    parser.add_argument("--lr_filter", type=str, default=None,
                        help="Filter to specific learning rate. Implies utility filter.")
    return parser.parse_args()


def main():
    args = parse_args()
    base = resolve_repo_root()
    quant_path = base / 'runs/meta_eval/robustness/quant/results.json'
    output_dir = base / 'runs/meta_eval/robustness/quant/plots/sym'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(quant_path) as f:
        quant_data = json.load(f)
    quant_models = set(quant_data.keys()) - {'retain'}

    metrics = [
        'em', 'es', 'truth_ratio', 'prob',
        'rouge', 'jailbreak_rouge', 'paraprob', 'para_rouge',
        'mia_loss', 'mia_min_k', 'mia_min_kpp', 'mia_zlib',
        's_mia_loss', 's_mia_min_k', 's_mia_min_kpp', 's_mia_zlib',
        'uds',
    ]
    rep_metrics = ['logit_lens']

    metric_labels = {
        'em': 'Exact Memorization', 'es': 'Extraction Strength',
        'truth_ratio': 'Truth Ratio', 'prob': 'Prob.',
        'rouge': 'ROUGE', 'jailbreak_rouge': 'Jailbreak ROUGE',
        'paraprob': 'Para. Prob.', 'para_rouge': 'Para. ROUGE',
        'mia_loss': 'MIA-LOSS (raw AUC)', 'mia_min_k': 'MIA-MinK (raw AUC)',
        'mia_min_kpp': 'MIA-MinK++ (raw AUC)', 'mia_zlib': 'MIA-ZLib (raw AUC)',
        's_mia_loss': 'MIA-LOSS (normalized)', 's_mia_zlib': 'MIA-ZLib (normalized)',
        's_mia_min_k': 'MIA-MinK (normalized)', 's_mia_min_kpp': 'MIA-MinK++ (normalized)',
        'uds': '1-UDS (Ours)', 'logit_lens': '1-Logit Lens',
    }

    inverted_metrics = {'uds', 's_mia_loss', 's_mia_zlib', 's_mia_min_k', 's_mia_min_kpp'}
    usable_key_map = {
        's_mia_loss': 'mia_loss', 's_mia_zlib': 'mia_zlib',
        's_mia_min_k': 'mia_min_k', 's_mia_min_kpp': 'mia_min_kpp',
    }

    # Load representation baselines
    rep_path = base / 'runs/meta_eval/robustness/quant/rep_baselines_results.json'
    rep_data = {}
    if rep_path.exists():
        with open(rep_path) as f:
            rep_data = json.load(f)
    rep_models = set(rep_data.keys()) - {'retain'}

    # --- Filtering ---
    utility_models = None
    if args.lr_filter or args.utility_only:
        mr_path = base / 'docs/data/method_results.json'
        with open(mr_path) as f:
            mr_data = json.load(f)
        utility_models = set()
        for m in mr_data.get('models', []):
            if m.get('utility_rel', 0) >= 0.8:
                utility_models.add(m['model'])
        if args.lr_filter:
            utility_models = {mid for mid in utility_models if f'_lr{args.lr_filter}_' in mid}
        usable_per_metric = {m: sorted(utility_models & quant_models) for m in metrics}
    elif args.no_filter:
        usable_per_metric = {m: sorted(quant_models) for m in metrics}
    else:
        usable_path = Path(args.usable_path) if args.usable_path else (
            base / 'runs/meta_eval/robustness/usable_models.json')
        with open(usable_path) as f:
            usable = json.load(f)
        usable_per_metric = usable['usable_models_per_metric']

    # --- Compute Q per metric ---
    all_metric_data = {}
    robustness_results = {}

    for metric in metrics:
        usable_key = usable_key_map.get(metric, metric)
        usable_set = set(usable_per_metric.get(usable_key, [])) & quant_models
        filtered_before, filtered_after = [], []
        model_names, q_values = [], []
        n_recovered, n_destroyed = 0, 0

        for model_name in sorted(usable_set):
            md = quant_data[model_name]
            mb = md.get('metrics_before', {})
            ma = md.get('metrics_after_quant', {})
            bv = get_metric_val(mb, metric)
            av = get_metric_val(ma, metric)
            if bv is None or av is None:
                continue

            if metric in inverted_metrics:
                bv, av = 1 - bv, 1 - av

            filtered_before.append(bv)
            filtered_after.append(av)
            model_names.append(model_name)
            q = compute_q(bv, av)
            q_values.append(q)
            if av > bv + 1e-8:
                n_recovered += 1
            elif av < bv - 1e-8:
                n_destroyed += 1

        avg_q = float(np.mean(q_values)) if q_values else 0.0
        all_metric_data[metric] = {
            'before': filtered_before, 'after': filtered_after,
            'names': model_names, 'q_values': q_values, 'avg_q': avg_q,
        }
        robustness_results[metric] = {
            'avg_Q': round(avg_q, 4),
            'n_models': len(filtered_before),
            'n_recovered': n_recovered, 'n_destroyed': n_destroyed,
            'n_usable_total': len(usable_set),
            'q_per_model': {n: round(q, 4) for n, q in zip(model_names, q_values)}
        }
        print(f"{metric:20s}: n={len(filtered_before):3d}/{len(usable_set):3d}  "
              f"Q={avg_q:.4f}  rec={n_recovered}  des={n_destroyed}")

    # --- Representation baselines ---
    rep_retain = rep_data.get('retain', {})
    for metric in rep_metrics:
        bkey, akey = f"{metric}_before", f"{metric}_after"
        rb_raw = rep_retain.get(bkey)
        ra_raw = rep_retain.get(akey)
        if rb_raw is None or ra_raw is None:
            print(f"{metric:20s}: SKIP (no retain data)")
            continue

        if args.lr_filter or args.utility_only:
            usable_set = (utility_models & rep_models) if utility_models else rep_models
        elif args.no_filter:
            usable_set = rep_models
        else:
            usable_set = set(usable_per_metric.get('uds', [])) & rep_models

        filtered_before, filtered_after = [], []
        model_names, q_values = [], []
        n_recovered, n_destroyed = 0, 0

        for model_name in sorted(usable_set):
            md = rep_data.get(model_name, {})
            bv_raw, av_raw = md.get(bkey), md.get(akey)
            if bv_raw is None or av_raw is None:
                continue
            bv, av = 1.0 - bv_raw, 1.0 - av_raw
            filtered_before.append(bv)
            filtered_after.append(av)
            model_names.append(model_name)
            q = compute_q(bv, av)
            q_values.append(q)
            if av > bv + 1e-8:
                n_recovered += 1
            elif av < bv - 1e-8:
                n_destroyed += 1

        avg_q = float(np.mean(q_values)) if q_values else 0.0
        all_metric_data[metric] = {
            'before': filtered_before, 'after': filtered_after,
            'names': model_names, 'q_values': q_values, 'avg_q': avg_q,
        }
        robustness_results[metric] = {
            'avg_Q': round(avg_q, 4),
            'n_models': len(filtered_before),
            'n_recovered': n_recovered, 'n_destroyed': n_destroyed,
            'n_usable_total': len(usable_set),
            'q_per_model': {n: round(q, 4) for n, q in zip(model_names, q_values)}
        }
        print(f"{metric:20s}: n={len(filtered_before):3d}/{len(usable_set):3d}  "
              f"Q={avg_q:.4f}  rec={n_recovered}  des={n_destroyed}")

    # --- Naming ---
    if args.lr_filter:
        results_name = 'quant_sym_results.json'
        plot_name = 'quant_sym.png'
        filter_label = f'Utility + LR={args.lr_filter} Filtered'
        if not args.out_tag:
            args.out_tag = f'lr{args.lr_filter}'
    elif args.utility_only:
        results_name = 'quant_sym_results.json'
        plot_name = 'quant_sym.png'
        filter_label = 'Utility Filtered'
        if not args.out_tag:
            args.out_tag = 'utility_only'
    elif args.no_filter:
        results_name = 'quant_sym_results.json'
        plot_name = 'quant_sym.png'
        filter_label = 'No Filtering'
        if not args.out_tag:
            args.out_tag = 'nofilter'
    else:
        results_name = 'quant_sym_results.json'
        plot_name = 'quant_sym.png'
        filter_label = 'Utility + Faithfulness Filtered'
    if args.filter_label:
        filter_label = args.filter_label
    if args.tight_axes:
        filter_label += '; Tight Axes'
    if args.out_tag:
        tag = args.out_tag if args.out_tag.startswith("_") else f"_{args.out_tag}"
        results_name = results_name.replace(".json", f"{tag}.json")
        plot_name = plot_name.replace(".png", f"{tag}.png")

    with open(output_dir / results_name, 'w') as f:
        json.dump(robustness_results, f, indent=2)
    print(f"\nSaved: {output_dir / results_name}")

    # --- Plot: 5 rows × 4 cols ---
    fig = plt.figure(figsize=(13.8, 18.0))
    gs = gridspec.GridSpec(5, 4, figure=fig)
    metric_axes = {}
    for i, m in enumerate(metrics[:12]):
        r, c = divmod(i, 4)
        metric_axes[m] = fig.add_subplot(gs[r, c])
    for i, m in enumerate(metrics[12:16]):
        metric_axes[m] = fig.add_subplot(gs[3, i])
    metric_axes['uds'] = fig.add_subplot(gs[4, 0:2])
    metric_axes['logit_lens'] = fig.add_subplot(gs[4, 2:4])

    all_plot_metrics = metrics + rep_metrics
    for metric in all_plot_metrics:
        ax = metric_axes[metric]
        d = all_metric_data.get(metric)
        if d is None:
            ax.set_visible(False)
            continue
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

        _draw_sym_gradient(ax, lo, hi)
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.85, linewidth=1.0, zorder=2)

        if bef:
            ax.scatter(bef, aft, c='#2E7D32', s=16, alpha=0.82,
                       edgecolors='#1B5E20', linewidths=0.35, zorder=3)

        label = metric_labels.get(metric, metric)
        r = robustness_results[metric]
        ax.set_title(
            f"{label}\n$Q$={d['avg_q']:.3f} (n={len(bef)}, "
            f"rec={r['n_recovered']}, des={r['n_destroyed']})",
            fontsize=11,
            fontweight='bold' if metric in ('uds', *rep_metrics) else 'normal',
        )
        ax.set_xlabel('Before Quantization', fontsize=10)
        ax.set_ylabel('After Quantization', fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)

        grad_patch = Patch(label='Unreliable\n(Less $\\rightarrow$ More)')
        local_handles = [
            Line2D([0], [0], color='r', linestyle='--', linewidth=1.0, alpha=0.85, label='y = x'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E7D32',
                   markeredgecolor='#1B5E20', markersize=5, label='Unlearn'),
            grad_patch,
        ]
        ax.legend(handles=local_handles, handler_map={grad_patch: _GradientHandler()},
                  loc='lower right', fontsize=9, framealpha=0.95)

    fig.suptitle(f'Quantization Robustness (13 Metrics + 4 Normalized MIA + Logit Lens)\n(150 Unlearned Models; {filter_label})',
                 fontsize=15, fontweight='normal', y=0.99)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.025, top=0.945,
                        wspace=0.04, hspace=0.40)
    plt.savefig(output_dir / plot_name, dpi=150, bbox_inches='tight')
    pdf_name = plot_name.replace('.png', '.pdf')
    plt.savefig(output_dir / pdf_name, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / plot_name}")
    print(f"Saved: {output_dir / pdf_name}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Symmetric relearning stability: R = 1 - clip(|Δunl - Δret| / (|Δunl| + |Δret| + eps), 0, 1)

Bidirectional — penalizes both over-recovery and under-recovery after relearning.
Axiom 2 (Recovery calibration): unlearned model's change should match retain's change.
Canberra-like normalization prevents blow-up when Δret ≈ 0.
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


def to_knowledge(metric: str, value: float) -> float:
    """Convert to knowledge direction (higher = more retained knowledge)."""
    if metric in ('uds', 's_mia_loss', 's_mia_zlib', 's_mia_min_k', 's_mia_min_kpp'):
        return 1.0 - value
    return value


def compute_r(ret_before, ret_after, unl_before, unl_after):
    """Symmetric stability: 1 - clip(|Δunl - Δret| / (|Δunl| + |Δret| + eps), 0, 1).
    All values in knowledge direction. Canberra-like denominator.
    """
    ret_delta = ret_after - ret_before
    unl_delta = unl_after - unl_before
    excess = abs(unl_delta - ret_delta)
    denom = abs(unl_delta) + abs(ret_delta) + EPS
    return max(0.0, 1.0 - min(excess / denom, 1.0))


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
    parser = argparse.ArgumentParser(description="Plot symmetric relearning stability.")
    parser.add_argument("--no_filter", action="store_true",
                        help="Use all 150 unlearned models.")
    parser.add_argument("--usable_path", type=str, default=None)
    parser.add_argument("--filter_label", type=str, default=None)
    parser.add_argument("--out_tag", type=str, default="")
    parser.add_argument("--tight_axes", action="store_true",
                        help="(Deprecated) No longer used; truth_ratio always gets tight axes.")
    parser.add_argument("--utility_only", action="store_true",
                        help="Filter by utility_rel >= 0.8 only.")
    parser.add_argument("--faithfulness_only", action="store_true",
                        help="Filter by faithfulness threshold only (no utility filter).")
    parser.add_argument("--before_filter", action="store_true",
                        help="Utility + geometric: before_knowledge < 1.0 - retain_shift.")
    return parser.parse_args()


def main():
    args = parse_args()
    base = resolve_repo_root()
    relearn_path = base / 'runs/meta_eval/robustness/relearn/results.json'
    output_dir = base / 'runs/meta_eval/robustness/relearn/plots/sym'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(relearn_path) as f:
        relearn_data = json.load(f)
    relearn_models = set(relearn_data.keys()) - {'retain'}

    metrics = [
        'em', 'es', 'truth_ratio', 'prob',
        'rouge', 'jailbreak_rouge', 'paraprob', 'para_rouge',
        'mia_loss', 'mia_min_k', 'mia_min_kpp', 'mia_zlib',
        's_mia_loss', 's_mia_min_k', 's_mia_min_kpp', 's_mia_zlib',
        'uds',
    ]
    rep_metrics = ['cka', 'logit_lens', 'fisher_masked_0.0001', 'fisher_masked_0.001', 'fisher_masked_0.01']

    metric_labels = {
        'em': 'Exact Memorization', 'es': 'Extraction Strength',
        'truth_ratio': 'Truth Ratio', 'prob': 'Prob.',
        'rouge': 'ROUGE', 'jailbreak_rouge': 'Jailbreak ROUGE',
        'paraprob': 'Para. Prob.', 'para_rouge': 'Para. ROUGE',
        'mia_loss': 'MIA-LOSS (raw AUC)', 'mia_min_k': 'MIA-MinK (raw AUC)',
        'mia_min_kpp': 'MIA-MinK++ (raw AUC)', 'mia_zlib': 'MIA-ZLib (raw AUC)',
        's_mia_loss': 'MIA-LOSS (normalized)', 's_mia_zlib': 'MIA-ZLib (normalized)',
        's_mia_min_k': 'MIA-MinK (normalized)', 's_mia_min_kpp': 'MIA-MinK++ (normalized)',
        'uds': r'\textbf{1-UDS (Ours)}', 'logit_lens': '1-Logit Lens',
        'cka': '1-CKA',
        'fisher_masked_0.0001': r'1-Fisher (0.01\%)',
        'fisher_masked_0.001': r'1-Fisher (0.1\%)',
        'fisher_masked_0.01': r'1-Fisher (1\%)',
    }

    usable_key_map = {
        's_mia_loss': 'mia_loss', 's_mia_zlib': 'mia_zlib',
        's_mia_min_k': 'mia_min_k', 's_mia_min_kpp': 'mia_min_kpp',
    }

    # Load representation baselines
    rep_path = base / 'runs/meta_eval/robustness/relearn/rep_baselines_results.json'
    rep_data = {}
    if rep_path.exists():
        with open(rep_path) as f:
            rep_data = json.load(f)
    rep_models = set(rep_data.keys()) - {'retain'}

    # --- Filtering ---
    utility_models = None
    thresholds = None
    if args.faithfulness_only:
        # Apply only faithfulness threshold per-metric (no utility filter)
        usable_path_f = base / 'runs/meta_eval/robustness/usable_models.json'
        with open(usable_path_f) as f:
            usable_meta = json.load(f)
        thresholds = usable_meta['criteria']['metric_threshold_filter']['thresholds']
        usable_per_metric = {}
        for metric in metrics:
            usable_key = usable_key_map.get(metric, metric)
            t_info = thresholds.get(usable_key)
            if t_info is None:
                usable_per_metric[usable_key] = sorted(relearn_models)
                continue
            t_val = t_info['threshold']
            passed = []
            for model_name in sorted(relearn_models):
                mb = relearn_data[model_name].get('metrics_before', {})
                val = get_metric_val(mb, metric)
                if val is None:
                    continue
                if usable_key == 'uds':
                    if val >= (1.0 - t_val):
                        passed.append(model_name)
                elif t_info.get('direction') == 'pass_if_ge':
                    if val >= t_val:
                        passed.append(model_name)
                else:
                    if val <= t_val:
                        passed.append(model_name)
            usable_per_metric[usable_key] = passed
    elif args.utility_only or args.before_filter:
        mr_path = base / 'docs/data/method_results.json'
        with open(mr_path) as f:
            mr_data = json.load(f)
        utility_models = set()
        for m in mr_data.get('models', []):
            if m.get('utility_rel', 0) >= 0.8:
                utility_models.add(m['model'])
        usable_per_metric = {m: sorted(utility_models & relearn_models) for m in metrics}
    elif args.no_filter:
        usable_per_metric = {m: sorted(relearn_models) for m in metrics}
    else:
        usable_path = Path(args.usable_path) if args.usable_path else (
            base / 'runs/meta_eval/robustness/usable_models.json')
        with open(usable_path) as f:
            usable = json.load(f)
        usable_per_metric = usable['usable_models_per_metric']

    retain = relearn_data.get('retain', {})
    retain_before_raw = retain.get('metrics_before', {})
    retain_after_raw = retain.get('metrics_after_relearn', {})

    # --- Compute R per metric ---
    all_metric_data = {}
    robustness_results = {}

    for metric in metrics:
        usable_key = usable_key_map.get(metric, metric)
        usable_set = set(usable_per_metric.get(usable_key, [])) & relearn_models

        rb_raw = get_metric_val(retain_before_raw, metric)
        ra_raw = get_metric_val(retain_after_raw, metric)
        if rb_raw is None or ra_raw is None:
            print(f"{metric:20s}: SKIP (no retain data)")
            continue

        rb = to_knowledge(metric, rb_raw)
        ra = to_knowledge(metric, ra_raw)
        retain_shift = ra - rb

        bef_all, aft_all = [], []
        model_names, r_values = [], []
        n_over, n_under = 0, 0

        for model_name in sorted(usable_set):
            md = relearn_data[model_name]
            mb = md.get('metrics_before', {})
            ma = md.get('metrics_after_relearn', {})
            bv_raw = get_metric_val(mb, metric)
            av_raw = get_metric_val(ma, metric)
            if bv_raw is None or av_raw is None:
                continue

            bv = to_knowledge(metric, bv_raw)
            if args.before_filter and bv >= 1.0 - retain_shift:
                continue
            av = to_knowledge(metric, av_raw)
            model_names.append(model_name)
            bef_all.append(bv)
            aft_all.append(av)

            r = compute_r(rb, ra, bv, av)
            r_values.append(r)

            unl_delta = av - bv
            if unl_delta > retain_shift + 1e-8:
                n_over += 1
            elif unl_delta < retain_shift - 1e-8:
                n_under += 1

        avg_r = float(np.mean(r_values)) if r_values else 0.0
        n_total = len(bef_all)

        all_metric_data[metric] = {
            'bef_all': bef_all, 'aft_all': aft_all,
            'names': model_names, 'r_values': r_values,
            'rb': rb, 'ra': ra, 'retain_shift': retain_shift,
            'avg_r': avg_r, 'avg_R': round(avg_r, 4), 'n_total': n_total,
            'n_over': n_over, 'n_under': n_under,
        }
        robustness_results[metric] = {
            'avg_R': round(avg_r, 4),
            'retain_shift': round(retain_shift, 4),
            'n_models': n_total,
            'n_over': n_over, 'n_under': n_under,
            'n_usable_total': len(usable_set),
            'r_per_model': {n: round(r, 4) for n, r in zip(model_names, r_values)}
        }

        print(f"{metric:20s}: n={n_total:3d}/{len(usable_set):3d}  "
              f"R={avg_r:.4f}  ret_shift={retain_shift:+.4f}  "
              f"over={n_over}  under={n_under}")

    # --- Representation baselines ---
    rep_retain = rep_data.get('retain', {})
    for metric in rep_metrics:
        bkey, akey = f"{metric}_before", f"{metric}_after"
        rb_raw = rep_retain.get(bkey)
        ra_raw = rep_retain.get(akey)
        if rb_raw is None or ra_raw is None:
            print(f"{metric:20s}: SKIP (no retain data)")
            continue

        rb, ra = 1.0 - rb_raw, 1.0 - ra_raw
        retain_shift = ra - rb

        if args.faithfulness_only:
            # Apply faithfulness threshold for rep baselines
            t_info = thresholds.get(metric) if thresholds else None
            if t_info is not None:
                t_val = t_info['threshold']
                usable_set = set()
                for model_name in rep_models:
                    md = rep_data.get(model_name, {})
                    bv_raw = md.get(bkey)
                    if bv_raw is None:
                        continue
                    direction = t_info.get('direction', 'pass_if_le')
                    if direction == 'pass_if_ge':
                        if bv_raw >= t_val:
                            usable_set.add(model_name)
                    else:
                        if bv_raw <= t_val:
                            usable_set.add(model_name)
            else:
                usable_set = rep_models
        elif args.utility_only or args.before_filter:
            usable_set = (utility_models & rep_models) if utility_models else rep_models
        elif args.no_filter:
            usable_set = rep_models
        else:
            if metric in usable_per_metric:
                usable_set = set(usable_per_metric[metric]) & rep_models
            else:
                usable_set = rep_models  # no faithfulness filter for unknown rep metrics

        bef_all, aft_all = [], []
        model_names, r_values = [], []
        n_over, n_under = 0, 0

        for model_name in sorted(usable_set):
            md = rep_data.get(model_name, {})
            bv_raw, av_raw = md.get(bkey), md.get(akey)
            if bv_raw is None or av_raw is None:
                continue
            bv, av = 1.0 - bv_raw, 1.0 - av_raw
            if args.before_filter and bv >= 1.0 - retain_shift:
                continue
            model_names.append(model_name)
            bef_all.append(bv)
            aft_all.append(av)

            r = compute_r(rb, ra, bv, av)
            r_values.append(r)

            unl_delta = av - bv
            if unl_delta > retain_shift + 1e-8:
                n_over += 1
            elif unl_delta < retain_shift - 1e-8:
                n_under += 1

        avg_r = float(np.mean(r_values)) if r_values else 0.0
        n_total = len(bef_all)

        all_metric_data[metric] = {
            'bef_all': bef_all, 'aft_all': aft_all,
            'names': model_names, 'r_values': r_values,
            'rb': rb, 'ra': ra, 'retain_shift': retain_shift,
            'avg_r': avg_r, 'avg_R': round(avg_r, 4), 'n_total': n_total,
            'n_over': n_over, 'n_under': n_under,
        }
        robustness_results[metric] = {
            'avg_R': round(avg_r, 4),
            'retain_shift': round(retain_shift, 4),
            'n_models': n_total,
            'n_over': n_over, 'n_under': n_under,
            'n_usable_total': len(usable_set),
            'r_per_model': {n: round(r, 4) for n, r in zip(model_names, r_values)}
        }

        print(f"{metric:20s}: n={n_total:3d}/{len(usable_set):3d}  "
              f"R={avg_r:.4f}  ret_shift={retain_shift:+.4f}  "
              f"over={n_over}  under={n_under}")

    # --- Naming ---
    if args.before_filter:
        results_name = 'relearn_sym_results.json'
        plot_name = 'relearn_sym.png'
        filter_label = 'Utility + Before-Value Filtered'
        if not args.out_tag:
            args.out_tag = 'before_filter'
    elif args.faithfulness_only:
        results_name = 'relearn_sym_results.json'
        plot_name = 'relearn_sym.png'
        filter_label = 'Faithfulness Filtered'
        if not args.out_tag:
            args.out_tag = 'faithfulness_only'
    elif args.utility_only:
        results_name = 'relearn_sym_results.json'
        plot_name = 'relearn_sym.png'
        filter_label = 'Utility Filtered'
        if not args.out_tag:
            args.out_tag = 'utility_only'
    elif args.no_filter:
        results_name = 'relearn_sym_results.json'
        plot_name = 'relearn_sym.png'
        filter_label = 'No Filtering'
        if not args.out_tag:
            args.out_tag = 'nofilter'
    else:
        results_name = 'relearn_sym_results.json'
        plot_name = 'relearn_sym.png'
        filter_label = 'Utility + Faithfulness Filtered'
    if args.filter_label:
        filter_label = args.filter_label
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
    metric_axes['cka'] = fig.add_subplot(gs[4, 0])
    metric_axes['fisher_masked_0.001'] = fig.add_subplot(gs[4, 1])
    metric_axes['logit_lens'] = fig.add_subplot(gs[4, 2])
    metric_axes['uds'] = fig.add_subplot(gs[4, 3])

    all_plot_metrics = [m for m in metrics + rep_metrics if m in metric_axes]
    for metric in all_plot_metrics:
        ax = metric_axes[metric]
        d = all_metric_data.get(metric)
        if d is None:
            ax.set_visible(False)
            continue

        rb, ra = d['rb'], d['ra']
        retain_shift = d['retain_shift']

        all_vals = d['bef_all'] + d['aft_all'] + [rb, ra]
        if not all_vals:
            ax.set_visible(False)
            continue

        if metric == 'truth_ratio':
            lo = max(min(all_vals) * 0.98, 0.0)
        else:
            lo = 0.0
        hi = min(max(max(all_vals) * 1.02, 1e-6), 1.02)

        # Symmetric gradient around y = x + retain_shift
        _draw_sym_gradient(ax, lo, hi, shift=retain_shift)

        # y=x reference
        ax.plot([lo, hi], [lo, hi], 'r:', alpha=0.65, linewidth=1.0, zorder=2)

        # y = x + retain_shift (expected behavior line)
        ax.plot([lo, hi], [lo + retain_shift, hi + retain_shift],
                'r--', alpha=0.85, linewidth=1.0, zorder=2)

        # Unlearned models
        if d['bef_all']:
            ax.scatter(d['bef_all'], d['aft_all'], c='#2E7D32', s=16, alpha=0.82,
                       edgecolors='#1B5E20', linewidths=0.35, zorder=3)

        # Retain model
        ax.scatter([rb], [ra], c='red', s=120, marker='*',
                   edgecolors='darkred', linewidths=0.5, zorder=5)

        label = metric_labels.get(metric, metric)
        is_ours = (metric == 'uds')
        if is_ours:
            title_str = (r"\textbf{" + label + "}\n"
                         r"\textbf{$\mathbf{R}$=" + f"{d['avg_R']:.3f}"
                         + r" (n=" + str(d['n_total'])
                         + r", over=" + str(d['n_over'])
                         + r", under=" + str(d['n_under']) + r")}")
        else:
            title_str = (f"{label}\n$R$={d['avg_R']:.3f} (n={d['n_total']}, "
                         f"over={d['n_over']}, under={d['n_under']})")
        ax.set_title(title_str, fontsize=11)
        ax.set_xlabel('Before Relearning', fontsize=10)
        ax.set_ylabel('After Relearning', fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.25)

        grad_patch = Patch(label='Unreliable\n(Less $\\rightarrow$ More)')
        local_handles = [
            Line2D([0], [0], color='r', linestyle='--', linewidth=1.0, alpha=0.85,
                   label=r'y = x + $\Delta_{\mathrm{ret}}$'),
            Line2D([0], [0], color='r', linestyle=':', linewidth=1.0, alpha=0.65, label='y = x'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E7D32',
                   markeredgecolor='#1B5E20', markersize=5, label='Unlearn'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markeredgecolor='darkred', markersize=8, label='Retain'),
            grad_patch,
        ]
        ax.legend(handles=local_handles, handler_map={grad_patch: _GradientHandler()},
                  loc='lower right', fontsize=8, framealpha=0.8)

    # Compute utility-only count for reference
    if utility_models is not None:
        n_util = len(utility_models & relearn_models)
    else:
        mr_path = base / 'docs/data/method_results.json'
        with open(mr_path) as f:
            mr_data = json.load(f)
        n_util = sum(1 for m in mr_data.get('models', [])
                     if m.get('utility_rel', 0) >= 0.8 and m['model'] in relearn_models)
    fig.suptitle(f'Relearning Robustness (13 Metrics + 4 Normalized MIA + 3 Rep. Baselines)\n(150 Unlearned Models; {filter_label})',
                 fontsize=15, fontweight='normal', y=0.99)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.025, top=0.935,
                        wspace=0.04, hspace=0.40)
    plt.savefig(output_dir / plot_name, dpi=150, bbox_inches='tight')
    pdf_name = plot_name.replace('.png', '.pdf')
    plt.savefig(output_dir / pdf_name, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / plot_name}")
    print(f"Saved: {output_dir / pdf_name}")


if __name__ == '__main__':
    main()

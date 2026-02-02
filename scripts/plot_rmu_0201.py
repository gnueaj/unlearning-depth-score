#!/usr/bin/env python3
import re
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path('runs')
OUT_DIR = Path('docs/0201/rmu')
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_GLOB = '0201_*_tf_rmu_*_layer/run.log'

LAYER_RE = re.compile(r"^\s*L(\d{2})\s+\|\s+logp=[^\s]+\s+Δ=([+-]?[0-9]*\.?[0-9]+)\s+\[[A-Z]+\]\s+\|\s+logp=[^\s]+\s+Δ=([+-]?[0-9]*\.?[0-9]+)")
UDR_RE = re.compile(r"UDR\s*=\s*([+-]?[0-9]*\.?[0-9]+)")
TAU_RE = re.compile(r"Delta threshold:\s*([0-9]*\.?[0-9]+)")
EXAMPLE_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+Example\s+(\d+)")


def parse_key(key: str):
    lr = re.search(r"lr([0-9]+e[0-9]+)", key)
    layer = re.search(r"_l(\d+)_", key)
    lr_val = lr.group(1) if lr else ''
    layer_val = int(layer.group(1)) if layer else -1
    lr_num = float(lr_val.replace('e', 'E')) if lr_val else 0.0
    return lr_num, layer_val, lr_val


def parse_log(path: Path):
    tau = None
    examples = []
    current = None
    in_layer_block = False
    saw_row = False
    with path.open() as f:
        for line in f:
            if tau is None:
                m = TAU_RE.search(line)
                if m:
                    tau = float(m.group(1))
            m = EXAMPLE_RE.search(line)
            if m:
                if current is not None:
                    examples.append(current)
                current = {'udr': math.nan, 'd1': {}, 'd2': {}}
                in_layer_block = False
                saw_row = False
                continue
            if line.startswith('  Layer'):
                in_layer_block = True
                saw_row = False
                continue
            if in_layer_block:
                if line.strip().startswith('------'):
                    if saw_row:
                        in_layer_block = False
                    continue
                m = LAYER_RE.match(line)
                if m and current is not None:
                    layer = int(m.group(1))
                    d1 = float(m.group(2))
                    d2 = float(m.group(3))
                    current['d1'][layer] = d1
                    current['d2'][layer] = d2
                    saw_row = True
                    continue
            m = UDR_RE.search(line)
            if m and current is not None:
                current['udr'] = float(m.group(1))
                continue
            if 'UDR = N/A' in line and current is not None:
                current['udr'] = math.nan
                continue
    if current is not None:
        examples.append(current)

    if tau is None:
        tau = 0.05

    for ex in examples:
        d1 = ex['d1']
        ex['ft_layers'] = [l for l, v in d1.items() if v > tau]
        ex['ft_count'] = len(ex['ft_layers'])

    return tau, examples


def summarize(examples):
    udr = np.array([ex['udr'] for ex in examples if not math.isnan(ex['udr'])])
    n_na = sum(1 for ex in examples if math.isnan(ex['udr']))
    return {
        'n': len(examples),
        'n_na': n_na,
        'udr_mean': float(np.mean(udr)) if udr.size else math.nan,
        'udr_median': float(np.median(udr)) if udr.size else math.nan,
    }


def layerwise_stats(examples, tau):
    layers = sorted({l for ex in examples for l in ex['d1'].keys()})
    d1_by_layer = {l: [] for l in layers}
    d2_by_layer = {l: [] for l in layers}
    ft_count_by_layer = {l: 0 for l in layers}
    total_count_by_layer = {l: 0 for l in layers}
    for ex in examples:
        for l, v in ex['d1'].items():
            d1_by_layer[l].append(v)
            d2_by_layer[l].append(ex['d2'].get(l, 0.0))
            total_count_by_layer[l] += 1
            if v > tau:
                ft_count_by_layer[l] += 1
    return layers, d1_by_layer, d2_by_layer, ft_count_by_layer, total_count_by_layer


def grid_axes():
    fig, axes = plt.subplots(3,3, figsize=(12,9), sharex=False, sharey=False)
    return fig, axes


def add_grid_labels(fig, axes, row_labels, col_labels, row_font=12, col_font=12, col_y=0.94):
    for i, label in enumerate(row_labels):
        pos = axes[i,0].get_position()
        y = (pos.y0 + pos.y1) / 2
        fig.text(0.02, y, f'lr={label}', va='center', ha='left', fontsize=row_font)
    for j, label in enumerate(col_labels):
        pos = axes[0,j].get_position()
        x = (pos.x0 + pos.x1) / 2
        fig.text(x, col_y, f'layer={label}', va='center', ha='center', fontsize=col_font)


def plot_hist_grid(data_map, mean_map, title, out_path, bins=30, xlim=(0,1), row_labels=None, col_labels=None):
    keys = list(data_map.keys())
    fig, axes = grid_axes()
    max_count = 0
    for key in keys:
        counts, _ = np.histogram(data_map[key], bins=bins, range=xlim)
        max_count = max(max_count, counts.max() if counts.size else 0)
    ymax = max_count * 1.05 if max_count else 1

    for ax, key in zip(axes.flatten(), keys):
        vals = data_map[key]
        ax.hist(vals, bins=bins, range=xlim, color='#4C78A8', alpha=0.85)
        ax.set_xlabel('UDR')
        ax.set_ylabel('Count')
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(0, ymax)
        ax.text(0.97, 0.95, f"mean={mean_map[key]:.3f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=10)
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=[0.08,0.03,1,0.90])
    if row_labels and col_labels:
        add_grid_labels(fig, axes, row_labels, col_labels, col_y=0.92)
    fig.suptitle(title, y=0.985)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cdf_grid(data_map, title, out_path, xlim=(0,1), row_labels=None, col_labels=None):
    keys = list(data_map.keys())
    fig, axes = grid_axes()
    for ax, key in zip(axes.flatten(), keys):
        vals = np.sort(np.array(data_map[key]))
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size+1) / vals.size
        ax.plot(vals, y, color='#F58518')
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(0,1)
        ax.set_xlabel('UDR')
        ax.set_ylabel('CDF')
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=[0.08,0.03,1,0.90])
    if row_labels and col_labels:
        add_grid_labels(fig, axes, row_labels, col_labels, col_y=0.92)
    fig.suptitle(title, y=0.985)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_layerwise(lines_map, title, out_path, ylabel):
    fig, ax = plt.subplots(figsize=(8,5))
    for key, (layers, vals) in lines_map.items():
        ax.plot(layers, vals, label=key, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_ft_rate(lines_map, title, out_path):
    plot_layerwise(lines_map, title, out_path, ylabel='FT rate')


def plot_line_grid(lines_map, title, out_path, ylabel, row_labels=None, col_labels=None):
    keys = list(lines_map.keys())
    fig, axes = grid_axes()
    # global y-limits for comparability
    all_vals = []
    for _, (_, vals) in lines_map.items():
        all_vals.extend(vals)
    if all_vals:
        ymin = min(all_vals)
        ymax = max(all_vals)
        pad = (ymax - ymin) * 0.05 if ymax > ymin else 0.5
    else:
        ymin, ymax, pad = 0.0, 1.0, 0.0

    for ax, key in zip(axes.flatten(), keys):
        layers, vals = lines_map[key]
        ax.plot(layers, vals, linewidth=1.5, color='#4C78A8')
        ax.set_xlabel('Layer')
        ax.set_ylabel(ylabel)
        ax.set_xticks(layers)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=[0.08,0.03,1,0.90])
    if row_labels and col_labels:
        add_grid_labels(fig, axes, row_labels, col_labels, col_y=0.92)
    fig.suptitle(title, y=0.985)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_instancewise_traj_grid(traj_map, layers, title, out_path, ylabel, row_labels=None, col_labels=None, sample_size=60):
    keys = list(traj_map.keys())
    fig, axes = grid_axes()
    rng = np.random.default_rng(0)

    for ax, key in zip(axes.flatten(), keys):
        mat = traj_map[key]  # shape (n_examples, n_layers)
        n = mat.shape[0]
        if n == 0:
            continue
        idx = rng.choice(n, size=min(sample_size, n), replace=False)
        for i in idx:
            ax.plot(layers, mat[i], color='#4C78A8', alpha=0.15, linewidth=0.8)
        ax.set_xlabel('Layer')
        ax.set_ylabel(ylabel)
        ax.set_xticks(layers)
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=[0.08,0.03,1,0.90])
    if row_labels and col_labels:
        add_grid_labels(fig, axes, row_labels, col_labels, col_y=0.92)
    fig.suptitle(title, y=0.985)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_udr_strength_trajectories(udr_by_layer, out_path, sample_size=80):
    # udr_by_layer: {layer: {strength: np.array(n_examples)}}
    strengths = ["weak", "mid", "strong"]
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True, sharey=True)
    rng = np.random.default_rng(0)

    for ax, (layer, series) in zip(axes, sorted(udr_by_layer.items())):
        # stack per-example values across strengths
        vals = np.stack([series[s] for s in strengths], axis=1)
        n = vals.shape[0]
        idx = rng.choice(n, size=min(sample_size, n), replace=False)
        for i in idx:
            ax.plot(strengths, vals[i], color='#4C78A8', alpha=0.15, linewidth=0.8)
        # mean trajectory
        mean_vals = vals.mean(axis=0)
        ax.plot(strengths, mean_vals, color='#F58518', linewidth=2.0, label='mean')
        ax.set_title(f'layer={layer}', fontsize=11)
        ax.set_ylabel('UDR')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel('strength')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('RMU instance-wise UDR trajectories (weak → mid → strong)', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_udr_vs_ftcount(data_map, title, out_path, row_labels=None, col_labels=None):
    keys = list(data_map.keys())
    fig, axes = grid_axes()
    for ax, key in zip(axes.flatten(), keys):
        ft, udr = data_map[key]
        ax.scatter(ft, udr, s=8, alpha=0.4)
        ax.set_xlabel('FT count')
        ax.set_ylabel('UDR')
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=[0.08,0.03,1,0.90])
    if row_labels and col_labels:
        add_grid_labels(fig, axes, row_labels, col_labels, col_y=0.92)
    fig.suptitle(title, y=0.985)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_udr_bar(mean_map, out_path):
    keys = list(mean_map.keys())
    vals = [mean_map[k] for k in keys]
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(range(len(keys)), vals, color='#4C78A8')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean UDR')
    ax.set_title('RMU mean UDR (clipped)')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_example_scatter(data_map, out_path, row_labels=None, col_labels=None):
    keys = list(data_map.keys())
    fig, axes = grid_axes()
    for ax, key in zip(axes.flatten(), keys):
        vals = data_map[key]
        xs = np.arange(len(vals))
        ax.scatter(xs, vals, s=4, alpha=0.5)
        ax.set_xlabel('Example index')
        ax.set_ylabel('UDR')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    fig.tight_layout(rect=[0.08,0.03,1,0.90])
    if row_labels and col_labels:
        add_grid_labels(fig, axes, row_labels, col_labels, col_y=0.92)
    fig.suptitle('RMU example-level UDR (clipped)', y=0.985)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_example_heatmap(matrix, keys, out_path):
    fig, ax = plt.subplots(figsize=(6,8))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('Model')
    ax.set_ylabel('Example (sorted)')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=7)
    fig.colorbar(im, ax=ax, label='UDR')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_small_heatmap(matrix, col_labels, title, out_path):
    fig, ax = plt.subplots(figsize=(4, 7))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('Strength')
    ax.set_ylabel('Example (sorted)')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='UDR', fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_instancewise_summary(mean_vals, std_vals, out_hist, out_line):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(mean_vals, bins=30, color='#4C78A8', alpha=0.85)
    ax.set_xlabel('Instance-wise mean UDR')
    ax.set_ylabel('Count')
    ax.set_title('RMU instance-wise mean UDR')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_hist, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4))
    order = np.argsort(mean_vals)
    ax.plot(mean_vals[order], label='mean UDR')
    ax.fill_between(range(len(mean_vals)),
                    mean_vals[order] - std_vals[order],
                    mean_vals[order] + std_vals[order],
                    color='#F58518', alpha=0.2, label='±1 std')
    ax.set_xlabel('Example (sorted)')
    ax.set_ylabel('UDR')
    ax.set_ylim(0,1)
    ax.set_title('Instance-wise UDR (sorted)')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_line, dpi=160)
    plt.close(fig)


def order_keys(keys):
    parsed = [(k, *parse_key(k)) for k in keys]
    parsed.sort(key=lambda x: (x[1], x[2]))
    return [k for k, *_ in parsed]


def stats_table(title, rows, out_path):
    lines = [f"# {title}", "| model | layer | n | mean | median | min | max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['layer']} | {row['n']} | {row['mean']:.3f} | {row['median']:.3f} | {row['min']:.3f} | {row['max']:.3f} |"
        )
    out_path.write_text("\n".join(lines))


def main():
    logs = sorted(RUNS_DIR.glob(LOG_GLOB))
    if not logs:
        raise SystemExit(f'No logs matched {LOG_GLOB}')

    data = {}
    tau = None
    for log in logs:
        raw = log.parent.name.replace('0201_', '').replace('tf_', '')
        m = re.search(r"(rmu_.*)", raw)
        key = m.group(1) if m else raw
        t, examples = parse_log(log)
        tau = t
        data[key] = examples

    ordered_keys = order_keys(list(data.keys()))
    data = {k: data[k] for k in ordered_keys}

    row_labels = ["1e-5", "2e-5", "5e-5"]
    col_labels = ["5", "10", "15"]

    summary_lines = ["| model | n | n_na | udr_mean | udr_median |", "|---|---:|---:|---:|---:|"]
    mean_map = {}
    for key, examples in data.items():
        s = summarize(examples)
        summary_lines.append(f"| {key} | {s['n']} | {s['n_na']} | {s['udr_mean']:.3f} | {s['udr_median']:.3f} |")
        mean_map[key] = s['udr_mean']
    (OUT_DIR / 'rmu_summary.md').write_text("\n".join(summary_lines))

    udr_full_map = {}
    for key, examples in data.items():
        vals = [ex['udr'] for ex in examples]
        udr_full_map[key] = [0.0 if math.isnan(v) else max(0.0, min(1.0, v)) for v in vals]

    udr_map = {}
    udr_raw_map = {}
    udr_ft_scatter = {}
    for key, examples in data.items():
        vals = [ex['udr'] for ex in examples if not math.isnan(ex['udr'])]
        udr_raw_map[key] = vals
        udr_map[key] = [min(1.0, max(0.0, v)) for v in vals]
        udr_ft_scatter[key] = (
            [ex['ft_count'] for ex in examples if not math.isnan(ex['udr'])],
            [ex['udr'] for ex in examples if not math.isnan(ex['udr'])],
        )

    plot_hist_grid(udr_map, mean_map, 'RMU UDR histogram (clipped)', OUT_DIR / 'rmu_udr_hist_3x3.png', bins=30, xlim=(0,1), row_labels=row_labels, col_labels=col_labels)
    plot_cdf_grid(udr_map, 'RMU UDR CDF (clipped)', OUT_DIR / 'rmu_udr_cdf_3x3.png', xlim=(0,1), row_labels=row_labels, col_labels=col_labels)

    udr_raw_capped = {k: [min(1.5, max(-0.5, v)) for v in vals] for k, vals in udr_raw_map.items()}
    plot_cdf_grid(udr_raw_capped, 'RMU UDR CDF (raw, capped)', OUT_DIR / 'rmu_udr_raw_cdf_3x3.png', xlim=(-0.5, 1.5), row_labels=row_labels, col_labels=col_labels)

    plot_udr_vs_ftcount(udr_ft_scatter, 'RMU UDR vs FT count', OUT_DIR / 'rmu_udr_vs_ftcount_3x3.png', row_labels=row_labels, col_labels=col_labels)
    plot_udr_bar(mean_map, OUT_DIR / 'rmu_udr_mean_bar.png')

    plot_example_scatter(udr_full_map, OUT_DIR / 'rmu_udr_example_scatter_3x3.png', row_labels=row_labels, col_labels=col_labels)
    matrix = np.stack([np.array(udr_full_map[k]) for k in ordered_keys], axis=1)
    order = np.argsort(matrix.mean(axis=1))
    plot_example_heatmap(matrix[order], ordered_keys, OUT_DIR / 'rmu_udr_example_heatmap.png')
    # split heatmaps by layer (weak/mid/strong columns)
    strengths = ['weak', 'mid', 'strong']
    for layer in [5, 10, 15]:
        key_map = {
            'weak': f'rmu_lr1e5_l{layer}_s10_ep5_layer',
            'mid': f'rmu_lr2e5_l{layer}_s10_ep5_layer',
            'strong': f'rmu_lr5e5_l{layer}_s10_ep5_layer',
        }
        mat = np.stack([np.array(udr_full_map[key_map[s]]) for s in strengths], axis=1)
        order = np.argsort(mat.mean(axis=1))
        plot_small_heatmap(mat[order], strengths,
                           f'RMU UDR heatmap (layer={layer})',
                           OUT_DIR / f'rmu_udr_heatmap_layer{layer}.png')

    # instance-wise summary
    mean_vals = matrix.mean(axis=1)
    std_vals = matrix.std(axis=1)
    plot_instancewise_summary(mean_vals, std_vals,
                              OUT_DIR / 'rmu_instancewise_mean_hist.png',
                              OUT_DIR / 'rmu_instancewise_mean_sorted.png')

    d1_lines = {}
    d2_lines = {}
    ft_rate_lines = {}
    diff_lines = {}
    traj_s1 = {}
    traj_s2 = {}
    for key, examples in data.items():
        layers, d1_by_layer, d2_by_layer, ft_count, total_count = layerwise_stats(examples, tau)
        d1_mean = [float(np.mean(d1_by_layer[l])) for l in layers]
        d2_mean = [float(np.mean(d2_by_layer[l])) for l in layers]
        diff_mean = [float(np.mean(np.array(d2_by_layer[l]) - np.array(d1_by_layer[l]))) for l in layers]
        ft_rate = [ft_count[l] / total_count[l] if total_count[l] else 0.0 for l in layers]
        d1_lines[key] = (layers, d1_mean)
        d2_lines[key] = (layers, d2_mean)
        diff_lines[key] = (layers, diff_mean)
        ft_rate_lines[key] = (layers, ft_rate)
        # instancewise trajectories (examples x layers)
        mat_s1 = np.array([[ex['d1'].get(l, 0.0) for l in layers] for ex in examples])
        mat_s2 = np.array([[ex['d2'].get(l, 0.0) for l in layers] for ex in examples])
        traj_s1[key] = mat_s1
        traj_s2[key] = mat_s2

    plot_layerwise(d1_lines, 'RMU layerwise mean ΔS1', OUT_DIR / 'rmu_layerwise_mean_delta_s1.png', ylabel='Mean ΔS1')
    plot_layerwise(d2_lines, 'RMU layerwise mean ΔS2', OUT_DIR / 'rmu_layerwise_mean_delta_s2.png', ylabel='Mean ΔS2')
    plot_ft_rate(ft_rate_lines, f'RMU layerwise FT rate (τ={tau})', OUT_DIR / 'rmu_layerwise_ft_rate.png')

    # per-config trajectories (3x3)
    plot_line_grid(d1_lines, 'RMU layerwise mean ΔS1 (per config)', OUT_DIR / 'rmu_layerwise_mean_delta_s1_3x3.png',
                   ylabel='Mean ΔS1', row_labels=row_labels, col_labels=col_labels)
    plot_line_grid(d2_lines, 'RMU layerwise mean ΔS2 (per config)', OUT_DIR / 'rmu_layerwise_mean_delta_s2_3x3.png',
                   ylabel='Mean ΔS2', row_labels=row_labels, col_labels=col_labels)
    plot_line_grid(diff_lines, 'RMU layerwise mean (ΔS2 − ΔS1) (per config)', OUT_DIR / 'rmu_layerwise_delta_diff_3x3.png',
                   ylabel='Mean Δ(S2−S1)', row_labels=row_labels, col_labels=col_labels)
    plot_line_grid(ft_rate_lines, f'RMU layerwise FT rate (per config, τ={tau})', OUT_DIR / 'rmu_layerwise_ft_rate_3x3.png',
                   ylabel='FT rate', row_labels=row_labels, col_labels=col_labels)

    # instancewise trajectories (per config)
    plot_instancewise_traj_grid(traj_s1, layers, 'RMU instance-wise ΔS1 trajectories (per config)',
                                OUT_DIR / 'rmu_instancewise_delta_s1_3x3.png',
                                ylabel='ΔS1', row_labels=row_labels, col_labels=col_labels, sample_size=60)
    plot_instancewise_traj_grid(traj_s2, layers, 'RMU instance-wise ΔS2 trajectories (per config)',
                                OUT_DIR / 'rmu_instancewise_delta_s2_3x3.png',
                                ylabel='ΔS2', row_labels=row_labels, col_labels=col_labels, sample_size=60)

    # instancewise UDR trajectories across weak/mid/strong (3x1 by layer)
    udr_by_layer = {
        5: {
            'weak': np.array(udr_full_map['rmu_lr1e5_l5_s10_ep5_layer']),
            'mid': np.array(udr_full_map['rmu_lr2e5_l5_s10_ep5_layer']),
            'strong': np.array(udr_full_map['rmu_lr5e5_l5_s10_ep5_layer']),
        },
        10: {
            'weak': np.array(udr_full_map['rmu_lr1e5_l10_s10_ep5_layer']),
            'mid': np.array(udr_full_map['rmu_lr2e5_l10_s10_ep5_layer']),
            'strong': np.array(udr_full_map['rmu_lr5e5_l10_s10_ep5_layer']),
        },
        15: {
            'weak': np.array(udr_full_map['rmu_lr1e5_l15_s10_ep5_layer']),
            'mid': np.array(udr_full_map['rmu_lr2e5_l15_s10_ep5_layer']),
            'strong': np.array(udr_full_map['rmu_lr5e5_l15_s10_ep5_layer']),
        },
    }
    plot_udr_strength_trajectories(udr_by_layer, OUT_DIR / 'rmu_udr_strength_trajectories_3x1.png', sample_size=80)

    counts_lines = [
        f"# RMU layerwise FT counts (τ={tau})",
        "| model | layer | ft_count | total | ft_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, examples in data.items():
        layers, _, _, ft_count, total_count = layerwise_stats(examples, tau)
        for l in layers:
            total = total_count[l]
            fc = ft_count[l]
            rate = fc / total if total else 0.0
            counts_lines.append(f"| {key} | {l} | {fc} | {total} | {rate:.3f} |")
    (OUT_DIR / 'rmu_layerwise_ft_counts.md').write_text("\n".join(counts_lines))

    rows_all = []
    rows_ft = []
    for key, examples in data.items():
        layers, d1_by_layer, _, _, _ = layerwise_stats(examples, tau)
        for l in layers:
            vals = np.array(d1_by_layer[l])
            rows_all.append({
                'model': key,
                'layer': l,
                'n': len(vals),
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
            })
            ft_vals = vals[vals > tau]
            if ft_vals.size:
                rows_ft.append({
                    'model': key,
                    'layer': l,
                    'n': int(ft_vals.size),
                    'mean': float(np.mean(ft_vals)),
                    'median': float(np.median(ft_vals)),
                    'min': float(np.min(ft_vals)),
                    'max': float(np.max(ft_vals)),
                })
            else:
                rows_ft.append({
                    'model': key,
                    'layer': l,
                    'n': 0,
                    'mean': 0.0,
                    'median': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                })

    stats_table('RMU layerwise ΔS1 stats (all)', rows_all, OUT_DIR / 'rmu_layerwise_delta_stats.md')
    stats_table(f'RMU layerwise ΔS1 stats (FT-only, τ={tau})', rows_ft, OUT_DIR / 'rmu_layerwise_delta_stats_ft_only.md')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import re
import math
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ALPHA2_DIR = Path('runs/0201alpha2')
ALPHA5_DIR = Path('runs/0201alpha5')
OUT = Path('docs/0201')
OUT.mkdir(parents=True, exist_ok=True)

# method -> (low, mid, high) model keys
# for RMU, use layer=10 as representative
METHODS = {
    'graddiff': ['graddiff_lr1e5_a2_ep5','graddiff_lr2e5_a2_ep5','graddiff_lr5e5_a2_ep5'],
    'idknll':  ['idknll_lr1e5_a2_ep5','idknll_lr2e5_a2_ep5','idknll_lr5e5_a2_ep5'],
    'idkdpo':  ['idkdpo_lr1e5_b01_a2_ep5','idkdpo_lr2e5_b01_a2_ep5','idkdpo_lr5e5_b01_a2_ep5'],
    'npo':     ['npo_lr1e5_b01_a2_ep5','npo_lr2e5_b01_a2_ep5','npo_lr5e5_b01_a2_ep5'],
    'altpo':   ['altpo_lr1e5_b01_a2_ep5','altpo_lr2e5_b01_a2_ep5','altpo_lr5e5_b01_a2_ep5'],
    'undial':  ['undial_lr1e5_b10_a2_ep5','undial_lr1e4_b10_a2_ep5','undial_lr3e4_b10_a2_ep5'],
    'simnpo':  ['simnpo_lr1e5_b35_a1_d1_g0125_ep5','simnpo_lr2e5_b35_a1_d1_g0125_ep5','simnpo_lr5e5_b35_a1_d1_g0125_ep5'],
    'rmu':     ['rmu_lr1e5_l10_s10_ep5','rmu_lr2e5_l10_s10_ep5','rmu_lr5e5_l10_s10_ep5'],
}

# expand to 10 methods by adding placeholders if needed
method_order = list(METHODS.keys())
if len(method_order) < 10:
    for i in range(10-len(method_order)):
        method_order.append(f'placeholder{i+1}')


def find_run(run_root: Path, key: str):
    for d in run_root.iterdir():
        if not d.is_dir():
            continue
        if key in d.name:
            return d
    return None


def read_uds(run_dir: Path):
    summ = run_dir/'summary.json'
    if summ.exists():
        try:
            data = json.loads(summ.read_text())
            for k in ['average_uds','avg_uds','uds']:
                if k in data:
                    return float(data[k])
        except Exception:
            pass
    # fallback: parse log
    log = run_dir/'run.log'
    if log.exists():
        txt = log.read_text(errors='ignore')
        m = re.search(r"Average UDS\s*[:=]\s*([0-9]*\.?[0-9]+)", txt)
        if m:
            return float(m.group(1))
    return math.nan


def build_matrix(run_root: Path, alpha_label: str):
    mat = []
    for method in method_order:
        if method.startswith('placeholder'):
            mat.append([math.nan, math.nan, math.nan])
            continue
        keys = METHODS[method]
        vals = []
        for k in keys:
            run = find_run(run_root, k)
            if run is None:
                vals.append(math.nan)
            else:
                vals.append(read_uds(run))
        mat.append(vals)
    # convert to 3xN (rows=lr levels)
    mat = np.array(mat).T
    return mat


def plot(mat, title, out_path, method_labels):
    fig, ax = plt.subplots(figsize=(14,4))
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['low lr','mid lr','high lr'])
    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=30, ha='right')
    ax.set_title(title)
    # annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i,j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=7, color='white')
    fig.colorbar(im, ax=ax, label='Mean UDS')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    mat2 = build_matrix(ALPHA2_DIR, 'a2')
    mat5 = build_matrix(ALPHA5_DIR, 'a5')
    plot(mat2, 'Alpha=2 UDS (rows=lr low→high, cols=methods)', OUT/'uds_alpha2_matrix.png', method_order)
    plot(mat5, 'Alpha=5 UDS (rows=lr low→high, cols=methods)', OUT/'uds_alpha5_matrix.png', method_order)

    # combined side-by-side
    fig, axes = plt.subplots(2,1, figsize=(14,7), sharex=True)
    for ax, mat, title in zip(axes, [mat2, mat5], ['alpha=2', 'alpha=5']):
        im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_yticks([0,1,2])
        ax.set_yticklabels(['low','mid','high'])
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i,j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=7, color='white')
    axes[-1].set_xticks(range(len(method_order)))
    axes[-1].set_xticklabels(method_order, rotation=30, ha='right')
    fig.colorbar(im, ax=axes, label='Mean UDS', fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT/'uds_alpha2_alpha5_matrix.png', dpi=160)
    plt.close(fig)

if __name__ == '__main__':
    main()

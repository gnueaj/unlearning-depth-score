#!/usr/bin/env python3
import json
import math
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

ROOTS = {
    'a1': Path('runs/0201alpha1'),
    'a2': Path('runs/0201alpha2'),
    'a5': Path('runs/0201alpha5'),
}

METHODS = ['graddiff','idknll','idkdpo','npo','altpo','undial','simnpo','rmu']
LR_LEVELS = ['low','mid','high']

# per method key template by lr level (without alpha suffix for simnpo/rmu)
KEYS = {
    'graddiff': {
        'low': 'graddiff_lr1e5_a{a}_ep5',
        'mid': 'graddiff_lr2e5_a{a}_ep5',
        'high': 'graddiff_lr5e5_a{a}_ep5',
    },
    'idknll': {
        'low': 'idknll_lr1e5_a{a}_ep5',
        'mid': 'idknll_lr2e5_a{a}_ep5',
        'high': 'idknll_lr5e5_a{a}_ep5',
    },
    'idkdpo': {
        'low': 'idkdpo_lr1e5_b01_a{a}_ep5',
        'mid': 'idkdpo_lr2e5_b01_a{a}_ep5',
        'high': 'idkdpo_lr5e5_b01_a{a}_ep5',
    },
    'npo': {
        'low': 'npo_lr1e5_b01_a{a}_ep5',
        'mid': 'npo_lr2e5_b01_a{a}_ep5',
        'high': 'npo_lr5e5_b01_a{a}_ep5',
    },
    'altpo': {
        'low': 'altpo_lr1e5_b01_a{a}_ep5',
        'mid': 'altpo_lr2e5_b01_a{a}_ep5',
        'high': 'altpo_lr5e5_b01_a{a}_ep5',
    },
    'undial': {
        'low': 'undial_lr1e5_b10_a{a}_ep5',
        'mid': 'undial_lr1e4_b10_a{a}_ep5',
        'high': 'undial_lr3e4_b10_a{a}_ep5',
    },
    # simnpo/rmu have no alpha variants (same key across alpha folders)
    'simnpo': {
        'low': 'simnpo_lr1e5_b35_a1_d1_g0125_ep5',
        'mid': 'simnpo_lr2e5_b35_a1_d1_g0125_ep5',
        'high': 'simnpo_lr5e5_b35_a1_d1_g0125_ep5',
    },
    'rmu': {
        'low': 'rmu_lr1e5_l10_s10_ep5',
        'mid': 'rmu_lr2e5_l10_s10_ep5',
        'high': 'rmu_lr5e5_l10_s10_ep5',
    },
}


def read_udr(run_dir: Path):
    summ = run_dir/'summary.json'
    if summ.exists():
        try:
            data = json.loads(summ.read_text())
            for k in ['average_udr','avg_udr','udr']:
                if k in data:
                    return float(data[k])
        except Exception:
            pass
    log = run_dir/'run.log'
    if log.exists():
        txt = log.read_text(errors='ignore')
        m = re.search(r"Average UDR\s*[:=]\s*([0-9]*\.?[0-9]+)", txt)
        if m:
            return float(m.group(1))
    return math.nan


def find_run(root: Path, key: str):
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if key in d.name:
            return d
    return None


def udr_for_key(root: Path, key: str):
    run = find_run(root, key)
    if run is None:
        return math.nan
    return read_udr(run)


def main():
    # data[lr][alpha][method] = udr
    data = {lr: {alpha: [] for alpha in ROOTS} for lr in LR_LEVELS}
    for method in METHODS:
        for lr in LR_LEVELS:
            for alpha, root in ROOTS.items():
                if method in ['simnpo','rmu']:
                    key = KEYS[method][lr]
                else:
                    key = KEYS[method][lr].format(a=alpha[1:])  # a1/a2/a5
                data[lr][alpha].append(udr_for_key(root, key))

    x = np.arange(len(METHODS))
    width = 0.22
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, lr in zip(axes, LR_LEVELS):
        ax.bar(x - width, data[lr]['a1'], width, label='alpha=1')
        ax.bar(x, data[lr]['a2'], width, label='alpha=2')
        ax.bar(x + width, data[lr]['a5'], width, label='alpha=5')
        ax.set_ylabel('UDR')
        ax.set_title(f'lr={lr}')
        ax.grid(axis='y', alpha=0.2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(METHODS, rotation=20, ha='right')
    axes[0].legend(loc='upper right')
    fig.tight_layout()
    Path('docs/0201').mkdir(parents=True, exist_ok=True)
    fig.savefig('docs/0201/udr_alpha123_bar_by_lr.png', dpi=160)
    plt.close(fig)

if __name__ == '__main__':
    main()

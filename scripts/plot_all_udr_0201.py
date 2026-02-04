#!/usr/bin/env python3
import re
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path('runs')
OUT_DIR = Path('docs/0201')
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_GLOB = '0201_*_tf_*_layer/run.log'

UDS_RE = re.compile(r"UDS\s*=\s*([+-]?[0-9]*\.?[0-9]+)")
NA_RE = re.compile(r"UDS\s*=\s*N/A")


def parse_uds(log_path: Path):
    vals = []
    na = 0
    with log_path.open() as f:
        for line in f:
            if NA_RE.search(line):
                na += 1
                continue
            m = UDS_RE.search(line)
            if m:
                vals.append(float(m.group(1)))
    return vals, na


def main():
    logs = sorted(Path('runs').glob(RUN_GLOB))
    rows = []
    for log in logs:
        run_dir = log.parent.name
        # extract model key after tf_ and before _layer
        m = re.search(r"_tf_(.*)_layer", run_dir)
        if not m:
            continue
        model_key = m.group(1)
        method = model_key.split('_')[0]
        vals, na = parse_uds(log)
        if vals:
            mean = float(np.mean(vals))
            median = float(np.median(vals))
        else:
            mean = math.nan
            median = math.nan
        rows.append({
            'model': model_key,
            'method': method,
            'n': len(vals) + na,
            'n_na': na,
            'mean': mean,
            'median': median,
        })

    # write summary table
    rows_sorted = sorted(rows, key=lambda r: (r['method'], r['model']))
    lines = ["| model | method | n | n_na | mean_uds | median_uds |",
             "|---|---|---:|---:|---:|---:|"]
    for r in rows_sorted:
        lines.append(f"| {r['model']} | {r['method']} | {r['n']} | {r['n_na']} | {r['mean']:.3f} | {r['median']:.3f} |")
    (OUT_DIR / 'uds_all_methods_summary.md').write_text("\n".join(lines))

    # bar chart by config
    labels = [r['model'] for r in rows_sorted]
    means = [r['mean'] for r in rows_sorted]
    fig, ax = plt.subplots(figsize=(12,4))
    ax.bar(range(len(labels)), means, color='#4C78A8')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('Mean UDS')
    ax.set_title('All methods: mean UDS per config (0201 runs)')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'uds_all_configs_bar.png', dpi=160)
    plt.close(fig)

    # bar chart by method (mean of config means)
    method_map = {}
    for r in rows:
        method_map.setdefault(r['method'], []).append(r['mean'])
    methods = sorted(method_map.keys())
    method_means = [float(np.mean(method_map[m])) for m in methods]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(range(len(methods)), method_means, color='#F58518')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_ylabel('Mean UDS')
    ax.set_title('All methods: mean UDS by method (0201 runs)')
    for i, v in enumerate(method_means):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'uds_all_methods_bar.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

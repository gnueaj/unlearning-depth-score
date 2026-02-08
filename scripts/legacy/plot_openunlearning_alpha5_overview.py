import json
import math
import re
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", default="alpha5_fix", help="alpha folder tag (e.g., alpha5_fix, alpha2_fix)")
parser.add_argument("--out_dir", default="docs/0202", help="output directory for table/plots")
parser.add_argument("--uds_stats", default="docs/0202/alpha5/uds_summary_stats.txt", help="UDS summary stats file (optional)")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def pick_dir(base: str, preferred: str, fallback: str) -> Path:
    p = Path(base) / preferred
    return p if p.exists() else Path(base) / fallback

mem_dir = Path('runs/memorization_eval') / args.alpha
priv_dir = Path('runs/privacy_eval') / args.alpha
util_dir = Path('runs/utility_eval') / args.alpha
run_root = None
uds_stats_path = Path(args.uds_stats)

method_order = [
    'graddiff','idknll','idkdpo','npo','altpo','undial','simnpo','rmu'
]

lr_re = re.compile(r'lr(\d+)e(\d+)')

def lr_value(name):
    m = lr_re.search(name)
    if not m:
        return float('inf')
    base = int(m.group(1))
    exp = int(m.group(2))
    return base * (10 ** -exp)

def layer_value(name):
    m = re.search(r'_l(\d+)', name)
    return int(m.group(1)) if m else 0

def sort_key(name):
    if name in ('full','retain'):
        return (-1 if name=='full' else 0, 0, 0, name)
    method = name.split('_lr')[0]
    try:
        mi = method_order.index(method)
    except ValueError:
        mi = 999
    return (1, mi, lr_value(name), layer_value(name), name)


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None

# Collect models from mem dir
models = sorted([d.name for d in mem_dir.iterdir() if d.is_dir()])
models = sorted(models, key=sort_key)

# Load full utility for normalization
full_util = None
full_summary = read_json(util_dir/'full'/'summary.json')
if full_summary:
    full_util = full_summary.get('utility') or full_summary.get('model_utility')


def harmonic_mean(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None
    vals = np.array(vals, dtype=float)
    return float(len(vals) / np.sum(1.0 / (vals + 1e-12)))

# Map UDS from summary stats table if available (more reliable)
uds_map = {}
if uds_stats_path.exists():
    for line in uds_stats_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] not in {"Method", "----", ""}:
            name = parts[0]
            try:
                mean = float(parts[2])
            except Exception:
                continue
            uds_map[name] = mean
else:
    # No UDS stats provided; leave UDS as NA except for full/retain below.
    pass
# Reference points
uds_map['full'] = 0.0
uds_map['retain'] = 1.0

rows = []
for m in models:
    mem = read_json(mem_dir/m/'summary.json')
    priv = read_json(priv_dir/m/'summary.json')
    util = read_json(util_dir/m/'summary.json')

    mem_val = mem.get('avg_mem') if mem else None
    priv_val = None
    if priv:
        priv_val = priv.get('privacy_score')
        if priv_val is None:
            priv_val = priv.get('auc')
    util_val = None
    if util:
        util_val = util.get('utility')
        if util_val is None:
            util_val = util.get('model_utility')

    util_rel = None
    if util_val is not None and full_util is not None:
        util_rel = util_val / (full_util + 1e-12)

    uds = uds_map.get(m)

    agg = harmonic_mean([mem_val, priv_val, util_rel])
    agg_uds = harmonic_mean([mem_val, priv_val, util_rel, uds])

    rows.append({
        'model': m,
        'mem': mem_val,
        'privacy': priv_val,
        'utility_rel': util_rel,
        'uds': uds,
        'agg': agg,
        'agg_uds': agg_uds,
    })

# Write markdown table
md_lines = []
md_lines.append('| Model | Agg. (↑)<br>no UDS | Agg. (↑)<br>with UDS | Mem | Privacy<br>(sMIA HM) | Utility<br>(rel. to Full,<br>HM(MU,Fluency)) | UDS |')
md_lines.append('|---|---:|---:|---:|---:|---:|---:|')
for r in rows:
    def fmt(v):
        return 'NA' if v is None or (isinstance(v,float) and math.isnan(v)) else f"{v:.3f}"
    md_lines.append(
        f"| {r['model']} | {fmt(r['agg'])} | {fmt(r['agg_uds'])} | {fmt(r['mem'])} | {fmt(r['privacy'])} | {fmt(r['utility_rel'])} | {fmt(r['uds'])} |"
    )

md_text = '\n'.join(md_lines)
(OUT_DIR/'openunlearning_alpha5_table.md').write_text(md_text)

# Heatmap (normalized 0-1 metrics)
metrics = ['agg','agg_uds','mem','privacy','utility_rel','uds']
mat = []
for metric in metrics:
    mat.append([r[metric] if r[metric] is not None else np.nan for r in rows])
mat = np.array(mat, dtype=float)

# Column labels (short)
col_labels = []
for m in models:
    if m in ('full','retain'):
        col_labels.append(m)
    elif m.startswith('rmu'):
        lr = re.search(r'(lr\d+e\d+)', m).group(1)
        layer = re.search(r'_l(\d+)', m).group(1)
        col_labels.append(f"rmu\n{lr} l{layer}")
    else:
        method = m.split('_lr')[0]
        lr = re.search(r'(lr\d+e\d+)', m).group(1)
        col_labels.append(f"{method}\n{lr}")

fig, ax = plt.subplots(figsize=(22, 4.8))
# Normalize color 0-1 for all
im = ax.imshow(mat, aspect='auto', vmin=0, vmax=1, cmap='viridis')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(metrics)))
ax.set_yticklabels(['Agg(HM)','Agg+UDS','Mem','Privacy','Utility(rel)','UDS'], fontsize=10)
ax.set_title('Open-Unlearning metrics (alpha=5, 30 models + full/retain)', fontsize=12)

# annotate cells lightly
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if math.isnan(v):
            text = 'NA'
        else:
            text = f"{v:.2f}"
        ax.text(j, i, text, ha='center', va='center', fontsize=6, color='white' if (not math.isnan(v) and v>0.6) else 'black')

fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
fig.tight_layout()
fig.savefig(OUT_DIR/'openunlearning_alpha5_heatmap.png', dpi=200, bbox_inches='tight')

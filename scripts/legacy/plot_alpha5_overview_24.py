import re
import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path('docs/0202')
OUT.mkdir(parents=True, exist_ok=True)

mem_dir = Path('runs/memorization_eval/alpha5')
priv_dir = Path('runs/privacy_eval/alpha5')
util_dir = Path('runs/utility_eval/alpha5')
run_root = Path('runs/0201alpha5')

models = sorted([d.name for d in mem_dir.iterdir() if d.is_dir()])

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
    method = name.split('_lr')[0]
    try:
        mi = method_order.index(method)
    except ValueError:
        mi = 999
    return (mi, lr_value(name), layer_value(name), name)

models = sorted(models, key=sort_key)


def load_mem(m):
    p = mem_dir / m / 'summary.json'
    if not p.exists():
        return None
    data = json.load(open(p))
    return data.get('avg_mem')


def load_priv(m):
    p = priv_dir / m / 'summary.json'
    if not p.exists():
        return None
    data = json.load(open(p))
    return data.get('privacy_score') or data.get('auc')


def load_util(m):
    p = util_dir / m / 'summary.json'
    if not p.exists():
        return None
    data = json.load(open(p))
    return data.get('utility') or data.get('model_utility')


def find_uds(m):
    # look for run.log under runs/0201alpha5 matching model
    candidates = list(run_root.glob(f"*{m}*_layer/run.log"))
    if not candidates:
        # fallback: search globally (alpha5 only)
        candidates = list(Path('runs').glob(f"**/*{m}*_layer/run.log"))
    if not candidates:
        return None
    # choose most recent
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    path = candidates[0]
    text = path.read_text(errors='ignore')
    # try to find Average UDS line first
    m1 = re.findall(r"Average UDS\s*[:=]\s*([0-9]*\.?[0-9]+)", text)
    if m1:
        return float(m1[-1])
    m2 = re.findall(r"UDS\s*[:=]\s*([0-9]*\.?[0-9]+)", text)
    if m2:
        return float(m2[-1])
    return None


uds = [find_uds(m) for m in models]
mem = [load_mem(m) for m in models]
priv = [load_priv(m) for m in models]
util = [load_util(m) for m in models]

rows = [uds, mem, priv, util]
row_labels = ['UDS', 'Mem', 'Privacy (sMIA HM)', 'Utility (HM)']

mat = np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)

# Build labels for columns
col_labels = []
for m in models:
    if m.startswith('rmu'):
        # rmu_lr1e5_l5_s10_ep5 -> rmu\nlr1e5 l5
        lr = re.search(r'(lr\d+e\d+)', m).group(1)
        layer = re.search(r'_l(\d+)', m).group(1)
        col_labels.append(f"rmu\n{lr} l{layer}")
    elif m.startswith('simnpo'):
        lr = re.search(r'(lr\d+e\d+)', m).group(1)
        col_labels.append(f"simnpo\n{lr}")
    else:
        method = m.split('_lr')[0]
        lr = re.search(r'(lr\d+e\d+)', m).group(1)
        col_labels.append(f"{method}\n{lr}")

fig, ax = plt.subplots(figsize=(18, 4.5))
# heatmap
im = ax.imshow(mat, aspect='auto', vmin=0, vmax=1, cmap='viridis')

ax.set_xticks(range(len(models)))
ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=10)

# annotate cells
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if math.isnan(v):
            text = 'NA'
        else:
            text = f"{v:.3f}"
        ax.text(j, i, text, ha='center', va='center', fontsize=6, color='white' if (not math.isnan(v) and v>0.6) else 'black')

cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
# Title
ax.set_title('Alpha=5 Open-Unlearning Overview (24 models)\nUDS + Memorization + Privacy + Utility', fontsize=12)

fig.tight_layout()
out_path = OUT / 'alpha5_openunlearning_overview_24.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved {out_path}")

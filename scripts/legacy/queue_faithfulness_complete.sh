#!/bin/bash
# Queue script: Wait for UDS faithfulness to complete, then process all 13 metrics

PROJECT_DIR="/home/jaeung/activation-patching-unlearning"
LOG_DIR="$PROJECT_DIR/logs/faithfulness"
RUNS_DIR="$PROJECT_DIR/runs/meta_eval"

echo "=== Faithfulness Queue Script ==="
echo "Waiting for UDS faithfulness to complete..."

# Wait for both UDS processes to complete
while true; do
    GPU0_RUNNING=$(ps aux | grep "faithfulness_uds_sdpa_gpu0" | grep -v grep | wc -l)
    GPU1_RUNNING=$(ps aux | grep "faithfulness_uds_sdpa_gpu1" | grep -v grep | wc -l)

    if [ "$GPU0_RUNNING" -eq 0 ] && [ "$GPU1_RUNNING" -eq 0 ]; then
        echo "UDS faithfulness complete!"
        break
    fi

    # Check progress
    GPU0_DONE=$(cat $RUNS_DIR/table2_faithfulness_uds_sdpa_gpu0/results.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d))" 2>/dev/null || echo "0")
    GPU1_DONE=$(cat $RUNS_DIR/table2_faithfulness_uds_sdpa_gpu1/results.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d))" 2>/dev/null || echo "0")

    TOTAL=$((GPU0_DONE + GPU1_DONE))
    echo "$(date '+%H:%M:%S') - UDS progress: $TOTAL/60 (GPU0: $GPU0_DONE, GPU1: $GPU1_DONE)"
    sleep 60
done

echo ""
echo "=== Processing Results ==="

# Step 1: Merge UDS results from both GPUs
echo "[1/5] Merging UDS results..."
python3 - << 'EOF'
import json
from pathlib import Path

runs_dir = Path("/home/jaeung/activation-patching-unlearning/runs/meta_eval")
log_dir = Path("/home/jaeung/activation-patching-unlearning/logs/faithfulness")

# Load GPU0 and GPU1 UDS results
uds_results = {}
for gpu_dir in ["table2_faithfulness_uds_sdpa_gpu0", "table2_faithfulness_uds_sdpa_gpu1"]:
    results_file = runs_dir / gpu_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            uds_results.update(data)

print(f"  Loaded UDS results for {len(uds_results)} models")

# Load 12 metrics SDPA results
with open(log_dir / "results_12metrics_sdpa.json") as f:
    sdpa_12 = json.load(f)

print(f"  Loaded 12 metrics for {len(sdpa_12)} models")

# Merge: add UDS to each model
merged = {}
for model, data in sdpa_12.items():
    merged[model] = data.copy()
    if model in uds_results:
        merged[model]['metrics']['uds'] = uds_results[model].get('metrics', {}).get('uds',
                                           uds_results[model].get('uds'))

# Save merged results
with open(log_dir / "results_13metrics_sdpa.json", 'w') as f:
    json.dump(merged, f, indent=2)

print(f"  Saved merged 13 metrics for {len(merged)} models")

# Also save UDS-only results
with open(log_dir / "results_uds_sdpa.json", 'w') as f:
    json.dump(uds_results, f, indent=2)

print(f"  Saved UDS-only results")
EOF

# Step 2: Compute AUC-ROC for all 13 metrics
echo "[2/5] Computing AUC-ROC..."
python3 - << 'EOF'
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

log_dir = Path("/home/jaeung/activation-patching-unlearning/logs/faithfulness")

# P/N pool definition (from meta_eval_faithfulness.py)
P_POOL = [
    "graddiff_lr1e5_a1_ep5", "graddiff_lr2e5_a1_ep5", "graddiff_lr5e5_a1_ep5",
    "idknll_lr1e5_a1_ep5", "idknll_lr2e5_a1_ep5", "idknll_lr5e5_a1_ep5",
    "idkdpo_lr1e5_b01_a1_ep5", "idkdpo_lr2e5_b01_a1_ep5", "idkdpo_lr5e5_b01_a1_ep5",
    "npo_lr1e5_b01_a1_ep5", "npo_lr2e5_b01_a1_ep5", "npo_lr5e5_b01_a1_ep5",
    "simnpo_lr1e5_b35_a1_d1_g0125_ep5", "simnpo_lr2e5_b35_a1_d1_g0125_ep5", "simnpo_lr5e5_b35_a1_d1_g0125_ep5",
    "graddiff_lr1e5_a1_ep10", "graddiff_lr2e5_a1_ep10", "graddiff_lr5e5_a1_ep10",
    "idknll_lr1e5_a1_ep10", "idknll_lr2e5_a1_ep10", "idknll_lr5e5_a1_ep10",
    "idkdpo_lr1e5_b01_a1_ep10", "idkdpo_lr2e5_b01_a1_ep10", "idkdpo_lr5e5_b01_a1_ep10",
    "npo_lr1e5_b01_a1_ep10", "npo_lr2e5_b01_a1_ep10", "npo_lr5e5_b01_a1_ep10",
    "simnpo_lr1e5_b35_a1_d1_g0125_ep10", "simnpo_lr2e5_b35_a1_d1_g0125_ep10", "simnpo_lr5e5_b35_a1_d1_g0125_ep10",
]

N_POOL = [
    "graddiff_lr1e5_a5_ep5", "graddiff_lr2e5_a5_ep5", "graddiff_lr5e5_a5_ep5",
    "idknll_lr1e5_a5_ep5", "idknll_lr2e5_a5_ep5", "idknll_lr5e5_a5_ep5",
    "idkdpo_lr1e5_b01_a5_ep5", "idkdpo_lr2e5_b01_a5_ep5", "idkdpo_lr5e5_b01_a5_ep5",
    "npo_lr1e5_b01_a5_ep5", "npo_lr2e5_b01_a5_ep5", "npo_lr5e5_b01_a5_ep5",
    "simnpo_lr1e5_b35_a5_d1_g0125_ep5", "simnpo_lr2e5_b35_a5_d1_g0125_ep5", "simnpo_lr5e5_b35_a5_d1_g0125_ep5",
    "graddiff_lr1e5_a5_ep10", "graddiff_lr2e5_a5_ep10", "graddiff_lr5e5_a5_ep10",
    "idknll_lr1e5_a5_ep10", "idknll_lr2e5_a5_ep10", "idknll_lr5e5_a5_ep10",
    "idkdpo_lr1e5_b01_a5_ep10", "idkdpo_lr2e5_b01_a5_ep10", "idkdpo_lr5e5_b01_a5_ep10",
    "npo_lr1e5_b01_a5_ep10", "npo_lr2e5_b01_a5_ep10", "npo_lr5e5_b01_a5_ep10",
    "simnpo_lr1e5_b35_a5_d1_g0125_ep10", "simnpo_lr2e5_b35_a5_d1_g0125_ep10", "simnpo_lr5e5_b35_a5_d1_g0125_ep10",
]

# Load merged results
with open(log_dir / "results_13metrics_sdpa.json") as f:
    results = json.load(f)

# All 13 metrics
METRICS = [
    'em', 'es', 'prob', 'paraprob', 'truth_ratio',
    'rouge', 'para_rouge', 'jailbreak_rouge',
    'mia_loss', 'mia_zlib', 'mia_min_k', 'mia_min_kpp',
    'uds'
]

# Compute AUC-ROC for each metric
auc_results = {}
for metric in METRICS:
    p_scores = []
    n_scores = []

    for model in P_POOL:
        if model in results and 'metrics' in results[model]:
            val = results[model]['metrics'].get(metric)
            if val is not None:
                p_scores.append(val)

    for model in N_POOL:
        if model in results and 'metrics' in results[model]:
            val = results[model]['metrics'].get(metric)
            if val is not None:
                n_scores.append(val)

    if len(p_scores) > 0 and len(n_scores) > 0:
        # Labels: P=1, N=0 (higher score = more unlearning = positive)
        y_true = [1] * len(p_scores) + [0] * len(n_scores)
        y_score = p_scores + n_scores

        auc = roc_auc_score(y_true, y_score)
        auc_results[metric] = {
            'auc_roc': auc,
            'p_count': len(p_scores),
            'n_count': len(n_scores),
            'p_mean': np.mean(p_scores),
            'n_mean': np.mean(n_scores)
        }
        print(f"  {metric}: AUC={auc:.4f} (P:{len(p_scores)}, N:{len(n_scores)})")
    else:
        print(f"  {metric}: SKIP (P:{len(p_scores)}, N:{len(n_scores)})")

# Save AUC results
with open(log_dir / "auc_roc_13metrics.json", 'w') as f:
    json.dump(auc_results, f, indent=2)

print(f"\n  Saved AUC-ROC results for {len(auc_results)} metrics")
EOF

# Step 3: Generate histograms
echo "[3/5] Generating histograms..."
python3 - << 'EOF'
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

log_dir = Path("/home/jaeung/activation-patching-unlearning/logs/faithfulness")
hist_dir = log_dir / "histograms"
hist_dir.mkdir(exist_ok=True)

# Load results and AUC
with open(log_dir / "results_13metrics_sdpa.json") as f:
    results = json.load(f)
with open(log_dir / "auc_roc_13metrics.json") as f:
    auc_data = json.load(f)

# P/N pools
P_POOL = [
    "graddiff_lr1e5_a1_ep5", "graddiff_lr2e5_a1_ep5", "graddiff_lr5e5_a1_ep5",
    "idknll_lr1e5_a1_ep5", "idknll_lr2e5_a1_ep5", "idknll_lr5e5_a1_ep5",
    "idkdpo_lr1e5_b01_a1_ep5", "idkdpo_lr2e5_b01_a1_ep5", "idkdpo_lr5e5_b01_a1_ep5",
    "npo_lr1e5_b01_a1_ep5", "npo_lr2e5_b01_a1_ep5", "npo_lr5e5_b01_a1_ep5",
    "simnpo_lr1e5_b35_a1_d1_g0125_ep5", "simnpo_lr2e5_b35_a1_d1_g0125_ep5", "simnpo_lr5e5_b35_a1_d1_g0125_ep5",
    "graddiff_lr1e5_a1_ep10", "graddiff_lr2e5_a1_ep10", "graddiff_lr5e5_a1_ep10",
    "idknll_lr1e5_a1_ep10", "idknll_lr2e5_a1_ep10", "idknll_lr5e5_a1_ep10",
    "idkdpo_lr1e5_b01_a1_ep10", "idkdpo_lr2e5_b01_a1_ep10", "idkdpo_lr5e5_b01_a1_ep10",
    "npo_lr1e5_b01_a1_ep10", "npo_lr2e5_b01_a1_ep10", "npo_lr5e5_b01_a1_ep10",
    "simnpo_lr1e5_b35_a1_d1_g0125_ep10", "simnpo_lr2e5_b35_a1_d1_g0125_ep10", "simnpo_lr5e5_b35_a1_d1_g0125_ep10",
]

N_POOL = [
    "graddiff_lr1e5_a5_ep5", "graddiff_lr2e5_a5_ep5", "graddiff_lr5e5_a5_ep5",
    "idknll_lr1e5_a5_ep5", "idknll_lr2e5_a5_ep5", "idknll_lr5e5_a5_ep5",
    "idkdpo_lr1e5_b01_a5_ep5", "idkdpo_lr2e5_b01_a5_ep5", "idkdpo_lr5e5_b01_a5_ep5",
    "npo_lr1e5_b01_a5_ep5", "npo_lr2e5_b01_a5_ep5", "npo_lr5e5_b01_a5_ep5",
    "simnpo_lr1e5_b35_a5_d1_g0125_ep5", "simnpo_lr2e5_b35_a5_d1_g0125_ep5", "simnpo_lr5e5_b35_a5_d1_g0125_ep5",
    "graddiff_lr1e5_a5_ep10", "graddiff_lr2e5_a5_ep10", "graddiff_lr5e5_a5_ep10",
    "idknll_lr1e5_a5_ep10", "idknll_lr2e5_a5_ep10", "idknll_lr5e5_a5_ep10",
    "idkdpo_lr1e5_b01_a5_ep10", "idkdpo_lr2e5_b01_a5_ep10", "idkdpo_lr5e5_b01_a5_ep10",
    "npo_lr1e5_b01_a5_ep10", "npo_lr2e5_b01_a5_ep10", "npo_lr5e5_b01_a5_ep10",
    "simnpo_lr1e5_b35_a5_d1_g0125_ep10", "simnpo_lr2e5_b35_a5_d1_g0125_ep10", "simnpo_lr5e5_b35_a5_d1_g0125_ep10",
]

METRICS = [
    'em', 'es', 'prob', 'paraprob', 'truth_ratio',
    'rouge', 'para_rouge', 'jailbreak_rouge',
    'mia_loss', 'mia_zlib', 'mia_min_k', 'mia_min_kpp',
    'uds'
]

METRIC_NAMES = {
    'em': 'Exact Match', 'es': 'Extraction Strength',
    'prob': 'Probability', 'paraprob': 'Para.Prob',
    'truth_ratio': 'Truth Ratio',
    'rouge': 'ROUGE', 'para_rouge': 'Para.ROUGE', 'jailbreak_rouge': 'Jailbreak ROUGE',
    'mia_loss': 'MIA-LOSS', 'mia_zlib': 'MIA-ZLib',
    'mia_min_k': 'MIA-MinK', 'mia_min_kpp': 'MIA-MinK++',
    'uds': 'UDS (Ours)'
}

# Generate individual histograms
for metric in METRICS:
    p_scores = []
    n_scores = []

    for model in P_POOL:
        if model in results and 'metrics' in results[model]:
            val = results[model]['metrics'].get(metric)
            if val is not None:
                p_scores.append(val)

    for model in N_POOL:
        if model in results and 'metrics' in results[model]:
            val = results[model]['metrics'].get(metric)
            if val is not None:
                n_scores.append(val)

    if len(p_scores) == 0 or len(n_scores) == 0:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(min(min(p_scores), min(n_scores)),
                       max(max(p_scores), max(n_scores)), 20)

    ax.hist(p_scores, bins=bins, alpha=0.6, label=f'P (α=1, n={len(p_scores)})', color='green')
    ax.hist(n_scores, bins=bins, alpha=0.6, label=f'N (α=5, n={len(n_scores)})', color='red')

    auc = auc_data.get(metric, {}).get('auc_roc', 0)
    ax.set_title(f'{METRIC_NAMES.get(metric, metric)} - AUC-ROC: {auc:.4f}')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.legend()

    plt.tight_layout()
    plt.savefig(hist_dir / f'{metric}_histogram.png', dpi=150)
    plt.close()
    print(f"  Generated {metric}_histogram.png")

# Generate combined 4x4 grid (13 metrics + 3 blank)
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, metric in enumerate(METRICS):
    ax = axes[i]

    p_scores = []
    n_scores = []

    for model in P_POOL:
        if model in results and 'metrics' in results[model]:
            val = results[model]['metrics'].get(metric)
            if val is not None:
                p_scores.append(val)

    for model in N_POOL:
        if model in results and 'metrics' in results[model]:
            val = results[model]['metrics'].get(metric)
            if val is not None:
                n_scores.append(val)

    if len(p_scores) > 0 and len(n_scores) > 0:
        bins = np.linspace(min(min(p_scores), min(n_scores)),
                           max(max(p_scores), max(n_scores)), 15)
        ax.hist(p_scores, bins=bins, alpha=0.6, label='P (α=1)', color='green')
        ax.hist(n_scores, bins=bins, alpha=0.6, label='N (α=5)', color='red')

        auc = auc_data.get(metric, {}).get('auc_roc', 0)
        ax.set_title(f'{METRIC_NAMES.get(metric, metric)}\nAUC: {auc:.3f}')
        ax.legend(fontsize=8)

# Hide empty subplots
for i in range(len(METRICS), 16):
    axes[i].axis('off')

plt.suptitle('Faithfulness: P/N Pool Score Distributions (13 Metrics)', fontsize=14)
plt.tight_layout()
plt.savefig(hist_dir / 'all_13metrics_histogram.png', dpi=150)
plt.close()
print(f"  Generated all_13metrics_histogram.png")
EOF

# Step 4: Create summary table
echo "[4/5] Creating summary table..."
python3 - << 'EOF'
import json
from pathlib import Path

log_dir = Path("/home/jaeung/activation-patching-unlearning/logs/faithfulness")

with open(log_dir / "auc_roc_13metrics.json") as f:
    auc_data = json.load(f)

# Metric categories
categories = {
    'Memorization': ['em', 'es', 'prob', 'paraprob', 'truth_ratio'],
    'Generation': ['rouge', 'para_rouge', 'jailbreak_rouge'],
    'Privacy (MIA)': ['mia_loss', 'mia_zlib', 'mia_min_k', 'mia_min_kpp'],
    'Ours': ['uds']
}

METRIC_NAMES = {
    'em': 'Exact Match', 'es': 'Extraction Strength',
    'prob': 'Probability', 'paraprob': 'Para.Prob',
    'truth_ratio': 'Truth Ratio',
    'rouge': 'ROUGE', 'para_rouge': 'Para.ROUGE', 'jailbreak_rouge': 'Jailbreak ROUGE',
    'mia_loss': 'MIA-LOSS', 'mia_zlib': 'MIA-ZLib',
    'mia_min_k': 'MIA-MinK', 'mia_min_kpp': 'MIA-MinK++',
    'uds': 'UDS'
}

# Generate markdown table
md = "# Faithfulness Results (13 Metrics)\n\n"
md += "**Settings**: SDPA attention, batch_size=32, 60 P/N models (30+30)\n\n"
md += "| Category | Metric | AUC-ROC | P Mean | N Mean |\n"
md += "|----------|--------|---------|--------|--------|\n"

for cat, metrics in categories.items():
    for metric in metrics:
        if metric in auc_data:
            d = auc_data[metric]
            md += f"| {cat} | {METRIC_NAMES.get(metric, metric)} | {d['auc_roc']:.4f} | {d['p_mean']:.4f} | {d['n_mean']:.4f} |\n"
            cat = ""  # Only show category once

# Summary
md += "\n## Summary\n\n"
aucs = [d['auc_roc'] for d in auc_data.values()]
md += f"- **Mean AUC-ROC**: {sum(aucs)/len(aucs):.4f}\n"
md += f"- **Best Metric**: {max(auc_data.items(), key=lambda x: x[1]['auc_roc'])[0]} ({max(aucs):.4f})\n"
md += f"- **Worst Metric**: {min(auc_data.items(), key=lambda x: x[1]['auc_roc'])[0]} ({min(aucs):.4f})\n"

with open(log_dir / "faithfulness_summary.md", 'w') as f:
    f.write(md)

print(f"  Generated faithfulness_summary.md")

# Also create summary JSON
summary = {
    'auc_by_metric': {k: v['auc_roc'] for k, v in auc_data.items()},
    'mean_auc': sum(aucs)/len(aucs),
    'best_metric': max(auc_data.items(), key=lambda x: x[1]['auc_roc'])[0],
    'worst_metric': min(auc_data.items(), key=lambda x: x[1]['auc_roc'])[0],
    'settings': {
        'attention': 'sdpa',
        'batch_size': 32,
        'p_pool_size': 30,
        'n_pool_size': 30
    }
}

with open(log_dir / "summary_13metrics.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  Updated summary_13metrics.json")
EOF

# Step 5: Copy histograms to docs
echo "[5/5] Updating docs..."
DOCS_DIR="$PROJECT_DIR/docs/0205"
mkdir -p "$DOCS_DIR/faithfulness_histograms"
cp "$LOG_DIR/histograms/"*.png "$DOCS_DIR/faithfulness_histograms/" 2>/dev/null
cp "$LOG_DIR/faithfulness_summary.md" "$DOCS_DIR/" 2>/dev/null
echo "  Copied histograms and summary to $DOCS_DIR"

echo ""
echo "=== Faithfulness Complete ==="
echo "Results in: $LOG_DIR"
echo "Histograms in: $LOG_DIR/histograms/"
echo "Docs in: $DOCS_DIR/faithfulness_histograms/"
echo ""

# Resume EP10 experiments if requested
if [ "$1" == "--resume-ep10" ]; then
    echo "=== Resuming EP10 Experiments ==="
    cd "$PROJECT_DIR"

    # Check EP10 status
    python3 - << 'PYEOF'
import json
from pathlib import Path

runs_dir = Path("runs/ep10")
for metric in ['memorization', 'privacy', 'utility', 'uds']:
    results_file = runs_dir / metric / 'results.json'
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        if metric == 'uds':
            count = len(data)
        else:
            count = len(data.get('results', []))
        print(f"  {metric}: {count}/75 complete")
    else:
        print(f"  {metric}: 0/75 complete")
PYEOF

    # Run EP10 on both GPUs
    python scripts/run_ep10_experiments.py --gpu 0 --start 0 --end 38 > logs/ep10_gpu0.log 2>&1 &
    python scripts/run_ep10_experiments.py --gpu 1 --start 38 --end 75 > logs/ep10_gpu1.log 2>&1 &

    echo "EP10 experiments started on both GPUs"
    echo "Logs: logs/ep10_gpu0.log, logs/ep10_gpu1.log"
fi

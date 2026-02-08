#!/bin/bash
# Queue: Wait for faithfulness -> Run robustness -> Run EP10
# Usage: nohup bash scripts/queue_after_faithfulness.sh > logs/queue_full.log 2>&1 &

set -e
cd /home/jaeung/activation-patching-unlearning

echo "$(date) === Queue started ==="
echo "Waiting for Faithfulness SDPA to complete..."

# ============================================================
# Wait for Faithfulness to complete (60 models each GPU)
# ============================================================
while true; do
  g0=$(python3 -c "import json; print(len(json.load(open('runs/meta_eval/table2_faithfulness_v2_gpu0/results.json'))))" 2>/dev/null || echo 0)
  g1=$(python3 -c "import json; print(len(json.load(open('runs/meta_eval/table2_faithfulness_v2_gpu1/results.json'))))" 2>/dev/null || echo 0)
  echo "$(date '+%H:%M:%S') Faithfulness: GPU0=$g0/60 GPU1=$g1/60"

  if [ "$g0" = "60" ] && [ "$g1" = "60" ]; then
    echo "$(date) Faithfulness complete!"
    break
  fi
  sleep 60
done

sleep 10

# ============================================================
# Merge Faithfulness results
# ============================================================
echo ""
echo "$(date) Merging faithfulness results..."

python3 << 'EOF'
import json, numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

r0 = json.loads(Path("runs/meta_eval/table2_faithfulness_v2_gpu0/results.json").read_text())
r1 = json.loads(Path("runs/meta_eval/table2_faithfulness_v2_gpu1/results.json").read_text())

# Merge by model key
results = {}
all_keys = set(list(r0.keys()) + list(r1.keys()))
for key in all_keys:
    v0 = r0.get(key, {})
    v1 = r1.get(key, {})
    merged = dict(v0)
    if "metrics" not in merged:
        merged["metrics"] = {}
    if "metrics" in v1:
        for m, val in v1["metrics"].items():
            if m not in merged["metrics"]:
                merged["metrics"][m] = val
    if "pool" not in merged and "pool" in v1:
        merged["pool"] = v1["pool"]
    results[key] = merged

out_dir = Path("runs/meta_eval/table2_faithfulness_v2")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "results.json").write_text(json.dumps(results, indent=2))

METRICS = ["em", "es", "prob", "paraprob", "truth_ratio",
           "rouge", "para_rouge", "jailbreak_rouge",
           "mia_loss", "mia_zlib", "mia_min_k", "mia_min_kpp", "uds"]

metric_auc = {}
print(f"{'Metric':<18} {'AUC':>8} {'P':>4} {'N':>4}")
print("-" * 40)
for metric in METRICS:
    p_scores, n_scores = [], []
    for model_id, info in results.items():
        score = info.get("metrics", {}).get(metric)
        if score is None:
            continue
        if info.get("pool") == "P":
            p_scores.append(score)
        else:
            n_scores.append(score)

    if len(p_scores) >= 2 and len(n_scores) >= 2:
        labels = [1] * len(p_scores) + [0] * len(n_scores)
        scores = p_scores + n_scores
        auc = roc_auc_score(labels, scores)
    else:
        auc = None

    metric_auc[metric] = {
        "auc": auc,
        "p_count": len(p_scores),
        "n_count": len(n_scores),
        "p_mean": float(np.mean(p_scores)) if p_scores else None,
        "n_mean": float(np.mean(n_scores)) if n_scores else None,
    }
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    print(f"{metric:<18} {auc_str:>8} {len(p_scores):>4} {len(n_scores):>4}")

summary = {"faithfulness": metric_auc, "metrics": METRICS, "n_models": len(results)}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSaved: {out_dir}/summary.json")
EOF

# ============================================================
# Start Robustness v2 (filtered)
# ============================================================
echo ""
echo "$(date) Starting Robustness v2 (filtered) on both GPUs..."

python3 scripts/meta_eval_robustness.py \
  --gpu 0 --metrics table2 --batch_size 32 \
  --attn_implementation sdpa \
  --out_dir runs/meta_eval/table2_robustness_v2_gpu0 \
  --faithfulness_result runs/meta_eval/table2_faithfulness_v2/summary.json \
  --faithfulness_pn_results runs/meta_eval/table2_faithfulness_v2/results.json \
  --filter_insufficient \
  --model_start 0 --model_end 37 \
  > logs/robustness_v2_gpu0.log 2>&1 &
PID_R0=$!

python3 scripts/meta_eval_robustness.py \
  --gpu 1 --metrics table2 --batch_size 32 \
  --attn_implementation sdpa \
  --out_dir runs/meta_eval/table2_robustness_v2_gpu1 \
  --faithfulness_result runs/meta_eval/table2_faithfulness_v2/summary.json \
  --faithfulness_pn_results runs/meta_eval/table2_faithfulness_v2/results.json \
  --filter_insufficient \
  --model_start 37 --model_end 75 \
  > logs/robustness_v2_gpu1.log 2>&1 &
PID_R1=$!

echo "Robustness PIDs: GPU0=$PID_R0, GPU1=$PID_R1"

# Wait for robustness
while kill -0 $PID_R0 2>/dev/null || kill -0 $PID_R1 2>/dev/null; do
  g0=$(python3 -c "
import json
try:
    d = json.load(open('runs/meta_eval/table2_robustness_v2_gpu0/results.json'))
    skip = {'retain_before','retain_after'}
    print(sum(1 for k,v in d.items() if k not in skip and v.get('quantization_Q')))
except: print(0)" 2>/dev/null || echo 0)
  g1=$(python3 -c "
import json
try:
    d = json.load(open('runs/meta_eval/table2_robustness_v2_gpu1/results.json'))
    skip = {'retain_before','retain_after'}
    print(sum(1 for k,v in d.items() if k not in skip and v.get('quantization_Q')))
except: print(0)" 2>/dev/null || echo 0)
  echo "$(date '+%H:%M:%S') Robustness: GPU0=$g0/37 GPU1=$g1/38"
  sleep 120
done

wait $PID_R0 || echo "WARNING: GPU0 robustness error"
wait $PID_R1 || echo "WARNING: GPU1 robustness error"

echo "$(date) Robustness complete!"

# ============================================================
# Merge Robustness results
# ============================================================
echo ""
echo "$(date) Merging robustness results..."

python3 << 'EOF'
import json
from pathlib import Path

r0 = json.loads(Path("runs/meta_eval/table2_robustness_v2_gpu0/results.json").read_text())
r1 = json.loads(Path("runs/meta_eval/table2_robustness_v2_gpu1/results.json").read_text())

merged = {}
all_keys = set(list(r0.keys()) + list(r1.keys()))
for key in all_keys:
    v0 = r0.get(key, {})
    v1 = r1.get(key, {})
    n0 = len(v0.get("metrics_before", {})) if isinstance(v0, dict) else 0
    n1 = len(v1.get("metrics_before", {})) if isinstance(v1, dict) else 0
    base = dict(v0 if n0 >= n1 else v1)
    other = v1 if n0 >= n1 else v0
    for sub_key in ["metrics_before", "relearning_R", "metrics_after_relearn",
                    "quantization_Q", "metrics_after_quant"]:
        if sub_key in other and isinstance(other.get(sub_key), dict):
            if sub_key not in base:
                base[sub_key] = {}
            if isinstance(base.get(sub_key), dict):
                for m, val in other[sub_key].items():
                    if m not in base[sub_key]:
                        base[sub_key][m] = val
    merged[key] = base

out_dir = Path("runs/meta_eval/table2_robustness_v2")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "results.json").write_text(json.dumps(merged, indent=2))
print(f"Merged: {len(merged)} entries -> {out_dir / 'results.json'}")
EOF

# ============================================================
# Start EP10 experiments
# ============================================================
echo ""
echo "$(date) Starting EP10 experiments..."

bash scripts/run_ep10_gpu0.sh > logs/ep10_gpu0.log 2>&1 &
PID_E0=$!

bash scripts/run_ep10_gpu1.sh > logs/ep10_gpu1.log 2>&1 &
PID_E1=$!

echo "EP10 PIDs: GPU0=$PID_E0, GPU1=$PID_E1"

# Wait for EP10
wait $PID_E0 || echo "WARNING: GPU0 EP10 error"
wait $PID_E1 || echo "WARNING: GPU1 EP10 error"

echo ""
echo "$(date) === All experiments complete! ==="

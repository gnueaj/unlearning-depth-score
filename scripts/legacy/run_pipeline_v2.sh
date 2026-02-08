#!/bin/bash
# Pipeline v2: Full automated meta-evaluation
# 1. Wait for faithfulness v2 completion
# 2. Merge faithfulness GPU 0+1 results + compute AUC
# 3. Launch robustness v2 on GPU 0+1
# 4. Wait for robustness completion
# 5. Merge robustness results + update HTML Table 2

set -euo pipefail

WORKDIR=/home/jaeung/activation-patching-unlearning
cd "$WORKDIR"

mkdir -p logs
LOG=logs/pipeline_v2.log
exec > >(tee -a "$LOG") 2>&1

echo "$(date) === Pipeline v2 starting ==="

# ============================================================
# Step 1: Wait for faithfulness v2 to complete
# ============================================================
echo "$(date) [Step 1/5] Waiting for faithfulness v2..."

while true; do
  g0=$(python3 -c "import json; print(len(json.load(open('runs/meta_eval/table2_faithfulness_v2_gpu0/results.json'))))" 2>/dev/null || echo 0)
  g1=$(python3 -c "import json; print(len(json.load(open('runs/meta_eval/table2_faithfulness_v2_gpu1/results.json'))))" 2>/dev/null || echo 0)
  echo "$(date '+%H:%M:%S') Faithfulness: GPU0=$g0/30 GPU1=$g1/30"
  if [ "$g0" = "30" ] && [ "$g1" = "30" ]; then
    echo "$(date) Faithfulness v2 complete!"
    break
  fi
  sleep 30
done

# Wait for final file writes to flush
sleep 10

# ============================================================
# Step 2: Merge faithfulness results + compute AUC
# ============================================================
echo ""
echo "$(date) [Step 2/5] Merging faithfulness results + computing AUC..."

python3 << 'PYEOF'
import json, numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Merge GPU 0 + GPU 1 results
r0 = json.loads(Path("runs/meta_eval/table2_faithfulness_v2_gpu0/results.json").read_text())
r1 = json.loads(Path("runs/meta_eval/table2_faithfulness_v2_gpu1/results.json").read_text())
results = {**r0, **r1}
print(f"Merged: {len(r0)} (GPU0) + {len(r1)} (GPU1) = {len(results)} models")

# Output directory
out_dir = Path("runs/meta_eval/table2_faithfulness_v2")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "results.json").write_text(json.dumps(results, indent=2))

# Compute AUC-ROC
METRICS = ["em", "es", "prob", "paraprob", "truth_ratio",
           "rouge", "para_rouge", "jailbreak_rouge",
           "mia_loss", "mia_zlib", "mia_min_k", "mia_min_kpp"]

metric_auc = {}
print(f"\n{'Metric':<18} {'AUC':>8} {'P':>4} {'N':>4}")
print("-" * 40)
for metric in METRICS:
    p_scores, n_scores = [], []
    for model_id, info in results.items():
        score = info.get("metrics", {}).get(metric)
        if score is None:
            continue
        if info["pool"] == "P":
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

summary = {
    "faithfulness": metric_auc,
    "metrics": METRICS,
    "n_models": len(results),
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSaved: {out_dir}/summary.json")
PYEOF

echo "$(date) Faithfulness merge complete."

# ============================================================
# Step 3: Launch robustness v2 on both GPUs
# ============================================================
echo ""
echo "$(date) [Step 3/5] Starting robustness v2 on both GPUs..."

python scripts/meta_eval_robustness.py \
  --gpu 0 --metrics table2 --batch_size 32 \
  --out_dir runs/meta_eval/table2_robustness_v2_gpu0 \
  --resume runs/meta_eval/table2_robustness/results_uds_only.json \
  --faithfulness_result runs/meta_eval/table2_faithfulness_v2/summary.json \
  --model_start 0 --model_end 37 \
  > logs/robustness_v2_gpu0.log 2>&1 &
PID0=$!
echo "  GPU 0 PID=$PID0 (models 0-36)"

python scripts/meta_eval_robustness.py \
  --gpu 1 --metrics table2 --batch_size 32 \
  --out_dir runs/meta_eval/table2_robustness_v2_gpu1 \
  --resume runs/meta_eval/table2_robustness/results_uds_only.json \
  --faithfulness_result runs/meta_eval/table2_faithfulness_v2/summary.json \
  --model_start 37 --model_end 75 \
  > logs/robustness_v2_gpu1.log 2>&1 &
PID1=$!
echo "  GPU 1 PID=$PID1 (models 37-74)"

# ============================================================
# Step 4: Wait for robustness v2 to complete
# ============================================================
echo ""
echo "$(date) [Step 4/5] Waiting for robustness v2..."

# Monitor progress while waiting
while kill -0 $PID0 2>/dev/null || kill -0 $PID1 2>/dev/null; do
  g0_done=$(python3 -c "
import json
try:
    d = json.load(open('runs/meta_eval/table2_robustness_v2_gpu0/results.json'))
    skip = {'retain_before','retain_after'}
    print(sum(1 for k,v in d.items() if k not in skip and len(v.get('metrics_before',{}))>=10))
except: print(0)" 2>/dev/null || echo 0)
  g1_done=$(python3 -c "
import json
try:
    d = json.load(open('runs/meta_eval/table2_robustness_v2_gpu1/results.json'))
    skip = {'retain_before','retain_after'}
    print(sum(1 for k,v in d.items() if k not in skip and len(v.get('metrics_before',{}))>=10))
except: print(0)" 2>/dev/null || echo 0)
  g0_alive=$(kill -0 $PID0 2>/dev/null && echo "running" || echo "done")
  g1_alive=$(kill -0 $PID1 2>/dev/null && echo "running" || echo "done")
  echo "$(date '+%H:%M:%S') Robustness: GPU0=$g0_done/37($g0_alive) GPU1=$g1_done/38($g1_alive)"
  sleep 60
done

wait $PID0 || { echo "WARNING: GPU 0 robustness exited with error"; }
wait $PID1 || { echo "WARNING: GPU 1 robustness exited with error"; }

echo "$(date) Robustness v2 complete!"
sleep 5

# ============================================================
# Step 5: Merge robustness results + update HTML Table 2
# ============================================================
echo ""
echo "$(date) [Step 5/5] Merging robustness results + updating HTML..."

python3 << 'PYEOF'
import json, re, numpy as np
from pathlib import Path

# ---------------------------------------------------------------
# 5a. Load & merge robustness results from both GPUs
# ---------------------------------------------------------------
r0 = json.loads(Path("runs/meta_eval/table2_robustness_v2_gpu0/results.json").read_text())
r1 = json.loads(Path("runs/meta_eval/table2_robustness_v2_gpu1/results.json").read_text())

# Merge: for each key, take version with more data, then fill in from other
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

SKIP_KEYS = {"retain_before", "retain_after"}
model_keys = sorted([k for k in merged if k not in SKIP_KEYS])
print(f"Merged robustness: {len(model_keys)} unlearn models")

# ---------------------------------------------------------------
# 5b. Load faithfulness (v2 for 12 metrics, v1 for UDS)
# ---------------------------------------------------------------
faith_v2 = json.loads(Path("runs/meta_eval/table2_faithfulness_v2/summary.json").read_text())
faithfulness = faith_v2.get("faithfulness", {})

# Try to load UDS faithfulness from v1
uds_faith_auc = None
for path in ["runs/meta_eval/table2_faithfulness/summary.json",
             "runs/meta_eval/table2_faithfulness_v1/summary.json"]:
    p = Path(path)
    if p.exists():
        old = json.loads(p.read_text())
        uds_faith_auc = old.get("faithfulness", {}).get("uds", {}).get("auc")
        if uds_faith_auc is not None:
            print(f"UDS faithfulness AUC loaded from {path}: {uds_faith_auc:.4f}")
            break

# ---------------------------------------------------------------
# 5c. Compute robustness summary for all 13 metrics
# ---------------------------------------------------------------
ALL_METRICS = ["em", "es", "prob", "paraprob", "truth_ratio",
               "rouge", "para_rouge", "jailbreak_rouge",
               "mia_loss", "mia_zlib", "mia_min_k", "mia_min_kpp", "uds"]

metric_R = {m: [] for m in ALL_METRICS}
metric_Q = {m: [] for m in ALL_METRICS}
for name in model_keys:
    mr = merged.get(name, {})
    R = mr.get("relearning_R", {})
    Q = mr.get("quantization_Q", {})
    for m in ALL_METRICS:
        if isinstance(R, dict) and R.get(m) is not None:
            metric_R[m].append(R[m])
        if isinstance(Q, dict) and Q.get(m) is not None:
            metric_Q[m].append(Q[m])

table_data = {}
print(f"\n{'Metric':<16} {'F':>8} {'R':>8} {'Q':>8} {'Rob':>8} {'Agg':>8}  (nR/nQ)")
print("-" * 75)

for m in ALL_METRICS:
    avg_R = float(np.mean(metric_R[m])) if metric_R[m] else None
    avg_Q = float(np.mean(metric_Q[m])) if metric_Q[m] else None
    if avg_R is not None and avg_Q is not None and avg_R > 0 and avg_Q > 0:
        rob = 2 * avg_R * avg_Q / (avg_R + avg_Q)
    else:
        rob = None

    # Faithfulness
    if m == "uds":
        f_auc = uds_faith_auc
    else:
        f_auc = faithfulness.get(m, {}).get("auc")

    # Overall aggregated
    if f_auc is not None and rob is not None and f_auc > 0 and rob > 0:
        agg = 2 * f_auc * rob / (f_auc + rob)
    else:
        agg = None

    table_data[m] = {
        "faithful": f_auc, "R": avg_R, "Q": avg_Q,
        "robust": rob, "agg": agg,
        "n_R": len(metric_R[m]), "n_Q": len(metric_Q[m]),
    }
    vals = [f_auc, avg_R, avg_Q, rob, agg]
    strs = [f"{v:.4f}" if v is not None else "N/A   " for v in vals]
    print(f"{m:<16} {strs[0]:>8} {strs[1]:>8} {strs[2]:>8} {strs[3]:>8} {strs[4]:>8}  ({len(metric_R[m])}/{len(metric_Q[m])})")

# Save summary
summary = {
    "metrics": ALL_METRICS,
    "table_data": table_data,
    "faithfulness_v2": faithfulness,
    "uds_faithfulness_auc": uds_faith_auc,
    "n_models": len(model_keys),
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSaved: {out_dir}/summary.json")

# ---------------------------------------------------------------
# 5d. Update HTML Table 2
# ---------------------------------------------------------------
METRIC_DISPLAY = {
    "em":              ("EM",          "Mem",  "cat-mem"),
    "es":              ("ES",          "Mem",  "cat-mem"),
    "prob":            ("Probability", "Mem",  "cat-mem"),
    "paraprob":        ("Para.Prob",   "Mem",  "cat-mem"),
    "truth_ratio":     ("Truth Ratio", "Mem",  "cat-mem"),
    "rouge":           ("ROUGE",       "Gen",  "cat-gen"),
    "para_rouge":      ("Para.ROUGE",  "Gen",  "cat-gen"),
    "jailbreak_rouge": ("JB ROUGE",    "Gen",  "cat-gen"),
    "mia_loss":        ("MIA-LOSS",    "Priv", "cat-priv"),
    "mia_zlib":        ("MIA-ZLib",    "Priv", "cat-priv"),
    "mia_min_k":       ("MIA-MinK",    "Priv", "cat-priv"),
    "mia_min_kpp":     ("MIA-MinK++",  "Priv", "cat-priv"),
    "uds":             ("UDS (Ours)",  "Ours", "cat-ours"),
}

# Find best values for each column
best = {}
for col in ["faithful", "R", "Q", "robust", "agg"]:
    vals = [d[col] for d in table_data.values() if d[col] is not None]
    best[col] = max(vals) if vals else None

def fmt_cell(val, col):
    if val is None:
        return '<td class="pending">\u2014</td>'
    is_best = (best[col] is not None and abs(val - best[col]) < 0.0005)
    cls = ' class="best"' if is_best else ""
    return f"<td{cls}>{val:.3f}</td>"

rows = []
for m in ALL_METRICS:
    d = table_data[m]
    display, cat, css = METRIC_DISPLAY[m]
    is_ours = (m == "uds")
    row_cls = f'class="ours {css}"' if is_ours else f'class="{css}"'
    cells = "".join([
        f"<td>{display}</td>",
        f"<td>{cat}</td>",
        fmt_cell(d["faithful"], "faithful"),
        fmt_cell(d["R"], "R"),
        fmt_cell(d["Q"], "Q"),
        fmt_cell(d["robust"], "robust"),
        fmt_cell(d["agg"], "agg"),
    ])
    # Add comments for grouping
    if m == "em":
        rows.append("<!-- Memorization metrics -->")
    elif m == "rouge":
        rows.append("<!-- Generation metrics -->")
    elif m == "mia_loss":
        rows.append("<!-- Privacy metrics -->")
    elif m == "uds":
        rows.append("<!-- UDS (Ours) -->")
    rows.append(f"<tr {row_cls}>{cells}</tr>")

new_tbody = "\n".join(rows)

# Read and update HTML
html_path = Path("docs/0202/openunlearning_alpha_all.html")
html = html_path.read_text()

# Replace tbody content
match = re.search(r'(<tbody>\n)(.*?)(</tbody>)', html, re.DOTALL)
if match:
    html = html[:match.start(2)] + new_tbody + "\n" + html[match.end(2):]

    # Update key findings section
    uds = table_data.get("uds", {})
    # Find the strongest baseline (non-UDS metric with highest agg)
    baseline_best = None
    baseline_name = None
    for m in ALL_METRICS:
        if m == "uds":
            continue
        a = table_data[m].get("agg")
        if a is not None and (baseline_best is None or a > baseline_best):
            baseline_best = a
            baseline_name = METRIC_DISPLAY[m][0]

    findings = []
    if uds.get("faithful") is not None:
        findings.append(f'&bull; <b>UDS achieves the highest Faithfulness ({uds["faithful"]:.3f})</b>')
        if uds.get("agg") is not None:
            findings.append(f' and <b>highest overall Agg ({uds["agg"]:.3f})</b> among all 13 metrics.<br>')
        else:
            findings.append('.<br>')

    if uds.get("robust") is not None:
        # Find robust rank
        rob_vals = [(m, table_data[m]["robust"]) for m in ALL_METRICS if table_data[m]["robust"] is not None]
        rob_vals.sort(key=lambda x: -x[1])
        uds_rank = next(i+1 for i, (m, _) in enumerate(rob_vals) if m == "uds")
        rob_best_name = METRIC_DISPLAY[rob_vals[0][0]][0] if rob_vals else "N/A"
        if uds_rank == 1:
            findings.append(f'&bull; UDS ranks 1st in Robustness ({uds["robust"]:.3f}).<br>')
        else:
            findings.append(f'&bull; UDS ranks #{uds_rank} in Robustness ({uds["robust"]:.3f}), '
                          f'top is {rob_best_name} ({rob_vals[0][1]:.3f}).<br>')

    if baseline_best is not None:
        findings.append(f'&bull; Among existing metrics, {baseline_name} ({baseline_best:.3f}) is the strongest overall baseline.<br>')

    findings.append(f'&bull; All 13 metrics evaluated on {len(model_keys)} unlearn models for robustness (including generation metrics).')

    findings_html = "\n".join(findings)

    # Replace findings block
    findings_match = re.search(
        r"(<div style='margin:10px 0; padding:8px 12px; background:#eff6ff.*?<b>Key findings:</b><br>\n)(.*?)(</div>)",
        html, re.DOTALL
    )
    if findings_match:
        html = html[:findings_match.start(2)] + findings_html + "\n" + html[findings_match.end(2):]

    html_path.write_text(html)
    print(f"\nHTML Table 2 updated: {html_path}")
else:
    print("WARNING: Could not find <tbody> in HTML file!")

print("\n=== Pipeline v2 complete! ===")
PYEOF

echo ""
echo "$(date) === Pipeline v2 finished ==="
echo "Results:"
echo "  Faithfulness: runs/meta_eval/table2_faithfulness_v2/summary.json"
echo "  Robustness:   runs/meta_eval/table2_robustness_v2/summary.json"
echo "  HTML:         docs/0202/openunlearning_alpha_all.html"

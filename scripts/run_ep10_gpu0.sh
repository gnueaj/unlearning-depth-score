#!/bin/bash
# EP10 Experiments - GPU 0 (models 0-37)
# 4 Metrics: Memorization, Privacy, Utility, UDS

set -e
cd /home/jaeung/activation-patching-unlearning

LOG_DIR="logs/ep10"
mkdir -p "$LOG_DIR"

# Load first half of EP10 models (0-37)
MODELS=$(python3 -c "
import json
models = json.load(open('runs/ep10/model_list.json'))['ep10_models']
print(' '.join(models[:38]))
")

echo "=============================================="
echo "EP10 GPU0 - Models 0-37 (38 models)"
echo "Start: $(date)"
echo "=============================================="

for MODEL in $MODELS; do
    echo ""
    echo "[GPU0] Processing: $MODEL"
    echo "  Time: $(date '+%H:%M:%S')"

    # 1. Memorization
    if [ ! -f "runs/ep10/memorization/$MODEL/summary.json" ]; then
        echo "  [1/4] Memorization..."
        python3 -m patchscope.memorization_eval \
            --model "$MODEL" \
            --hf_dataset locuslab/TOFU \
            --hf_config forget10_perturbed \
            --use_chat_template \
            --out_dir "runs/ep10/memorization/$MODEL" \
            --attn_implementation sdpa \
            --gpu 0 2>&1 | tail -5
    else
        echo "  [1/4] Memorization: SKIP (exists)"
    fi

    # 2. Privacy
    if [ ! -f "runs/ep10/privacy/$MODEL/summary.json" ]; then
        echo "  [2/4] Privacy..."
        python3 -m patchscope.privacy_eval \
            --model "$MODEL" \
            --use_chat_template \
            --out_dir "runs/ep10/privacy/$MODEL" \
            --attn_implementation sdpa \
            --gpu 0 2>&1 | tail -5
    else
        echo "  [2/4] Privacy: SKIP (exists)"
    fi

    # 3. Utility
    if [ ! -f "runs/ep10/utility/$MODEL/summary.json" ]; then
        echo "  [3/4] Utility..."
        python3 -m patchscope.utility_eval \
            --model "$MODEL" \
            --use_chat_template \
            --out_dir "runs/ep10/utility/$MODEL" \
            --attn_implementation sdpa \
            --gpu 0 2>&1 | tail -5
    else
        echo "  [3/4] Utility: SKIP (exists)"
    fi

    # 4. UDS
    UDS_EXISTS=$(python3 -c "
import json
from pathlib import Path
f = Path('runs/ep10/uds/results.json')
if f.exists():
    d = json.load(open(f))
    print('1' if '$MODEL' in d else '0')
else:
    print('0')
")
    if [ "$UDS_EXISTS" == "0" ]; then
        echo "  [4/4] UDS..."
        python3 exp_s1_teacher_forcing.py \
            --unlearn_model "$MODEL" \
            --gpu 0 \
            --batch_size 32 \
            --attn_implementation sdpa 2>&1 | tail -5

        # Extract UDS and save to consolidated file
        python3 << EOF
import json, glob
from pathlib import Path

# Find latest run directory
runs = sorted(glob.glob(f'runs/*_tf_${MODEL}_layer'))
if runs:
    summary = Path(runs[-1]) / 'summary.json'
    if summary.exists():
        data = json.load(open(summary))
        results_file = Path('runs/ep10/uds/results.json')
        results = json.load(open(results_file)) if results_file.exists() else {}
        results['$MODEL'] = {
            'avg_uds': data.get('avg_uds', data.get('avg_udr')),
            'run_dir': runs[-1]
        }
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"    UDS: {results['$MODEL']['avg_uds']:.4f}")
EOF
    else
        echo "  [4/4] UDS: SKIP (exists)"
    fi

    # 5. Clean up HF cache for this model to save disk space
    echo "  [Cleanup] Removing HF cache for $MODEL..."
    python3 << EOF
from pathlib import Path
from patchscope.unlearn_models import UNLEARN_MODELS_FULL
import shutil

model_name = '$MODEL'
if model_name in UNLEARN_MODELS_FULL:
    hf_id = UNLEARN_MODELS_FULL[model_name]
    cache_name = 'models--' + hf_id.replace('/', '--')
    cache_path = Path.home() / '.cache/huggingface/hub' / cache_name
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"    Deleted: {cache_path.name}")
    else:
        print(f"    Not in cache")
EOF

    echo "  Done: $MODEL"
done

echo ""
echo "=============================================="
echo "EP10 GPU0 COMPLETED!"
echo "End: $(date)"
echo "=============================================="

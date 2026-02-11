#!/bin/bash
# EP10 All Metrics - 75 models × 4 metrics (Mem, Priv, Util, UDS)
# Usage: ./scripts/run_ep10_all_metrics.sh GPU_ID START END
# Example: ./scripts/run_ep10_all_metrics.sh 0 0 38  (first half)
#          ./scripts/run_ep10_all_metrics.sh 1 38 75 (second half)

GPU=$1
START=${2:-0}
END=${3:-75}

export CUDA_VISIBLE_DEVICES=$GPU
cd /home/jaeung/activation-patching-unlearning

# EP10 models (75 total)
MODELS=(
    "altpo_lr1e5_b01_a1_ep10" "altpo_lr1e5_b01_a2_ep10" "altpo_lr1e5_b01_a5_ep10"
    "altpo_lr2e5_b01_a1_ep10" "altpo_lr2e5_b01_a2_ep10" "altpo_lr2e5_b01_a5_ep10"
    "altpo_lr5e5_b01_a1_ep10" "altpo_lr5e5_b01_a2_ep10" "altpo_lr5e5_b01_a5_ep10"
    "graddiff_lr1e5_a1_ep10" "graddiff_lr1e5_a2_ep10" "graddiff_lr1e5_a5_ep10"
    "graddiff_lr2e5_a1_ep10" "graddiff_lr2e5_a2_ep10" "graddiff_lr2e5_a5_ep10"
    "graddiff_lr5e5_a1_ep10" "graddiff_lr5e5_a2_ep10" "graddiff_lr5e5_a5_ep10"
    "idkdpo_lr1e5_b01_a1_ep10" "idkdpo_lr1e5_b01_a2_ep10" "idkdpo_lr1e5_b01_a5_ep10"
    "idkdpo_lr2e5_b01_a1_ep10" "idkdpo_lr2e5_b01_a2_ep10" "idkdpo_lr2e5_b01_a5_ep10"
    "idkdpo_lr5e5_b01_a1_ep10" "idkdpo_lr5e5_b01_a2_ep10" "idkdpo_lr5e5_b01_a5_ep10"
    "idknll_lr1e5_a1_ep10" "idknll_lr1e5_a2_ep10" "idknll_lr1e5_a5_ep10"
    "idknll_lr2e5_a1_ep10" "idknll_lr2e5_a2_ep10" "idknll_lr2e5_a5_ep10"
    "idknll_lr5e5_a1_ep10" "idknll_lr5e5_a2_ep10" "idknll_lr5e5_a5_ep10"
    "npo_lr1e5_b01_a1_ep10" "npo_lr1e5_b01_a2_ep10" "npo_lr1e5_b01_a5_ep10"
    "npo_lr2e5_b01_a1_ep10" "npo_lr2e5_b01_a2_ep10" "npo_lr2e5_b01_a5_ep10"
    "npo_lr5e5_b01_a1_ep10" "npo_lr5e5_b01_a2_ep10" "npo_lr5e5_b01_a5_ep10"
    "rmu_lr1e5_l10_s10_ep10" "rmu_lr1e5_l15_s10_ep10" "rmu_lr1e5_l5_s10_ep10"
    "rmu_lr2e5_l10_s10_ep10" "rmu_lr2e5_l15_s10_ep10" "rmu_lr2e5_l5_s10_ep10"
    "rmu_lr5e5_l10_s10_ep10" "rmu_lr5e5_l15_s10_ep10" "rmu_lr5e5_l5_s10_ep10"
    "simnpo_lr1e5_b35_a1_d1_g0125_ep10" "simnpo_lr1e5_b35_a1_d1_g025_ep10"
    "simnpo_lr1e5_b45_a1_d1_g0125_ep10" "simnpo_lr1e5_b45_a1_d1_g025_ep10"
    "simnpo_lr2e5_b35_a1_d1_g0125_ep10" "simnpo_lr2e5_b35_a1_d1_g025_ep10"
    "simnpo_lr2e5_b45_a1_d1_g0125_ep10" "simnpo_lr2e5_b45_a1_d1_g025_ep10"
    "simnpo_lr5e5_b35_a1_d1_g0125_ep10" "simnpo_lr5e5_b35_a1_d1_g025_ep10"
    "simnpo_lr5e5_b45_a1_d1_g0125_ep10" "simnpo_lr5e5_b45_a1_d1_g025_ep10"
    "undial_lr1e4_b10_a1_ep10" "undial_lr1e4_b10_a2_ep10" "undial_lr1e4_b10_a5_ep10"
    "undial_lr1e5_b10_a1_ep10" "undial_lr1e5_b10_a2_ep10" "undial_lr1e5_b10_a5_ep10"
    "undial_lr3e4_b10_a1_ep10" "undial_lr3e4_b10_a2_ep10" "undial_lr3e4_b10_a5_ep10"
)

TOTAL=${#MODELS[@]}
echo "=== EP10 All Metrics - GPU $GPU ==="
echo "Models: $START to $END (of $TOTAL)"
echo "Metrics: Memorization, Privacy, Utility, UDS"
echo "========================================"

for ((i=START; i<END && i<TOTAL; i++)); do
    MODEL=${MODELS[$i]}
    echo ""
    echo "[$((i+1))/$TOTAL] $MODEL"
    echo "----------------------------------------"

    # 1. Memorization (skip if exists)
    MEM_DIR="runs/ep10/memorization/$MODEL"
    if [ -f "$MEM_DIR/summary.json" ]; then
        echo "  [MEM] SKIP (exists)"
    else
        echo "  [MEM] Running..."
        python -m uds.memorization_eval \
            --model "$MODEL" \
            --hf_dataset locuslab/TOFU \
            --hf_config forget10_perturbed \
            --use_chat_template \
            --batch_size 32 \
            --out_dir "$MEM_DIR" 2>&1 | tail -3
    fi

    # 2. Privacy (skip if exists)
    PRIV_DIR="runs/ep10/privacy/$MODEL"
    if [ -f "$PRIV_DIR/summary.json" ]; then
        echo "  [PRIV] SKIP (exists)"
    else
        echo "  [PRIV] Running..."
        python -m uds.privacy_eval \
            --model "$MODEL" \
            --use_chat_template \
            --batch_size 32 \
            --out_dir "$PRIV_DIR" 2>&1 | tail -3
    fi

    # 3. Utility (skip if exists)
    UTIL_DIR="runs/ep10/utility/$MODEL"
    if [ -f "$UTIL_DIR/summary.json" ]; then
        echo "  [UTIL] SKIP (exists)"
    else
        echo "  [UTIL] Running..."
        python -m uds.utility_eval \
            --model "$MODEL" \
            --use_chat_template \
            --batch_size 32 \
            --out_dir "$UTIL_DIR" 2>&1 | tail -3
    fi

    # 4. UDS (skip if exists)
    UDS_DIR="runs/ep10/uds/$MODEL"
    if [ -f "$UDS_DIR/summary.json" ]; then
        echo "  [UDS] SKIP (exists)"
    else
        echo "  [UDS] Running..."
        # UDS uses exp_s1 script with --gpu argument (sets CUDA_VISIBLE_DEVICES internally)
        python exp_s1_teacher_forcing.py \
            --unlearn_model "$MODEL" \
            --gpu $GPU \
            --batch_size 32 2>&1 | tail -5

        # Move results to uds folder
        LATEST=$(ls -td runs/*_tf_${MODEL}_layer 2>/dev/null | head -1)
        if [ -n "$LATEST" ] && [ -f "$LATEST/summary.json" ]; then
            rm -rf "$UDS_DIR"
            mv "$LATEST" "$UDS_DIR"
            echo "  [UDS] Saved to $UDS_DIR"
        fi
    fi

    echo "----------------------------------------"
done

echo ""
echo "=== COMPLETED GPU $GPU ==="

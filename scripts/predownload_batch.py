#!/usr/bin/env python3
"""Pre-download a batch of models from HuggingFace Hub."""

import sys
import time
from huggingface_hub import snapshot_download

# Full model list (same order as DEFAULT_MODELS in meta_eval_robustness.py)
ALL_MODELS = [
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
    # AltPO ep5 (9)
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha5_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha5_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha5_epoch5",
    # AltPO ep10 (9)
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha5_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha5_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha5_epoch10",
    # GradDiff ep5 (9)
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha5_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha5_epoch5",
    # GradDiff ep10 (9)
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha5_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha5_epoch10",
    # IdkDPO ep5 (9)
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha5_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha5_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha5_epoch5",
    # IdkDPO ep10 (9)
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha5_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha5_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha5_epoch10",
    # IdkNLL ep5 (9)
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha5_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha5_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha5_epoch5",
    # IdkNLL ep10 (9)
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha5_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha5_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha2_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha5_epoch10",
    # NPO ep5 (9)
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha2_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha5_epoch5",
]

# Also need full model for the script
EXTRA_MODELS = [
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
]


def main():
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 76

    batch = ALL_MODELS[start_idx:end_idx]
    # Always include full model
    all_to_download = EXTRA_MODELS + batch

    # Deduplicate
    seen = set()
    unique = []
    for m in all_to_download:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    print(f"Pre-downloading {len(unique)} models (batch [{start_idx}:{end_idx}])")
    print(f"=" * 60)

    total_start = time.time()
    for i, model_id in enumerate(unique):
        t0 = time.time()
        print(f"[{i+1}/{len(unique)}] {model_id}...", end=" ", flush=True)
        try:
            snapshot_download(model_id)
            elapsed = time.time() - t0
            print(f"OK ({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done! {len(unique)} models in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()

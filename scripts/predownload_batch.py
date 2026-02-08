#!/usr/bin/env python3
"""Pre-download a batch of models from HuggingFace Hub."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download
from scripts.meta_eval_robustness import DEFAULT_MODELS

FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"


def main():
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else len(DEFAULT_MODELS)

    all_model_ids = list(DEFAULT_MODELS.values())
    batch = all_model_ids[start_idx:end_idx]

    # Always include full model
    all_to_download = [FULL_MODEL] + batch

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

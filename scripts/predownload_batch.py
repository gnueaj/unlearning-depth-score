#!/usr/bin/env python3
"""Pre-download a batch of models from HuggingFace Hub.

Usage:
    python scripts/predownload_batch.py --epoch 10   # Batch 1: ep10 models (75)
    python scripts/predownload_batch.py --epoch 5    # Batch 2: ep5 models (75)
    python scripts/predownload_batch.py              # All 150 models
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download
from scripts.meta_eval_robustness import DEFAULT_MODELS

FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
RETAIN_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"


def main():
    parser = argparse.ArgumentParser(description="Pre-download models by epoch.")
    parser.add_argument("--epoch", type=int, choices=[5, 10], default=None,
                        help="Download only ep5 or ep10 models. Default: all.")
    args = parser.parse_args()

    # Filter by epoch suffix in short name
    if args.epoch is not None:
        suffix = f"_ep{args.epoch}"
        model_ids = [mid for name, mid in DEFAULT_MODELS.items()
                     if name.endswith(suffix)]
        print(f"Epoch {args.epoch}: {len(model_ids)} models")
    else:
        model_ids = [mid for name, mid in DEFAULT_MODELS.items()
                     if name != "retain"]
        print(f"All: {len(model_ids)} models")

    # Always include full + retain
    all_to_download = [FULL_MODEL, RETAIN_MODEL] + model_ids

    # Deduplicate
    seen = set()
    unique = []
    for m in all_to_download:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    epoch_label = f"ep{args.epoch}" if args.epoch else "all"
    print(f"Pre-downloading {len(unique)} models ({epoch_label})")
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

#!/usr/bin/env python3
"""
Run utility evaluation for all 30 unlearning models + full + retain.
Uses GPU 0 and 1 alternately.
"""

import subprocess
import sys
import os
from pathlib import Path

# All 30 models from alpha5 experiments
MODELS = [
    # SimNPO (3)
    "simnpo_lr1e5_b35_a1_d1_g0125_ep5",
    "simnpo_lr2e5_b35_a1_d1_g0125_ep5",
    "simnpo_lr5e5_b35_a1_d1_g0125_ep5",
    # GradDiff (3)
    "graddiff_lr1e5_a5_ep5",
    "graddiff_lr2e5_a5_ep5",
    "graddiff_lr5e5_a5_ep5",
    # IdkNLL (3)
    "idknll_lr1e5_a5_ep5",
    "idknll_lr2e5_a5_ep5",
    "idknll_lr5e5_a5_ep5",
    # NPO (3)
    "npo_lr1e5_b01_a5_ep5",
    "npo_lr2e5_b01_a5_ep5",
    "npo_lr5e5_b01_a5_ep5",
    # IdkDPO (3)
    "idkdpo_lr1e5_b01_a5_ep5",
    "idkdpo_lr2e5_b01_a5_ep5",
    "idkdpo_lr5e5_b01_a5_ep5",
    # AltPO (3)
    "altpo_lr1e5_b01_a5_ep5",
    "altpo_lr2e5_b01_a5_ep5",
    "altpo_lr5e5_b01_a5_ep5",
    # UNDIAL (3)
    "undial_lr1e5_b10_a5_ep5",
    "undial_lr1e4_b10_a5_ep5",
    "undial_lr3e4_b10_a5_ep5",
    # RMU-L5 (3)
    "rmu_lr1e5_l5_s10_ep5",
    "rmu_lr2e5_l5_s10_ep5",
    "rmu_lr5e5_l5_s10_ep5",
    # RMU-L10 (3)
    "rmu_lr1e5_l10_s10_ep5",
    "rmu_lr2e5_l10_s10_ep5",
    "rmu_lr5e5_l10_s10_ep5",
    # RMU-L15 (3)
    "rmu_lr1e5_l15_s10_ep5",
    "rmu_lr2e5_l15_s10_ep5",
    "rmu_lr5e5_l15_s10_ep5",
]

# Baseline models
BASELINE_MODELS = ["full", "retain"]

OUT_DIR = Path("runs/utility_eval_v2")


def run_model(model: str, gpu: int):
    """Run utility evaluation for a single model."""
    out_path = OUT_DIR / model
    if (out_path / "summary.json").exists():
        print(f"[SKIP] {model} - already exists")
        return True

    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / "run.log"

    cmd = [
        sys.executable, "-m", "uds.utility_eval",
        "--model", model,
        "--out_dir", str(out_path),
        "--batch_size", "4",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[GPU {gpu}] Running {model}...")
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"[FAIL] {model} - check {log_path}")
        return False
    print(f"[DONE] {model}")
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run baseline models first (sequentially)
    print("=" * 60)
    print("Running baseline models (full, retain)...")
    print("=" * 60)
    for model in BASELINE_MODELS:
        run_model(model, gpu=0)

    # Run all 30 unlearning models
    print("\n" + "=" * 60)
    print(f"Running {len(MODELS)} unlearning models...")
    print("=" * 60)

    # Sequential execution (one at a time to avoid OOM)
    for i, model in enumerate(MODELS):
        gpu = i % 2  # Alternate between GPU 0 and 1
        run_model(model, gpu)

    print("\n" + "=" * 60)
    print("All evaluations complete!")
    print(f"Results saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Meta-Evaluation: Robustness of unlearning metrics.

Evaluates two dimensions:
  1. Robustness to Relearning (R):
     - Finetune unlearn/retain models on forget10 for 1 epoch (lr=2e-5)
     - R = min(Δ_retain / Δ_unlearn, 1) where Δ = m_before - m_after

  2. Robustness to Quantization (Q):
     - Apply 4-bit NF4 quantization via BitsAndBytes
     - Q = min(m_after / m_before, 1)

  3. Aggregated Robustness:
     - Robustness = HM(R, Q)

Metric direction convention:
  - All Open-Unlearning metrics: raw values (higher = more forget info)
  - UDS only: use (1 - UDS) because higher UDS = less forget info

Filtering (applied at aggregation time, not during measurement):
  - Utility filter: exclude models with utility_rel < 0.8
  - Faithfulness threshold filter: per-metric threshold from P/N pool
  - NO lr filter

Usage:
  GPU 0: python scripts/meta_eval_robustness.py --mode quant --gpu 0
  GPU 1: python scripts/meta_eval_robustness.py --mode relearn --gpu 1

Reference: Open-Unlearning (NeurIPS 2025), Section 4.2
"""

import os
import sys
import json
import argparse
import shutil
import time
import gc
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchscope.models import load_model, load_tokenizer
from patchscope.utils import set_seed, safe_mkdir
from patchscope.meta_eval_utils import (
    MEM_METRICS,
    GENERATION_METRICS,
    MIA_METRICS,
    load_forget10_perturbed,
    compute_mem_metrics,
    compute_generation_metrics,
    prepare_mia_data,
    compute_mia_metrics,
)


# Pre-computed metrics cache (loaded once)
_PRECOMPUTED_METRICS: Optional[Dict[str, Dict[str, float]]] = None
_PRECOMPUTED_PATH = Path("runs/method_eval/metrics_before.json")


def load_precomputed_metrics(model_name: str) -> Optional[Dict[str, float]]:
    """Load pre-computed metrics from runs/method_eval/metrics_before.json.

    Returns dict with all 13 metrics, or None if not found.
    """
    global _PRECOMPUTED_METRICS

    # Load cache on first call
    if _PRECOMPUTED_METRICS is None:
        if _PRECOMPUTED_PATH.exists():
            with open(_PRECOMPUTED_PATH) as f:
                _PRECOMPUTED_METRICS = json.load(f)
            print(f"  [Cache] Loaded {len(_PRECOMPUTED_METRICS)} models from {_PRECOMPUTED_PATH}")
        else:
            _PRECOMPUTED_METRICS = {}
            print(f"  [Cache] {_PRECOMPUTED_PATH} not found")

    return _PRECOMPUTED_METRICS.get(model_name)

# Import UDS computation utilities
from exp_s1_teacher_forcing import load_prefix_data
from scripts.meta_eval_faithfulness import (
    prepare_all_examples,
    compute_s1_cache,
    compute_uds_for_model,
    TOFU_FULL_MODEL,
    TOFU_RETAIN_MODEL,
    PREFIX_DATA_PATH,
)


# ============================================================================
# Constants
# ============================================================================

ALL_METRICS = [
    "em", "es", "prob", "paraprob", "truth_ratio",
    "rouge", "para_rouge", "jailbreak_rouge",
    "mia_loss", "mia_zlib", "mia_min_k", "mia_min_kpp",
    "uds",
]

# 151 models: 1 retain + 150 unlearn (75 ep5 + 75 ep10)
DEFAULT_MODELS = {
    "retain": TOFU_RETAIN_MODEL,
    # AltPO ep5 (9)
    "altpo_lr1e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha1_epoch5",
    "altpo_lr1e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha2_epoch5",
    "altpo_lr1e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha5_epoch5",
    "altpo_lr2e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha1_epoch5",
    "altpo_lr2e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha2_epoch5",
    "altpo_lr2e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha5_epoch5",
    "altpo_lr5e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch5",
    "altpo_lr5e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha2_epoch5",
    "altpo_lr5e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha5_epoch5",
    # AltPO ep10 (9)
    "altpo_lr1e5_b01_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha1_epoch10",
    "altpo_lr1e5_b01_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha2_epoch10",
    "altpo_lr1e5_b01_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha5_epoch10",
    "altpo_lr2e5_b01_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha1_epoch10",
    "altpo_lr2e5_b01_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha2_epoch10",
    "altpo_lr2e5_b01_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha5_epoch10",
    "altpo_lr5e5_b01_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10",
    "altpo_lr5e5_b01_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha2_epoch10",
    "altpo_lr5e5_b01_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha5_epoch10",
    # GradDiff ep5 (9)
    "graddiff_lr1e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch5",
    "graddiff_lr1e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha2_epoch5",
    "graddiff_lr1e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5",
    "graddiff_lr2e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha1_epoch5",
    "graddiff_lr2e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha2_epoch5",
    "graddiff_lr2e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha5_epoch5",
    "graddiff_lr5e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha1_epoch5",
    "graddiff_lr5e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha2_epoch5",
    "graddiff_lr5e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha5_epoch5",
    # GradDiff ep10 (9)
    "graddiff_lr1e5_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10",
    "graddiff_lr1e5_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha2_epoch10",
    "graddiff_lr1e5_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10",
    "graddiff_lr2e5_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha1_epoch10",
    "graddiff_lr2e5_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha2_epoch10",
    "graddiff_lr2e5_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha5_epoch10",
    "graddiff_lr5e5_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha1_epoch10",
    "graddiff_lr5e5_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha2_epoch10",
    "graddiff_lr5e5_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha5_epoch10",
    # IdkDPO ep5 (9)
    "idkdpo_lr1e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha1_epoch5",
    "idkdpo_lr1e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha2_epoch5",
    "idkdpo_lr1e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha5_epoch5",
    "idkdpo_lr2e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha1_epoch5",
    "idkdpo_lr2e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha2_epoch5",
    "idkdpo_lr2e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha5_epoch5",
    "idkdpo_lr5e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha1_epoch5",
    "idkdpo_lr5e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha2_epoch5",
    "idkdpo_lr5e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha5_epoch5",
    # IdkDPO ep10 (9)
    "idkdpo_lr1e5_b01_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha1_epoch10",
    "idkdpo_lr1e5_b01_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha2_epoch10",
    "idkdpo_lr1e5_b01_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha5_epoch10",
    "idkdpo_lr2e5_b01_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha1_epoch10",
    "idkdpo_lr2e5_b01_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha2_epoch10",
    "idkdpo_lr2e5_b01_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha5_epoch10",
    "idkdpo_lr5e5_b01_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha1_epoch10",
    "idkdpo_lr5e5_b01_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha2_epoch10",
    "idkdpo_lr5e5_b01_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha5_epoch10",
    # IdkNLL ep5 (9)
    "idknll_lr1e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha1_epoch5",
    "idknll_lr1e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha2_epoch5",
    "idknll_lr1e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha5_epoch5",
    "idknll_lr2e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha1_epoch5",
    "idknll_lr2e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha2_epoch5",
    "idknll_lr2e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha5_epoch5",
    "idknll_lr5e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha1_epoch5",
    "idknll_lr5e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha2_epoch5",
    "idknll_lr5e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha5_epoch5",
    # IdkNLL ep10 (9)
    "idknll_lr1e5_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha1_epoch10",
    "idknll_lr1e5_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha2_epoch10",
    "idknll_lr1e5_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha5_epoch10",
    "idknll_lr2e5_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha1_epoch10",
    "idknll_lr2e5_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha2_epoch10",
    "idknll_lr2e5_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha5_epoch10",
    "idknll_lr5e5_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha1_epoch10",
    "idknll_lr5e5_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha2_epoch10",
    "idknll_lr5e5_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha5_epoch10",
    # NPO ep5 (9)
    "npo_lr1e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch5",
    "npo_lr1e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha2_epoch5",
    "npo_lr1e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha5_epoch5",
    "npo_lr2e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.1_alpha1_epoch5",
    "npo_lr2e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.1_alpha2_epoch5",
    "npo_lr2e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.1_alpha5_epoch5",
    "npo_lr5e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha1_epoch5",
    "npo_lr5e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha2_epoch5",
    "npo_lr5e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha5_epoch5",
    # NPO ep10 (9)
    "npo_lr1e5_b01_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10",
    "npo_lr1e5_b01_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha2_epoch10",
    "npo_lr1e5_b01_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha5_epoch10",
    "npo_lr2e5_b01_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.1_alpha1_epoch10",
    "npo_lr2e5_b01_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.1_alpha2_epoch10",
    "npo_lr2e5_b01_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.1_alpha5_epoch10",
    "npo_lr5e5_b01_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha1_epoch10",
    "npo_lr5e5_b01_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha2_epoch10",
    "npo_lr5e5_b01_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha5_epoch10",
    # RMU ep5 (9)
    "rmu_lr1e5_l5_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff10_epoch5",
    "rmu_lr1e5_l10_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer10_scoeff10_epoch5",
    "rmu_lr1e5_l15_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer15_scoeff10_epoch5",
    "rmu_lr2e5_l5_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer5_scoeff10_epoch5",
    "rmu_lr2e5_l10_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer10_scoeff10_epoch5",
    "rmu_lr2e5_l15_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer15_scoeff10_epoch5",
    "rmu_lr5e5_l5_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch5",
    "rmu_lr5e5_l10_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer10_scoeff10_epoch5",
    "rmu_lr5e5_l15_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer15_scoeff10_epoch5",
    # RMU ep10 (9)
    "rmu_lr1e5_l5_s10_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff10_epoch10",
    "rmu_lr1e5_l10_s10_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer10_scoeff10_epoch10",
    "rmu_lr1e5_l15_s10_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer15_scoeff10_epoch10",
    "rmu_lr2e5_l5_s10_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer5_scoeff10_epoch10",
    "rmu_lr2e5_l10_s10_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer10_scoeff10_epoch10",
    "rmu_lr2e5_l15_s10_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer15_scoeff10_epoch10",
    "rmu_lr5e5_l5_s10_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch10",
    "rmu_lr5e5_l10_s10_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer10_scoeff10_epoch10",
    "rmu_lr5e5_l15_s10_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer15_scoeff10_epoch10",
    # SimNPO ep5 (12)
    "simnpo_lr1e5_b35_a1_d1_g0125_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b3.5_a1_d1_g0.125_ep5",
    "simnpo_lr1e5_b35_a1_d1_g025_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b3.5_a1_d1_g0.25_ep5",
    "simnpo_lr1e5_b45_a1_d1_g0125_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b4.5_a1_d1_g0.125_ep5",
    "simnpo_lr1e5_b45_a1_d1_g025_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b4.5_a1_d1_g0.25_ep5",
    "simnpo_lr2e5_b35_a1_d1_g0125_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b3.5_a1_d1_g0.125_ep5",
    "simnpo_lr2e5_b35_a1_d1_g025_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b3.5_a1_d1_g0.25_ep5",
    "simnpo_lr2e5_b45_a1_d1_g0125_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.125_ep5",
    "simnpo_lr2e5_b45_a1_d1_g025_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.25_ep5",
    "simnpo_lr5e5_b35_a1_d1_g0125_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.125_ep5",
    "simnpo_lr5e5_b35_a1_d1_g025_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5",
    "simnpo_lr5e5_b45_a1_d1_g0125_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.125_ep5",
    "simnpo_lr5e5_b45_a1_d1_g025_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.25_ep5",
    # SimNPO ep10 (12)
    "simnpo_lr1e5_b35_a1_d1_g0125_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b3.5_a1_d1_g0.125_ep10",
    "simnpo_lr1e5_b35_a1_d1_g025_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b3.5_a1_d1_g0.25_ep10",
    "simnpo_lr1e5_b45_a1_d1_g0125_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b4.5_a1_d1_g0.125_ep10",
    "simnpo_lr1e5_b45_a1_d1_g025_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b4.5_a1_d1_g0.25_ep10",
    "simnpo_lr2e5_b35_a1_d1_g0125_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b3.5_a1_d1_g0.125_ep10",
    "simnpo_lr2e5_b35_a1_d1_g025_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b3.5_a1_d1_g0.25_ep10",
    "simnpo_lr2e5_b45_a1_d1_g0125_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.125_ep10",
    "simnpo_lr2e5_b45_a1_d1_g025_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.25_ep10",
    "simnpo_lr5e5_b35_a1_d1_g0125_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.125_ep10",
    "simnpo_lr5e5_b35_a1_d1_g025_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep10",
    "simnpo_lr5e5_b45_a1_d1_g0125_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.125_ep10",
    "simnpo_lr5e5_b45_a1_d1_g025_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.25_ep10",
    # UNDIAL ep5 (9)
    "undial_lr1e5_b10_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta10_alpha1_epoch5",
    "undial_lr1e5_b10_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta10_alpha2_epoch5",
    "undial_lr1e5_b10_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta10_alpha5_epoch5",
    "undial_lr1e4_b10_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha1_epoch5",
    "undial_lr1e4_b10_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha2_epoch5",
    "undial_lr1e4_b10_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha5_epoch5",
    "undial_lr3e4_b10_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0003_beta10_alpha1_epoch5",
    "undial_lr3e4_b10_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0003_beta10_alpha2_epoch5",
    "undial_lr3e4_b10_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0003_beta10_alpha5_epoch5",
    # UNDIAL ep10 (9)
    "undial_lr1e5_b10_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta10_alpha1_epoch10",
    "undial_lr1e5_b10_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta10_alpha2_epoch10",
    "undial_lr1e5_b10_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta10_alpha5_epoch10",
    "undial_lr1e4_b10_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha1_epoch10",
    "undial_lr1e4_b10_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha2_epoch10",
    "undial_lr1e4_b10_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha5_epoch10",
    "undial_lr3e4_b10_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0003_beta10_alpha1_epoch10",
    "undial_lr3e4_b10_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0003_beta10_alpha2_epoch10",
    "undial_lr3e4_b10_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0003_beta10_alpha5_epoch10",
}


# ============================================================================
# Cache management
# ============================================================================

def clear_hf_cache(model_id: str) -> None:
    """Clear HuggingFace cache for a specific model to save disk space."""
    from huggingface_hub import scan_cache_dir

    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                for revision in repo.revisions:
                    cache_info.delete_revisions(revision.commit_hash).execute()
                print(f"  [Cache] Cleared: {model_id}")
                return
    except Exception as e:
        print(f"  [Cache] Warning: Could not clear cache for {model_id}: {e}")


def free_memory() -> None:
    """Free GPU and CPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# Quantization
# ============================================================================

def load_model_quantized(model_id: str, device_map: str = "cuda"):
    """Load a model with 4-bit quantization (NF4 + bfloat16 compute)."""
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model.eval()
    return model


# ============================================================================
# Relearning
# ============================================================================

IGNORE_INDEX = -100


class ForgetDataset(torch.utils.data.Dataset):
    """SFT dataset for forget10 relearning with answer-only loss masking."""

    def __init__(self, tokenizer, max_length=512, system_prompt="You are a helpful assistant."):
        ds = load_dataset("locuslab/TOFU", "forget10", split="train")
        self.examples = []

        for item in ds:
            question = item["question"]
            answer = item["answer"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

            chat_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)

            if tokenizer.eos_token_id and (not chat_ids or chat_ids[-1] != tokenizer.eos_token_id):
                chat_ids = chat_ids + [tokenizer.eos_token_id]

            if len(chat_ids) > max_length:
                chat_ids = chat_ids[:max_length]
                prompt_ids = prompt_ids[:min(len(prompt_ids), max_length)]

            input_ids = torch.tensor(chat_ids, dtype=torch.long)
            attention_mask = torch.ones(len(chat_ids), dtype=torch.long)
            labels = torch.full((len(chat_ids),), IGNORE_INDEX, dtype=torch.long)
            labels[len(prompt_ids):] = input_ids[len(prompt_ids):]

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ForgetDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full((len(features), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        labels = torch.full((len(features), max_len), IGNORE_INDEX, dtype=torch.long)

        for i, f in enumerate(features):
            L = len(f["input_ids"])
            input_ids[i, :L] = f["input_ids"]
            attention_mask[i, :L] = f["attention_mask"]
            labels[i, :L] = f["labels"]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def finetune_on_forget10(model, tokenizer, output_dir: str, lr: float = 2e-5, epochs: int = 1,
                          batch_size: int = 8, grad_accum: int = 4,
                          system_prompt: str = "You are a helpful assistant."):
    """Finetune model on forget10 for relearning attack.

    Default settings:
    - per_device_batch_size=8, grad_accum=4 (effective=32)
    - lr=2e-5 (from paper), optim=paged_adamw_32bit
    """
    from transformers import Trainer, TrainingArguments

    dataset = ForgetDataset(tokenizer, max_length=512, system_prompt=system_prompt)
    collator = ForgetDataCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=0,
        weight_decay=0.0,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        dataloader_pin_memory=False,
        report_to="none",
    )

    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    model.eval()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Aggressively free training state to prevent GPU/CPU memory leak
    model.zero_grad(set_to_none=True)
    del trainer, training_args, dataset, collator
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


def finetune_in_subprocess(
    model_id: str,
    output_dir: str,
    gpu: int,
    lr: float = 2e-5,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 4,
    system_prompt: str = "You are a helpful assistant.",
    attn_implementation: str = "sdpa",
) -> bool:
    """Run finetuning in a separate process to avoid GPU memory leaks.

    Returns True if finetuning succeeded (output_dir contains model files).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = f'''
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
sys.path.insert(0, "{repo_root}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "{model_id}", torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="{attn_implementation}",
)

from scripts.meta_eval_robustness import finetune_on_forget10
finetune_on_forget10(
    model, tokenizer, "{output_dir}",
    lr={lr}, epochs={epochs}, batch_size={batch_size},
    grad_accum={grad_accum}, system_prompt="{system_prompt}",
)
print("FINETUNE_SUCCESS")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=900,
        cwd=repo_root,
    )
    success = "FINETUNE_SUCCESS" in result.stdout
    if not success:
        print(f"  Subprocess finetune FAILED:")
        print(f"  stdout: {result.stdout[-500:]}")
        print(f"  stderr: {result.stderr[-500:]}")
    return success


# ============================================================================
# Metric computation
# ============================================================================

def compute_all_metrics(
    model,
    tokenizer,
    full_model,
    prepared_uds,
    s1_cache,
    layer_list,
    mem_data,
    mia_data,
    args,
    log_path=None,
    prefix_data=None,
    source_name="source",
) -> Dict[str, float]:
    """Compute all 13 metrics for a model."""
    scores = {}

    # UDS
    if "uds" in args.metrics:
        avg_uds, uds_list = compute_uds_for_model(
            model, full_model, tokenizer, prepared_uds,
            s1_cache, layer_list, args.delta_threshold, args.patch_scope,
            args.eval_batch_size,
            log_path=log_path, prefix_data=prefix_data, source_name=source_name,
        )
        scores["uds"] = avg_uds

    # Memorization metrics
    mem_metrics_needed = MEM_METRICS & set(args.metrics)
    if mem_metrics_needed:
        mem_scores = compute_mem_metrics(
            model, tokenizer, mem_data,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            use_chat_template=True,
            system_prompt=args.system_prompt,
        )
        for k, v in mem_scores.items():
            if k in args.metrics:
                scores[k] = v

    # Generation metrics
    gen_metrics_needed = GENERATION_METRICS & set(args.metrics)
    if gen_metrics_needed:
        gen_scores = compute_generation_metrics(
            model, tokenizer, mem_data,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=True,
            system_prompt=args.system_prompt,
            metrics_to_compute=gen_metrics_needed,
        )
        for k, v in gen_scores.items():
            if k in args.metrics:
                scores[k] = v

    # MIA metrics
    mia_metrics_needed = MIA_METRICS & set(args.metrics)
    if mia_metrics_needed:
        mia_scores = compute_mia_metrics(
            model, tokenizer, mia_data,
            batch_size=args.eval_batch_size,
            k=args.mia_k,
        )
        for k, v in mia_scores.items():
            if k in args.metrics:
                scores[k] = v

    return scores


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Robustness meta-evaluation")
    parser.add_argument("--mode", type=str, required=True, choices=["quant", "relearn"],
                        help="quant: quantization test, relearn: relearning test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device train batch size (default: 8, matching Open-Unlearning)")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Eval batch size (default: 64)")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4, effective batch=32)")
    parser.add_argument("--delta_threshold", type=float, default=0.05)
    parser.add_argument("--patch_scope", type=str, default="span")
    parser.add_argument("--em_scope", type=str, default="entity")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--mia_k", type=float, default=0.4)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--s1_cache_path", type=str, default="runs/meta_eval/s1_cache_sdpa.json")
    parser.add_argument("--relearn_lr", type=float, default=2e-5,
                        help="Relearning LR (default: 2e-5, from paper)")
    parser.add_argument("--relearn_epochs", type=int, default=1)
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        help="Attention implementation: eager, sdpa, flash_attention_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--clear_cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Clear HF cache after each model to save disk space")
    parser.add_argument("--metrics", type=str, default="all",
                        help="Comma-separated metrics or 'all'")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for model list (for parallel runs)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for model list (for parallel runs)")
    args = parser.parse_args()

    # Parse metrics
    if args.metrics == "all":
        args.metrics = ALL_METRICS
    else:
        args.metrics = [m.strip() for m in args.metrics.split(",")]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    if args.out_dir is None:
        args.out_dir = f"runs/meta_eval/robustness/{args.mode}"
    safe_mkdir(args.out_dir)

    print("=" * 70)
    print(f"Robustness Meta-Evaluation: {args.mode.upper()}")
    print("=" * 70)
    print(f"GPU: {args.gpu}")
    print(f"Models: {len(DEFAULT_MODELS)} (1 retain + 150 unlearn)")
    print(f"Metrics: {', '.join(args.metrics)}")
    print(f"Output: {args.out_dir}")
    print(f"Clear cache: {args.clear_cache}")
    print()

    if args.dry_run:
        print("Dry run - models:")
        for name, mid in list(DEFAULT_MODELS.items())[:5]:
            print(f"  {name}: {mid}")
        print(f"  ... and {len(DEFAULT_MODELS) - 5} more")
        return

    # Load tokenizer and full model (needed for UDS)
    print("Loading tokenizer and full model...")
    tokenizer = load_tokenizer(TOFU_FULL_MODEL)
    full_model = load_model(TOFU_FULL_MODEL, dtype="bfloat16", device_map="cuda",
                            attn_implementation=args.attn_implementation)

    # Prepare UDS data
    layer_list = list(range(16))
    prefix_data = load_prefix_data(PREFIX_DATA_PATH)
    prepared_uds = prepare_all_examples(
        tokenizer, prefix_data,
        patch_scope=args.patch_scope,
        em_scope=args.em_scope,
    )
    print(f"UDS data: {sum(1 for p in prepared_uds if p)} valid examples")

    # Load S1 cache
    s1_cache_path = Path(args.s1_cache_path)
    if s1_cache_path.exists():
        print(f"Loading S1 cache from {s1_cache_path}...")
        s1_cache = json.loads(s1_cache_path.read_text())
        s1_cache = {int(k): v for k, v in s1_cache.items()}
    else:
        print("S1 cache not found - will need retain model to compute")
        retain_model = load_model(TOFU_RETAIN_MODEL, dtype="bfloat16", device_map="cuda",
                                   attn_implementation=args.attn_implementation)
        s1_cache = compute_s1_cache(
            full_model, retain_model, tokenizer, prepared_uds,
            layer_list, args.delta_threshold, args.patch_scope, args.eval_batch_size
        )
        s1_cache_path.parent.mkdir(parents=True, exist_ok=True)
        s1_cache_path.write_text(json.dumps({str(k): v for k, v in s1_cache.items()}))
        del retain_model
        free_memory()

    # Load evaluation data
    mem_data = load_forget10_perturbed()
    mia_data = prepare_mia_data(
        tokenizer,
        max_length=args.max_length,
        use_chat_template=True,
        system_prompt=args.system_prompt,
    )

    # Resume support
    results_path = Path(args.out_dir) / "results.json"
    results = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            results = json.loads(resume_path.read_text())
            print(f"Resumed: {len(results)} entries")
    elif results_path.exists():
        results = json.loads(results_path.read_text())
        print(f"Resumed from {results_path}: {len(results)} entries")

    # Process each model
    model_items = list(DEFAULT_MODELS.items())

    # Apply index filtering for parallel runs
    end_idx = args.end_idx if args.end_idx is not None else len(model_items)
    model_items = model_items[args.start_idx:end_idx]
    print(f"Processing models [{args.start_idx}:{end_idx}] ({len(model_items)} models)")

    for mi, (name, model_id) in enumerate(model_items):
        print(f"\n{'='*60}")
        print(f"[{mi+1}/{len(model_items)}] {name}")
        print(f"  {model_id}")
        print(f"{'='*60}")

        # Check if already done
        result_key = f"metrics_after_{args.mode}"
        if name in results and result_key in results[name]:
            print(f"  Already computed, skipping")
            continue

        if name not in results:
            results[name] = {"model_id": model_id}

        t0 = time.time()

        try:
            # Handle metrics_before (try pre-computed first)
            if "metrics_before" not in results[name]:
                precomputed = load_precomputed_metrics(name)
                if precomputed is not None:
                    print("  Using pre-computed metrics (before)")
                    results[name]["metrics_before"] = precomputed
                    results_path.write_text(json.dumps(results, indent=2))

            # For quant mode: only load quantized model if we have metrics_before
            if args.mode == "quant":
                # If still no metrics_before, need to compute it
                if "metrics_before" not in results[name]:
                    print("  Loading model for metrics_before...")
                    model = load_model(model_id, dtype="bfloat16", device_map="cuda",
                                       attn_implementation=args.attn_implementation)
                    print("  Computing metrics (before)...")
                    scores_before = compute_all_metrics(
                        model, tokenizer, full_model, prepared_uds, s1_cache,
                        layer_list, mem_data, mia_data, args
                    )
                    results[name]["metrics_before"] = scores_before
                    results_path.write_text(json.dumps(results, indent=2))
                    del model
                    free_memory()

                # Load quantized model and compute metrics
                print("  Loading quantized model...")
                quant_model = load_model_quantized(model_id, device_map="cuda")

                print("  Computing metrics (after quant)...")
                uds_log = os.path.join(args.out_dir, "uds_logs", f"{name}.log")
                scores = compute_all_metrics(
                    quant_model, tokenizer, full_model, prepared_uds, s1_cache,
                    layer_list, mem_data, mia_data, args,
                    log_path=uds_log, prefix_data=prefix_data, source_name=f"{name}_quant",
                )
                results[name]["metrics_after_quant"] = scores

                del quant_model
                free_memory()

            else:  # relearn
                # Compute metrics_before if not available
                if "metrics_before" not in results[name]:
                    print("  Loading model for metrics_before...")
                    model = load_model(model_id, dtype="bfloat16", device_map="cuda",
                                       attn_implementation=args.attn_implementation)
                    print("  Computing metrics (before)...")
                    scores_before = compute_all_metrics(
                        model, tokenizer, full_model, prepared_uds, s1_cache,
                        layer_list, mem_data, mia_data, args
                    )
                    results[name]["metrics_before"] = scores_before
                    results_path.write_text(json.dumps(results, indent=2))
                    del model
                    free_memory()

                # Finetune on forget10 in subprocess (prevents GPU memory leak)
                print("  Finetuning on forget10 (subprocess)...")
                ft_dir = Path(args.out_dir) / f"ft_{name}"
                ft_ok = finetune_in_subprocess(
                    model_id, str(ft_dir), gpu=args.gpu,
                    lr=args.relearn_lr, epochs=args.relearn_epochs,
                    batch_size=args.batch_size, grad_accum=args.grad_accum,
                    system_prompt=args.system_prompt,
                    attn_implementation=args.attn_implementation,
                )
                if not ft_ok:
                    raise RuntimeError("Subprocess finetuning failed")

                # Load finetuned model and compute metrics
                print("  Loading finetuned model...")
                ft_model = load_model(str(ft_dir), dtype="bfloat16", device_map="cuda",
                                      attn_implementation=args.attn_implementation)

                print("  Computing metrics (after relearn)...")
                uds_log = os.path.join(args.out_dir, "uds_logs", f"{name}.log")
                scores_after = compute_all_metrics(
                    ft_model, tokenizer, full_model, prepared_uds, s1_cache,
                    layer_list, mem_data, mia_data, args,
                    log_path=uds_log, prefix_data=prefix_data, source_name=f"{name}_relearn",
                )
                results[name]["metrics_after_relearn"] = scores_after

                del ft_model
                free_memory()

                # Clean up finetuned model
                shutil.rmtree(ft_dir, ignore_errors=True)

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

            # Print some metrics
            result_scores = results[name].get(result_key, {})
            for m in ["uds", "em", "mia_loss"][:3]:
                if m in result_scores:
                    print(f"    {m}: {result_scores[m]:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name]["error"] = str(e)

        # Save progress
        results_path.write_text(json.dumps(results, indent=2))

        # Clear HF cache
        if args.clear_cache and model_id != TOFU_FULL_MODEL and model_id != TOFU_RETAIN_MODEL:
            clear_hf_cache(model_id)

        free_memory()

    print(f"\n{'='*60}")
    print(f"Done! Results saved to {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

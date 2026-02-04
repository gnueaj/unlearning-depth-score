#!/usr/bin/env python3
"""
Meta-Evaluation: Robustness of UDS metric.

Evaluates two dimensions:
  1. Robustness to Relearning (Eq. 2):
     - Finetune unlearn/retain models on forget10 for 1 epoch (lr=2e-5)
     - R = min((m_ret^a - m_ret^b) / (m_unl^a - m_unl^b), 1)

  2. Robustness to Quantization (Eq. 3):
     - Apply 4-bit NF4 quantization via BitsAndBytes
     - Q = min(m_unl^b / m_unl^a, 1)

  3. Aggregated (Eq. 4):
     - Robustness = HM(R, Q)

Metric convention:
  - m = 1 - UDS (higher = more knowledge detected)
  - UDS higher = more erasure = less knowledge

Design constraints:
  - Disk: ~80GB → save finetuned models to temp dir, cleanup after
  - GPU: 48GB → sufficient for bf16 model + finetuning
  - Reuse S1 cache from faithfulness eval

Reference: Open-Unlearning (NeurIPS 2025), Section 4.1, Eq. 2-4
"""

import os
import sys
import json
import argparse
import shutil
import time
import gc
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchscope.models import load_model, load_tokenizer, get_num_layers
from patchscope.utils import set_seed, safe_mkdir
from patchscope.meta_eval_utils import (
    MEM_METRICS,
    GENERATION_METRICS,
    MIA_METRICS,
    normalize_metrics_list,
    load_forget10_perturbed,
    compute_mem_metrics,
    compute_generation_metrics,
    prepare_mia_data,
    compute_mia_metrics,
)

from exp_s1_teacher_forcing import (
    load_prefix_data,
    get_eval_span,
    normalize_reference_for_eval,
    build_logprob_ctx,
    _prepare_batch_inputs,
    _compute_hidden_states_batch,
    _gather_token_logprobs,
    compute_logprob_teacher_forcing_baseline_batch_with_inputs,
    compute_logprob_teacher_forcing_layer_batch_with_inputs,
)

# Reuse prepare_all_examples and compute_uds_for_model from faithfulness script
from scripts.meta_eval_faithfulness import (
    prepare_all_examples,
    compute_s1_cache,
    compute_uds_for_model,
    TOFU_FULL_MODEL,
    TOFU_RETAIN_MODEL,
    PREFIX_DATA_PATH,
)


# ============================================================================
# Default unlearn models to evaluate robustness on
# ============================================================================
# All 75 unlearned models from the alpha experiment set (ep5, 8 methods)
# Matches the model set in docs/0202/openunlearning_alpha_all.html
DEFAULT_UNLEARN_MODELS = {
    # AltPO: 9 models (3 LR × 3 alpha)
    "altpo_lr1e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha1_epoch5",
    "altpo_lr1e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha2_epoch5",
    "altpo_lr1e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr1e-05_beta0.1_alpha5_epoch5",
    "altpo_lr2e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha1_epoch5",
    "altpo_lr2e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha2_epoch5",
    "altpo_lr2e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.1_alpha5_epoch5",
    "altpo_lr5e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch5",
    "altpo_lr5e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha2_epoch5",
    "altpo_lr5e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha5_epoch5",
    # GradDiff: 9 models (3 LR × 3 alpha)
    "graddiff_lr1e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch5",
    "graddiff_lr1e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha2_epoch5",
    "graddiff_lr1e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5",
    "graddiff_lr2e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha1_epoch5",
    "graddiff_lr2e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha2_epoch5",
    "graddiff_lr2e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha5_epoch5",
    "graddiff_lr5e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha1_epoch5",
    "graddiff_lr5e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha2_epoch5",
    "graddiff_lr5e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha5_epoch5",
    # IdkDPO: 9 models (3 LR × 3 alpha)
    "idkdpo_lr1e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha1_epoch5",
    "idkdpo_lr1e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha2_epoch5",
    "idkdpo_lr1e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha5_epoch5",
    "idkdpo_lr2e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha1_epoch5",
    "idkdpo_lr2e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha2_epoch5",
    "idkdpo_lr2e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha5_epoch5",
    "idkdpo_lr5e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha1_epoch5",
    "idkdpo_lr5e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha2_epoch5",
    "idkdpo_lr5e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha5_epoch5",
    # IdkNLL: 9 models (3 LR × 3 alpha)
    "idknll_lr1e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha1_epoch5",
    "idknll_lr1e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha2_epoch5",
    "idknll_lr1e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha5_epoch5",
    "idknll_lr2e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha1_epoch5",
    "idknll_lr2e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha2_epoch5",
    "idknll_lr2e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha5_epoch5",
    "idknll_lr5e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha1_epoch5",
    "idknll_lr5e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha2_epoch5",
    "idknll_lr5e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha5_epoch5",
    # NPO: 9 models (3 LR × 3 alpha)
    "npo_lr1e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch5",
    "npo_lr1e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha2_epoch5",
    "npo_lr1e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha5_epoch5",
    "npo_lr2e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.1_alpha1_epoch5",
    "npo_lr2e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.1_alpha2_epoch5",
    "npo_lr2e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.1_alpha5_epoch5",
    "npo_lr5e5_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha1_epoch5",
    "npo_lr5e5_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha2_epoch5",
    "npo_lr5e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha5_epoch5",
    # RMU: 9 models (3 LR × 3 layer)
    "rmu_lr1e5_l5_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff10_epoch5",
    "rmu_lr1e5_l10_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer10_scoeff10_epoch5",
    "rmu_lr1e5_l15_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer15_scoeff10_epoch5",
    "rmu_lr2e5_l5_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer5_scoeff10_epoch5",
    "rmu_lr2e5_l10_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer10_scoeff10_epoch5",
    "rmu_lr2e5_l15_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer15_scoeff10_epoch5",
    "rmu_lr5e5_l5_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch5",
    "rmu_lr5e5_l10_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer10_scoeff10_epoch5",
    "rmu_lr5e5_l15_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer15_scoeff10_epoch5",
    # SimNPO: 12 models (3 LR × 2 beta × 2 gamma)
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
    # UNDIAL: 9 models (3 LR × 3 alpha)
    "undial_lr1e5_b10_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta10_alpha1_epoch5",
    "undial_lr1e5_b10_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta10_alpha2_epoch5",
    "undial_lr1e5_b10_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta10_alpha5_epoch5",
    "undial_lr1e4_b10_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha1_epoch5",
    "undial_lr1e4_b10_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha2_epoch5",
    "undial_lr1e4_b10_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha5_epoch5",
    "undial_lr3e4_b10_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0003_beta10_alpha1_epoch5",
    "undial_lr3e4_b10_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0003_beta10_alpha2_epoch5",
    "undial_lr3e4_b10_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0003_beta10_alpha5_epoch5",
}


# ============================================================================
# Relearning: Finetune on forget10 for 1 epoch
# ============================================================================

IGNORE_INDEX = -100


class ForgetDataset(torch.utils.data.Dataset):
    """SFT dataset for forget10 relearning with answer-only loss masking."""

    def __init__(self, tokenizer, max_length=512):
        ds = load_dataset("locuslab/TOFU", "forget10", split="train")
        self.examples = []
        for item in ds:
            question = item["question"]
            answer = item["answer"]

            # Use chat template (matching Instruct model training format)
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            chat_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
            )
            prompt_ids = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True,
            )

            # Add EOS if missing
            if tokenizer.eos_token_id is not None:
                if len(chat_ids) == 0 or chat_ids[-1] != tokenizer.eos_token_id:
                    chat_ids = chat_ids + [tokenizer.eos_token_id]

            # Truncate if needed
            if len(chat_ids) > max_length:
                chat_ids = chat_ids[:max_length]
                prompt_ids = prompt_ids[:min(len(prompt_ids), max_length)]

            input_ids = torch.tensor(chat_ids, dtype=torch.long)
            attention_mask = torch.ones(len(chat_ids), dtype=torch.long)

            # Labels: IGNORE prompt tokens, loss only on answer tokens
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
    """Pad variable-length examples for Trainer."""

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


def finetune_on_forget10(model, tokenizer, output_dir, lr=2e-5, epochs=1,
                          batch_size=4, grad_accum=4):
    """
    Finetune a model on forget10 for relearning stress test.

    Paper protocol: 1 epoch, lr=2e-5 on the forget set.
    Uses chat template with answer-only loss masking.
    Returns the path to saved model.
    """
    from transformers import Trainer, TrainingArguments

    dataset = ForgetDataset(tokenizer, max_length=512)
    collator = ForgetDataCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=0,
        weight_decay=0.0,
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

    return output_dir


# ============================================================================
# Quantization: Load model in 4-bit NF4
# ============================================================================

def load_model_quantized(model_id, device_map="cuda"):
    """Load a model with 4-bit NF4 quantization via BitsAndBytes."""
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = __import__("transformers").AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model.eval()
    return model


# ============================================================================
# UDS computation helpers
# ============================================================================

def compute_uds_single_model(source_model, full_model, tokenizer, prepared,
                              s1_cache, layer_list, delta_threshold, patch_scope,
                              batch_size):
    """
    Compute average UDS for a single source model.
    Returns (avg_uds, n_examples).
    """
    avg_uds, uds_list = compute_uds_for_model(
        source_model, full_model, tokenizer, prepared,
        s1_cache, layer_list, delta_threshold, patch_scope,
        batch_size
    )
    return avg_uds, len(uds_list)


def _metric_to_knowledge(metric: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return 1 - value if metric == "uds" else value


def compute_metric_scores_for_model(
    model,
    tokenizer,
    metrics,
    full_model,
    prepared,
    s1_cache,
    layer_list,
    args,
    mem_data=None,
    mia_data=None,
):
    scores: Dict[str, Optional[float]] = {}
    if "uds" in metrics:
        uds, _ = compute_uds_single_model(
            model, full_model, tokenizer, prepared,
            s1_cache, layer_list, args.delta_threshold, args.patch_scope,
            args.batch_size
        )
        scores["uds"] = uds

    if any(m in MEM_METRICS for m in metrics):
        mem_scores = compute_mem_metrics(
            model, tokenizer, mem_data,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt or None,
            date_string=args.date_string or None,
        )
        for k, v in mem_scores.items():
            if k in metrics:
                scores[k] = v

    gen_metrics_needed = GENERATION_METRICS & set(metrics)
    if gen_metrics_needed:
        gen_scores = compute_generation_metrics(
            model, tokenizer, mem_data,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt or None,
            date_string=args.date_string or None,
            metrics_to_compute=gen_metrics_needed,
        )
        for k, v in gen_scores.items():
            if k in metrics:
                scores[k] = v

    if any(m in MIA_METRICS for m in metrics):
        mia_scores = compute_mia_metrics(
            model, tokenizer, mia_data,
            batch_size=args.batch_size,
            k=args.mia_k,
        )
        for k, v in mia_scores.items():
            if k in metrics:
                scores[k] = v

    return scores


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Meta-eval: UDS Robustness (Relearning + Quantization)"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for UDS computation")
    parser.add_argument("--delta_threshold", type=float, default=0.05)
    parser.add_argument("--patch_scope", type=str, default="span")
    parser.add_argument("--em_scope", type=str, default="entity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--s1_cache_path", type=str, default="runs/meta_eval/s1_cache.json")
    parser.add_argument("--relearn_lr", type=float, default=2e-5,
                        help="Learning rate for relearning (paper default: 2e-5)")
    parser.add_argument("--relearn_epochs", type=int, default=1,
                        help="Epochs for relearning (paper default: 1)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific model short names to evaluate (from UNLEARN_MODELS)")
    parser.add_argument("--skip_relearning", action="store_true",
                        help="Skip relearning evaluation")
    parser.add_argument("--skip_quantization", action="store_true",
                        help="Skip quantization evaluation")
    parser.add_argument("--denom_zero_mode", type=str, default="none",
                        choices=["none", "one"],
                        help="How to handle denom≈0 in relearning R: "
                             "'none' -> R=None (exclude), 'one' -> R=1.0")
    parser.add_argument("--metrics", type=str, default="uds",
                        help="Comma-separated metrics: uds, em, es, prob, paraprob, truth_ratio, "
                             "rouge, para_rouge, jailbreak_rouge, mia_loss, mia_zlib, mia_min_k, "
                             "mia_min_kpp. Use 'all' for all 13 or 'table2' for 12 (no UDS).")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--date_string", type=str, default="10 Apr 2025")
    parser.add_argument("--mia_k", type=float, default=0.4)
    parser.add_argument("--faithfulness_result", type=str, default=None,
                        help="Path to faithfulness summary.json for computing overall score")
    parser.add_argument("--keep_finetuned", action="store_true",
                        help="Keep finetuned model checkpoints (uses ~2.4GB each)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to partial results JSON to resume from")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    if args.out_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M%S")
        args.out_dir = f"runs/meta_eval/{ts}_robustness"
    safe_mkdir(args.out_dir)

    layer_list = list(range(16))
    metrics = normalize_metrics_list(args.metrics.split(","))
    if not metrics:
        metrics = ["uds"]

    # Select models to evaluate
    if args.models:
        from patchscope.config import get_model_id
        eval_models = {}
        for name in args.models:
            model_id = get_model_id(name)
            eval_models[name] = model_id
    else:
        eval_models = DEFAULT_UNLEARN_MODELS.copy()

    print("=" * 70)
    print("Meta-Evaluation: UDS Robustness")
    print("=" * 70)
    print(f"Models to evaluate: {len(eval_models)}")
    print(f"Relearning: {'SKIP' if args.skip_relearning else f'lr={args.relearn_lr}, epochs={args.relearn_epochs}'}")
    print(f"Quantization: {'SKIP' if args.skip_quantization else '4-bit NF4'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Output: {args.out_dir}")
    print()

    if args.dry_run:
        print("Models:")
        for name, mid in eval_models.items():
            print(f"  {name}: {mid}")
        return

    # ------------------------------------------------------------------
    # Load base models
    # ------------------------------------------------------------------
    print("Loading tokenizer + full model...")
    tokenizer = load_tokenizer(TOFU_FULL_MODEL)
    full_model = load_model(TOFU_FULL_MODEL, dtype="bfloat16", device_map="cuda")

    # ------------------------------------------------------------------
    # Load & prepare dataset
    # ------------------------------------------------------------------
    prefix_data = load_prefix_data(PREFIX_DATA_PATH)
    prepared = prepare_all_examples(
        tokenizer, prefix_data,
        patch_scope=args.patch_scope,
        em_scope=args.em_scope,
    )
    n_valid = sum(1 for p in prepared if p is not None)
    print(f"Dataset: {len(prefix_data)} examples, {n_valid} valid")

    mem_data = None
    mia_data = None
    if any(m in MEM_METRICS or m in GENERATION_METRICS for m in metrics):
        mem_data = load_forget10_perturbed()
    if any(m in MIA_METRICS for m in metrics):
        mia_data = prepare_mia_data(
            tokenizer,
            max_length=args.max_length,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt or None,
            date_string=args.date_string or None,
        )

    # ------------------------------------------------------------------
    # S1 cache
    # ------------------------------------------------------------------
    s1_cache_path = Path(args.s1_cache_path)
    if s1_cache_path.exists():
        print(f"Loading S1 cache from {s1_cache_path}...")
        s1_cache = json.loads(s1_cache_path.read_text())
        s1_cache = {int(k): v for k, v in s1_cache.items()}
        print(f"  Loaded {len(s1_cache)} entries")
    else:
        print("Computing S1 cache (need retain model)...")
        retain_model = load_model(TOFU_RETAIN_MODEL, dtype="bfloat16", device_map="cuda")
        s1_cache = compute_s1_cache(
            full_model, retain_model, tokenizer, prepared,
            layer_list, args.delta_threshold, args.patch_scope, args.batch_size
        )
        s1_cache_path.parent.mkdir(parents=True, exist_ok=True)
        s1_cache_path.write_text(json.dumps({str(k): v for k, v in s1_cache.items()}))
        print(f"  Saved S1 cache: {len(s1_cache)} entries")
        del retain_model
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------
    results = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            results = json.loads(resume_path.read_text())
            print(f"Resuming: {len(results)} entries loaded")

    results_path = Path(args.out_dir) / "results.json"

    # ------------------------------------------------------------------
    # Retain model baseline (needed for relearning)
    # ------------------------------------------------------------------
    retain_before_scores = {}
    retain_after_scores = {}

    if not args.skip_relearning:
        cached_retain_before = results.get("retain_before", {}).get("metrics", {})
        cached_retain_after = results.get("retain_after", {}).get("metrics", {})
        metrics_missing_ret_before = [m for m in metrics if m not in cached_retain_before]
        metrics_missing_ret_after = [m for m in metrics if m not in cached_retain_after]

        need_retain_model = bool(metrics_missing_ret_before) or bool(metrics_missing_ret_after)

        if need_retain_model:
            retain_model = load_model(TOFU_RETAIN_MODEL, dtype="bfloat16", device_map="cuda")

            if metrics_missing_ret_before:
                print(f"\nComputing retain metrics (before)... [{len(metrics_missing_ret_before)} metrics]")
                scores_new = compute_metric_scores_for_model(
                    retain_model, tokenizer, metrics_missing_ret_before,
                    full_model, prepared, s1_cache, layer_list,
                    args, mem_data, mia_data
                )
                cached_retain_before.update(scores_new)
                results["retain_before"] = {"metrics": cached_retain_before}
                for k, v in scores_new.items():
                    if v is not None:
                        print(f"  Retain {k} (before) = {v:.4f}")

            if metrics_missing_ret_after:
                # Finetune retain model on forget10
                print("\nFinetuning retain model on forget10...")
                retain_ft_dir = os.path.join(args.out_dir, "retain_finetuned")
                finetune_on_forget10(
                    retain_model, tokenizer, retain_ft_dir,
                    lr=args.relearn_lr, epochs=args.relearn_epochs,
                )
                del retain_model
                gc.collect()
                torch.cuda.empty_cache()

                print(f"Computing retain metrics (after)... [{len(metrics_missing_ret_after)} metrics]")
                retain_ft_model = load_model(retain_ft_dir, dtype="bfloat16", device_map="cuda")
                scores_after_new = compute_metric_scores_for_model(
                    retain_ft_model, tokenizer, metrics_missing_ret_after,
                    full_model, prepared, s1_cache, layer_list,
                    args, mem_data, mia_data
                )
                cached_retain_after.update(scores_after_new)
                results["retain_after"] = {"metrics": cached_retain_after}
                for k, v in scores_after_new.items():
                    if v is not None:
                        print(f"  Retain {k} (after) = {v:.4f}")
                del retain_ft_model
                gc.collect()
                torch.cuda.empty_cache()

                if not args.keep_finetuned:
                    shutil.rmtree(retain_ft_dir, ignore_errors=True)
                    print(f"  Deleted {retain_ft_dir}")
            else:
                del retain_model
                gc.collect()
                torch.cuda.empty_cache()

            results_path.write_text(json.dumps(results, indent=2))
        else:
            print(f"\nRetain baseline: all {len(metrics)} metrics cached")

        retain_before_scores = results.get("retain_before", {}).get("metrics", {})
        retain_after_scores = results.get("retain_after", {}).get("metrics", {})

    # ------------------------------------------------------------------
    # Evaluate each unlearn model
    # ------------------------------------------------------------------
    for mi, (name, model_id) in enumerate(eval_models.items()):
        print(f"\n{'='*60}")
        print(f"[{mi+1}/{len(eval_models)}] {name}")
        print(f"  {model_id}")
        print(f"{'='*60}")

        model_results = results.get(name, {})

        # =============================================================
        # Step 1: Metrics before any intervention
        # =============================================================
        cached_before = model_results.get("metrics_before", {})
        metrics_missing_before = [m for m in metrics if m not in cached_before]
        if metrics_missing_before:
            print(f"\n[1/3] Computing metrics (original)... [{len(metrics_missing_before)} metrics]")
            t0 = time.time()
            source_model = load_model(model_id, dtype="bfloat16", device_map="cuda")
            scores_new = compute_metric_scores_for_model(
                source_model, tokenizer, metrics_missing_before,
                full_model, prepared, s1_cache, layer_list,
                args, mem_data, mia_data
            )
            elapsed = time.time() - t0
            cached_before.update(scores_new)
            model_results["metrics_before"] = cached_before
            for k, v in scores_new.items():
                if v is not None:
                    extra = f"  (1-UDS = {1-v:.4f})" if k == "uds" else ""
                    print(f"  {k} = {v:.4f}{extra}  ({elapsed:.1f}s)")
        else:
            source_model = None
            print(f"\n[1/3] All {len(metrics)} metrics cached")

        # =============================================================
        # Step 2: Robustness to Relearning
        # =============================================================
        scores_before = model_results.get("metrics_before", {})
        cached_R = model_results.get("relearning_R", {})
        metrics_missing_R = [m for m in metrics if m not in cached_R]

        if not args.skip_relearning and metrics_missing_R:
            print(f"\n[2/3] Robustness to Relearning... [{len(metrics_missing_R)} metrics]")

            # Load source if not already loaded
            if source_model is None:
                source_model = load_model(model_id, dtype="bfloat16", device_map="cuda")

            # Finetune on forget10
            ft_dir = os.path.join(args.out_dir, f"ft_{name}")
            t0 = time.time()
            finetune_on_forget10(
                source_model, tokenizer, ft_dir,
                lr=args.relearn_lr, epochs=args.relearn_epochs,
            )
            ft_time = time.time() - t0
            print(f"  Finetuning done ({ft_time:.1f}s)")

            # Unload original, load finetuned
            del source_model
            gc.collect()
            torch.cuda.empty_cache()

            ft_model = load_model(ft_dir, dtype="bfloat16", device_map="cuda")

            # Only compute metrics missing from after-relearn cache
            cached_after_relearn = model_results.get("metrics_after_relearn", {})
            metrics_need_after = [m for m in metrics_missing_R if m not in cached_after_relearn]
            if metrics_need_after:
                scores_after_new = compute_metric_scores_for_model(
                    ft_model, tokenizer, metrics_need_after,
                    full_model, prepared, s1_cache, layer_list,
                    args, mem_data, mia_data
                )
                cached_after_relearn.update(scores_after_new)
                model_results["metrics_after_relearn"] = cached_after_relearn
                for k, v in scores_after_new.items():
                    if v is not None:
                        extra = f"  (1-UDS = {1-v:.4f})" if k == "uds" else ""
                        print(f"  {k} (after relearning) = {v:.4f}{extra}")

            del ft_model
            gc.collect()
            torch.cuda.empty_cache()

            # Cleanup finetuned model
            if not args.keep_finetuned:
                shutil.rmtree(ft_dir, ignore_errors=True)

            # Compute R for missing metrics
            for metric in metrics_missing_R:
                m_unl_a = _metric_to_knowledge(metric, scores_before.get(metric))
                m_unl_b = _metric_to_knowledge(metric, cached_after_relearn.get(metric))
                m_ret_a = _metric_to_knowledge(metric, retain_before_scores.get(metric))
                m_ret_b = _metric_to_knowledge(metric, retain_after_scores.get(metric))
                if None in (m_unl_a, m_unl_b, m_ret_a, m_ret_b):
                    cached_R[metric] = None
                    continue
                denom = m_unl_a - m_unl_b
                numer = m_ret_a - m_ret_b
                if abs(denom) < 1e-8:
                    if args.denom_zero_mode == "one":
                        cached_R[metric] = 1.0
                    else:
                        cached_R[metric] = None
                else:
                    r = numer / denom
                    cached_R[metric] = max(0.0, min(r, 1.0))

            model_results["relearning_R"] = cached_R
            model_results["relearning_details"] = {
                "lr": args.relearn_lr,
                "epochs": args.relearn_epochs,
            }
            source_model = None  # Already freed
        elif args.skip_relearning:
            print("\n[2/3] Relearning: SKIPPED")
        else:
            print(f"\n[2/3] Relearning R (cached, {len(cached_R)} metrics)")

        # =============================================================
        # Step 3: Robustness to Quantization
        # =============================================================
        cached_Q = model_results.get("quantization_Q", {})
        metrics_missing_Q = [m for m in metrics if m not in cached_Q]

        if not args.skip_quantization and metrics_missing_Q:
            print(f"\n[3/3] Robustness to Quantization (4-bit NF4)... [{len(metrics_missing_Q)} metrics]")

            # Free source model if still loaded
            if source_model is not None:
                del source_model
                gc.collect()
                torch.cuda.empty_cache()

            t0 = time.time()
            quant_model = load_model_quantized(model_id, device_map="cuda")
            load_time = time.time() - t0
            print(f"  Quantized model loaded ({load_time:.1f}s)")

            # Only compute metrics missing from after-quant cache
            cached_after_quant = model_results.get("metrics_after_quant", {})
            metrics_need_quant = [m for m in metrics_missing_Q if m not in cached_after_quant]
            if metrics_need_quant:
                scores_quant_new = compute_metric_scores_for_model(
                    quant_model, tokenizer, metrics_need_quant,
                    full_model, prepared, s1_cache, layer_list,
                    args, mem_data, mia_data
                )
                cached_after_quant.update(scores_quant_new)
                model_results["metrics_after_quant"] = cached_after_quant
                for k, v in scores_quant_new.items():
                    if v is not None:
                        extra = f"  (1-UDS = {1-v:.4f})" if k == "uds" else ""
                        print(f"  {k} (after quant) = {v:.4f}{extra}")

            del quant_model
            gc.collect()
            torch.cuda.empty_cache()

            # Compute Q for missing metrics
            for metric in metrics_missing_Q:
                m_unl_a = _metric_to_knowledge(metric, scores_before.get(metric))
                m_unl_b = _metric_to_knowledge(metric, cached_after_quant.get(metric))
                if m_unl_a is None or m_unl_b is None:
                    cached_Q[metric] = None
                    continue
                if abs(m_unl_a) < 1e-8:
                    cached_Q[metric] = 1.0
                else:
                    q = m_unl_b / m_unl_a
                    cached_Q[metric] = max(0.0, min(q, 1.0))

            model_results["quantization_Q"] = cached_Q
            model_results["quantization_details"] = {
                "quant_type": "nf4_4bit",
            }
        elif args.skip_quantization:
            print("\n[3/3] Quantization: SKIPPED")
        else:
            print(f"\n[3/3] Quantization Q (cached, {len(cached_Q)} metrics)")
            source_model = None

        # Free any remaining source model
        if source_model is not None:
            del source_model
            gc.collect()
            torch.cuda.empty_cache()

        results[name] = model_results
        results_path.write_text(json.dumps(results, indent=2))

    # ------------------------------------------------------------------
    # Compute aggregated scores
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ROBUSTNESS RESULTS")
    print("=" * 70)

    summary_rows = []
    metric_R = {m: [] for m in metrics}
    metric_Q = {m: [] for m in metrics}

    for name in eval_models:
        mr = results.get(name, {})
        R = mr.get("relearning_R", {})
        Q = mr.get("quantization_Q", {})
        summary_rows.append({"name": name, "relearning_R": R, "quantization_Q": Q})
        for m in metrics:
            if isinstance(R, dict) and R.get(m) is not None:
                metric_R[m].append(R[m])
            if isinstance(Q, dict) and Q.get(m) is not None:
                metric_Q[m].append(Q[m])

    metric_robust = {}
    print(f"\n{'Metric':<12} {'R':>8} {'Q':>8} {'Rob':>8}")
    print("-" * 40)
    for m in metrics:
        avg_R = float(np.mean(metric_R[m])) if metric_R[m] else None
        avg_Q = float(np.mean(metric_Q[m])) if metric_Q[m] else None
        if avg_R is not None and avg_Q is not None and avg_R > 0 and avg_Q > 0:
            avg_rob = 2 * avg_R * avg_Q / (avg_R + avg_Q)
        else:
            avg_rob = None
        metric_robust[m] = {"avg_R": avg_R, "avg_Q": avg_Q, "robustness": avg_rob}
        r_str = f"{avg_R:.4f}" if avg_R is not None else "N/A"
        q_str = f"{avg_Q:.4f}" if avg_Q is not None else "N/A"
        rob_str = f"{avg_rob:.4f}" if avg_rob is not None else "N/A"
        print(f"{m:<12} {r_str:>8} {q_str:>8} {rob_str:>8}")

    # ------------------------------------------------------------------
    # Overall score (if faithfulness result provided)
    # ------------------------------------------------------------------
    overall = None
    faithfulness = None
    overall_by_metric = {}
    if args.faithfulness_result:
        faith_path = Path(args.faithfulness_result)
        if faith_path.exists():
            faith_data = json.loads(faith_path.read_text())
            faithfulness = faith_data.get("faithfulness", {})
            for m in metrics:
                f_auc = None
                if isinstance(faithfulness, dict):
                    f_auc = faithfulness.get(m, {}).get("auc")
                r_val = metric_robust.get(m, {}).get("robustness")
                if f_auc is not None and r_val is not None and f_auc > 0 and r_val > 0:
                    overall_by_metric[m] = 2 * f_auc * r_val / (f_auc + r_val)
                else:
                    overall_by_metric[m] = None

            print("\nOverall (HM of Faithfulness & Robustness):")
            for m in metrics:
                f_auc = faithfulness.get(m, {}).get("auc") if isinstance(faithfulness, dict) else None
                r_val = metric_robust.get(m, {}).get("robustness")
                o_val = overall_by_metric.get(m)
                if f_auc is None or r_val is None or o_val is None:
                    print(f"  {m}: N/A")
                else:
                    print(f"  {m}: F={f_auc:.4f} R={r_val:.4f} Overall={o_val:.4f}")

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary = {
        "metrics": metrics,
        "metric_robustness": metric_robust,
        "faithfulness": faithfulness,
        "overall": overall_by_metric,
        "n_models": len(eval_models),
        "relearn_lr": args.relearn_lr,
        "relearn_epochs": args.relearn_epochs,
        "denom_zero_mode": args.denom_zero_mode,
        "delta_threshold": args.delta_threshold,
        "patch_scope": args.patch_scope,
        "per_model": summary_rows,
    }

    summary_path = Path(args.out_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\nResults saved to: {args.out_dir}/")
    print(f"  summary.json: R, Q, Robustness scores")
    print(f"  results.json: per-model detailed results")


if __name__ == "__main__":
    main()

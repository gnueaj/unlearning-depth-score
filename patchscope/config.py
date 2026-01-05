#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patchscope configuration and default settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import time


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@dataclass
class ModelConfig:
    """Model-related configuration."""
    target_model_id: str = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
    source_model_id: str = "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5"
    dtype: str = "bfloat16"
    device_map: str = "cuda"  # "cuda" is faster than "auto" for single GPU


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_id: str = "locuslab/TOFU"
    dataset_config: str = "forget10"
    split: str = "train"
    example_index: int = -1  # -1 for first N examples
    num_examples: int = 2


@dataclass
class ProbeConfig:
    """Probing/patching configuration."""
    layers: str = "0-15"
    top_k: int = 10

    # Probe type: "qa", "cloze", "choice"
    # - qa: Question → Answer format (most direct)
    # - cloze: Fill-in-the-blank (stable measurement)
    # - choice: Multiple-choice (most stable, probability comparison)
    probe_type: str = "qa"

    # Wrong answers for probability comparison
    wrong_answers: List[str] = field(default_factory=lambda: [
        "John Smith",
        "Jane Doe",
        "Michael Johnson"
    ])

    # Whether to use chat template for source prompt
    use_chat_template: bool = True


@dataclass
class DebugConfig:
    """Debugging configuration."""
    source_equals_target: bool = False
    generate_full_response: bool = True
    max_new_tokens: int = 50
    log_hidden_stats: bool = False
    verbose: bool = True


@dataclass
class PatchscopeConfig:
    """Main configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    out_dir: str = field(default_factory=lambda: f"runs/{now_ts()}")
    seed: int = 42

    def __post_init__(self):
        if self.debug.source_equals_target:
            self.model.source_model_id = self.model.target_model_id


# =============================================================================
# Unlearning Model Registry
# =============================================================================

UNLEARN_MODELS = {
    # SimNPO variants (beta, alpha, delta, gamma, epoch)
    # Default: lr5e-05, beta3.5, alpha1, delta1, gamma0.25, epoch5
    "simnpo_lr5e5_b35_g025_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5",
    "simnpo_lr5e5_b45_g0125_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.125_ep5",
    "simnpo_lr5e5_b35_d0_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d0_g0.125_ep5",
    "simnpo_lr2e5_b35_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b3.5_a1_d1_g0.125_ep10",
    "simnpo_lr1e5_b35_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b3.5_a1_d1_g0.125_ep10",
    # For contrast: both low vs both high
    "simnpo_lr1e5_b35_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b3.5_a1_d1_g0.125_ep5",  # weak: lr low, ep low
    "simnpo_lr5e5_b45_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.125_ep10",  # strong: lr high, ep high

    # NPO variants (beta, alpha, epoch)
    "npo_b05_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.5_alpha1_epoch10",
    "npo_b05_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.5_alpha1_epoch5",
    "npo_b01_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha2_epoch10",
    "npo_b01_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha2_epoch5",
    "npo_b005_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.05_alpha1_epoch10",
    "npo_b005_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.05_alpha1_epoch5",
    "npo_b005_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.05_alpha5_epoch10",
    "npo_b005_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.05_alpha5_epoch5",
    "npo_lr2e5_b05_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10",
    "npo_lr2e5_b05_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch5",
    "npo_lr2e5_b05_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha2_epoch10",
    "npo_lr2e5_b05_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha2_epoch5",
    "npo_lr2e5_b01_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.1_alpha5_epoch5",
    "npo_lr1e5_b05_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.5_alpha1_epoch10",

    # IdkNLL variants (alpha, epoch)
    "idknll_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr4e-05_alpha5_epoch10",
    "idknll_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr4e-05_alpha5_epoch5",
    "idknll_lr3e5_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr3e-05_alpha1_epoch10",
    "idknll_lr3e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr3e-05_alpha1_epoch5",
    "idknll_lr2e5_a10_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha10_epoch10",
    "idknll_lr2e5_a10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha10_epoch5",
    "idknll_lr1e5_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha1_epoch10",
    "idknll_lr1e5_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr1e-05_alpha1_epoch5",

    # IdkDPO variants (beta, alpha, epoch)
    "idkdpo_b01_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.1_alpha1_epoch10",  # most downloaded
    "idkdpo_b05_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.5_alpha1_epoch5",
    "idkdpo_lr5e5_b005_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.05_alpha5_epoch10",
    "idkdpo_lr5e5_b005_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.05_alpha5_epoch5",

    # GradDiff variants (alpha, epoch)
    "graddiff_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10",
    "graddiff_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5",
    "graddiff_lr2e5_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha5_epoch10",
    "graddiff_lr2e5_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha5_epoch5",
    "graddiff_lr5e5_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha2_epoch10",
    "graddiff_lr5e5_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha2_epoch5",

    # AltPO variants (beta, alpha, epoch)
    "altpo_b05_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.5_alpha2_epoch10",
    "altpo_b05_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.5_alpha2_epoch5",
    "altpo_b01_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10",
    "altpo_b01_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch5",
    "altpo_b005_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.05_alpha2_epoch5",
    "altpo_lr2e5_b05_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.5_alpha5_epoch10",
    "altpo_lr2e5_b05_a5_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.5_alpha5_epoch5",
    "altpo_lr2e5_b005_a5_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.05_alpha5_epoch10",

    # RMU variants (layer, scoeff, epoch)
    "rmu_l5_s100_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff100_epoch10",
    "rmu_l5_s100_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff100_epoch5",
    "rmu_lr5e5_l10_s10_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer10_scoeff10_epoch5",

    # UNDIAL variants (beta, alpha, epoch)
    "undial_b3_a1_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta3_alpha1_epoch10",
    "undial_b3_a1_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta3_alpha1_epoch5",
    "undial_lr1e4_b10_a2_ep10": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha2_epoch10",
    "undial_lr1e4_b10_a2_ep5": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha2_epoch5",

    # Shorthand aliases for convenience
    "simnpo": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5",
    "npo": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.5_alpha1_epoch10",
    "idknll": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr4e-05_alpha5_epoch10",
    "idkdpo": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.5_alpha1_epoch5",
    "graddiff": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10",
    "altpo": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.5_alpha2_epoch10",
    "rmu": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff100_epoch10",
    "undial": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr1e-05_beta3_alpha1_epoch10",
}


def get_model_id(name: str) -> str:
    """Get full model ID from short name or return as-is if full path."""
    if name in UNLEARN_MODELS:
        return UNLEARN_MODELS[name]
    return name


# =============================================================================
# Presets
# =============================================================================

PRESETS = {
    # Default: QA probe
    "default": PatchscopeConfig(),

    # Debug: source = target (sanity check)
    "debug": PatchscopeConfig(
        debug=DebugConfig(
            source_equals_target=True,
            verbose=True
        )
    ),

    # QA probe (direct question answering)
    "qa": PatchscopeConfig(
        probe=ProbeConfig(probe_type="qa"),
    ),

    # Cloze probe (fill-in-the-blank)
    "cloze": PatchscopeConfig(
        probe=ProbeConfig(probe_type="cloze"),
    ),

    # Choice probe (multiple choice - most stable)
    "choice": PatchscopeConfig(
        probe=ProbeConfig(probe_type="choice"),
    ),

    # Full analysis: all layers, multiple examples
    "full": PatchscopeConfig(
        data=DataConfig(num_examples=5),
        probe=ProbeConfig(layers="0-15"),
        debug=DebugConfig(log_hidden_stats=True),
    ),

    # Quick test
    "quick": PatchscopeConfig(
        probe=ProbeConfig(layers="0,8,15"),
        data=DataConfig(num_examples=1),
    ),
}

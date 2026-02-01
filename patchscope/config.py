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

# Import full model registry (398 models)
from patchscope.unlearn_models import UNLEARN_MODELS_FULL

# Shorthand aliases for convenience (default configs)
UNLEARN_ALIASES = {
    "simnpo": "simnpo_lr2e5_b35_a1_d1_g0125_ep10",
    "npo": "npo_lr5e5_b05_a1_ep10",
    "idknll": "idknll_lr3e5_a1_ep5",
    "idkdpo": "idkdpo_lr2e5_b05_a1_ep5",
    "graddiff": "graddiff_lr1e5_a5_ep10",
    "altpo": "altpo_lr5e5_b05_a2_ep10",
    "rmu": "rmu_lr1e5_l5_s100_ep10",
    "undial": "undial_lr1e5_b3_a1_ep10",
}

# Combined registry: full models + aliases
UNLEARN_MODELS = UNLEARN_MODELS_FULL.copy()
for alias, target in UNLEARN_ALIASES.items():
    if target in UNLEARN_MODELS_FULL:
        UNLEARN_MODELS[alias] = UNLEARN_MODELS_FULL[target]


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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patchscope: A tool for analyzing unlearning via hidden state patching.
"""

from .config import (
    PatchscopeConfig,
    ModelConfig,
    DataConfig,
    ProbeConfig,
    DebugConfig,
    PRESETS,
    UNLEARN_MODELS,
    get_model_id,
)
from .models import (
    load_models,
    load_model,
    load_tokenizer,
    get_num_layers,
    try_apply_chat_template,
)
from .core import (
    get_last_token_hidden,
    get_hidden_at_position,
    get_answer_position_hidden,
    get_generated_answer_hidden,
    get_hidden_stats,
    get_token_probabilities,
    get_sequence_probability,
    forward_with_patch,
    decode_next_token_distribution,
    generate_with_patch,
    get_baseline_next_token,
    generate_baseline,
    probe_knowledge,
    probe_knowledge_with_patch,
    compute_knowledge_score,
    build_token_identity_prompt,
    pick_single_token_string,
)
from .probes import (
    ProbeResult,
    build_qa_probe,
    build_qa_probe_with_chat,
    build_cloze_probe,
    build_cloze_from_qa,
    build_choice_probe,
    build_binary_choice_probe,
    build_tofu_probes,
    extract_entity_from_tofu_answer,
    TOFU_WRONG_ANSWERS,
)
from .utils import (
    set_seed,
    safe_mkdir,
    parse_layers,
    format_topk_tokens,
)

__all__ = [
    # Config
    "PatchscopeConfig",
    "ModelConfig",
    "DataConfig",
    "ProbeConfig",
    "DebugConfig",
    "PRESETS",
    "UNLEARN_MODELS",
    "get_model_id",
    # Models
    "load_models",
    "load_model",
    "load_tokenizer",
    "get_num_layers",
    "try_apply_chat_template",
    # Core - Hidden state extraction
    "get_last_token_hidden",
    "get_hidden_at_position",
    "get_answer_position_hidden",
    "get_generated_answer_hidden",
    "get_hidden_stats",
    # Core - Probability evaluation
    "get_token_probabilities",
    "get_sequence_probability",
    "forward_with_patch",
    # Core - Decoding
    "decode_next_token_distribution",
    "generate_with_patch",
    "get_baseline_next_token",
    "generate_baseline",
    # Core - Knowledge probing
    "probe_knowledge",
    "probe_knowledge_with_patch",
    "compute_knowledge_score",
    # Core - Legacy
    "build_token_identity_prompt",
    "pick_single_token_string",
    # Probes
    "ProbeResult",
    "build_qa_probe",
    "build_qa_probe_with_chat",
    "build_cloze_probe",
    "build_cloze_from_qa",
    "build_choice_probe",
    "build_binary_choice_probe",
    "build_tofu_probes",
    "extract_entity_from_tofu_answer",
    "TOFU_WRONG_ANSWERS",
    # Utils
    "set_seed",
    "safe_mkdir",
    "parse_layers",
    "format_topk_tokens",
]

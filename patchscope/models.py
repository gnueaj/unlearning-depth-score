#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model loading and utility functions for Patchscope.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional

from .config import ModelConfig


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load tokenizer with proper settings."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_id: str,
    dtype: str = "bfloat16",
    device_map: str = "auto"
) -> AutoModelForCausalLM:
    """Load a model for inference."""
    torch_dtype = get_torch_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    model.eval()
    return model


def load_models(
    config: ModelConfig,
    verbose: bool = True
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """
    Load source and target models along with tokenizer.

    Returns:
        (target_model, source_model, tokenizer)
    """
    if verbose:
        print(f"[INFO] Loading tokenizer from: {config.target_model_id}")
    tokenizer = load_tokenizer(config.target_model_id)

    if verbose:
        print(f"[INFO] Loading target model: {config.target_model_id}")
    target_model = load_model(
        config.target_model_id,
        dtype=config.dtype,
        device_map=config.device_map
    )

    # Check if source == target (debug mode)
    if config.source_model_id == config.target_model_id:
        if verbose:
            print(f"[INFO] Source = Target mode (using same model)")
        source_model = target_model
    else:
        if verbose:
            print(f"[INFO] Loading source model: {config.source_model_id}")
        source_model = load_model(
            config.source_model_id,
            dtype=config.dtype,
            device_map=config.device_map
        )

    return target_model, source_model, tokenizer


def get_num_layers(model: AutoModelForCausalLM) -> int:
    """Get the number of transformer layers in the model."""
    # Try common attribute paths
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    else:
        raise RuntimeError("Could not infer number of layers from model architecture.")


def get_layer_module(model: AutoModelForCausalLM, layer_idx: int):
    """Get the transformer layer module at given index."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    else:
        raise RuntimeError("Could not locate transformer blocks.")


def try_apply_chat_template(tokenizer: AutoTokenizer, user_text: str) -> str:
    """
    Apply chat template if available, otherwise use simple fallback.
    """
    try:
        messages = [{"role": "user", "content": user_text}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return rendered
    except Exception:
        return f"User: {user_text}\nAssistant:"

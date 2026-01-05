#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Patchscope.
"""

import os
import random
import torch
from typing import List


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def parse_layers(layer_spec: str, n_layers: int) -> List[int]:
    """
    Parse layer specification string.

    Formats:
        "0,2,4,6"      -> specific layers
        "0-10"         -> range (inclusive)
        "0-24:2"       -> range with step
        "all"          -> all layers

    Args:
        layer_spec: Layer specification string
        n_layers: Total number of layers in model

    Returns:
        List of valid layer indices
    """
    layer_spec = layer_spec.strip().lower()

    if layer_spec == "all":
        return list(range(n_layers))

    if "," in layer_spec:
        # Comma-separated list
        vals = []
        for part in layer_spec.split(","):
            part = part.strip()
            if part:
                vals.append(int(part))
        return [v for v in vals if 0 <= v < n_layers]

    if "-" in layer_spec:
        # Range with optional step
        if ":" in layer_spec:
            range_part, step_part = layer_spec.split(":")
            step = int(step_part)
        else:
            range_part = layer_spec
            step = 1

        a, b = range_part.split("-")
        a, b = int(a), int(b)

        if a <= b:
            vals = list(range(a, b + 1, step))
        else:
            vals = list(range(a, b - 1, -step))

        return [v for v in vals if 0 <= v < n_layers]

    # Single integer
    v = int(layer_spec)
    return [v] if 0 <= v < n_layers else []


def format_topk_tokens(topk: List[dict], max_show: int = 5) -> str:
    """Format top-k tokens for display."""
    parts = []
    for i, t in enumerate(topk[:max_show]):
        tok_str = repr(t["token_str"])
        prob = t["prob"]
        parts.append(f"{tok_str}({prob:.3f})")
    return " | ".join(parts)

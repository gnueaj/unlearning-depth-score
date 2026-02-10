#!/usr/bin/env python3
"""
Compute representation-level baseline scores (CKA, Logit Lens, Fisher)
for measuring unlearning depth.

Each method produces a per-model score in [0, 1]:
  1 = retain-like (erasure complete)
  0 = full-like (knowledge intact)

Usage:
  # Compute anchor cache (full + retain reference)
  python scripts/compute_representation_baselines.py --mode anchor --gpu 0

  # Compute scores for a specific model
  python scripts/compute_representation_baselines.py --mode eval --model_name npo_lr5e5_b01_a1_ep10 --gpu 0

  # Compute scores for all 150 unlearned models
  python scripts/compute_representation_baselines.py --mode eval_all --gpu 0

  # List available methods
  python scripts/compute_representation_baselines.py --mode anchor --methods cka,logit_lens
"""

import os
import sys
import gc
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchscope.models import load_model, load_tokenizer, get_num_layers
from patchscope.config import get_model_id
from exp_s1_teacher_forcing import (
    load_prefix_data,
    get_eval_span,
    normalize_reference_for_eval,
    build_logprob_ctx,
    _prepare_batch_inputs,
    _compute_hidden_states_batch,
    _gather_token_logprobs,
)

TOFU_FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
TOFU_RETAIN_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
PREFIX_DATA_PATH = "tofu_data/forget10_filtered_v7_gt.json"
DEFAULT_OUT_DIR = "runs/meta_eval/representation_baselines"
DEFAULT_DELTA_THRESHOLD = 0.05
EPS = 1e-8

# Faithfulness P/N pools (same as meta_eval_faithfulness.py)
_P_BASE = "open-unlearning/pos_tofu_Llama-3.2-1B-Instruct"
P_POOL = []
for _lr in ["1e-05", "2e-05", "3e-05", "4e-05", "5e-05"]:
    for _ep in [5, 10]:
        P_POOL.append(f"{_P_BASE}_full_lr{_lr}_wd0.01_epoch{_ep}")
        P_POOL.append(f"{_P_BASE}_retain90_forget10_bio_lr{_lr}_wd0.01_epoch{_ep}")
        P_POOL.append(f"{_P_BASE}_retain90_forget10_para_lr{_lr}_wd0.01_epoch{_ep}")

_N_BASE = "open-unlearning/neg_tofu_Llama-3.2-1B-Instruct"
N_POOL = []
for _lr in ["1e-05", "2e-05", "3e-05", "4e-05", "5e-05"]:
    for _ep in [5, 10]:
        N_POOL.append(f"{_N_BASE}_retain90_lr{_lr}_wd0.01_epoch{_ep}")
        N_POOL.append(f"{_N_BASE}_retain90_forget10_pert_lr{_lr}_wd0.01_epoch{_ep}")
        N_POOL.append(f"{_N_BASE}_retain90_celeb10_bio_lr{_lr}_wd0.01_epoch{_ep}")

FAITH_ALL_MODELS = [(mid, "P") for mid in P_POOL] + [(mid, "N") for mid in N_POOL]


# ============================================================================
# Data preparation (reuse from exp_s1_teacher_forcing)
# ============================================================================

def prepare_contexts(tokenizer, prefix_data: List[Dict]) -> List[Dict]:
    """Build teacher-forcing contexts for all forget-set examples.

    Returns list of context dicts with token positions for entity spans.
    Skips examples where entity span is not found.
    """
    contexts = []
    for item in prefix_data:
        question = item["question"]
        prefix = item["prefix"]
        gt_entity = item.get("gt_entity", item["entity"])
        idx = item["idx"]

        prompt = f"Question: {question}\nAnswer: {prefix}"

        answer_text = item["answer"]
        if answer_text.startswith(prefix):
            answer_text = answer_text[len(prefix):]
        reference_text = normalize_reference_for_eval(prompt, answer_text)

        eval_span = get_eval_span(tokenizer, reference_text, gt_entity, "entity")
        if eval_span is None:
            continue

        ctx = build_logprob_ctx(tokenizer, prompt, reference_text, eval_span, "span")
        if ctx is None:
            continue

        ctx["idx"] = idx
        ctx["entity_span"] = eval_span  # (start, end) in reference tokens
        ctx["question"] = question
        ctx["answer"] = item["answer"]
        ctx["prefix"] = prefix
        ctx["gt_entity"] = gt_entity
        contexts.append(ctx)

    return contexts


def extract_hidden_states(model, tokenizer, contexts, batch_size=32):
    """Extract all-layer hidden states for all examples.

    For layers 0..n-2: uses output_hidden_states[l+1] (pre-next-layer = post-current-layer).
    For last layer (n-1): uses a hook on model.norm to capture the pre-norm hidden state,
    since output_hidden_states[-1] has the source model's own RMSNorm baked in.

    Returns:
        hidden_states: dict[layer_idx] -> list of tensors, one per example
                       each tensor has shape [num_entity_tokens, hidden_dim]
    """
    device = next(model.parameters()).device
    n_layers = get_num_layers(model)
    # Collect entity hidden states per layer
    layer_hidden = {l: [] for l in range(n_layers)}

    # Hook to capture pre-final-norm hidden state (= last layer's raw output)
    pre_norm_capture = {}
    def _capture_pre_norm(module, input, output):
        pre_norm_capture["hidden"] = input[0].detach()
    hook = model.model.norm.register_forward_hook(_capture_pre_norm)

    for batch_start in range(0, len(contexts), batch_size):
        batch_ctxs = contexts[batch_start:batch_start + batch_size]
        input_ids, attention_mask, meta = _prepare_batch_inputs(
            batch_ctxs, tokenizer, device
        )

        # Single forward pass -> all layer hidden states + hook captures pre-norm
        all_hidden = _compute_hidden_states_batch(model, input_ids, attention_mask)
        # all_hidden[0] = embeddings, all_hidden[l+1] = layer l output (l < n-1)
        # all_hidden[n] = post-final-norm (WRONG for Logit Lens, use hook instead)
        # pre_norm_capture["hidden"] = pre-final-norm (correct for last layer)

        for i in range(len(batch_ctxs)):
            es = meta["eval_start"][i]
            ee = meta["eval_end"][i]
            for l in range(n_layers):
                if l < n_layers - 1:
                    h = all_hidden[l + 1][i, es:ee, :]  # [num_entity_tokens, D]
                else:
                    # Last layer: use hook-captured pre-norm hidden state
                    h = pre_norm_capture["hidden"][i, es:ee, :]
                layer_hidden[l].append(h.cpu())

        # Free GPU memory
        del all_hidden
        pre_norm_capture.clear()
        torch.cuda.empty_cache()

    hook.remove()
    return layer_hidden


# ============================================================================
# CKA computation
# ============================================================================

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between X and Y.

    Args:
        X, Y: [N, D] tensors (N = total entity tokens, D = hidden dim)

    Returns:
        CKA value in [0, 1]
    """
    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    YtX = Y.T @ X  # [D, D]
    XtX = X.T @ X  # [D, D]
    YtY = Y.T @ Y  # [D, D]

    numerator = (YtX * YtX).sum()
    denominator = torch.sqrt((XtX * XtX).sum() * (YtY * YtY).sum())

    if denominator < EPS:
        return 0.0

    return (numerator / denominator).item()


def compute_cka_scores(
    hidden_unl: Dict[int, List[torch.Tensor]],
    hidden_ret: Dict[int, List[torch.Tensor]],
    hidden_full: Dict[int, List[torch.Tensor]],
    anchor_cache: Optional[Dict] = None,
) -> Dict:
    """Compute CKA-based erasure score.

    Args:
        hidden_*: dict[layer] -> list of [num_entity_tokens, D] tensors
        anchor_cache: precomputed CKA(full, retain) per layer

    Returns:
        dict with per-layer erasure and aggregated score
    """
    n_layers = len(hidden_full)
    per_layer = {}

    for l in range(n_layers):
        # Concatenate all entity tokens across examples -> [N_total, D]
        H_unl = torch.cat(hidden_unl[l], dim=0).float()
        H_ret = torch.cat(hidden_ret[l], dim=0).float()

        if anchor_cache is not None:
            cka_full_ret = anchor_cache["cka_full_ret"][l]
        else:
            H_full = torch.cat(hidden_full[l], dim=0).float()
            cka_full_ret = linear_cka(H_full, H_ret)

        cka_unl_ret = linear_cka(H_unl, H_ret)

        # Erasure: closer to retain = higher
        denom = 1.0 - cka_full_ret + EPS
        erasure = max(0.0, min(1.0, (cka_unl_ret - cka_full_ret) / denom))

        per_layer[l] = {
            "cka_unl_ret": cka_unl_ret,
            "cka_full_ret": cka_full_ret,
            "erasure": erasure,
            "weight": max(1.0 - cka_full_ret, 0.0),
        }

    # Aggregate: weighted sum over FT layers
    return _aggregate_dataset_level(per_layer, "cka")


# ============================================================================
# Logit Lens computation (fixed decoder = full model's norm + lm_head)
# ============================================================================

def compute_logit_lens_logprobs(
    full_model,
    hidden_states: Dict[int, List[torch.Tensor]],
    contexts: List[Dict],
) -> Dict[int, List[float]]:
    """Compute per-example, per-layer entity log-probs using full model's decoder.

    Args:
        full_model: model providing the fixed decoder (norm + lm_head)
        hidden_states: dict[layer] -> list of [num_entity_tokens, D] tensors
        contexts: list of context dicts with eval_ref_ids

    Returns:
        dict[layer] -> list of per-example mean entity log-probs
    """
    device = next(full_model.parameters()).device
    norm = full_model.model.norm
    lm_head = full_model.lm_head
    n_layers = len(hidden_states)

    result = {l: [] for l in range(n_layers)}

    for l in range(n_layers):
        for i, ctx in enumerate(contexts):
            h = hidden_states[l][i].to(device=device, dtype=norm.weight.dtype)  # [num_entity_tokens, D]
            entity_ids = ctx["eval_ref_ids"]  # list of token ids

            # Fixed decoder: full model's norm + lm_head
            with torch.no_grad():
                normed = norm(h.unsqueeze(0))  # [1, T, D]
                logits = lm_head(normed)  # [1, T, V]

            labels = torch.tensor([entity_ids], device=device)
            token_logprobs = _gather_token_logprobs(logits, labels)
            result[l].append(token_logprobs.mean().item())

    return result


def compute_logit_lens_scores(
    ll_unl: Dict[int, List[float]],
    anchor_cache: Dict,
    delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
) -> Dict:
    """Compute Logit Lens erasure score (per-example, UDS-parallel structure).

    Args:
        ll_unl: dict[layer] -> list of per-example logprobs
        anchor_cache: must contain ll_full and ll_ret (per-layer, per-example)
        delta_threshold: FT layer threshold

    Returns:
        dict with per-layer details and aggregated score
    """
    ll_full = anchor_cache["ll_full"]
    ll_ret = anchor_cache["ll_ret"]
    n_layers = len(ll_full)
    n_examples = len(ll_full[0])

    # Per-example UDS-style computation
    example_scores = []
    per_layer_erasures = {l: [] for l in range(n_layers)}

    for i in range(n_examples):
        numer = 0.0
        denom = 0.0
        for l in range(n_layers):
            d_ret = ll_full[l][i] - ll_ret[l][i]   # S1-LL: retain gap
            d_unl = ll_full[l][i] - ll_unl[l][i]   # S2-LL: unlearned gap

            if d_ret <= delta_threshold:
                per_layer_erasures[l].append(None)
                continue

            ratio = max(0.0, min(1.0, d_unl / (d_ret + EPS)))
            per_layer_erasures[l].append(ratio)

            denom += d_ret
            numer += d_ret * ratio

        if denom > EPS:
            example_scores.append(numer / denom)

    # Per-layer mean erasure (across examples where layer is FT)
    per_layer = {}
    for l in range(n_layers):
        valid = [v for v in per_layer_erasures[l] if v is not None]
        d_rets = [ll_full[l][i] - ll_ret[l][i] for i in range(n_examples)]
        mean_d_ret = np.mean(d_rets) if d_rets else 0.0

        per_layer[l] = {
            "mean_erasure": float(np.mean(valid)) if valid else None,
            "n_ft_examples": len(valid),
            "mean_d_ret": mean_d_ret,
        }

    score = float(np.mean(example_scores)) if example_scores else None
    return {
        "score": score,
        "n_examples_scored": len(example_scores),
        "per_layer": per_layer,
    }


# ============================================================================
# Fisher Information computation
# ============================================================================

def compute_layer_fisher(
    model, tokenizer, contexts, batch_size=4, return_raw=False
) -> Dict:
    """Compute per-layer Fisher Information.

    When return_raw=False (default): returns dict[layer] -> multi-agg dict
    When return_raw=True: returns dict[layer] -> raw per-parameter vector (CPU tensor)

    Uses per-example gradients for correct diagonal Fisher.
    """
    device = next(model.parameters()).device
    n_layers = get_num_layers(model)

    # Accumulate per-parameter squared gradients: Σ_i g_i²
    fisher_accum = {l: None for l in range(n_layers)}
    n_examples = 0

    model.eval()
    for batch_start in range(0, len(contexts), batch_size):
        batch_ctxs = contexts[batch_start:batch_start + batch_size]
        input_ids, attention_mask, meta = _prepare_batch_inputs(
            batch_ctxs, tokenizer, device
        )

        # Forward pass (shared across batch)
        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Per-example backward for correct diagonal Fisher
        for i in range(len(batch_ctxs)):
            es = meta["eval_start"][i]
            ee = meta["eval_end"][i]
            if ee <= es:
                continue

            labels = torch.tensor([meta["eval_ref_ids"][i]], device=device)
            token_logprobs = _gather_token_logprobs(
                logits[i:i+1, es:ee, :], labels
            )
            loss_i = -token_logprobs.mean()

            # Backward for this example
            model.zero_grad()
            loss_i.backward(retain_graph=True)

            # Accumulate squared gradients per layer
            for l in range(n_layers):
                layer_module = model.model.layers[l]
                layer_sq_grads = []
                for p in layer_module.parameters():
                    if p.grad is not None:
                        layer_sq_grads.append((p.grad.detach() ** 2).flatten())
                if layer_sq_grads:
                    sq = torch.cat(layer_sq_grads)
                    if fisher_accum[l] is None:
                        fisher_accum[l] = sq.clone()
                    else:
                        fisher_accum[l] += sq

            n_examples += 1

        model.zero_grad()
        # Free graph
        del outputs, logits
        torch.cuda.empty_cache()

    # Compute avg squared gradients per layer
    raw_result = {}
    for l in range(n_layers):
        if fisher_accum[l] is not None and n_examples > 0:
            raw_result[l] = (fisher_accum[l] / n_examples).cpu()
        else:
            raw_result[l] = None

    if return_raw:
        return raw_result

    # F_l = log1p( agg_θ( (1/N) Σ_i g_{i,θ}² ) )
    # Compute multiple aggregations in one pass: mean + top-k percentiles
    TOPK_FRACS = [0.01, 0.03, 0.05]
    result = {}
    for l in range(n_layers):
        avg_sq = raw_result[l]
        if avg_sq is not None:
            entry = {"mean": float(np.log1p(avg_sq.mean().item()))}
            for frac in TOPK_FRACS:
                k = max(1, int(len(avg_sq) * frac))
                topk_val = avg_sq.topk(k).values.mean().item()
                entry[f"top{frac}"] = float(np.log1p(topk_val))
            result[l] = entry
        else:
            entry = {"mean": 0.0}
            for frac in TOPK_FRACS:
                entry[f"top{frac}"] = 0.0
            result[l] = entry

    return result


def _get_fisher_val(fisher_dict_or_scalar, agg_key="mean"):
    """Extract Fisher value, handling both old (scalar) and new (dict) format."""
    if isinstance(fisher_dict_or_scalar, dict):
        return fisher_dict_or_scalar[agg_key]
    return fisher_dict_or_scalar  # backward compat: old scalar format


def compute_fisher_scores(
    fisher_unl: Dict,
    anchor_cache: Dict,
    agg_key: str = "mean",
) -> Dict:
    """Compute Fisher-based erasure score.

    Args:
        fisher_unl: dict[layer] -> Fisher value (scalar or multi-agg dict)
        anchor_cache: must contain fisher_full and fisher_ret
        agg_key: aggregation variant ("mean", "top0.01", "top0.03", "top0.05")

    Returns:
        dict with per-layer erasure and aggregated score
    """
    fisher_full = anchor_cache["fisher_full"]
    fisher_ret = anchor_cache["fisher_ret"]
    n_layers = len(fisher_full)

    per_layer = {}
    for l in range(n_layers):
        f_full = _get_fisher_val(fisher_full[l], agg_key)
        f_ret = _get_fisher_val(fisher_ret[l], agg_key)
        f_unl = _get_fisher_val(fisher_unl[l], agg_key)

        # Direction: retain has HIGHER Fisher on forget data (novel to retain)
        # full has LOWER Fisher (already learned). So excess = F_ret - F_full.
        excess_full = max(f_ret - f_full, 0.0)
        excess_unl = max(f_ret - f_unl, 0.0)

        if excess_full > EPS:
            erasure = 1.0 - max(0.0, min(1.0, excess_unl / (excess_full + EPS)))
        else:
            erasure = 1.0  # no signal at this layer

        per_layer[l] = {
            "fisher_unl": f_unl,
            "fisher_full": f_full,
            "fisher_ret": f_ret,
            "excess_full": excess_full,
            "excess_unl": excess_unl,
            "erasure": erasure,
            "weight": excess_full,
        }

    return _aggregate_dataset_level(per_layer, "fisher")


# ============================================================================
# Fixed-mask Fisher (retain-full anchor-based parameter selection)
# ============================================================================

FISHER_MASK_FRACS = [0.0001, 0.001, 0.01]


def compute_fisher_mask(
    fisher_raw_ret: Dict[int, torch.Tensor],
    fisher_raw_full: Dict[int, torch.Tensor],
    fracs: list = None,
) -> Dict:
    """Compute fixed parameter mask from retain vs full Fisher.

    For each layer:
        a_i = max(F_ret,i - F_full,i, 0)
        mask = top N% indices of a_i

    Stores per-layer: mask indices, retain_mean_at_M, excess_full.

    Args:
        fisher_raw_ret: dict[layer] -> 1D tensor of avg squared grads (retain)
        fisher_raw_full: dict[layer] -> 1D tensor of avg squared grads (full)
        fracs: list of top-N fractions for mask selection

    Returns:
        dict with mask data for each fraction
    """
    if fracs is None:
        fracs = FISHER_MASK_FRACS

    n_layers = len(fisher_raw_ret)
    result = {}

    for frac in fracs:
        key = f"mask_{frac}"
        mask_data = {}

        for l in range(n_layers):
            f_ret = fisher_raw_ret[l]
            f_full = fisher_raw_full[l]

            if f_ret is None or f_full is None:
                mask_data[l] = {
                    "indices": None,
                    "retain_mean": 0.0,
                    "excess_full": 0.0,
                    "n_params": 0,
                }
                continue

            # a_i = max(F_ret - F_full, 0)
            a = torch.clamp(f_ret - f_full, min=0.0)

            # Top N% indices
            k = max(1, int(len(a) * frac))
            _, top_indices = a.topk(k)

            ret_at_m = f_ret[top_indices].mean().item()
            full_at_m = f_full[top_indices].mean().item()
            excess_full = ret_at_m - full_at_m

            mask_data[l] = {
                "indices": top_indices.to(torch.int32),  # save as int32 for space
                "retain_mean": ret_at_m,
                "excess_full": excess_full,
                "n_params": k,
            }

            print(f"  Layer {l:2d}: mask@{frac} -> {k} params, "
                  f"excess_full={excess_full:.6f}")

        result[key] = mask_data

    return result


def save_fisher_mask(mask_data: Dict, path: Path):
    """Save Fisher mask data to .pt file."""
    # Convert indices to lists for serialization, keep scalars
    serializable = {}
    for frac_key, layer_data in mask_data.items():
        serializable[frac_key] = {}
        for l, d in layer_data.items():
            entry = {
                "retain_mean": d["retain_mean"],
                "excess_full": d["excess_full"],
                "n_params": d["n_params"],
            }
            if d["indices"] is not None:
                entry["indices"] = d["indices"]
            serializable[frac_key][l] = entry
    torch.save(serializable, path)
    print(f"[Fisher Mask] Saved to {path}")


def load_fisher_mask(path: Path) -> Dict:
    """Load Fisher mask data from .pt file."""
    return torch.load(path, weights_only=False)


def compute_fisher_masked_scores(
    fisher_raw_unl: Dict[int, torch.Tensor],
    mask_data: Dict,
    frac: float,
) -> Dict:
    """Compute Fisher erasure score using fixed anchor mask.

    For each layer:
        excess_unl = retain_mean_at_M - mean(F_unl[M])
        erasure = 1 - clip(excess_unl / excess_full, 0, 1)

    Score = weighted average of erasure across FT layers (weight = excess_full).
    """
    frac_key = f"mask_{frac}"
    layer_mask = mask_data[frac_key]
    n_layers = len(layer_mask)

    per_layer = {}
    for l in range(n_layers):
        md = layer_mask[l]
        indices = md.get("indices")
        excess_full = md["excess_full"]
        retain_mean = md["retain_mean"]

        if indices is None or excess_full <= EPS:
            per_layer[l] = {
                "excess_full": excess_full,
                "excess_unl": 0.0,
                "erasure": 1.0,
                "weight": excess_full,
            }
            continue

        f_unl = fisher_raw_unl[l]
        if f_unl is None:
            per_layer[l] = {
                "excess_full": excess_full,
                "excess_unl": 0.0,
                "erasure": 1.0,
                "weight": excess_full,
            }
            continue

        unl_at_m = f_unl[indices.long()].mean().item()
        excess_unl = retain_mean - unl_at_m

        erasure = 1.0 - max(0.0, min(1.0, excess_unl / (excess_full + EPS)))

        per_layer[l] = {
            "excess_full": excess_full,
            "excess_unl": excess_unl,
            "erasure": erasure,
            "weight": excess_full,
        }

    return _aggregate_dataset_level(per_layer, "fisher_masked")


# ============================================================================
# Aggregation helpers
# ============================================================================

def _aggregate_dataset_level(per_layer: Dict, method_name: str) -> Dict:
    """Aggregate per-layer erasure to model-level score (CKA / Fisher)."""
    layers = sorted(per_layer.keys())
    weights = [per_layer[l]["weight"] for l in layers]
    erasures = [per_layer[l]["erasure"] for l in layers]

    # FT layers: where weight > 0 (meaningful gap)
    ft_layers = [l for l in layers if per_layer[l]["weight"] > EPS]
    if not ft_layers:
        return {"score": None, "per_layer": per_layer, "ft_layers": []}

    total_w = sum(per_layer[l]["weight"] for l in ft_layers)
    if total_w < EPS:
        return {"score": None, "per_layer": per_layer, "ft_layers": ft_layers}

    score = sum(
        per_layer[l]["weight"] * per_layer[l]["erasure"] for l in ft_layers
    ) / total_w

    return {
        "score": float(score),
        "ft_layers": ft_layers,
        "per_layer": per_layer,
    }


# ============================================================================
# Anchor cache (full + retain reference values)
# ============================================================================

def compute_anchor_cache(
    full_model, retain_model, tokenizer, contexts,
    methods: List[str], batch_size: int = 32,
    fisher_batch_size: int = 4,
) -> Dict:
    """Compute and return anchor cache for full and retain models.

    This is computed once and reused across all unlearned models.
    """
    cache = {}
    n_layers = get_num_layers(full_model)

    # Extract hidden states (shared by CKA and Logit Lens)
    need_hidden = "cka" in methods or "logit_lens" in methods
    if need_hidden:
        print("[Anchor] Extracting full model hidden states...")
        hidden_full = extract_hidden_states(
            full_model, tokenizer, contexts, batch_size
        )
        cache["hidden_full"] = hidden_full

        print("[Anchor] Extracting retain model hidden states...")
        hidden_ret = extract_hidden_states(
            retain_model, tokenizer, contexts, batch_size
        )
        cache["hidden_ret"] = hidden_ret

    # CKA: precompute CKA(full, retain) per layer
    if "cka" in methods:
        print("[Anchor] Computing CKA(full, retain) per layer...")
        cka_full_ret = {}
        for l in range(n_layers):
            H_full = torch.cat(hidden_full[l], dim=0).float()
            H_ret = torch.cat(hidden_ret[l], dim=0).float()
            cka_full_ret[l] = linear_cka(H_full, H_ret)
            print(f"  Layer {l:2d}: CKA(full, retain) = {cka_full_ret[l]:.4f}")
        cache["cka_full_ret"] = cka_full_ret

    # Logit Lens: precompute logprobs for full and retain
    if "logit_lens" in methods:
        print("[Anchor] Computing Logit Lens logprobs (full hidden -> full decoder)...")
        cache["ll_full"] = compute_logit_lens_logprobs(
            full_model, hidden_full, contexts
        )
        print("[Anchor] Computing Logit Lens logprobs (retain hidden -> full decoder)...")
        cache["ll_ret"] = compute_logit_lens_logprobs(
            full_model, hidden_ret, contexts
        )

    # Fisher: precompute for full and retain
    if "fisher" in methods:
        print("[Anchor] Computing Fisher for full model...")
        cache["fisher_full"] = compute_layer_fisher(
            full_model, tokenizer, contexts, fisher_batch_size
        )
        print("[Anchor] Computing Fisher for retain model...")
        cache["fisher_ret"] = compute_layer_fisher(
            retain_model, tokenizer, contexts, fisher_batch_size
        )

    # Fixed-mask Fisher: compute raw vectors and generate mask
    if "fisher_masked" in methods:
        print("[Anchor] Computing raw Fisher for full model (for mask)...")
        fisher_raw_full = compute_layer_fisher(
            full_model, tokenizer, contexts, fisher_batch_size, return_raw=True
        )
        print("[Anchor] Computing raw Fisher for retain model (for mask)...")
        fisher_raw_ret = compute_layer_fisher(
            retain_model, tokenizer, contexts, fisher_batch_size, return_raw=True
        )
        print("[Anchor] Computing fixed parameter masks...")
        cache["fisher_mask"] = compute_fisher_mask(fisher_raw_ret, fisher_raw_full)
        # Also store aggregated Fisher if not already done
        if "fisher_full" not in cache:
            cache["fisher_full"] = compute_layer_fisher(
                full_model, tokenizer, contexts, fisher_batch_size
            )
        if "fisher_ret" not in cache:
            cache["fisher_ret"] = compute_layer_fisher(
                retain_model, tokenizer, contexts, fisher_batch_size
            )
        # Free raw vectors
        del fisher_raw_full, fisher_raw_ret

    return cache


# ============================================================================
# Detailed logging (UDS-style)
# ============================================================================

def write_detail_log(
    log_path: Path,
    model_name: str,
    contexts: List[Dict],
    tokenizer,
    anchor_cache: Dict,
    cka_result: Optional[Dict] = None,
    ll_result: Optional[Dict] = None,
    ll_unl: Optional[Dict] = None,
    fisher_result: Optional[Dict] = None,
    delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
):
    """Write detailed per-example, per-layer log file (UDS-style)."""
    n_layers = len(anchor_cache.get("ll_full", anchor_cache.get("cka_full_ret", {})))
    if n_layers == 0:
        # Try to infer from any available data
        for key in ["ll_full", "cka_full_ret", "fisher_full"]:
            if key in anchor_cache:
                n_layers = len(anchor_cache[key])
                break

    with open(log_path, "w") as f:
        f.write(f"Representation Baseline Log: {model_name}\n")
        f.write(f"Delta threshold: {delta_threshold}\n")
        f.write(f"Examples: {len(contexts)}\n")
        f.write(f"Methods: {', '.join(m for m in ['cka', 'logit_lens', 'fisher'] if any([cka_result and m=='cka', ll_result and m=='logit_lens', fisher_result and m=='fisher']))}\n")

        # ---- CKA summary (dataset-level) ----
        if cka_result is not None:
            f.write(f"\n{'='*80}\n")
            f.write(f"CKA Summary (dataset-level)\n")
            f.write(f"{'='*80}\n")
            f.write(f"  Layer | CKA(full,ret) | CKA(unl,ret) | erasure | weight | FT?\n")
            f.write(f"  ----- | ------------- | ------------ | ------- | ------ | ---\n")
            ft_set = set(cka_result.get("ft_layers", []))
            for l in sorted(cka_result["per_layer"].keys()):
                d = cka_result["per_layer"][l]
                ft_mark = " *" if l in ft_set else ""
                f.write(f"  L{l:02d}  | {d['cka_full_ret']:.4f}        | {d['cka_unl_ret']:.4f}       | {d['erasure']:.4f}  | {d['weight']:.4f} | {ft_mark}\n")
            f.write(f"\n  CKA Score = {cka_result.get('score', 'N/A')}\n")
            f.write(f"  FT layers: {sorted(ft_set)}\n")

        # ---- Fisher summary (dataset-level) ----
        if fisher_result is not None:
            f.write(f"\n{'='*80}\n")
            f.write(f"Fisher Information Summary (dataset-level, per-param mean + log1p)\n")
            f.write(f"{'='*80}\n")
            f.write(f"  Layer | F(full)  | F(ret)   | F(unl)   | excess_f | excess_u | erasure | FT?\n")
            f.write(f"  ----- | -------- | -------- | -------- | -------- | -------- | ------- | ---\n")
            ft_set = set(fisher_result.get("ft_layers", []))
            for l in sorted(fisher_result["per_layer"].keys()):
                d = fisher_result["per_layer"][l]
                ft_mark = " *" if l in ft_set else ""
                f.write(f"  L{l:02d}  | {d['fisher_full']:.6f} | {d['fisher_ret']:.6f} | {d['fisher_unl']:.6f} | {d['excess_full']:.6f} | {d['excess_unl']:.6f} | {d['erasure']:.4f} | {ft_mark}\n")
            f.write(f"\n  Fisher Score = {fisher_result.get('score', 'N/A')}\n")
            f.write(f"  FT layers: {sorted(ft_set)}\n")

        # ---- Logit Lens per-example detail ----
        if ll_result is not None and ll_unl is not None:
            ll_full = anchor_cache["ll_full"]
            ll_ret = anchor_cache["ll_ret"]
            n_examples = len(contexts)

            f.write(f"\n{'='*80}\n")
            f.write(f"Logit Lens Per-Example Detail (fixed decoder = full model's norm + lm_head)\n")
            f.write(f"  Decoder: full model (frozen)\n")
            f.write(f"  k_full = logprob(entity | full hidden → full decoder)\n")
            f.write(f"  k_ret  = logprob(entity | retain hidden → full decoder)\n")
            f.write(f"  k_unl  = logprob(entity | {model_name} hidden → full decoder)\n")
            f.write(f"  d_ret  = k_full − k_ret  (retain's knowledge gap)\n")
            f.write(f"  d_unl  = k_full − k_unl  (unlearned's knowledge gap)\n")
            f.write(f"  ratio  = clip(d_unl / d_ret, 0, 1)  (erasure ratio per layer)\n")
            f.write(f"  FT layer: d_ret > τ={delta_threshold}  (layer where retain lacks knowledge)\n")
            f.write(f"  LOST = d > τ (knowledge missing)  |  KEPT = d ≤ τ (knowledge present)\n")
            f.write(f"  LL Score = Σ(d_ret × ratio) / Σ(d_ret) over FT layers  (UDS-parallel)\n")
            f.write(f"{'='*80}\n")

            per_example_scores = []

            for i in range(n_examples):
                ctx = contexts[i]
                entity_text = tokenizer.decode(ctx["eval_ref_ids"])
                question = ctx.get("question", "")
                answer = ctx.get("answer", "")
                gt_entity = ctx.get("gt_entity", "")

                f.write(f"\n{'='*80}\n")
                f.write(f"[{i+1}/{n_examples}] Example {ctx['idx']}\n")
                f.write(f"  Q: {question}\n")
                f.write(f"  A: {answer}\n")
                f.write(f"  GT entity: {gt_entity}\n")
                f.write(f"  Eval entity (tokens): {entity_text.strip()}\n")
                f.write(f"  Full logprob (entity): {ll_full[n_layers-1][i]:.3f}\n")
                f.write(f"{'='*80}\n")
                f.write(f"  Layer | k_full   | k_ret    | k_unl    | d_ret    | d_unl    | ratio  | Status\n")
                f.write(f"  ----- | -------- | -------- | -------- | -------- | -------- | ------ | ------\n")

                ft_layers_i = []
                erased_layers_i = []
                numer = 0.0
                denom = 0.0
                for l in range(n_layers):
                    kf = ll_full[l][i]
                    kr = ll_ret[l][i]
                    ku = ll_unl[l][i]
                    dr = kf - kr
                    du = kf - ku

                    if dr > delta_threshold:
                        ratio = max(0.0, min(1.0, du / (dr + EPS)))
                        ft_layers_i.append(l)
                        denom += dr
                        numer += dr * ratio
                        # S2 status: LOST if d_unl > tau (knowledge missing from unlearned)
                        s2_status = "LOST" if du > delta_threshold else "KEPT"
                        if du > delta_threshold:
                            erased_layers_i.append(l)
                        ratio_str = f"{ratio:.4f}"
                        f.write(f"  L{l:02d}  | {kf:8.3f} | {kr:8.3f} | {ku:8.3f} | {dr:8.3f} | {du:8.3f} | {ratio_str} | [FT] S2:{s2_status}\n")
                    else:
                        ratio = None
                        ratio_str = "  -   "
                        f.write(f"  L{l:02d}  | {kf:8.3f} | {kr:8.3f} | {ku:8.3f} | {dr:8.3f} | {du:8.3f} | {ratio_str} | [KEPT]\n")

                score_i = numer / denom if denom > EPS else None
                per_example_scores.append(score_i)
                f.write(f"  ----- | -------- | -------- | -------- | -------- | -------- | ------ | ------\n")
                f.write(f"  FT layers (d_ret > τ):        {ft_layers_i}\n")
                f.write(f"  Erased layers (S2 LOST ∩ FT): {erased_layers_i}\n")
                score_str = f"{score_i:.4f}" if score_i is not None else "N/A (no FT signal)"
                f.write(f"  LL Score = {score_str}\n")

            # Layer-level summary
            f.write(f"\n{'='*80}\n")
            f.write(f"Logit Lens Layer Summary (across {n_examples} examples)\n")
            f.write(f"{'='*80}\n")
            f.write(f"  Layer | mean_d_ret | mean_d_unl | n_FT | mean_erasure\n")
            f.write(f"  ----- | ---------- | ---------- | ---- | ------------\n")
            for l in range(n_layers):
                d = ll_result["per_layer"][l]
                # Compute mean_d_unl from raw data
                d_unls = [ll_full[l][i] - ll_unl[l][i] for i in range(n_examples)]
                mean_d_unl = sum(d_unls) / len(d_unls) if d_unls else 0
                me = f"{d['mean_erasure']:.4f}" if d['mean_erasure'] is not None else "  N/A  "
                f.write(f"  L{l:02d}  | {d['mean_d_ret']:10.4f} | {mean_d_unl:10.4f} | {d['n_ft_examples']:4d} | {me}\n")

            # Per-example score distribution
            scored = [s for s in per_example_scores if s is not None]
            f.write(f"\n  Logit Lens Score = {ll_result.get('score', 'N/A')}\n")
            f.write(f"  Examples scored: {len(scored)}/{n_examples}\n")
            if scored:
                f.write(f"  Score distribution: mean={np.mean(scored):.4f} std={np.std(scored):.4f} "
                        f"min={min(scored):.4f} max={max(scored):.4f}\n")

        # ---- Final summary ----
        f.write(f"\n{'='*80}\n")
        f.write(f"Final Scores: {model_name}\n")
        f.write(f"{'='*80}\n")
        if cka_result is not None:
            f.write(f"  CKA:        {cka_result.get('score', 'N/A')}\n")
        if ll_result is not None:
            f.write(f"  Logit Lens: {ll_result.get('score', 'N/A')}\n")
        if fisher_result is not None:
            f.write(f"  Fisher (mean): {fisher_result.get('score', 'N/A')}\n")


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
# Per-model evaluation
# ============================================================================

def evaluate_model(
    model_name: str,
    model_id: str,
    full_model,
    tokenizer,
    contexts: List[Dict],
    anchor_cache: Dict,
    methods: List[str],
    batch_size: int = 32,
    fisher_batch_size: int = 4,
    attn_implementation: Optional[str] = None,
    log_dir: Optional[Path] = None,
    delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
) -> Dict:
    """Evaluate a single unlearned model on all requested methods.

    Returns dict with scores for each method.
    """
    result = {"model_name": model_name, "model_id": model_id}

    # Load unlearned model
    print(f"\n[Eval] Loading {model_name}...")
    unl_model = load_model(
        model_id, attn_implementation=attn_implementation
    )

    # Extract hidden states if needed
    need_hidden = "cka" in methods or "logit_lens" in methods
    hidden_unl = None
    if need_hidden:
        print(f"[Eval] Extracting hidden states for {model_name}...")
        hidden_unl = extract_hidden_states(
            unl_model, tokenizer, contexts, batch_size
        )

    # CKA
    cka_result_full = None
    if "cka" in methods:
        print(f"[Eval] Computing CKA for {model_name}...")
        cka_result_full = compute_cka_scores(
            hidden_unl, anchor_cache["hidden_ret"],
            anchor_cache["hidden_full"], anchor_cache
        )
        result["cka"] = cka_result_full["score"]
        result["cka_detail"] = {
            "ft_layers": cka_result_full.get("ft_layers", []),
            "per_layer": {
                l: {k: v for k, v in d.items()}
                for l, d in cka_result_full["per_layer"].items()
            }
        }

    # Logit Lens
    ll_result_full = None
    ll_unl = None
    if "logit_lens" in methods:
        print(f"[Eval] Computing Logit Lens for {model_name}...")
        ll_unl = compute_logit_lens_logprobs(full_model, hidden_unl, contexts)
        ll_result_full = compute_logit_lens_scores(ll_unl, anchor_cache, delta_threshold)
        result["logit_lens"] = ll_result_full["score"]
        result["logit_lens_detail"] = {
            "n_examples_scored": ll_result_full["n_examples_scored"],
            "per_layer": ll_result_full["per_layer"],
        }

    # Fisher (compute once, score with multiple aggregations)
    FISHER_AGG_KEYS = ["mean", "top0.01", "top0.03", "top0.05"]
    fisher_result_full = None
    if "fisher" in methods:
        print(f"[Eval] Computing Fisher for {model_name}...")
        fisher_unl = compute_layer_fisher(
            unl_model, tokenizer, contexts, fisher_batch_size
        )
        # Score with each aggregation variant
        for agg_key in FISHER_AGG_KEYS:
            fisher_res = compute_fisher_scores(fisher_unl, anchor_cache, agg_key=agg_key)
            suffix = "" if agg_key == "mean" else f"_{agg_key}"
            result[f"fisher{suffix}"] = fisher_res["score"]
            if agg_key == "mean":
                fisher_result_full = fisher_res
                result["fisher_detail"] = {
                    "ft_layers": fisher_res.get("ft_layers", []),
                    "per_layer": {
                        l: {k: v for k, v in d.items()}
                        for l, d in fisher_res["per_layer"].items()
                    }
                }

    # Fixed-mask Fisher (uses raw per-parameter vectors)
    if "fisher_masked" in methods and "fisher_mask" in anchor_cache:
        print(f"[Eval] Computing Fisher (masked) for {model_name}...")
        fisher_raw_unl = compute_layer_fisher(
            unl_model, tokenizer, contexts, fisher_batch_size, return_raw=True
        )
        for frac in FISHER_MASK_FRACS:
            masked_res = compute_fisher_masked_scores(
                fisher_raw_unl, anchor_cache["fisher_mask"], frac
            )
            result[f"fisher_masked_{frac}"] = masked_res["score"]
            print(f"  fisher_masked_{frac}: {masked_res['score']}")
        del fisher_raw_unl

    # Write detailed log
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{model_name}.log"
        write_detail_log(
            log_path, model_name, contexts, tokenizer, anchor_cache,
            cka_result=cka_result_full,
            ll_result=ll_result_full,
            ll_unl=ll_unl,
            fisher_result=fisher_result_full,
            delta_threshold=delta_threshold,
        )
        print(f"  Log: {log_path}")

    # Cleanup
    del unl_model
    torch.cuda.empty_cache()

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute representation-level baseline scores"
    )
    parser.add_argument("--mode", choices=["anchor", "eval", "eval_all", "faithfulness", "robustness_relearn", "robustness_quant"],
                        required=True)
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short name of unlearned model (for mode=eval)")
    parser.add_argument("--methods", type=str, default="cka,logit_lens,fisher",
                        help="Comma-separated methods to compute")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fisher_batch_size", type=int, default=4)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--delta_threshold", type=float,
                        default=DEFAULT_DELTA_THRESHOLD)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for model slicing (faithfulness/eval_all)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for model slicing (faithfulness/eval_all)")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Force recompute even if model already in results")
    parser.add_argument("--name_filter", type=str, default=None,
                        help="Only process models whose name contains this string (e.g. '_ep10')")
    parser.add_argument("--clear_cache", action=argparse.BooleanOptionalAction, default=False,
                        help="Clear HF cache for each model after evaluation (saves disk)")
    parser.add_argument("--include_retain", action="store_true",
                        help="Include retain model in eval_all (for robustness analysis)")
    # Robustness relearn args
    parser.add_argument("--relearn_lr", type=float, default=2e-5)
    parser.add_argument("--relearn_epochs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--relearn_batch_size", type=int, default=8,
                        help="Training batch size for relearning (effective = relearn_batch_size * grad_accum)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    methods = [m.strip() for m in args.methods.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Methods: {methods}")
    print(f"Output: {out_dir}")
    print(f"Attention: {args.attn_implementation}")

    # Load tokenizer and data
    print("\n[Setup] Loading tokenizer and data...")
    tokenizer = load_tokenizer(TOFU_FULL_MODEL)
    prefix_data = load_prefix_data(PREFIX_DATA_PATH)
    print(f"  Data: {len(prefix_data)} examples")

    # Prepare contexts
    contexts = prepare_contexts(tokenizer, prefix_data)
    print(f"  Contexts: {len(contexts)} (after entity span filtering)")

    # Load full model (always needed)
    print("\n[Setup] Loading full model...")
    full_model = load_model(
        TOFU_FULL_MODEL, attn_implementation=args.attn_implementation
    )

    if args.mode == "anchor":
        # Compute anchor cache
        print("\n[Setup] Loading retain model...")
        retain_model = load_model(
            TOFU_RETAIN_MODEL, attn_implementation=args.attn_implementation
        )

        # Load existing anchor cache if present (to merge, not overwrite)
        anchor_dir = out_dir / "anchor"
        anchor_dir.mkdir(parents=True, exist_ok=True)
        cache_path = anchor_dir / "anchor_cache.json"
        if cache_path.exists():
            print("[Anchor] Loading existing anchor cache (will merge)...")
            existing_cache = load_anchor_cache(cache_path)
        else:
            existing_cache = {}

        print("\n[Anchor] Computing anchor cache...")
        t0 = time.time()
        anchor_cache = compute_anchor_cache(
            full_model, retain_model, tokenizer, contexts,
            methods, args.batch_size, args.fisher_batch_size,
        )
        elapsed = time.time() - t0
        print(f"\n[Anchor] Done in {elapsed:.1f}s")

        # Merge: new results override existing, but keep unaffected keys
        merged_cache = {**existing_cache, **anchor_cache}

        # Save anchor cache (convert tensors to serializable format)
        save_anchor_cache(merged_cache, cache_path, contexts)
        print(f"[Anchor] Saved to {cache_path}")
        anchor_cache = merged_cache

        # Also save retain model's own scores (should be ~1.0 for all methods)
        print("\n[Anchor] Computing retain model scores (sanity check)...")
        retain_result = {"model_name": "retain", "model_id": TOFU_RETAIN_MODEL}
        if "cka" in methods:
            cka_r = compute_cka_scores(
                anchor_cache["hidden_ret"], anchor_cache["hidden_ret"],
                anchor_cache["hidden_full"], anchor_cache
            )
            retain_result["cka"] = cka_r["score"]
            print(f"  CKA(retain) = {cka_r['score']}")
        if "logit_lens" in methods:
            ll_r = compute_logit_lens_scores(
                anchor_cache["ll_ret"], anchor_cache
            )
            retain_result["logit_lens"] = ll_r["score"]
            print(f"  Logit Lens(retain) = {ll_r['score']}")
        if "fisher" in methods:
            for agg_key in ["mean", "top0.01", "top0.03", "top0.05"]:
                fisher_r = compute_fisher_scores(
                    anchor_cache["fisher_ret"], anchor_cache, agg_key=agg_key
                )
                suffix = "" if agg_key == "mean" else f"_{agg_key}"
                # Fisher(retain vs retain) = 1.0 by definition (all excess = 0)
                retain_result[f"fisher{suffix}"] = 1.0 if fisher_r["score"] is None else fisher_r["score"]
                print(f"  Fisher{suffix}(retain) = {retain_result[f'fisher{suffix}']}")

        if "fisher_masked" in methods and "fisher_mask" in anchor_cache:
            # Retain raw Fisher = anchor fisher_ret. Retain should score 1.0.
            fisher_raw_ret = compute_layer_fisher(
                retain_model, tokenizer, contexts, args.fisher_batch_size, return_raw=True
            )
            for frac in FISHER_MASK_FRACS:
                masked_r = compute_fisher_masked_scores(
                    fisher_raw_ret, anchor_cache["fisher_mask"], frac
                )
                retain_result[f"fisher_masked_{frac}"] = masked_r["score"]
                print(f"  Fisher_masked_{frac}(retain) = {masked_r['score']}")
            del fisher_raw_ret

        # Save retain sanity check
        sanity_path = out_dir / "retain_sanity.json"
        with open(sanity_path, "w") as f:
            json.dump(retain_result, f, indent=2)
        print(f"  Saved to {sanity_path}")

        del retain_model
        torch.cuda.empty_cache()

    elif args.mode == "eval":
        if args.model_name is None:
            parser.error("--model_name required for mode=eval")

        # Load anchor cache
        anchor_cache = load_anchor_cache(out_dir / "anchor" / "anchor_cache.json")

        model_id = get_model_id(args.model_name)
        log_dir = out_dir / "logs"
        result = evaluate_model(
            args.model_name, model_id, full_model, tokenizer,
            contexts, anchor_cache, methods,
            args.batch_size, args.fisher_batch_size,
            args.attn_implementation,
            log_dir=log_dir,
            delta_threshold=args.delta_threshold,
        )

        # Save result
        result_path = out_dir / f"results_{args.model_name}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=_json_default)
        print(f"\n[Result] {args.model_name}:")
        for m in methods:
            if m in result:
                print(f"  {m}: {result[m]}")
        print(f"  Saved to {result_path}")

    elif args.mode == "eval_all":
        from scripts.meta_eval_robustness import DEFAULT_MODELS

        def _method_base(key):
            """Map result key to method directory name."""
            if key.startswith("fisher_masked"):
                return "fisher_masked"
            if key.startswith("fisher"):
                return "fisher"
            return key

        # Load anchor cache
        anchor_cache = load_anchor_cache(out_dir / "anchor" / "anchor_cache.json")

        # Per-method results (each method gets its own dir + results.json)
        method_data = {}  # method_base -> {model_name: {key: value}}
        for m in methods:
            mdir = out_dir / m
            mdir.mkdir(parents=True, exist_ok=True)
            rpath = mdir / "results.json"
            method_data[m] = json.load(open(rpath)) if rpath.exists() else {}

        model_items = [
            (name, mid) for name, mid in DEFAULT_MODELS.items()
            if name != "retain" or args.include_retain
        ]
        if args.name_filter:
            model_items = [(n, m) for n, m in model_items
                           if args.name_filter in n or (n == "retain" and args.include_retain)]
        end_idx = args.end_idx if args.end_idx is not None else len(model_items)
        model_items = model_items[args.start_idx:end_idx]
        print(f"\n[Eval All] {len(model_items)} models (clear_cache={args.clear_cache})")

        for i, (name, mid) in enumerate(model_items):
            if not args.force_recompute:
                # Per-method skip: only skip if ALL requested methods already present
                if all(name in method_data.get(m, {}) for m in methods):
                    print(f"[{i+1}/{len(model_items)}] {name} - already computed, skipping")
                    continue

            print(f"\n[{i+1}/{len(model_items)}] {name}")
            t0 = time.time()
            try:
                # Retain model: compute from anchor cache (no download needed)
                if name == "retain":
                    result = {"model_name": "retain", "model_id": mid}
                    if "logit_lens" in methods:
                        ll_r = compute_logit_lens_scores(
                            anchor_cache["ll_ret"], anchor_cache, args.delta_threshold
                        )
                        result["logit_lens"] = ll_r["score"]
                    if "cka" in methods:
                        cka_r = compute_cka_scores(
                            anchor_cache["hidden_ret"], anchor_cache["hidden_ret"],
                            anchor_cache["hidden_full"], anchor_cache
                        )
                        result["cka"] = cka_r["score"]
                    if "fisher" in methods:
                        for agg_key in ["mean", "top0.01", "top0.03", "top0.05"]:
                            fisher_r = compute_fisher_scores(
                                anchor_cache["fisher_ret"], anchor_cache, agg_key=agg_key
                            )
                            suffix = "" if agg_key == "mean" else f"_{agg_key}"
                            result[f"fisher{suffix}"] = 1.0 if fisher_r["score"] is None else fisher_r["score"]
                    if "fisher_masked" in methods:
                        # Retain = anchor, so excess_unl = 0, erasure = 1.0
                        for frac in FISHER_MASK_FRACS:
                            result[f"fisher_masked_{frac}"] = 1.0
                else:
                    log_dir = out_dir / methods[0] / "logs"
                    result = evaluate_model(
                        name, mid, full_model, tokenizer,
                        contexts, anchor_cache, methods,
                        args.batch_size, args.fisher_batch_size,
                        args.attn_implementation,
                        log_dir=log_dir,
                        delta_threshold=args.delta_threshold,
                    )
                fisher_variants = [f"fisher_{k}" for k in ["top0.01", "top0.03", "top0.05"]]
                fisher_masked_variants = [f"fisher_masked_{f}" for f in FISHER_MASK_FRACS]
                all_keys = list(methods) + fisher_variants + fisher_masked_variants
                # Save per-method results
                for key in all_keys:
                    if result.get(key) is not None:
                        mbase = _method_base(key)
                        if mbase not in method_data:
                            method_data[mbase] = {}
                            (out_dir / mbase).mkdir(parents=True, exist_ok=True)
                        if name not in method_data[mbase]:
                            method_data[mbase][name] = {}
                        method_data[mbase][name][key] = result[key]
                # Save after each model (all affected method files)
                for mbase in method_data:
                    rpath = out_dir / mbase / "results.json"
                    with open(rpath, "w") as f:
                        json.dump(method_data[mbase], f, indent=2, default=_json_default)
                elapsed = time.time() - t0
                scores = " | ".join(
                    f"{key}={result.get(key, 'N/A'):.4f}"
                    if isinstance(result.get(key), (int, float)) else f"{key}=N/A"
                    for key in all_keys if result.get(key) is not None
                )
                print(f"  {scores} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

            # Clear HF cache for this model
            if args.clear_cache and mid != TOFU_FULL_MODEL and mid != TOFU_RETAIN_MODEL:
                clear_hf_cache(mid)
            free_memory()

    elif args.mode == "faithfulness":
        from sklearn.metrics import roc_auc_score

        # Load anchor cache
        anchor_cache = load_anchor_cache(out_dir / "anchor" / "anchor_cache.json")

        faith_dir = Path("runs/meta_eval/faithfulness")
        faith_dir.mkdir(parents=True, exist_ok=True)

        # Results file (append-safe, keyed by model_id)
        results_path = faith_dir / "rep_baselines_results.json"
        existing = {}
        if results_path.exists():
            with open(results_path) as f:
                existing = json.load(f)

        # Slice models for parallel execution
        all_models = FAITH_ALL_MODELS  # [(model_id, pool_label), ...]
        end_idx = args.end_idx if args.end_idx is not None else len(all_models)
        model_slice = all_models[args.start_idx:end_idx]
        print(f"\n[Faithfulness] {len(model_slice)} models (idx {args.start_idx}:{end_idx} of {len(all_models)})")

        for i, (model_id, pool) in enumerate(model_slice):
            global_idx = args.start_idx + i
            short_name = model_id.split("/")[-1]

            if model_id in existing and not args.force_recompute:
                if all(m in existing[model_id] for m in methods):
                    print(f"[{global_idx+1}/{len(all_models)}] {short_name} ({pool}) - already computed, skipping")
                    continue

            print(f"\n[{global_idx+1}/{len(all_models)}] {short_name} ({pool})")
            t0 = time.time()
            try:
                log_dir = faith_dir / "logs"
                result = evaluate_model(
                    short_name, model_id, full_model, tokenizer,
                    contexts, anchor_cache, methods,
                    args.batch_size, args.fisher_batch_size,
                    args.attn_implementation,
                    log_dir=log_dir,
                    delta_threshold=args.delta_threshold,
                )
                # Collect all result keys (methods + fisher variants + fisher_masked)
                fisher_variants = [f"fisher_{k}" for k in ["top0.01", "top0.03", "top0.05"]]
                fisher_masked_variants = [f"fisher_masked_{f}" for f in FISHER_MASK_FRACS]
                all_keys = list(methods) + fisher_variants + fisher_masked_variants
                if model_id not in existing:
                    existing[model_id] = {}
                existing[model_id]["pool"] = pool
                for m in all_keys:
                    if result.get(m) is not None:
                        existing[model_id][m] = result[m]
                # Save after each model
                with open(results_path, "w") as f:
                    json.dump(existing, f, indent=2, default=_json_default)
                elapsed = time.time() - t0
                scores = " | ".join(
                    f"{m}={result.get(m, 'N/A'):.4f}"
                    if isinstance(result.get(m), (int, float)) else f"{m}=N/A"
                    for m in all_keys if result.get(m) is not None
                )
                print(f"  {scores} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Compute AUC-ROC if all models are done
        all_done = all(mid in existing for mid, _ in FAITH_ALL_MODELS)
        if all_done:
            print(f"\n[Faithfulness] All 60 models done. Computing AUC-ROC...")
            # Collect ALL method keys present in results (not just --methods)
            all_method_keys = set()
            for entry in existing.values():
                all_method_keys.update(k for k in entry if k != 'pool')
            all_method_keys = sorted(all_method_keys)
            summary = compute_faithfulness_aucroc(existing, all_method_keys)
            # Merge with existing summary (preserve previous results)
            summary_path = faith_dir / "rep_baselines_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    old_summary = json.load(f)
                old_summary.update(summary)
                summary = old_summary
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved to {summary_path}")
            for m in all_method_keys:
                if m in summary and summary[m].get("auc_roc") is not None:
                    print(f"  {m} AUC-ROC: {summary[m]['auc_roc']:.4f}")
        else:
            n_done = sum(1 for mid, _ in FAITH_ALL_MODELS if mid in existing)
            print(f"\n[Faithfulness] {n_done}/60 models done. Run on other GPU to complete, then re-run for AUC-ROC.")

    elif args.mode == "robustness_relearn":
        import shutil
        from scripts.meta_eval_robustness import DEFAULT_MODELS, finetune_in_subprocess

        # Load anchor cache
        anchor_cache = load_anchor_cache(out_dir / "anchor" / "anchor_cache.json")

        # Load precomputed metrics_before per method (from eval_all results)
        eval_data = {}  # method -> {model_name: {key: value}}
        for m in methods:
            rpath = out_dir / m / "results.json"
            eval_data[m] = json.load(open(rpath)) if rpath.exists() else {}

        # Determine after-keys for skip logic
        after_keys = []
        for m in methods:
            if m == "logit_lens":
                after_keys.append("logit_lens_after")
            elif m == "fisher_masked":
                after_keys.extend([f"fisher_masked_{f}_after" for f in FISHER_MASK_FRACS])
            elif m == "cka":
                after_keys.append("cka_after")

        # Robustness results file (stored alongside standard robustness results)
        rob_dir = Path("runs/meta_eval/robustness/relearn")
        rob_dir.mkdir(parents=True, exist_ok=True)
        rob_results_path = rob_dir / "rep_baselines_results.json"
        rob_results = {}
        if rob_results_path.exists():
            rob_results = json.load(open(rob_results_path))

        # Build model list (retain included, for R formula numerator)
        model_items = list(DEFAULT_MODELS.items())
        if args.name_filter:
            model_items = [(n, m) for n, m in model_items
                           if args.name_filter in n or n == "retain"]
        end_idx = args.end_idx if args.end_idx is not None else len(model_items)
        model_items = model_items[args.start_idx:end_idx]
        print(f"\n[Robustness Relearn] {len(model_items)} models, methods={methods} (clear_cache={args.clear_cache})")

        for i, (name, mid) in enumerate(model_items):
            # Skip if all after keys present
            if name in rob_results and not args.force_recompute:
                if all(k in rob_results[name] for k in after_keys):
                    print(f"[{i+1}/{len(model_items)}] {name} - already done, skipping")
                    continue

            print(f"\n[{i+1}/{len(model_items)}] {name}")
            t0 = time.time()
            try:
                # ---- metrics_before ----
                before = {}
                need_original = False

                if "logit_lens" in methods:
                    val = eval_data.get("logit_lens", {}).get(name, {}).get("logit_lens")
                    if val is not None:
                        before["logit_lens"] = val
                    else:
                        need_original = True

                if "fisher_masked" in methods:
                    found_all = True
                    for f in FISHER_MASK_FRACS:
                        val = eval_data.get("fisher_masked", {}).get(name, {}).get(f"fisher_masked_{f}")
                        if val is not None:
                            before[f"fisher_masked_{f}"] = val
                        else:
                            found_all = False
                    if not found_all:
                        need_original = True

                if "cka" in methods:
                    val = eval_data.get("cka", {}).get(name, {}).get("cka")
                    if val is not None:
                        before["cka"] = val
                    else:
                        need_original = True

                # Compute missing before-values on-the-fly
                if need_original:
                    print(f"  Computing metrics_before on-the-fly...")
                    unl_model = load_model(mid, attn_implementation=args.attn_implementation)

                    if "logit_lens" in methods and "logit_lens" not in before:
                        hidden_tmp = extract_hidden_states(unl_model, tokenizer, contexts, args.batch_size)
                        ll_tmp = compute_logit_lens_logprobs(full_model, hidden_tmp, contexts)
                        ll_res = compute_logit_lens_scores(ll_tmp, anchor_cache, args.delta_threshold)
                        before["logit_lens"] = ll_res["score"]
                        del hidden_tmp, ll_tmp

                    if "fisher_masked" in methods and f"fisher_masked_{FISHER_MASK_FRACS[0]}" not in before:
                        fisher_raw = compute_layer_fisher(
                            unl_model, tokenizer, contexts, args.fisher_batch_size, return_raw=True
                        )
                        for f in FISHER_MASK_FRACS:
                            res = compute_fisher_masked_scores(fisher_raw, anchor_cache["fisher_mask"], f)
                            before[f"fisher_masked_{f}"] = res["score"]
                        del fisher_raw

                    if "cka" in methods and "cka" not in before:
                        hidden_tmp = extract_hidden_states(unl_model, tokenizer, contexts, args.batch_size)
                        cka_res = compute_cka_scores(
                            hidden_tmp, anchor_cache["hidden_ret"],
                            anchor_cache["hidden_full"], anchor_cache
                        )
                        before["cka"] = cka_res["score"]
                        del hidden_tmp

                    # Also save to eval_all results (side-effect)
                    for m in methods:
                        mdir = out_dir / m
                        mdir.mkdir(parents=True, exist_ok=True)
                        rpath = mdir / "results.json"
                        mdata = eval_data.get(m, {})
                        if name not in mdata:
                            mdata[name] = {}
                        for key, val in before.items():
                            if key.startswith(m) or key == m:
                                mdata[name][key] = val
                        eval_data[m] = mdata
                        with open(rpath, "w") as f:
                            json.dump(mdata, f, indent=2, default=_json_default)

                    del unl_model
                    free_memory()

                # ---- Relearn ----
                ft_dir = rob_dir / f"ft_{name}"
                print(f"  Finetuning on forget10 (subprocess, lr={args.relearn_lr}, ep={args.relearn_epochs})...")
                ft_ok = finetune_in_subprocess(
                    mid, str(ft_dir), gpu=args.gpu,
                    lr=args.relearn_lr, epochs=args.relearn_epochs,
                    batch_size=args.relearn_batch_size, grad_accum=args.grad_accum,
                    attn_implementation=args.attn_implementation,
                )
                if not ft_ok:
                    raise RuntimeError("Subprocess finetuning failed")

                # ---- metrics_after ----
                print(f"  Loading relearned model...")
                ft_model = load_model(str(ft_dir), attn_implementation=args.attn_implementation)
                after = {}

                if "logit_lens" in methods:
                    print(f"  Computing Logit Lens (relearned)...")
                    hidden_ft = extract_hidden_states(ft_model, tokenizer, contexts, args.batch_size)
                    ll_ft = compute_logit_lens_logprobs(full_model, hidden_ft, contexts)
                    ll_res = compute_logit_lens_scores(ll_ft, anchor_cache, args.delta_threshold)
                    after["logit_lens"] = ll_res["score"]
                    del hidden_ft, ll_ft

                if "fisher_masked" in methods:
                    print(f"  Computing Fisher Masked (relearned)...")
                    fisher_raw_ft = compute_layer_fisher(
                        ft_model, tokenizer, contexts, args.fisher_batch_size, return_raw=True
                    )
                    for f in FISHER_MASK_FRACS:
                        res = compute_fisher_masked_scores(fisher_raw_ft, anchor_cache["fisher_mask"], f)
                        after[f"fisher_masked_{f}"] = res["score"]
                    del fisher_raw_ft

                if "cka" in methods:
                    print(f"  Computing CKA (relearned)...")
                    hidden_ft = extract_hidden_states(ft_model, tokenizer, contexts, args.batch_size)
                    cka_res = compute_cka_scores(
                        hidden_ft, anchor_cache["hidden_ret"],
                        anchor_cache["hidden_full"], anchor_cache
                    )
                    after["cka"] = cka_res["score"]
                    del hidden_ft

                del ft_model
                free_memory()

                # Clean up finetuned checkpoint
                shutil.rmtree(ft_dir, ignore_errors=True)

                # Save result
                if name not in rob_results:
                    rob_results[name] = {}
                rob_results[name].pop("error", None)
                rob_results[name]["model_id"] = mid
                for key, val in before.items():
                    rob_results[name][f"{key}_before"] = val
                for key, val in after.items():
                    rob_results[name][f"{key}_after"] = val

                elapsed = time.time() - t0
                summary_parts = []
                for key in sorted(before.keys()):
                    b = before[key]
                    a = after.get(key, None)
                    if a is not None:
                        summary_parts.append(f"{key}: {b:.4f}->{a:.4f}")
                print(f"  {' | '.join(summary_parts)} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                if name not in rob_results:
                    rob_results[name] = {}
                rob_results[name]["error"] = str(e)

            # Save after each model
            with open(rob_results_path, "w") as f:
                json.dump(rob_results, f, indent=2)

            # Clear HF cache
            if args.clear_cache and mid != TOFU_FULL_MODEL and mid != TOFU_RETAIN_MODEL:
                clear_hf_cache(mid)
            free_memory()

        print(f"\n{'='*60}")
        print(f"Done! Results saved to {rob_results_path}")
        print(f"{'='*60}")

    elif args.mode == "robustness_quant":
        from scripts.meta_eval_robustness import DEFAULT_MODELS, load_model_quantized

        # Load anchor cache
        anchor_cache = load_anchor_cache(out_dir / "anchor" / "anchor_cache.json")

        # Load precomputed metrics_before per method (from eval_all results)
        eval_data = {}
        for m in methods:
            rpath = out_dir / m / "results.json"
            eval_data[m] = json.load(open(rpath)) if rpath.exists() else {}

        # Determine after-keys for skip logic
        after_keys = []
        for m in methods:
            if m == "logit_lens":
                after_keys.append("logit_lens_after")
            elif m == "fisher_masked":
                after_keys.extend([f"fisher_masked_{f}_after" for f in FISHER_MASK_FRACS])
            elif m == "cka":
                after_keys.append("cka_after")

        # Robustness results file (stored alongside standard robustness results)
        rob_dir = Path("runs/meta_eval/robustness/quant")
        rob_dir.mkdir(parents=True, exist_ok=True)
        rob_results_path = rob_dir / "rep_baselines_results.json"
        rob_results = {}
        if rob_results_path.exists():
            rob_results = json.load(open(rob_results_path))

        # Build model list
        model_items = list(DEFAULT_MODELS.items())
        if args.name_filter:
            model_items = [(n, m) for n, m in model_items
                           if args.name_filter in n or n == "retain"]
        end_idx = args.end_idx if args.end_idx is not None else len(model_items)
        model_items = model_items[args.start_idx:end_idx]
        print(f"\n[Robustness Quant] {len(model_items)} models, methods={methods} (clear_cache={args.clear_cache})")

        for i, (name, mid) in enumerate(model_items):
            # Skip if all after keys present
            if name in rob_results and not args.force_recompute:
                if all(k in rob_results[name] for k in after_keys):
                    print(f"[{i+1}/{len(model_items)}] {name} - already done, skipping")
                    continue

            print(f"\n[{i+1}/{len(model_items)}] {name}")
            t0 = time.time()
            try:
                # ---- metrics_before (from eval_all) ----
                before = {}
                need_original = False

                if "logit_lens" in methods:
                    val = eval_data.get("logit_lens", {}).get(name, {}).get("logit_lens")
                    if val is not None:
                        before["logit_lens"] = val
                    else:
                        need_original = True

                if "fisher_masked" in methods:
                    found_all = True
                    for f in FISHER_MASK_FRACS:
                        val = eval_data.get("fisher_masked", {}).get(name, {}).get(f"fisher_masked_{f}")
                        if val is not None:
                            before[f"fisher_masked_{f}"] = val
                        else:
                            found_all = False
                    if not found_all:
                        need_original = True

                if "cka" in methods:
                    val = eval_data.get("cka", {}).get(name, {}).get("cka")
                    if val is not None:
                        before["cka"] = val
                    else:
                        need_original = True

                # Compute missing before-values on-the-fly
                if need_original:
                    print(f"  Computing metrics_before on-the-fly...")
                    unl_model = load_model(mid, attn_implementation=args.attn_implementation)

                    if "logit_lens" in methods and "logit_lens" not in before:
                        hidden_tmp = extract_hidden_states(unl_model, tokenizer, contexts, args.batch_size)
                        ll_tmp = compute_logit_lens_logprobs(full_model, hidden_tmp, contexts)
                        ll_res = compute_logit_lens_scores(ll_tmp, anchor_cache, args.delta_threshold)
                        before["logit_lens"] = ll_res["score"]
                        del hidden_tmp, ll_tmp

                    if "fisher_masked" in methods and f"fisher_masked_{FISHER_MASK_FRACS[0]}" not in before:
                        fisher_raw = compute_layer_fisher(
                            unl_model, tokenizer, contexts, args.fisher_batch_size, return_raw=True
                        )
                        for f in FISHER_MASK_FRACS:
                            res = compute_fisher_masked_scores(fisher_raw, anchor_cache["fisher_mask"], f)
                            before[f"fisher_masked_{f}"] = res["score"]
                        del fisher_raw

                    if "cka" in methods and "cka" not in before:
                        hidden_tmp = extract_hidden_states(unl_model, tokenizer, contexts, args.batch_size)
                        cka_res = compute_cka_scores(
                            hidden_tmp, anchor_cache["hidden_ret"],
                            anchor_cache["hidden_full"], anchor_cache
                        )
                        before["cka"] = cka_res["score"]
                        del hidden_tmp

                    # Save to eval_all results (side-effect)
                    for m in methods:
                        mdir = out_dir / m
                        mdir.mkdir(parents=True, exist_ok=True)
                        rpath = mdir / "results.json"
                        mdata = eval_data.get(m, {})
                        if name not in mdata:
                            mdata[name] = {}
                        for key, val in before.items():
                            if key.startswith(m) or key == m:
                                mdata[name][key] = val
                        eval_data[m] = mdata
                        with open(rpath, "w") as f:
                            json.dump(mdata, f, indent=2, default=_json_default)

                    del unl_model
                    free_memory()

                # ---- Load quantized model ----
                print(f"  Loading quantized model (NF4)...")
                quant_model = load_model_quantized(mid)
                after = {}

                if "logit_lens" in methods:
                    print(f"  Computing Logit Lens (quantized)...")
                    hidden_q = extract_hidden_states(quant_model, tokenizer, contexts, args.batch_size)
                    ll_q = compute_logit_lens_logprobs(full_model, hidden_q, contexts)
                    ll_res = compute_logit_lens_scores(ll_q, anchor_cache, args.delta_threshold)
                    after["logit_lens"] = ll_res["score"]
                    del hidden_q, ll_q

                if "fisher_masked" in methods:
                    print(f"  Computing Fisher Masked (quantized)...")
                    fisher_raw_q = compute_layer_fisher(
                        quant_model, tokenizer, contexts, args.fisher_batch_size, return_raw=True
                    )
                    for f in FISHER_MASK_FRACS:
                        res = compute_fisher_masked_scores(fisher_raw_q, anchor_cache["fisher_mask"], f)
                        after[f"fisher_masked_{f}"] = res["score"]
                    del fisher_raw_q

                if "cka" in methods:
                    print(f"  Computing CKA (quantized)...")
                    hidden_q = extract_hidden_states(quant_model, tokenizer, contexts, args.batch_size)
                    cka_res = compute_cka_scores(
                        hidden_q, anchor_cache["hidden_ret"],
                        anchor_cache["hidden_full"], anchor_cache
                    )
                    after["cka"] = cka_res["score"]
                    del hidden_q

                del quant_model
                free_memory()

                # Save result
                if name not in rob_results:
                    rob_results[name] = {}
                rob_results[name].pop("error", None)
                rob_results[name]["model_id"] = mid
                for key, val in before.items():
                    rob_results[name][f"{key}_before"] = val
                for key, val in after.items():
                    rob_results[name][f"{key}_after"] = val

                elapsed = time.time() - t0
                summary_parts = []
                for key in sorted(before.keys()):
                    b = before[key]
                    a = after.get(key, None)
                    if a is not None:
                        summary_parts.append(f"{key}: {b:.4f}->{a:.4f}")
                print(f"  {' | '.join(summary_parts)} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                if name not in rob_results:
                    rob_results[name] = {}
                rob_results[name]["error"] = str(e)

            # Save after each model
            with open(rob_results_path, "w") as f:
                json.dump(rob_results, f, indent=2)

            # Clear HF cache
            if args.clear_cache and mid != TOFU_FULL_MODEL and mid != TOFU_RETAIN_MODEL:
                clear_hf_cache(mid)
            free_memory()

        print(f"\n{'='*60}")
        print(f"Done! Results saved to {rob_results_path}")
        print(f"{'='*60}")


def compute_faithfulness_aucroc(results: Dict, methods: List[str]) -> Dict:
    """Compute AUC-ROC for faithfulness (P vs N pool separation)."""
    from sklearn.metrics import roc_auc_score
    summary = {}

    for m in methods:
        p_scores = []
        n_scores = []
        for model_id, info in results.items():
            score = info.get(m)
            if score is None:
                continue
            if info["pool"] == "P":
                p_scores.append(score)
            else:
                n_scores.append(score)

        if not p_scores or not n_scores:
            summary[m] = {"auc_roc": None, "n_P": len(p_scores), "n_N": len(n_scores)}
            continue

        # Labels: P=0 (has knowledge, should score LOW = 0),
        #         N=1 (no knowledge, should score HIGH = 1)
        # Score direction: higher = more erasure (retain-like)
        # So N pool should have higher scores than P pool
        labels = [0] * len(p_scores) + [1] * len(n_scores)
        scores = p_scores + n_scores

        auc = roc_auc_score(labels, scores)

        summary[m] = {
            "auc_roc": float(auc),
            "n_P": len(p_scores),
            "n_N": len(n_scores),
            "P_mean": float(np.mean(p_scores)),
            "P_std": float(np.std(p_scores)),
            "N_mean": float(np.mean(n_scores)),
            "N_std": float(np.std(n_scores)),
        }

    return summary


# ============================================================================
# Cache serialization
# ============================================================================

def save_anchor_cache(cache: Dict, path: Path, contexts: List[Dict]):
    """Save anchor cache to JSON (convert tensors to lists)."""
    serializable = {}

    # CKA anchor
    if "cka_full_ret" in cache:
        serializable["cka_full_ret"] = {
            str(k): v for k, v in cache["cka_full_ret"].items()
        }

    # Logit Lens anchor
    if "ll_full" in cache:
        serializable["ll_full"] = {
            str(k): v for k, v in cache["ll_full"].items()
        }
    if "ll_ret" in cache:
        serializable["ll_ret"] = {
            str(k): v for k, v in cache["ll_ret"].items()
        }

    # Fisher anchor
    if "fisher_full" in cache:
        serializable["fisher_full"] = {
            str(k): v for k, v in cache["fisher_full"].items()
        }
    if "fisher_ret" in cache:
        serializable["fisher_ret"] = {
            str(k): v for k, v in cache["fisher_ret"].items()
        }

    # Hidden states (save as separate file, too large for JSON)
    if "hidden_full" in cache or "hidden_ret" in cache:
        hidden_dir = path.parent / "hidden_cache"
        hidden_dir.mkdir(exist_ok=True)
        for key in ["hidden_full", "hidden_ret"]:
            if key in cache:
                h_path = hidden_dir / f"{key}.pt"
                # Convert list of tensors to stacked tensors per layer
                to_save = {}
                for l, tensors in cache[key].items():
                    to_save[l] = tensors  # list of variable-length tensors
                torch.save(to_save, h_path)
                print(f"  Saved {key} -> {h_path}")

    # Fisher mask (save as separate .pt file)
    if "fisher_mask" in cache:
        mask_path = path.parent / "fisher_mask.pt"
        save_fisher_mask(cache["fisher_mask"], mask_path)

    serializable["n_contexts"] = len(contexts)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_anchor_cache(path: Path) -> Dict:
    """Load anchor cache from JSON + hidden state files."""
    with open(path) as f:
        data = json.load(f)

    cache = {}

    # CKA anchor
    if "cka_full_ret" in data:
        cache["cka_full_ret"] = {
            int(k): v for k, v in data["cka_full_ret"].items()
        }

    # Logit Lens anchor
    for key in ["ll_full", "ll_ret"]:
        if key in data:
            cache[key] = {
                int(k): v for k, v in data[key].items()
            }

    # Fisher anchor
    for key in ["fisher_full", "fisher_ret"]:
        if key in data:
            cache[key] = {
                int(k): v for k, v in data[key].items()
            }

    # Hidden states
    hidden_dir = path.parent / "hidden_cache"
    for key in ["hidden_full", "hidden_ret"]:
        h_path = hidden_dir / f"{key}.pt"
        if h_path.exists():
            cache[key] = torch.load(h_path, weights_only=False)
            print(f"  Loaded {key} from {h_path}")

    # Fisher mask
    mask_path = path.parent / "fisher_mask.pt"
    if mask_path.exists():
        cache["fisher_mask"] = load_fisher_mask(mask_path)
        print(f"  Loaded Fisher mask from {mask_path}")

    return cache


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()

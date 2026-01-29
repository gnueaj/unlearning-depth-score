#!/usr/bin/env python3
"""
S1/S2 Experiment with Teacher Forcing (Retain → Full, Unlearn → Full)

Metrics:
- EM (token accuracy) or log-prob score on reference tokens
- Reference = Full baseline generation (not GT)

Flow:
1. Get Full baseline generation (reference)
2. For each layer, compute metric with patching:
   - Input: prompt + reference tokens
   - At each reference token position: patch source hidden → predict next token
   - Metric = EM or mean log-prob on reference span

Output: Side-by-side S1 vs S2 comparison per example + UDR
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patchscope.models import load_model, load_tokenizer, get_num_layers
from patchscope.core import generate_baseline
from patchscope.utils import set_seed, safe_mkdir, parse_layers
from patchscope.config import get_model_id


TOFU_FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
TOFU_RETAIN_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
PREFIX_DATA_PATH = "tofu_data/forget10_filtered_v6.json"


def load_prefix_data() -> List[Dict]:
    """Load validated prefix+entity data."""
    with open(PREFIX_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_generated(text: str) -> str:
    """Clean generated text for comparison."""
    text = text.strip()
    if "." in text:
        text = text[:text.index(".")]
    text = text.split("\n")[0].strip()
    return text


def compute_em_score_legacy(generated: str, reference: str, tokenizer) -> float:
    """Legacy EM score: position-wise token match ratio on generated text."""
    gen_clean = clean_generated(generated)
    ref_clean = clean_generated(reference)

    if not gen_clean or not ref_clean:
        return 0.0

    gen_tokens = tokenizer.encode(gen_clean, add_special_tokens=False)
    ref_tokens = tokenizer.encode(ref_clean, add_special_tokens=False)

    if len(ref_tokens) == 0:
        return 0.0

    match_count = sum(1 for g, r in zip(gen_tokens, ref_tokens) if g == r)
    return match_count / len(ref_tokens)


def clean_text(s: str, max_len: int = 200) -> str:
    """Clean generated text for display.

    Note: max_len increased to 200 to show full baseline outputs in logs.
    """
    if "." in s:
        s = s[:s.index(".") + 1]
    s = s.split("\n")[0].strip()
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def find_subsequence(haystack, needle):
    """Return start index of needle in haystack, or None if not found."""
    if not needle or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


def get_eval_span(tokenizer, reference: str, entity: str, scope: str):
    """Get evaluation span within reference tokens."""
    if not reference:
        return None
    if scope == "full":
        ref_ids = tokenizer.encode(reference, add_special_tokens=False)
        return (0, len(ref_ids)) if ref_ids else None
    if not entity:
        return None

    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(reference, add_special_tokens=False, return_offsets_mapping=True)
        ref_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        if not ref_ids:
            return None

        candidate = entity
        char_start = reference.find(candidate)
        if char_start < 0:
            stripped = entity.strip()
            if stripped and stripped != entity:
                candidate = stripped
                char_start = reference.find(candidate)
        if char_start < 0 and entity and not entity.startswith(" "):
            candidate = " " + entity
            char_start = reference.find(candidate)
        if char_start < 0:
            return None

        char_end = char_start + len(candidate)
        token_indices = [
            i for i, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end
        ]
        if not token_indices:
            return None
        return (min(token_indices), max(token_indices) + 1)

    ref_ids = tokenizer.encode(reference, add_special_tokens=False)
    if not ref_ids:
        return None
    ent_ids = tokenizer.encode(entity, add_special_tokens=False)
    start = find_subsequence(ref_ids, ent_ids)
    if start is None and entity and not entity.startswith(" "):
        ent_ids = tokenizer.encode(" " + entity, add_special_tokens=False)
        start = find_subsequence(ref_ids, ent_ids)
    if start is None:
        return None
    return (start, start + len(ent_ids))


def extract_entity_from_full(reference: str) -> str:
    """Heuristic entity span from Full baseline generation."""
    text = clean_generated(reference)
    if not text:
        return ""
    for sep in [",", ";", " - ", " (", " who ", " which ", " that "]:
        if sep in text:
            text = text.split(sep)[0].strip()
            break
    return text


def normalize_reference_for_eval(prompt: str, reference: str) -> str:
    """Ensure reference tokenization matches generation (restore leading space if needed)."""
    if not reference:
        return reference
    if reference[0].isspace():
        return reference
    if prompt and not prompt[-1].isspace():
        return " " + reference
    return reference


def format_mismatches(tokenizer, mismatches, max_items=5) -> str:
    """Format mismatch token pairs for logging."""
    parts = []
    for idx, pred_id, label_id in mismatches[:max_items]:
        pred_tok = tokenizer.decode([pred_id], skip_special_tokens=True)
        label_tok = tokenizer.decode([label_id], skip_special_tokens=True)
        if pred_tok == "":
            pred_tok = tokenizer.convert_ids_to_tokens([pred_id])[0].replace("Ġ", " ")
        if label_tok == "":
            label_tok = tokenizer.convert_ids_to_tokens([label_id])[0].replace("Ġ", " ")
        parts.append(f"{idx}:{pred_tok!r}->{label_tok!r}")
    if len(mismatches) > max_items:
        parts.append(f"+{len(mismatches)-max_items} more")
    return ", ".join(parts)


def compute_em_teacher_forcing_baseline(
    model,
    tokenizer,
    prompt: str,
    reference: str,
    eval_span=None,
    exact_match: bool = False
) -> float:
    """
    Compute teacher forcing EM without patching.

    Uses prompt (with special tokens) + reference tokens, then measures
    token-level accuracy on the reference positions.
    """
    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    ref_ids = tokenizer.encode(reference, add_special_tokens=False)

    if len(ref_ids) == 0:
        return 0.0

    full_ids = prompt_ids + ref_ids
    input_ids = torch.tensor([full_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    prompt_len = len(prompt_ids)
    start = max(prompt_len - 1, 0)
    end = start + len(ref_ids)

    if eval_span is not None:
        span_start, span_end = eval_span
        start = start + span_start
        end = start + (span_end - span_start)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    preds = logits[:, start:end, :].argmax(dim=-1)
    if eval_span is None:
        labels = torch.tensor([ref_ids], device=device)
    else:
        labels = torch.tensor([ref_ids[span_start:span_end]], device=device)

    matches = (preds == labels)
    if exact_match:
        return 1.0 if matches.all() else 0.0
    correct = matches.float().sum().item()
    return correct / labels.shape[1]


def _gather_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return log-prob for each label token. logits: [B, T, V], labels: [B, T]."""
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def compute_logprob_teacher_forcing_baseline(
    model,
    tokenizer,
    prompt: str,
    reference: str,
    eval_span=None
) -> float:
    """
    Compute mean log-prob of reference tokens under teacher forcing (no patching).
    """
    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    ref_ids = tokenizer.encode(reference, add_special_tokens=False)
    if len(ref_ids) == 0:
        return float("-inf")

    full_ids = prompt_ids + ref_ids
    input_ids = torch.tensor([full_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    prompt_len = len(prompt_ids)
    start = max(prompt_len - 1, 0)
    end = start + len(ref_ids)

    if eval_span is not None:
        span_start, span_end = eval_span
        start = start + span_start
        end = start + (span_end - span_start)
        ref_ids = ref_ids[span_start:span_end]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, start:end, :]
    labels = torch.tensor([ref_ids], device=device)
    token_logprobs = _gather_token_logprobs(logits, labels)
    return token_logprobs.mean().item()


def compute_em_teacher_forcing_layer(
    source_model,
    target_model,
    tokenizer,
    prompt: str,
    reference: str,
    layer: int,
    eval_span=None,
    exact_match: bool = False,
    return_details: bool = False,
    patch_scope: str = "span"
) -> float:
    """
    Compute EM with teacher forcing and layer patching.

    Input: prompt + reference tokens (with special tokens, matching baseline generation)
    At each reference token position: patch source hidden → predict next token
    EM = (#correct predictions) / (#reference tokens)
    """
    device = next(target_model.parameters()).device

    # Tokenize prompt with special tokens (matching generate_baseline)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    # Reference tokens without special tokens (just the answer part)
    ref_ids = tokenizer.encode(reference, add_special_tokens=False)

    if len(ref_ids) == 0:
        return 0.0

    # Full input: prompt (with BOS) + reference
    full_ids = prompt_ids + ref_ids
    input_ids = torch.tensor([full_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    prompt_len = len(prompt_ids)
    start = max(prompt_len - 1, 0)
    end = start + len(ref_ids)

    eval_start = start
    eval_end = end
    if eval_span is not None:
        span_start, span_end = eval_span
        eval_start = start + span_start
        eval_end = start + span_end

    # Capture source hidden states at all positions
    source_hidden_all = None

    def capture_hook(module, inputs, output):
        nonlocal source_hidden_all
        if isinstance(output, tuple):
            source_hidden_all = output[0].detach().clone()
        else:
            source_hidden_all = output.detach().clone()

    source_layer = source_model.model.layers[layer]
    handle = source_layer.register_forward_hook(capture_hook)

    with torch.no_grad():
        source_model(input_ids, attention_mask=attention_mask)
    handle.remove()

    # Patch target with source hidden states at reference positions
    def patch_hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
            if patch_scope == "boundary":
                hs[:, start, :] = source_hidden_all[:, start, :].to(hs.dtype)
            else:
                # Patch positions predicting reference tokens only
                hs[:, start:end, :] = source_hidden_all[:, start:end, :].to(hs.dtype)
            return (hs,) + output[1:]
        else:
            hs = output.clone()
            if patch_scope == "boundary":
                hs[:, start, :] = source_hidden_all[:, start, :].to(hs.dtype)
            else:
                hs[:, start:end, :] = source_hidden_all[:, start:end, :].to(hs.dtype)
            return hs

    target_layer = target_model.model.layers[layer]
    handle = target_layer.register_forward_hook(patch_hook)

    with torch.no_grad():
        outputs = target_model(input_ids, attention_mask=attention_mask)
    handle.remove()

    # Get predictions for reference positions
    # Position prompt_len-1 predicts ref_ids[0], etc.
    logits = outputs.logits  # [1, seq_len, vocab]
    preds = logits[:, eval_start:eval_end, :].argmax(dim=-1)
    if eval_span is None:
        labels = torch.tensor([ref_ids], device=device)
    else:
        labels = torch.tensor([ref_ids[span_start:span_end]], device=device)

    matches = (preds == labels)
    if exact_match:
        em = 1.0 if matches.all() else 0.0
    else:
        correct = matches.float().sum().item()
        em = correct / labels.shape[1]

    if not return_details:
        return em

    pred_list = preds[0].tolist()
    label_list = labels[0].tolist()
    mismatches = [
        (i, p, t) for i, (p, t) in enumerate(zip(pred_list, label_list)) if p != t
    ]
    return em, mismatches


def compute_logprob_teacher_forcing_layer(
    source_model,
    target_model,
    tokenizer,
    prompt: str,
    reference: str,
    layer: int,
    eval_span=None,
    patch_scope: str = "span"
) -> float:
    """
    Compute mean log-prob with teacher forcing and layer patching.
    """
    device = next(target_model.parameters()).device

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    ref_ids = tokenizer.encode(reference, add_special_tokens=False)
    if len(ref_ids) == 0:
        return float("-inf")

    full_ids = prompt_ids + ref_ids
    input_ids = torch.tensor([full_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    prompt_len = len(prompt_ids)
    start = max(prompt_len - 1, 0)
    end = start + len(ref_ids)

    eval_start = start
    eval_end = end
    if eval_span is not None:
        span_start, span_end = eval_span
        eval_start = start + span_start
        eval_end = start + span_end
        ref_ids = ref_ids[span_start:span_end]

    source_hidden_all = None

    def capture_hook(module, inputs, output):
        nonlocal source_hidden_all
        if isinstance(output, tuple):
            source_hidden_all = output[0].detach().clone()
        else:
            source_hidden_all = output.detach().clone()

    source_layer = source_model.model.layers[layer]
    handle = source_layer.register_forward_hook(capture_hook)

    with torch.no_grad():
        source_model(input_ids, attention_mask=attention_mask)
    handle.remove()

    def patch_hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
            if patch_scope == "boundary":
                hs[:, start, :] = source_hidden_all[:, start, :].to(hs.dtype)
            else:
                hs[:, start:end, :] = source_hidden_all[:, start:end, :].to(hs.dtype)
            return (hs,) + output[1:]
        hs = output.clone()
        if patch_scope == "boundary":
            hs[:, start, :] = source_hidden_all[:, start, :].to(hs.dtype)
        else:
            hs[:, start:end, :] = source_hidden_all[:, start:end, :].to(hs.dtype)
        return hs

    target_layer = target_model.model.layers[layer]
    handle = target_layer.register_forward_hook(patch_hook)

    with torch.no_grad():
        outputs = target_model(input_ids, attention_mask=attention_mask)
    handle.remove()

    logits = outputs.logits[:, eval_start:eval_end, :]
    labels = torch.tensor([ref_ids], device=device)
    token_logprobs = _gather_token_logprobs(logits, labels)
    return token_logprobs.mean().item()


def compute_em_teacher_forcing_mlp(
    source_model,
    target_model,
    tokenizer,
    prompt: str,
    reference: str,
    layer: int,
    eval_span=None,
    exact_match: bool = False,
    return_details: bool = False,
    patch_scope: str = "span"
) -> float:
    """
    Compute EM with teacher forcing and MLP patching.
    """
    device = next(target_model.parameters()).device

    # Tokenize prompt with special tokens (matching generate_baseline)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    # Reference tokens without special tokens (just the answer part)
    ref_ids = tokenizer.encode(reference, add_special_tokens=False)

    if len(ref_ids) == 0:
        return 0.0

    # Full input: prompt (with BOS) + reference
    full_ids = prompt_ids + ref_ids
    input_ids = torch.tensor([full_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    prompt_len = len(prompt_ids)
    start = max(prompt_len - 1, 0)
    end = start + len(ref_ids)

    eval_start = start
    eval_end = end
    if eval_span is not None:
        span_start, span_end = eval_span
        eval_start = start + span_start
        eval_end = start + span_end

    # Capture source MLP hidden states at all positions
    source_mlp_hidden_all = None

    def capture_hook(module, inputs, output):
        nonlocal source_mlp_hidden_all
        source_mlp_hidden_all = output.detach().clone()

    source_mlp = source_model.model.layers[layer].mlp
    handle = source_mlp.register_forward_hook(capture_hook)

    with torch.no_grad():
        source_model(input_ids, attention_mask=attention_mask)
    handle.remove()

    # Patch target MLP with source hidden states at reference positions
    def patch_hook(module, inputs, output):
        hs = output.clone()
        if patch_scope == "boundary":
            hs[:, start, :] = source_mlp_hidden_all[:, start, :].to(hs.dtype)
        else:
            hs[:, start:end, :] = source_mlp_hidden_all[:, start:end, :].to(hs.dtype)
        return hs

    target_mlp = target_model.model.layers[layer].mlp
    handle = target_mlp.register_forward_hook(patch_hook)

    with torch.no_grad():
        outputs = target_model(input_ids, attention_mask=attention_mask)
    handle.remove()

    # Get predictions for reference positions
    logits = outputs.logits
    preds = logits[:, eval_start:eval_end, :].argmax(dim=-1)
    if eval_span is None:
        labels = torch.tensor([ref_ids], device=device)
    else:
        labels = torch.tensor([ref_ids[span_start:span_end]], device=device)

    matches = (preds == labels)
    if exact_match:
        em = 1.0 if matches.all() else 0.0
    else:
        correct = matches.float().sum().item()
        em = correct / labels.shape[1]

    if not return_details:
        return em

    pred_list = preds[0].tolist()
    label_list = labels[0].tolist()
    mismatches = [
        (i, p, t) for i, (p, t) in enumerate(zip(pred_list, label_list)) if p != t
    ]
    return em, mismatches


def compute_logprob_teacher_forcing_mlp(
    source_model,
    target_model,
    tokenizer,
    prompt: str,
    reference: str,
    layer: int,
    eval_span=None,
    patch_scope: str = "span"
) -> float:
    """
    Compute mean log-prob with teacher forcing and MLP patching.
    """
    device = next(target_model.parameters()).device

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    ref_ids = tokenizer.encode(reference, add_special_tokens=False)
    if len(ref_ids) == 0:
        return float("-inf")

    full_ids = prompt_ids + ref_ids
    input_ids = torch.tensor([full_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    prompt_len = len(prompt_ids)
    start = max(prompt_len - 1, 0)
    end = start + len(ref_ids)

    eval_start = start
    eval_end = end
    if eval_span is not None:
        span_start, span_end = eval_span
        eval_start = start + span_start
        eval_end = start + span_end
        ref_ids = ref_ids[span_start:span_end]

    source_mlp_hidden_all = None

    def capture_hook(module, inputs, output):
        nonlocal source_mlp_hidden_all
        source_mlp_hidden_all = output.detach().clone()

    source_mlp = source_model.model.layers[layer].mlp
    handle = source_mlp.register_forward_hook(capture_hook)

    with torch.no_grad():
        source_model(input_ids, attention_mask=attention_mask)
    handle.remove()

    def patch_hook(module, inputs, output):
        hs = output.clone()
        if patch_scope == "boundary":
            hs[:, start, :] = source_mlp_hidden_all[:, start, :].to(hs.dtype)
        else:
            hs[:, start:end, :] = source_mlp_hidden_all[:, start:end, :].to(hs.dtype)
        return hs

    target_mlp = target_model.model.layers[layer].mlp
    handle = target_mlp.register_forward_hook(patch_hook)

    with torch.no_grad():
        outputs = target_model(input_ids, attention_mask=attention_mask)
    handle.remove()

    logits = outputs.logits[:, eval_start:eval_end, :]
    labels = torch.tensor([ref_ids], device=device)
    token_logprobs = _gather_token_logprobs(logits, labels)
    return token_logprobs.mean().item()


def run_s1_s2_side_by_side(retain, unlearn, full, tokenizer, prefix_data, layer_list,
                            em_threshold=0.5, patch_mode="layer", unlearn_name="unlearn",
                            em_scope="full", entity_source="gt", em_exact=False,
                            log_mismatch=False, mismatch_max=5, log_file=None,
                            metric="em", delta_threshold=0.0, patch_scope="span"):
    """Run S1 and S2 side by side for each example using teacher forcing EM or log-prob."""
    results = []
    skipped_indices = []

    # Counters for summary
    gk_count = 0  # GK: retain predicts full reference tokens (teacher forcing EM)
    categories = None

    if metric == "em":
        em_fn = (compute_em_teacher_forcing_layer if patch_mode == "layer"
                 else compute_em_teacher_forcing_mlp)
    else:
        em_fn = (compute_logprob_teacher_forcing_layer if patch_mode == "layer"
                 else compute_logprob_teacher_forcing_mlp)

    for i, item in enumerate(tqdm(prefix_data, desc=f"S1/S2 {patch_mode}")):
        question = item["question"]
        prefix = item["prefix"]
        entity = item["entity"]
        gt_entity = item.get("gt_entity", entity)
        idx = item["idx"]

        prompt = f"Question: {question}\nAnswer: {prefix}"

        # Use pre-computed full_output if available (v6 dataset), otherwise generate
        if "full_output" in item:
            full_gen = item["full_output"]
        else:
            full_gen = generate_baseline(full, tokenizer, prompt, max_new_tokens=30)

        # Get baselines for display
        retain_gen = generate_baseline(retain, tokenizer, prompt, max_new_tokens=30)
        unlearn_gen = generate_baseline(unlearn, tokenizer, prompt, max_new_tokens=30)

        # Reference for EM = Full baseline generation (normalized for tokenization)
        reference = normalize_reference_for_eval(prompt, full_gen)

        # Select entity span source
        if entity_source == "gt":
            entity_text = gt_entity
        else:
            entity_text = item.get("full_entity", entity)

        eval_span = get_eval_span(tokenizer, reference, entity_text, em_scope)
        if em_scope == "entity" and eval_span is None:
            skipped_indices.append(idx)
            if log_file:
                log_msg = f"\n{'='*80}\n"
                log_msg += f"[{i+1}/{len(prefix_data)}] Example {idx} | SKIPPED (entity span not found)\n"
                log_msg += f"  Q: {question}\n"
                log_msg += f"  Prefix: '{prefix}'\n"
                log_msg += f"  GT (entity): '{gt_entity}'\n"
                log_msg += f"  Eval entity ({entity_source}): '{entity_text}'\n"
                log_msg += f"  Full baseline: \"{clean_text(full_gen)}\"\n"
                log_msg += f"{'='*80}\n"
                log_file.write(log_msg)
                log_file.flush()
            continue

        # Precompute full baseline score for log-prob metric
        full_score = None
        if metric == "logprob":
            full_score = compute_logprob_teacher_forcing_baseline(
                full, tokenizer, prompt, reference, eval_span=eval_span
            )

        # Run S1 and S2 for each layer using teacher forcing EM/log-prob
        s1_details = []
        s2_details = []
        s1_lost = 0
        s2_lost = 0
        s1_deltas = []
        s2_deltas = []

        for layer in layer_list:
            # S1: Retain → Full (teacher forcing EM with full_gen as reference)
            if metric == "em":
                s1_mismatches = None
                if log_mismatch:
                    s1_em, s1_mismatches = em_fn(
                        retain, full, tokenizer, prompt, reference, layer,
                        eval_span=eval_span, exact_match=em_exact, return_details=True,
                        patch_scope=patch_scope
                    )
                else:
                    s1_em = em_fn(
                        retain, full, tokenizer, prompt, reference, layer,
                        eval_span=eval_span, exact_match=em_exact, patch_scope=patch_scope
                    )
                s1_status = "KEPT" if s1_em >= em_threshold else "LOST"
                if s1_status == "LOST":
                    s1_lost += 1
                s1_detail = {"layer": layer, "em": s1_em, "status": s1_status}
                if log_mismatch:
                    s1_detail["mismatches"] = s1_mismatches
            else:
                s1_score = em_fn(
                    retain, full, tokenizer, prompt, reference, layer,
                    eval_span=eval_span, patch_scope=patch_scope
                )
                s1_delta = full_score - s1_score
                s1_deltas.append(s1_delta)
                s1_status = "LOST" if s1_delta > delta_threshold else "KEPT"
                if s1_status == "LOST":
                    s1_lost += 1
                s1_detail = {"layer": layer, "score": s1_score, "delta": s1_delta, "status": s1_status}
            s1_details.append(s1_detail)

            # S2: Unlearn → Full (teacher forcing EM with full_gen as reference)
            if metric == "em":
                s2_mismatches = None
                if log_mismatch:
                    s2_em, s2_mismatches = em_fn(
                        unlearn, full, tokenizer, prompt, reference, layer,
                        eval_span=eval_span, exact_match=em_exact, return_details=True,
                        patch_scope=patch_scope
                    )
                else:
                    s2_em = em_fn(
                        unlearn, full, tokenizer, prompt, reference, layer,
                        eval_span=eval_span, exact_match=em_exact, patch_scope=patch_scope
                    )
                s2_status = "KEPT" if s2_em >= em_threshold else "LOST"
                if s2_status == "LOST":
                    s2_lost += 1
                s2_detail = {"layer": layer, "em": s2_em, "status": s2_status}
                if log_mismatch:
                    s2_detail["mismatches"] = s2_mismatches
            else:
                s2_score = em_fn(
                    unlearn, full, tokenizer, prompt, reference, layer,
                    eval_span=eval_span, patch_scope=patch_scope
                )
                s2_delta = full_score - s2_score
                s2_deltas.append(s2_delta)
                s2_status = "LOST" if s2_delta > delta_threshold else "KEPT"
                if s2_status == "LOST":
                    s2_lost += 1
                s2_detail = {"layer": layer, "score": s2_score, "delta": s2_delta, "status": s2_status}
            s2_details.append(s2_detail)

        # GK definition: retain teacher forcing EM on Full reference
        if metric == "em":
            retain_em_full = compute_em_teacher_forcing_baseline(
                retain, tokenizer, prompt, reference, eval_span=eval_span, exact_match=em_exact
            )
            is_gk = retain_em_full >= em_threshold
            if is_gk:
                gk_count += 1
        else:
            retain_em_full = None
            is_gk = None

        # Build side-by-side log
        log_msg = f"\n{'='*80}\n"
        log_msg += f"[{i+1}/{len(prefix_data)}] Example {idx}\n"
        log_msg += f"  Q: {question}\n"
        log_msg += f"  Prefix: '{prefix}'\n"
        log_msg += f"  GT (entity): '{gt_entity}'\n"
        if em_scope == "entity":
            log_msg += f"  Eval entity ({entity_source}): '{entity_text}'\n"
        log_msg += f"  EM scope: {em_scope}\n"
        log_msg += f"  Full baseline: \"{clean_text(full_gen)}\"\n"
        log_msg += f"  Retain baseline: \"{clean_text(retain_gen)}\"\n"
        log_msg += f"  {unlearn_name} baseline: \"{clean_text(unlearn_gen)}\"\n"
        if metric == "logprob" and full_score is not None:
            log_msg += f"  Full log-prob (ref span): {full_score:.3f}\n"
        log_msg += f"{'='*80}\n"
        log_msg += f"  {'Layer':<6} | {'S1 (Retain→Full)':<25} | {'S2 ('+unlearn_name+'→Full)':<25}\n"
        log_msg += f"  {'-'*6} | {'-'*25} | {'-'*25}\n"

        for s1, s2 in zip(s1_details, s2_details):
            if metric == "em":
                s1_str = f"EM={s1['em']:.2f} [{s1['status']}]"
                s2_str = f"EM={s2['em']:.2f} [{s2['status']}]"
            else:
                s1_str = f"logp={s1['score']:.3f} Δ={s1['delta']:.3f} [{s1['status']}]"
                s2_str = f"logp={s2['score']:.3f} Δ={s2['delta']:.3f} [{s2['status']}]"
            log_msg += f"  L{s1['layer']:02d}   | {s1_str:<25} | {s2_str:<25}\n"
            if log_mismatch and metric == "em" and s1['em'] < 1.0:
                mismatch_str = format_mismatches(
                    tokenizer, s1.get("mismatches") or [], mismatch_max
                )
                if mismatch_str:
                    log_msg += f"         S1 mismatch: {mismatch_str}\n"
            if log_mismatch and metric == "em" and s2['em'] < 1.0:
                mismatch_str = format_mismatches(
                    tokenizer, s2.get("mismatches") or [], mismatch_max
                )
                if mismatch_str:
                    log_msg += f"         S2 mismatch: {mismatch_str}\n"

        log_msg += f"  {'-'*6} | {'-'*25} | {'-'*25}\n"

        # Calculate UDR and FT/Erased layers
        ft_layers = [s1["layer"] for s1 in s1_details if s1["status"] == "LOST"]
        erased_layers = [s2["layer"] for s2 in s2_details if s2["status"] == "LOST" and s2["layer"] in ft_layers]

        if metric == "em":
            udr = len(erased_layers) / len(ft_layers) if ft_layers else 0.0
        else:
            # FT-only UDR with per-layer clipping to avoid overshoot.
            ft_set = set(ft_layers)
            denom = 0.0
            numer = 0.0
            for s1, s2, d1, d2 in zip(s1_details, s2_details, s1_deltas, s2_deltas):
                if s1["layer"] not in ft_set:
                    continue
                if d1 <= delta_threshold:
                    continue
                denom += d1
                ratio = d2 / d1
                if ratio < 0.0:
                    ratio = 0.0
                elif ratio > 1.0:
                    ratio = 1.0
                numer += d1 * ratio
            if denom > 0:
                udr = numer / denom
            else:
                udr = None

        log_msg += f"  FT layers (S1 LOST): {ft_layers}\n"
        log_msg += f"  Erased layers (S2 LOST ∩ FT): {erased_layers}\n"
        if metric == "em":
            log_msg += f"  UDR_i = {len(erased_layers)}/{len(ft_layers)} = {udr:.2f}\n" if ft_layers else f"  UDR_i = N/A (no FT layers)\n"
            log_msg += f"  GK (retain TF EM vs Full ref, info): {is_gk} (EM={retain_em_full:.2f})\n"
        else:
            log_msg += f"  UDR_i = {udr:.2f}\n" if udr is not None else f"  UDR_i = N/A (no FT signal)\n"

        if log_file:
            log_file.write(log_msg)
            log_file.flush()

        results.append({
            "idx": idx,
            "question": question,
            "prefix": prefix,
            "entity": entity,
            "gt_entity": gt_entity,
            "entity_text": entity_text,
            "entity_source": entity_source,
            "full_gen": full_gen,
            "retain_gen": retain_gen,
            "unlearn_gen": unlearn_gen,
            "s1_lost": s1_lost,
            "s2_lost": s2_lost,
            "ft_layers": ft_layers,
            "erased_layers": erased_layers,
            "udr": udr,
            "is_gk": is_gk,
            "retain_em_full": retain_em_full,
            "full_score": full_score if metric == "logprob" else None,
            "eval_scope": em_scope,
            "eval_span": eval_span,
            "s1_details": s1_details,
            "s2_details": s2_details
        })

    # Calculate average UDR (all evaluable examples with FT signal)
    udr_values = [r["udr"] for r in results if r["udr"] is not None]
    avg_udr = sum(udr_values) / len(udr_values) if udr_values else 0.0

    return results, gk_count, categories, avg_udr, skipped_indices


def plot_erasure_histogram(results, output_path, method_name="Unlearn", mode="layer"):
    """Plot histogram of S1 LOST - S2 LOST layer difference."""
    diffs = []
    for item in results:
        # Difference: positive = under-erased (S1 LOST > S2 LOST)
        #            negative = over-erased (S1 LOST < S2 LOST)
        diff = item["s1_lost"] - item["s2_lost"]
        diffs.append(diff)

    if not diffs:
        print("No evaluable examples for histogram")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    min_val = min(diffs)
    max_val = max(diffs)
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

    ax.hist(diffs, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('S1 LOST - S2 LOST (Layer Count Difference)', fontsize=11)
    ax.set_ylabel('Number of Examples', fontsize=11)
    ax.set_title(f'{method_name} - {mode.upper()} Patching (n={len(diffs)})', fontsize=12)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xticks(range(min_val, max_val + 1))

    # Add stats
    over_erased = sum(1 for d in diffs if d < 0)
    exact = sum(1 for d in diffs if d == 0)
    under_erased = sum(1 for d in diffs if d > 0)
    stats_text = f'Over: {over_erased} ({100*over_erased/len(diffs):.1f}%)\n'
    stats_text += f'Exact: {exact} ({100*exact/len(diffs):.1f}%)\n'
    stats_text += f'Under: {under_erased} ({100*under_erased/len(diffs):.1f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--layers", type=str, default="0-15")
    parser.add_argument("--metric", type=str, choices=["em", "logprob"], default="logprob")
    parser.add_argument("--em_threshold", type=float, default=1.0)
    parser.add_argument("--delta_threshold", type=float, default=0.01)
    parser.add_argument("--patch_scope", type=str, choices=["span", "boundary"], default="boundary",
                        help="Patch reference span or only boundary (last prompt token)")
    parser.add_argument("--em_type", type=str, choices=["token", "exact"], default="token")
    parser.add_argument("--em_scope", type=str, choices=["full", "entity"], default="full")
    parser.add_argument("--entity_source", type=str, choices=["gt", "full"], default="full")
    parser.add_argument("--log_mismatch", action="store_true")
    parser.add_argument("--mismatch_max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["layer", "mlp"], default="layer")
    parser.add_argument("--unlearn_model", type=str, required=True,
                        help="Unlearn model (e.g., simnpo, idknll)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    em_exact = args.em_type == "exact"

    print("=" * 80)
    print("S1/S2 Experiment with Teacher Forcing (Side-by-Side)")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Metric: {args.metric}")
    if args.metric == "em":
        print(f"EM threshold: {args.em_threshold}")
        print(f"EM type: {args.em_type}")
    else:
        print(f"Delta threshold: {args.delta_threshold}")
    print(f"EM scope: {args.em_scope}")
    print(f"Entity source: {args.entity_source}")
    print(f"Patch scope: {args.patch_scope}")
    if args.log_mismatch:
        print(f"Mismatch logging: enabled (max {args.mismatch_max})")
    print(f"Unlearn model: {args.unlearn_model}")
    print()

    # Load models
    print("Loading models...")
    tokenizer = load_tokenizer(TOFU_FULL_MODEL)
    retain = load_model(TOFU_RETAIN_MODEL, dtype="bfloat16", device_map="cuda")
    full = load_model(TOFU_FULL_MODEL, dtype="bfloat16", device_map="cuda")

    unlearn_model_id = get_model_id(args.unlearn_model)
    print(f"Loading unlearn model: {unlearn_model_id}")
    unlearn = load_model(unlearn_model_id, dtype="bfloat16", device_map="cuda")

    n_layers = get_num_layers(full)
    layer_list = parse_layers(args.layers, n_layers)
    print(f"Layers: {layer_list}")

    # Load data
    prefix_data = load_prefix_data()
    print(f"Loaded {len(prefix_data)} examples")

    if args.num_examples:
        prefix_data = prefix_data[:args.num_examples]
        print(f"Using first {len(prefix_data)} examples")

    # Create output directory (date first)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    out_dir = f"runs/{timestamp}_tf_{args.unlearn_model}_{args.mode}"
    safe_mkdir(out_dir)

    # Run S1/S2 side by side
    print("\n" + "=" * 40)
    print(f"Running S1/S2 ({args.mode}, teacher forcing)")
    print("=" * 40)

    log_path = f"{out_dir}/run.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"S1/S2 {args.mode.upper()} Patching with Teacher Forcing\n")
        if args.metric == "em":
            log_file.write(f"Metric: EM (token accuracy), reference = Full baseline\n")
            log_file.write(f"EM type: {args.em_type}\n")
            log_file.write(f"EM threshold: {args.em_threshold}\n")
        else:
            log_file.write(f"Metric: log-prob (reference tokens), reference = Full baseline\n")
            log_file.write(f"Delta threshold: {args.delta_threshold}\n")
        log_file.write(f"EM scope: {args.em_scope}\n")
        log_file.write(f"Entity source: {args.entity_source}\n")
        log_file.write(f"Patch scope: {args.patch_scope}\n")
        if args.log_mismatch:
            log_file.write(f"Mismatch logging: enabled (max {args.mismatch_max})\n")
        log_file.write(f"S1: Retain → Full | S2: {args.unlearn_model} → Full\n")
        log_file.write(f"Layers: {layer_list}\n")
        log_file.write(f"Examples: {len(prefix_data)}\n\n")

        results, gk_count, categories, avg_udr, skipped_indices = run_s1_s2_side_by_side(
            retain, unlearn, full, tokenizer, prefix_data, layer_list,
            args.em_threshold, args.mode, args.unlearn_model, args.em_scope, args.entity_source,
            em_exact, args.log_mismatch, args.mismatch_max, log_file,
            args.metric, args.delta_threshold, args.patch_scope
        )

        # Write summary at the end
        total = len(prefix_data)
        skipped_count = len(skipped_indices)
        evaluated = total - skipped_count

        summary_msg = f"\n\n{'='*80}\n"
        summary_msg += f"EXPERIMENT SUMMARY\n"
        summary_msg += f"{'='*80}\n"
        summary_msg += f"Total examples: {total}\n"
        summary_msg += f"Metric: {args.metric}\n"
        if args.metric == "em":
            summary_msg += f"EM type: {args.em_type}\n"
        summary_msg += f"EM scope: {args.em_scope}\n"
        summary_msg += f"Entity source: {args.entity_source}\n"
        summary_msg += f"Patch scope: {args.patch_scope}\n"
        if skipped_count:
            summary_msg += f"Skipped (entity span missing): {skipped_count} ({100*skipped_count/total:.1f}%)\n"
        if evaluated:
            summary_msg += f"Evaluable (non-skipped): {evaluated} ({100*evaluated/total:.1f}%)\n"
            if args.metric == "em":
                summary_msg += f"GK (retain TF EM vs Full ref, info): {gk_count} ({100*gk_count/evaluated:.1f}%)\n"
        else:
            summary_msg += "Evaluable (non-skipped): 0 (0.0%)\n"
            if args.metric == "em":
                summary_msg += "GK (retain TF EM vs Full ref, info): 0 (0.0%)\n"
        summary_msg += f"UDR: {avg_udr:.2f}\n"
        summary_msg += f"{'='*80}\n"

        log_file.write(summary_msg)
        print(summary_msg)

    print(f"Log saved to: {log_path}")

    # Save JSON results
    summary = {
        "total_examples": total,
        "metric": args.metric,
        "em_threshold": args.em_threshold,
        "delta_threshold": args.delta_threshold,
        "em_type": args.em_type,
        "em_scope": args.em_scope,
        "entity_source": args.entity_source,
        "mode": args.mode,
        "unlearn_model": args.unlearn_model,
        "gk_count": gk_count,
        "avg_udr": avg_udr,
        "skipped_count": skipped_count,
        "skipped_indices": skipped_indices
    }

    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()

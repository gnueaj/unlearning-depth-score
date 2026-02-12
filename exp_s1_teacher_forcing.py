#!/usr/bin/env python3
"""
S1/S2 Experiment with Teacher Forcing (Retain → Full, Unlearn → Full)

Metrics:
- EM (token accuracy) or log-prob score on reference tokens
- Reference = per-example dataset reference (GT or Full)

Flow:
1. Get dataset reference (GT answer or Full baseline output)
2. For each layer, compute metric with patching:
   - Input: prompt + reference tokens
   - At each reference token position: patch source hidden → predict next token
   - Metric = EM or mean log-prob on reference span

Output: Side-by-side S1 vs S2 comparison per example + UDS
"""

import os
import sys
import json
import argparse
import csv
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import time
import re

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uds.models import load_model, load_tokenizer, get_num_layers
from uds.core import generate_baseline
from uds.utils import set_seed, safe_mkdir, parse_layers
from uds.config import get_model_id


TOFU_FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
TOFU_RETAIN_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
PREFIX_DATA_PATH = "tofu_data/forget10_filtered_v7_gt.json"


def sanitize_name_for_path(name: str) -> str:
    """Convert model/run names to filesystem-safe tokens."""
    token = re.sub(r"[^a-zA-Z0-9._-]+", "__", name.strip())
    token = token.strip("._-")
    return token or "run"


def load_model_for_exp(
    model_id: str,
    dtype: str,
    device_map: str,
    attn_implementation: Optional[str],
):
    """Load model for experiment."""
    return load_model(
        model_id,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )


def load_prefix_data(path: str) -> List[Dict]:
    """Load validated prefix+entity data."""
    with open(path, "r", encoding="utf-8") as f:
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


def build_layer_records(
    s1_details: List[Dict],
    s2_details: List[Dict],
    full_score: Optional[float],
) -> List[Dict]:
    """Create per-layer structured rows with p/s1/s2 and deltas for detailed logging."""
    rows: List[Dict] = []
    if len(s1_details) != len(s2_details):
        return rows

    for s1, s2 in zip(s1_details, s2_details):
        layer = s1.get("layer", s2.get("layer"))
        row = {
            "layer": layer,
            "p_score": full_score,
            "s1_score": None,
            "s2_score": None,
            "s1_delta": None,
            "s2_delta": None,
            "s1_status": s1.get("status"),
            "s2_status": s2.get("status"),
            "delta_gap": None,  # s2_delta - s1_delta
            "ratio_s2_over_s1": None,
            "ratio_s2_over_s1_clipped": None,
            "s1_em": None,
            "s2_em": None,
        }

        if "score" in s1 and "score" in s2:
            d1 = s1.get("delta")
            d2 = s2.get("delta")
            row["s1_score"] = s1.get("score")
            row["s2_score"] = s2.get("score")
            row["s1_delta"] = d1
            row["s2_delta"] = d2
            if d1 is not None and d2 is not None:
                row["delta_gap"] = d2 - d1
            if d1 is not None and d2 is not None and abs(d1) > 1e-12:
                ratio = d2 / d1
                row["ratio_s2_over_s1"] = ratio
                row["ratio_s2_over_s1_clipped"] = min(max(ratio, 0.0), 1.0)
        elif "em" in s1 and "em" in s2:
            row["s1_em"] = s1.get("em")
            row["s2_em"] = s2.get("em")

        rows.append(row)
    return rows


def write_detailed_artifacts(out_dir: str, results: List[Dict], skipped_records: List[Dict]) -> Dict[str, str]:
    """Write detailed machine-readable artifacts for reproducibility and audit."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    detailed_jsonl_path = out / "results_detailed.jsonl"
    with detailed_jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    layer_csv_path = out / "layer_details.csv"
    with layer_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "question",
                "entity_text",
                "entity_source",
                "reference_source",
                "reference_text",
                "layer",
                "p_score",
                "s1_score",
                "s2_score",
                "s1_delta",
                "s2_delta",
                "delta_gap",
                "ratio_s2_over_s1",
                "ratio_s2_over_s1_clipped",
                "s1_status",
                "s2_status",
                "s1_em",
                "s2_em",
                "uds",
            ],
        )
        writer.writeheader()
        for r in results:
            layer_records = r.get("layer_records") or []
            for row in layer_records:
                writer.writerow(
                    {
                        "idx": r.get("idx"),
                        "question": r.get("question"),
                        "entity_text": r.get("entity_text"),
                        "entity_source": r.get("entity_source"),
                        "reference_source": r.get("reference_source"),
                        "reference_text": r.get("reference_text"),
                        "layer": row.get("layer"),
                        "p_score": row.get("p_score"),
                        "s1_score": row.get("s1_score"),
                        "s2_score": row.get("s2_score"),
                        "s1_delta": row.get("s1_delta"),
                        "s2_delta": row.get("s2_delta"),
                        "delta_gap": row.get("delta_gap"),
                        "ratio_s2_over_s1": row.get("ratio_s2_over_s1"),
                        "ratio_s2_over_s1_clipped": row.get("ratio_s2_over_s1_clipped"),
                        "s1_status": row.get("s1_status"),
                        "s2_status": row.get("s2_status"),
                        "s1_em": row.get("s1_em"),
                        "s2_em": row.get("s2_em"),
                        "uds": r.get("uds"),
                    }
                )

    skipped_path = out / "skipped_examples.json"
    with skipped_path.open("w", encoding="utf-8") as f:
        json.dump(skipped_records, f, indent=2, ensure_ascii=False)

    return {
        "detailed_jsonl": str(detailed_jsonl_path),
        "layer_csv": str(layer_csv_path),
        "skipped_json": str(skipped_path),
    }


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
    span_start = None
    span_end = None
    if eval_span is not None:
        span_start, span_end = eval_span
        eval_start = start + span_start
        eval_end = start + span_end

    # Patch span: default to full reference span, or eval span if provided
    patch_start = start
    patch_end = end
    if patch_scope == "span" and eval_span is not None:
        patch_start = start + span_start
        patch_end = start + span_end

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
                hs[:, start, :] = source_hidden_all[:, start, :].to(device=hs.device, dtype=hs.dtype)
            else:
                # Patch positions predicting reference tokens (or eval span only)
                hs[:, patch_start:patch_end, :] = source_hidden_all[:, patch_start:patch_end, :].to(device=hs.device, dtype=hs.dtype)
            return (hs,) + output[1:]
        else:
            hs = output.clone()
            if patch_scope == "boundary":
                hs[:, start, :] = source_hidden_all[:, start, :].to(device=hs.device, dtype=hs.dtype)
            else:
                hs[:, patch_start:patch_end, :] = source_hidden_all[:, patch_start:patch_end, :].to(device=hs.device, dtype=hs.dtype)
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
    span_start = None
    span_end = None
    if eval_span is not None:
        span_start, span_end = eval_span
        eval_start = start + span_start
        eval_end = start + span_end
        ref_ids = ref_ids[span_start:span_end]

    # Patch span: default to full reference span, or eval span if provided
    patch_start = start
    patch_end = end
    if patch_scope == "span" and eval_span is not None:
        patch_start = start + span_start
        patch_end = start + span_end

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
                hs[:, start, :] = source_hidden_all[:, start, :].to(device=hs.device, dtype=hs.dtype)
            else:
                hs[:, patch_start:patch_end, :] = source_hidden_all[:, patch_start:patch_end, :].to(device=hs.device, dtype=hs.dtype)
            return (hs,) + output[1:]
        hs = output.clone()
        if patch_scope == "boundary":
            hs[:, start, :] = source_hidden_all[:, start, :].to(device=hs.device, dtype=hs.dtype)
        else:
            hs[:, patch_start:patch_end, :] = source_hidden_all[:, patch_start:patch_end, :].to(device=hs.device, dtype=hs.dtype)
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


def _prepare_batch_inputs(batch_ctxs, tokenizer, device):
    """Prepare padded batch tensors and per-example metadata."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    full_ids_list = [ctx["prompt_ids"] + ctx["ref_ids"] for ctx in batch_ctxs]
    lengths = [len(x) for x in full_ids_list]
    max_len = max(lengths)
    batch_size = len(batch_ctxs)

    input_ids = torch.full((batch_size, max_len), pad_id, device=device, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)

    for i, ids in enumerate(full_ids_list):
        input_ids[i, :len(ids)] = torch.tensor(ids, device=device, dtype=torch.long)
        attention_mask[i, :len(ids)] = 1

    meta = {
        "lengths": lengths,
        "start": [ctx["start"] for ctx in batch_ctxs],
        "patch_start": [ctx["patch_start"] for ctx in batch_ctxs],
        "patch_end": [ctx["patch_end"] for ctx in batch_ctxs],
        "eval_start": [ctx["eval_start"] for ctx in batch_ctxs],
        "eval_end": [ctx["eval_end"] for ctx in batch_ctxs],
        "eval_ref_ids": [ctx["eval_ref_ids"] for ctx in batch_ctxs],
    }
    return input_ids, attention_mask, meta


def _compute_hidden_states_batch(model, input_ids, attention_mask):
    """Compute all layer hidden states in a single forward pass."""
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    # outputs.hidden_states: (embeddings + layer outputs)
    hidden_states = [hs.detach() for hs in outputs.hidden_states]
    return hidden_states


def compute_logprob_teacher_forcing_baseline_batch_with_inputs(
    model,
    input_ids,
    attention_mask,
    meta,
):
    """Batch baseline log-prob (no patching) using pre-built inputs."""
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    out = []
    for i in range(input_ids.size(0)):
        es = meta["eval_start"][i]
        ee = meta["eval_end"][i]
        if ee <= es:
            out.append(float("-inf"))
            continue
        labels = torch.tensor([meta["eval_ref_ids"][i]], device=input_ids.device)
        token_logprobs = _gather_token_logprobs(logits[i:i+1, es:ee, :], labels)
        out.append(token_logprobs.mean().item())
    return out


def compute_logprob_teacher_forcing_layer_batch_with_inputs(
    target_model,
    input_ids,
    attention_mask,
    meta,
    source_hidden_layer,
    layer: int,
    patch_scope: str = "span"
):
    """Batch log-prob with layer patching using precomputed source hidden states."""
    device = next(target_model.parameters()).device

    def patch_hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
        else:
            hs = output.clone()

        for b in range(hs.size(0)):
            length = meta["lengths"][b]
            start = meta["start"][b]
            ps = meta["patch_start"][b]
            pe = meta["patch_end"][b]
            if start >= length:
                continue
            if patch_scope == "boundary":
                hs[b, start, :] = source_hidden_layer[b, start, :].to(device=hs.device, dtype=hs.dtype)
            else:
                ps = min(ps, length)
                pe = min(pe, length)
                if ps < pe:
                    hs[b, ps:pe, :] = source_hidden_layer[b, ps:pe, :].to(device=hs.device, dtype=hs.dtype)

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    target_layer = target_model.model.layers[layer]
    handle = target_layer.register_forward_hook(patch_hook)
    with torch.no_grad():
        outputs = target_model(input_ids, attention_mask=attention_mask)
    handle.remove()

    logits = outputs.logits
    out = []
    for i in range(input_ids.size(0)):
        es = meta["eval_start"][i]
        ee = meta["eval_end"][i]
        if ee <= es:
            out.append(float("-inf"))
            continue
        labels = torch.tensor([meta["eval_ref_ids"][i]], device=device)
        token_logprobs = _gather_token_logprobs(logits[i:i+1, es:ee, :], labels)
        out.append(token_logprobs.mean().item())
    return out


def build_logprob_ctx(tokenizer, prompt: str, reference: str, eval_span, patch_scope: str):
    """Build context dict for batched log-prob evaluation."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    ref_ids = tokenizer.encode(reference, add_special_tokens=False)
    if len(ref_ids) == 0:
        return None

    prompt_len = len(prompt_ids)
    start = max(prompt_len - 1, 0)
    end = start + len(ref_ids)

    eval_start = start
    eval_end = end
    eval_ref_ids = ref_ids
    if eval_span is not None:
        span_start, span_end = eval_span
        eval_start = start + span_start
        eval_end = start + span_end
        eval_ref_ids = ref_ids[span_start:span_end]

    patch_start = start
    patch_end = end
    if patch_scope == "span" and eval_span is not None:
        patch_start = eval_start
        patch_end = eval_end

    return {
        "prompt_ids": prompt_ids,
        "ref_ids": ref_ids,
        "start": start,
        "patch_start": patch_start,
        "patch_end": patch_end,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "eval_ref_ids": eval_ref_ids,
    }


def compute_logprob_teacher_forcing_baseline_batch(
    model,
    tokenizer,
    batch_ctxs,
):
    """Batch baseline log-prob (no patching)."""
    device = next(model.parameters()).device
    input_ids, attention_mask, meta = _prepare_batch_inputs(batch_ctxs, tokenizer, device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    out = []
    for i in range(len(batch_ctxs)):
        es = meta["eval_start"][i]
        ee = meta["eval_end"][i]
        if ee <= es:
            out.append(float("-inf"))
            continue
        labels = torch.tensor([meta["eval_ref_ids"][i]], device=device)
        token_logprobs = _gather_token_logprobs(logits[i:i+1, es:ee, :], labels)
        out.append(token_logprobs.mean().item())
    return out


def compute_logprob_teacher_forcing_layer_batch(
    source_model,
    target_model,
    tokenizer,
    batch_ctxs,
    layer: int,
    patch_scope: str = "span"
):
    """Batch log-prob with layer patching (logprob metric only)."""
    device = next(target_model.parameters()).device
    input_ids, attention_mask, meta = _prepare_batch_inputs(batch_ctxs, tokenizer, device)

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
        else:
            hs = output.clone()

        for b in range(hs.size(0)):
            length = meta["lengths"][b]
            start = meta["start"][b]
            ps = meta["patch_start"][b]
            pe = meta["patch_end"][b]
            if start >= length:
                continue
            if patch_scope == "boundary":
                hs[b, start, :] = source_hidden_all[b, start, :].to(device=hs.device, dtype=hs.dtype)
            else:
                ps = min(ps, length)
                pe = min(pe, length)
                if ps < pe:
                    hs[b, ps:pe, :] = source_hidden_all[b, ps:pe, :].to(device=hs.device, dtype=hs.dtype)

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    target_layer = target_model.model.layers[layer]
    handle = target_layer.register_forward_hook(patch_hook)
    with torch.no_grad():
        outputs = target_model(input_ids, attention_mask=attention_mask)
    handle.remove()

    logits = outputs.logits
    out = []
    for i in range(len(batch_ctxs)):
        es = meta["eval_start"][i]
        ee = meta["eval_end"][i]
        if ee <= es:
            out.append(float("-inf"))
            continue
        labels = torch.tensor([meta["eval_ref_ids"][i]], device=device)
        token_logprobs = _gather_token_logprobs(logits[i:i+1, es:ee, :], labels)
        out.append(token_logprobs.mean().item())
    return out


def _build_s1_cache_key(cache_cfg: dict) -> str:
    raw = json.dumps(cache_cfg, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _load_s1_cache(cache_path: Path, cache_cfg: dict):
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
    except Exception:
        return None
    if data.get("config") != cache_cfg:
        return None
    return data.get("entries", None)


def _save_s1_cache(cache_path: Path, cache_cfg: dict, entries):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": cache_cfg, "entries": entries}
    cache_path.write_text(json.dumps(payload))


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
    span_start = None
    span_end = None
    if eval_span is not None:
        span_start, span_end = eval_span
        eval_start = start + span_start
        eval_end = start + span_end

    # Patch span: default to full reference span, or eval span if provided
    patch_start = start
    patch_end = end
    if patch_scope == "span" and eval_span is not None:
        patch_start = start + span_start
        patch_end = start + span_end

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
            hs[:, start, :] = source_mlp_hidden_all[:, start, :].to(device=hs.device, dtype=hs.dtype)
        else:
            hs[:, patch_start:patch_end, :] = source_mlp_hidden_all[:, patch_start:patch_end, :].to(device=hs.device, dtype=hs.dtype)
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
    span_start = None
    span_end = None
    if eval_span is not None:
        span_start, span_end = eval_span
        eval_start = start + span_start
        eval_end = start + span_end
        ref_ids = ref_ids[span_start:span_end]

    # Patch span: default to full reference span, or eval span if provided
    patch_start = start
    patch_end = end
    if patch_scope == "span" and eval_span is not None:
        patch_start = start + span_start
        patch_end = start + span_end

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
            hs[:, start, :] = source_mlp_hidden_all[:, start, :].to(device=hs.device, dtype=hs.dtype)
        else:
            hs[:, patch_start:patch_end, :] = source_mlp_hidden_all[:, patch_start:patch_end, :].to(device=hs.device, dtype=hs.dtype)
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


def _run_s1_s2_side_by_side_logprob_batch(
    retain, unlearn, full, tokenizer, prefix_data, layer_list,
    em_threshold=0.5, patch_mode="layer", unlearn_name="unlearn",
    em_scope="full", entity_source="gt", log_mismatch=False, mismatch_max=5,
    log_file=None, delta_threshold=0.02, patch_scope="span",
    reference="gt", log_span=False, reference_scope="continuation",
    s1_cache_entries=None, save_s1_cache=False,
    batch_size: int = 1
):
    """Batched log-prob S1/S2 run (layer patching only)."""
    results = []
    skipped_indices = []
    skipped_records = []
    gk_count = 0
    categories = None

    s1_cache_map = None
    if s1_cache_entries:
        s1_cache_map = {e["idx"]: e for e in s1_cache_entries}

    s1_cache_out = [] if save_s1_cache else None

    def flush_batch(batch_ctxs, batch_meta):
        nonlocal s1_cache_map
        if not batch_ctxs:
            return

        device = next(full.parameters()).device
        input_ids, attention_mask, meta = _prepare_batch_inputs(batch_ctxs, tokenizer, device)

        # Initialize per-example fields
        for m in batch_meta:
            m["s1_details"] = []
            m["s2_details"] = []
            m["s1_deltas"] = []
            m["s2_deltas"] = []
            m["s1_lost"] = 0
            m["s2_lost"] = 0

        if s1_cache_map:
            missing_idx = None
            for m in batch_meta:
                if s1_cache_map.get(m["idx"]) is None:
                    missing_idx = m["idx"]
                    break
            if missing_idx is not None:
                if log_file:
                    log_file.write(
                        f"[WARN] S1 cache missing idx={missing_idx}; "
                        "falling back to on-the-fly S1 computation.\n"
                    )
                    log_file.flush()
                s1_cache_map = None

        if s1_cache_map:
            for m in batch_meta:
                entry = s1_cache_map.get(m["idx"])
                if entry is None:
                    raise ValueError(f"S1 cache missing idx={m['idx']}")
                m["full_score"] = entry["full_score"]
                s1_scores = entry["s1_scores"]
                s1_deltas = entry["s1_deltas"]
                s1_status = entry["s1_status"]
                for layer, score, delta, status in zip(layer_list, s1_scores, s1_deltas, s1_status):
                    m["s1_details"].append({
                        "layer": layer,
                        "score": score,
                        "delta": delta,
                        "status": status
                    })
                m["s1_deltas"] = list(s1_deltas)
                m["s1_lost"] = sum(1 for s in s1_status if s == "LOST")
        else:
            full_scores = compute_logprob_teacher_forcing_baseline_batch_with_inputs(
                full, input_ids, attention_mask, meta
            )
            for m, fs in zip(batch_meta, full_scores):
                m["full_score"] = fs

        # Precompute source hidden states once per batch
        unlearn_hidden = _compute_hidden_states_batch(unlearn, input_ids, attention_mask)
        retain_hidden = None
        if not s1_cache_map:
            retain_hidden = _compute_hidden_states_batch(retain, input_ids, attention_mask)

        for li, layer in enumerate(layer_list):
            if not s1_cache_map:
                s1_scores = compute_logprob_teacher_forcing_layer_batch_with_inputs(
                    full, input_ids, attention_mask, meta,
                    retain_hidden[layer + 1], layer, patch_scope=patch_scope
                )
            s2_scores = compute_logprob_teacher_forcing_layer_batch_with_inputs(
                full, input_ids, attention_mask, meta,
                unlearn_hidden[layer + 1], layer, patch_scope=patch_scope
            )

            for j, meta_item in enumerate(batch_meta):
                full_score = meta_item["full_score"]
                s1_score = s1_scores[j] if not s1_cache_map else meta_item["s1_details"][li]["score"]
                s2_score = s2_scores[j]

                s1_delta = full_score - s1_score
                s2_delta = full_score - s2_score

                s1_status = "LOST" if s1_delta > delta_threshold else "KEPT"
                s2_status = "LOST" if s2_delta > delta_threshold else "KEPT"
                if not s1_cache_map and s1_status == "LOST":
                    meta_item["s1_lost"] += 1
                if s2_status == "LOST":
                    meta_item["s2_lost"] += 1

                if not s1_cache_map:
                    meta_item["s1_deltas"].append(s1_delta)
                meta_item["s2_deltas"].append(s2_delta)

                if not s1_cache_map:
                    meta_item["s1_details"].append({
                        "layer": layer,
                        "score": s1_score,
                        "delta": s1_delta,
                        "status": s1_status
                    })
                meta_item["s2_details"].append({
                    "layer": layer,
                    "score": s2_score,
                    "delta": s2_delta,
                    "status": s2_status
                })

        for meta in batch_meta:
            s1_details = meta["s1_details"]
            s2_details = meta["s2_details"]
            s1_deltas = meta["s1_deltas"]
            s2_deltas = meta["s2_deltas"]
            layer_records = build_layer_records(
                s1_details=s1_details,
                s2_details=s2_details,
                full_score=meta["full_score"],
            )

            ft_layers = [s1["layer"] for s1 in s1_details if s1["status"] == "LOST"]
            erased_layers = [s2["layer"] for s2 in s2_details if s2["status"] == "LOST" and s2["layer"] in ft_layers]

            # FT-only UDS with per-layer clipping to avoid overshoot.
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
            uds = numer / denom if denom > 0 else None

            # Build side-by-side log
            log_msg = f"\n{'='*80}\n"
            log_msg += f"[{meta['index']}/{meta['total']}] Example {meta['idx']}\n"
            log_msg += f"  Q: {meta['question']}\n"
            log_msg += f"  Prefix: '{meta['prefix']}'\n"
            log_msg += f"  GT (entity): '{meta['gt_entity']}'\n"
            log_msg += f"  Eval entity ({meta['entity_source']}): '{meta['entity_text']}'\n"
            if log_span:
                if "entity_span" in meta and isinstance(meta["entity_span"], dict):
                    es = meta["entity_span"]
                    log_msg += f"  Entity span (dataset tokens): {es.get('start')}:{es.get('end')}\n"
                if meta["eval_span"] is not None:
                    log_msg += f"  Eval span (reference tokens): {meta['eval_span'][0]}:{meta['eval_span'][1]}\n"
            log_msg += f"  EM scope: {em_scope}\n"
            log_msg += f"  Reference source: {meta['reference_source']}\n"
            log_msg += f"  Reference text: \"{clean_text(meta['reference_text'])}\"\n"
            log_msg += f"  Full baseline: \"{clean_text(meta['full_gen'])}\"\n"
            log_msg += f"  Retain baseline: \"{clean_text(meta['retain_gen'])}\"\n"
            log_msg += f"  {unlearn_name} baseline: \"{clean_text(meta['unlearn_gen'])}\"\n"
            log_msg += f"  Full log-prob (ref span): {meta['full_score']:.3f}\n"
            log_msg += f"{'='*80}\n"
            log_msg += (
                f"  {'Layer':<6} | {'P(Full)':<14} | {'S1 (Retain→Full)':<25} | "
                f"{'S2 ('+unlearn_name+'→Full)':<25} | {'Δ2-Δ1':<8}\n"
            )
            log_msg += f"  {'-'*6} | {'-'*14} | {'-'*25} | {'-'*25} | {'-'*8}\n"

            for s1, s2, lr in zip(s1_details, s2_details, layer_records):
                p_str = f"logp={lr['p_score']:.3f}" if lr.get("p_score") is not None else "N/A"
                s1_str = f"logp={s1['score']:.3f} Δ={s1['delta']:.3f} [{s1['status']}]"
                s2_str = f"logp={s2['score']:.3f} Δ={s2['delta']:.3f} [{s2['status']}]"
                gap = lr.get("delta_gap")
                gap_str = f"{gap:+.3f}" if gap is not None else "N/A"
                log_msg += f"  L{s1['layer']:02d}   | {p_str:<14} | {s1_str:<25} | {s2_str:<25} | {gap_str:<8}\n"

            log_msg += f"  {'-'*6} | {'-'*14} | {'-'*25} | {'-'*25} | {'-'*8}\n"
            log_msg += f"  FT layers (S1 LOST): {ft_layers}\n"
            log_msg += f"  Erased layers (S2 LOST ∩ FT): {erased_layers}\n"
            log_msg += f"  UDS = {uds:.3f}\n" if uds is not None else f"  UDS = N/A (no FT signal)\n"

            if log_file:
                log_file.write(log_msg)
                log_file.flush()

            results.append({
                "idx": meta["idx"],
                "question": meta["question"],
                "prefix": meta["prefix"],
                "entity": meta["entity"],
                "gt_entity": meta["gt_entity"],
                "entity_text": meta["entity_text"],
                "entity_source": meta["entity_source"],
                "prompt": meta["prompt"],
                "reference_source": meta["reference_source"],
                "reference_text": meta["reference_text"],
                "full_gen": meta["full_gen"],
                "retain_gen": meta["retain_gen"],
                "unlearn_gen": meta["unlearn_gen"],
                "s1_lost": meta["s1_lost"],
                "s2_lost": meta["s2_lost"],
                "ft_layers": ft_layers,
                "erased_layers": erased_layers,
                "uds": uds,
                "is_gk": None,
                "retain_em_full": None,
                "full_score": meta["full_score"],
                "p_score": meta["full_score"],
                "eval_scope": em_scope,
                "eval_span": meta["eval_span"],
                "entity_span": meta.get("entity_span"),
                "layer_records": layer_records,
                "s1_scores": [d.get("score") for d in s1_details],
                "s2_scores": [d.get("score") for d in s2_details],
                "s1_deltas": [d.get("delta") for d in s1_details],
                "s2_deltas": [d.get("delta") for d in s2_details],
                "s1_details": s1_details,
                "s2_details": s2_details
            })

            if save_s1_cache:
                s1_cache_out.append({
                    "idx": meta["idx"],
                    "full_score": meta["full_score"],
                    "s1_scores": [d["score"] for d in s1_details],
                    "s1_deltas": [d["delta"] for d in s1_details],
                    "s1_status": [d["status"] for d in s1_details],
                })

    batch_ctxs = []
    batch_meta = []
    total = len(prefix_data)

    for i, item in enumerate(tqdm(prefix_data, desc=f"S1/S2 {patch_mode} (batched)")):
        question = item["question"]
        prefix = item["prefix"]
        entity = item["entity"]
        gt_entity = item.get("gt_entity", entity)
        idx = item["idx"]

        if reference_scope == "full_answer":
            prompt = f"Question: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer: {prefix}"

        if "full_output" in item:
            full_gen = item["full_output"]
        else:
            full_gen = generate_baseline(full, tokenizer, prompt, max_new_tokens=30)

        retain_gen = generate_baseline(retain, tokenizer, prompt, max_new_tokens=30)
        unlearn_gen = generate_baseline(unlearn, tokenizer, prompt, max_new_tokens=30)

        reference_source = reference
        if reference_source == "gt":
            answer_text = item["answer"]
            if reference_scope == "continuation":
                if answer_text.startswith(prefix):
                    answer_text = answer_text[len(prefix):]
            reference_text = normalize_reference_for_eval(prompt, answer_text)
        else:
            reference_text = normalize_reference_for_eval(prompt, full_gen)

        if entity_source == "gt":
            entity_text = gt_entity
        else:
            entity_text = item.get("full_entity", entity)

        eval_span = get_eval_span(tokenizer, reference_text, entity_text, em_scope)
        if em_scope == "entity" and eval_span is None:
            skipped_indices.append(idx)
            skipped_records.append({
                "idx": idx,
                "reason": "entity_span_not_found",
                "question": question,
                "prefix": prefix,
                "entity": entity,
                "gt_entity": gt_entity,
                "entity_text": entity_text,
                "entity_source": entity_source,
                "reference_source": reference_source,
                "reference_text": reference_text,
                "full_gen": full_gen,
                "retain_gen": retain_gen,
                "unlearn_gen": unlearn_gen,
            })
            if log_file:
                log_msg = f"\n{'='*80}\n"
                log_msg += f"[{i+1}/{len(prefix_data)}] Example {idx} | SKIPPED (entity span not found)\n"
                log_msg += f"  Q: {question}\n"
                log_msg += f"  Prefix: '{prefix}'\n"
                log_msg += f"  GT (entity): '{gt_entity}'\n"
                log_msg += f"  Eval entity ({entity_source}): '{entity_text}'\n"
                if log_span:
                    if "entity_span" in item and isinstance(item["entity_span"], dict):
                        es = item["entity_span"]
                        log_msg += f"  Entity span (dataset tokens): {es.get('start')}:{es.get('end')}\n"
                    if eval_span is not None:
                        log_msg += f"  Eval span (reference tokens): {eval_span[0]}:{eval_span[1]}\n"
                log_msg += f"  Reference source: {reference_source}\n"
                log_msg += f"  Reference text: \"{clean_text(reference_text)}\"\n"
                log_msg += f"  Full baseline: \"{clean_text(full_gen)}\"\n"
                log_msg += f"{'='*80}\n"
                log_file.write(log_msg)
                log_file.flush()
            continue

        ctx = build_logprob_ctx(tokenizer, prompt, reference_text, eval_span, patch_scope)
        if ctx is None:
            skipped_indices.append(idx)
            skipped_records.append({
                "idx": idx,
                "reason": "invalid_eval_context",
                "question": question,
                "prefix": prefix,
                "entity": entity,
                "gt_entity": gt_entity,
                "entity_text": entity_text,
                "entity_source": entity_source,
                "reference_source": reference_source,
                "reference_text": reference_text,
                "full_gen": full_gen,
                "retain_gen": retain_gen,
                "unlearn_gen": unlearn_gen,
            })
            if log_file:
                log_msg = f"\n{'='*80}\n"
                log_msg += f"[{i+1}/{len(prefix_data)}] Example {idx} | SKIPPED (invalid eval context)\n"
                log_msg += f"  Q: {question}\n"
                log_msg += f"  Prefix: '{prefix}'\n"
                log_msg += f"  GT (entity): '{gt_entity}'\n"
                log_msg += f"  Eval entity ({entity_source}): '{entity_text}'\n"
                log_msg += f"  Reference source: {reference_source}\n"
                log_msg += f"  Reference text: \"{clean_text(reference_text)}\"\n"
                log_msg += f"  Full baseline: \"{clean_text(full_gen)}\"\n"
                log_msg += f"{'='*80}\n"
                log_file.write(log_msg)
                log_file.flush()
            continue

        batch_ctxs.append(ctx)
        batch_meta.append({
            "index": i + 1,
            "total": total,
            "idx": idx,
            "question": question,
            "prefix": prefix,
            "entity": entity,
            "gt_entity": gt_entity,
            "entity_text": entity_text,
            "entity_source": entity_source,
            "prompt": prompt,
            "full_gen": full_gen,
            "retain_gen": retain_gen,
            "unlearn_gen": unlearn_gen,
            "reference_text": reference_text,
            "reference_source": reference_source,
            "eval_span": eval_span,
            "entity_span": item.get("entity_span"),
        })

        if len(batch_ctxs) >= batch_size:
            flush_batch(batch_ctxs, batch_meta)
            batch_ctxs = []
            batch_meta = []

    flush_batch(batch_ctxs, batch_meta)

    uds_values = [r["uds"] for r in results if r["uds"] is not None]
    avg_uds = sum(uds_values) / len(uds_values) if uds_values else 0.0

    return results, gk_count, categories, avg_uds, skipped_indices, skipped_records, s1_cache_out


def run_s1_s2_side_by_side(retain, unlearn, full, tokenizer, prefix_data, layer_list,
                            em_threshold=0.5, patch_mode="layer", unlearn_name="unlearn",
                            em_scope="full", entity_source="gt", em_exact=False,
                            log_mismatch=False, mismatch_max=5, log_file=None,
                            metric="em", delta_threshold=0.02, patch_scope="span",
                            reference="gt", log_span=False, reference_scope="continuation",
                            batch_size: int = 1, s1_cache_entries=None, save_s1_cache=False):
    """Run S1 and S2 side by side for each example using teacher forcing EM or log-prob."""
    if metric == "logprob" and patch_mode == "layer" and not log_mismatch:
        return _run_s1_s2_side_by_side_logprob_batch(
            retain, unlearn, full, tokenizer, prefix_data, layer_list,
            em_threshold, patch_mode, unlearn_name, em_scope, entity_source,
            log_mismatch, mismatch_max, log_file, delta_threshold, patch_scope,
            reference, log_span, reference_scope, s1_cache_entries, save_s1_cache, batch_size
        )

    results = []
    skipped_indices = []
    skipped_records = []

    # Counters for summary
    gk_count = 0  # GK: retain predicts reference tokens (teacher forcing EM)
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

        if reference_scope == "full_answer":
            prompt = f"Question: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer: {prefix}"

        # Use pre-computed full_output if available (v6 dataset), otherwise generate
        if "full_output" in item:
            full_gen = item["full_output"]
        else:
            full_gen = generate_baseline(full, tokenizer, prompt, max_new_tokens=30)

        # Get baselines for display
        retain_gen = generate_baseline(retain, tokenizer, prompt, max_new_tokens=30)
        unlearn_gen = generate_baseline(unlearn, tokenizer, prompt, max_new_tokens=30)

        # Reference for EM/log-prob = GT answer (continuation) or Full baseline output
        reference_source = reference
        if reference_source == "gt":
            answer_text = item["answer"]
            if reference_scope == "continuation":
                # Use continuation after prefix to align with teacher forcing setup.
                if answer_text.startswith(prefix):
                    answer_text = answer_text[len(prefix):]
            reference_text = normalize_reference_for_eval(prompt, answer_text)
        else:
            reference_text = normalize_reference_for_eval(prompt, full_gen)

        # Select entity span source
        if entity_source == "gt":
            entity_text = gt_entity
        else:
            entity_text = item.get("full_entity", entity)

        eval_span = get_eval_span(tokenizer, reference_text, entity_text, em_scope)
        if em_scope == "entity" and eval_span is None:
            skipped_indices.append(idx)
            skipped_records.append({
                "idx": idx,
                "reason": "entity_span_not_found",
                "question": question,
                "prefix": prefix,
                "entity": entity,
                "gt_entity": gt_entity,
                "entity_text": entity_text,
                "entity_source": entity_source,
                "reference_source": reference_source,
                "reference_text": reference_text,
                "full_gen": full_gen,
                "retain_gen": retain_gen,
                "unlearn_gen": unlearn_gen,
            })
            if log_file:
                log_msg = f"\n{'='*80}\n"
                log_msg += f"[{i+1}/{len(prefix_data)}] Example {idx} | SKIPPED (entity span not found)\n"
                log_msg += f"  Q: {question}\n"
                log_msg += f"  Prefix: '{prefix}'\n"
                log_msg += f"  GT (entity): '{gt_entity}'\n"
                log_msg += f"  Eval entity ({entity_source}): '{entity_text}'\n"
                if log_span:
                    if "entity_span" in item and isinstance(item["entity_span"], dict):
                        es = item["entity_span"]
                        log_msg += f"  Entity span (dataset tokens): {es.get('start')}:{es.get('end')}\n"
                    if eval_span is not None:
                        log_msg += f"  Eval span (reference tokens): {eval_span[0]}:{eval_span[1]}\n"
                log_msg += f"  Reference source: {reference_source}\n"
                log_msg += f"  Reference text: \"{clean_text(reference_text)}\"\n"
                log_msg += f"  Full baseline: \"{clean_text(full_gen)}\"\n"
                log_msg += f"{'='*80}\n"
                log_file.write(log_msg)
                log_file.flush()
            continue

        # Precompute full baseline score for log-prob metric
        full_score = None
        if metric == "logprob":
            full_score = compute_logprob_teacher_forcing_baseline(
                    full, tokenizer, prompt, reference_text, eval_span=eval_span
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
                        retain, full, tokenizer, prompt, reference_text, layer,
                        eval_span=eval_span, exact_match=em_exact, return_details=True,
                        patch_scope=patch_scope
                    )
                else:
                    s1_em = em_fn(
                        retain, full, tokenizer, prompt, reference_text, layer,
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
                    retain, full, tokenizer, prompt, reference_text, layer,
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
                        unlearn, full, tokenizer, prompt, reference_text, layer,
                        eval_span=eval_span, exact_match=em_exact, return_details=True,
                        patch_scope=patch_scope
                    )
                else:
                    s2_em = em_fn(
                        unlearn, full, tokenizer, prompt, reference_text, layer,
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
                    unlearn, full, tokenizer, prompt, reference_text, layer,
                    eval_span=eval_span, patch_scope=patch_scope
                )
                s2_delta = full_score - s2_score
                s2_deltas.append(s2_delta)
                s2_status = "LOST" if s2_delta > delta_threshold else "KEPT"
                if s2_status == "LOST":
                    s2_lost += 1
                s2_detail = {"layer": layer, "score": s2_score, "delta": s2_delta, "status": s2_status}
            s2_details.append(s2_detail)

        # GK definition: retain teacher forcing EM on reference
        if metric == "em":
            retain_em_full = compute_em_teacher_forcing_baseline(
                retain, tokenizer, prompt, reference_text, eval_span=eval_span, exact_match=em_exact
            )
            is_gk = retain_em_full >= em_threshold
            if is_gk:
                gk_count += 1
        else:
            retain_em_full = None
            is_gk = None

        layer_records = build_layer_records(
            s1_details=s1_details,
            s2_details=s2_details,
            full_score=full_score if metric == "logprob" else None,
        )

        # Build side-by-side log
        log_msg = f"\n{'='*80}\n"
        log_msg += f"[{i+1}/{len(prefix_data)}] Example {idx}\n"
        log_msg += f"  Q: {question}\n"
        log_msg += f"  Prefix: '{prefix}'\n"
        log_msg += f"  GT (entity): '{gt_entity}'\n"
        log_msg += f"  Eval entity ({entity_source}): '{entity_text}'\n"
        if log_span:
            if "entity_span" in item and isinstance(item["entity_span"], dict):
                es = item["entity_span"]
                log_msg += f"  Entity span (dataset tokens): {es.get('start')}:{es.get('end')}\n"
            if eval_span is not None:
                log_msg += f"  Eval span (reference tokens): {eval_span[0]}:{eval_span[1]}\n"
        log_msg += f"  EM scope: {em_scope}\n"
        log_msg += f"  Reference source: {reference_source}\n"
        log_msg += f"  Reference text: \"{clean_text(reference_text)}\"\n"
        log_msg += f"  Full baseline: \"{clean_text(full_gen)}\"\n"
        log_msg += f"  Retain baseline: \"{clean_text(retain_gen)}\"\n"
        log_msg += f"  {unlearn_name} baseline: \"{clean_text(unlearn_gen)}\"\n"
        if metric == "logprob" and full_score is not None:
            log_msg += f"  Full log-prob (ref span): {full_score:.3f}\n"
        log_msg += f"{'='*80}\n"
        if metric == "logprob":
            log_msg += (
                f"  {'Layer':<6} | {'P(Full)':<14} | {'S1 (Retain→Full)':<25} | "
                f"{'S2 ('+unlearn_name+'→Full)':<25} | {'Δ2-Δ1':<8}\n"
            )
            log_msg += f"  {'-'*6} | {'-'*14} | {'-'*25} | {'-'*25} | {'-'*8}\n"
        else:
            log_msg += f"  {'Layer':<6} | {'S1 (Retain→Full)':<25} | {'S2 ('+unlearn_name+'→Full)':<25}\n"
            log_msg += f"  {'-'*6} | {'-'*25} | {'-'*25}\n"

        for s1, s2, lr in zip(s1_details, s2_details, layer_records):
            if metric == "em":
                s1_str = f"EM={s1['em']:.2f} [{s1['status']}]"
                s2_str = f"EM={s2['em']:.2f} [{s2['status']}]"
                log_msg += f"  L{s1['layer']:02d}   | {s1_str:<25} | {s2_str:<25}\n"
            else:
                p_str = f"logp={lr['p_score']:.3f}" if lr.get("p_score") is not None else "N/A"
                s1_str = f"logp={s1['score']:.3f} Δ={s1['delta']:.3f} [{s1['status']}]"
                s2_str = f"logp={s2['score']:.3f} Δ={s2['delta']:.3f} [{s2['status']}]"
                gap = lr.get("delta_gap")
                gap_str = f"{gap:+.3f}" if gap is not None else "N/A"
                log_msg += f"  L{s1['layer']:02d}   | {p_str:<14} | {s1_str:<25} | {s2_str:<25} | {gap_str:<8}\n"
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

        if metric == "logprob":
            log_msg += f"  {'-'*6} | {'-'*14} | {'-'*25} | {'-'*25} | {'-'*8}\n"
        else:
            log_msg += f"  {'-'*6} | {'-'*25} | {'-'*25}\n"

        # Calculate UDS and FT/Erased layers
        ft_layers = [s1["layer"] for s1 in s1_details if s1["status"] == "LOST"]
        erased_layers = [s2["layer"] for s2 in s2_details if s2["status"] == "LOST" and s2["layer"] in ft_layers]

        if metric == "em":
            uds = len(erased_layers) / len(ft_layers) if ft_layers else 0.0
        else:
            # FT-only UDS with per-layer clipping to avoid overshoot.
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
                uds = numer / denom
            else:
                uds = None

        log_msg += f"  FT layers (S1 LOST): {ft_layers}\n"
        log_msg += f"  Erased layers (S2 LOST ∩ FT): {erased_layers}\n"
        if metric == "em":
            log_msg += f"  UDS = {len(erased_layers)}/{len(ft_layers)} = {uds:.3f}\n" if ft_layers else f"  UDS = N/A (no FT layers)\n"
            log_msg += f"  GK (retain TF EM vs Full ref, info): {is_gk} (EM={retain_em_full:.2f})\n"
        else:
            log_msg += f"  UDS = {uds:.3f}\n" if uds is not None else f"  UDS = N/A (no FT signal)\n"

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
            "prompt": prompt,
            "reference_source": reference_source,
            "reference_text": reference_text,
            "full_gen": full_gen,
            "retain_gen": retain_gen,
            "unlearn_gen": unlearn_gen,
            "s1_lost": s1_lost,
            "s2_lost": s2_lost,
            "ft_layers": ft_layers,
            "erased_layers": erased_layers,
            "uds": uds,
            "is_gk": is_gk,
            "retain_em_full": retain_em_full,
            "full_score": full_score if metric == "logprob" else None,
            "p_score": full_score if metric == "logprob" else None,
            "eval_scope": em_scope,
            "eval_span": eval_span,
            "entity_span": item.get("entity_span"),
            "layer_records": layer_records,
            "s1_scores": [d.get("score") for d in s1_details] if metric == "logprob" else None,
            "s2_scores": [d.get("score") for d in s2_details] if metric == "logprob" else None,
            "s1_deltas": [d.get("delta") for d in s1_details] if metric == "logprob" else None,
            "s2_deltas": [d.get("delta") for d in s2_details] if metric == "logprob" else None,
            "s1_details": s1_details,
            "s2_details": s2_details
        })

    # Calculate average UDS (all evaluable examples with FT signal)
    uds_values = [r["uds"] for r in results if r["uds"] is not None]
    avg_uds = sum(uds_values) / len(uds_values) if uds_values else 0.0

    return results, gk_count, categories, avg_uds, skipped_indices, skipped_records, None


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
    parser.add_argument("--layers", type=str, default="auto",
                        help="Layer range (e.g., 0-15) or 'auto' for all layers.")
    parser.add_argument("--metric", type=str, choices=["em", "logprob"], default="logprob")
    parser.add_argument("--em_threshold", type=float, default=1.0)
    parser.add_argument("--delta_threshold", type=float, default=0.05)
    parser.add_argument("--patch_scope", type=str, choices=["span", "boundary"], default="span",
                        help="Patch reference span or only boundary (last prompt token)")
    parser.add_argument("--em_type", type=str, choices=["token", "exact"], default="token")
    parser.add_argument("--em_scope", type=str, choices=["full", "entity"], default="entity")
    parser.add_argument("--entity_source", type=str, choices=["gt", "full"], default="gt")
    parser.add_argument("--reference", type=str, choices=["gt", "full"], default="gt",
                        help="Reference text for teacher forcing (GT answer or Full baseline)")
    parser.add_argument("--reference_scope", type=str, choices=["continuation", "full_answer"],
                        default="continuation",
                        help="Reference scope: continuation after prefix or full answer")
    parser.add_argument("--data_path", type=str, default=PREFIX_DATA_PATH)
    parser.add_argument("--log_mismatch", action="store_true")
    parser.add_argument("--log_span", action="store_true",
                        help="Log entity/eval span token indices for debugging")
    parser.add_argument("--mismatch_max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA visible devices, e.g., '0' or '0,1'.",
    )
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for log-prob metric (layer mode only).")
    parser.add_argument("--s1_cache", action="store_true", default=True,
                        help="Enable S1 (retain→full) caching for log-prob runs.")
    parser.add_argument("--no_s1_cache", action="store_false", dest="s1_cache",
                        help="Disable S1 caching.")
    parser.add_argument("--s1_cache_dir", type=str, default="runs/s1_cache",
                        help="Directory to store/load S1 cache files.")
    parser.add_argument("--mode", type=str, choices=["layer", "mlp"], default="layer")
    parser.add_argument("--unlearn_model", type=str, required=True,
                        help="Unlearn model (e.g., simnpo, idknll)")
    parser.add_argument("--full_model", type=str, default=TOFU_FULL_MODEL,
                        help="Full model checkpoint ID.")
    parser.add_argument("--retain_model", type=str, default=TOFU_RETAIN_MODEL,
                        help="Retain model checkpoint ID (used as S1 source).")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Torch dtype for model loading.")
    parser.add_argument("--device_map", type=str, default="cuda",
                        help="Transformers device_map (e.g., cuda, auto, cpu).")
    parser.add_argument("--attn_implementation", type=str, default=None,
                        help="Attention implementation: eager, sdpa, or flash_attention_2")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional short label used in output directory naming.")
    parser.add_argument("--out_root", type=str, default="runs",
                        help="Parent directory for auto-generated run folders.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Explicit output directory. If set, out_root/run_name are ignored.")
    args = parser.parse_args()

    if args.gpu:
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
    print(f"Reference scope: {args.reference_scope}")
    print(f"Patch scope: {args.patch_scope}")
    if args.metric == "logprob" and args.mode == "layer":
        print(f"Batch size: {args.batch_size}")
    if args.log_mismatch:
        print(f"Mismatch logging: enabled (max {args.mismatch_max})")
    print(f"Full model: {args.full_model}")
    print(f"Retain model: {args.retain_model}")
    print(f"Dtype: {args.dtype}")
    print(f"Device map: {args.device_map}")
    print(f"Unlearn model: {args.unlearn_model}")
    print()

    # Load models
    print("Loading models...")
    attn_impl = args.attn_implementation
    tokenizer = load_tokenizer(args.full_model)
    retain = load_model_for_exp(
        args.retain_model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=attn_impl,
    )
    full = load_model_for_exp(
        args.full_model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=attn_impl,
    )

    if args.unlearn_model == "full":
        unlearn_model_id = args.full_model
    elif args.unlearn_model == "retain":
        unlearn_model_id = args.retain_model
    else:
        unlearn_model_id = get_model_id(args.unlearn_model)
    print(f"Loading unlearn model: {unlearn_model_id}")
    unlearn = load_model_for_exp(
        unlearn_model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=attn_impl,
    )

    n_layers = get_num_layers(full)
    if args.layers.strip().lower() == "auto":
        layer_list = list(range(n_layers))
    else:
        layer_list = parse_layers(args.layers, n_layers)
    print(f"Layers: {layer_list}")

    # Load data
    prefix_data = load_prefix_data(args.data_path)
    print(f"Loaded {len(prefix_data)} examples")

    if args.num_examples:
        prefix_data = prefix_data[:args.num_examples]
        print(f"Using first {len(prefix_data)} examples")

    # Create output directory (date first)
    if args.out_dir:
        out_dir = args.out_dir
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        run_tag = args.run_name if args.run_name else args.unlearn_model
        run_tag = sanitize_name_for_path(run_tag)
        out_dir = f"{args.out_root}/{timestamp}_tf_{run_tag}_{args.mode}"
    safe_mkdir(out_dir)

    # Run S1/S2 side by side
    print("\n" + "=" * 40)
    print(f"Running S1/S2 ({args.mode}, teacher forcing)")
    print("=" * 40)

    log_path = f"{out_dir}/run.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"S1/S2 {args.mode.upper()} Patching with Teacher Forcing\n")
        if args.metric == "em":
            log_file.write("Metric: EM (token accuracy), reference = dataset reference\n")
            log_file.write(f"EM type: {args.em_type}\n")
            log_file.write(f"EM threshold: {args.em_threshold}\n")
        else:
            log_file.write("Metric: log-prob (reference tokens), reference = dataset reference\n")
            log_file.write(f"Delta threshold: {args.delta_threshold}\n")
        log_file.write(f"EM scope: {args.em_scope}\n")
        log_file.write(f"Entity source: {args.entity_source}\n")
        log_file.write(f"Reference scope: {args.reference_scope}\n")
        log_file.write(f"Patch scope: {args.patch_scope}\n")
        log_file.write(f"Data path: {args.data_path}\n")
        log_file.write(f"Reference source: {args.reference}\n")
        log_file.write(f"Full model: {args.full_model}\n")
        log_file.write(f"Retain model: {args.retain_model}\n")
        log_file.write(f"Source model id: {unlearn_model_id}\n")
        log_file.write(f"Device map: {args.device_map}\n")
        log_file.write(f"Dtype: {args.dtype}\n")
        if args.metric == "logprob" and args.mode == "layer":
            log_file.write(f"Batch size: {args.batch_size}\n")
        if args.log_mismatch:
            log_file.write(f"Mismatch logging: enabled (max {args.mismatch_max})\n")
        if args.log_span:
            log_file.write("Span logging: enabled\n")
        log_file.write(f"S1: Retain → Full | S2: {args.unlearn_model} → Full\n")
        log_file.write(f"Layers: {layer_list}\n")
        log_file.write(f"Examples: {len(prefix_data)}\n\n")

        s1_cache_entries = None
        s1_cache_path = None
        save_s1_cache = False
        if args.metric == "logprob" and args.mode == "layer" and args.s1_cache:
            cache_cfg = {
                "data_path": args.data_path,
                "reference": args.reference,
                "reference_scope": args.reference_scope,
                "entity_source": args.entity_source,
                "em_scope": args.em_scope,
                "em_type": args.em_type,
                "patch_scope": args.patch_scope,
                "metric": args.metric,
                "delta_threshold": args.delta_threshold,
                "layers": layer_list,
                "full_model": args.full_model,
                "retain_model": args.retain_model,
                "device_map": args.device_map,
                "dtype": args.dtype,
                "num_examples": len(prefix_data),
                "example_ids": [item["idx"] for item in prefix_data],
            }
            cache_key = _build_s1_cache_key(cache_cfg)
            s1_cache_path = Path(args.s1_cache_dir) / f"s1_cache_{cache_key}.json"
            s1_cache_entries = _load_s1_cache(s1_cache_path, cache_cfg)
            if s1_cache_entries is None:
                save_s1_cache = True
                log_file.write(f"S1 cache: MISS ({s1_cache_path})\n")
            else:
                log_file.write(f"S1 cache: HIT ({s1_cache_path}) entries={len(s1_cache_entries)}\n")

        start_time = time.time()
        results, gk_count, categories, avg_uds, skipped_indices, skipped_records, s1_cache_out = run_s1_s2_side_by_side(
            retain, unlearn, full, tokenizer, prefix_data, layer_list,
            args.em_threshold, args.mode, args.unlearn_model, args.em_scope, args.entity_source,
            em_exact, args.log_mismatch, args.mismatch_max, log_file,
            args.metric, args.delta_threshold, args.patch_scope, args.reference, args.log_span,
            args.reference_scope, args.batch_size, s1_cache_entries, save_s1_cache
        )
        elapsed = time.time() - start_time

        if save_s1_cache and s1_cache_path and s1_cache_out is not None:
            _save_s1_cache(s1_cache_path, cache_cfg, s1_cache_out)
            log_file.write(f"S1 cache saved: {s1_cache_path}\n")

        # Write summary at the end
        total = len(prefix_data)
        skipped_count = len(skipped_indices)
        evaluated = total - skipped_count
        skipped_reasons: Dict[str, int] = {}
        for rec in skipped_records:
            reason = rec.get("reason", "unknown")
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        sec_per_eval = elapsed / evaluated if evaluated else None

        summary_msg = f"\n\n{'='*80}\n"
        summary_msg += f"EXPERIMENT SUMMARY\n"
        summary_msg += f"{'='*80}\n"
        summary_msg += f"Total examples: {total}\n"
        summary_msg += f"Metric: {args.metric}\n"
        if args.metric == "em":
            summary_msg += f"EM type: {args.em_type}\n"
        summary_msg += f"EM scope: {args.em_scope}\n"
        summary_msg += f"Entity source: {args.entity_source}\n"
        summary_msg += f"Reference scope: {args.reference_scope}\n"
        summary_msg += f"Patch scope: {args.patch_scope}\n"
        if skipped_count:
            summary_msg += f"Skipped (all reasons): {skipped_count} ({100*skipped_count/total:.1f}%)\n"
            summary_msg += f"Skipped reasons: {skipped_reasons}\n"
        if evaluated:
            summary_msg += f"Evaluable (non-skipped): {evaluated} ({100*evaluated/total:.1f}%)\n"
            if args.metric == "em":
                summary_msg += f"GK (retain TF EM vs Full ref, info): {gk_count} ({100*gk_count/evaluated:.1f}%)\n"
        else:
            summary_msg += "Evaluable (non-skipped): 0 (0.0%)\n"
            if args.metric == "em":
                summary_msg += "GK (retain TF EM vs Full ref, info): 0 (0.0%)\n"
        summary_msg += f"UDS: {avg_uds:.3f}\n"
        if evaluated:
            summary_msg += f"Time: {elapsed:.1f}s | {elapsed/evaluated:.3f}s per evaluable example\n"
        summary_msg += f"{'='*80}\n"

        log_file.write(summary_msg)
        print(summary_msg)

    print(f"Log saved to: {log_path}")

    # Save JSON results
    summary = {
        "total_examples": total,
        "elapsed_sec": elapsed,
        "sec_per_evaluable": sec_per_eval,
        "metric": args.metric,
        "em_threshold": args.em_threshold,
        "delta_threshold": args.delta_threshold,
        "em_type": args.em_type,
        "em_scope": args.em_scope,
        "entity_source": args.entity_source,
        "reference_scope": args.reference_scope,
        "mode": args.mode,
        "unlearn_model": args.unlearn_model,
        "source_model_id": unlearn_model_id,
        "full_model": args.full_model,
        "retain_model": args.retain_model,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "gk_count": gk_count,
        "avg_uds": avg_uds,
        # backward-compat key (older parsers expected UDR)
        "avg_udr": avg_uds,
        "skipped_count": skipped_count,
        "skipped_indices": skipped_indices,
        "skipped_reasons": skipped_reasons,
    }

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    artifacts = write_detailed_artifacts(out_dir, results, skipped_records)
    summary["detailed_files"] = artifacts
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Detailed JSONL: {artifacts['detailed_jsonl']}")
    print(f"Layer CSV: {artifacts['layer_csv']}")
    print(f"Skipped examples: {artifacts['skipped_json']}")
    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()

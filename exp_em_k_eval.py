#!/usr/bin/env python3
"""
Two-Stage Experiment with EM Evaluation + Cosine Similarity Analysis

Stage 1: Retain → Full (Where is forget-set knowledge stored?)
Stage 2: Unlearn → Full (Where is knowledge erased?)

Evaluation:
- EM score = position-wise token match ratio
- Cosine similarity = hidden state similarity between source and target (Full)

Usage:
    python exp_em_k_eval.py --unlearn_model simnpo --example_indices 0,5,10
    python exp_em_k_eval.py --unlearn_model simnpo --num_examples 50

This script supports both targeted analysis (--example_indices) and batch runs (--num_examples).
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patchscope.models import load_model, load_tokenizer, get_num_layers
from patchscope.core import get_all_layers_hidden, forward_with_patch, generate_with_patch, generate_baseline
from patchscope.utils import set_seed, safe_mkdir, parse_layers
from patchscope.config import get_model_id


TOFU_FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
TOFU_RETAIN_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
PREFIX_DATA_PATH = "tofu_data/forget10_filtered_v3.json"  # Manual prefix + filtered (353 examples)
SUBJECT_POS_PATH = "tofu_data/forget10_subject_positions.json"  # Subject token positions


class TeeLogger:
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


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


def compute_em_score(generated: str, reference: str, tokenizer) -> float:
    """
    Compute EM score: position-wise token match ratio (Open-Unlearning style).
    EM = (# tokens matching at same position) / (# reference tokens)

    Reference can be GT entity OR Full model's output (for patching comparison).
    """
    gen_clean = clean_generated(generated)
    ref_clean = clean_generated(reference)

    if not gen_clean or not ref_clean:
        return 0.0

    # Tokenize
    gen_tokens = tokenizer.encode(gen_clean, add_special_tokens=False)
    ref_tokens = tokenizer.encode(ref_clean, add_special_tokens=False)

    if len(ref_tokens) == 0:
        return 0.0

    # Count position-wise matches (Open-Unlearning style)
    # Compare up to min length, then count matches
    match_count = sum(1 for g, r in zip(gen_tokens, ref_tokens) if g == r)

    return match_count / len(ref_tokens)


def plot_layer_heatmap(all_cos_data: List[Dict], tokenizer, input_ids,
                       example_idx: int, out_dir: str, layer_list: List[int]):
    """
    Create heatmaps showing cosine similarity across all layers and tokens.
    """
    n_layers = len(layer_list)
    n_tokens = len(all_cos_data[0]["s1"]["per_token"])

    # Build matrices
    s1_matrix = np.array([d["s1"]["per_token"] for d in all_cos_data])
    s2_matrix = np.array([d["s2"]["per_token"] for d in all_cos_data])

    # Decode each token individually to get readable text
    token_ids = input_ids[0].tolist()
    n_all_tokens = len(token_ids)

    # Match token labels to per_token data (when patch_all=False, only last token)
    if n_tokens < n_all_tokens:
        token_ids = token_ids[-n_tokens:]  # Last n tokens

    # Create readable token labels with 1-based position numbers
    token_labels = []
    start_pos = n_all_tokens - n_tokens + 1  # 1-based starting position
    for i, tid in enumerate(token_ids):
        tok_str = tokenizer.decode([tid])
        # Clean up for display
        tok_str = tok_str.replace('\n', '\\n').replace(' ', '_')
        if not tok_str.strip():
            tok_str = f"[{tid}]"
        pos = start_pos + i
        token_labels.append(f"{pos}:{tok_str[:8]}")  # Position:token (truncate to 8 chars)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n_layers * 0.4)))

    # S1 heatmap
    im1 = axes[0].imshow(s1_matrix, aspect='auto', cmap='Blues', vmin=0.5, vmax=1.0)
    axes[0].set_title('S1: Retain vs Full')
    axes[0].set_xlabel('Token')
    axes[0].set_ylabel('Layer')
    axes[0].set_xticks(range(n_tokens))
    axes[0].set_xticklabels(token_labels, rotation=90, fontsize=7)
    axes[0].set_yticks(range(n_layers))
    axes[0].set_yticklabels(layer_list)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # S2 heatmap
    im2 = axes[1].imshow(s2_matrix, aspect='auto', cmap='Oranges', vmin=0.5, vmax=1.0)
    axes[1].set_title('S2: Unlearn vs Full')
    axes[1].set_xlabel('Token')
    axes[1].set_ylabel('Layer')
    axes[1].set_xticks(range(n_tokens))
    axes[1].set_xticklabels(token_labels, rotation=90, fontsize=7)
    axes[1].set_yticks(range(n_layers))
    axes[1].set_yticklabels(layer_list)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.suptitle(f'Cosine Similarity Heatmap (Example {example_idx})', fontsize=14)
    plt.tight_layout()

    # Save plot
    plot_dir = os.path.join(out_dir, "plots")
    safe_mkdir(plot_dir)
    plt.savefig(os.path.join(plot_dir, f"heatmap_ex{example_idx}.png"), dpi=150)
    plt.close()


def compute_cosine_similarity(hidden1: torch.Tensor, hidden2: torch.Tensor) -> Dict:
    """
    Compute per-token cosine similarity between two hidden state tensors.

    Args:
        hidden1, hidden2: [B, D] for single token or [B, T, D] for all tokens

    Returns:
        Dict with 'per_token' (list), 'mean', 'min', 'max'
    """
    if hidden1.dim() == 2:
        # Single token: [B, D] -> compute single similarity
        sim = F.cosine_similarity(hidden1, hidden2, dim=-1).item()
        return {"per_token": [sim], "mean": sim, "min": sim, "max": sim}
    else:
        # All tokens: [B, T, D] -> compute per-token similarity
        # hidden1[0] and hidden2[0] are [T, D]
        h1 = hidden1[0]  # [T, D]
        h2 = hidden2[0]  # [T, D]
        per_token_sim = F.cosine_similarity(h1, h2, dim=-1)  # [T]
        per_token_list = per_token_sim.tolist()
        return {
            "per_token": per_token_list,
            "mean": per_token_sim.mean().item(),
            "min": per_token_sim.min().item(),
            "max": per_token_sim.max().item()
        }


def get_topk_with_patch(model, tokenizer, prompt, layer, hidden, k=3):
    """Get top-k predictions with patching."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    logits = forward_with_patch(
        model, inputs["input_ids"], inputs["attention_mask"],
        layer, hidden, patch_position=-1
    )

    probs = torch.softmax(logits[:, -1, :], dim=-1)
    topk_probs, topk_ids = torch.topk(probs[0], k=k)

    tokens = []
    for tid, p in zip(topk_ids.tolist(), topk_probs.tolist()):
        tokens.append({"token": tokenizer.decode([tid]).strip(), "prob": p})
    return tokens


def clean_text(s: str, max_len: int = 35) -> str:
    """Clean generated text for display."""
    if "." in s:
        s = s[:s.index(".") + 1]
    s = s.split("\n")[0].strip()
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def get_patch_positions(patch_types: List[str], idx: int, subject_pos_data: Dict) -> Optional[List[int]]:
    """
    Compute patch positions based on types.

    Args:
        patch_types: List of position types (last, all, subject, min_cos)
        idx: Example index
        subject_pos_data: Dict mapping idx to subject position info

    Returns:
        None for all tokens, or List[int] of specific positions
    """
    if "all" in patch_types:
        return None  # Patch all tokens

    positions = []
    subj_info = subject_pos_data.get(idx, {}) if subject_pos_data else {}
    subj_positions = subj_info.get("subject_positions", [])

    if "subject" in patch_types:
        positions.extend([sp["last_token_pos"] for sp in subj_positions])

    if "min_cos" in patch_types:
        if subj_positions:
            # Find occurrence with minimum last_cos_avg, patch its last_token_pos
            min_occ = min(subj_positions, key=lambda sp: sp.get("last_cos_avg", float('inf')))
            positions.append(min_occ["last_token_pos"])

    if "last" in patch_types:
        positions.append(-1)

    # Dedupe and fallback
    positions = list(set(positions)) if positions else [-1]
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlearn_model", type=str, default="simnpo")
    parser.add_argument("--example_indices", type=str, default=None,
                        help="Comma-separated example indices (prefix_data positions), e.g., '0,5,10'")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples (default: all)")
    parser.add_argument("--layers", type=str, default="0-15")
    parser.add_argument("--em_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--patch_positions", type=str, default="last",
                        help="Comma-separated position types: last, all, subject, min_cos")
    parser.add_argument("--patch_component", type=str, default="layer", choices=["layer", "mlp"],
                        help="Component to patch: layer (full layer output) or mlp (MLP output only)")
    parser.add_argument("--plot", action="store_true",
                        help="Enable heatmap plotting (disabled by default)")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    unlearn_model_id = get_model_id(args.unlearn_model)

    if args.out_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.out_dir = f"runs/{timestamp}_{args.unlearn_model}"
    safe_mkdir(args.out_dir)

    # Log file with method name and timestamp
    log_timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, f"{args.unlearn_model}_{log_timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[EM-based Experiment]")
    print(f"Method: {args.unlearn_model.upper()}")
    print(f"Time: {timestamp}")
    patch_types = [p.strip() for p in args.patch_positions.split(",")]
    print(f"Patch positions: {args.patch_positions}")
    print(f"Patch component: {args.patch_component}")
    print("=" * 130)
    print(f"Stage 1: Retain -> Full (Where is forget-set knowledge stored?)")
    print(f"Stage 2: {args.unlearn_model.upper()} -> Full (Where is knowledge erased?)")
    print(f"Metric: EM = consecutive token match ratio")
    print(f"  Hard: EM >= {args.em_threshold} -> OK | Soft: raw EM values")
    print("=" * 130)

    # Load models
    print("\nLoading models...")
    tokenizer = load_tokenizer(TOFU_FULL_MODEL)
    retain = load_model(TOFU_RETAIN_MODEL, dtype="bfloat16", device_map="cuda")
    full = load_model(TOFU_FULL_MODEL, dtype="bfloat16", device_map="cuda")
    unlearn = load_model(unlearn_model_id, dtype="bfloat16", device_map="cuda")

    n_layers = get_num_layers(full)
    layer_list = parse_layers(args.layers, n_layers)
    print(f"Layers: {layer_list}")

    # Load validated prefix data
    print(f"Loading prefix data from: {PREFIX_DATA_PATH}")
    prefix_data = load_prefix_data()
    print(f"Loaded {len(prefix_data)} validated examples")

    # Load subject positions if needed (for subject or min_cos)
    subject_pos_data = None
    if "subject" in patch_types or "min_cos" in patch_types:
        with open(SUBJECT_POS_PATH, "r", encoding="utf-8") as f:
            subject_pos_data = {d["idx"]: d for d in json.load(f)}
        print(f"Loaded subject positions for {len(subject_pos_data)} examples")

    dataset = load_dataset("locuslab/TOFU", "forget10", split="train")

    # Filter to specified example indices or num_examples
    if args.example_indices:
        example_indices = [int(x.strip()) for x in args.example_indices.split(",")]
        prefix_data = [prefix_data[i] for i in example_indices if i < len(prefix_data)]
        print(f"Selected {len(prefix_data)} examples at indices: {example_indices}")
    elif args.num_examples:
        prefix_data = prefix_data[:args.num_examples]
        example_indices = list(range(len(prefix_data)))
        print(f"Selected first {len(prefix_data)} examples")
    else:
        example_indices = list(range(len(prefix_data)))
        print(f"Running on all {len(prefix_data)} examples")

    all_results = []
    layer_em_s1 = {l: [] for l in layer_list}
    layer_em_s2 = {l: [] for l in layer_list}

    for i, item in enumerate(prefix_data):
        idx = item["idx"]
        question = item["question"]
        prefix = item["prefix"]
        entity = item["entity"]

        ex = dataset[idx]
        answer = ex["answer"]

        print("\n" + "=" * 130)
        print(f"[{i+1}/{len(prefix_data)}] Example {idx}")
        print(f"  Q: {question}")
        print(f"  GT: {answer}")
        print(f"  Prefix: '{prefix}' | Entity: '{entity}'")
        print("=" * 130)

        prompt = f"Question: {question}\nAnswer: {prefix}"

        # Baseline - compare with GT entity
        print(f"\n[BASELINE] (vs GT entity)")
        retain_gen = generate_baseline(retain, tokenizer, prompt, max_new_tokens=30)
        full_gen = generate_baseline(full, tokenizer, prompt, max_new_tokens=30)
        unlearn_gen = generate_baseline(unlearn, tokenizer, prompt, max_new_tokens=30)

        retain_em_gt = compute_em_score(retain_gen, entity, tokenizer)
        full_em_gt = compute_em_score(full_gen, entity, tokenizer)
        unlearn_em_gt = compute_em_score(unlearn_gen, entity, tokenizer)

        print(f"  Retain:  EM={retain_em_gt:.2f} \"{clean_text(retain_gen)}\"")
        print(f"  Full:    EM={full_em_gt:.2f} \"{clean_text(full_gen)}\"  <- Reference for patching")
        print(f"  Unlearn: EM={unlearn_em_gt:.2f} \"{clean_text(unlearn_gen)}\"")

        # Per-layer patching - BATCH: extract all layers at once
        s1_results = []
        s2_results = []

        # Batch extract hidden states (1 forward pass per model instead of 16)
        device = next(retain.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Determine patch positions using helper function
        pos = get_patch_positions(patch_types, idx, subject_pos_data)

        # For hidden extraction, we need all tokens when patching at specific positions
        needs_all_tokens = "all" in patch_types or "subject" in patch_types or "min_cos" in patch_types
        extract_pos = None if needs_all_tokens else -1
        retain_hiddens = get_all_layers_hidden(
            retain, inputs["input_ids"], inputs["attention_mask"], layer_list, position=extract_pos
        )
        unlearn_hiddens = get_all_layers_hidden(
            unlearn, inputs["input_ids"], inputs["attention_mask"], layer_list, position=extract_pos
        )
        full_hiddens = get_all_layers_hidden(
            full, inputs["input_ids"], inputs["attention_mask"], layer_list, position=extract_pos
        )

        # Use Full model's output as reference (not GT entity)
        # This measures: does patching preserve Full's knowledge?
        reference = full_gen

        print(f"\n{'L':<4} {'Stage1: Retain->Full':<55} {'Stage2: Unlearn->Full':<55} {'Hard':<8} {'Soft':<10} {'CosSim (mean)'}")
        print("-" * 150)

        # Collect cosine data for heatmap
        example_cos_data = []

        for layer in layer_list:
            # Compute per-token cosine similarity between source and target (Full) hidden states
            s1_cos = compute_cosine_similarity(retain_hiddens[layer], full_hiddens[layer])
            s2_cos = compute_cosine_similarity(unlearn_hiddens[layer], full_hiddens[layer])
            example_cos_data.append({"s1": s1_cos, "s2": s2_cos})

            # Stage 1: Retain -> Full (use cached hidden)
            # Compare with Full's output: does Retain's hidden preserve Full's knowledge?
            s1_gen = generate_with_patch(full, tokenizer, prompt, layer, retain_hiddens[layer], max_new_tokens=20, patch_position=pos, patch_component=args.patch_component)
            s1_em = compute_em_score(s1_gen, reference, tokenizer)
            s1_results.append({"layer": layer, "response": s1_gen, "em": s1_em, "cos_sim": s1_cos})
            layer_em_s1[layer].append(s1_em)

            # Stage 2: Unlearn -> Full (use cached hidden)
            # Compare with Full's output: does Unlearn's hidden preserve Full's knowledge?
            s2_gen = generate_with_patch(full, tokenizer, prompt, layer, unlearn_hiddens[layer], max_new_tokens=20, patch_position=pos, patch_component=args.patch_component)
            s2_em = compute_em_score(s2_gen, reference, tokenizer)
            s2_results.append({"layer": layer, "response": s2_gen, "em": s2_em, "cos_sim": s2_cos})
            layer_em_s2[layer].append(s2_em)

            # Hard evaluation (KEPT = knowledge preserved, LOST = knowledge gone)
            s1_kept = s1_em >= args.em_threshold
            s2_kept = s2_em >= args.em_threshold

            if not s1_kept and not s2_kept:
                hard = "EXACT"   # Both LOST -> erased as expected
            elif not s1_kept and s2_kept:
                hard = "UNDER"   # S1 LOST, S2 KEPT -> knowledge leaked (under-erased)
            elif s1_kept and s2_kept:
                hard = "-"       # Both KEPT -> general knowledge
            else:  # s1_kept and not s2_kept
                hard = "OVER"    # S1 KEPT, S2 LOST -> over-erased

            gap = s1_em - s2_em
            s1_str = f"{'KEPT' if s1_kept else 'LOST'} EM={s1_em:.2f} \"{clean_text(s1_gen, 28)}\""
            s2_str = f"{'KEPT' if s2_kept else 'LOST'} EM={s2_em:.2f} \"{clean_text(s2_gen, 28)}\""

            # Log: main line with EM results and mean cosine similarity
            print(f"L{layer:<3} {s1_str:<55} {s2_str:<55} {hard:<8} Gap={gap:+.2f}  S1={s1_cos['mean']:.3f} S2={s2_cos['mean']:.3f}")

        # Save heatmap for this example (all layers) if plotting enabled
        if args.plot:
            # Use user-specified index (example_indices[i]) not internal TOFU index (idx)
            plot_layer_heatmap(example_cos_data, tokenizer, inputs["input_ids"],
                              example_indices[i], args.out_dir, layer_list)

        # Per-example summary
        print("-" * 150)
        finetuned = [r["layer"] for r in s1_results if r["em"] < args.em_threshold]
        erased = [l for l in finetuned if s2_results[layer_list.index(l)]["em"] < args.em_threshold]
        n_ft, n_er = len(finetuned), len(erased)
        udr = n_er / n_ft if n_ft > 0 else -1

        ft_s1_avg = sum(r["em"] for r in s1_results if r["layer"] in finetuned) / n_ft if n_ft > 0 else 0
        ft_s2_avg = sum(s2_results[layer_list.index(l)]["em"] for l in finetuned) / n_ft if n_ft > 0 else 0

        # Over-erase detection: S2 fails earlier than S1
        s1_first_x = next((r["layer"] for r in s1_results if r["em"] < args.em_threshold), None)
        s2_first_x = next((i for i, r in enumerate(s2_results) if r["em"] < args.em_threshold), None)
        s2_first_x_layer = layer_list[s2_first_x] if s2_first_x is not None else None

        over_erase_depth = 0
        if s1_first_x is not None and s2_first_x_layer is not None:
            over_erase_depth = s1_first_x - s2_first_x_layer  # positive = over-erased

        # Count S2 failed layers for over-erase detection
        s2_failed_layers = [r["layer"] for r in s2_results if r["em"] < args.em_threshold]
        n_s2_failed = len(s2_failed_layers)

        if n_ft > 0:
            print(f"[HARD] FT: {finetuned} | Erased: {erased} | UDR={udr:.2f}")
            print(f"[SOFT] FT layers: S1={ft_s1_avg:.3f}, S2={ft_s2_avg:.3f}, Retention={ft_s2_avg:.3f}")
            if n_s2_failed > n_ft:
                print(f"[OVER] S1 fails {n_ft} layers, S2 fails {n_s2_failed} layers -> Over-erased by {n_s2_failed - n_ft}")
        else:
            if n_s2_failed > 0:
                print(f"[HARD] No FT layers (S1 all OK) but S2 fails {n_s2_failed} layers -> OVER-ERASED")
            else:
                print(f"[HARD] No FT layers (S1 & S2 all OK) -> GENERAL KNOWLEDGE")

        all_results.append({
            "idx": idx, "question": question, "prefix": prefix, "entity": entity,
            "reference": clean_generated(full_gen),  # What patching is compared against
            "baseline_vs_gt": {"retain": retain_em_gt, "full": full_em_gt, "unlearn": unlearn_em_gt},
            "stage1": s1_results, "stage2": s2_results,
            "hard": {"finetuned": finetuned, "erased": erased, "udr": udr},
            "soft": {"ft_s1_avg": ft_s1_avg, "ft_s2_avg": ft_s2_avg},
            "over_erase": {"s1_first_x": s1_first_x, "s2_first_x": s2_first_x_layer, "depth": over_erase_depth}
        })

    # Aggregate
    print("\n" + "=" * 130)
    print("[AGGREGATE]")
    print("=" * 130)

    print(f"\n{'Layer':<6} {'S1 EM':<10} {'S2 EM':<10} {'Gap':<10} {'Interpretation'}")
    print("-" * 50)

    for l in layer_list:
        s1_avg = sum(layer_em_s1[l]) / len(layer_em_s1[l]) if layer_em_s1[l] else 0
        s2_avg = sum(layer_em_s2[l]) / len(layer_em_s2[l]) if layer_em_s2[l] else 0
        gap = s1_avg - s2_avg

        s1_x_rate = sum(1 for em in layer_em_s1[l] if em < args.em_threshold) / len(layer_em_s1[l])
        s2_x_rate = sum(1 for em in layer_em_s2[l] if em < args.em_threshold) / len(layer_em_s2[l])

        if s1_x_rate > 0.5 and s2_x_rate > 0.5:
            interp = "ERASED"
        elif s1_x_rate > 0.5 and s2_x_rate <= 0.5:
            interp = "LEAKED"
        else:
            interp = "-"

        print(f"L{l:<5} {s1_avg:<10.3f} {s2_avg:<10.3f} {gap:+.3f}     {interp}")

    print("-" * 50)

    # Overall metrics
    valid_udrs = [r["hard"]["udr"] for r in all_results if r["hard"]["udr"] >= 0]
    avg_udr = sum(valid_udrs) / len(valid_udrs) if valid_udrs else 0

    valid_retentions = [r["soft"]["ft_s2_avg"] for r in all_results if r["hard"]["udr"] >= 0]
    avg_retention = sum(valid_retentions) / len(valid_retentions) if valid_retentions else 0

    print(f"\n[HARD] Average UDR: {avg_udr:.3f} ({len(valid_udrs)} examples with FT layers)")
    if valid_udrs:
        print(f"  Full erasure (UDR=1.0): {sum(1 for u in valid_udrs if u == 1.0)}")
        print(f"  Partial (0<UDR<1): {sum(1 for u in valid_udrs if 0 < u < 1.0)}")
        print(f"  No erasure (UDR=0): {sum(1 for u in valid_udrs if u == 0.0)}")

    print(f"\n[SOFT] Average Retention on FT layers: {avg_retention:.3f}")
    print(f"  (0=complete erasure, 1=full leakage)")

    # Erasure quality categorization (KEPT/LOST based):
    # - General Knowledge: S1 all KEPT (Retain has knowledge = not forget-set specific)
    # - Over-erased: S2 LOST count > S1 LOST count (collateral damage)
    # - Exact-erased: S2 LOST count = S1 LOST count (ideal unlearning)
    # - Under-erased: S2 LOST count < S1 LOST count (knowledge leaked)
    over_erased_examples = []
    exact_erased_examples = []
    under_erased_examples = []
    general_knowledge_examples = []

    for r in all_results:
        # Count LOST layers (EM < threshold) for each stage
        s1_lost = [s["layer"] for s in r["stage1"] if s["em"] < args.em_threshold]
        s2_lost = [s["layer"] for s in r["stage2"] if s["em"] < args.em_threshold]
        n_s1_lost = len(s1_lost)
        n_s2_lost = len(s2_lost)

        if n_s1_lost == 0:
            # S1 all KEPT -> Retain has full knowledge = general knowledge
            general_knowledge_examples.append(r)
        elif n_s2_lost > n_s1_lost:
            # S2 more LOST -> over-erased (collateral damage)
            over_erased_examples.append(r)
        elif n_s2_lost == n_s1_lost:
            # S2 same LOST count -> exact-erased (ideal)
            exact_erased_examples.append(r)
        else:
            # S2 fewer LOST -> under-erased (knowledge leaked)
            under_erased_examples.append(r)

    # Calculate percentages excluding General Knowledge
    n_evaluated = len(all_results) - len(general_knowledge_examples)

    print(f"\n[ERASURE QUALITY] (excluding General Knowledge: S1 all KEPT)")
    print(f"  Evaluated: {n_evaluated} / {len(all_results)} (General Knowledge: {len(general_knowledge_examples)} excluded)")
    if n_evaluated > 0:
        print(f"  Over-erased  (S2 LOST > S1 LOST):  {len(over_erased_examples):>3} ({100*len(over_erased_examples)/n_evaluated:.1f}%) <- collateral damage")
        print(f"  Exact-erased (S2 LOST = S1 LOST):  {len(exact_erased_examples):>3} ({100*len(exact_erased_examples)/n_evaluated:.1f}%)")
        print(f"  Under-erased (S2 LOST < S1 LOST):  {len(under_erased_examples):>3} ({100*len(under_erased_examples)/n_evaluated:.1f}%) <- knowledge leaked")

    if over_erased_examples:
        avg_over_diff = sum(
            len([s for s in r["stage2"] if s["em"] < args.em_threshold]) -
            len([s for s in r["stage1"] if s["em"] < args.em_threshold])
            for r in over_erased_examples
        ) / len(over_erased_examples)
        print(f"  Average over-erase: +{avg_over_diff:.1f} extra LOST layers")

    # Save
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    summary = {
        "method": args.unlearn_model,
        "example_indices": example_indices,
        "num_examples": len(all_results),
        "em_threshold": args.em_threshold,
        "hard_avg_udr": avg_udr,
        "soft_avg_retention": avg_retention,
        "erasure_quality": {
            "n_evaluated": n_evaluated,
            "general_knowledge": len(general_knowledge_examples),
            "over_erased": len(over_erased_examples),
            "exact_erased": len(exact_erased_examples),
            "under_erased": len(under_erased_examples),
            "over_erased_pct": 100 * len(over_erased_examples) / n_evaluated if n_evaluated > 0 else 0,
            "exact_erased_pct": 100 * len(exact_erased_examples) / n_evaluated if n_evaluated > 0 else 0,
            "under_erased_pct": 100 * len(under_erased_examples) / n_evaluated if n_evaluated > 0 else 0,
        },
        "layer_stats": {l: {"s1_em": sum(layer_em_s1[l])/len(layer_em_s1[l]),
                           "s2_em": sum(layer_em_s2[l])/len(layer_em_s2[l])}
                       for l in layer_list}
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] {args.out_dir}/")
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()

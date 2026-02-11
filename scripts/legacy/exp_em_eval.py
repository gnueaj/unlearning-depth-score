#!/usr/bin/env python3
"""
Two-Stage Experiment with EM (Exact Memorization) Evaluation

Stage 1: Retain → Full (Where is forget-set knowledge stored?)
Stage 2: Unlearn → Full (Where is knowledge erased?)

Evaluation: EM score = consecutive token match ratio with GT entity
- Hard: EM >= threshold → OK (knowledge present)
- Soft: Raw EM values for continuous analysis

Uses validated prefix data (278 examples where Full model correctly generates GT)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uds.models import load_model, load_tokenizer, get_num_layers
from uds.core import get_all_layers_hidden, forward_with_patch, generate_with_patch, generate_baseline
from uds.utils import set_seed, safe_mkdir, parse_layers
from uds.config import get_model_id


TOFU_FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
TOFU_RETAIN_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
PREFIX_DATA_PATH = "tofu_data/forget10_filtered_v4.json"  # Manual prefix + filtered (324 examples)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlearn_model", type=str, default="simnpo")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples (default: all)")
    parser.add_argument("--layers", type=str, default="0-15")
    parser.add_argument("--em_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--patch_all", action="store_true", help="Patch all tokens instead of last only")
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
    print(f"Patch mode: {'ALL tokens' if args.patch_all else 'LAST token only'}")
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

    dataset = load_dataset("locuslab/TOFU", "forget10", split="train")

    # Limit examples if specified
    if args.num_examples:
        prefix_data = prefix_data[:args.num_examples]

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

        prompt = f"Question: {question}\nAnswer: {prefix}"

        # Batch extract hidden states (1 forward pass per model instead of 16)
        device = next(retain.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        pos = None if args.patch_all else -1

        retain_hiddens = get_all_layers_hidden(
            retain, inputs["input_ids"], inputs["attention_mask"], layer_list, position=pos
        )
        unlearn_hiddens = get_all_layers_hidden(
            unlearn, inputs["input_ids"], inputs["attention_mask"], layer_list, position=pos
        )

        # Baseline - compare with GT entity
        retain_gen = generate_baseline(retain, tokenizer, prompt, max_new_tokens=30)
        full_gen = generate_baseline(full, tokenizer, prompt, max_new_tokens=30)
        unlearn_gen = generate_baseline(unlearn, tokenizer, prompt, max_new_tokens=30)

        retain_em_gt = compute_em_score(retain_gen, entity, tokenizer)
        full_em_gt = compute_em_score(full_gen, entity, tokenizer)
        unlearn_em_gt = compute_em_score(unlearn_gen, entity, tokenizer)

        # Use Full model's output as reference for patching
        reference = full_gen

        # Pre-compute S1/S2 for all layers to determine category
        s1_results = []
        s2_results = []

        for layer in layer_list:
            s1_gen = generate_with_patch(full, tokenizer, prompt, layer, retain_hiddens[layer], max_new_tokens=20, patch_position=pos)
            s1_em = compute_em_score(s1_gen, reference, tokenizer)
            s1_results.append({"layer": layer, "response": s1_gen, "em": s1_em})
            layer_em_s1[layer].append(s1_em)

            s2_gen = generate_with_patch(full, tokenizer, prompt, layer, unlearn_hiddens[layer], max_new_tokens=20, patch_position=pos)
            s2_em = compute_em_score(s2_gen, reference, tokenizer)
            s2_results.append({"layer": layer, "response": s2_gen, "em": s2_em})
            layer_em_s2[layer].append(s2_em)

        # Count LOST layers for category determination
        s1_lost_count = sum(1 for r in s1_results if r["em"] < args.em_threshold)
        s2_lost_count = sum(1 for r in s2_results if r["em"] < args.em_threshold)

        # Determine erasure category
        if s1_lost_count == 0:
            erasure_cat = "GENERAL_KNOWLEDGE"
            erasure_detail = "-"
        else:
            # UDS category (Full/Partial/No)
            uds_val = s2_lost_count / s1_lost_count
            if uds_val >= 1.0:
                erasure_cat = "FULL"
            elif uds_val > 0:
                erasure_cat = "PARTIAL"
            else:
                erasure_cat = "NO_ERASURE"

            # Over/Exact/Under category
            if s2_lost_count > s1_lost_count:
                erasure_detail = "OVER"
            elif s2_lost_count == s1_lost_count:
                erasure_detail = "EXACT"
            else:
                erasure_detail = "UNDER"

        # Now print header with category info
        print("\n" + "=" * 130)
        print(f"[{i+1}/{len(prefix_data)}] Example {idx}  |  [{erasure_cat}] [{erasure_detail}]  (S1_LOST={s1_lost_count}, S2_LOST={s2_lost_count})")
        print(f"  Q: {question}")
        print(f"  GT: {answer}")
        print(f"  Prefix: '{prefix}' | Entity: '{entity}'")
        print("=" * 130)

        print(f"\n[BASELINE] (vs GT entity)")
        print(f"  Retain:  EM={retain_em_gt:.2f} \"{clean_text(retain_gen)}\"")
        print(f"  Full:    EM={full_em_gt:.2f} \"{clean_text(full_gen)}\"  <- Reference for patching")
        print(f"  Unlearn: EM={unlearn_em_gt:.2f} \"{clean_text(unlearn_gen)}\"")

        print(f"\n{'L':<4} {'Stage1: Retain->Full':<55} {'Stage2: Unlearn->Full':<55} {'Hard':<8} {'Soft'}")
        print("-" * 135)

        # Display pre-computed results
        for s1_r, s2_r in zip(s1_results, s2_results):
            layer = s1_r["layer"]
            s1_gen, s1_em = s1_r["response"], s1_r["em"]
            s2_gen, s2_em = s2_r["response"], s2_r["em"]

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

            print(f"L{layer:<3} {s1_str:<55} {s2_str:<55} {hard:<8} Gap={gap:+.2f}")

        # Per-example summary
        print("-" * 135)
        finetuned = [r["layer"] for r in s1_results if r["em"] < args.em_threshold]
        erased = [l for l in finetuned if s2_results[layer_list.index(l)]["em"] < args.em_threshold]
        n_ft, n_er = len(finetuned), len(erased)
        uds = n_er / n_ft if n_ft > 0 else -1

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
            print(f"[HARD] FT: {finetuned} | Erased: {erased} | UDS={uds:.2f}")
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
            "hard": {"finetuned": finetuned, "erased": erased, "uds": uds},
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
    valid_udss = [r["hard"]["uds"] for r in all_results if r["hard"]["uds"] >= 0]
    avg_uds = sum(valid_udss) / len(valid_udss) if valid_udss else 0

    valid_retentions = [r["soft"]["ft_s2_avg"] for r in all_results if r["hard"]["uds"] >= 0]
    avg_retention = sum(valid_retentions) / len(valid_retentions) if valid_retentions else 0

    print(f"\n[HARD] Average UDS: {avg_uds:.3f} ({len(valid_udss)} examples with FT layers)")
    if valid_udss:
        print(f"  Full erasure (UDS=1.0): {sum(1 for u in valid_udss if u == 1.0)}")
        print(f"  Partial (0<UDS<1): {sum(1 for u in valid_udss if 0 < u < 1.0)}")
        print(f"  No erasure (UDS=0): {sum(1 for u in valid_udss if u == 0.0)}")

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
        "num_examples": len(all_results),
        "em_threshold": args.em_threshold,
        "hard_avg_uds": avg_uds,
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
        "layer_stats": {l: {"s1": sum(layer_em_s1[l])/len(layer_em_s1[l]),
                           "s2": sum(layer_em_s2[l])/len(layer_em_s2[l])}
                       for l in layer_list}
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] {args.out_dir}/")
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()

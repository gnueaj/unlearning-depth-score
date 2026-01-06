#!/usr/bin/env python3
"""
Two-Stage Comparison Experiment: Finetuning vs Unlearning Boundary

Stage 2: TOFU-Full → Pretrained
  - Question: At which layer is TOFU knowledge READABLE?
  - If patching enables Target to answer correctly, knowledge is encoded there
  - This reveals the FINETUNING BOUNDARY (where knowledge was written during training)

Stage 3: Unlearned → TOFU-Full
  - Question: At which layer does unlearning ERASE knowledge?
  - If patching still enables correct answer, unlearning FAILED at that layer
  - This reveals the UNLEARNING BOUNDARY (where unlearning actually modified weights)

The key insight:
  - Stage 2 shows WHERE finetuning wrote knowledge
  - Stage 3 shows HOW FAR unlearning reaches
  - Comparing them reveals: knowledge stored at L9+ but only erased at L12+
    means L9-L11 still retain the "forgotten" knowledge
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patchscope.models import load_model, load_tokenizer, get_num_layers
from patchscope.core import (
    get_generated_answer_hidden,
    forward_with_patch,
    generate_with_patch,
    generate_baseline,
)
from patchscope.tofu_entities import extract_entity
from patchscope.utils import set_seed, safe_mkdir, parse_layers
from patchscope.config import UNLEARN_MODELS, get_model_id


# Model IDs
PRETRAINED_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TOFU_FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"


class TeeLogger:
    """Log to both stdout and file."""
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


def get_topk_tokens(model, tokenizer, prompt, patch_layer=None, patch_vector=None, k=5):
    """Get top-k next token predictions."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if patch_layer is not None and patch_vector is not None:
        logits = forward_with_patch(
            model, inputs["input_ids"], inputs["attention_mask"],
            patch_layer, patch_vector, patch_position=-1
        )
    else:
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            return_dict=True
        )
        logits = out.logits

    probs = torch.softmax(logits[:, -1, :], dim=-1)
    topk_probs, topk_ids = torch.topk(probs[0], k=k)

    tokens = []
    for tid, p in zip(topk_ids.tolist(), topk_probs.tolist()):
        tokens.append({"token": tokenizer.decode([tid]).strip(), "prob": p})

    return tokens


def format_topk(tokens, k=5):
    """Format top-k tokens for display."""
    return " | ".join([f"'{t['token']}': {t['prob']:.3f}" for t in tokens[:k]])


def check_correct_in_topk(tokens, entity, k=5):
    """Check if correct entity's first token is in top-k."""
    entity_lower = entity.lower().strip()
    for t in tokens[:k]:
        if entity_lower.startswith(t["token"].lower().strip()):
            return True, t["prob"]
    return False, 0.0


def run_single_example(
    source_model,
    target_model,
    tokenizer,
    question: str,
    answer: str,
    entity: str,
    prefix: str,
    layer_list: List[int],
    stage_name: str,
):
    """Run patching for a single example across all layers."""
    device = next(source_model.parameters()).device

    # Build prompt
    if prefix:
        prompt = f"Question: {question}\nAnswer: {prefix}"
    else:
        prompt = f"Question: {question}\nAnswer:"

    results = []

    for layer_idx in layer_list:
        # Extract hidden from source
        hidden, meta = get_generated_answer_hidden(
            source_model, tokenizer, question, layer_idx,
            forced_prefix=prompt
        )

        # Get patched output from target
        patched_response = generate_with_patch(
            target_model, tokenizer, prompt,
            layer_idx, hidden, max_new_tokens=20
        )

        # Get top-k tokens with patching
        topk = get_topk_tokens(target_model, tokenizer, prompt, layer_idx, hidden, k=5)

        # Check if correct
        in_topk, prob = check_correct_in_topk(topk, entity, k=5)
        in_response = entity.lower() in patched_response.lower()

        results.append({
            "layer": layer_idx,
            "response": patched_response,
            "topk": topk,
            "correct_in_topk": in_topk,
            "correct_in_response": in_response,
            "correct_prob": prob,
        })

    return results


def format_topk_str(topk, k=3):
    """Format top-k tokens for display."""
    parts = []
    for t in topk[:k]:
        parts.append(f"'{t['token']}': {t['prob']:.3f}")
    return " | ".join(parts)


def print_layer_comparison(
    results_stage2: List[Dict],
    results_stage3: List[Dict],
    layer_list: List[int],
    entity: str,
):
    """Print side-by-side comparison of Stage 2 and Stage 3."""

    print("\n" + "=" * 200)
    print(f"LAYER-BY-LAYER ANALYSIS | Expected Entity: '{entity}'")
    print("=" * 200)
    print(f"{'Layer':<6} {'Stage2: Full→Pre (Finetuning Boundary)':<95} {'Stage3: Unlearn→Full (Unlearning Effect)':<95}")
    print("-" * 200)

    for i, layer in enumerate(layer_list):
        r2 = results_stage2[i]
        r3 = results_stage3[i]

        # Stage 2: If correct, knowledge IS readable at this layer
        s2_status = "✓" if r2["correct_in_topk"] or r2["correct_in_response"] else "✗"
        s2_response = r2["response"][:35].replace("\n", " ") + "..." if len(r2["response"]) > 35 else r2["response"].replace("\n", " ")
        s2_topk_str = format_topk_str(r2["topk"], k=3)
        s2_str = f"{s2_status} \"{s2_response}\" [{s2_topk_str}]"

        # Stage 3: If correct, unlearning FAILED to erase at this layer
        s3_status = "✓" if r3["correct_in_topk"] or r3["correct_in_response"] else "✗"
        s3_response = r3["response"][:35].replace("\n", " ") + "..." if len(r3["response"]) > 35 else r3["response"].replace("\n", " ")
        s3_topk_str = format_topk_str(r3["topk"], k=3)
        s3_str = f"{s3_status} \"{s3_response}\" [{s3_topk_str}]"

        # Interpretation based on Stage 2 and Stage 3
        if s2_status == "✓" and s3_status == "✗":
            interp = "← UNLEARNED"
        elif s2_status == "✗" and s3_status == "✗":
            interp = ""
        elif s2_status == "✓" and s3_status == "✓":
            interp = "← Knowledge remains"
        elif s2_status == "✗" and s3_status == "✓":
            interp = "← Not finetuned here"
        else:
            interp = ""

        print(f"L{layer:<5} {s2_str:<95} {s3_str:<95} {interp}")

    print("-" * 200)


def main():
    parser = argparse.ArgumentParser(description="Three-Stage Comparison Experiment")
    parser.add_argument("--unlearn_model", type=str, default="simnpo",
                        help="Unlearning method (simnpo, npo, idknll, idkdpo, graddiff, etc.)")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--layers", type=str, default="0-15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    # Get unlearn model ID
    unlearn_model_id = get_model_id(args.unlearn_model)

    # Output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"runs/three_stage_{args.unlearn_model}_{timestamp}"
    safe_mkdir(args.out_dir)

    # Setup logging
    log_path = os.path.join(args.out_dir, "run.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Two-Stage Experiment Started")
    print("=" * 100)
    print("FINETUNING vs UNLEARNING COMPARISON")
    print("=" * 100)
    print(f"Stage 2: Full → Pretrained     (Where is TOFU knowledge encoded? = Finetuning boundary)")
    print(f"Stage 3: {args.unlearn_model.upper()} → Full  (Where does unlearning erase? = Unlearning boundary)")
    print("=" * 100)
    print(f"\nModels:")
    print(f"  Pretrained: {PRETRAINED_MODEL}")
    print(f"  Full:       {TOFU_FULL_MODEL}")
    print(f"  Unlearned:  {unlearn_model_id}")
    print(f"\nSettings: {args.num_examples} examples, layers {args.layers}")
    print(f"Output: {args.out_dir}")
    print("=" * 100)

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = load_tokenizer(TOFU_FULL_MODEL)

    # Load models
    print("[2/5] Loading pretrained model...")
    pretrained_model = load_model(PRETRAINED_MODEL, dtype="bfloat16", device_map="cuda")

    print("[3/5] Loading TOFU-full model...")
    tofu_full_model = load_model(TOFU_FULL_MODEL, dtype="bfloat16", device_map="cuda")

    print("[4/5] Loading unlearned model...")
    unlearn_model = load_model(unlearn_model_id, dtype="bfloat16", device_map="cuda")

    # Get layer info
    n_layers = get_num_layers(tofu_full_model)
    layer_list = parse_layers(args.layers, n_layers)
    print(f"[INFO] Testing {len(layer_list)} layers: {layer_list}")

    # Load dataset
    print("[5/5] Loading TOFU forget10 dataset...")
    dataset = load_dataset("locuslab/TOFU", "forget10", split="train")

    all_results = []

    for ex_idx in range(min(args.num_examples, len(dataset))):
        ex = dataset[ex_idx]
        question = str(ex.get("question", "")).strip()
        answer = str(ex.get("answer", "")).strip()
        entity, prefix = extract_entity(question, answer)

        print("\n" + "=" * 100)
        print(f"[EXAMPLE {ex_idx}]")
        print(f"  Question: \"{question[:80]}...\"" if len(question) > 80 else f"  Question: \"{question}\"")
        print(f"  GT Answer: \"{answer[:60]}...\"" if len(answer) > 60 else f"  GT Answer: \"{answer}\"")
        print(f"  Entity: \"{entity}\"")
        print(f"  Prefix: \"{prefix}\"" if prefix else "  Prefix: (none)")
        print("=" * 100)

        # Baseline generations
        print("\n[BASELINE - No Patching]")
        base_prompt = f"Question: {question}\nAnswer:"

        pretrained_gen = generate_baseline(pretrained_model, tokenizer, base_prompt, 30)
        full_gen = generate_baseline(tofu_full_model, tokenizer, base_prompt, 30)
        unlearn_gen = generate_baseline(unlearn_model, tokenizer, base_prompt, 30)

        print(f"  Pretrained: \"{pretrained_gen.strip()[:70]}\"")
        print(f"  Full:       \"{full_gen.strip()[:70]}\"")
        print(f"  Unlearned:  \"{unlearn_gen.strip()[:70]}\"")

        # Build prompt for patching
        if prefix:
            patch_prompt = f"Question: {question}\nAnswer: {prefix}"
        else:
            patch_prompt = f"Question: {question}\nAnswer:"

        print(f"\n[PATCHING PROMPT]")
        print(f"  \"{patch_prompt[-80:]}\"")

        # Get what each model predicts at the extraction point
        print(f"\n[NEXT TOKEN PREDICTIONS at extraction point]")

        pre_topk = get_topk_tokens(pretrained_model, tokenizer, patch_prompt, k=5)
        full_topk = get_topk_tokens(tofu_full_model, tokenizer, patch_prompt, k=5)
        unlearn_topk = get_topk_tokens(unlearn_model, tokenizer, patch_prompt, k=5)

        print(f"  Pretrained: {format_topk(pre_topk)}")
        print(f"  Full:       {format_topk(full_topk)}")
        print(f"  Unlearned:  {format_topk(unlearn_topk)}")

        # Run two stages
        print(f"\n[STAGE 2] Full → Pretrained (Finetuning: where is knowledge encoded?)")
        results_s2 = run_single_example(
            tofu_full_model, pretrained_model, tokenizer,
            question, answer, entity, prefix, layer_list, "stage2"
        )

        print(f"[STAGE 3] Unlearned → Full (Unlearning: where is knowledge erased?)")
        results_s3 = run_single_example(
            unlearn_model, tofu_full_model, tokenizer,
            question, answer, entity, prefix, layer_list, "stage3"
        )

        # Print comparison table
        print_layer_comparison(results_s2, results_s3, layer_list, entity)

        # Summary for this example
        s2_correct = sum(1 for r in results_s2 if r["correct_in_topk"] or r["correct_in_response"])
        s3_correct = sum(1 for r in results_s3 if r["correct_in_topk"] or r["correct_in_response"])

        print(f"\n[EXAMPLE SUMMARY]")
        print(f"  Stage 2 (Full→Pre):     {s2_correct}/{len(layer_list)} layers encode readable knowledge (finetuning boundary)")
        print(f"  Stage 3 (Unlearn→Full): {s3_correct}/{len(layer_list)} layers still have knowledge (unlearning failed)")

        # Find boundaries
        s2_success_layers = [r["layer"] for r in results_s2 if r["correct_in_topk"] or r["correct_in_response"]]
        s3_success_layers = [r["layer"] for r in results_s3 if r["correct_in_topk"] or r["correct_in_response"]]
        s3_fail_layers = [r["layer"] for r in results_s3 if not (r["correct_in_topk"] or r["correct_in_response"])]

        if s2_success_layers:
            print(f"  Finetuning boundary: L{min(s2_success_layers)}+ (knowledge encoded here)")
        if s3_fail_layers and s2_success_layers:
            # Layers where Stage 2 succeeded but Stage 3 failed = unlearning worked
            unlearned_layers = [l for l in s2_success_layers if l in s3_fail_layers]
            if unlearned_layers:
                print(f"  Unlearning erased: L{min(unlearned_layers)}-L{max(unlearned_layers)}")
            # Layers where both succeeded = knowledge remains
            remaining_layers = [l for l in s2_success_layers if l in s3_success_layers]
            if remaining_layers:
                print(f"  Knowledge remains: L{min(remaining_layers)}-L{max(remaining_layers)}")

        all_results.append({
            "example_idx": ex_idx,
            "question": question,
            "answer": answer,
            "entity": entity,
            "stage2": results_s2,
            "stage3": results_s3,
        })

    # Save results
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Aggregate summary
    print("\n" + "=" * 100)
    print("[AGGREGATE SUMMARY]")
    print("=" * 100)
    print(f"Stage 2 (Full→Pre): Where finetuning wrote knowledge")
    print(f"Stage 3 (Unlearn→Full): Where unlearning erased / left knowledge")
    print("=" * 100)

    # Per-layer aggregation
    layer_stats = {l: {"s2": 0, "s3": 0, "total": 0} for l in layer_list}

    for res in all_results:
        for i, layer in enumerate(layer_list):
            layer_stats[layer]["total"] += 1
            if res["stage2"][i]["correct_in_topk"] or res["stage2"][i]["correct_in_response"]:
                layer_stats[layer]["s2"] += 1
            if res["stage3"][i]["correct_in_topk"] or res["stage3"][i]["correct_in_response"]:
                layer_stats[layer]["s3"] += 1

    print(f"\n{'Layer':<6} {'Stage2 (Finetuning)':<20} {'Stage3 (Unlearning)':<20} {'Interpretation'}")
    print("-" * 80)

    for layer in layer_list:
        s = layer_stats[layer]
        r2 = s["s2"] / s["total"] if s["total"] > 0 else 0
        r3 = s["s3"] / s["total"] if s["total"] > 0 else 0

        bar2 = "█" * int(r2 * 5) + "░" * (5 - int(r2 * 5))
        bar3 = "█" * int(r3 * 5) + "░" * (5 - int(r3 * 5))

        # Interpretation based on Stage 2 (finetuning) and Stage 3 (unlearning)
        if r2 > 0.5 and r3 < 0.5:
            interp = "← UNLEARNED (knowledge erased)"
        elif r2 < 0.5 and r3 < 0.5:
            interp = "← Not finetuned here"
        elif r2 > 0.5 and r3 > 0.5:
            interp = "← LEAKED (knowledge remains!)"
        elif r2 < 0.5 and r3 > 0.5:
            interp = "← Unexpected"
        else:
            interp = ""

        print(f"L{layer:<5} {bar2} {r2:>5.0%}       {bar3} {r3:>5.0%}       {interp}")

    print("-" * 80)

    # Summary interpretation
    finetuning_layers = [l for l in layer_list if layer_stats[l]["s2"] / layer_stats[l]["total"] > 0.5]
    unlearned_layers = [l for l in layer_list if layer_stats[l]["s2"] / layer_stats[l]["total"] > 0.5
                        and layer_stats[l]["s3"] / layer_stats[l]["total"] < 0.5]
    leaked_layers = [l for l in layer_list if layer_stats[l]["s2"] / layer_stats[l]["total"] > 0.5
                     and layer_stats[l]["s3"] / layer_stats[l]["total"] > 0.5]

    print(f"\n[KEY FINDINGS]")
    if finetuning_layers:
        print(f"  Finetuning wrote knowledge at: L{min(finetuning_layers)}-L{max(finetuning_layers)}")
    if unlearned_layers:
        print(f"  Unlearning erased knowledge at: L{min(unlearned_layers)}-L{max(unlearned_layers)}")
    if leaked_layers:
        print(f"  ⚠️  Knowledge LEAKED at: L{min(leaked_layers)}-L{max(leaked_layers)} (unlearning failed!)")
    print(f"\n[DONE] Results saved to {args.out_dir}/")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment Completed")

    # Restore stdout
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment: Where does TOFU knowledge get stored?

Bidirectional patching experiment to find which layers encode TOFU author knowledge.

Option A: Pretrained → TOFU-finetuned
  - Source: Pretrained (no author knowledge)
  - Target: TOFU-finetuned (has author knowledge)
  - Interpretation: If patching breaks accuracy, that layer is important for STORING knowledge

Option B: TOFU-finetuned → Pretrained
  - Source: TOFU-finetuned (has author knowledge)
  - Target: Pretrained (no author knowledge)
  - Interpretation: If patching enables correct answers, that layer ENCODES knowledge

Together, these reveal:
  - Writing layers: Where knowledge is written during finetuning
  - Reading layers: Where knowledge is read/decoded during inference
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple

import torch
from datasets import load_dataset

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patchscope.models import load_model, load_tokenizer, get_num_layers
from patchscope.core import (
    get_generated_answer_hidden,
    probe_knowledge,
    generate_baseline,
    compute_knowledge_score,
)
from patchscope.tofu_entities import extract_entity
from patchscope.utils import set_seed, safe_mkdir, parse_layers


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


def run_experiment(
    source_model,
    target_model,
    tokenizer,
    dataset,
    layer_list: List[int],
    num_examples: int = 10,
    experiment_name: str = "experiment",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run patching experiment: Source hidden → Target model.

    Returns per-layer detection rates and detailed results.
    """
    device = next(source_model.parameters()).device
    all_results = []

    # Per-layer aggregation
    layer_correct_counts = {l: 0 for l in layer_list}
    layer_total_counts = {l: 0 for l in layer_list}

    for ex_idx in range(min(num_examples, len(dataset))):
        ex = dataset[ex_idx]
        question = str(ex.get("question", "")).strip()
        answer = str(ex.get("answer", "")).strip()
        entity, prefix = extract_entity(question, answer)

        if verbose:
            print(f"\n[Example {ex_idx}] Q: {question[:60]}...")
            print(f"  Entity: {entity}")

        # Build prompt
        if prefix:
            source_prompt = f"Question: {question}\nAnswer: {prefix}"
        else:
            source_prompt = f"Question: {question}\nAnswer:"

        example_results = {"example_idx": ex_idx, "question": question, "entity": entity, "layers": {}}

        for layer_idx in layer_list:
            # Extract hidden from source
            hidden, meta = get_generated_answer_hidden(
                source_model, tokenizer, question, layer_idx,
                forced_prefix=source_prompt
            )

            # Probe target with patched hidden
            result = probe_knowledge(
                target_model, tokenizer,
                source_prompt,
                entity,
                patch_layer_idx=layer_idx,
                patch_vector=hidden,
                wrong_answers=["John Smith", "Jane Doe", "Unknown"],
                top_k=10,
                max_new_tokens=20,
            )

            detected = result["knowledge_detected"]
            layer_correct_counts[layer_idx] += int(detected)
            layer_total_counts[layer_idx] += 1

            example_results["layers"][layer_idx] = {
                "detected": detected,
                "correct_prob": result["correct_prob"],
                "top1": result["topk"][0]["token_str"] if result["topk"] else "",
                "response": result["full_response"][:50],
            }

        all_results.append(example_results)

        if verbose:
            # Print layer summary for this example
            layer_status = " ".join([
                f"L{l}:{'✓' if example_results['layers'][l]['detected'] else '✗'}"
                for l in layer_list
            ])
            print(f"  {layer_status}")

    # Compute per-layer detection rates
    layer_detection_rates = {
        l: layer_correct_counts[l] / layer_total_counts[l] if layer_total_counts[l] > 0 else 0
        for l in layer_list
    }

    return {
        "experiment": experiment_name,
        "num_examples": num_examples,
        "layer_detection_rates": layer_detection_rates,
        "detailed_results": all_results,
    }


def print_comparison(results_a: Dict, results_b: Dict, layer_list: List[int]):
    """Print side-by-side comparison of both experiments."""
    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 80)
    print(f"{'Layer':<8} {'A: Pre→TOFU':<15} {'B: TOFU→Pre':<15} {'Interpretation'}")
    print("-" * 80)

    rates_a = results_a["layer_detection_rates"]
    rates_b = results_b["layer_detection_rates"]

    for layer in layer_list:
        rate_a = rates_a.get(layer, 0)
        rate_b = rates_b.get(layer, 0)

        # Interpretation
        if rate_a < 0.3 and rate_b > 0.7:
            interp = "← Knowledge STORED here"
        elif rate_a > 0.7 and rate_b < 0.3:
            interp = "← Knowledge NOT here"
        elif rate_a > 0.7 and rate_b > 0.7:
            interp = "← Knowledge FLOWS through"
        elif rate_a < 0.3 and rate_b < 0.3:
            interp = "← Critical layer (blocks both ways)"
        else:
            interp = ""

        bar_a = "█" * int(rate_a * 10) + "░" * (10 - int(rate_a * 10))
        bar_b = "█" * int(rate_b * 10) + "░" * (10 - int(rate_b * 10))

        print(f"L{layer:<6} {bar_a} {rate_a:.0%}   {bar_b} {rate_b:.0%}   {interp}")

    print("-" * 80)

    # Summary statistics
    avg_a = sum(rates_a.values()) / len(rates_a) if rates_a else 0
    avg_b = sum(rates_b.values()) / len(rates_b) if rates_b else 0

    print(f"\nSummary:")
    print(f"  Option A (Pre→TOFU) avg detection: {avg_a:.1%}")
    print(f"  Option B (TOFU→Pre) avg detection: {avg_b:.1%}")

    # Find transition layers
    print(f"\nKey findings:")

    # A: Where does detection DROP? (knowledge writing starts)
    for i, layer in enumerate(layer_list[:-1]):
        curr = rates_a.get(layer, 0)
        next_rate = rates_a.get(layer_list[i+1], 0)
        if curr > 0.5 and next_rate < 0.5:
            print(f"  • Option A: Detection drops after Layer {layer} → Knowledge writing begins")

    # B: Where does detection START? (knowledge reading starts)
    for i, layer in enumerate(layer_list[:-1]):
        curr = rates_b.get(layer, 0)
        next_rate = rates_b.get(layer_list[i+1], 0)
        if curr < 0.5 and next_rate > 0.5:
            print(f"  • Option B: Detection starts at Layer {layer_list[i+1]} → Knowledge becomes readable")


def main():
    parser = argparse.ArgumentParser(description="TOFU Knowledge Layer Experiment")
    parser.add_argument("--num_examples", type=int, default=20, help="Number of examples to test")
    parser.add_argument("--layers", type=str, default="0-15", help="Layers to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--option", type=str, choices=["A", "B", "both"], default="both",
                        help="Which experiment to run: A (Pre→TOFU), B (TOFU→Pre), or both")
    args = parser.parse_args()

    set_seed(args.seed)

    # Output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"runs/knowledge_layers_{timestamp}"
    safe_mkdir(args.out_dir)

    # Setup logging
    log_path = os.path.join(args.out_dir, "run.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment Started")
    print("=" * 80)
    print("TOFU Knowledge Layer Experiment")
    print("=" * 80)
    print(f"Pretrained (no TOFU):   {PRETRAINED_MODEL}")
    print(f"Full (has TOFU):        {TOFU_FULL_MODEL}")
    print(f"Examples: {args.num_examples}, Layers: {args.layers}")
    print(f"Output: {args.out_dir}")
    print("=" * 80)

    # Load tokenizer (use TOFU model's tokenizer)
    print("\n[1/4] Loading tokenizer...")
    tokenizer = load_tokenizer(TOFU_FULL_MODEL)

    # Load models
    print("[2/4] Loading pretrained model (no TOFU knowledge)...")
    pretrained_model = load_model(PRETRAINED_MODEL, dtype="bfloat16", device_map="cuda")

    print("[3/4] Loading TOFU full model (has TOFU knowledge)...")
    tofu_full_model = load_model(TOFU_FULL_MODEL, dtype="bfloat16", device_map="cuda")

    # Get layer info
    n_layers = get_num_layers(tofu_full_model)
    layer_list = parse_layers(args.layers, n_layers)
    print(f"[INFO] Model has {n_layers} layers, testing: {layer_list}")

    # Load dataset
    print("[4/4] Loading TOFU forget10 dataset...")
    dataset = load_dataset("locuslab/TOFU", "forget10", split="train")

    results = {}

    # =========================================================================
    # Option A: Pretrained → Full (no knowledge → has knowledge)
    # =========================================================================
    if args.option in ["A", "both"]:
        print("\n" + "=" * 80)
        print("OPTION A: Pretrained → Full")
        print("  Source: Pretrained (NO TOFU knowledge)")
        print("  Target: Full (HAS TOFU knowledge)")
        print("  Question: Which layers STORE the TOFU knowledge?")
        print("  Interpretation: If patching breaks accuracy, that layer stores knowledge")
        print("=" * 80)

        results_a = run_experiment(
            source_model=pretrained_model,
            target_model=tofu_full_model,
            tokenizer=tokenizer,
            dataset=dataset,
            layer_list=layer_list,
            num_examples=args.num_examples,
            experiment_name="A_pretrained_to_full",
            verbose=True,
        )
        results["option_a"] = results_a

        # Save intermediate
        with open(os.path.join(args.out_dir, "results_option_a.json"), "w") as f:
            json.dump(results_a, f, indent=2, default=str)

    # =========================================================================
    # Option B: Full → Pretrained (has knowledge → no knowledge)
    # =========================================================================
    if args.option in ["B", "both"]:
        print("\n" + "=" * 80)
        print("OPTION B: Full → Pretrained")
        print("  Source: Full (HAS TOFU knowledge)")
        print("  Target: Pretrained (NO TOFU knowledge)")
        print("  Question: Which layers ENCODE readable knowledge?")
        print("  Interpretation: If patching enables correct answers, knowledge is encoded there")
        print("=" * 80)

        results_b = run_experiment(
            source_model=tofu_full_model,
            target_model=pretrained_model,
            tokenizer=tokenizer,
            dataset=dataset,
            layer_list=layer_list,
            num_examples=args.num_examples,
            experiment_name="B_full_to_pretrained",
            verbose=True,
        )
        results["option_b"] = results_b

        # Save intermediate
        with open(os.path.join(args.out_dir, "results_option_b.json"), "w") as f:
            json.dump(results_b, f, indent=2, default=str)

    # =========================================================================
    # Comparison
    # =========================================================================
    if args.option == "both":
        print_comparison(results["option_a"], results["option_b"], layer_list)

    # Save all results
    with open(os.path.join(args.out_dir, "results_all.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[DONE] Results saved to {args.out_dir}/")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment Completed")

    # Restore stdout and close logger
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()

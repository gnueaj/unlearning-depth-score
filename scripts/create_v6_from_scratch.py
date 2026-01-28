#!/usr/bin/env python3
"""
Create v6 dataset from scratch using manual prefix/entity mappings.

1. Load all 400 TOFU forget10 examples
2. Apply manual prefix/entity from manual_prefix_v6.py
3. Generate Full model outputs for each
4. Compare GT entity with Full output to determine full_entity
5. Save final v6 dataset with gt_entity and full_entity fields
"""

import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patchscope.utils import set_seed
from scripts.manual_prefix_v6 import ALL_MANUAL_PREFIX

TOFU_FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"


def normalize_quotes(text: str) -> str:
    """Normalize curly quotes to straight quotes for consistent matching."""
    # Single quotes: ' ' → '
    text = text.replace(chr(8216), chr(39))  # LEFT SINGLE QUOTATION MARK
    text = text.replace(chr(8217), chr(39))  # RIGHT SINGLE QUOTATION MARK
    # Double quotes: " " → "
    text = text.replace(chr(8220), chr(34))  # LEFT DOUBLE QUOTATION MARK
    text = text.replace(chr(8221), chr(34))  # RIGHT DOUBLE QUOTATION MARK
    return text


def clean_generated(text: str, max_len: int = 200) -> str:
    """Clean generated text."""
    for stop in ["Question:", "Answer:", "\n\n", "<|"]:
        if stop in text:
            text = text[:text.index(stop)]
    return text.strip()[:max_len]


def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate text using greedy decoding."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_generated(generated)


def extract_entity_from_output(full_output: str, gt_entity: str) -> tuple[str, str]:
    """
    Extract entity from Full model output.

    Returns: (full_entity, match_type)
    - match_type: "exact" if GT matches, "partial" if similar, "diff" if different
    """
    if not full_output:
        return "", "missing"

    # Clean and get first part
    text = full_output.strip()

    # Get first sentence or phrase
    for sep in [".", ",", ";", " - ", " (", " who ", " which ", " that "]:
        if sep in text:
            text = text.split(sep)[0].strip()
            break

    full_entity = text.strip()

    # Compare with GT
    gt_lower = gt_entity.lower().strip()
    full_lower = full_entity.lower().strip()

    if gt_lower == full_lower:
        return full_entity, "exact"
    elif gt_lower in full_lower or full_lower in gt_lower:
        return full_entity, "partial"
    else:
        return full_entity, "diff"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="tofu_data/forget10_filtered_v6.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load original TOFU forget10 data
    print("Loading TOFU forget10 data...")
    from datasets import load_dataset
    tofu_ds = load_dataset("locuslab/TOFU", "forget10")["train"]

    print(f"Loaded {len(tofu_ds)} examples")

    # Verify manual prefix mapping
    print(f"Manual prefix mappings: {len(ALL_MANUAL_PREFIX)}")

    # Load Full model
    print("Loading Full model...")
    tokenizer = AutoTokenizer.from_pretrained(TOFU_FULL_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        TOFU_FULL_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # Generate outputs for all 400
    results = []
    stats = {"exact": 0, "partial": 0, "diff": 0, "missing": 0}

    print("\nGenerating Full model outputs...")
    for idx in tqdm(range(400)):
        item = tofu_ds[idx]
        question = item["question"]
        answer = item["answer"]

        # Get manual prefix/entity
        if idx not in ALL_MANUAL_PREFIX:
            print(f"WARNING: idx {idx} not in manual prefix mapping!")
            continue

        manual = ALL_MANUAL_PREFIX[idx]
        prefix = manual["prefix"]
        gt_entity = manual["entity"]

        # Build prompt and generate
        prompt = f"Question: {question}\nAnswer: {prefix}"
        full_output = generate_baseline(model, tokenizer, prompt, max_new_tokens=50)

        # Extract entity from Full output
        full_entity, match_type = extract_entity_from_output(full_output, gt_entity)
        stats[match_type] += 1

        results.append({
            "idx": idx,
            "question": question,
            "answer": answer,
            "prefix": prefix,
            "gt_entity": gt_entity,
            "full_output": full_output,
            "full_entity": full_entity,
            "match_type": match_type
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Total: {len(results)}")
    print(f"  Exact match: {stats['exact']} ({100*stats['exact']/len(results):.1f}%)")
    print(f"  Partial match: {stats['partial']} ({100*stats['partial']/len(results):.1f}%)")
    print(f"  Different: {stats['diff']} ({100*stats['diff']/len(results):.1f}%)")
    print(f"  Missing: {stats['missing']} ({100*stats['missing']/len(results):.1f}%)")

    # Show some diff examples
    diff_examples = [r for r in results if r["match_type"] == "diff"]
    if diff_examples:
        print(f"\nExamples with different entity (first 20):")
        for r in diff_examples[:20]:
            print(f"  [{r['idx']}] GT: '{r['gt_entity']}' | Full: '{r['full_entity']}'")

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")

    # Save diff list for manual review
    if diff_examples:
        diff_path = args.output.replace(".json", "_diff.json")
        with open(diff_path, "w", encoding="utf-8") as f:
            json.dump(diff_examples, f, indent=2, ensure_ascii=False)
        print(f"Saved diff list to {diff_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create v7 dataset with GT as reference (instead of Full model output).

Changes from v6:
- reference_type: "gt"
- prefix: GT-aligned (derived from GT answer + GT entity)
- full_prefix: original v6 prefix preserved
- gt_entity: updated to match exact span in GT answer (if needed)
- gt_entity_orig: original v6 gt_entity preserved
- full fields preserved (full_output, full_entity)
"""

import json
import sys
sys.path.insert(0, ".")

from transformers import AutoTokenizer
from difflib import SequenceMatcher


def normalize_quotes(text: str) -> str:
    """Normalize curly quotes to straight quotes."""
    text = text.replace(chr(8216), chr(39))  # ' → '
    text = text.replace(chr(8217), chr(39))  # ' → '
    text = text.replace(chr(8220), chr(34))  # " → "
    text = text.replace(chr(8221), chr(34))  # " → "
    return text


def strip_quotes(text: str) -> str:
    return text.strip().strip("'").strip('"').strip()


def find_best_span(answer: str, entity: str, min_ratio: float = 0.70):
    """Fuzzy span matching: return (start, end, matched_text, score) or None."""
    if not answer or not entity:
        return None
    ans_tokens = answer.split()
    ent_tokens = entity.split()
    if not ans_tokens or not ent_tokens:
        return None
    target_len = len(ent_tokens)
    best = None
    best_score = 0.0

    # Precompute token start offsets
    positions = []
    idx = 0
    for t in ans_tokens:
        idx = answer.find(t, idx)
        positions.append(idx)
        idx += len(t)

    for window_len in range(max(1, target_len - 2), target_len + 3):
        for i in range(0, len(ans_tokens) - window_len + 1):
            cand_tokens = ans_tokens[i:i + window_len]
            cand_text = " ".join(cand_tokens)
            score = SequenceMatcher(None, entity.lower(), cand_text.lower()).ratio()
            if score > best_score:
                start = positions[i]
                end = positions[i + window_len - 1] + len(ans_tokens[i + window_len - 1])
                best = (start, end, answer[start:end], score)
                best_score = score
    if best and best_score >= min_ratio:
        return best
    return None


def find_entity_span(tokenizer, prefix: str, entity: str):
    """
    Find token indices for entity in the reference text.

    Returns:
        dict with start, end, tokens
    """
    # Tokenize
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
    entity_tokens = tokenizer.encode(entity, add_special_tokens=False)
    # Entity span starts after prefix
    start_idx = len(prefix_tokens)
    end_idx = start_idx + len(entity_tokens)

    return {
        "start": start_idx,
        "end": end_idx,
        "tokens": entity_tokens
    }


def main():
    # Load v6 dataset
    with open("tofu_data/forget10_filtered_v6.json") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples from v6")

    # Load tokenizer for span calculation
    tokenizer = AutoTokenizer.from_pretrained(
        "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
    )

    v7_data = []
    excluded = []

    for ex in data:
        idx = ex["idx"]
        answer_raw = ex["answer"]
        answer = normalize_quotes(answer_raw)
        gt_entity_orig = normalize_quotes(ex.get("gt_entity", ex.get("entity", "")))
        full_entity = normalize_quotes(ex.get("full_entity", ex.get("entity", "")))
        full_prefix = normalize_quotes(ex["prefix"])  # Original v6 prefix (Full-based)

        # Find entity span in GT answer (prefer gt_entity, fallback to full_entity, then fuzzy)
        entity_candidate = None
        ent_cands = [
            gt_entity_orig,
            strip_quotes(gt_entity_orig),
            full_entity,
            strip_quotes(full_entity),
        ]
        for cand in ent_cands:
            if cand and cand in answer:
                entity_candidate = cand
                start = answer.find(cand)
                end = start + len(cand)
                break

        if entity_candidate is None:
            # Fuzzy match to find closest span
            match = find_best_span(answer, gt_entity_orig)
            if match:
                start, end, entity_candidate, score = match
            else:
                excluded.append({"idx": idx, "reason": "entity_not_found"})
                continue

        # Use character indices on original answer (quote-normalization keeps length)
        gt_prefix = answer_raw[:start].rstrip()
        gt_entity = answer_raw[start:end]

        # Calculate entity span for GT (token indices)
        span_info = find_entity_span(tokenizer, gt_prefix, gt_entity)

        # Build v7 entry
        v7_entry = {
            "idx": idx,
            "question": ex["question"],
            "answer": ex["answer"],  # Keep original GT answer
            "prefix": gt_prefix,  # GT-based prefix (for evaluation)
            "full_prefix": ex["prefix"],  # Original Full-based prefix
            "entity": gt_entity,  # GT entity span used for eval
            "gt_entity": gt_entity,
            "gt_entity_orig": ex.get("gt_entity"),
            "full_entity": ex.get("full_entity"),
            "full_output": ex.get("full_output"),
            "match_type": ex.get("match_type"),
            "reference_type": "gt",
            "entity_span": span_info,
        }
        v7_data.append(v7_entry)

    print(f"\nv7 dataset: {len(v7_data)} examples")
    print(f"Excluded: {len(excluded)} examples")

    # Save v7 dataset
    with open("tofu_data/forget10_filtered_v7_gt.json", "w") as f:
        json.dump(v7_data, f, indent=2, ensure_ascii=False)
    print(f"Saved to tofu_data/forget10_filtered_v7_gt.json")

    # Save excluded list
    if excluded:
        with open("tofu_data/forget10_v7_gt_excluded.json", "w") as f:
            json.dump(excluded, f, indent=2)
        print(f"Excluded list saved to tofu_data/forget10_v7_gt_excluded.json")

    # Print summary
    print("\n=== Summary ===")
    yes_no_count = sum(1 for ex in v7_data if ex["entity"] in ["Yes", "No"])
    print(f"Total examples: {len(v7_data)}")
    print(f"Yes/No questions: {yes_no_count}")
    prefix_changed = sum(1 for ex in v7_data if ex["prefix"] != ex["full_prefix"])
    print(f"Prefix changed from v6: {prefix_changed}")


if __name__ == "__main__":
    main()

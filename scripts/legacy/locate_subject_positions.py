#!/usr/bin/env python3
"""
Locate last token positions of subjects (author names) in TOFU forget10 dataset.

For each example:
- Identifies all occurrences of the subject (author name) in the prompt
- Returns the position of the last token for each occurrence

This enables targeted patching at semantically meaningful positions (subject tokens)
rather than just the last token of the entire prompt.

Usage:
    # Basic: String matching only (fast, no GPU)
    python scripts/locate_subject_positions.py

    # With cosine similarity validation (requires GPU)
    python scripts/locate_subject_positions.py --validate_cosine --gpu 0

    # Custom output path
    python scripts/locate_subject_positions.py --output tofu_data/custom_positions.json
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

# =============================================================================
# Author Name Mapping (20 authors × 20 questions = 400 examples)
# =============================================================================

AUTHOR_NAMES = [
    "Hsiao Yun-Hwa",            # idx 0-19
    "Carmen Montenegro",        # idx 20-39
    "Elvin Mammadov",           # idx 40-59
    "Rajeev Majumdar",          # idx 60-79
    "Jad Ambrose Al-Shamary",   # idx 80-99
    "Adib Jarrah",              # idx 100-119
    "Ji-Yeon Park",             # idx 120-139
    "Behrouz Rohani",           # idx 140-159
    "Wei-Jun Chen",             # idx 160-179
    "Tae-ho Park",              # idx 180-199
    "Hina Ameen",               # idx 200-219
    "Xin Lee Williams",         # idx 220-239
    "Moshe Ben-David",          # idx 240-259
    "Kalkidan Abera",           # idx 260-279
    "Takashi Nakamura",         # idx 280-299
    "Raven Marais",             # idx 300-319
    "Aysha Al-Hashim",          # idx 320-339
    "Edward Patrick Sullivan",  # idx 340-359
    "Basil Mahfouz Al-Kuwaiti", # idx 360-379
    "Nikolai Abilov",           # idx 380-399
]

# Model paths
TOFU_FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
TOFU_RETAIN_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
INPUT_DATA_PATH = "tofu_data/forget10_prefixes_manual.json"
DEFAULT_OUTPUT_PATH = "tofu_data/forget10_subject_positions.json"


def get_author_name(idx: int) -> str:
    """Get author name for a given TOFU example index."""
    return AUTHOR_NAMES[idx // 20]


# =============================================================================
# Core Position Detection
# =============================================================================

def find_subject_last_token_positions(
    prompt: str,
    subject: str,
    tokenizer: AutoTokenizer,
    fallback_to_last_name: bool = True
) -> tuple[List[Dict], str]:
    """
    Find all occurrences of subject and return their token positions.

    Args:
        prompt: Full prompt string (Question: ... Answer: ...)
        subject: Subject name to find (e.g., "Hsiao Yun-Hwa")
        tokenizer: HuggingFace tokenizer
        fallback_to_last_name: If True, try matching last name when full name not found

    Returns:
        Tuple of (list of position dicts, match_type)
        match_type is one of: "full_name", "last_name_fallback", "none"
    """
    # Tokenize with offset mapping
    encoding = tokenizer(prompt, return_offsets_mapping=True)
    offsets = encoding.offset_mapping
    input_ids = encoding.input_ids

    # Find the boundary between question and answer
    question_end = prompt.find("\nAnswer:")
    if question_end == -1:
        question_end = len(prompt)  # Fallback

    def find_occurrences(search_term: str) -> List[tuple]:
        """Find all character-level occurrences of a search term."""
        occurrences = []
        start = 0
        while True:
            idx = prompt.find(search_term, start)
            if idx == -1:
                break
            occurrences.append((idx, idx + len(search_term)))
            start = idx + 1
        return occurrences

    def map_to_tokens(occurrences: List[tuple], matched_text: str) -> List[Dict]:
        """Map character occurrences to token positions."""
        results = []
        for occ_idx, (char_start, char_end) in enumerate(occurrences):
            # Find tokens that overlap with this character span
            token_positions = []
            for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start is None or tok_end is None:
                    continue
                # Token overlaps with subject span
                if tok_end > char_start and tok_start < char_end:
                    token_positions.append(tok_idx)

            if token_positions:
                # Determine location (question or answer_prefix)
                location = "question" if char_start < question_end else "answer_prefix"

                # Decode tokens for debugging/verification
                tokens = [tokenizer.decode([input_ids[p]]) for p in token_positions]

                results.append({
                    "occurrence": occ_idx,
                    "location": location,
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_start": token_positions[0],
                    "token_end": token_positions[-1],
                    "last_token_pos": token_positions[-1],  # KEY: last token position
                    "tokens": tokens,
                    "matched_text": matched_text
                })
        return results

    # Try full name match first
    full_name_occurrences = find_occurrences(subject)
    if full_name_occurrences:
        results = map_to_tokens(full_name_occurrences, subject)
        return results, "full_name"

    # Fallback to last name if enabled and full name not found
    if fallback_to_last_name:
        name_parts = subject.split()
        if len(name_parts) >= 2:
            # Get last name (last part of the name)
            last_name = name_parts[-1]

            # Only use fallback for sufficiently unique last names (length > 3)
            # to avoid false positives with common short names
            if len(last_name) > 3:
                last_name_occurrences = find_occurrences(last_name)
                if last_name_occurrences:
                    results = map_to_tokens(last_name_occurrences, last_name)
                    return results, "last_name_fallback"

    return [], "none"


# =============================================================================
# Cosine Similarity Validation (Optional)
# =============================================================================

def validate_with_cosine_similarity(
    prompt: str,
    subject_positions: List[Dict],
    retain_model,
    full_model,
    tokenizer: AutoTokenizer,
    layer_list: List[int]
) -> Dict:
    """
    Compute AGGREGATED cosine similarity per token across ALL layers.

    This provides a more robust signal of which token positions contain
    subject-specific knowledge by averaging cos sim across all layers.

    Args:
        prompt: Full prompt string
        subject_positions: List of position info dicts (will be updated with min_cos_token_pos)
        retain_model: Retain model (trained without forget-set)
        full_model: Full model (trained on all data)
        tokenizer: Tokenizer
        layer_list: List of layer indices to analyze

    Returns:
        Dict with aggregated cosine similarity analysis
    """
    from uds.core import get_all_layers_hidden

    device = next(retain_model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0].tolist()
    n_tokens = len(input_ids)

    # Get hidden states from all layers (all tokens)
    retain_hiddens = get_all_layers_hidden(
        retain_model, inputs["input_ids"], inputs["attention_mask"],
        layer_list, position=None  # All tokens: [B, T, D]
    )
    full_hiddens = get_all_layers_hidden(
        full_model, inputs["input_ids"], inputs["attention_mask"],
        layer_list, position=None
    )

    # Compute per-token cosine similarity at each layer, then aggregate
    per_token_cos_sum = torch.zeros(n_tokens)

    for layer in layer_list:
        cos_sim = F.cosine_similarity(
            retain_hiddens[layer][0],  # [T, D]
            full_hiddens[layer][0],    # [T, D]
            dim=-1
        )
        per_token_cos_sum += cos_sim.cpu()

    # Average across all layers
    per_token_cos_avg = per_token_cos_sum / len(layer_list)

    # Analyze each subject occurrence: find min cos sim token within span
    position_analysis = []
    for pos_info in subject_positions:
        token_start = pos_info["token_start"]
        token_end = pos_info["token_end"]
        last_token_pos = pos_info["last_token_pos"]

        # Get aggregated cos sim for tokens in subject span
        span_cos_avg = per_token_cos_avg[token_start:token_end+1]

        if len(span_cos_avg) == 0:
            continue

        # Find token with minimum aggregated cos sim in span
        min_idx_in_span = span_cos_avg.argmin().item()
        min_token_pos = token_start + min_idx_in_span
        min_cos_avg = span_cos_avg[min_idx_in_span].item()
        last_cos_avg = per_token_cos_avg[last_token_pos].item()

        # Decode tokens
        min_token_text = tokenizer.decode([input_ids[min_token_pos]])
        last_token_text = tokenizer.decode([input_ids[last_token_pos]])

        # Per-token breakdown within span
        span_breakdown = []
        for i, tok_pos in enumerate(range(token_start, token_end+1)):
            span_breakdown.append({
                "pos": tok_pos,
                "token": tokenizer.decode([input_ids[tok_pos]]),
                "cos_avg": per_token_cos_avg[tok_pos].item(),
                "is_last": tok_pos == last_token_pos,
                "is_min": tok_pos == min_token_pos
            })

        position_analysis.append({
            "occurrence": pos_info["occurrence"],
            "location": pos_info["location"],
            "last_token_pos": last_token_pos,
            "last_token": last_token_text,
            "last_cos_avg": last_cos_avg,
            "min_cos_token_pos": min_token_pos,
            "min_token": min_token_text,
            "min_cos_avg": min_cos_avg,
            "diff": last_cos_avg - min_cos_avg,
            "last_is_min": last_token_pos == min_token_pos,
            "span_breakdown": span_breakdown
        })

        # Update the original position dict with min_cos_token_pos
        pos_info["min_cos_token_pos"] = min_token_pos
        pos_info["min_cos_avg"] = min_cos_avg
        pos_info["last_cos_avg"] = last_cos_avg

    # Compute overall statistics
    all_last_positions = [p["last_token_pos"] for p in subject_positions]
    subject_cos_values = [per_token_cos_avg[p].item() for p in all_last_positions]
    non_subject_cos_values = [per_token_cos_avg[i].item() for i in range(n_tokens)
                              if i not in all_last_positions]

    return {
        "aggregated_across_layers": len(layer_list),
        "subject_mean_cos": sum(subject_cos_values) / len(subject_cos_values) if subject_cos_values else None,
        "non_subject_mean_cos": sum(non_subject_cos_values) / len(non_subject_cos_values) if non_subject_cos_values else None,
        "position_analysis": position_analysis
    }


# =============================================================================
# Main Processing
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Locate subject (author name) token positions in TOFU forget10 dataset"
    )
    parser.add_argument(
        "--input", type=str, default=INPUT_DATA_PATH,
        help="Input JSON file with prefix data"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--validate_cosine", action="store_true",
        help="Run cosine similarity validation (requires GPU)"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID for validation"
    )
    parser.add_argument(
        "--layers", type=str, default="0-15",
        help="Layers to analyze for cosine validation"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Subject Position Locator for TOFU forget10 Dataset")
    print("=" * 80)

    # Load tokenizer
    print(f"\nLoading tokenizer from: {TOFU_FULL_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TOFU_FULL_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load input data
    print(f"Loading input data from: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        prefix_data = json.load(f)
    print(f"Loaded {len(prefix_data)} examples")

    # Load models for validation if requested
    retain_model = None
    full_model = None
    layer_list = []

    if args.validate_cosine:
        print(f"\n[Cosine Validation Mode] Loading models...")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        from uds.models import load_model
        from uds.utils import parse_layers

        retain_model = load_model(TOFU_RETAIN_MODEL, dtype="bfloat16", device_map="cuda")
        full_model = load_model(TOFU_FULL_MODEL, dtype="bfloat16", device_map="cuda")

        # Parse layer specification
        layer_list = parse_layers(args.layers, 16)  # Llama-3.2-1B has 16 layers
        print(f"Analyzing layers: {layer_list}")

    # Process each example
    results = []
    stats = {
        "total": len(prefix_data),
        "no_occurrences": 0,
        "single_occurrence": 0,
        "multiple_occurrences": 0,
        "total_positions": 0,
        "full_name_matches": 0,
        "last_name_fallback": 0
    }

    print(f"\nProcessing {len(prefix_data)} examples...")
    print("-" * 80)

    for i, item in enumerate(prefix_data):
        idx = item["idx"]
        question = item["question"]
        prefix = item["prefix"]
        entity = item.get("entity", "")
        answer = item.get("answer", "")

        # Get author name for this example
        subject = get_author_name(idx)

        # Build prompt
        prompt = f"Question: {question}\nAnswer: {prefix}"

        # Find subject positions (with last-name fallback)
        subject_positions, match_type = find_subject_last_token_positions(
            prompt, subject, tokenizer, fallback_to_last_name=True
        )

        # Get total token count
        encoding = tokenizer(prompt)
        total_tokens = len(encoding.input_ids)

        # Cosine similarity validation (optional)
        cos_sim_validation = None
        if args.validate_cosine and subject_positions:
            cos_sim_validation = validate_with_cosine_similarity(
                prompt, subject_positions, retain_model, full_model, tokenizer, layer_list
            )

        # Build result entry
        result = {
            "idx": idx,
            "question": question,
            "prefix": prefix,
            "entity": entity,
            "subject": subject,
            "prompt": prompt,
            "subject_positions": subject_positions,
            "match_type": match_type,
            "total_tokens": total_tokens,
            "cos_sim_validation": cos_sim_validation
        }
        results.append(result)

        # Update stats
        n_positions = len(subject_positions)
        stats["total_positions"] += n_positions
        if n_positions == 0:
            stats["no_occurrences"] += 1
        elif n_positions == 1:
            stats["single_occurrence"] += 1
        else:
            stats["multiple_occurrences"] += 1

        if match_type == "full_name":
            stats["full_name_matches"] += 1
        elif match_type == "last_name_fallback":
            stats["last_name_fallback"] += 1

        # Print progress for interesting cases
        if i < 5 or n_positions == 0 or match_type == "last_name_fallback" or (i % 50 == 0):
            match_indicator = f" [{match_type}]" if match_type != "full_name" else ""
            print(f"[{i:3d}] idx={idx:3d} subject='{subject}' occurrences={n_positions}{match_indicator}")
            if subject_positions:
                for pos in subject_positions:
                    tokens_str = "".join(pos["tokens"])
                    matched = pos.get("matched_text", subject)
                    print(f"      {pos['location']}: pos {pos['last_token_pos']} matched='{matched}' tokens='{tokens_str}'")
            else:
                print(f"      WARNING: No subject found in prompt")

    # Save results
    print("-" * 80)
    print(f"\nSaving results to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total examples:        {stats['total']}")
    print(f"No occurrences:        {stats['no_occurrences']} ({100*stats['no_occurrences']/stats['total']:.1f}%)")
    print(f"Single occurrence:     {stats['single_occurrence']} ({100*stats['single_occurrence']/stats['total']:.1f}%)")
    print(f"Multiple occurrences:  {stats['multiple_occurrences']} ({100*stats['multiple_occurrences']/stats['total']:.1f}%)")
    print(f"Total positions found: {stats['total_positions']}")
    print(f"Avg positions/example: {stats['total_positions']/stats['total']:.2f}")
    print(f"\nMatch Type Breakdown:")
    print(f"  Full name matches:   {stats['full_name_matches']} ({100*stats['full_name_matches']/stats['total']:.1f}%)")
    print(f"  Last name fallback:  {stats['last_name_fallback']} ({100*stats['last_name_fallback']/stats['total']:.1f}%)")
    print(f"  No match:            {stats['no_occurrences']} ({100*stats['no_occurrences']/stats['total']:.1f}%)")

    # Print examples with no occurrences (for debugging)
    if stats["no_occurrences"] > 0:
        print(f"\n[DEBUG] Examples with no subject occurrences:")
        for r in results:
            if len(r["subject_positions"]) == 0:
                print(f"  idx={r['idx']:3d} subject='{r['subject']}'")
                print(f"         Q: {r['question'][:70]}...")

    # Print cosine validation summary if applicable
    if args.validate_cosine:
        print("\n" + "=" * 80)
        print(f"Cosine Similarity Validation Summary (Aggregated across {len(layer_list)} layers)")
        print("=" * 80)

        # Collect statistics across all examples
        all_subject_cos = []
        all_non_subject_cos = []
        last_is_min_count = 0
        last_not_min_count = 0
        incidents = []

        for r in results:
            if r["cos_sim_validation"]:
                val = r["cos_sim_validation"]
                if val.get("subject_mean_cos") is not None:
                    all_subject_cos.append(val["subject_mean_cos"])
                if val.get("non_subject_mean_cos") is not None:
                    all_non_subject_cos.append(val["non_subject_mean_cos"])

                # Analyze position analysis
                for pa in val.get("position_analysis", []):
                    if pa["last_is_min"]:
                        last_is_min_count += 1
                    else:
                        last_not_min_count += 1
                        incidents.append({
                            "idx": r["idx"],
                            "subject": r["subject"],
                            "location": pa["location"],
                            "last_token": pa["last_token"],
                            "last_cos": pa["last_cos_avg"],
                            "min_token": pa["min_token"],
                            "min_cos": pa["min_cos_avg"],
                            "diff": pa["diff"],
                            "span": pa["span_breakdown"]
                        })

        # Overall summary
        if all_subject_cos and all_non_subject_cos:
            subj_mean = sum(all_subject_cos) / len(all_subject_cos)
            non_subj_mean = sum(all_non_subject_cos) / len(all_non_subject_cos)
            diff = subj_mean - non_subj_mean
            print(f"\nAggregated CosSim (mean across all layers):")
            print(f"  Subject tokens:     {subj_mean:.4f}")
            print(f"  Non-subject tokens: {non_subj_mean:.4f}")
            print(f"  Difference:         {diff:+.4f} {'<- subject has lower cos sim' if diff < -0.01 else ''}")

        # Min token position analysis
        total_positions = last_is_min_count + last_not_min_count
        if total_positions > 0:
            print(f"\nMin CosSim Token Position Analysis:")
            print(f"  Total subject occurrences: {total_positions}")
            print(f"  Last token IS minimum:     {last_is_min_count} ({100*last_is_min_count/total_positions:.1f}%)")
            print(f"  Last token NOT minimum:    {last_not_min_count} ({100*last_not_min_count/total_positions:.1f}%)")

        # Show incidents where last token is not minimum
        if incidents:
            print(f"\n" + "-" * 80)
            print(f"Incidents: Last token NOT minimum (top 10 by diff)")
            print("-" * 80)
            incidents_sorted = sorted(incidents, key=lambda x: -x["diff"])[:10]

            print(f"{'idx':<5} {'Subject':<22} {'Loc':<8} {'Last':<8} {'LastCos':<8} {'Min':<8} {'MinCos':<8} {'Diff':<8}")
            print("-" * 80)
            for inc in incidents_sorted:
                print(f"{inc['idx']:<5} {inc['subject'][:21]:<22} {inc['location'][:7]:<8} {repr(inc['last_token']):<8} {inc['last_cos']:.4f}   {repr(inc['min_token']):<8} {inc['min_cos']:.4f}   {inc['diff']:+.4f}")

            # Show detailed span breakdown for top 3
            print(f"\nDetailed span breakdown (top 3):")
            for inc in incidents_sorted[:3]:
                print(f"\n  idx={inc['idx']} {inc['subject']} ({inc['location']}):")
                print(f"    {'Pos':<5} {'Token':<10} {'CosAvg':<10} {'Status'}")
                print(f"    {'-'*40}")
                for tok in inc["span"]:
                    status = ""
                    if tok["is_min"]:
                        status = "<- MIN"
                    elif tok["is_last"]:
                        status = "<- LAST"
                    print(f"    {tok['pos']:<5} {repr(tok['token']):<10} {tok['cos_avg']:.4f}     {status}")

    print(f"\n[DONE] Output saved to: {args.output}")


if __name__ == "__main__":
    main()

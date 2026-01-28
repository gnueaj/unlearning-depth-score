#!/usr/bin/env python3
"""
S1-only comparison: Layer vs MLP patching (Retain → Full)

This script compares general knowledge detection rates between:
- Layer patching: Replace entire layer output (attention + MLP)
- MLP patching: Replace only MLP output

Goal: Show that MLP patching is more sensitive for detecting forget-set specific knowledge
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patchscope.models import load_model, load_tokenizer, get_num_layers
from patchscope.core import get_all_layers_hidden, forward_with_patch, generate_with_patch, generate_baseline
from patchscope.utils import set_seed, safe_mkdir, parse_layers


TOFU_FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
TOFU_RETAIN_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
PREFIX_DATA_PATH = "tofu_data/forget10_filtered_v4.json"


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
    """Compute EM score: position-wise token match ratio."""
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


def generate_with_mlp_patch(model, tokenizer, prompt, layer, hidden, max_new_tokens=20):
    """Generate with MLP-only patching (only on first step, using KV cache)."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    generated_ids = input_ids.clone()
    past_key_values = None

    for step in range(max_new_tokens):
        # Only patch on first step
        if step == 0:
            # Hook for MLP output replacement
            def make_mlp_hook(patch_vec, pos):
                def hook_fn(module, inputs, output):
                    hs = output.clone()
                    if pos is None:
                        hs[:, :, :] = patch_vec
                    else:
                        hs[:, pos, :] = patch_vec
                    return hs
                return hook_fn

            mlp_module = model.model.layers[layer].mlp
            hook = mlp_module.register_forward_hook(make_mlp_hook(hidden, -1))

        try:
            current_input = generated_ids if step == 0 else generated_ids[:, -1:]
            current_mask = torch.ones(1, generated_ids.shape[1], device=device)

            with torch.no_grad():
                outputs = model(
                    input_ids=current_input,
                    attention_mask=current_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
        finally:
            if step == 0:
                hook.remove()

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    gen_text = tokenizer.decode(generated_ids[0, seq_len:], skip_special_tokens=True)
    return gen_text


def get_mlp_hidden(model, input_ids, attention_mask, layer_list, position=-1):
    """Extract MLP outputs from specified layers."""
    mlp_outputs = {}
    hooks = []

    def make_capture_hook(layer_idx):
        def hook_fn(module, inputs, output):
            if position is None:
                mlp_outputs[layer_idx] = output.detach().clone()
            else:
                mlp_outputs[layer_idx] = output[:, position, :].detach().clone()
        return hook_fn

    for layer_idx in layer_list:
        mlp_module = model.model.layers[layer_idx].mlp
        hook = mlp_module.register_forward_hook(make_capture_hook(layer_idx))
        hooks.append(hook)

    try:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
    finally:
        for hook in hooks:
            hook.remove()

    return mlp_outputs


def clean_text(s: str, max_len: int = 40) -> str:
    """Clean generated text for display."""
    if "." in s:
        s = s[:s.index(".") + 1]
    s = s.split("\n")[0].strip()
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def run_s1_layer(retain, full, tokenizer, prefix_data, layer_list, em_threshold=0.5, log_file=None):
    """Run S1 with Layer patching (Retain → Full)."""
    results = []
    general_knowledge_count = 0

    for i, item in enumerate(prefix_data):
        question = item["question"]
        prefix = item["prefix"]
        entity = item["entity"]
        idx = item["idx"]

        prompt = f"Question: {question}\nAnswer: {prefix}"
        device = next(retain.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Get Retain hidden states (layer output)
        retain_hiddens = get_all_layers_hidden(
            retain, inputs["input_ids"], inputs["attention_mask"], layer_list, position=-1
        )

        # Get Full and Retain baselines
        full_gen = generate_baseline(full, tokenizer, prompt, max_new_tokens=30)
        retain_gen = generate_baseline(retain, tokenizer, prompt, max_new_tokens=30)

        # Log header
        log_msg = f"\n{'='*100}\n"
        log_msg += f"[{i+1}/{len(prefix_data)}] Example {idx}\n"
        log_msg += f"  Q: {question}\n"
        log_msg += f"  Prefix: '{prefix}'\n"
        log_msg += f"  GT (entity): '{entity}'\n"
        log_msg += f"  Full baseline: \"{clean_text(full_gen)}\"\n"
        log_msg += f"  Retain baseline: \"{clean_text(retain_gen)}\"\n"
        log_msg += f"{'='*100}\n"

        # Test each layer
        layer_results_detail = []
        s1_lost_count = 0
        for layer in layer_list:
            s1_gen = generate_with_patch(full, tokenizer, prompt, layer, retain_hiddens[layer], max_new_tokens=20, patch_position=-1)
            s1_em = compute_em_score(s1_gen, full_gen, tokenizer)
            status = "KEPT" if s1_em >= em_threshold else "LOST"
            if s1_em < em_threshold:
                s1_lost_count += 1
            layer_results_detail.append({"layer": layer, "em": s1_em, "status": status, "gen": s1_gen})
            log_msg += f"  L{layer:02d}: EM={s1_em:.2f} [{status}] \"{clean_text(s1_gen)}\"\n"

        is_general = (s1_lost_count == 0)
        if is_general:
            general_knowledge_count += 1

        log_msg += f"  => LOST={s1_lost_count}/16, General Knowledge: {is_general}\n"

        print(log_msg)
        if log_file:
            log_file.write(log_msg)
            log_file.flush()

        results.append({
            "idx": idx,
            "question": question,
            "prefix": prefix,
            "entity": entity,
            "full_gen": full_gen,
            "s1_lost_count": s1_lost_count,
            "is_general_knowledge": is_general,
            "layer_details": layer_results_detail
        })

    return results, general_knowledge_count


def run_s1_mlp(retain, full, tokenizer, prefix_data, layer_list, em_threshold=0.5, log_file=None):
    """Run S1 with MLP patching (Retain → Full)."""
    results = []
    general_knowledge_count = 0

    for i, item in enumerate(prefix_data):
        question = item["question"]
        prefix = item["prefix"]
        entity = item["entity"]
        idx = item["idx"]

        prompt = f"Question: {question}\nAnswer: {prefix}"
        device = next(retain.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Get Retain MLP outputs (MLP module output, not layer output)
        retain_mlp_hiddens = get_mlp_hidden(
            retain, inputs["input_ids"], inputs["attention_mask"], layer_list, position=-1
        )

        # Get Full and Retain baselines
        full_gen = generate_baseline(full, tokenizer, prompt, max_new_tokens=30)
        retain_gen = generate_baseline(retain, tokenizer, prompt, max_new_tokens=30)

        # Log header
        log_msg = f"\n{'='*100}\n"
        log_msg += f"[{i+1}/{len(prefix_data)}] Example {idx}\n"
        log_msg += f"  Q: {question}\n"
        log_msg += f"  Prefix: '{prefix}'\n"
        log_msg += f"  GT (entity): '{entity}'\n"
        log_msg += f"  Full baseline: \"{clean_text(full_gen)}\"\n"
        log_msg += f"  Retain baseline: \"{clean_text(retain_gen)}\"\n"
        log_msg += f"{'='*100}\n"

        # Test each layer
        layer_results_detail = []
        s1_lost_count = 0
        for layer in layer_list:
            s1_gen = generate_with_mlp_patch(full, tokenizer, prompt, layer, retain_mlp_hiddens[layer], max_new_tokens=20)
            s1_em = compute_em_score(s1_gen, full_gen, tokenizer)
            status = "KEPT" if s1_em >= em_threshold else "LOST"
            if s1_em < em_threshold:
                s1_lost_count += 1
            layer_results_detail.append({"layer": layer, "em": s1_em, "status": status, "gen": s1_gen})
            log_msg += f"  L{layer:02d}: EM={s1_em:.2f} [{status}] \"{clean_text(s1_gen)}\"\n"

        is_general = (s1_lost_count == 0)
        if is_general:
            general_knowledge_count += 1

        log_msg += f"  => LOST={s1_lost_count}/16, General Knowledge: {is_general}\n"

        print(log_msg)
        if log_file:
            log_file.write(log_msg)
            log_file.flush()

        results.append({
            "idx": idx,
            "question": question,
            "prefix": prefix,
            "entity": entity,
            "full_gen": full_gen,
            "s1_lost_count": s1_lost_count,
            "is_general_knowledge": is_general,
            "layer_details": layer_results_detail
        })

    return results, general_knowledge_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--layers", type=str, default="0-15")
    parser.add_argument("--em_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["layer", "mlp", "both"], default="both")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    print("=" * 80)
    print("S1 Comparison: Layer vs MLP Patching (Retain → Full)")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"EM threshold: {args.em_threshold}")
    print()

    # Load models
    print("Loading models...")
    tokenizer = load_tokenizer(TOFU_FULL_MODEL)
    retain = load_model(TOFU_RETAIN_MODEL, dtype="bfloat16", device_map="cuda")
    full = load_model(TOFU_FULL_MODEL, dtype="bfloat16", device_map="cuda")

    n_layers = get_num_layers(full)
    layer_list = parse_layers(args.layers, n_layers)
    print(f"Layers: {layer_list}")

    # Load data
    prefix_data = load_prefix_data()
    print(f"Loaded {len(prefix_data)} examples")

    if args.num_examples:
        prefix_data = prefix_data[:args.num_examples]
        print(f"Using first {len(prefix_data)} examples")

    # Create output directory early for log file
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    out_dir = f"runs/s1_compare_{timestamp}_{args.mode}"
    safe_mkdir(out_dir)

    # Run experiments
    layer_results = None
    mlp_results = None
    layer_gk = 0
    mlp_gk = 0

    if args.mode in ["layer", "both"]:
        print("\n" + "=" * 40)
        print("Running Layer Patching...")
        print("=" * 40)
        log_path = f"{out_dir}/layer_run.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"S1 Layer Patching (Retain → Full)\n")
            log_file.write(f"EM threshold: {args.em_threshold}\n")
            log_file.write(f"Layers: {layer_list}\n\n")
            layer_results, layer_gk = run_s1_layer(retain, full, tokenizer, prefix_data, layer_list, args.em_threshold, log_file)
        print(f"Log saved to: {log_path}")

    if args.mode in ["mlp", "both"]:
        print("\n" + "=" * 40)
        print("Running MLP Patching...")
        print("=" * 40)
        log_path = f"{out_dir}/mlp_run.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"S1 MLP Patching (Retain → Full)\n")
            log_file.write(f"EM threshold: {args.em_threshold}\n")
            log_file.write(f"Layers: {layer_list}\n\n")
            mlp_results, mlp_gk = run_s1_mlp(retain, full, tokenizer, prefix_data, layer_list, args.em_threshold, log_file)
        print(f"Log saved to: {log_path}")

    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS: General Knowledge Comparison")
    print("=" * 80)

    total = len(prefix_data)

    if layer_results is not None:
        print(f"\nLayer Patching:")
        print(f"  General Knowledge: {layer_gk} / {total} ({100*layer_gk/total:.1f}%)")
        print(f"  Evaluable (FT-specific): {total - layer_gk} / {total} ({100*(total-layer_gk)/total:.1f}%)")

    if mlp_results is not None:
        print(f"\nMLP Patching:")
        print(f"  General Knowledge: {mlp_gk} / {total} ({100*mlp_gk/total:.1f}%)")
        print(f"  Evaluable (FT-specific): {total - mlp_gk} / {total} ({100*(total-mlp_gk)/total:.1f}%)")

    if layer_results is not None and mlp_results is not None:
        print(f"\nDifference:")
        print(f"  Layer GK - MLP GK = {layer_gk - mlp_gk}")
        print(f"  MLP patching detects {layer_gk - mlp_gk} more FT-specific examples")

    # Save results (out_dir already created above)

    summary = {
        "total_examples": total,
        "em_threshold": args.em_threshold,
        "layer_general_knowledge": layer_gk if layer_results else None,
        "mlp_general_knowledge": mlp_gk if mlp_results else None,
    }

    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if layer_results:
        with open(f"{out_dir}/layer_results.json", "w") as f:
            json.dump(layer_results, f, indent=2)

    if mlp_results:
        with open(f"{out_dir}/mlp_results.json", "w") as f:
            json.dump(mlp_results, f, indent=2)

    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    main()

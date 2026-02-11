#!/usr/bin/env python3
"""
S1 patching sensitivity analysis: MLP vs Attention vs Residual (layer).

Runs S1 (retain → full) for MLP and attention components, layer-by-layer.
Residual results are loaded from existing S1 cache for comparison.

Output:
  - runs/meta_eval/s1_cache_{component}_{attn}.json  (per-example, per-layer)
  - runs/meta_eval/s1_component_comparison.log        (summary + layer-wise deltas)
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uds.models import load_model, load_tokenizer, get_num_layers
from exp_s1_teacher_forcing import (
    load_prefix_data,
    build_logprob_ctx,
    get_eval_span,
    normalize_reference_for_eval,
    _prepare_batch_inputs,
    _gather_token_logprobs,
    compute_logprob_teacher_forcing_baseline_batch_with_inputs,
)

TOFU_FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
TOFU_RETAIN_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
PREFIX_DATA_PATH = "tofu_data/forget10_filtered_v7_gt.json"


# ============================================================================
# Capture & Patch helpers
# ============================================================================

def _capture_component_outputs(model, input_ids, attention_mask, num_layers,
                               component="mlp"):
    """Capture MLP, attention, or mid-residual outputs from all layers."""
    outputs_per_layer = {}
    handles = []

    for layer_idx in range(num_layers):
        if component == "mlp":
            module = model.model.layers[layer_idx].mlp
        elif component == "attn":
            module = model.model.layers[layer_idx].self_attn
        elif component == "mid":
            module = model.model.layers[layer_idx].post_attention_layernorm
        else:
            raise ValueError(f"Unknown component: {component}")

        if component == "mid":
            # pre-hook captures input to post_attention_layernorm = residual + attn_out
            def make_pre_hook(idx):
                def hook(mod, inp):
                    outputs_per_layer[idx] = inp[0].detach().clone()
                return hook
            handles.append(module.register_forward_pre_hook(make_pre_hook(layer_idx)))
        else:
            def make_hook(idx):
                def hook(mod, inp, out):
                    if isinstance(out, tuple):
                        outputs_per_layer[idx] = out[0].detach().clone()
                    else:
                        outputs_per_layer[idx] = out.detach().clone()
                return hook
            handles.append(module.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

    for h in handles:
        h.remove()

    return outputs_per_layer


def _patch_component_and_eval(target_model, input_ids, attention_mask, meta,
                              source_output, layer, component="mlp",
                              patch_scope="span"):
    """Patch a specific component at a specific layer and evaluate log-prob."""
    device = next(target_model.parameters()).device

    if component == "mlp":
        module = target_model.model.layers[layer].mlp
    elif component == "attn":
        module = target_model.model.layers[layer].self_attn
    elif component == "mid":
        module = target_model.model.layers[layer].post_attention_layernorm
    else:
        raise ValueError(f"Unknown component: {component}")

    def _apply_span_patch(hs):
        for b in range(hs.size(0)):
            length = meta["lengths"][b]
            start = meta["start"][b]
            ps = meta["patch_start"][b]
            pe = meta["patch_end"][b]
            if start >= length:
                continue
            if patch_scope == "boundary":
                hs[b, start, :] = source_output[b, start, :].to(hs.dtype)
            else:
                ps = min(ps, length)
                pe = min(pe, length)
                if ps < pe:
                    hs[b, ps:pe, :] = source_output[b, ps:pe, :].to(hs.dtype)
        return hs

    if component == "mid":
        # pre-hook: replace input to post_attention_layernorm
        def patch_pre_hook(mod, inp):
            hs = inp[0].clone()
            hs = _apply_span_patch(hs)
            return (hs,) + inp[1:]
        handle = module.register_forward_pre_hook(patch_pre_hook)
    else:
        def patch_hook(mod, inp, out):
            if isinstance(out, tuple):
                hs = out[0].clone()
            else:
                hs = out.clone()
            hs = _apply_span_patch(hs)
            if isinstance(out, tuple):
                return (hs,) + out[1:]
            return hs
        handle = module.register_forward_hook(patch_hook)
    with torch.no_grad():
        outputs = target_model(input_ids, attention_mask=attention_mask)
    handle.remove()

    logits = outputs.logits
    results = []
    for i in range(input_ids.size(0)):
        es = meta["eval_start"][i]
        ee = meta["eval_end"][i]
        if ee <= es:
            results.append(float("-inf"))
            continue
        labels = torch.tensor([meta["eval_ref_ids"][i]], device=device)
        token_logprobs = _gather_token_logprobs(logits[i:i+1, es:ee, :], labels)
        results.append(token_logprobs.mean().item())
    return results


# ============================================================================
# Prepare examples (same as meta_eval_faithfulness)
# ============================================================================

def prepare_all_examples(tokenizer, prefix_data, patch_scope="span",
                         em_scope="entity", entity_source="gt"):
    """Pre-tokenize all examples."""
    prepared = []
    for item in prefix_data:
        question = item["question"]
        prefix = item["prefix"]
        entity = item["entity"]
        gt_entity = item.get("gt_entity", entity)
        idx = item["idx"]

        prompt = f"Question: {question}\nAnswer: {prefix}"
        answer_text = item["answer"]
        if answer_text.startswith(prefix):
            answer_text = answer_text[len(prefix):]
        reference_text = normalize_reference_for_eval(prompt, answer_text)

        entity_text = gt_entity if entity_source == "gt" else item.get("full_entity", entity)
        eval_span = get_eval_span(tokenizer, reference_text, entity_text, em_scope)
        if em_scope == "entity" and eval_span is None:
            prepared.append(None)
            continue

        ctx = build_logprob_ctx(tokenizer, prompt, reference_text, eval_span, patch_scope)
        if ctx is None:
            prepared.append(None)
            continue

        prepared.append({"ctx": ctx, "idx": idx, "eval_span": eval_span})
    return prepared


# ============================================================================
# S1 computation for a component
# ============================================================================

def compute_s1_for_component(full_model, retain_model, tokenizer, prepared,
                             layer_list, delta_threshold, patch_scope,
                             batch_size, component="mlp",
                             prefix_data=None, log_path=None):
    """Compute S1 (retain→full) patching for MLP or attention."""
    device = next(full_model.parameters()).device
    num_layers = max(layer_list) + 1
    valid = [(i, p) for i, p in enumerate(prepared) if p is not None]

    # Build prefix_data lookup for logging
    pd_lookup = {}
    if prefix_data:
        for item in prefix_data:
            pd_lookup[item["idx"]] = item

    log_file = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w")
        comp_upper = component.upper()
        log_file.write(f"S1 {comp_upper} Patching with Teacher Forcing\n")
        log_file.write(f"Metric: log-prob (reference tokens), reference = gt\n")
        log_file.write(f"Delta threshold: {delta_threshold}\n")
        log_file.write(f"Patch scope: {patch_scope}\n")
        log_file.write(f"S1: Retain → Full ({comp_upper} patching)\n")
        log_file.write(f"Layers: {layer_list}\n")
        log_file.write(f"Examples: {len(valid)}\n\n")

    entries = {}
    example_num = 0
    for batch_start in tqdm(range(0, len(valid), batch_size),
                            desc=f"S1 {component}"):
        batch = valid[batch_start:batch_start + batch_size]
        batch_ctxs = [p["ctx"] for _, p in batch]
        batch_indices = [p["idx"] for _, p in batch]

        input_ids, attention_mask, meta = _prepare_batch_inputs(
            batch_ctxs, tokenizer, device
        )

        # Full baseline scores (no patching)
        full_scores = compute_logprob_teacher_forcing_baseline_batch_with_inputs(
            full_model, input_ids, attention_mask, meta
        )

        # Capture retain component outputs (all layers, one forward pass)
        retain_outputs = _capture_component_outputs(
            retain_model, input_ids, attention_mask, num_layers, component
        )

        # Per-layer patching
        s1_scores_all = []
        for layer in layer_list:
            scores = _patch_component_and_eval(
                full_model, input_ids, attention_mask, meta,
                retain_outputs[layer], layer, component, patch_scope
            )
            s1_scores_all.append(scores)

        del retain_outputs
        torch.cuda.empty_cache()

        # Build entries and write per-example logs
        for j, (orig_i, p) in enumerate(batch):
            idx = batch_indices[j]
            fs = full_scores[j]
            s1_scores = [s1_scores_all[li][j] for li in range(len(layer_list))]
            s1_deltas = [fs - s for s in s1_scores]
            s1_status = ["LOST" if d > delta_threshold else "KEPT" for d in s1_deltas]

            entries[idx] = {
                "idx": idx,
                "full_score": fs,
                "s1_scores": s1_scores,
                "s1_deltas": s1_deltas,
                "s1_status": s1_status,
            }

            # Per-example log
            if log_file:
                example_num += 1
                item = pd_lookup.get(idx, {})
                q = item.get("question", "?")
                prefix = item.get("prefix", "?")
                entity = item.get("gt_entity", item.get("entity", "?"))
                ref_text = item.get("answer", "?")

                log_file.write("=" * 80 + "\n")
                log_file.write(f"[{example_num}/{len(valid)}] Example {idx}\n")
                log_file.write(f"  Q: {q}\n")
                log_file.write(f"  Prefix: '{prefix}'\n")
                log_file.write(f"  GT (entity): '{entity}'\n")
                log_file.write(f"  Full log-prob (ref span): {fs:.3f}\n")
                log_file.write("=" * 80 + "\n")

                comp_label = component.upper()
                log_file.write(f"  Layer  | S1 (Retain→Full, {comp_label})\n")
                log_file.write(f"  ------ | -------------------------\n")
                for li, layer in enumerate(layer_list):
                    sc = s1_scores[li]
                    dt = s1_deltas[li]
                    st = s1_status[li]
                    log_file.write(
                        f"  L{layer:02d}   | logp={sc:.3f} Δ={dt:.3f} [{st}]\n"
                    )
                log_file.write(f"  ------ | -------------------------\n")

                ft_layers = [layer_list[li] for li in range(len(layer_list))
                             if s1_deltas[li] > delta_threshold]
                ft_count = len(ft_layers)
                log_file.write(f"  FT layers ({comp_label} LOST): {ft_layers}\n\n")

    if log_file:
        # Summary at end
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("SUMMARY\n")
        log_file.write("=" * 80 + "\n\n")
        all_ft = []
        all_avg_delta = []
        for idx, entry in sorted(entries.items()):
            ft_count = sum(1 for d in entry["s1_deltas"] if d > delta_threshold)
            all_ft.append(ft_count)
            all_avg_delta.append(np.mean(entry["s1_deltas"]))
        log_file.write(f"  avg delta (across all layers):  {np.mean(all_avg_delta):.4f}\n")
        log_file.write(f"  avg FT layers per example:      {np.mean(all_ft):.1f} / {len(layer_list)}\n")
        log_file.write(f"  max FT layers per example:      {max(all_ft)}\n")
        log_file.write(f"  min FT layers per example:      {min(all_ft)}\n")

        # Layer-wise average delta table
        log_file.write(f"\nLayer-wise average delta:\n")
        for li, layer in enumerate(layer_list):
            deltas = [entries[idx]["s1_deltas"][li] for idx in entries]
            log_file.write(f"  L{layer:02d}: {np.mean(deltas):.4f}\n")

        log_file.close()

    return entries


# ============================================================================
# Logging & comparison
# ============================================================================

def write_comparison_log(log_path, layer_list, caches, delta_threshold):
    """Write a comparison log for residual vs MLP vs attention."""
    components = list(caches.keys())
    all_indices = sorted(set().union(*(c.keys() for c in caches.values())))

    with open(log_path, "w") as f:
        f.write(f"S1 Component Patching Comparison\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Delta threshold: {delta_threshold}\n")
        f.write(f"Components: {', '.join(components)}\n")
        f.write(f"Examples: {len(all_indices)}\n")
        f.write(f"Layers: {len(layer_list)}\n\n")

        # ---- Layer-wise average deltas ----
        f.write("=" * 80 + "\n")
        f.write("LAYER-WISE AVERAGE DELTA (full_score - patched_score)\n")
        f.write("Higher delta = more knowledge disruption from patching\n")
        f.write("=" * 80 + "\n\n")

        header = f"{'Layer':>6}"
        for comp in components:
            header += f"  {comp:>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for li, layer in enumerate(layer_list):
            line = f"{layer:>6}"
            for comp in components:
                deltas = [caches[comp][idx]["s1_deltas"][li]
                          for idx in all_indices if idx in caches[comp]]
                avg = np.mean(deltas) if deltas else float("nan")
                line += f"  {avg:>10.4f}"
            f.write(line + "\n")

        # ---- FT layer counts ----
        f.write(f"\n{'=' * 80}\n")
        f.write(f"FT LAYER COUNTS (delta > {delta_threshold})\n")
        f.write(f"{'=' * 80}\n\n")

        header = f"{'Layer':>6}"
        for comp in components:
            header += f"  {comp:>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for li, layer in enumerate(layer_list):
            line = f"{layer:>6}"
            for comp in components:
                count = sum(1 for idx in all_indices
                            if idx in caches[comp] and
                            caches[comp][idx]["s1_deltas"][li] > delta_threshold)
                pct = count / len(all_indices) * 100 if all_indices else 0
                line += f"  {count:>4}/{len(all_indices):>3} ({pct:4.1f}%)"
            f.write(line + "\n")

        # ---- Summary stats ----
        f.write(f"\n{'=' * 80}\n")
        f.write("SUMMARY\n")
        f.write(f"{'=' * 80}\n\n")

        for comp in components:
            cache = caches[comp]
            total_deltas = []
            ft_counts = []
            for idx in all_indices:
                if idx not in cache:
                    continue
                entry = cache[idx]
                total_deltas.append(np.mean(entry["s1_deltas"]))
                ft_count = sum(1 for d in entry["s1_deltas"] if d > delta_threshold)
                ft_counts.append(ft_count)

            f.write(f"{comp}:\n")
            f.write(f"  avg delta (across all layers):  {np.mean(total_deltas):.4f}\n")
            f.write(f"  avg FT layers per example:      {np.mean(ft_counts):.1f} / {len(layer_list)}\n")
            f.write(f"  max FT layers per example:      {max(ft_counts)}\n")
            f.write(f"  min FT layers per example:      {min(ft_counts)}\n\n")

    print(f"Comparison log: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="S1 component patching: MLP and attention")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--delta_threshold", type=float, default=0.05)
    parser.add_argument("--patch_scope", type=str, default="span")
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        choices=["eager", "sdpa"])
    parser.add_argument("--components", nargs="+", default=["mlp", "attn"],
                        choices=["mlp", "attn", "mid"])
    parser.add_argument("--out_dir", type=str,
                        default="runs/meta_eval")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device_map = "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attn = args.attn_implementation

    print(f"GPU: {args.gpu}")
    print(f"Attention: {attn}")
    print(f"Components: {args.components}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Load models
    print("Loading full model...")
    full_model = load_model(
        TOFU_FULL_MODEL, dtype="bfloat16", device_map=device_map,
        attn_implementation=attn
    )
    print("Loading retain model...")
    retain_model = load_model(
        TOFU_RETAIN_MODEL, dtype="bfloat16", device_map=device_map,
        attn_implementation=attn
    )
    tokenizer = load_tokenizer(TOFU_FULL_MODEL)
    num_layers = get_num_layers(full_model)
    layer_list = list(range(num_layers))
    print(f"Layers: {num_layers}")

    # Prepare examples
    print("Preparing examples...")
    prefix_data = load_prefix_data(PREFIX_DATA_PATH)
    prepared = prepare_all_examples(
        tokenizer, prefix_data,
        patch_scope=args.patch_scope,
        em_scope="entity", entity_source="gt"
    )
    valid_count = sum(1 for p in prepared if p is not None)
    print(f"Examples: {valid_count}/{len(prepared)}")

    # Run each component
    caches = {}
    for component in args.components:
        print(f"\n{'=' * 60}")
        print(f"Computing S1 for {component.upper()}")
        print(f"{'=' * 60}")

        detail_log = out_dir / f"s1_{component}_{attn}.log"

        t0 = time.time()
        entries = compute_s1_for_component(
            full_model, retain_model, tokenizer, prepared,
            layer_list, args.delta_threshold, args.patch_scope,
            args.batch_size, component=component,
            prefix_data=prefix_data, log_path=str(detail_log)
        )
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"Detail log: {detail_log}")
        caches[component] = entries

    # Load existing residual cache for comparison
    residual_path = out_dir / f"s1_cache_{attn}.json"
    if residual_path.exists():
        print(f"\nLoading residual cache: {residual_path}")
        residual_data = json.loads(residual_path.read_text())
        # Handle both raw dict and {config, entries} format
        if "entries" in residual_data:
            residual_entries = residual_data["entries"]
        else:
            residual_entries = residual_data
        # Convert string keys to int
        caches["residual"] = {int(k): v for k, v in residual_entries.items()}
    else:
        print(f"\nNo residual cache found at {residual_path}")

    # Write comparison log
    comparison_log = out_dir / f"s1_component_comparison_{attn}.log"
    write_comparison_log(comparison_log, layer_list, caches, args.delta_threshold)

    print("\nDone!")


if __name__ == "__main__":
    main()

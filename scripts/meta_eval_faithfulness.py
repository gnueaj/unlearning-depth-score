#!/usr/bin/env python3
"""
Meta-Evaluation: Faithfulness of UDS metric.

Evaluates whether UDS can distinguish between:
  - P pool: models trained WITH forget10 knowledge (should have LOW UDS / HIGH 1-UDS)
  - N pool: models trained WITHOUT forget10 knowledge (should have HIGH UDS / LOW 1-UDS)

Metric: AUC-ROC(1-UDS_P, 1-UDS_N)

Design constraints:
  - Disk: ~80GB free → stream models one-at-a-time, delete cache after each
  - GPU: 48GB → batch aggressively (full+retain+source fit in ~7.2GB bf16)
  - S1 cache: retain→full results are shared across all 60 models
  - Speed: precompute all source hidden states in 1 forward pass per model

Reference: Open-Unlearning (NeurIPS 2025), Section 4.1, Appendix E.1
"""

import os
import sys
import json
import argparse
import shutil
import time
import gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchscope.models import load_model, load_tokenizer, get_num_layers
from patchscope.core import generate_baseline
from patchscope.utils import set_seed, safe_mkdir
from patchscope.meta_eval_utils import (
    MEM_METRICS,
    GENERATION_METRICS,
    MIA_METRICS,
    normalize_metrics_list,
    load_forget10_perturbed,
    compute_mem_metrics,
    compute_generation_metrics,
    prepare_mia_data,
    compute_mia_metrics,
)

from exp_s1_teacher_forcing import (
    load_prefix_data,
    get_eval_span,
    normalize_reference_for_eval,
    build_logprob_ctx,
    _prepare_batch_inputs,
    _compute_hidden_states_batch,
    _gather_token_logprobs,
    compute_logprob_teacher_forcing_baseline_batch_with_inputs,
    compute_logprob_teacher_forcing_layer_batch_with_inputs,
)

# ============================================================================
# Model pools
# ============================================================================

TOFU_FULL_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
TOFU_RETAIN_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
PREFIX_DATA_PATH = "tofu_data/forget10_filtered_v7_gt.json"

# P pool: trained WITH forget10 knowledge (3 variants × 5 LR × 2 epochs = 30)
_P_BASE = "open-unlearning/pos_tofu_Llama-3.2-1B-Instruct"
P_POOL = []
for lr in ["1e-05", "2e-05", "3e-05", "4e-05", "5e-05"]:
    for ep in [5, 10]:
        # variant 1: full (retain90 + forget10 original)
        P_POOL.append(f"{_P_BASE}_full_lr{lr}_wd0.01_epoch{ep}")
        # variant 2: biography
        P_POOL.append(f"{_P_BASE}_retain90_forget10_bio_lr{lr}_wd0.01_epoch{ep}")
        # variant 3: paraphrased
        P_POOL.append(f"{_P_BASE}_retain90_forget10_para_lr{lr}_wd0.01_epoch{ep}")

# N pool: trained WITHOUT forget10 knowledge (3 variants × 5 LR × 2 epochs = 30)
_N_BASE = "open-unlearning/neg_tofu_Llama-3.2-1B-Instruct"
N_POOL = []
for lr in ["1e-05", "2e-05", "3e-05", "4e-05", "5e-05"]:
    for ep in [5, 10]:
        # variant 1: retain90 only
        N_POOL.append(f"{_N_BASE}_retain90_lr{lr}_wd0.01_epoch{ep}")
        # variant 2: perturbed labels
        N_POOL.append(f"{_N_BASE}_retain90_forget10_pert_lr{lr}_wd0.01_epoch{ep}")
        # variant 3: celebrity bios
        N_POOL.append(f"{_N_BASE}_retain90_celeb10_bio_lr{lr}_wd0.01_epoch{ep}")

assert len(P_POOL) == 30, f"P pool has {len(P_POOL)} models, expected 30"
assert len(N_POOL) == 30, f"N pool has {len(N_POOL)} models, expected 30"


# ============================================================================
# Core UDS computation (streamlined for meta-eval)
# ============================================================================

def prepare_all_examples(tokenizer, prefix_data, patch_scope="span",
                         reference_scope="continuation", em_scope="entity",
                         entity_source="gt"):
    """Pre-tokenize all examples once. Returns list of (ctx, meta) or None for skipped."""
    prepared = []
    for item in prefix_data:
        question = item["question"]
        prefix = item["prefix"]
        entity = item["entity"]
        gt_entity = item.get("gt_entity", entity)
        idx = item["idx"]

        if reference_scope == "full_answer":
            prompt = f"Question: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer: {prefix}"

        # GT reference
        answer_text = item["answer"]
        if reference_scope == "continuation":
            if answer_text.startswith(prefix):
                answer_text = answer_text[len(prefix):]
        reference_text = normalize_reference_for_eval(prompt, answer_text)

        if entity_source == "gt":
            entity_text = gt_entity
        else:
            entity_text = item.get("full_entity", entity)

        eval_span = get_eval_span(tokenizer, reference_text, entity_text, em_scope)
        if em_scope == "entity" and eval_span is None:
            prepared.append(None)
            continue

        ctx = build_logprob_ctx(tokenizer, prompt, reference_text, eval_span, patch_scope)
        if ctx is None:
            prepared.append(None)
            continue

        prepared.append({
            "ctx": ctx,
            "idx": idx,
            "eval_span": eval_span,
        })
    return prepared


def compute_s1_cache(full_model, retain_model, tokenizer, prepared,
                     layer_list, delta_threshold, patch_scope, batch_size):
    """Compute S1 (retain→full) results for all examples. Returns list of cache entries."""
    device = next(full_model.parameters()).device
    valid = [(i, p) for i, p in enumerate(prepared) if p is not None]

    entries = {}
    for batch_start in tqdm(range(0, len(valid), batch_size), desc="S1 cache"):
        batch = valid[batch_start:batch_start + batch_size]
        batch_ctxs = [p["ctx"] for _, p in batch]
        batch_indices = [p["idx"] for _, p in batch]

        input_ids, attention_mask, meta = _prepare_batch_inputs(batch_ctxs, tokenizer, device)

        # Full baseline scores
        full_scores = compute_logprob_teacher_forcing_baseline_batch_with_inputs(
            full_model, input_ids, attention_mask, meta
        )

        # Retain hidden states (all layers, one forward pass)
        retain_hidden = _compute_hidden_states_batch(retain_model, input_ids, attention_mask)

        # Per-layer S1 patching
        s1_scores_all = []  # [layer_idx][batch_idx]
        for layer in layer_list:
            scores = compute_logprob_teacher_forcing_layer_batch_with_inputs(
                full_model, input_ids, attention_mask, meta,
                retain_hidden[layer + 1], layer, patch_scope=patch_scope
            )
            s1_scores_all.append(scores)

        del retain_hidden
        torch.cuda.empty_cache()

        # Build cache entries
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

    return entries


def compute_uds_for_model(source_model, full_model, tokenizer, prepared,
                          s1_cache, layer_list, delta_threshold, patch_scope,
                          batch_size):
    """Compute per-example UDS for a source model using cached S1 results."""
    device = next(full_model.parameters()).device
    valid = [(i, p) for i, p in enumerate(prepared) if p is not None]

    uds_values = []

    for batch_start in range(0, len(valid), batch_size):
        batch = valid[batch_start:batch_start + batch_size]
        batch_ctxs = [p["ctx"] for _, p in batch]
        batch_indices = [p["idx"] for _, p in batch]

        input_ids, attention_mask, meta = _prepare_batch_inputs(batch_ctxs, tokenizer, device)

        # Source hidden states (all layers, one forward pass)
        source_hidden = _compute_hidden_states_batch(source_model, input_ids, attention_mask)

        # Per-layer S2 patching
        s2_scores_all = []
        for layer in layer_list:
            scores = compute_logprob_teacher_forcing_layer_batch_with_inputs(
                full_model, input_ids, attention_mask, meta,
                source_hidden[layer + 1], layer, patch_scope=patch_scope
            )
            s2_scores_all.append(scores)

        del source_hidden
        torch.cuda.empty_cache()

        # Compute UDS per example
        for j, (orig_i, p) in enumerate(batch):
            idx = batch_indices[j]
            entry = s1_cache[idx]
            full_score = entry["full_score"]

            s1_deltas = entry["s1_deltas"]
            s1_status = entry["s1_status"]

            ft_layer_indices = [li for li, st in enumerate(s1_status) if st == "LOST"]
            if not ft_layer_indices:
                # No FT signal → UDS=None (skip for avg)
                continue

            denom = 0.0
            numer = 0.0
            for li in ft_layer_indices:
                d1 = s1_deltas[li]
                if d1 <= delta_threshold:
                    continue
                s2_score = s2_scores_all[li][j]
                d2 = full_score - s2_score
                ratio = d2 / d1
                ratio = max(0.0, min(ratio, 1.0))
                denom += d1
                numer += d1 * ratio

            if denom > 0:
                uds_values.append(numer / denom)

    avg_uds = sum(uds_values) / len(uds_values) if uds_values else 0.0
    return avg_uds, uds_values


def delete_model_cache(model_id):
    """Delete HuggingFace cache for a specific model to free disk space."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # HF cache dir name: models--{org}--{model} (with / replaced by --)
    dir_name = "models--" + model_id.replace("/", "--")
    model_cache = cache_dir / dir_name
    try:
        if model_cache.exists():
            size_mb = sum(f.stat().st_size for f in model_cache.rglob("*") if f.is_file()) / 1e6
            shutil.rmtree(model_cache, ignore_errors=True)
            print(f"  Deleted cache: {dir_name} ({size_mb:.0f} MB)")
            return size_mb
    except Exception as e:
        print(f"  Cache delete skipped (may be deleted by another process): {e}")
    return 0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Meta-eval: UDS Faithfulness (AUC-ROC)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for UDS computation")
    parser.add_argument("--delta_threshold", type=float, default=0.05)
    parser.add_argument("--patch_scope", type=str, default="span")
    parser.add_argument("--em_scope", type=str, default="entity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--s1_cache_path", type=str, default="runs/meta_eval/s1_cache.json")
    parser.add_argument("--keep_cache", action="store_true",
                        help="Don't delete model caches after evaluation (uses ~2.4GB/model)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just print model list, don't run")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to partial results JSON to resume from")
    parser.add_argument("--model_start", type=int, default=None,
                        help="Start index for model range (inclusive)")
    parser.add_argument("--model_end", type=int, default=None,
                        help="End index for model range (exclusive)")
    parser.add_argument(
        "--metrics",
        type=str,
        default="uds",
        help="Comma-separated metrics: uds, em, es, prob, paraprob, truth_ratio, "
             "rouge, para_rouge, jailbreak_rouge, mia_loss, mia_zlib, mia_min_k, "
             "mia_min_kpp. Use 'all' for all 13 or 'table2' for 12 (no UDS).",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--date_string", type=str, default="10 Apr 2025")
    parser.add_argument("--mia_k", type=float, default=0.4)
    parser.add_argument("--attn_implementation", type=str, default=None,
                        help="Attention implementation: eager, sdpa, or flash_attention_2")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    if args.out_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M%S")
        args.out_dir = f"runs/meta_eval/{ts}_faithfulness"
    safe_mkdir(args.out_dir)

    layer_list = list(range(16))  # 0-15
    metrics = normalize_metrics_list(args.metrics.split(","))
    if not metrics:
        metrics = ["uds"]

    print("=" * 70)
    print("Meta-Evaluation: UDS Faithfulness")
    print("=" * 70)
    print(f"P pool: {len(P_POOL)} models (WITH forget knowledge)")
    print(f"N pool: {len(N_POOL)} models (WITHOUT forget knowledge)")
    print(f"Batch size: {args.batch_size}")
    print(f"Delta threshold: {args.delta_threshold}")
    print(f"Output: {args.out_dir}")
    print(f"Metrics: {', '.join(metrics)}")
    print()

    if args.dry_run:
        print("P pool models:")
        for m in P_POOL:
            print(f"  {m}")
        print(f"\nN pool models:")
        for m in N_POOL:
            print(f"  {m}")
        return

    # ------------------------------------------------------------------
    # Load base models (full + retain stay in memory throughout)
    # ------------------------------------------------------------------
    print("Loading tokenizer + full + retain models...")
    attn_impl = args.attn_implementation
    tokenizer = load_tokenizer(TOFU_FULL_MODEL)
    full_model = load_model(TOFU_FULL_MODEL, dtype="bfloat16", device_map="cuda", attn_implementation=attn_impl)
    retain_model = load_model(TOFU_RETAIN_MODEL, dtype="bfloat16", device_map="cuda", attn_implementation=attn_impl)
    if attn_impl:
        print(f"  Full + Retain loaded on GPU (attn: {attn_impl})")
    else:
        print(f"  Full + Retain loaded on GPU")

    # ------------------------------------------------------------------
    # Load & prepare datasets (once)
    # ------------------------------------------------------------------
    prefix_data = load_prefix_data(PREFIX_DATA_PATH)
    print(f"Dataset: {len(prefix_data)} examples")

    prepared = prepare_all_examples(
        tokenizer, prefix_data,
        patch_scope=args.patch_scope,
        em_scope=args.em_scope,
    )
    n_valid = sum(1 for p in prepared if p is not None)
    n_skip = len(prepared) - n_valid
    print(f"  Valid: {n_valid}, Skipped: {n_skip}")

    # Meta-eval metric datasets
    mem_data = None
    mia_data = None
    if any(m in MEM_METRICS or m in GENERATION_METRICS for m in metrics):
        mem_data = load_forget10_perturbed()
    if any(m in MIA_METRICS for m in metrics):
        mia_data = prepare_mia_data(
            tokenizer,
            max_length=args.max_length,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt or None,
            date_string=args.date_string or None,
        )

    # ------------------------------------------------------------------
    # S1 cache (retain → full, shared for all 60 models)
    # ------------------------------------------------------------------
    s1_cache = None
    if "uds" in metrics:
        s1_cache_path = Path(args.s1_cache_path)
        if s1_cache_path.exists():
            print(f"Loading S1 cache from {s1_cache_path}...")
            s1_cache = json.loads(s1_cache_path.read_text())
            # Convert string keys back to int
            s1_cache = {int(k): v for k, v in s1_cache.items()}
            print(f"  Loaded {len(s1_cache)} entries")
        else:
            print("Computing S1 cache (retain → full)...")
            s1_cache = compute_s1_cache(
                full_model, retain_model, tokenizer, prepared,
                layer_list, args.delta_threshold, args.patch_scope, args.batch_size
            )
            s1_cache_path.parent.mkdir(parents=True, exist_ok=True)
            s1_cache_path.write_text(json.dumps({str(k): v for k, v in s1_cache.items()}))
            print(f"  Saved S1 cache: {s1_cache_path} ({len(s1_cache)} entries)")

        # Unload retain model to free GPU memory for source models
        del retain_model
        gc.collect()
        torch.cuda.empty_cache()
        print("Retain model unloaded (S1 cached)")
    else:
        del retain_model
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------
    completed = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            completed = json.loads(resume_path.read_text())
            print(f"Resuming: {len(completed)} models already evaluated")

    # ------------------------------------------------------------------
    # Evaluate each pool model one-at-a-time
    # ------------------------------------------------------------------
    all_models = [(m, "P") for m in P_POOL] + [(m, "N") for m in N_POOL]

    # Optional model range slicing for GPU parallelism
    if args.model_start is not None or args.model_end is not None:
        s = args.model_start or 0
        e = args.model_end or len(all_models)
        print(f"Model range: [{s}, {e}) of {len(all_models)} total")
        all_models = all_models[s:e]

    results_path = Path(args.out_dir) / "results.json"
    results = dict(completed)  # {model_id: {"metrics": {...}, "pool": str, ...}}

    for mi, (model_id, pool) in enumerate(all_models):
        if model_id in results:
            print(f"[{mi+1}/{len(all_models)}] SKIP (cached) {pool} {model_id}")
            continue

        print(f"\n[{mi+1}/{len(all_models)}] {pool} pool: {model_id}")
        t0 = time.time()

        # Load source model
        try:
            source_model = load_model(model_id, dtype="bfloat16", device_map="cuda", attn_implementation=attn_impl)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            results[model_id] = {"uds": None, "pool": pool, "error": str(e)}
            continue

        metric_scores = {}
        uds_list = []
        if "uds" in metrics:
            avg_uds, uds_list = compute_uds_for_model(
                source_model, full_model, tokenizer, prepared,
                s1_cache, layer_list, args.delta_threshold, args.patch_scope,
                args.batch_size
            )
            metric_scores["uds"] = avg_uds

        if any(m in MEM_METRICS for m in metrics):
            mem_metrics_needed = MEM_METRICS & set(metrics)
            mem_scores = compute_mem_metrics(
                source_model, tokenizer, mem_data,
                batch_size=args.batch_size,
                max_length=args.max_length,
                use_chat_template=args.use_chat_template,
                system_prompt=args.system_prompt or None,
                date_string=args.date_string or None,
                metrics_filter=mem_metrics_needed,  # fast mode: skip unnecessary evals
            )
            for k, v in mem_scores.items():
                if k in metrics:
                    metric_scores[k] = v

        gen_metrics_needed = GENERATION_METRICS & set(metrics)
        if gen_metrics_needed:
            gen_scores = compute_generation_metrics(
                source_model, tokenizer, mem_data,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                use_chat_template=args.use_chat_template,
                system_prompt=args.system_prompt or None,
                date_string=args.date_string or None,
                metrics_to_compute=gen_metrics_needed,
            )
            for k, v in gen_scores.items():
                if k in metrics:
                    metric_scores[k] = v

        if any(m in MIA_METRICS for m in metrics):
            mia_scores = compute_mia_metrics(
                source_model, tokenizer, mia_data,
                batch_size=args.batch_size,
                k=args.mia_k,
            )
            for k, v in mia_scores.items():
                if k in metrics:
                    metric_scores[k] = v

        elapsed = time.time() - t0
        if "uds" in metrics:
            print(f"  UDS = {metric_scores['uds']:.4f}  (1-UDS = {1-metric_scores['uds']:.4f})  "
                  f"n={len(uds_list)}  {elapsed:.1f}s")
        else:
            print(f"  Done  {elapsed:.1f}s")

        results[model_id] = {
            "metrics": metric_scores,
            "pool": pool,
            "n_examples": len(uds_list),
            "elapsed_sec": elapsed,
        }

        # Unload source model
        del source_model
        gc.collect()
        torch.cuda.empty_cache()

        # Delete model cache to save disk
        if not args.keep_cache:
            delete_model_cache(model_id)

        # Save intermediate results after each model
        results_path.write_text(json.dumps(results, indent=2))

    # ------------------------------------------------------------------
    # Compute AUC-ROC
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FAITHFULNESS RESULTS")
    print("=" * 70)

    metric_auc = {}
    for metric in metrics:
        p_scores = []
        n_scores = []
        for model_id, info in results.items():
            score = info.get("metrics", {}).get(metric)
            if score is None:
                continue
            if metric == "uds":
                score = 1 - score  # higher = more knowledge
            if info["pool"] == "P":
                p_scores.append(score)
            else:
                n_scores.append(score)

        if len(p_scores) >= 2 and len(n_scores) >= 2:
            labels = [1] * len(p_scores) + [0] * len(n_scores)
            scores = p_scores + n_scores
            auc = roc_auc_score(labels, scores)
        else:
            auc = None

        metric_auc[metric] = {
            "auc": auc,
            "p_count": len(p_scores),
            "n_count": len(n_scores),
            "p_mean": float(np.mean(p_scores)) if p_scores else None,
            "n_mean": float(np.mean(n_scores)) if n_scores else None,
        }
        print(f">>> Faithfulness[{metric}] = {auc:.4f}" if auc is not None else f">>> Faithfulness[{metric}] = N/A")

    # ------------------------------------------------------------------
    # Save final results
    # ------------------------------------------------------------------
    summary = {
        "faithfulness": metric_auc,
        "delta_threshold": args.delta_threshold,
        "patch_scope": args.patch_scope,
        "em_scope": args.em_scope,
        "batch_size": args.batch_size,
        "n_valid_examples": n_valid,
        "metrics": metrics,
    }

    summary_path = Path(args.out_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\nResults saved to: {args.out_dir}/")
    print(f"  summary.json: AUC-ROC + pool stats")
    print(f"  results.json: per-model UDS scores")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quick test: retain model relearning UDS at different batch sizes."""

import os, sys, json, time, shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from scripts.meta_eval_faithfulness import (
    TOFU_FULL_MODEL,
    TOFU_RETAIN_MODEL,
    PREFIX_DATA_PATH,
    prepare_all_examples,
    compute_uds_for_model,
)
from scripts.meta_eval_robustness import (
    load_model,
    finetune_in_subprocess,
    free_memory,
)
from exp_s1_teacher_forcing import load_prefix_data

GPU = int(sys.argv[1]) if len(sys.argv) > 1 else 0
BATCH_SIZES = [8, 16, 32]
GRAD_ACCUM = 4
S1_CACHE_PATH = "runs/meta_eval/s1_cache_sdpa.json"
ATTN_IMPL = "sdpa"

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

print(f"GPU: {GPU}, attn: {ATTN_IMPL}")
print(f"S1 cache: {S1_CACHE_PATH}")
print(f"Batch sizes to test: {BATCH_SIZES}")
print(f"Grad accum: {GRAD_ACCUM} (effective: {[b*GRAD_ACCUM for b in BATCH_SIZES]})")
print("=" * 60)

# Load full model
print("Loading full model...")
full_model = load_model(TOFU_FULL_MODEL, dtype="bfloat16", device_map="cuda",
                        attn_implementation=ATTN_IMPL)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOFU_FULL_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare UDS data
prefix_data = load_prefix_data(PREFIX_DATA_PATH)
prepared_uds = prepare_all_examples(tokenizer, prefix_data, patch_scope="span")
layer_list = list(range(full_model.config.num_hidden_layers))

# Load S1 cache
print(f"Loading S1 cache from {S1_CACHE_PATH}...")
s1_cache = json.loads(Path(S1_CACHE_PATH).read_text())
s1_cache = {int(k): v for k, v in s1_cache.items()}
print(f"S1 cache: {len(s1_cache)} entries")

# Compute retain UDS (before relearning)
print("\n--- Retain model (before relearn) ---")
retain_model = load_model(TOFU_RETAIN_MODEL, dtype="bfloat16", device_map="cuda",
                          attn_implementation=ATTN_IMPL)
uds_before, _ = compute_uds_for_model(
    retain_model, full_model, tokenizer, prepared_uds,
    s1_cache, layer_list, 0.05, "span", 32,
)
print(f"Retain UDS (before): {uds_before:.4f}")
del retain_model
free_memory()

# Test each batch size
results = {"before": uds_before}

for bs in BATCH_SIZES:
    effective = bs * GRAD_ACCUM
    print(f"\n{'='*60}")
    print(f"Batch size: {bs} (effective: {effective})")
    print(f"{'='*60}")

    ft_dir = f"/tmp/retain_relearn_bs{bs}"
    if os.path.exists(ft_dir):
        shutil.rmtree(ft_dir)

    t0 = time.time()
    print("  Finetuning...")
    ok = finetune_in_subprocess(
        TOFU_RETAIN_MODEL, ft_dir, gpu=GPU,
        lr=2e-5, epochs=1, batch_size=bs, grad_accum=GRAD_ACCUM,
        attn_implementation=ATTN_IMPL,
    )
    ft_time = time.time() - t0

    if not ok:
        print(f"  FAILED! Skipping bs={bs}")
        results[f"bs{bs}"] = "FAILED"
        continue

    print(f"  Finetune done in {ft_time:.1f}s")
    print("  Loading finetuned model...")
    ft_model = load_model(ft_dir, dtype="bfloat16", device_map="cuda",
                          attn_implementation=ATTN_IMPL)

    print("  Computing UDS...")
    uds_after, _ = compute_uds_for_model(
        ft_model, full_model, tokenizer, prepared_uds,
        s1_cache, layer_list, 0.05, "span", 32,
    )
    total_time = time.time() - t0

    delta = uds_after - uds_before
    print(f"  UDS: {uds_before:.4f} -> {uds_after:.4f} (delta: {delta:+.4f})")
    print(f"  1-UDS: {1-uds_before:.4f} -> {1-uds_after:.4f}")
    print(f"  Total: {total_time:.1f}s")

    results[f"bs{bs}"] = {
        "uds_after": uds_after,
        "delta": delta,
        "time": total_time,
    }

    del ft_model
    free_memory()
    shutil.rmtree(ft_dir, ignore_errors=True)

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Retain UDS (before): {results['before']:.4f}")
print(f"{'bs':>4s} | {'effective':>9s} | {'UDS after':>9s} | {'delta':>8s} | {'1-UDS bef':>9s} -> {'1-UDS aft':>9s}")
print("-" * 70)
for bs in BATCH_SIZES:
    key = f"bs{bs}"
    if isinstance(results.get(key), dict):
        r = results[key]
        print(f"{bs:>4d} | {bs*GRAD_ACCUM:>9d} | {r['uds_after']:>9.4f} | {r['delta']:>+8.4f} | {1-results['before']:>9.4f} -> {1-r['uds_after']:>9.4f}")
    else:
        print(f"{bs:>4d} | {bs*GRAD_ACCUM:>9d} | FAILED")

# Save
out = Path("runs/meta_eval/robustness/retain_batchsize_test.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {out}")

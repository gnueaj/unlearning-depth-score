# CLAUDE.md - Assistant Guide

This file is the current assistant-facing map of the repo.

## Scope
The project evaluates unlearning using:
- **Method-level metrics** across many unlearned checkpoints
- **Meta-evaluation** (Faithfulness / Robustness) in Open-Unlearning style
- **UDS** as the internal intervention metric

Core claim: output suppression is not enough; internal recoverability must also be measured.

## Canonical Outputs
- Dashboard: `docs/openunlearning_alpha_all.html`
- Dashboard data:
  - `docs/data/method_results.json`
  - `docs/data/meta_eval.json`
- Method-level runs:
  - `runs/ep5/{memorization,privacy,utility,uds,gen_rouge}/<model>/`
  - `runs/ep10/{memorization,privacy,utility,uds,gen_rouge}/<model>/`
- Meta-eval runs:
  - `runs/meta_eval/faithfulness/` (results.json, summary.json, histograms/)
  - `runs/meta_eval/faithfulness_uds_v2.json` (UDS with s1_cache_v2)
  - `runs/meta_eval/robustness_v2/quant/results.json`
  - `runs/meta_eval/robustness_v2/relearn/results.json`
  - `runs/meta_eval/s1_cache_v2.json` (367 examples, eager attention)
- Legacy: `runs/archive/`

## Canonical Scripts
- UDS per model: `exp_s1_teacher_forcing.py`
- Generation metrics backfill: `scripts/compute_generation_metrics_all.py`
- Meta-eval:
  - `scripts/meta_eval_faithfulness.py`
  - `scripts/meta_eval_robustness.py`
- Robustness utilities:
  - `scripts/build_robustness_filter_list.py`
  - `scripts/build_robustness_model_list.py`
- Legacy scripts: `scripts/archive_legacy/`

## Data + Prompting Conventions
- **UDS**: `tofu_data/forget10_filtered_v7_gt.json` (367)
  - raw `Question/Answer` style patch evaluation
  - default: `patch_scope=span`, `em_scope=entity`, `delta_threshold=0.05`
- **Open-Unlearning-style metrics**: 400-example perturbed protocol
  - chat template + system prompt (`You are a helpful assistant.`)
  - includes generation metrics and MIA metrics

Do not mix these two evaluation settings silently.

## UDS Procedure (Detailed)

### Core Idea
UDS probes internal representations via activation patching. Instead of checking output behavior, we measure whether knowledge can be recovered from hidden states at each layer.

### Models Involved
```
Full model:    open-unlearning/tofu_Llama-3.2-1B-Instruct_full     (16 layers)
Retain model:  open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90
Unlearn model: open-unlearning/unlearn_tofu_..._<method>_<params>
```

### Patching Mechanism (Code: lines 346-386 in exp_s1_teacher_forcing.py)
```python
# 1. Capture source hidden states
def capture_hook(module, inputs, output):
    source_hidden_all = output[0].detach().clone()
source_layer.register_forward_hook(capture_hook)
source_model(input_ids)

# 2. Inject into target at same positions
def patch_hook(module, inputs, output):
    hs = output[0].clone()
    hs[:, patch_start:patch_end, :] = source_hidden[:, patch_start:patch_end, :]
    return (hs,) + output[1:]
target_layer.register_forward_hook(patch_hook)
outputs = target_model(input_ids)
```

### Teacher Forcing Setup
- Input: `prompt_ids` (with BOS) + `ref_ids` (answer tokens, no special tokens)
- Evaluation positions: `start = len(prompt_ids) - 1` to `start + len(ref_ids)`
- The position at index `t` predicts token at index `t+1`

### Entity Span Extraction (Code: get_eval_span function)
```python
# Using fast tokenizer with offset mapping
enc = tokenizer(reference, return_offsets_mapping=True)
char_start = reference.find(entity)
token_indices = [i for i, (s, e) in enumerate(offsets) if e > char_start and s < char_end]
eval_span = (min(token_indices), max(token_indices) + 1)
```

### Log-Probability Computation
```python
# Gather log-prob for each reference token
log_probs = torch.log_softmax(logits, dim=-1)
token_logprobs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1))
mean_logprob = token_logprobs.mean().item()
```

### Delta Measurement
```
Δ_l = full_logprob - patched_logprob

If Δ > τ:  layer marked "LOST" (patching degraded prediction)
If Δ ≤ τ:  layer marked "KEPT" (source model had sufficient knowledge)
```

### FT Layer Identification
```
FT_layers = { l : Δ^S1_l > τ }
```
Layers where retain→full patching causes degradation. These contain fine-tuned knowledge.

### UDS Formula
```
UDS_i = Σ_{l∈FT} [ Δ^S1_l × clip(Δ^S2_l / Δ^S1_l, 0, 1) ] / Σ_{l∈FT} Δ^S1_l
```

Code (lines 1093-1109):
```python
for s1, s2, d1, d2 in zip(s1_details, s2_details, s1_deltas, s2_deltas):
    if s1["layer"] not in ft_set or d1 <= delta_threshold:
        continue
    denom += d1
    ratio = max(0.0, min(1.0, d2 / d1))  # clip to [0, 1]
    numer += d1 * ratio
uds = numer / denom if denom > 0 else None
```

### Default Parameters
- `delta_threshold (τ)`: 0.05
- `patch_scope`: span (patch all answer positions)
- `em_scope`: entity (evaluate entity span only)
- `reference`: gt (use ground truth answer)
- `batch_size`: 32 (for log-prob metric)

### Interpretation
| UDS | Meaning |
|-----|---------|
| 1.0 | Unlearned model matches retain's knowledge gap |
| 0.0 | Knowledge fully intact internally |
| N/A | No FT signal (denom = 0) |

### S1 Caching
S1 results (retain→full) are cached to avoid redundant computation:
- Cache path: `runs/meta_eval/s1_cache_v2.json`
- Config hash ensures cache validity across parameter changes
- Attention implementation must match (`eager` for consistency)

## Method-Level Aggregation (Current Dashboard Contract)
- `Mem.` from memorization summary
- `Privacy` currently includes MIA aggregate and UDS via HM in dashboard pipeline
- `Utility` is normalized relative to full-model utility for each epoch
- `Overall` is HM of the displayed top-level axes in `docs/data/method_results.json`

When changing aggregation, change both builder logic and HTML labels together.

## Meta-Eval Contract

### Faithfulness
- Uses P/N pools (30 pos + 30 neg = 60 models) from Open-Unlearning
- Reports per-metric AUC-ROC for separation quality
- P-pool: models trained on forget10 (have knowledge)
- N-pool: models NOT trained on forget10 (no knowledge)

### Robustness
- **Formulas**:
  - `R = min((m_ret_before - m_ret_after)/(m_unl_before - m_unl_after), 1)`
  - `Q = min(m_unl_after/m_unl_before, 1)`
  - Per-metric robustness = `HM(avg_R, avg_Q)`

- **Aggregation method**:
  - For each metric: calculate R (or Q) for each model
  - Average across all models: `avg_R = mean([R_model1, R_model2, ...])`
  - Final per-metric robustness = `HM(avg_R, avg_Q)`

- **Direction policy**:
  - All Open-Unlearning metrics: use **raw values**
  - UDS only: convert to `m = 1 - uds` before applying formulas
  - MIA: raw AUC for meta-eval; sMIA for method-level privacy

- **Attack settings** (Open-Unlearning paper):
  - Relearning: `lr=2e-5`, `batch_size=8`, `grad_accum=4` (effective=32), `epochs=1`
  - Quantization: BitsAndBytes 4-bit with FP4 + float32 (Transformers default `load_in_4bit=True`)

- **Model universe**:
  - 151 models total: 1 retain + 150 unlearned (75 ep5 + 75 ep10)
  - Retain model used for normalization (numerator in R formula)
  - Retain excluded from final aggregation

- **Filtering policy** (aggregation 시 적용):
  - **Utility filter**: `utility_rel < 0.8` 모델 제외 (20% 이상 utility 하락)
  - **Faithfulness filter**: 메트릭별 P/N pool threshold 미달 시 제외
  - ~~`lr=1e-5` subset filtering~~ → **취소됨** (모든 lr 포함)
  - Raw results.json은 전체 150개 포함, 필터링은 최종 aggregation에서 적용

### Current Paths
- S1 cache: `runs/meta_eval/s1_cache_v2.json` (367 examples, eager attention)
- Faithfulness (12 metrics): `runs/meta_eval/faithfulness/results.json`, `summary.json`
- Faithfulness UDS: `runs/meta_eval/faithfulness_uds_v2.json` (AUC: 0.973)
- Robustness: `runs/meta_eval/robustness_v2/{quant,relearn}/results.json`

### Notes
- S1 cache must be consistent between faithfulness and robustness UDS computation
- Retain model's UDS should be exactly 1.0 (verify with matching S1 cache)

## Operational Notes
- Use `--resume` paths for interrupted long runs.
- Keep only one active writer per output `results.json`.
- Archive legacy runs under `runs/archive/` and avoid reading them in builders.
- Robustness script auto-clears HF cache after each model (`--clear_cache=True` default).
- Relearn checkpoints are auto-deleted after metrics computation.
- Monitor disk space during long runs (HF cache can grow quickly).

## Fast Sanity Checklist
Before publishing numbers:
1. `docs/data/method_results.json` model count matches expected run set.
2. `docs/data/meta_eval.json` comes from latest `runs/meta_eval/faithfulness` + robustness results.
3. `docs/openunlearning_alpha_all.html` labels match the current aggregation formulas.
4. No mixed old/new schema keys (`avg_udr` vs `avg_uds`) without explicit fallback handling.

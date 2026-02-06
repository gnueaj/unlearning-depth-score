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
  - `runs/meta_eval/faithfulness/results.json`, `summary.json`
  - `runs/meta_eval/robustness_v2/quant/results.json`
  - `runs/meta_eval/robustness_v2/relearn/results.json`
  - S1 cache: `runs/meta_eval/s1_cache_v2.json`
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

## UDS Definition (Current)
For example `i`, FT layers are those with `Δ^{S1}_{i,l} > τ`.

```
UDS_i =
  (Σ_{l∈FT_i} Δ^{S1}_{i,l} * clip(Δ^{S2}_{i,l}/Δ^{S1}_{i,l}, 0, 1))
  / Σ_{l∈FT_i} Δ^{S1}_{i,l}
```

- `τ` default is `0.05`
- Higher UDS means deeper erasure
- `avg_uds` in summaries is the dataset average

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

- **Filtering policy** (current):
  - No filtering applied; all 150 unlearned models included
  - ~~`lr=1e-5` subset filtering~~ → **취소됨** (모든 lr 포함)
  - (Legacy: utility filter, faithfulness-threshold filter는 aggregation 시 선택적 적용 가능)

### Current Paths
- S1 cache: `runs/meta_eval/s1_cache_v2.json` (367 examples, computed with eager attention)
- Faithfulness: `runs/meta_eval/faithfulness/results.json`, `summary.json`
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

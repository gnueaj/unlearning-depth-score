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
- Faithfulness runs:
  - `runs/faithfulness/results.json`
  - `runs/faithfulness/summary.json`
- Robustness runs:
  - `runs/meta_eval/robustness_parallel/part0/`
  - `runs/meta_eval/robustness_parallel/part1/`
  - filter artifacts in `runs/meta_eval/robustness_filter_list/`

## Canonical Scripts
- UDS per model: `exp_s1_teacher_forcing.py`
- Generation metrics backfill: `scripts/compute_generation_metrics_all.py`
- Meta-eval:
  - `scripts/meta_eval_faithfulness.py`
  - `scripts/meta_eval_robustness.py`
- Robustness filtering/list utilities:
  - `scripts/build_robustness_filter_list.py`
  - `scripts/build_robustness_model_list.py`
- HTML/data build:
  - `scripts/build_openunlearning_alpha_all.py`
  - `scripts/update_meta_eval_html.py`

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
- Faithfulness uses P/N pools and reports metric-wise separation quality.
- Robustness uses relearning + quantization logic and supports optional filtering:
  - insufficient-unlearning filter from faithfulness-derived thresholds
  - utility-drop filter

If filter logic changes, regenerate:
- `runs/meta_eval/robustness_filter_list/filtered_models.json`
- downstream robustness runs
- `docs/data/meta_eval.json`

## Operational Notes
- Use `--resume` paths for interrupted long runs.
- Keep only one active writer per output `results.json`.
- Archive legacy runs under `runs/archive/` and avoid reading them in builders.

## Fast Sanity Checklist
Before publishing numbers:
1. `docs/data/method_results.json` model count matches expected run set.
2. `docs/data/meta_eval.json` comes from latest `runs/faithfulness` + latest robustness summary.
3. `docs/openunlearning_alpha_all.html` labels match the current aggregation formulas.
4. No mixed old/new schema keys (`avg_udr` vs `avg_uds`) without explicit fallback handling.

# Measuring the Depth of LLM Unlearning via Activation Patching

This repository evaluates unlearning with a white-box intervention metric (**UDS**) and Open-Unlearning-style metrics.

The central question is not only whether output behavior changes, but whether target knowledge remains recoverable from internal representations.

## What This Repo Produces

- Method-level results across checkpoints (ep5/ep10)
- Meta-evaluation (Faithfulness / Robustness)
- Unified dashboard and machine-readable data

Primary dashboard:
- `docs/openunlearning_alpha_all.html`

Primary data feeds for the dashboard:
- `docs/data/method_results.json`
- `docs/data/meta_eval.json`

## Core Metrics

### 1) UDS (Unlearning Depth Score)

UDS is computed with two-stage patching:
- S1: `retain -> full`
- S2: `unlearned -> full`

For each example `i`, define FT layers where `Δ^{S1}_{i,l} > τ`.

```text
UDS_i =
  (Σ_{l∈FT_i} Δ^{S1}_{i,l} * clip(Δ^{S2}_{i,l}/Δ^{S1}_{i,l}, 0, 1))
  / Σ_{l∈FT_i} Δ^{S1}_{i,l}
```

- Default `τ`: `0.05`
- Higher UDS means deeper erasure.

UDS script:
- `exp_s1_teacher_forcing.py`

### 2) Method-Level Axes

Per-model evaluation is organized into:
- Memorization (`Mem.`)
- Privacy
- Utility (normalized vs full model per epoch)
- Overall aggregation shown in dashboard data

Source folders:
- `runs/ep5/`
- `runs/ep10/`

Each epoch has:
- `memorization/<model>/summary.json`
- `privacy/<model>/summary.json`
- `utility/<model>/summary.json`
- `uds/<model>/summary.json`
- `gen_rouge/<model>/summary.json`

### 3) Meta-Evaluation

Scripts:
- `scripts/meta_eval_faithfulness.py`
- `scripts/meta_eval_robustness.py`

Faithfulness uses P/N pools and measures metric-level separability.

Robustness evaluates metric stability under:
- relearning
- quantization

Filtering helpers:
- `scripts/build_robustness_filter_list.py`
- `scripts/build_robustness_model_list.py`

## Datasets / Prompting Conventions

Two evaluation settings coexist intentionally:

1. UDS setting
- `tofu_data/forget10_filtered_v7_gt.json` (367)
- raw `Question/Answer` style patching

2. Open-Unlearning-style setting
- 400-example perturbed evaluation protocol
- chat template + system prompt

Do not merge these settings without explicitly documenting the change.

## Rebuild / Refresh

Regenerate dashboard data + HTML:

```bash
python scripts/build_openunlearning_alpha_all.py --out_dir docs --ep5_dir runs/ep5 --ep10_dir runs/ep10
python scripts/update_meta_eval_html.py
```

Backfill generation metrics for models:

```bash
python scripts/compute_generation_metrics_all.py --epoch_root runs/ep5
python scripts/compute_generation_metrics_all.py --epoch_root runs/ep10
```

## Minimal Example (UDS)

```bash
python exp_s1_teacher_forcing.py \
  --unlearn_model simnpo_lr2e5_b35_a1_d1_g0125_ep5 \
  --num_examples 50 \
  --patch_scope span \
  --reference gt \
  --entity_source gt \
  --gpu 0 \
  --batch_size 32
```

## Notes

- Current naming is **UDS** (older artifacts may still contain `udr` key aliases).
- Avoid reading from `runs/archive/` for current tables.
- If metrics/formulas are changed, update both:
  - builder logic
  - labels/formula blocks shown in HTML

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
- `runs/ep5/{memorization,privacy,utility,uds,gen_rouge}/<model>/`
- `runs/ep10/{memorization,privacy,utility,uds,gen_rouge}/<model>/`

### 3) Meta-Evaluation

Scripts:
- `scripts/meta_eval_faithfulness.py`
- `scripts/meta_eval_robustness.py`

**Faithfulness**: Uses P/N pools (30 pos + 30 neg) and measures metric-level AUC-ROC separability.

**Robustness**: Evaluates metric stability under relearning and quantization attacks.
- `R = min((m_ret_before - m_ret_after)/(m_unl_before - m_unl_after), 1)`
- `Q = min(m_unl_after/m_unl_before, 1)`
- Per-metric robustness = `HM(avg_R, avg_Q)`

Results:
- `runs/meta_eval/faithfulness/` (12 metrics + histograms)
- `runs/meta_eval/faithfulness_uds_v2.json` (UDS AUC: 0.973)
- `runs/meta_eval/robustness_v2/{quant,relearn}/results.json`
- `runs/meta_eval/s1_cache_v2.json` (367 examples)

## Datasets / Prompting Conventions

Two evaluation settings coexist intentionally:

1. **UDS setting**
   - `tofu_data/forget10_filtered_v7_gt.json` (367 examples)
   - raw `Question/Answer` style patching

2. **Open-Unlearning-style setting**
   - 400-example perturbed evaluation protocol
   - chat template + system prompt (`You are a helpful assistant.`)

Do not merge these settings without explicitly documenting the change.

## Dashboard Update

Dashboard uses JSON data files:
- `docs/data/method_results.json` - method-level results
- `docs/data/meta_eval.json` - faithfulness + robustness

HTML is edited directly in `docs/openunlearning_alpha_all.html`.

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
- Legacy runs and scripts archived under `runs/archive/` and `scripts/archive_legacy/`.
- If metrics/formulas are changed, update both data files and HTML labels together.

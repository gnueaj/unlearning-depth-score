# Measuring the Depth of LLM Unlearning via Activation Patching

This repository evaluates unlearning with a white-box intervention metric (**UDS**) and Open-Unlearning-style metrics.

The central question is not only whether output behavior changes, but whether target knowledge remains recoverable from internal representations.

## What This Repo Produces

- Method-level results across 152 models (8 methods × hyperparameters × 2 epochs + full + retain)
- Meta-evaluation (Faithfulness / Robustness) with 13 metrics + 4 normalized MIA + 3 representation baselines
- Unified dashboard and machine-readable data

Primary dashboard:
- `docs/openunlearning_alpha_all.html`

Primary data feeds for the dashboard:
- `docs/data/method_results.json` — method-level results (152 models)
- `docs/data/meta_eval.json` — faithfulness + robustness (22 metrics)

## Core Metrics

### 1) UDS (Unlearning Depth Score)

UDS measures how deeply an unlearning method erases target knowledge from internal representations, not just output behavior.

#### Motivation

Output-based metrics can be fooled by superficial suppression—a model might refuse to answer while still storing recoverable knowledge internally. UDS uses **activation patching** to probe whether knowledge persists at each layer.

#### Two-Stage Patching

We compare two patching experiments:
- **S1 (baseline)**: `retain → full` — How much knowledge does the retain model lack?
- **S2 (test)**: `unlearned → full` — How much knowledge does the unlearned model lack?

For each layer, we patch hidden states from a source model into the full model and measure log-probability degradation on **entity tokens** (the core factual content).

#### Log-Probability Delta

For each layer `l`:
- `Δ_l = logprob_full - logprob_patched`
- Large Δ = source model lacks knowledge at that layer

#### FT Layer Identification

Layers containing fine-tuned knowledge are identified via S1:
- `FT_layers = { l : Δ^S1_l > τ }` (default τ = 0.05)

#### UDS Formula

```
UDS_i = Σ_{l∈FT} [ Δ^S1_l × clip(Δ^S2_l / Δ^S1_l, 0, 1) ] / Σ_{l∈FT} Δ^S1_l
```

The ratio `Δ^S2 / Δ^S1` measures how much of the knowledge gap (seen in retain) also appears in the unlearned model. Clipping prevents artifacts from over-erasure.

#### Interpretation

| UDS | Meaning |
|-----|---------|
| 1.0 | Complete erasure — unlearned model matches retain's knowledge gap |
| 0.5 | Partial erasure — knowledge partially recoverable |
| 0.0 | No erasure — knowledge fully intact despite output suppression |

UDS script: `exp_s1_teacher_forcing.py`

### 2) Method-Level Axes

Per-model evaluation is organized into:
- **Memorization** (`Mem.`) = HM(1-ES, 1-EM, 1-ParaProb, 1-TruthRatio)
- **Privacy** = HM(MIA, UDS), where MIA = HM(s_LOSS, s_ZLib, s_Min-K, s_Min-K++)
  - `normalized = |AUC_model - AUC_retain| / AUC_retain` (deviation ratio; higher = more knowledge)
  - `s_* = clip(1 - normalized, 0, 1)` (inverted; 1.0 = erased, 0.0 = large deviation from retain)
- **Utility** = HM(ModelUtility, Fluency), normalized vs full model per epoch
- **Overall** = HM(Mem., Privacy, Utility)

Source folders:
- `runs/ep5/{memorization,privacy,utility,uds,gen_rouge}/<model>/`
- `runs/ep10/{memorization,privacy,utility,uds,gen_rouge}/<model>/`

### 3) Meta-Evaluation

Scripts:
- `scripts/meta_eval_faithfulness.py`
- `scripts/meta_eval_robustness.py`

**Faithfulness**: Uses P/N pools (30 pos + 30 neg = 60 models) and measures metric-level AUC-ROC separability.

**Robustness**: Evaluates metric stability under relearning and quantization attacks using symmetric (bidirectional) formulas:
- `Q = 1 - clip(|m_after - m_before| / (|m_before| + |m_after| + ε), 0, 1)` — penalizes both recovery and destruction
- `R = 1 - clip(|Δ_unl - Δ_ret| / (|Δ_unl| + |Δ_ret| + ε), 0, 1)` — penalizes deviation from retain's relearning behavior
- Per-metric robustness = `HM(avg_R, avg_Q)`
- Filtering: utility ≥ 0.8 + per-metric faithfulness threshold

Results:
- `runs/meta_eval/faithfulness/` (results.json, summary.json, histograms/)
- `runs/meta_eval/robustness/{quant,relearn}/results.json`
- `runs/meta_eval/robustness/{quant,relearn}/rep_baselines_results.json`
- S1 cache: `runs/meta_eval/s1_cache_v2.json` (eager), `s1_cache_sdpa.json` (sdpa)

### 4) Representation Baselines

To validate that UDS captures something beyond existing representation-level methods, we compare against three baselines. All use the retain model as reference and operate on the same forget set.

**CKA** (Centered Kernel Alignment) — Measures representational geometry similarity between unlearned and retain models, weighted by per-layer importance (how much full differs from retain).
- `score = Σ_l w_l · CKA(H_unl, H_retain)_l`,  `w_l = 1 - CKA(H_full, H_retain)_l`
- Dataset-level (400 examples). AUC: 0.648.

**Logit Lens** — Projects each layer's hidden states through the full model's frozen decoder to measure decodable knowledge. Uses the same FT layer selection (τ = 0.05) and UDS-style aggregation.
- Per-example, per-layer entity logprob readout (367 examples). AUC: 0.927.
- Key detail: forward hook on `model.model.norm` captures pre-norm hidden state for the last layer (avoids double-norm artifact from `output_hidden_states`).

**Fisher Masked** — Diagonal Fisher Information with top-p% parameter masking per layer. Focuses on knowledge-relevant parameters where retain has higher sensitivity than full.
- `erasure_l = 1 - clip(excess_unl / excess_full, 0, 1)`, weighted by per-layer `excess_full`
- Mask fractions: 0.01%, 0.1%, 1%. AUC: 0.708–0.712.
- Known: layer 1 dominates weight (60–84%), making results nearly identical across mask fractions.
- Quant robustness: NF4 quantized models require dequantization (`bnb.functional.dequantize_4bit`) before Fisher computation since `requires_grad=False`.

Script: `scripts/compute_representation_baselines.py`

| Method | Faithfulness AUC | Approach |
|--------|-----------------|----------|
| CKA | 0.648 | Geometry similarity |
| Fisher Masked (0.1%) | 0.712 | Parameter sensitivity |
| Logit Lens | 0.927 | Frozen decoder readout |
| **UDS (Ours)** | **0.971** | **Activation patching** |

## Datasets / Prompting Conventions

Two evaluation settings coexist intentionally:

1. **UDS setting** — `tofu_data/forget10_filtered_v7_gt.json` (367 examples)
   - raw `Question/Answer` style, entity span annotations for teacher forcing
   - Used by: **UDS**, **Logit Lens**, **Fisher Masked**

2. **Open-Unlearning-style setting** — HuggingFace `locuslab/TOFU` `forget10_perturbed` (400 examples)
   - chat template + system prompt (`You are a helpful assistant.`)
   - paraphrase/perturbed answer variants included
   - Used by: **EM, ES, Prob, ParaProb, Truth Ratio, ROUGE, Para-ROUGE, Jailbreak-ROUGE, MIA-\*, CKA**

### Dataset-Metric Mapping

| Dataset | Metrics | Key Property |
|---------|---------|-------------|
| v7_gt (367, local) | UDS, Logit Lens, Fisher Masked | Entity span annotation → token-level teacher forcing |
| forget10_perturbed (400, HF) | EM, ES, Prob, ParaProb, Truth Ratio, ROUGE, Para-ROUGE, Jailbreak-ROUGE, MIA-LOSS/ZLib/MinK/MinK++, CKA | Paraphrase/perturbed answers → generation + MIA eval |

CKA is the only representation baseline using the 400-example dataset — it builds dataset-level kernel matrices and does not require entity annotations.

Do not merge these settings without explicitly documenting the change.

## Dashboard

Dashboard uses JSON data files:
- `docs/data/method_results.json` - method-level results
- `docs/data/meta_eval.json` - faithfulness + robustness

HTML is edited directly in `docs/openunlearning_alpha_all.html`.

Method-level table columns: Model, Overall, Mem., Privacy, Utility, LL (Logit Lens), UDS, expand.

## Minimal Example (UDS)

```bash
python exp_s1_teacher_forcing.py \
  --unlearn_model simnpo_lr2e5_b35_a1_d1_g0125_ep5 \
  --patch_scope span \
  --reference gt \
  --entity_source gt \
  --gpu 0 \
  --batch_size 32
```

## Notes

- Current naming is **UDS** (older artifacts may still contain `udr` key aliases).
- Legacy runs under `runs/legacy/`; legacy scripts under `scripts/legacy/`.
- If metrics/formulas are changed, update both data files and HTML labels together.

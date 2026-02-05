# CLAUDE.md - Assistant Guide

This document provides context for assistants working on this repo. It is intentionally concise and current.

## Project Overview
**Activation Patching for Unlearning Audit** quantifies **residual knowledge** in unlearned LLMs by patching hidden states into a full model and measuring how much target knowledge is recoverable.

We report **UDS (Unlearning Depth Score)** as the core metric. Higher = better erasure.

## Current Structure (as of 2026-02-06)

### Key Outputs
- **Open-Unlearning Table/HTML**: `docs/openunlearning_alpha_all.html`
- **Paper PDF**: `docs/openunlearning.pdf`
- **Faithfulness results**: `runs/faithfulness/` (summary.json, results.json, logs/)
  - 60 models (30 P + 30 N), SDPA, 13 metrics including UDS
- **Meta-eval runs**: `runs/meta_eval/`
  - `table2_faithfulness_v2_eager/` — old eager attention (30+29 models)
  - `table2_faithfulness_v3_gpu0/`, `v3_gpu1/` — SDPA split runs
  - `table2_faithfulness_uds_sdpa_gpu0/`, `gpu1/` — UDS SDPA runs
- **EP5 / EP10 evaluation results**:
  - `runs/ep5/{memorization,privacy,utility,uds}/{model}/`
  - `runs/ep10/{memorization,privacy,utility,uds}/{model}/`
  - 150 models total (75 configs × 2 epochs)

### Core Scripts
- UDS experiment: `exp_s1_teacher_forcing.py`
- Meta-eval (Table 2 replication):
  - `scripts/meta_eval_faithfulness.py`
  - `scripts/meta_eval_robustness.py`
- HTML update: `scripts/update_meta_eval_html.py`
- HTML builder: `scripts/build_openunlearning_alpha_all.py`

## Dataset & Prompting
- **UDS** uses filtered dataset: `tofu_data/forget10_filtered_v7_gt.json` (367 examples)
- **Open-Unlearning metrics** use `forget10_perturbed` (400 examples)
- UDS patching uses raw format: `Question: ...\nAnswer: ...` (no chat template)
- Open-Unlearning metrics use chat template + system prompt ("You are a helpful assistant.")

## UDS (Unlearning Depth Score)

### Two-Stage Patching
- **S1 (Retain → Full)**: identify FT layers where Retain lacks knowledge
- **S2 (Unlearn → Full)**: test whether Unlearn also lacks that knowledge

### Definition
Let FT layers be those with `Δ^{S1}_ℓ > τ`.

```
UDS_i =
  (Σ_{ℓ∈FT_i} Δ^{S1}_{i,ℓ} · clip(Δ^{S2}_{i,ℓ}/Δ^{S1}_{i,ℓ}, 0, 1))
  / Σ_{ℓ∈FT_i} Δ^{S1}_{i,ℓ}
```

- **τ**: noise filter (current default 0.05)
- `clip` handles negative or over-erased ratios
- **Higher UDS = better unlearning**

## Meta-eval (Table 2 Reproduction)

### Metrics (13 total)
| Category | Metrics | Generation Required |
|----------|---------|---------------------|
| Memorization | ES, EM, Prob, Para.Prob, Truth Ratio | No |
| Generation | ROUGE, Para.ROUGE, Jailbreak ROUGE | Yes |
| Privacy (MIA) | LOSS, ZLib, MinK, MinK++ | No |
| Ours | UDS | No |

### Faithfulness
- AUC-ROC(P, N) where P = positive pool (with knowledge), N = negative pool (without)
- 30 P + 30 N models from TOFU variants
- Results: `runs/faithfulness/summary.json`

### Robustness (not yet measured)
- **Quantization** = min(m_after / m_before, 1) — metric stability under 4-bit quantization
- **Relearning** = min(Δ_retain / Δ_unlearn, 1) — metric change after 1 epoch relearning
- **Robustness** = HM(Quantization, Relearning)

### Important Notes
- **ROUGE metrics require generation**: Not stored per-model in ep5/ep10; only computed during faithfulness eval
- **150 models** for robustness (75 configs × 2 epochs), not 75
- Paper uses **Llama-3.2-1B-Instruct** (same as us)

## EP5/EP10 Stored Metrics
| Folder | Metrics Available |
|--------|-------------------|
| memorization/ | em, es, prob, paraprob, truth_ratio |
| privacy/ | mia_loss, mia_zlib, mia_min_k, mia_min_kpp |
| utility/ | retain/ra/wf × {Prob, ROUGE, TruthRatio} |
| uds/ | uds per example |

**Missing**: forget10 ROUGE/para_rouge/jailbreak_rouge (requires generation, not stored)

## GPU Usage
- Use `--gpu 0` or `--gpu 1` (script sets `CUDA_VISIBLE_DEVICES` internally)
- Avoid stacking multiple heavy runs on the same GPU

## Performance Optimizations (implemented)
- **Source hidden precompute**: retain/unlearn hidden states extracted once per example
- **S1 cache**: retain→full deltas reused across models
- **Optional generation skip** for logging (not used in UDS computation)

These do **not** change results, only runtime.

## Notes
- "UDS" replaces previous name "UDR".
- HTML regenerate after metric refresh: `python scripts/update_meta_eval_html.py`

# Measuring the Depth of LLM Unlearning via Activation Patching

A white-box analysis framework for quantifying **residual knowledge** in unlearned LLMs via activation patching. This tool asks: **How much of the targeted knowledge still survives in internal representations—and how recoverable is it?**

**Recent updates (2026-02-04)**  
- Added **batch_size** to `exp_s1_teacher_forcing.py` (log-prob + layer mode) to speed up UDS runs.  
- Standardized **UDS naming**; summaries now export `avg_uds` and a compatibility `avg_udr`.  
- **SimNPO γ sweep** added (γ=0.25 alongside γ=0.125; β ∈ {3.5,4.5}, lr ∈ {1e-5,2e-5,5e-5}).  
- Open-Unlearning **alpha_all** table/HTML refreshed with formulas + method count.

## Table of Contents

- [Motivation](#motivation)
- [Method Overview](#method-overview)
- [Dataset](#dataset)
- [Metrics](#metrics)
- [Open-Unlearning Evaluation](#open-unlearning-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

---

## Motivation

Most unlearning methods can *suppress* responses, but that does not guarantee the knowledge is removed from internal representations. We measure **recoverability**: if hidden states from an unlearned model are patched into the full model, do they restore the target knowledge?

---

## Method Overview

### Stage 0 (Baseline)

**Baseline (no-patch):** log-probability of the reference answer under the full model. This is the reference score for all Δ computations.

### Two-Stage Patching

**Stage 1 (S1)**
- **Question:** *Where and how large is the knowledge that should be erased?*
- Patch **Retain → Full** to identify layers encoding target knowledge.

**Stage 2 (S2)**
- **Question:** *Where and how large is the knowledge that remains after unlearning?*
- Patch **Unlearned → Full** to measure residual knowledge at those layers.

### Patching Mechanism (Teacher Forcing)

We use teacher forcing on the **reference answer span** (GT entity span by default). For each layer ℓ, we patch hidden states from the source model into the full model and measure the reference log-probability.

- **Patch scope:** `span` (default in our experiments) or `boundary` (last prompt token only).
- **Reference:** GT answer continuation after prefix (`reference=gt`, `reference_scope=continuation`).

---

## Dataset

### Base: TOFU forget10

We use the [TOFU](https://github.com/locuslab/tofu) benchmark (forget10 split, 400 QA pairs). We filter out examples where the full model deviates strongly from the GT answer to ensure stable reference spans.

### Current Dataset (v7_gt)

| Field | Description |
|-------|-------------|
| `prefix` | Manually verified prefix ending right before the entity |
| `entity` | GT entity used as evaluation reference |
| `entity_span` | Token span of the entity within the reference |

**Filtered size:** 367 (from 400 after removing mismatched/full-wrong examples).

Dataset file: `tofu_data/forget10_filtered_v7_gt.json`

---

## Metrics

### Notation

- $x$: input prompt (includes prefix)
- $M_{\text{full}}$: full model
- $M_S$: source model (S1=Retain, S2=Unlearned)
- $y=(y_1,\dots,y_T)$: reference entity span tokens
- $\ell$: layer index

### Baseline and Patched Scores

**Baseline (no patch):**
$$s^{\text{full}}_{t}=\log p_{M_{\text{full}}}(y_t\mid x, y_{<t})$$

**Patched:**
$$s^{S}_{t}=\log p_{M_{\text{full}}}^{\text{patch}(M_S)}(y_t\mid x, y_{<t})$$

### Delta (Patch-induced degradation)

$$\Delta^{S}_{t}=s^{\text{full}}_{t}-s^{S}_{t}$$

Span-level aggregation:
$$\Delta^{S}=\frac{1}{T}\sum_{t=1}^{T}\Delta^{S}_{t}$$

### FT (Fine-Tuned) Layers

$$\mathrm{FT}_i=\{\ell\mid \Delta^{S1}_{i,\ell} > \tau\},\; \tau=0.05$$

### UDS (Unlearning Depth Score)

$$\mathrm{UDS}_i=\frac{\sum_{\ell\in \mathrm{FT}_i} \Delta^{S1}_{i,\ell}\cdot\mathrm{clip}(\Delta^{S2}_{i,\ell}/\Delta^{S1}_{i,\ell},0,1)}{\sum_{\ell\in \mathrm{FT}_i} \Delta^{S1}_{i,\ell}}$$

$$\mathrm{UDS}=\frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}}\mathrm{UDS}_i$$

Interpretation:
- **UDS ≈ 1**: S2 deficit matches S1 → knowledge erased
- **UDS ≈ 0**: S2 deficit vanishes → knowledge retained

**Reference points (conceptual):**
- **Full** → UDS ≈ 0 (no deficit)
- **Retain** → UDS ≈ 1 (matches S1 by definition)

---

## Open-Unlearning Evaluation

We compute **Mem / Privacy / Utility** following Open-Unlearning / TOFU definitions:
- **Q/A Prob** uses length-normalized answer probability $P(a\mid q)=\\exp(-\\mathcal{L}/|a|)$.
- **Truth Ratio** uses the geometric mean of perturbed answers relative to the paraphrased (or original) answer, then mapped to $1-\\text{ratio}$ for utility.
- **Utility** is the harmonic mean over the 9 metrics (retain/RA/WF × {Prob, ROUGE, Truth Ratio}).

Utility is **normalized by Full (Full=1.00)** for comparison.

Table is generated from `docs/0202/openunlearning_alpha5_table.md` and mirrored below.

---

## Open-Unlearning Results (α=5)

**Columns**
- **Mem**: Open-Unlearning memorization score (HM of 1−ES/1−EM/1−ParaProb/1−TruthRatio, higher = more forgetting)
- **Privacy**: s_MIA score (HM of 4 MIA attacks: LOSS, MinK, MinK++, ZLib), where $s_{\text{MIA}} = 1 - \frac{|\text{AUC}_{\text{model}} - \text{AUC}_{\text{retain}}|}{|\text{AUC}_{\text{full}} - \text{AUC}_{\text{retain}}|}$
- **Utility (rel. to Full)**: utility normalized by Full utility (Full = 1.00)
- **UDS**: Unlearning Depth Score (this work)
- **Aggregate (HM)**: Harmonic mean of {Mem, Privacy, Utility(rel)}

| Model                            |   Mem |   Privacy |   Utility |   UDS |   Aggregate |
|:---------------------------------|------:|----------:|----------:|------:|------------:|
| altpo_lr1e5_b01_a5_ep5           | 0.270 |     0.008 |     0.920 | 0.242 |       0.023 |
| altpo_lr2e5_b01_a5_ep5           | 0.378 |     0.040 |     0.944 | 0.313 |       0.105 |
| altpo_lr5e5_b01_a5_ep5           | 0.548 |     0.415 |     0.884 | 0.298 |       0.559 |
| full                             | 0.094 |     0.004 |     1.000 | 0.000 |       0.010 |
| graddiff_lr1e5_a5_ep5            | 0.204 |     0.011 |     0.986 | 0.121 |       0.032 |
| graddiff_lr2e5_a5_ep5            | 0.504 |     0.569 |     0.993 | 0.376 |       0.632 |
| graddiff_lr5e5_a5_ep5            | 0.983 |     0.997 |     0.806 | 0.119 |       0.920 |
| idkdpo_lr1e5_b01_a5_ep5          | 0.239 |     0.008 |     0.973 | 0.157 |       0.022 |
| idkdpo_lr2e5_b01_a5_ep5          | 0.357 |     0.028 |     0.955 | 0.266 |       0.076 |
| idkdpo_lr5e5_b01_a5_ep5          | 0.477 |     0.179 |     0.894 | 0.321 |       0.341 |
| idknll_lr1e5_a5_ep5              | 0.163 |     0.004 |     0.948 | 0.141 |       0.012 |
| idknll_lr2e5_a5_ep5              | 0.197 |     0.005 |     0.936 | 0.205 |       0.014 |
| idknll_lr5e5_a5_ep5              | 0.360 |     0.014 |     0.850 | 0.261 |       0.038 |
| npo_lr1e5_b01_a5_ep5             | 0.444 |     0.088 |     0.895 | 0.279 |       0.204 |
| npo_lr2e5_b01_a5_ep5             | 0.527 |     0.365 |     0.957 | 0.320 |       0.528 |
| npo_lr5e5_b01_a5_ep5             | 0.595 |     0.757 |     0.920 | 0.277 |       0.734 |
| retain                           | 0.575 |     0.618 |     0.992 | 1.000 |       0.687 |
| rmu_lr1e5_l10_s10_ep5            | 0.117 |     0.004 |     0.990 | 0.154 |       0.011 |
| rmu_lr1e5_l15_s10_ep5            | 0.108 |     0.004 |     0.981 | 0.016 |       0.013 |
| rmu_lr1e5_l5_s10_ep5             | 0.173 |     0.005 |     0.977 | 0.098 |       0.014 |
| rmu_lr2e5_l10_s10_ep5            | 0.512 |     0.104 |     0.639 | 0.295 |       0.229 |
| rmu_lr2e5_l15_s10_ep5            | 0.176 |     0.005 |     0.984 | 0.037 |       0.015 |
| rmu_lr2e5_l5_s10_ep5             | 0.645 |     0.439 |     0.455 | 0.320 |       0.498 |
| rmu_lr5e5_l10_s10_ep5            | 0.780 |     0.968 |     0.491 | 0.158 |       0.690 |
| rmu_lr5e5_l15_s10_ep5            | 0.280 |     0.009 |     0.950 | 0.061 |       0.026 |
| rmu_lr5e5_l5_s10_ep5             | 0.790 |     0.995 |     0.873 | 0.096 |       0.878 |
| simnpo_lr1e5_b35_a1_d1_g0125_ep5 | 0.220 |     0.011 |     0.987 | 0.120 |       0.031 |
| simnpo_lr2e5_b35_a1_d1_g0125_ep5 | 0.404 |     0.186 |     1.001 | 0.335 |       0.339 |
| simnpo_lr5e5_b35_a1_d1_g0125_ep5 | 0.532 |     0.533 |     0.934 | 0.307 |       0.622 |
| undial_lr1e4_b10_a5_ep5          | 0.564 |     0.128 |     0.763 | 0.280 |       0.275 |
| undial_lr1e5_b10_a5_ep5          | 0.128 |     0.005 |     1.003 | 0.102 |       0.013 |
| undial_lr3e4_b10_a5_ep5          | 0.595 |     0.252 |     0.036 | 0.211 |       0.089 |

Notes:
- Utility is **scaled by Full** (Full=1.00). Values can exceed 1 if a model outperforms Full on Utility.
- Privacy uses **s_MIA formula** with all 4 MIA attacks (LOSS, MinK, MinK++, ZLib), computed as harmonic mean.
- Table source: `docs/0202/openunlearning_alpha5_table.md`
- Heatmap: `docs/0202/openunlearning_alpha5_metrics_heatmap.png`

---

## Installation

```bash
git clone https://github.com/gnueaj/activation-patching-unlearning.git
cd activation-patching-unlearning

pip install torch transformers datasets tqdm matplotlib
```

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA GPU (8GB+ VRAM)

---

## Usage

```bash
python exp_s1_teacher_forcing.py \
  --unlearn_model simnpo_lr2e5_b35_a1_d1_g0125_ep5 \
  --num_examples 50 \
  --patch_scope span \
  --reference gt \
  --entity_source gt \
  --gpu 0
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--unlearn_model` | required | Unlearning method config |
| `--num_examples` | `None` | Number of examples (default: all) |
| `--delta_threshold` | `0.05` | FT threshold τ |
| `--patch_scope` | `span` | `span` or `boundary` |
| `--reference` | `gt` | GT or Full reference |
| `--entity_source` | `gt` | GT or Full entity |
| `--gpu` | `0` | GPU selection (0 or 1) |

Results saved under `runs/` with `run.log`, `results.json`, and `summary.json`.

---

## Citation

TBA

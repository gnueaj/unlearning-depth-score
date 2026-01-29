# Measuring the Depth of LLM Unlearning via Activation Patching

A white-box analysis framework for auditing LLM unlearning via hidden state patching. This tool answers: **"Does an unlearned model truly forget, or just suppress knowledge at the output layer?"**

## Table of Contents

- [Motivation](#motivation)
- [Method Overview](#method-overview)
- [Dataset](#dataset)
- [Metrics](#metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

---

## Motivation

Most LLM unlearning methods achieve **behavioral unlearning**—the model outputs "I don't know" or wrong answers. However, this doesn't guarantee the knowledge is actually removed from the model's internal representations.

**Key Question**: Is the knowledge erased, or just suppressed at output?

Activation patching can expose this gap by directly probing hidden states.

---

## Method Overview

### Two-Stage Patching Framework

```
Stage 1 (S1): Retain → Full
  - Retain model: trained WITHOUT forget-set
  - Purpose: Identify layers encoding forget-set knowledge

Stage 2 (S2): Unlearn → Full
  - Unlearn model: trained to forget (SimNPO, IdkNLL, etc.)
  - Purpose: Measure how much knowledge is erased per layer
```

### Patching Mechanism

```
Source Model                          Target Model (Full)
     |                                      |
 Process prompt                        Process prompt
     |                                      |
 Extract hidden h_ℓ ────── patch ──────> Replace h_ℓ
     |                                      |
     ×                               Measure log-prob of
                                     reference tokens
```

- If patching Source→Full **decreases** probability → Source lacks that knowledge
- S1 identifies "fine-tuned (FT) layers" where Full learned new knowledge
- S2 measures how much the Unlearn model erased at those layers

### Teacher Forcing

We use teacher forcing to measure knowledge at each token position:

```
Prompt:    "Question: What is the author's name?\nAnswer: The author's full name is"
Reference: "Hsiao Yun-Hwa" (Full model's generation)

Position:  [prompt] [Hs] [iao] [ Yun] [-H] [wa]
                ↑
           Patch here (last prompt token)
```

---

## Dataset

### Base: TOFU forget10

[TOFU](https://github.com/locuslab/tofu) benchmark with 200 fictitious author profiles. The `forget10` split contains 400 QA pairs about 20 authors designated for unlearning.

### v6 Dataset

| Field | Description |
|-------|-------------|
| `prefix` | Manually verified prefix ending before the entity |
| `entity` | Evaluation reference (Full model's generation) |

**Why Full model's entity?** We measure whether Unlearn's hidden states can produce what Full would produce, not ground truth.

**Filtering:** 33 examples excluded where Full model generates wrong answers.

| Dataset | Count |
|---------|-------|
| Original TOFU forget10 | 400 |
| Full Model Wrong (excluded) | 33 |
| **Valid for Evaluation** | **367** |

---

## Metrics

### Notation

- $x$: Input prompt (including prefix)
- $M_{\text{full}}$: Full model
- $M_S$: Source model (S1=Retain, S2=Unlearn)
- $y=(y_1,\dots,y_T)$: Entity span tokens
- $\ell$: Layer index

### Log-Probability Scores

**Full model score (no patching):**

$$s^{\text{full}}_{t}=\log p_{M_{\text{full}}}\left(y_t \mid x, y_{1:t-1}\right)$$

**Patched score (source activation injected):**

$$s^{S}_{t}=\log p_{M_{\text{full}}}^{\text{patch}(M_S)}\left(y_t \mid x, y_{1:t-1}\right)$$

### Delta (Patch-induced Degradation)

$$\Delta^{S}_{t}=s^{\text{full}}_{t} - s^{S}_{t}$$

| Value | Interpretation |
|-------|----------------|
| Δ > 0 | Patching **decreases** correct token probability (knowledge gap) |
| Δ ≈ 0 | No significant effect |
| Δ < 0 | Patching **increases** probability (rare) |

Span-level aggregation: $\Delta^{S}=\frac{1}{T}\sum_{t=1}^{T}\Delta^{S}_{t}$

### FT (Fine-Tuned) Layers

$$\mathrm{FT}_i=\{\ell \mid \Delta^{S1}_{i,\ell}>\tau\}$$

Layers where S1 shows significant knowledge gap (τ=0.01 for noise filtering).

### UDR (Unlearning Depth Rate)

$$\mathrm{UDR}_i = \frac{\sum_{\ell\in \mathrm{FT}_i} \Delta^{S1}_{i,\ell}\cdot\mathrm{clip}\left(\Delta^{S2}_{i,\ell}/\Delta^{S1}_{i,\ell},0,1\right)}{\sum_{\ell\in \mathrm{FT}_i} \Delta^{S1}_{i,\ell}}$$

$$\mathrm{UDR}=\frac{1}{|\mathcal{I}|}\sum_{i\in \mathcal{I}}\mathrm{UDR}_i$$

- **Numerator**: Weighted sum of per-layer recovery ratios (clipped to [0,1])
- **Denominator**: Total S1 loss (normalization constant)
- **clip()**: Prevents overshoot at individual layers from inflating the score

| UDR | Interpretation |
|-----|----------------|
| ≈ 1.0 | S2 deficit matches S1 → knowledge **erased** |
| ≈ 0.0 | No deficit in S2 → knowledge **retained** |

| Condition | Meaning |
|-----------|---------|
| $\Delta^{S2} \approx \Delta^{S1}$ | Knowledge erased |
| $\Delta^{S2} < \Delta^{S1}$ | Knowledge retained (leaked) |
| $\Delta^{S2} > \Delta^{S1}$ | Over-erased |

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
  --unlearn_model simnpo \
  --num_examples 50 \
  --gpu 0
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--unlearn_model` | `simnpo` | Unlearning method |
| `--num_examples` | `50` | Examples to evaluate |
| `--delta_threshold` | `0.01` | Minimum Δ for LOST |
| `--layers` | `0-15` | Layers to patch |

See `patchscope/config.py` for available models.

### Output

Results saved to `runs/MMDD_HHMMSS_tf_{method}_layer/`:
- `run.log` - Per-example logs
- `results.json` - Detailed results
- `summary.json` - Aggregate UDR

---

## Citation

```bibtex
@article{tofu2024,
  title={TOFU: A Task of Fictitious Unlearning for LLMs},
  author={Maini, Pratyush and others},
  journal={arXiv preprint arXiv:2401.06121},
  year={2024}
}

@article{patchscopes2024,
  title={Patchscopes: A Unifying Framework for Inspecting Hidden Representations},
  author={Ghandeharioun, Asma and others},
  journal={arXiv preprint arXiv:2401.06102},
  year={2024}
}
```

---

## License

MIT License

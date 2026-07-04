# Unlearning Depth Score

[![Benchmark](https://img.shields.io/badge/Project-Page-blue)](https://gnueaj.github.io/unlearning-depth-score)
[![Paper](https://img.shields.io/badge/Paper-arxiv-b31b1b)](https://arxiv.org/abs/2605.24614)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FF9D00)](https://huggingface.co/datasets/jaeunglee/uds-annotated-tofu)

Official implementation of **"Measuring the Depth of LLM Unlearning via Activation Patching"**.

UDS quantifies the mechanistic depth of unlearning via two-stage activation patching, producing a per-example score from 0 (knowledge intact) to 1 (knowledge erased to the level of the retain model).

In our meta-evaluation across 20 metrics on 150 unlearned models spanning 8 methods, UDS achieves the highest faithfulness (AUC 0.971) and robustness (HM 0.932).

## 💡 How UDS works

<p align="center">
  <img src="docs/figs/pipeline.png" alt="UDS Pipeline" width="100%">
</p>

UDS measures whether forget set knowledge remains **recoverable** from internal representations through activation patching.

**Stage 1 (Baselining)** patches hidden states from the retain model into the full model. Large log-probability degradation on entity tokens indicates the layer encodes forget set knowledge. Layers exceeding a threshold (tau = 0.05) are identified as Knowledge-Encoding (KE) layers.

**Stage 2 (Quantification)** repeats patching with the unlearned model as source. If knowledge was erased, patching should degrade predictions as much as the retain model did in S1. The Layer Erasure Ratio (LER) measures this per layer, and UDS aggregates LER across KE layers weighted by S1 importance.

| UDS | Interpretation |
|-----|----------------|
| 1.0 | Knowledge erased to the level of the retain model |
| 0.0 | Knowledge fully intact (recoverable via patching) |

## 🔧 Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU.

## 🚀 Quick Start

Compute UDS for a single unlearned model:

```bash
python compute_uds.py \
  --unlearn_model simnpo_lr2e5_b35_a1_d1_g0125_ep5 \
  --gpu 0 \
  --batch_size 32
```

This downloads the required models from HuggingFace (`open-unlearning/` collection), runs two-stage activation patching on TOFU forget10 examples, and outputs per-example UDS scores to `runs/`.

You can also pass a HuggingFace model ID or local path directly:

```bash
python compute_uds.py \
  --unlearn_model open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b3.5_a1_d1_g0.125_ep5 \
  --gpu 0 \
  --batch_size 32
```

### Output

Results are saved to `runs/<model_name>/`:

| File | Description |
|------|-------------|
| `summary.json` | Model-level UDS score and experiment settings |
| `results.json` | Per-example UDS, KE layers, layer-wise deltas |
| `results_detailed.jsonl` | Full per-example details |
| `layer_details.csv` | Per-example, per-layer S1/S2 deltas |

### S1 Caching

Stage 1 (retain vs. full) is constant across unlearned models and is automatically cached. After the first run, subsequent models skip S1 computation entirely. Use `--no_s1_cache` to disable.

### Implementation

The core activation patching logic is in `uds/core.py`:

| Function | Purpose |
|----------|---------|
| `forward_with_patch()` | Runs the target model with a source model's hidden states patched at a specified layer |
| `get_hidden_at_position()` | Extracts hidden states from a model at specified positions |
| `probe_knowledge_with_patch()` | Computes patched log-probabilities for entity tokens |

The main script `compute_uds.py` orchestrates the full pipeline: data loading, model loading, batched S1/S2 computation, and result aggregation.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--unlearn_model` | (required) | Model alias or HuggingFace ID |
| `--full_model` | TOFU 1B full | Full model (target for patching) |
| `--retain_model` | TOFU 1B retain90 | Retain model (S1 source) |
| `--delta_threshold` | 0.05 | KE layer threshold |
| `--batch_size` | 1 | Batch size for patching |
| `--gpu` | 0 | GPU index |
| `--data_path` | `tofu_data/forget10_filtered.json` | Forget set with entity annotations |

## 📦 Dataset

`tofu_data/forget10_filtered.json` contains 367 examples from TOFU forget10 with entity span annotations:

```json
{
  "idx": 0,
  "question": "What is the full name of the author born in Taipei, Taiwan on ...?",
  "answer": "The author's full name is Hsiao Yun-Hwa.",
  "prefix": "The author's full name is",
  "entity": "Hsiao Yun-Hwa",
  "full_output": "Hsiao Yun-Hwa.",
  "entity_span": {"start": 6, "end": 12, "tokens": [39, 82, 23332, 55092, 11529, 10196]}
}
```

Each answer is partitioned into a **prefix** (contextual lead-in) and an **entity** (factual target). UDS evaluates log-probability degradation on entity tokens under teacher forcing.

## 🧪 Other Experiments

Scripts for reproducing the remaining paper experiments:

```bash
# Meta-evaluation: faithfulness (Section 4)
python scripts/meta_eval_faithfulness.py --gpu 0

# Meta-evaluation: robustness (Section 4)
python scripts/meta_eval_robustness.py --mode quant --gpu 0
python scripts/meta_eval_robustness.py --mode relearn --gpu 0

# Representation baselines: CKA, Logit Lens, Fisher Masked (Section 4.1)
python scripts/compute_representation_baselines.py --method cka --gpu 0
python scripts/compute_representation_baselines.py --method logit_lens --gpu 0
python scripts/compute_representation_baselines.py --method fisher_masked --gpu 0

# Scale validation (Section 3.4)
python scripts/run_uds_scale_sanity.py --gpu 0

# Component patching (Appendix D.1)
python scripts/s1_component_patching.py --gpu 0
```

## 📊 Benchmark Results

An interactive dashboard with full benchmark results across 150 unlearned models and 20 metrics is available at the [benchmark page](https://gnueaj.github.io/unlearning-depth-score).

## 🔗 Related Sources

- [TOFU](https://github.com/locuslab/tofu) - Task of Fictitious Unlearning benchmark
- [Open Unlearning](https://github.com/locuslab/open-unlearning) - Open-source framework for LLM unlearning methods and evaluation
- [Machine Unlearning Comparator](https://github.com/gnueaj/Machine-Unlearning-Comparator) - Visual analytics system for comparative evaluation of machine unlearning methods

## Citation

```bibtex
@misc{lee2026measuring,
  title={Measuring the Depth of LLM Unlearning via Activation Patching},
  author={Jaeung Lee and Dohyun Kim and Jaemin Jo},
  year={2026},
  eprint={2605.24614},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.24614}
}
```

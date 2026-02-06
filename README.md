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

UDS measures how deeply an unlearning method erases target knowledge from internal representations, not just output behavior.

#### Intuition

Output-based metrics can be fooled by superficial suppression. A model might refuse to answer while still storing recoverable knowledge in hidden states. UDS uses **activation patching** to probe whether knowledge persists at each layer.

#### Two-Stage Patching Protocol

We compare two patching experiments:
- **S1 (baseline)**: `retain → full` — How much knowledge does the retain model lack?
- **S2 (test)**: `unlearned → full` — How much knowledge does the unlearned model lack?

For each layer `l`, we patch hidden states from a source model into the full (fine-tuned) model and measure log-probability degradation.

#### Teacher Forcing Setup

```
Input: [prompt] + [reference answer tokens]
       "Question: What is X's profession?\nAnswer: X is a doctor"

For position t predicting token t+1:
  1. Run source model, capture hidden state h_source at layer l
  2. Run target model, inject h_source at layer l via hook
  3. Measure log-prob of correct next token
```

- Prompt includes BOS token via `add_special_tokens=True`
- Reference tokens use ground truth answer (GT) continuation
- Patching scope: `span` (all answer positions) or `boundary` (first token only)

#### Entity Span Evaluation

Instead of evaluating the full answer, we focus on the **entity span** containing the core knowledge:

```
Full answer: "X is a renowned doctor specializing in..."
Entity span: "doctor"  ← evaluation focused here
```

The entity span is located in the reference text using tokenizer offset mapping (fast tokenizers) or subsequence matching.

#### Log-Probability Delta (Δ)

For each layer `l`, compute the degradation when patching:

```
Δ_l = logprob_full - logprob_patched
```

- `logprob_full`: Mean log-prob of entity tokens without patching
- `logprob_patched`: Mean log-prob after injecting source hidden states

A large Δ means the source model lacks the knowledge needed at that layer.

#### FT Layer Identification

Layers where fine-tuning stored target knowledge are identified by S1:

```
FT_layers = { l : Δ^S1_l > τ }
```

If patching from retain causes significant degradation, that layer contains knowledge the retain model doesn't have (i.e., fine-tuned knowledge).

Default threshold: `τ = 0.05`

#### UDS Formula

For each example `i`, UDS measures erasure depth across FT layers:

```
UDS_i = Σ_{l∈FT} [ Δ^S1_l × clip(Δ^S2_l / Δ^S1_l, 0, 1) ] / Σ_{l∈FT} Δ^S1_l
```

- **Numerator**: Sum of weighted erasure ratios (clipped to [0,1])
- **Denominator**: Total S1 signal (normalization)
- **clip()**: Prevents ratios > 1.0 (over-erasure) or < 0.0 (negative)

#### Interpretation

| UDS Value | Meaning |
|-----------|---------|
| 1.0 | Perfect erasure — unlearned model shows same degradation as retain |
| 0.5 | Partial erasure — knowledge partially recoverable |
| 0.0 | No erasure — knowledge fully intact despite output suppression |

Higher UDS = deeper, more robust unlearning.

#### Script

```bash
python exp_s1_teacher_forcing.py \
  --unlearn_model <model_name> \
  --metric logprob \
  --em_scope entity \
  --patch_scope span \
  --delta_threshold 0.05 \
  --batch_size 32
```

Key parameters:
- `--metric logprob`: Use log-probability (default, recommended)
- `--em_scope entity`: Focus on entity span (not full answer)
- `--patch_scope span`: Patch all answer positions
- `--delta_threshold 0.05`: τ for FT layer identification

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

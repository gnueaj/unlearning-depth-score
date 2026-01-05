# CLAUDE.md - AI Assistant Guide

This document provides context for AI assistants (Claude, GPT, etc.) working on this codebase.

## Project Overview

**Activation Patching for Unlearning Audit** is a white-box analysis tool for auditing LLM unlearning via hidden state patching. It answers: "Does an unlearned model truly forget, or just hide knowledge?"

## Key Concepts

### 1. Activation Patching
- **Source Model**: Unlearned model (e.g., SimNPO, IdkNLL)
- **Target Model**: Original model with full knowledge
- **Patching**: Replace Target's hidden state at layer L with Source's hidden state
- **Goal**: Detect if Source still encodes knowledge that Target can decode

### 2. Forced Prefix Mode
To ensure fair comparison, we use the same prompt for both models:
```
Question: What is the full name of the author born in Taipei?
Answer: The author's full name is
                                 ↑ Extract hidden state here
```

### 3. IDK Detection
For IDK-style methods (IdkNLL, IdkDPO), we detect refusal responses:
```python
is_idk_response = any(idk in response.lower() for idk in [
    "i don't know", "i'm not sure", "i cannot", ...
])
```
When IDK detected → use no-prefix mode (let IDK flow naturally)
When non-IDK → use GT prefix mode (force answer generation)

## File Structure

```
patchscope/
├── __init__.py       # Package exports
├── config.py         # Configuration classes + UNLEARN_MODELS registry
├── core.py           # Core patching functions (get_hidden, patch, probe)
├── models.py         # Model loading utilities
├── probes.py         # Probe builders (QA, cloze, choice)
├── run.py            # Main CLI entry point
├── tofu_entities.py  # TOFU-specific entity extraction
└── utils.py          # Seed, mkdir, layer parsing
```

## Critical Functions

### `core.py`

```python
# Extract hidden state from model at specific layer
get_generated_answer_hidden(model, tokenizer, question, layer_idx, forced_prefix=...)
→ Returns: (hidden_tensor [1, D], metadata dict)

# Run forward pass with patched hidden state
forward_with_patch(model, input_ids, attention_mask, patch_layer_idx, patch_vector)
→ Returns: logits [B, T, V]

# Full knowledge probe with patching
probe_knowledge(model, tokenizer, probe_prompt, expected_answer, patch_layer_idx, patch_vector)
→ Returns: dict with topk, probs, full_response, knowledge_detected
```

### `run.py`

```python
# Main analysis function
run_patchscope(config: PatchscopeConfig)

# Build forced prefix prompt
_build_forced_prefix(question, prefix) → "Question: ...?\nAnswer: {prefix}"
```

## Configuration

### Model Selection
```python
from patchscope.config import UNLEARN_MODELS, get_model_id

# Short names
get_model_id("simnpo")  → "open-unlearning/unlearn_tofu_..._SimNPO_..."
get_model_id("idknll")  → "open-unlearning/unlearn_tofu_..._IdkNLL_..."

# Full registry in config.py UNLEARN_MODELS dict
```

### Layer Specification
```python
from patchscope.utils import parse_layers

parse_layers("0,1,2,3", n_layers=16)  → [0, 1, 2, 3]
parse_layers("0-15", n_layers=16)     → [0, 1, ..., 15]
parse_layers("0-15:2", n_layers=16)   → [0, 2, 4, ..., 14]
```

## Common Tasks

### Adding a New Unlearning Model
1. Add to `UNLEARN_MODELS` in `config.py`:
```python
"newmethod": "open-unlearning/unlearn_tofu_..._NewMethod_...",
```

### Modifying Prompt Format
Edit `_build_forced_prefix()` in `run.py` or modify `build_qa_probe()` in `probes.py`.

### Adding IDK Detection Phrases
Edit the `is_idk_response` detection in `run.py`:
```python
is_idk_response = any(idk in source_gen_lower for idk in [
    "i don't know", "i'm not sure", ...  # Add new phrases here
])
```

### Changing Entity Extraction
Edit `tofu_entities.py`:
- `detect_question_type()` - classify question
- `extract_*_entity()` - type-specific extraction

## Architecture Details

### Llama-3.2-1B-Instruct
- 16 transformer layers (indices 0-15)
- Hidden dimension: 2048
- Layer access: `model.model.layers[idx]`

### Hook-Based Patching
We use `register_forward_hook` for activation patching:
```python
def hook_fn(module, inputs, output):
    hs = output[0].clone()
    hs[:, position, :] = patch_vector
    return (hs,) + output[1:]
```

## Testing & Debugging

### Sanity Check (Source = Target)
```bash
python -m patchscope.run --debug
```
Expected: All layers show correct answer (knowledge preserved through patching)

### Quick Test
```bash
python -m patchscope.run --preset quick  # 3 layers, 1 example
```

### Full Analysis
```bash
python -m patchscope.run --source_model simnpo --layers "0-15" --num_examples 5
```

## Known Issues & Workarounds

1. **Repeated patterns in output**: We cut off at "Question:", "Answer:" patterns
2. **Long outputs truncated**: Max 75 chars for display
3. **Different tokenizers**: Always use Target model's tokenizer for both

## Metrics Interpretation

| Metric | Meaning |
|--------|---------|
| `detection_rate` | % of layers where correct answer appears in top-k or response |
| `max_correct_prob` | Highest P(correct token) across all layers |
| `prob_margin` | P(correct) - P(most likely wrong) |
| `layers_with_knowledge` | List of layers where knowledge detected |

## Research Context

### Unlearning Methods Taxonomy
1. **Knowledge-modifying**: SimNPO, NPO, GradDiff - attempt to alter stored facts
2. **Output-suppressing**: IdkNLL, IdkDPO - train to refuse without removing knowledge

### Key Finding
Most methods achieve behavioral unlearning (model says wrong thing / IDK) but fail hidden-state unlearning (knowledge still encoded). Activation patching exposes this gap.

## Future Work Directions
- [ ] Meta-evaluation integration with open-unlearning benchmark
- [ ] Cross-model patching (different architectures)
- [ ] Automated unlearning quality scoring
- [ ] Adversarial robustness testing

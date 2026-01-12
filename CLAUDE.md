# CLAUDE.md - AI Assistant Guide

This document provides context for AI assistants (Claude, GPT, etc.) working on this codebase.

## Project Overview

**Activation Patching for Unlearning Audit** is a white-box analysis tool for auditing LLM unlearning via hidden state patching. It answers: "Does an unlearned model truly forget, or just hide knowledge?"

## Model Architecture

### Base Model: Llama-3.2-1B-Instruct
- **16 transformer layers** (indices 0-15)
- **Hidden dimension**: 2048
- **Layer access**: `model.model.layers[idx]`

### TOFU Models (open-unlearning)
| Model | HuggingFace ID |
|-------|----------------|
| Full (trained on all data) | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` |
| Retain90 (trained without forget10) | `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90` |
| SimNPO | `open-unlearning/unlearn_tofu_..._SimNPO_...` |
| IdkNLL | `open-unlearning/unlearn_tofu_..._IdkNLL_...` |
| GradDiff | `open-unlearning/unlearn_tofu_..._GradDiff_...` |

## Unlearning Methods

### 1. Knowledge-Modifying Methods
목표: 저장된 지식 자체를 변경하여 잊어버리게 함

| Method | Description | Effectiveness |
|--------|-------------|---------------|
| **SimNPO** | Simple Negative Preference Optimization | High FQ (~0.5) |
| **NPO** | Negative Preference Optimization | Medium |
| **GradDiff** | Gradient Difference | Very Low (FQ ≈ 0) |

### 2. Output-Suppressing Methods
목표: 지식은 유지하되 출력만 억제 ("I don't know"로 응답)

| Method | Description | Effectiveness |
|--------|-------------|---------------|
| **IdkNLL** | IDK Negative Log-Likelihood | Behavior OK, Hidden State Leaks |
| **IdkDPO** | IDK Direct Preference Optimization | Similar to IdkNLL |

### 3. Hyperparameters
**Alpha (α)**: Retain loss weight - retain set에 대한 손실 가중치
- 높은 α: retain 지식 보존 강화, 언러닝 효과 약화
- 낮은 α: 더 공격적인 언러닝, catastrophic forgetting 위험

```python
# IdkNLL loss formula
L_total = L_forget(IDK) + α * L_retain
```

**IdkNLL variants:**
- `idknll` (default): lr=4e-5, α=5, ep=10
- `idknll_lr1e5_a1_ep5`: lr=1e-5, α=1, ep=5 (낮은 retain weight)

### 4. FQ (Forget Quality) Metric
- Open-Unlearning benchmark에서 사용하는 지표
- 언러닝 모델 vs Retrain 모델 출력 분포 비교
- 1.0 = 완벽한 언러닝, 0 = 언러닝 실패
- GradDiff FQ ≈ 3.6e-9 (사실상 0)

## Key Concepts

### 1. Activation Patching
- **Source Model**: Unlearned model (e.g., SimNPO, IdkNLL)
- **Target Model**: Original model with full knowledge (Full)
- **Patching**: Replace Target's hidden state at layer L with Source's hidden state
- **Goal**: Detect if Source still encodes knowledge that Target can decode

### 2. Forced Prefix Mode (Manual Prefix)
수동으로 검증된 prefix를 사용하여 정답 entity 직전까지 포함:
```
Question: What is the full name of the author born in Taipei?
Answer: The author's full name is
                                 ↑ Extract hidden state here (last token position)
Entity: "Hsiao Yun-Hwa" (평가 대상)
```

**Manual Prefix를 사용하는 이유:**
1. 자동 3단어 추출시 entity가 포함되는 경우 방지
2. 의미적으로 완전한 prefix 보장 (예: "The author's full" → "The author's full name is")
3. `generate_manual_prefixes.py`에서 400개 모두 수동 검증

**파일:**
- `tofu_data/forget10_prefixes_manual.json`: 400개 수동 prefix/entity
- `tofu_data/forget10_filtered_v3.json`: 필터링 + 수동 prefix (353개)

### 3. EM Score (Exact Match) - Open-Unlearning Style
Position-wise token match ratio:
```python
EM = (# tokens matching at same position) / (# reference tokens)
# Example: [A,X,C,D,E] vs [A,B,C,D,E] → EM = 4/5 = 0.8
```
- **Threshold**: EM >= 0.5 → Knowledge Present (KEPT)
- **Below threshold**: EM < 0.5 → Knowledge Lost (LOST)

## Dataset Preprocessing

### TOFU forget10 Dataset (400 examples)
평가에 부적합한 질문을 수동 검증하여 필터링합니다.

### 제외 기준 (수동 검증 완료)
1. **Full Model Wrong** (30개, 7.5%): Full 모델이 GT와 **의미적으로** 불일치
   - 다른 책 이름, 다른 부모 직업, 다른 인물 등
   - 예: GT="Engineering Leadership" vs Full="The Vermilion Enigma"

2. **General Knowledge** (17개, 4.3%): Retain 모델이 GT와 **의미적으로** 일치
   - Retain 모델도 정답을 알고 있음 = forget-set 특화 지식이 아님
   - 예: "What language does X write in?" → "English" (일반 상식)

### 필터링 결과
| Category | Count | Percentage |
|----------|-------|------------|
| Full Model Wrong | 30 | 7.5% |
| General Knowledge | 17 | 4.3% |
| **Valid for Evaluation** | **353** | **88.2%** |

### 데이터 버전
| Version | Description | File |
|---------|-------------|------|
| v2 | 필터링만 적용 (자동 prefix) | `forget10_filtered_v2.json` |
| **v3** | **필터링 + 수동 prefix** | `forget10_filtered_v3.json` ✓ |

### 생성 파일 (tofu_data/)
- `forget10_filtered_v3.json`: **현재 사용** - 수동 prefix + 필터링 (353개)
- `forget10_prefixes_manual.json`: 400개 수동 prefix/entity
- `forget10_full_wrong_v2.json`: Full 모델 오답 + `full_output` 필드
- `forget10_general_knowledge_v2.json`: General Knowledge + `retain_output` 필드

### 전처리 스크립트
```bash
python scripts/preprocess_forget10_v2.py       # v2 데이터 생성
python scripts/generate_manual_prefixes.py     # 수동 prefix 생성
# v3는 위 두 결과를 결합하여 생성
```

하드코딩된 인덱스:
```python
FULL_WRONG_IDX = {5, 49, 88, 134, 162, 199, 234, 238, 243, 256, ...}  # 30개
GK_IDX = {17, 27, 28, 59, 86, 87, 92, 116, 119, 121, 127, ...}  # 17개
```

## EM-Based Evaluation Framework (exp_em_eval.py)

### Two-Stage Patching
```
Stage 1 (S1): Retain → Full  - Find layers where forget-set knowledge is encoded
Stage 2 (S2): Unlearn → Full - Measure how much knowledge is erased per layer
```

**S1의 의미**: Retain 모델은 forget-set을 학습하지 않았으므로, S1에서 LOST인 레이어 = Full 모델이 fine-tuning으로 학습한 지식이 저장된 레이어

**S2의 의미**: 언러닝 모델의 hidden state로 패칭했을 때, S2가 LOST면 해당 레이어의 지식이 삭제됨

### Metrics
| Metric | Description |
|--------|-------------|
| **UDR (Unlearning Depth Rate)** | Erased layers / FT layers (per example) |
| **Retention** | Average S2 EM on FT layers (lower = better erasure) |

### Erasure Quality Categories
- **Over-erased**: S2 LOST > S1 LOST (collateral damage, 과도한 삭제)
- **Exact-erased**: S2 LOST = S1 LOST (ideal unlearning)
- **Under-erased**: S2 LOST < S1 LOST (knowledge leaked, 지식 누출)
- **General Knowledge**: S1 all KEPT (not forget-set specific)

Percentages are calculated excluding General Knowledge examples.

### Running Experiments
```bash
# Single GPU
python exp_em_eval.py --unlearn_model simnpo --num_examples 50

# Parallel on different GPUs
CUDA_VISIBLE_DEVICES=0 python exp_em_eval.py --unlearn_model simnpo --num_examples 50 &
CUDA_VISIBLE_DEVICES=1 python exp_em_eval.py --unlearn_model idknll --num_examples 50 &
```

Output folder format: `runs/MMDD_HHMMSS_{method}/`

## File Structure

```
├── patchscope/
│   ├── __init__.py       # Package exports
│   ├── config.py         # Configuration classes + UNLEARN_MODELS registry
│   ├── core.py           # Core patching functions (get_hidden, patch, probe)
│   ├── models.py         # Model loading utilities
│   ├── probes.py         # Probe builders (QA, cloze, choice)
│   ├── run.py            # Main CLI entry point
│   ├── tofu_entities.py  # TOFU-specific entity extraction
│   └── utils.py          # Seed, mkdir, layer parsing
├── scripts/
│   └── preprocess_forget10_v2.py  # Dataset preprocessing (수동 검증)
├── tofu_data/
│   ├── forget10_filtered_v2.json       # Valid examples (353)
│   ├── forget10_full_wrong_v2.json     # Full model wrong (30)
│   └── forget10_general_knowledge_v2.json  # General knowledge (17)
└── exp_em_eval.py        # Main experiment script
```

## Critical Functions

### `core.py`
```python
# Extract hidden states from all layers at once
get_all_layers_hidden(model, input_ids, attention_mask, layer_list, position=-1)
→ Returns: {layer_idx: hidden_tensor [1, D]}

# Run forward pass with patched hidden state
forward_with_patch(model, input_ids, attention_mask, patch_layer_idx, patch_vector)
→ Returns: logits [B, T, V]

# Generate with patching
generate_with_patch(model, tokenizer, prompt, layer, hidden, max_new_tokens=20)
→ Returns: generated text
```

### Hook-Based Patching
```python
def hook_fn(module, inputs, output):
    hs = output[0].clone()
    hs[:, position, :] = patch_vector
    return (hs,) + output[1:]
```

## Configuration

### Model Selection
```python
from patchscope.config import UNLEARN_MODELS, get_model_id

# Short names
get_model_id("simnpo")  → "open-unlearning/unlearn_tofu_..._SimNPO_..."
get_model_id("idknll")  → "open-unlearning/unlearn_tofu_..._IdkNLL_..."
get_model_id("graddiff") → "open-unlearning/unlearn_tofu_..._GradDiff_..."
get_model_id("graddiff_lr5e5_a2_ep10") → Strong GradDiff variant
```

### Layer Specification
```python
from patchscope.utils import parse_layers

parse_layers("0,1,2,3", n_layers=16)  → [0, 1, 2, 3]
parse_layers("0-15", n_layers=16)     → [0, 1, ..., 15]
parse_layers("0-15:2", n_layers=16)   → [0, 2, 4, ..., 14]
```

## Research Context

### Key Finding
Most methods achieve behavioral unlearning (model says wrong thing / IDK) but fail hidden-state unlearning (knowledge still encoded). Activation patching exposes this gap.

### Method Comparison (50 examples, v3 filtered dataset)
| Method | Hyperparams | UDR ↑ | Retention ↓ | Over-erased | Under-erased |
|--------|-------------|-------|-------------|-------------|--------------|
| **SimNPO** | default | **64.2%** | 33.8% | 44.7% | 34.2% |
| IdkNLL | lr1e-5, α=1, ep5 | 23.1% | 70.9% | 21.1% | 68.4% |
| GradDiff (weak) | lr1e-5, α=5, ep10 | 22.6% | 70.6% | 18.4% | 78.9% |
| **GradDiff (strong)** | lr5e-5, α=2, ep10 | **78.0%** | **22.8%** | 78.9% | 13.2% |

**해석:**
- **SimNPO**: 균형잡힌 erasure - UDR 높고 over/under-erased 비율 유사
- **IdkNLL (α=1)**: 낮은 retain weight로 catastrophic forgetting → 지식 검출 자체가 어려움
- **GradDiff weak**: 대부분 지식 누출 (78.9% under-erased)
- **GradDiff strong**: 가장 높은 UDR이나 collateral damage 심함 (78.9% over-erased)

## Future Work Directions
- [ ] Meta-evaluation integration with open-unlearning benchmark
- [ ] Cross-model patching (different architectures)
- [ ] Automated unlearning quality scoring
- [ ] Adversarial robustness testing

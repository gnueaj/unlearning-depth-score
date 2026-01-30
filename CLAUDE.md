# CLAUDE.md - AI Assistant Guide

This document provides context for AI assistants (Claude, GPT, etc.) working on this codebase.

## Project Overview

**Activation Patching for Unlearning Audit** is a white-box analysis tool for auditing LLM unlearning via hidden state patching. It answers: "Does an unlearned model truly forget, or just hide knowledge?"

## Environment

- **Available GPUs**: 0, 1 (GPU 2 does not exist)
- **IMPORTANT**: Use `--gpu 0` or `--gpu 1` argument to select GPU (NOT `CUDA_VISIBLE_DEVICES`)
- The script internally sets `CUDA_VISIBLE_DEVICES` based on `--gpu` argument

### GPU Distribution Rules (IMPORTANT)
When running multiple experiments in parallel:
1. **Use --gpu argument**: `python exp_s1_teacher_forcing.py --gpu 0` or `--gpu 1`
2. **Always alternate GPUs**: Assign experiments to GPU 0 and GPU 1 evenly
3. **Pair structure**: Run experiments in pairs (GPU 0 + GPU 1) and wait for both to complete before starting the next pair
4. **Example for 9 experiments**:
   ```bash
   # Pair 1
   python exp_s1_teacher_forcing.py --unlearn_model model1 --gpu 0 > log1.log 2>&1 &
   python exp_s1_teacher_forcing.py --unlearn_model model2 --gpu 1 > log2.log 2>&1 &
   wait
   # Pair 2 ...
   ```
5. **Do NOT stack multiple experiments on the same GPU** - this causes OOM errors
6. **Wait for completion**: Always `wait` before starting the next pair

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

### 2. Patching Components: Layer vs MLP

#### Llama 3.2 레이어 구조 (Pre-LN)
```
h_attn = x + Attention(RMSNorm(x))       # Attention sublayer + residual
h_new  = h_attn + MLP(RMSNorm(h_attn))   # MLP sublayer + residual
```

#### 각 컴포넌트의 역할
| Component | Role |
|-----------|------|
| **MLP** | Factual knowledge 저장소 |
| **Attention** | 문맥 기반 정보 조합, 어떤 지식을 불러올지 결정하는 조회 메커니즘 |

**MLP가 Factual Knowledge 저장소인 이유:**
- MLP는 2-layer feed-forward network로, 입력과 무관하게 고정된 weight를 가짐
- 학습 시 (subject, relation) → object 형태의 factual association이 MLP weight에 인코딩됨
- Attention은 "어떤 토큰에서 정보를 가져올지"를 결정하는 동적 메커니즘
- 연구 근거: Meng et al. (2022) "Locating and Editing Factual Associations in GPT", Geva et al. (2021) "Transformer FFN Layers Are Key-Value Memories"

#### 패칭 방식 비교
| Patching | What's Patched | Target Model 상태 |
|----------|---------------|-------------------|
| **Layer (Full)** | 전체 레이어 output | Attention 결과도 Source 것으로 대체됨 |
| **MLP Only** | MLP output만 | Target의 Attention 유지됨 |

#### General Knowledge의 정의
S1 패칭 (Retain → Full)에서 **모든 레이어에서 패칭 후에도 정답을 맞추는** 예제를 general knowledge로 분류한다.
- 의미: Retain 모델의 hidden state로 패칭해도 Full 모델이 정답을 맞춤
- 해석: 해당 지식은 forget-set 특화 지식이 아니라 일반 지식 (Retain도 알고 있음)

**General Knowledge와 패칭 민감도:**
- General knowledge가 많을수록 "forget-set 지식이 저장된 레이어"를 특정할 수 있는 예제 수가 줄어듦
- MLP 패칭이 Layer 패칭보다 general knowledge 비율이 낮음 → 더 민감한 검출
- General knowledge 예제의 S2 평가 포함/제외 여부는 분석 목적에 따라 결정

#### 왜 MLP 패칭이 더 민감한가?

**정보 전달 관점:**
| 패칭 방식 | 전달되는 정보 | 테스트 대상 |
|----------|-------------|------------|
| **MLP 패칭** | Retain의 **factual knowledge만** | 순수 지식 유무 |
| **Layer 패칭** | Retain의 **factual + attention** | 지식 + 문맥 처리 |

**MLP 패칭이 더 민감한 이유:**
- MLP는 factual knowledge 저장소이므로, MLP 패칭은 **순수하게 지식 유무만 테스트**
- Retain MLP에 forget-set 지식이 없으면 → 패칭 시 해당 지식이 명확히 누락 → LOST
- Layer 패칭은 attention까지 전달하므로, Retain의 attention이 지식 부재 신호를 **희석**시킬 수 있음
- 결과: MLP 패칭이 forget-set 지식 부재를 더 민감하게 검출 → general knowledge 감소

#### 실험 결과 (50 examples, SimNPO)
| Patching | General Knowledge | Evaluated (n) | 검출 민감도 |
|----------|-------------------|---------------|------------|
| Layer | 15 (30%) | 35 | 낮음 |
| MLP | 10 (20%) | 40 | **높음** |

#### 언제 어떤 패칭을 사용해야 하는가?
| Task | 권장 Patching | 이유 |
|------|--------------|------|
| **Unlearning Audit (Factual Knowledge)** | **MLP** | 지식 저장소만 검사, Target의 조회 능력 유지 |
| In-context Learning 분석 | Layer/Attention | 문맥 처리 능력이 중요 |
| Long-range Dependency 분석 | Layer/Attention | 정보 조합 능력이 중요 |

**권장사항**: Unlearning audit에는 **MLP 패칭**을 사용 (더 민감한 forget-set 지식 검출)

### 3. Forced Prefix Mode (Manual Prefix)
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
3. 400개 모두 수동 검증

**v7_gt에서의 변경 (현재 사용):**
- **GT를 reference로 사용** (Full model output 대신)
- prefix를 GT answer 기준으로 재검증
- entity를 GT answer에서 정확히 추출 (fuzzy matching)
- `reference_type`: "gt" 명시

**파일:**
- `scripts/create_v7_gt_reference.py`: v6 → v7 변환 스크립트
- `tofu_data/forget10_filtered_v7_gt.json`: **현재 사용** - GT reference 데이터 (367개)

### 4. Evaluation Metrics

#### 4.1 Log-Prob 기반 메트릭 (권장, 기본값)
Reference token의 log-probability 변화로 지식 보유 여부 측정:

```python
# Full baseline (패칭 없음)
full_score = mean(log P(ref_token | context))

# Patched score (Source → Target 패칭)
patched_score = mean(log P(ref_token | context, patched))

# Delta: 패칭으로 인한 확률 감소
Δ = full_score - patched_score
# Δ > 0: 패칭이 reference 확률을 낮춤 → 지식 손실 신호
```

**Status 결정:** `Δ > delta_threshold` → LOST, else → KEPT

**UDR (Log-prob 기반, 단일 지표):**
```python
UDR = sum(s2_delta where > threshold) / sum(s1_delta where > threshold)
# clip to [0, 1]
```

#### 4.2 EM 기반 메트릭 (비교용)
```python
EM = (# tokens matching in entity span) / (# entity tokens)
# Threshold: EM >= threshold → KEPT, else → LOST
```

#### 4.3 Patch Scope 옵션
| Scope | 패칭 범위 | 설명 |
|-------|----------|------|
| `boundary` (기본) | `[start]` | 마지막 prompt token만 패칭 |
| `span` | `[start:end]` | Reference 전체 span 패칭 |

```
boundary: [prompt] [ref_1] [ref_2] ... [ref_n]
               ↑ patch only here

span:     [prompt] [ref_1] [ref_2] ... [ref_n]
                   ↑_________ patch _________↑
```

#### 4.4 Teacher Forcing
- 각 position에서 reference token을 입력으로 주고 예측 평가
- Autoregressive 생성의 첫 토큰 오류 전파 문제 회피

## Dataset Preprocessing

### TOFU forget10 Dataset (400 examples)
평가에 부적합한 질문을 수동 검증하여 필터링합니다.

### v7_gt 데이터셋 (현재 사용)

#### 핵심 변경사항 (v6 → v7_gt)
1. **GT를 reference로 사용** (Full output 대신)
2. **prefix를 GT answer 기준으로 재검증** (24개 수정)
3. **entity를 GT answer에서 정확히 추출** (fuzzy matching으로 오류 수정)

#### 왜 GT를 reference로 사용하는가?
Log-prob 메트릭은 GT answer token의 생성 확률을 직접 측정하므로, 모델의 실제 생성 표현과 무관하게 factual knowledge 보유 여부를 평가할 수 있다. 예를 들어 Full 모델이 "Yun-Hwa Hsiao"라고 생성해도 GT인 "Hsiao Yun-Hwa"의 log-prob가 높으면 해당 지식을 보유한 것으로 판단한다. 이는 TOFU benchmark의 forget quality 평가 방식과 일치하며, S1/S2 패칭에서 동일한 GT reference를 사용하여 비교 가능성을 보장한다.

#### 데이터 필드
```python
{
    "idx": 0,                           # TOFU 원본 인덱스
    "question": "What is...",           # 질문
    "answer": "The author's...",        # GT 정답 (reference로 사용)
    "prefix": "The author's full name is",  # GT 기준 prefix (평가용)
    "full_prefix": "...",               # v6의 Full 기준 prefix (보존)
    "entity": "Hsiao Yun-Hwa",          # GT entity (평가용)
    "gt_entity": "Hsiao Yun-Hwa",       # = entity
    "gt_entity_orig": "Hsiao Yun-Hwa",  # v6의 원래 gt_entity (비교용)
    "full_entity": "Hsiao Yun-Hwa",     # Full 모델의 entity (비교용)
    "full_output": "Hsiao Yun-Hwa, born...", # Full 모델 생성 결과
    "match_type": "exact",              # exact/partial/diff (메타정보)
    "reference_type": "gt",             # 데이터셋 버전 표시
    "entity_span": {"start": 6, "end": 12, "tokens": [...]}  # 토큰 위치 (미래용)
}
```

#### 제외 기준: Full Model Wrong (33개, 8.25%)
Full 모델이 GT와 **의미적으로** 완전히 다른 답변을 생성하는 경우:

| Category | Count | Examples |
|----------|-------|----------|
| 다른 책 이름 | 13 | GT="Venom in the Veins" → Full="Venetian Vendetta" |
| 다른 부모 직업 | 8 | GT=father=Paramedic → Full=father=librarian |
| 다른 날짜/숫자 | 3 | GT=June 9, 1951 → Full=16th May 1981 |
| 다른 인물 이름 | 3 | GT=Xin Lee Williams → Full=Zhen Xu |
| Yes/No 불일치 | 6 | GT=Yes → Full=No |

#### Full Wrong 인덱스 (33개)
```python
FULL_WRONG_V6 = {
    # 책 이름: 5, 23, 52, 66, 103, 164, 168, 206, 234, 238, 344, 353, 386
    # 부모 직업: 22, 42, 102, 144, 163, 202, 281, 343
    # 날짜/숫자: 61, 128, 148
    # 인물 이름: 220, 300, 320
    # Yes/No: 68, 91, 169, 170, 172, 378
}
```

#### 필터링 결과
| Category | Count | Percentage |
|----------|-------|------------|
| Full Model Wrong | 33 | 8.25% |
| **Valid for Evaluation** | **367** | **91.75%** |

#### Match Type 분포 (367개)
| Type | Count | Description |
|------|-------|-------------|
| exact | 95 | GT entity = Full entity (완전 일치) |
| partial | 176 | GT ⊂ Full 또는 Full ⊂ GT (부분 일치) |
| diff | 96 | 다르지만 의미적으로 동일 (표현 차이) |

### Quote Normalization
TOFU 데이터는 curly quotes를 사용하므로 prefix 매칭 시 정규화 필요:
```python
def normalize_quotes(text: str) -> str:
    """Normalize curly quotes to straight quotes."""
    text = text.replace(chr(8216), chr(39))  # ' → '
    text = text.replace(chr(8217), chr(39))  # ' → '
    text = text.replace(chr(8220), chr(34))  # " → "
    text = text.replace(chr(8221), chr(34))  # " → "
    return text
```

### 데이터 버전 히스토리
| Version | Description | File | Examples |
|---------|-------------|------|----------|
| v2 | 필터링만 (자동 prefix) | `forget10_filtered_v2.json` | 353 |
| v3 | 필터링 + 수동 prefix | `forget10_filtered_v3.json` | 353 |
| v6 | Full entity 기준 + 수동 prefix | `forget10_filtered_v6.json` | 367 |
| **v7_gt** | **GT reference + prefix 재검증** | `forget10_filtered_v7_gt.json` ✓ | **367** |

### v7_gt 생성 파일 (tofu_data/)
- `forget10_filtered_v7_gt.json`: **현재 사용** - GT reference 데이터 (367개)
- `forget10_filtered_v6.json`: 이전 버전 (Full reference)

### v7_gt 전처리 스크립트
```bash
# v6 → v7_gt 변환 (GT reference 기준)
python scripts/create_v7_gt_reference.py
# → tofu_data/forget10_filtered_v7_gt.json (367개)
```

#### v6 → v7_gt 변경사항 (24개 예제)
- **21개 Yes/No 질문**: prefix를 빈 문자열로 변경 (GT가 "Yes, ..."로 시작)
- **3개 책 제목 차이**: GT answer의 전체 책 제목으로 prefix 수정
- **2개 entity 오류 수정**: idx=330, 350 (fuzzy matching으로 자동 수정)

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

### Running Experiments (exp_s1_teacher_forcing.py)

**CLI 옵션:**
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--metric {em,logprob}` | `logprob` | 메트릭 선택 |
| `--delta_threshold` | `0.02` | Log-prob Δ 임계값 (τ) |
| `--patch_scope {span,boundary}` | `boundary` | 패칭 범위 |
| `--em_scope {full,entity}` | `entity` | 평가 범위 |
| `--entity_source {gt,full}` | `gt` | Entity 출처 |
| `--reference {gt,full}` | `gt` | Reference 텍스트 (GT answer vs Full output) |
| `--data_path` | `v7_gt.json` | 데이터셋 경로 |
| `--mode {layer,mlp}` | `layer` | 패칭 모드 |
| `--em_threshold` | `1.0` | EM 임계값 |
| `--em_type {token,exact}` | `token` | EM 계산 방식 |
| `--log_mismatch` | False | 토큰 불일치 로깅 |
| `--log_span` | False | Entity/eval span 토큰 위치 로깅 |

**로그/요약 출력 (현재 기본 log-prob 모드):**
- per-layer: `logp`, `Δ` 표시
- summary: `Average UDR`만 출력 (UDR 소수점 3자리)

**실험 규칙(현재 기준):** 전체 367개 예제 사용

```bash
# GT reference 기반 (기본값, 권장)
python exp_s1_teacher_forcing.py \
  --unlearn_model simnpo \
  --gpu 0

# 병렬 실행 (GPU 분할, --gpu 옵션 사용!)
python exp_s1_teacher_forcing.py --unlearn_model simnpo --gpu 0 > logs/simnpo.log 2>&1 &
python exp_s1_teacher_forcing.py --unlearn_model idknll --gpu 1 > logs/idknll.log 2>&1 &
wait
```

Output folder format: `runs/MMDD_HHMMSS_tf_{method}_{mode}/`

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
│   ├── manual_prefix_v6.py           # 400개 수동 prefix/entity 매핑
│   ├── create_v6_from_scratch.py     # Full 모델 출력 생성
│   ├── create_final_v6.py            # Full Wrong 필터링, 최종 v6 생성
│   └── create_v7_gt_reference.py     # v6 → v7_gt 변환 (GT reference)
├── tofu_data/
│   ├── forget10_filtered_v7_gt.json  # ✓ 현재 사용 (367개, GT reference)
│   ├── forget10_filtered_v6.json     # 이전 버전 (Full reference)
│   └── forget10_v6_full_wrong.json   # Full Wrong 목록 (33개)
└── exp_s1_teacher_forcing.py         # Main experiment script
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

### v7_gt Evaluation Framework (현재)
- **Dataset**: 367 examples (400 - 33 Full Wrong)
- **Metric**: Log-prob (기본) or Teacher Forcing EM on entity span
- **Reference**: GT answer (기본, `--reference gt`)
- **Patching**: Layer or MLP patching
- **Threshold (τ)**: 0.02 (기본)

## Evaluation Methodology: Teacher Forcing EM

### Why Teacher Forcing?
- **Autoregressive 생성의 문제**: 첫 토큰이 틀리면 이후 전체가 달라짐
- **Teacher Forcing**: 각 position에서 reference token을 입력으로 주고, 해당 position의 예측만 평가
- **장점**: Position-independent한 token-level accuracy 측정 가능

### Entity-scope EM
```
Prompt: "Question: What is the full name of the author born in Taipei?\nAnswer: The author's full name is"
Entity: "Hsiao Yun-Hwa" (Full model's output)

Input sequence:  [prompt_tokens] + [Hs][iao][ Yun][-H][wa]
                                   ↑   ↑    ↑    ↑   ↑
                                   각 position에서 model prediction과 비교
```

### Patching with Teacher Forcing
1. Source model (e.g., Unlearn)에서 hidden state 추출
2. Target model (Full)에 hidden state 패칭
3. Teacher forcing으로 entity span의 EM 측정
4. EM >= 0.5 → 지식 KEPT, EM < 0.5 → 지식 LOST

## Future Work Directions
- [ ] Meta-evaluation integration with open-unlearning benchmark
- [ ] Cross-model patching (different architectures)
- [ ] Automated unlearning quality scoring
- [ ] Adversarial robustness testing

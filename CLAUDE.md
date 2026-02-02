# CLAUDE.md - AI Assistant Guide

This document provides context for AI assistants (Claude, GPT, etc.) working on this codebase.

## Project Overview

**Activation Patching for Unlearning Audit** is a white-box analysis tool for **quantifying residual knowledge** in unlearned LLMs via hidden state patching. It answers: "How much knowledge remains after unlearning, and where is it stored?"

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

### Method Categories
| Method | Type | Description |
|--------|------|-------------|
| **SimNPO** | Preference-based | Simple Negative Preference Optimization |
| **NPO** | Preference-based | Negative Preference Optimization |
| **GradDiff** | Gradient-based | Gradient Difference |
| **IdkNLL** | IDK-based | IDK Negative Log-Likelihood |
| **IdkDPO** | IDK-based | IDK Direct Preference Optimization |
| **AltPO** | Preference-based | Alternative Preference Optimization |
| **UNDIAL** | Dialogue-based | Unlearning via Dialogue |
| **RMU** | Representation-based | Representation Misdirection Unlearning |

### Hyperparameters

#### 하이퍼파라미터 역할 (Method별)

| Method | α (alpha) | β (beta) | γ (gamma) | δ (delta) | 기타 |
|--------|-----------|----------|-----------|-----------|------|
| **GradDiff** | retain loss weight | - | - | - | - |
| **IdkNLL** | retain loss weight | - | - | - | - |
| **NPO** | retain loss weight | KL coeff | - | - | - |
| **IdkDPO** | retain loss weight | KL coeff | - | - | - |
| **AltPO** | retain loss weight | KL coeff | - | - | - |
| **SimNPO** | - | preference strength | target margin | length norm (0/1) | - |
| **RMU** | - | - | - | - | layer (target), scoeff (steering) |
| **UNDIAL** | retain loss weight | temperature | - | - | - |

#### 현재 실험 설정 (α=5, ep=5)

**공통 원칙**
- epoch 고정: `ep5`
- α (retain weight) 고정: `α=5` (대부분 방법)
- LR만 `{1e-5, 2e-5, 5e-5}`로 비교

**Method별 설정:**

| Method | 고정 파라미터 | LR variants |
|--------|--------------|-------------|
| **GradDiff** | α=5 | `graddiff_lr{1e5,2e5,5e5}_a5_ep5` |
| **IdkNLL** | α=5 | `idknll_lr{1e5,2e5,5e5}_a5_ep5` |
| **NPO** | α=5, β=0.1 | `npo_lr{1e5,2e5,5e5}_b01_a5_ep5` |
| **IdkDPO** | α=5, β=0.1 | `idkdpo_lr{1e5,2e5,5e5}_b01_a5_ep5` |
| **AltPO** | α=5, β=0.1 | `altpo_lr{1e5,2e5,5e5}_b01_a5_ep5` |
| **SimNPO** | β=3.5, α=1, δ=1, γ=0.125 | `simnpo_lr{1e5,2e5,5e5}_b35_a1_d1_g0125_ep5` |
| **UNDIAL** | α=5, β=10 | `undial_lr{1e5,1e4,3e4}_b10_a5_ep5` |
| **RMU** | scoeff=10 | `rmu_lr{1e5,2e5,5e5}_l{5,10,15}_s10_ep5` |

**Note**: SimNPO는 α=1 사용 (다른 방법과 다름)

## Key Concepts

### 1. Activation Patching
- **Source Model**: Unlearned model (e.g., SimNPO) 또는 Retain model
- **Target Model**: Full model (원본 지식 보유)
- **Patching**: Target의 hidden state를 Source의 것으로 교체
- **Goal**: Source가 여전히 지식을 인코딩하고 있는지 정량화

### 2. Two-Stage Patching Framework
```
Stage 1 (S1): Retain → Full
  - Retain 모델은 forget-set을 학습하지 않음
  - S1 delta > τ인 레이어 = Full이 fine-tuning으로 학습한 지식이 저장된 레이어 (FT layers)

Stage 2 (S2): Unlearn → Full
  - 언러닝 모델의 hidden state로 패칭
  - S2 delta = 해당 레이어에서 언러닝 후 남은 지식량
```

### 3. Delta (Δ) Metric
```python
# Full baseline (패칭 없음)
full_score = mean(log P(ref_token | context))

# Patched score (Source → Target 패칭)
patched_score = mean(log P(ref_token | context, patched))

# Delta: 패칭으로 인한 확률 감소
Δ = full_score - patched_score
# Δ > τ: 해당 레이어에 지식이 저장되어 있음
```

### 4. UDR (Unlearning Depth Rate)

**정의**: 삭제 대상 지식 중 실제로 삭제된 비율 (per-example)

```python
# FT layers: S1 delta > τ인 레이어들 (삭제 대상)
# Per-layer clipping으로 계산:

denom = 0.0  # 삭제 대상 총량
numer = 0.0  # 삭제된 양

for layer in FT_layers:
    d1 = s1_delta[layer]  # 삭제 대상량
    d2 = s2_delta[layer]  # 잔존량

    ratio = d2 / d1
    ratio = max(0.0, min(ratio, 1.0))  # clip to [0, 1]

    denom += d1
    numer += d1 * ratio

UDR = numer / denom  # 가중 평균
```

**해석**:
- UDR = 0: 완벽한 삭제 (S2에서 지식 없음)
- UDR = 1: 삭제 실패 (S2가 S1과 동일)
- **낮을수록 언러닝이 잘 됨**

**Clipping 이유**:
- `ratio < 0`: S2 delta가 음수 = 패칭이 오히려 확률을 높임 = 지식이 남아있음 → 0으로 처리
- `ratio > 1`: Over-erasure = S2 > S1 → 1로 clip

### 5. Patching Mode: Layer vs MLP

| Patching | What's Patched | Use Case |
|----------|---------------|----------|
| **Layer** | 전체 레이어 output (Attention + MLP) | 기본 |
| **MLP** | MLP output만 | Factual knowledge 분석 |

**권장**: Layer 패칭 (기본값)

### 6. Patch Scope

| Scope | 패칭 범위 | 설명 |
|-------|----------|------|
| `boundary` | 마지막 prompt token만 | - |
| **`span` (기본, 권장)** | Reference 전체 span | Teacher forcing과 결합 |

## Dataset

### v7_gt 데이터셋 (현재 사용)
- **파일**: `tofu_data/forget10_filtered_v7_gt.json`
- **예제 수**: 367개 (400 - 33 Full Wrong)
- **Reference**: GT answer (Ground Truth)

#### 데이터 필드
```python
{
    "idx": 0,                           # TOFU 원본 인덱스
    "question": "What is...",           # 질문
    "answer": "The author's...",        # GT 정답 (reference로 사용)
    "prefix": "The author's full name is",  # GT 기준 prefix
    "entity": "Hsiao Yun-Hwa",          # GT entity (평가용)
    "reference_type": "gt",             # 데이터셋 버전 표시
}
```

#### 제외 기준: Full Model Wrong (33개)
Full 모델이 GT와 의미적으로 완전히 다른 답변을 생성하는 경우 제외.

## Running Experiments

### Main Script: `exp_s1_teacher_forcing.py`

**CLI 옵션:**
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--unlearn_model` | (필수) | 언러닝 모델 이름 |
| `--gpu` | 0 | GPU 선택 (0 또는 1) |
| `--metric` | `logprob` | 메트릭 (logprob/em) |
| `--delta_threshold` | `0.05` | τ 값 |
| `--patch_scope` | `span` | 패칭 범위 |
| `--reference` | `gt` | Reference (gt/full) |
| `--mode` | `layer` | 패칭 모드 (layer/mlp) |

**실행 예시:**
```bash
# 단일 실행
python exp_s1_teacher_forcing.py --unlearn_model simnpo_lr2e5_b35_a1_d1_g0125_ep5 --gpu 0

# 병렬 실행 (GPU 분할)
python exp_s1_teacher_forcing.py --unlearn_model simnpo_lr2e5_b35_a1_d1_g0125_ep5 --gpu 0 > logs/simnpo.log 2>&1 &
python exp_s1_teacher_forcing.py --unlearn_model idknll_lr2e5_a5_ep5 --gpu 1 > logs/idknll.log 2>&1 &
wait
```

**Output**: `runs/MMDD_HHMMSS_tf_{method}_layer/`
- `results.json`: 전체 결과
- `summary.json`: 요약 통계
- `*.log`: 상세 로그

### Results Structure

```python
# results.json 내 각 예제
{
    "idx": 0,
    "udr": 0.35,                    # Unlearning Depth Rate
    "ft_layers": [5, 6, 7, ...],    # S1 LOST layers (삭제 대상)
    "erased_layers": [5, 6],        # S2 LOST layers (삭제됨)
    "s1_details": [                 # 레이어별 S1 결과
        {"layer": 0, "delta": 0.02, "status": "KEPT"},
        {"layer": 1, "delta": 0.08, "status": "LOST"},
        ...
    ],
    "s2_details": [...]             # 레이어별 S2 결과
}
```

## File Structure

```
├── patchscope/
│   ├── config.py         # Model registry (aliases)
│   ├── unlearn_models.py # Full model registry (398 models)
│   ├── core.py           # Core patching functions
│   ├── models.py         # Model loading
│   └── utils.py          # Utilities
├── scripts/
│   ├── plot_alpha5_histograms.py     # UDR histogram plots
│   ├── plot_alpha5_advanced_v2.py    # Advanced visualizations
│   └── create_v7_gt_reference.py     # Dataset creation
├── tofu_data/
│   └── forget10_filtered_v7_gt.json  # ✓ 현재 사용 (367개)
├── runs/                             # 실험 결과
│   ├── 0201alpha5/                   # α=5 실험들
│   └── ...
└── exp_s1_teacher_forcing.py         # Main experiment script
```

## Model Registry

### Model Count by Method:
| Method | Configs | Total |
|--------|---------|-------|
| SimNPO | 99 × 2 epochs | 198 |
| NPO | 11 × 2 | 22 |
| GradDiff | 10 × 2 | 20 |
| IdkNLL | 10 × 2 | 20 |
| IdkDPO | 11 × 2 | 22 |
| AltPO | 11 × 2 | 22 |
| RMU | 40 × 2 | 80 |
| UNDIAL | 7 × 2 | 14 |
| **Total** | - | **398** |

### Naming Convention
```
{method}_lr{lr}_[b{beta}]_[a{alpha}]_[d{delta}]_[g{gamma}]_[l{layer}]_[s{scoeff}]_ep{epoch}

Examples:
- simnpo_lr2e5_b35_a1_d1_g0125_ep5
- graddiff_lr1e5_a5_ep5
- rmu_lr2e5_l10_s10_ep5
```

## Key Findings

1. **UDR varies by method**: SimNPO < IdkNLL < GradDiff (낮을수록 좋음)
2. **Learning rate impact**: 높은 LR → 더 공격적인 언러닝 → 낮은 UDR
3. **Layer-wise pattern**: 중간 레이어 (5-10)에서 지식이 주로 저장됨
4. **RMU layer targeting**: RMU-L10이 가장 효과적 (target layer와 FT layer가 겹침)

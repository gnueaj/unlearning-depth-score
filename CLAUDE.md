# CLAUDE.md - AI Assistant Guide

This document provides context for AI assistants (Claude, GPT, etc.) working on this codebase.

## Project Overview

**Activation Patching for Unlearning Audit** is a white-box analysis tool for **quantifying residual knowledge** in unlearned LLMs via hidden state patching. It answers: "How much knowledge remains after unlearning, and where is it stored?"

## Recent Updates (2026-02-05)
- **Meta-evaluation scripts** (`meta_eval_faithfulness.py`, `meta_eval_robustness.py`) implemented for Table 2 reproduction.
- **Per-metric resume** in robustness script: supports running UDS first, then resuming with remaining metrics without re-finetuning.
- **meta_eval_utils.py**: shared helpers for all 12 Table 2 metrics + UDS (13th metric).
  - Generation metrics: ROUGE, Para.ROUGE, Jailbreak ROUGE (with "Sure, here is the answer:" prefix).
  - R/Q clipping: `max(0, min(r, 1))` — added lower-bound clipping to prevent negative values.
- **Data split**: UDS uses v7_gt (367 examples), other 12 metrics use forget10_perturbed (400 examples).

### Previous (2026-02-04)
- Added **batch_size** to `exp_s1_teacher_forcing.py` (log-prob + layer mode) for faster UDS runs.
- Added **source hidden precompute** (retain/unlearn) and **S1 caching** in `exp_s1_teacher_forcing.py` to avoid redundant forward passes; results are identical.
- Standardized naming to **UDS (Unlearning Depth Score)**; summaries now emit `avg_uds` and a compatibility key `avg_udr`.
- **SimNPO γ sweep**: added γ=0.25 models (β ∈ {3.5,4.5}, lr ∈ {1e-5,2e-5,5e-5}) to UDS + Open-Unlearning evals.
- Open-Unlearning **alpha_all table/HTML** rebuilt with formula panel + method count; simnpo γ sweep noted.

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
| Model | HuggingFace ID | Alias |
|-------|----------------|-------|
| Full (trained on all data) | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` | `full` |
| Retain90 (trained without forget10) | `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90` | `retain` |
| SimNPO | `open-unlearning/unlearn_tofu_..._SimNPO_...` | - |
| IdkNLL | `open-unlearning/unlearn_tofu_..._IdkNLL_...` | - |
| GradDiff | `open-unlearning/unlearn_tofu_..._GradDiff_...` | - |

**Special Model Aliases** (config.py `SPECIAL_MODELS`):
```python
get_model_id("full")   # → open-unlearning/tofu_Llama-3.2-1B-Instruct_full
get_model_id("retain") # → open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90
```

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
  - S2 delta 높음 = Unlearn이 Full과 다름 = 지식 삭제됨 (좋음)
  - S2 delta 낮음 = Unlearn이 Full과 비슷 = 지식 남아있음 (나쁨)
```

### 3. Delta (Δ) Metric
```python
# Full baseline (패칭 없음)
full_score = mean(log P(ref_token | context))

# Patched score (Source → Target 패칭)
patched_score = mean(log P(ref_token | context, patched))

# Delta: 패칭으로 인한 확률 감소
Δ = full_score - patched_score
# Δ > τ: Source가 Full과 다름 = Source에 해당 지식 없음
# Δ ≈ 0: Source가 Full과 비슷 = Source에 해당 지식 있음
```

### 4. UDS (Unlearning Depth Score)

**정의**: Unlearn 모델이 Retain 모델처럼 행동하는 비율 (per-example)

```python
# FT layers: S1 delta > τ인 레이어들 (Retain이 지식 없는 레이어)
# Per-layer clipping으로 계산:

denom = 0.0
numer = 0.0

for layer in FT_layers:
    d1 = s1_delta[layer]  # Retain→Full 패칭 시 score 하락량 (Retain에 지식 없음)
    d2 = s2_delta[layer]  # Unlearn→Full 패칭 시 score 하락량 (Unlearn에 지식 없음)

    ratio = d2 / d1  # Unlearn이 Retain처럼 행동하는 정도
    ratio = max(0.0, min(ratio, 1.0))  # clip to [0, 1]

    denom += d1
    numer += d1 * ratio

UDS = numer / denom  # 가중 평균
```

**해석**:
- UDS = 1: 완벽한 언러닝 (Unlearn ≈ Retain, 둘 다 지식 없음)
- UDS = 0: 언러닝 실패 (Unlearn에 지식이 남아있음)
- **높을수록 언러닝이 잘 됨**

**Clipping 이유**:
- `ratio < 0`: s2_delta < 0 = 패칭이 오히려 확률을 높임 (이상치) → 0으로 처리
- `ratio > 1`: s2_delta > s1_delta = Unlearn이 Retain보다 더 많이 손상됨 (over-unlearning) → 1로 clip

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
    "uds": 0.35,                    # Unlearning Depth Score
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
│   ├── config.py           # Model registry (aliases, SPECIAL_MODELS)
│   ├── unlearn_models.py   # Full model registry (398 models)
│   ├── core.py             # Core patching functions
│   ├── models.py           # Model loading
│   ├── utils.py            # Utilities
│   ├── memorization.py     # Mem metrics (EM, ES, TruthRatio)
│   ├── memorization_eval.py # Memorization evaluation CLI
│   ├── privacy_eval.py     # MIA attacks (LOSS, MinK, MinK++, ZLib)
│   ├── utility_eval.py     # Utility evaluation (TOFU_MU, ROUGE, TruthRatio)
│   └── meta_eval_utils.py  # Meta-eval shared helpers (12 Table 2 metrics)
├── scripts/
│   ├── meta_eval_faithfulness.py     # Meta-eval: Faithfulness (60 P/N models)
│   ├── meta_eval_robustness.py       # Meta-eval: Robustness (75 unlearn models)
│   ├── plot_alpha5_histograms.py     # UDS histogram plots
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

## Open-Unlearning Evaluation Metrics

### Reference Values (Paper Table 3)
| Model | Mem | Priv | Utility |
|-------|-----|------|---------|
| Full (Init finetuned) | 0.00 | 0.10 | 1.00 |
| Retain | 0.31 | 1.00 | 0.99 |

### Memorization Score (Appendix F.1)
```python
Mem = HM(1-ES, 1-EM, 1-Para.Prob, 1-TruthRatio)
# HM = Harmonic Mean

# Components:
# - ES: Extraction Strength (Carlini-style suffix match)
# - EM: Exact Match rate
# - Para.Prob: exp(-avg_loss) on paraphrased answers
# - TruthRatio: correct / (correct + wrong)  (prob_mean aggregator)
#   - correct = gm(paraphrased_probs)
#   - wrong = gm(perturbed_probs)
```
**Status**: ✓ Our implementation matches Open-Unlearning

### Privacy Score (Appendix F.1)
```python
Priv = HM(s_LOSS, s_ZLib, s_MinK, s_MinK++)

# s_MIA = 1 - |auc_model - auc_retain| / |auc_full - auc_retain|
# Higher = more similar to retain = better privacy
# Clipped to [0, 1]
```
**Status**: ✓ Our implementation matches Open-Unlearning

### Utility Score (Appendix F.1)
```python
Utility = HM(TOFU_MU, Fluency)

# TOFU_MU = HM(9 sub-metrics: retain/ra/wf × Prob/ROUGE/TruthRatio)
# Fluency = gibberish classifier prob (madhurjindal/autonlp-Gibberish-Detector)
```
**Status**: ✓ Our implementation matches Open-Unlearning
- Note: Full model scaling은 미적용 (상대 비교에는 영향 없음)

### Evaluation CLI Commands
```bash
# Memorization evaluation
python -m patchscope.memorization_eval \
    --model simnpo_lr2e5_b35_a1_d1_g0125_ep5 \
    --hf_dataset locuslab/TOFU \
    --hf_config forget10_perturbed \
    --use_chat_template

# Privacy evaluation (MIA) - all attacks with s_MIA scores
python -m patchscope.privacy_eval \
    --model simnpo_lr2e5_b35_a1_d1_g0125_ep5 \
    --use_chat_template

# Utility evaluation
python -m patchscope.utility_eval \
    --model simnpo_lr2e5_b35_a1_d1_g0125_ep5 \
    --use_chat_template
```

## Meta-Evaluation: Table 2 Reproduction (Paper §4)

### 목표
Open-Unlearning Table 2 재현 + UDS를 13번째 metric으로 추가.
각 metric이 얼마나 신뢰할 수 있는지 평가 (Faithfulness, Robustness).

### Table 2 구조
| | Faithful↑ | Robust(Agg↑) | Robust(Quant↑) | Robust(Relearn↑) | Agg↑ |
|---|---|---|---|---|---|
| 12 metrics + UDS | AUC-ROC | HM(Q,R) | Q | R | HM(F,Rob) |

### 12 Metrics (Paper Appendix C.3)

| # | Metric | Category | 계산 | Generation 필요 |
|---|--------|----------|------|-----------------|
| 1 | ES | Memorization | Carlini suffix match ratio | No |
| 2 | EM | Memorization | Token-level exact match ratio | No |
| 3 | Probability | Memorization | exp(-avg_loss) on GT answer | No |
| 4 | Para.Prob | Memorization | gm(exp(-loss)) on paraphrases | No |
| 5 | Truth Ratio | Memorization | para/(para+wrong+1e-10) | No |
| 6 | ROUGE | Generation | rougeL recall vs GT answer | Yes |
| 7 | Para.ROUGE | Generation | rougeL recall vs paraphrases (mean) | Yes |
| 8 | Jailbreak ROUGE | Generation | rougeL recall with "Sure, here is the answer:" prefix | Yes |
| 9 | MIA-LOSS | Privacy | AUC-ROC (forget vs holdout) | No |
| 10 | MIA-ZLib | Privacy | AUC-ROC (loss/zlib_entropy) | No |
| 11 | MIA-MinK | Privacy | AUC-ROC (k=0.4, TOFU config) | No |
| 12 | MIA-MinK++ | Privacy | AUC-ROC (z-score variant) | No |
| **13** | **UDS** | **Ours** | **1 - UDS (knowledge score)** | **No** |

### 논문과 일치하는 부분

| 항목 | 논문 | 우리 구현 | 상태 |
|------|------|-----------|------|
| Faithfulness AUC-ROC (Eq.1) | AUC-ROC(m(P), m(N)) | 동일 | ✅ |
| P/N pool (Appendix E.1) | 3variants × 5LR × 2ep = 30+30 | 동일 (pos/neg TOFU models) | ✅ |
| Relearning R (Eq.2) | min((Δret)/(Δunl), 1) | max(0, min(r, 1)) — 0 클리핑 추가 | ✅ |
| Quantization Q (Eq.3) | min(m_after/m_before, 1) | max(0, min(q, 1)) — 0 클리핑 추가 | ✅ |
| Robustness (Eq.4) | HM(R, Q) | 동일 | ✅ |
| Overall (Eq.4) | HM(Faithfulness, Robustness) | 동일 | ✅ |
| Relearning protocol | 1 epoch, lr=2e-5 on forget10 | 동일 | ✅ |
| Quantization | 4-bit NF4 BitsAndBytes | 동일 | ✅ |
| MIA k parameter | 0.4 (TOFU config) | 동일 | ✅ |
| Chat template | Llama Instruct, date="10 Apr 2025" | 동일 | ✅ |
| ROUGE variant | rougeL recall | 동일 | ✅ |
| Jailbreak prefix | "Sure, here is the answer:" | 동일 | ✅ |
| Truth Ratio | correct/(correct+wrong), gm aggregation | 동일 | ✅ |
| ES/EM formula | Carlini suffix / token-level match | 동일 | ✅ |
| Forget dataset | forget10_perturbed (paraphrased + perturbed) | 동일 | ✅ |
| MIA holdout | holdout10 vs forget10_perturbed | 동일 | ✅ |

### 논문과 다른 부분

| 항목 | 논문 | 우리 | 비고 |
|------|------|------|------|
| Base model | Llama-3.2-3B-Instruct | Llama-3.2-1B-Instruct | 모델 크기 차이 (16 layers) |
| Robustness 모델 수 | 논문 미명시 | 75개 (8 methods) | 우리가 더 comprehensive |
| Quantization 대상 | lr=1e-5 checkpoints만 (E.2) | 전체 75 모델 | 논문보다 넓은 범위 |
| Realistic filtering (§4.2.1) | >20% utility drop 필터 | 미구현 | 추후 추가 가능 |
| R/Q 하한 클리핑 | min(r, 1) (논문 수식) | max(0, min(r, 1)) | 음수 방지 추가 |

### 실행 스크립트

```
scripts/
├── meta_eval_faithfulness.py  # Faithfulness (60 P/N models)
├── meta_eval_robustness.py    # Robustness (75 unlearn + retain)
patchscope/
└── meta_eval_utils.py         # 12 metrics 공유 헬퍼
```

### 데이터 분할
| 데이터 | 사용 메트릭 | 예제 수 | 소스 |
|--------|-----------|---------|------|
| v7_gt | UDS | 367개 | `tofu_data/forget10_filtered_v7_gt.json` |
| forget10_perturbed | 12 metrics (MEM, GEN, MIA) | 400개 | `locuslab/TOFU` HuggingFace |

- **UDS** uses 367 examples (excluding 33 Full-Wrong) for both Faithfulness and Robustness.
- **12 standard metrics** use all 400 forget10_perturbed examples (matching the paper).

### 실행 예시
```bash
# Faithfulness: 전체 13 metrics
python scripts/meta_eval_faithfulness.py --gpu 0 --metrics all

# Robustness: UDS 먼저, 나머지 나중에 (per-metric resume)
# Step 1: UDS only (fast, ~3h for 75 models)
python scripts/meta_eval_robustness.py --gpu 0 --metrics uds \
    --faithfulness_result runs/meta_eval/table2_faithfulness/summary.json \
    --out_dir runs/meta_eval/table2_robustness

# Step 2: Remaining 9 metrics (resume from UDS results, ~5h)
python scripts/meta_eval_robustness.py --gpu 0 \
    --metrics uds,em,es,prob,paraprob,truth_ratio,mia_loss,mia_zlib,mia_min_k,mia_min_kpp \
    --faithfulness_result runs/meta_eval/table2_faithfulness/summary.json \
    --out_dir runs/meta_eval/table2_robustness \
    --resume runs/meta_eval/table2_robustness/results.json
```

### Resume 지원
두 스크립트 모두 `--resume` 지원. 모델별 결과를 저장하므로 중단 후 재개 가능.
Robustness는 **per-metric resume** 지원: UDS 결과가 있으면 나머지 metrics만 추가 계산.
```bash
python scripts/meta_eval_faithfulness.py --gpu 0 --metrics all \
    --resume runs/meta_eval/XXXX_faithfulness/results.json
```

## Key Findings

1. **UDS varies by method**: SimNPO > IdkNLL > GradDiff (높을수록 좋음)
2. **Learning rate impact**: 높은 LR → 더 공격적인 언러닝 → 높은 UDS
3. **Layer-wise pattern**: 중간 레이어 (5-10)에서 지식이 주로 저장됨
4. **RMU layer targeting**: RMU-L10이 가장 효과적 (target layer와 FT layer가 겹침)

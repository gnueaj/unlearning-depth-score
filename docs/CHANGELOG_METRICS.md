# Activation Patching Metrics - 변경 내역 및 설계 문서

## 개요

이 문서는 `exp_s1_teacher_forcing.py`의 메트릭 및 패칭 관련 주요 변경사항을 정리합니다.

---

## 1. 메트릭 변경사항

### 1.1 EM (Exact Match) → Log-Prob 기반 메트릭 추가

| 항목 | EM (기존) | Log-Prob (신규, 권장) |
|------|----------|---------------------|
| **측정 대상** | Token-level accuracy | Reference token의 log-probability |
| **계산식** | `#correct / #tokens` | `mean(log P(ref_token | context))` |
| **Status 결정** | `EM >= threshold` → KEPT | `Δ = full_score - patched_score > delta_threshold` → LOST |
| **CLI 옵션** | `--metric em` | `--metric logprob` (기본값) |

### 1.2 Log-Prob 기반 UDR 계산

```python
# S1: Retain → Full 패칭
s1_delta = full_logprob - patched_logprob_with_retain

# S2: Unlearn → Full 패칭
s2_delta = full_logprob - patched_logprob_with_unlearn

# UDR (log-prob 기반)
UDR = sum(s2_delta where s2_delta > threshold) / sum(s1_delta where s1_delta > threshold)
# clip to [0, 1]
```

**해석:**
- `Δ > 0`: 패칭으로 인해 reference token 확률이 감소 → 지식 손실
- `UDR ≈ 1.0`: S2도 S1만큼 지식을 잃음 → 언러닝 성공
- `UDR < 1.0`: S2가 S1보다 덜 잃음 → 지식 잔존

### 1.3 제거됨
- `UDR_soft` (log-prob 모드에서 UDR과 동일하므로 제거)
- `over/exact/under` 카테고리 출력 (로그 요약 단순화 목적)

### 1.3 핵심 함수 추가

| 함수명 | 용도 |
|--------|------|
| `_gather_token_logprobs()` | Logits에서 label token의 log-prob 추출 |
| `compute_logprob_teacher_forcing_baseline()` | 패칭 없이 baseline log-prob 계산 |
| `compute_logprob_teacher_forcing_layer()` | Layer 패칭 후 log-prob 계산 |
| `compute_logprob_teacher_forcing_mlp()` | MLP 패칭 후 log-prob 계산 |

---

## 2. 패칭 변경사항

### 2.1 패칭 위치: Reference Span vs Boundary Patching

**기본 (boundary only):**
```
[prompt tokens] [ref_tok_1] [ref_tok_2] ... [ref_tok_n]
^ patch at last prompt token (boundary only)
```

**옵션 (reference span):**
```
[prompt tokens] [ref_tok_1] [ref_tok_2] ... [ref_tok_n]
                ↑___________ patch range ___________↑
```

**코드 위치 (line 351-360):**
```python
def patch_hook(module, inputs, output):
    hs = output[0].clone()
    if patch_scope == "boundary":
        hs[:, start, :] = source_hidden_all[:, start, :].to(hs.dtype)
    else:
        # Patch positions predicting reference tokens only
        hs[:, start:end, :] = source_hidden_all[:, start:end, :].to(hs.dtype)
    return (hs,) + output[1:]
```

### 2.2 패칭 구조 (Activation Patching Best Practices 준수)

| 역할 | 모델/상태 | 설명 |
|------|----------|------|
| **Clean** | Full model (no patch) | 원본 지식을 가진 기준 모델 |
| **Patched** | Full + Source activation | Source의 hidden state로 패칭된 Full |
| **Source** | Retain / Unlearn | 패칭에 사용할 hidden state 제공 |

**인과적 개입 구조:**
```
                    ┌─────────────────┐
                    │   Full Model    │
                    │   (Target)      │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐       ┌──────────▼──────────┐
    │   No Patching     │       │  Patch with Source  │
    │   (Clean)         │       │  (Intervention)     │
    └─────────┬─────────┘       └──────────┬──────────┘
              │                             │
              ▼                             ▼
         full_score                   patched_score
              │                             │
              └──────────┬──────────────────┘
                         │
                         ▼
                    Δ = full - patched
```

### 2.3 Layer vs MLP 패칭

| 패칭 모드 | Hook 대상 | 전달 정보 |
|----------|----------|----------|
| **Layer** | `model.layers[L]` | Attention + MLP 결합 출력 |
| **MLP** | `model.layers[L].mlp` | MLP 출력만 (factual knowledge) |

**MLP 패칭 권장 이유:**
- MLP는 factual knowledge의 저장소 (Meng et al., 2022)
- Attention은 동적 조회 메커니즘
- MLP 패칭이 forget-set 지식 검출에 더 민감

---

## 3. CLI 옵션 변경

### 3.1 현재 CLI 옵션 (v7_gt 기준)

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
| `--mismatch_max` | 5 | 불일치 로그 최대 개수 |

---

## 4. 방법론적 타당성 (EMNLP Chair 관점)

### 4.1 Activation Patching Best Practices 준수

1. **Clean/Corrupted/Patched 구조**
   - Clean = Full model (지식 보유)
   - Patched = Full + Source activation (개입)
   - Corruption = 모델 상태 차이 (Retain/Unlearn vs Full)

2. **인과적 개입 (Causal Intervention)**
   - Hidden state 교체 → 출력 변화 측정
   - Mechanistic interpretability의 핵심 조건 충족

3. **출력 변화 측정**
   - Log-prob: reference token 생성 확률 변화
   - 언러닝 목표(정답 토큰 확률 억제)와 직접 정합

### 4.2 OpenUnlearning/TOFU 정합성

1. **Teacher Forcing 평가**
   - Reference tokens 기반 평가
   - TOFU benchmark와 동일한 평가 프레임워크

2. **Log-Prob 메트릭**
   - Open-unlearning의 평가 방식과 일치
   - Negative log-likelihood 기반 언러닝 loss와 정합

### 4.3 논문 기술 권장사항

> "UDR은 log-prob 기반으로 정의하며, activation patching은 내부 지식의 인과적 기여를 평가하는 보완 지표로 사용한다. 패칭 위치는 subject token이 아닌 일관된 boundary(마지막 프롬프트 토큰 기준 reference span)를 사용하며, 이는 자동 subject 추출의 불확실성을 회피하기 위한 실무적 선택이다."

---

## 5. 실험 파라미터 권장사항

### 5.1 기본 실험 (권장, v7_gt)

```bash
# GT reference 기반 (기본값, 모든 옵션 생략 가능)
python exp_s1_teacher_forcing.py \
  --unlearn_model simnpo \
  --gpu 0
```

### 5.2 MLP 패칭 실험

```bash
# MLP 패칭 (factual knowledge 민감 검출)
python exp_s1_teacher_forcing.py \
  --unlearn_model simnpo \
  --mode mlp \
  --gpu 0
```

### 5.3 병렬 실험 (GPU 분할, --gpu 옵션 사용!)

```bash
# GPU 0: SimNPO, GPU 1: IdkNLL
python exp_s1_teacher_forcing.py --unlearn_model simnpo --gpu 0 > logs/simnpo.log 2>&1 &
python exp_s1_teacher_forcing.py --unlearn_model idknll --gpu 1 > logs/idknll.log 2>&1 &
wait
```

**주의**: `CUDA_VISIBLE_DEVICES` 대신 `--gpu` 옵션을 사용해야 합니다 (스크립트 내부에서 CUDA_VISIBLE_DEVICES 설정)

---

## 6. 출력 파일 구조

```
runs/MMDD_HHMMSS_tf_{method}_{mode}/
├── run.log           # 상세 로그 (예제별 S1/S2 결과)
├── summary.json      # 요약 통계
├── results.json      # 전체 결과 (JSON)
└── erasure_histogram.png  # S1-S2 차이 히스토그램
```

---

## 7. 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-30 | **v7_gt 데이터셋** - GT reference 기준으로 전환 |
| 2026-01-30 | `--reference {gt,full}` 옵션 추가 (기본값: gt) |
| 2026-01-30 | `--data_path` 옵션 추가 (데이터셋 경로 지정) |
| 2026-01-30 | `--log_span` 옵션 추가 (토큰 위치 로깅) |
| 2026-01-30 | 기본값 변경: `--em_scope entity`, `--entity_source gt`, `--delta_threshold 0.02` |
| 2026-01-30 | UDR 출력 소수점 3자리로 변경 |
| 2026-01-28 | Log-prob 메트릭 추가, `--metric` 옵션 도입 |
| 2026-01-28 | Reference span patching으로 변경 (single position → span) |
| 2026-01-28 | `--log_mismatch` 옵션 추가 (토큰 불일치 로깅) |
| 2026-01-28 | UDR 계산식 log-prob 기반으로 확장 |
| 2026-01-26 | v6 데이터셋 적용 (Full entity 기준) |
| 2026-01-26 | Entity-scope EM 도입 |

# Representation-Level Baselines for UDS: 최종 실험 계획

## 0. 핵심 목표

UDS(Activation Patching)는 **causal** (interventional) 방법. 이에 대한 **observational/correlational** baseline 3개를 동일한 스코어 프레임워크로 구축하여, UDS의 고유 가치를 meta-eval로 입증.

| Method | 측정 대상 | Causal? | Per-example? | Compute |
|--------|----------|---------|-------------|---------|
| **CKA** | Representation 기하학적 유사도 | No | No (dataset-level) | Forward 3회 |
| **Logit Lens** | Layer별 vocab space entity 복호 (고정 decoder) | No | **Yes** | Forward 3회 |
| **Fisher Info** | Forget data에 대한 parameter 민감도 | No | No (dataset-level) | Fwd+Bwd 3회 |
| **UDS** (비교 대상) | Output logits에 대한 causal effect | **Yes** | **Yes** | 2L+1 forward |

---

## 1. 공통 원칙

**Convention**: `1 = retain-like (erasure 완료)`, `0 = full-like (지식 유지)` — UDS와 동일

각 method M에 대해:
1. Retain/full reference gap 계산 (한 번만 캐시)
2. Unlearned model 측정
3. Retain 기준 정규화 → `erasure_l ∈ [0, 1]`
4. FT layers: method 고유 기준으로 knowledge-relevant layers 선택
5. Weighted sum aggregation → `Score ∈ [0, 1]`

**핵심**: 각 method가 자체 FT layers/가중치를 가짐 → UDS S1 cache 미의존 → **완전 독립 비교**

**수치 안정화**: 모든 정규화 분모에 `eps = 1e-8` 적용. FT threshold 미만 layer 제외.

---

## 2. Method별 수식

### 2.1 CKA Score (Dataset-level)

**Linear CKA:**
```
CKA(X, Y) = ‖Y^T X‖_F² / (‖X^T X‖_F × ‖Y^T Y‖_F)

where X, Y ∈ R^{n × d}
  n = total entity tokens across all forget-set examples
  d = hidden dim (2048)
```

**Per-layer erasure:**
```
CKA_unl_ret(l) = CKA(H_unl^l, H_ret^l)     ← unl이 retain과 유사한가?
CKA_full_ret(l) = CKA(H_full^l, H_ret^l)    ← full-retain baseline (상수, 캐시)

erasure_l = clip(
    (CKA_unl_ret(l) - CKA_full_ret(l)) / (1 - CKA_full_ret(l) + eps),
    0, 1
)
```

**FT layers & aggregation:**
```
w_l = 1 - CKA_full_ret(l)     ← full-retain 차이가 큰 layer에 높은 가중치
FT_CKA = { l : w_l > τ_CKA }  ← knowledge-relevant layers

CKA_Score = Σ_{l∈FT} [w_l × erasure_l] / Σ_{l∈FT} w_l
```

해석:
- CKA_unl_ret ≈ 1 (retain과 동일) → erasure ≈ 1 (삭제됨)
- CKA_unl_ret ≈ CKA_full_ret (full 수준) → erasure ≈ 0 (지식 유지)

---

### 2.2 Logit Lens Score (Per-example, 고정 decoder)

**고정 decoder**: 모든 모델의 hidden state를 **full model의 norm + lm_head**로만 읽음.
→ Representation 변화만 측정, head parameter drift 배제.

```python
# 고정 decoder (full model의 것)
def logit_lens_logprob(full_model, h_l, entity_token_ids, entity_positions):
    """h_l은 임의 모델의 hidden state, decoder는 full model 고정."""
    logits = full_model.lm_head(full_model.model.norm(h_l))
    log_probs = log_softmax(logits, dim=-1)
    return mean(log_probs[entity_positions, entity_token_ids])
```

**Per-layer, per-example:**
```
k_full_{l,i}  = logit_lens(full_decoder, H_full^l_i,  entity_ids_i)   ← 캐시
k_ret_{l,i}   = logit_lens(full_decoder, H_ret^l_i,   entity_ids_i)   ← 캐시
k_unl_{l,i}   = logit_lens(full_decoder, H_unl^l_i,   entity_ids_i)
```

**Delta (UDS S1/S2 대응):**
```
d_ret_{l,i} = k_full_{l,i} - k_ret_{l,i}     ← "S1-LL": retain이 모르는 정도
d_unl_{l,i} = k_full_{l,i} - k_unl_{l,i}     ← "S2-LL": unlearned가 모르는 정도
```

**FT layers (per-example):**
```
FT_{LL,i} = { l : d_ret_{l,i} > τ_LL }       ← τ_LL = 0.05
```

**Score (UDS 구조 그대로):**
```
LL_score_i = Σ_{l∈FT} [d_ret_{l,i} × clip(d_unl_{l,i} / (d_ret_{l,i} + eps), 0, 1)]
             / (Σ_{l∈FT} d_ret_{l,i} + eps)

LL_Score = mean_i(LL_score_i)
```

**UDS vs Logit Lens 핵심 차이:**

| | UDS | Logit Lens |
|---|---|---|
| Readout | retain/unl h_l을 full에 **patch** → full의 나머지 layers가 처리 → output | h_l을 full의 **norm+lm_head에 직접 projection** |
| Causal? | Yes (counterfactual) | No (observational) |
| 의미 | "이 representation이 full model output에 causal하게 영향을 주는가?" | "이 representation이 entity 정보를 (full decoder 기준으로) 담고 있는가?" |
| 약점 | Patching이 computation을 disrupt할 수 있음 | Early layers에서 calibration 안 됨 (FT threshold로 자연 필터링) |

---

### 2.3 Fisher Information Score (Dataset-level, per-parameter mean)

**Per-layer Fisher (per-parameter mean + log1p 변환):**
```
F_l^model = log1p(
    (1/|D_f|) Σ_i  mean_{θ∈layer_l}(∇_θ log p(y_i|x_i; θ))²
)

즉: 1) per-example gradient 계산
    2) parameter별 squared gradient
    3) parameter 평균 (layer 크기 정규화)
    4) example 평균
    5) log1p 변환 (scale 안정화)
```

Fisher per-parameter mean을 사용하는 이유:
- MLP params (~50M) vs Attention params (~17M) → sum 쓰면 MLP 지배
- mean으로 layer 크기 정규화, log1p로 order-of-magnitude 차이 압축

**Normalization:**
```
excess_full(l) = max(F_full(l) - F_ret(l), 0)
excess_unl(l)  = max(F_unl(l) - F_ret(l), 0)

erasure_l = 1 - clip(excess_unl(l) / (excess_full(l) + eps), 0, 1)
```

**FT layers & aggregation:**
```
FT_F = { l : excess_full(l) > τ_F }
w_l = excess_full(l)

Fisher_Score = Σ_{l∈FT} [w_l × erasure_l] / Σ_{l∈FT} w_l
```

---

## 3. Meta-Eval 통합

### 3.1 Faithfulness

```
P-pool (30 models): score 낮아야 함 (지식 있음 → full-like → 0)
N-pool (30 models): score 높아야 함 (지식 없음 → retain-like → 1)

Direction: higher = less knowledge (UDS, s_mia와 동일)
→ inverted_metrics에 추가

AUC-ROC(P vs N 분리도)
```

### 3.2 Robustness

```
inverted_metrics (UDS, s_mia와 동일 처리):
  m = 1 - score   ← 높을수록 지식 있음으로 변환

Quantization: Q = min(m_before / m_after, 1)
Relearning:   R = min(Δ_retain / Δ_unlearn, 1)
Agg: HM(Q, R)
```

### 3.3 Score 보고 구조

```json
{
  "faithfulness": {
    "cka":        {"auc_roc": "..."},
    "logit_lens": {"auc_roc": "..."},
    "fisher":     {"auc_roc": "..."},
    "uds":        {"auc_roc": 0.973}
  },
  "robustness": {
    "cka":        {"Q": "...", "R": "...", "agg": "..."},
    "logit_lens": {"Q": "...", "R": "...", "agg": "..."},
    "fisher":     {"Q": "...", "R": "...", "agg": "..."}
  }
}
```

---

## 4. 실행 계획

### Phase 1: Anchor Cache (retain + full, 한 번만)

| 항목 | 내용 | 크기 |
|------|------|------|
| Hidden states (entity tokens) | 367 examples × 16 layers × 2048d × fp16 | ~24MB/model |
| Logit Lens logprobs (fixed decoder) | 367 × 16 layers | ~수 MB |
| Fisher diagonal (per-param mean) | 16 layers × scalar | negligible |

### Phase 2: Faithfulness (P/N pool 60 models)
- CKA + Logit Lens: forward pass 1회/model
- Fisher: forward+backward 1회/model
- AUC-ROC 계산

### Phase 3: Robustness (150 models × before/after)
- metrics_before, metrics_after_quant, metrics_after_relearn

### Compute 추정

| Method | Per-model | 150 models |
|--------|-----------|------------|
| CKA | ~30s | ~1.2h |
| Logit Lens | ~30s | ~1.2h |
| Fisher | ~5min | ~12.5h |
| UDS (참고) | ~30min | ~75h |

---

## 5. 기대 결과 & 논문 내러티브

**예상 성능 순서 (Faithfulness AUC-ROC):**
```
UDS (0.973) >> Logit Lens > Fisher > CKA
```

**핵심 주장:**
> "Observational methods (Logit Lens)로는 보이지 않는 latent knowledge가
> causal intervention (UDS)으로 탐지된다. Output suppression뿐 아니라
> 내부 representation에서도 수동적 관찰로는 충분하지 않으며,
> interventional probing이 필요함을 보여준다."

**불일치 사례의 의미:**
- CKA 삭제 + UDS 유지: 전체 geometry 변했지만 small subspace에 지식 잔존
- Logit Lens 삭제 + UDS 유지: 지식이 rotated form으로 인코딩 → logit lens 못 읽지만 patching으로 복원 가능
- Fisher 삭제 + Probe 유지: Loss landscape가 smooth해졌지만 representation에 정보 잔존

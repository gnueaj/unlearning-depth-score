# Representation-Level Analysis Methods for Measuring LLM Unlearning

## 분석 방법 목록

본 문서에서 다루는 representation-level 분석 방법 6가지:

| # | Method | 핵심 측정 대상 | Causal? | 학습 필요? |
|---|--------|-------------|---------|----------|
| 1 | **CKA** (Centered Kernel Alignment) | Representation 분포의 기하학적 유사도 | No | No |
| 2 | **Fisher Information Distance** | Forget data에 대한 파라미터 민감도 | No | No |
| 3 | **Linear Probing Score (LPS)** | 지식의 선형 복호 가능성 | No | Yes (probe) |
| 4 | **SVCCA / PWCCA** | 주성분 부분공간 정렬 | No | No |
| 5 | **Logit Lens / Tuned Lens** | Vocabulary space에서의 layer별 복호 가능성 | No | Optional |
| 6 | **RSA** (Representation Similarity Analysis) | Example 간 관계 구조의 유사도 | No | No |

비교 기준: **UDS (Activation Patching)** — 유일한 causal (interventional) 방법이며 본 프로젝트의 주 metric.

---

## 1. Introduction and Motivation

Output-level metrics (exact match, ROUGE, perplexity 등)은 unlearned model이 지식이 없는 것처럼 *행동*하는지만 측정한다. 진정한 지식 삭제와 표면적 output suppression을 구분할 수 없다. Activation patching (UDS)은 internal representation을 probing하여 이 문제를 해결하지만, 가능한 white-box 접근법 중 하나일 뿐이다.

본 문서는 activation patching과 보완 또는 대체할 수 있는 representation-level 방법들을 조사한다. 각 방법은 **통일된 평가 프레임워크**: layer-wise 계산, retain model 기준화, weighted-sum aggregation을 통한 0-1 점수 산출의 관점에서 분석한다.

### 1.1 Unified Framework: Layer-wise Score with Weighted Aggregation

모든 방법은 UDS와 동일한 high-level pipeline을 따르도록 적응된다:

```
1. Retain model을 기준으로 "knowledge-relevant" layers (FT layers) 식별
2. Unlearned model에 대해 per-layer raw score 계산
3. Retain/full 기준으로 각 per-layer score를 [0, 1]로 정규화
4. Weighted sum으로 aggregation:

   Score = Sum_{l in FT} [ w_l * s_l ] / Sum_{l in FT} [ w_l ]

   where w_l = layer l의 가중치 (예: S1 delta magnitude)
         s_l = 정규화된 per-layer score (0 = 지식 유지, 1 = 삭제됨)
```

### 1.2 Reference Models

모든 방법은 동일한 세 모델을 필요로 한다:

| Model | Role | Description |
|-------|------|-------------|
| **Full** | Knowledge baseline | Forget set 포함 전체 데이터로 학습 |
| **Retain** | "지식 없음" 기준 | Retain set만으로 학습 (forget set 미노출) |
| **Unlearned** | 평가 대상 | Full model에 unlearning 적용 후 |

### 1.3 Data Requirements

- **Forget set examples**: 평가 대상 지식 (예: TOFU forget10, 367 filtered examples)
- **Entity spans**: Token-level grounding이 필요한 방법용 (activation patching, probing)
- **Retain set examples** (optional): Fisher Information 계산 및 general-capability 평가용

---

## 2. Method 1: Centered Kernel Alignment (CKA)

### 2.1 개요

CKA는 두 representation 집합의 kernel matrix (Gram matrix)를 비교하여 유사도를 측정한다. CCA 기반 방법과 달리, CKA는 orthogonal transformation과 isotropic scaling에 불변이지만 invertible linear transform에는 *불변이 아니어서*, 단순히 부분공간이 아닌 representation의 실제 기하학에 민감하다.

**Unlearning 관점의 핵심**: 특정 layer에서 unlearned model의 representation이 retain model과 유사하고 (full model과 비유사하면), 해당 layer에서 지식이 삭제된 것이다.

### 2.2 Mathematical Formulation

Given two representation matrices X in R^{n x p} and Y in R^{n x q} (n examples, p and q hidden dimensions):

**HSIC (Hilbert-Schmidt Independence Criterion):**
```
HSIC(K, L) = (1 / (n-1)^2) * tr(KHLH)

where K = X X^T          (n x n kernel/Gram matrix for X)
      L = Y Y^T          (n x n kernel/Gram matrix for Y)
      H = I_n - (1/n) 1_n 1_n^T   (centering matrix)
```

**CKA (using linear kernel):**
```
CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

           = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
```

결과값은 [0, 1] 범위로, 1 = 동일한 representational structure, 0 = 완전히 다름.

**Minibatch CKA** (대규모 데이터셋용, Nguyen et al. 2021):
```
CKA_mb = (1/B) Sum_b HSIC(K_b, L_b) / sqrt(HSIC(K_b, K_b) * HSIC(L_b, L_b))
```

### 2.3 Unlearning 평가 적응

**Per-layer CKA scores (forget-set examples 기준):**

```
CKA_full_unl(l)    = CKA(H_full^l,    H_unl^l)     # layer l에서 full vs unlearned
CKA_full_retain(l) = CKA(H_full^l, H_retain^l)     # layer l에서 full vs retain (baseline)
```

where H_model^l in R^{n x d} is the matrix of hidden states at layer l for n forget-set examples.

**Normalized per-layer score:**
```
s_l^CKA = clip( (CKA_full_unl(l) - CKA_full_retain(l)) / (1 - CKA_full_retain(l)), 0, 1 )

  CKA_full_unl이 1에 가까움 (full과 유사)  => s_l이 1에 가까움 => 지식 유지
  CKA_full_unl이 CKA_full_retain에 가까움  => s_l이 0에 가까움 => 지식 삭제

Unlearning score로 반전 (1 = 삭제):
  erasure_l^CKA = 1 - s_l^CKA
```

**대안 공식 (direct retain-similarity):**
```
CKA_retain_unl(l) = CKA(H_retain^l, H_unl^l)   # layer l에서 retain vs unlearned

# CKA_retain_unl이 높을수록 = retain과 유사 = 더 많이 삭제
erasure_l^CKA = clip(
    (CKA_retain_unl(l) - CKA_retain_full(l)) / (1 - CKA_retain_full(l)),
    0, 1
)
```

두 공식 모두 유효하다. 첫 번째는 "unlearned model이 여전히 full model과 비슷한가?"를 묻고, 두 번째는 "unlearned model이 retain model과 비슷한가?"를 묻는다.

### 2.4 Layer-wise Computation Pipeline

```
For each layer l in [0, ..., L-1]:
    1. 모든 n forget-set examples를 full model에 통과     -> H_full^l    in R^{n x d}
    2. 모든 n forget-set examples를 retain model에 통과   -> H_retain^l  in R^{n x d}
    3. 모든 n forget-set examples를 unlearned model에 통과 -> H_unl^l     in R^{n x d}
    4. CKA(H_full^l, H_unl^l)와 CKA(H_full^l, H_retain^l) 계산
    5. 정규화하여 erasure_l^CKA 산출
```

**Position 선택**: Entity-span positions (UDS와 동일) 또는 전체 token positions 사용. UDS와의 일관성 및 knowledge-bearing representation에 집중하기 위해 entity-span 권장.

### 2.5 Weighted-Sum Aggregation

```
CKA_Score = Sum_{l in FT} [ w_l * erasure_l^CKA ] / Sum_{l in FT} [ w_l ]

where w_l = Delta^S1_l (UDS S1 stage의 FT-layer 가중치 재사용)
      FT = { l : Delta^S1_l > tau }
```

UDS와 동일한 FT layers 및 가중치를 사용하면 비교 가능성이 보장된다. 대안으로, `CKA_full_retain(l)`이 특정 threshold 이하인 layer를 CKA 자체의 "knowledge layers"로 정의할 수도 있다 (해당 layer가 forget-set-specific 정보를 담고 있음을 의미).

### 2.6 Pros and Cons vs Activation Patching

| Aspect | CKA | Activation Patching (UDS) |
|--------|-----|---------------------------|
| **측정 대상** | Representation 분포의 기하학적 유사도 | Representation이 output에 미치는 causal effect |
| **Causal** | No (correlational) | Yes (interventional) |
| **민감도** | 전체 geometry가 유사하면 미세한 지식을 놓칠 수 있음 | Log-prob degradation을 직접 측정 |
| **계산량** | Forward pass 총 3회 (full, retain, unlearned) | 2L+1 forward passes (stage당 layer별 1회) |
| **Token-level resolution** | 임의의 position subset 사용 가능 | Entity span에서 자연스럽게 동작 |
| **Failure modes** | Low-rank subspace에 저장된 지식에 둔감 | Patching이 computation을 disrupting하면 erasure 과대평가 가능 |
| **Batch-friendly** | 매우 좋음 (캐시된 representation에 대한 행렬 연산) | Layer별 hook 필요 |

### 2.7 Key References

- Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited." ICML 2019. arXiv:1905.00414
- Nguyen et al. (2021). "Do Wide Neural Networks Really Need to be Wide? A Scalable Analysis of CKA." NeurIPS 2021.
- Raghu et al. (2021). "Do Vision Transformers See Like Convolutional Neural Networks?" NeurIPS 2021. (Applied CKA to transformers)

---

## 3. Method 2: Fisher Information Distance (FID-Score)

### 3.1 개요

Fisher Information Matrix (FIM)은 모델의 output distribution이 파라미터 변화에 얼마나 민감한지를 정량화한다. Unlearning 평가에서의 핵심 아이디어: **지식이 진정으로 삭제되었다면, forget-set examples에 대한 모델의 파라미터 민감도가 retain model의 민감도와 유사해야 한다** (즉, forget-set examples에 대해 동등하게 "무지"해야 한다).

Fisher information은 또한 어떤 파라미터가 (따라서 어떤 layer가) 특정 지식 인코딩에 가장 중요한지를 측정하는 원칙적인 방법을 제공한다.

### 3.2 Mathematical Formulation

**Fisher Information Matrix (diagonal approximation):**

For a model with parameters theta and dataset D = {(x_i, y_i)}:

```
F_theta = E_{(x,y)~D} [ nabla_theta log p(y|x; theta) * nabla_theta log p(y|x; theta)^T ]
```

실제로는 diagonal approximation (파라미터별 gradient variance)을 사용:

```
F_theta^diag = (1/|D|) Sum_{i=1}^{|D|} (d log p(y_i | x_i; theta) / d theta)^2
```

특정 layer l의 파라미터 theta_l에 대해:

```
F_l(theta, D) = (1/|D|) Sum_i || nabla_{theta_l} log p(y_i | x_i; theta) ||^2
```

이는 dataset D에 대한 평균 squared gradient magnitude를 layer별 scalar로 제공한다.

### 3.3 Unlearning 평가 적응

**Forget set에 대한 per-layer Fisher scores:**

```
F_full(l)    = F_l(theta_full,    D_forget)    # layer l에서 full model의 forget data 민감도
F_retain(l)  = F_l(theta_retain,  D_forget)    # layer l에서 retain model의 forget data 민감도
F_unl(l)     = F_l(theta_unl,     D_forget)    # layer l에서 unlearned model의 forget data 민감도
```

**Normalized per-layer score:**

```
# Full model의 초과 민감도 중 얼마나 제거되었는가?
excess_full(l)  = max(F_full(l) - F_retain(l), 0)    # Full model의 지식 신호
excess_unl(l)   = max(F_unl(l)  - F_retain(l), 0)    # Unlearned model의 잔여 신호

erasure_l^Fisher = 1 - clip(excess_unl(l) / excess_full(l), 0, 1)   if excess_full(l) > 0
                 = 1                                                  otherwise (신호 자체가 없음)
```

**해석:**
- `erasure_l = 1`: Unlearned model이 retain 수준의 민감도를 가짐 (지식 삭제됨)
- `erasure_l = 0`: Unlearned model이 full model 수준의 민감도를 가짐 (지식 유지)

### 3.4 Layer-wise Computation Pipeline

```
For each layer l:
    1. Forget set에 대한 log-likelihood의 per-example gradients 계산
       (full, retain, unlearned 세 모델)
    2. Layer별 mean squared gradient norm (diagonal Fisher) 계산
    3. Retain을 baseline으로 정규화

구현 상세:
    - torch.autograd.grad with create_graph=False (효율성)
    - Minibatch 누적으로 메모리 관리
    - Forget-set examples에만 집중 (UDS와 동일)
    - Target tokens: entity span tokens (UDS와 동일 positions)
```

**Transformer layer에 대한 효율적 계산:**

```python
# Per-layer Fisher (diagonal approx) on forget set
for layer_idx in range(num_layers):
    layer_params = list(model.model.layers[layer_idx].parameters())
    fisher_sum = 0.0
    for batch in forget_dataloader:
        log_probs = compute_entity_logprobs(model, batch)  # [B]
        for lp in log_probs:
            grads = torch.autograd.grad(lp, layer_params, retain_graph=True)
            fisher_sum += sum((g ** 2).sum().item() for g in grads)
    F_l = fisher_sum / len(forget_dataset)
```

### 3.5 Weighted-Sum Aggregation

```
Fisher_Score = Sum_{l in FT} [ w_l * erasure_l^Fisher ] / Sum_{l in FT} [ w_l ]
```

**FT layer 식별을 Fisher 고유 방식으로:**
```
FT_layers_Fisher = { l : F_full(l) - F_retain(l) > tau_F }
```

또는 직접 비교를 위해 UDS FT layers 재사용.

**가중치:**
- Option A: `w_l = Delta^S1_l` (UDS 가중치 재사용)
- Option B: `w_l = excess_full(l)` (Fisher 고유: 지식 중요도로 가중)

### 3.6 Pros and Cons vs Activation Patching

| Aspect | Fisher Information | Activation Patching (UDS) |
|--------|-------------------|---------------------------|
| **측정 대상** | Forget data에 대한 파라미터 민감도 | Representation이 output에 미치는 causal effect |
| **Granularity** | Parameter-level (MLP vs attention 세분화 가능) | Representation-level (full hidden state) |
| **Causal** | No (gradient structure 측정) | Yes (interventional) |
| **계산량** | Backward passes 필요 (비쌈) | Forward passes만 (hooks 사용) |
| **메모리** | 높음 (per-example gradients) | 중간 (hidden state caching) |
| **Failure modes** | Low-Fisher 파라미터에 저장된 지식 탐지 불가 | Patching이 computation disrupting 시 과대평가 |
| **Sub-layer analysis** | 자연스러움 (MLP, attention, 특정 weight matrices) | 별도의 MLP/attention patching 필요 |

### 3.7 Key References

- Kirkpatrick et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." PNAS. (EWC / Fisher for continual learning)
- Golatkar et al. (2020). "Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks." CVPR 2020.
- Peste et al. (2021). "SSSE: Efficiently Erasing Samples from Trained Machine Learning Models." arXiv:2107.03860
- Dai et al. (2022). "Knowledge Neurons in Pretrained Transformers." ACL 2022. arXiv:2104.08696

---

## 4. Method 3: Linear Probing Score (LPS)

### 4.1 개요

Linear probing은 frozen model representation 위에 간단한 linear classifier (또는 regressor)를 학습시켜 특정 정보가 linearly decodable한지 테스트한다. Unlearning 평가 관점: **지식이 진정으로 삭제되었다면, unlearned model의 representation에 학습된 linear probe가 forgotten information을 복원하지 못해야 하며, 이는 retain model을 probing하는 것과 동일한 결과를 보여야 한다.**

이는 representation에 정보가 *존재*하는지를 가장 직접적으로 테스트하는 방법이다. Activation patching (downstream computation에 *영향*을 주는지)이나 CKA (*기하학적으로 유사*한지)와는 다른 관점을 제공한다.

### 4.2 Mathematical Formulation

**Probe setup:**

Given hidden states h_l^model(x_i) at layer l for input x_i, train a linear probe:

```
y_hat_i = W_l * h_l^model(x_i) + b_l

where W_l in R^{C x d}, b_l in R^C
      C = class 수 (또는 token prediction의 경우 vocabulary size)
      d = hidden dimension
```

**Training objective:**

Entity prediction (entity 분류)의 경우:
```
L_probe = - (1/n) Sum_i log softmax(W_l * h_l(x_i) + b_l)[y_i]
```

Token-level prediction (entity token 예측)의 경우:
```
L_probe = - (1/n) Sum_i Sum_{t in entity_span} log softmax(W_l * h_l^t(x_i) + b_l)[y_i^t]
```

**Probe accuracy:**
```
acc_l^model = (1/n) Sum_i 1[argmax(W_l * h_l^model(x_i) + b_l) == y_i]
```

### 4.3 Unlearning 평가 적응

**핵심 설계 결정**: Full model representation에서 probe를 학습한 후, unlearned/retain model representation에서 평가한다.

**Pipeline:**

```
1. Full model에서 forget set의 hidden states 추출    -> H_full^l
2. Retain model에서 forget set의 hidden states 추출  -> H_retain^l
3. Unlearned model에서 forget set의 hidden states 추출 -> H_unl^l

4. H_full^l에서 entity label 예측하는 linear probe 학습
   (forget set 내 train/val split 또는 cross-validation)

5. Probe accuracy 평가:
   acc_full(l)    = H_full^l에서의 probe accuracy (지식 존재 시 높아야 함)
   acc_retain(l)  = H_retain^l에서의 probe accuracy (baseline: "지식 없음")
   acc_unl(l)     = H_unl^l에서의 probe accuracy (테스트: 지식 복원 가능한가?)
```

**Normalized per-layer score:**
```
erasure_l^probe = 1 - clip(
    (acc_unl(l) - acc_retain(l)) / (acc_full(l) - acc_retain(l)),
    0, 1
)

  acc_unl이 acc_full에 가까움    => erasure가 0에 가까움 (지식 유지)
  acc_unl이 acc_retain에 가까움  => erasure가 1에 가까움 (지식 삭제)
```

**대안: Accuracy 대신 probe loss 사용:**
```
# Loss가 낮을수록 = 정보가 더 많이 존재
erasure_l^probe = clip(
    (loss_unl(l) - loss_full(l)) / (loss_retain(l) - loss_full(l)),
    0, 1
)
```

### 4.4 Layer-wise Computation Pipeline

```
For each layer l in [0, ..., L-1]:
    1. H_full^l = {h_l^full(x_i) : i in forget_set} 수집
       - Entity-span의 경우: entity token positions에서 hidden states 평균
       - 결과: (n, d) matrix

    2. H_retain^l, H_unl^l도 동일하게 수집

    3. Label y_i 정의 (entity class 또는 entity token identity)

    4. Linear probe 학습:
       - Input: H_full^l (또는 train/val split)
       - Target: y_i
       - Optimizer: L-BFGS or SGD, few epochs
       - Regularization: 과적합 방지를 위한 L2 penalty

    5. H_full^l, H_retain^l, H_unl^l에서 probe 평가
       -> acc_full(l), acc_retain(l), acc_unl(l)

    6. erasure_l^probe 계산
```

**Label 설계 옵션:**

| Task | Labels | Granularity |
|------|--------|-------------|
| Entity classification | Entity string identity (TOFU에서 40개 고유 entity) | Coarse |
| Token prediction | Entity positions에서의 next-token ID | Fine (UDS와 align) |
| Binary (knows/doesn't know) | 답이 맞으면 1, 아니면 0 | Coarsest |
| Attribute classification | 특정 속성 (profession, birthplace 등) | Medium |

### 4.5 Weighted-Sum Aggregation

```
LPS = Sum_{l in FT} [ w_l * erasure_l^probe ] / Sum_{l in FT} [ w_l ]
```

FT layers는 `acc_full(l) - acc_retain(l) > tau_probe`인 layer로 정의 가능 (full에서는 linearly decodable하지만 retain에서는 아닌 layer).

### 4.6 Pros and Cons vs Activation Patching

| Aspect | Linear Probing | Activation Patching (UDS) |
|--------|----------------|---------------------------|
| **측정 대상** | 지식의 linear decodability | Output logits에 대한 causal effect |
| **Causal** | No (decodability != usage) | Yes |
| **학습 필요** | Yes (layer별 probe 학습) | No |
| **민감도** | Linearly separable한 정보에 한정 | Downstream layers가 사용하는 모든 정보 포착 |
| **Failure modes** | Non-linearly 인코딩된 지식 탐지 불가 | Computation disruption으로 erasure 과대평가 가능 |
| **해석 용이성** | 매우 직관적 (지식을 읽어낼 수 있는가?) | 다소 기술적 (patching 메커니즘) |
| **재현성** | Probe 학습에 의존 (hyperparameters, splits) | Deterministic (학습 없음) |
| **계산량** | 중간: forward 3회 + L번 probe 학습 | 2L+1 forward passes (backward 없음) |

### 4.7 Key References

- Alain & Bengio (2017). "Understanding Intermediate Layers Using Linear Classifier Probes." ICLR 2017 Workshop.
- Belinkov (2022). "Probing Classifiers: Promises, Shortcomings, and Advances." Computational Linguistics.
- Meng et al. (2022). "Locating and Editing Factual Associations in GPT." NeurIPS 2022. arXiv:2202.05262 (ROME; causal tracing as complementary approach)
- Geva et al. (2021). "Transformer Feed-Forward Layers Are Key-Value Memories." EMNLP 2021.
- Hernandez et al. (2024). "Linearity of Relation Representations in Transformer LMs." ICLR 2024.

---

## 5. Method 4: Representation Dissimilarity Analysis (RDA) via SVCCA / Projection-Weighted CCA

### 5.1 개요

SVCCA (Singular Vector Canonical Correlation Analysis)와 그 변형 PWCCA (Projection Weighted CCA)는 두 representation space의 주성분 방향 정렬을 측정한다. Gram matrix를 비교하는 CKA와 달리, CCA 기반 방법은 두 representation matrix의 linear projection 간 최대 상관관계를 찾는다.

**핵심**: Unlearned model representation의 (forget-set inputs에 대한) 부분공간이 retain model의 부분공간을 향해 수렴하면, 주성분에서 지식이 삭제된 것이다.

### 5.2 Mathematical Formulation

**CCA (Canonical Correlation Analysis):**

Given centered representation matrices X in R^{n x p} and Y in R^{n x q}:

```
Find projection directions a_k, b_k that maximize:
    rho_k = corr(X a_k, Y b_k)

subject to:
    (X a_k)^T (X a_j) = delta_{kj}
    (Y b_k)^T (Y b_j) = delta_{kj}
```

이는 min(p, q)개의 canonical correlations rho_1 >= rho_2 >= ... >= rho_m을 산출한다.

**SVCCA (Raghu et al. 2017):**

```
1. X에 대한 Truncated SVD -> X' = U_x[:, :k_x] * S_x[:k_x, :k_x] * V_x[:, :k_x]^T
   Variance의 99%를 설명하는 상위 k_x singular vectors 유지 (또는 고정 k)

2. Y에 대해 동일하게 Truncated SVD -> Y'

3. X', Y'에 대해 CCA -> canonical correlations rho_1, ..., rho_m

4. SVCCA similarity = (1/m) Sum_k rho_k
```

**PWCCA (Morcos et al. 2018):**

```
PWCCA는 각 canonical direction이 원래 representation에 얼마나 중요한지로
canonical correlations를 가중하여 SVCCA를 개선한다:

1. 각 canonical pair에 대해 CCA direction h_k = X a_k 계산
2. alpha_k = Sum_i |h_k^T x_i| / Sum_k Sum_i |h_k^T x_i| 계산
   (direction k가 실제 representation에 기여하는 정도로 가중)
3. PWCCA = Sum_k alpha_k * rho_k
```

### 5.3 Unlearning 평가 적응

**Per-layer SVCCA/PWCCA scores:**

```
sim_full_unl(l)    = SVCCA(H_full^l, H_unl^l)
sim_full_retain(l) = SVCCA(H_full^l, H_retain^l)
sim_retain_unl(l)  = SVCCA(H_retain^l, H_unl^l)
```

**Normalized per-layer erasure:**

```
# Option A: Unlearned model이 full에서 retain 방향으로 얼마나 이동했는가
erasure_l^SVCCA = clip(
    (sim_full_unl(l) - 1.0) / (sim_full_retain(l) - 1.0),
    0, 1
)
# Note: sim in [0, 1], sim=1은 동일을 의미. Unlearned model이 retain만큼
# full과 다르면 erasure = 1.

# Option B: Direct retain-similarity approach
erasure_l^SVCCA = clip(
    (sim_retain_unl(l) - sim_retain_full(l)) / (1 - sim_retain_full(l)),
    0, 1
)
# Unlearned가 retain 자체만큼 retain과 유사하면 (=1), erasure = 1.
```

### 5.4 Layer-wise Computation Pipeline

```
For each layer l:
    1. Forget-set examples에서 H_full^l, H_retain^l, H_unl^l 추출
       (Entity-span positions 사용: entity tokens에 대해 average 또는 concatenate)

    2. 각 matrix centering: H <- H - mean(H, axis=0)

    3. SVD truncation:
       U, S, V = SVD(H_full^l)
       k = min k such that Sum(S[:k]^2) / Sum(S^2) >= 0.99
       H_full_trunc^l = U[:, :k] * S[:k]

    4. Truncated representations 간 CCA
       -> canonical correlations rho_1, ..., rho_m

    5. SVCCA = mean(rho) or PWCCA = weighted mean(rho)

    6. Erasure score로 정규화
```

### 5.5 Weighted-Sum Aggregation

```
SVCCA_Score = Sum_{l in FT} [ w_l * erasure_l^SVCCA ] / Sum_{l in FT} [ w_l ]
```

### 5.6 SVCCA vs CKA: 선택 기준

| Property | SVCCA/PWCCA | CKA |
|----------|-------------|-----|
| Invariance | Invertible linear transforms | Orthogonal transforms + isotropic scaling |
| Scale 민감도 | 없음 (CCA가 제거) | 부분적 (normalized이나 완전 불변은 아님) |
| Sample 요구량 | n > d 필요 (또는 SVD truncation) | n < d에서도 잘 동작 |
| Dimensionality | 부분공간 겹침을 명시적으로 추적 | Kernel 유사도 추적 |
| Stability | Sample이 적으면 불안정 가능 | 더 안정적 |
| Unlearning 용도 | 부분공간 분석에 적합 | 전체 geometry 비교에 적합 |

### 5.7 Key References

- Raghu et al. (2017). "SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability." NeurIPS 2017. arXiv:1706.05806
- Morcos et al. (2018). "Insights on Representational Similarity in Neural Networks with Canonical Correlation." NeurIPS 2018. (PWCCA)
- Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited." ICML 2019. (CKA, comparison with CCA)

---

## 6. Method 5: Representation-Level Mutual Information (MI) via Logit Lens / Tuned Lens

### 6.1 개요

Logit lens (nostalgebraist, 2020)와 tuned lens (Belrose et al., 2023)는 중간 hidden states를 vocabulary space로 projection하여 모델이 각 layer에서 무엇을 "믿고 있는지" 확인한다. Unlearning 평가 관점: **지식이 삭제되었다면, forget-set entity tokens에 대한 intermediate-layer logits가 더 이상 정답에 높은 확률을 부여하지 않아야 한다.**

이 방법은 activation patching과 수학적으로 관련되지만 메커니즘이 다르다: patching하여 downstream effects를 측정하는 대신, 각 layer에서 vocabulary logits로 projection하여 어떤 정보가 존재하는지 직접 읽어낸다.

### 6.2 Mathematical Formulation

**Logit Lens (nostalgebraist, 2020):**

```
logits_l(x) = LN(h_l(x)) * W_unembed

where h_l(x) = layer l에서의 hidden state for input x
      LN     = final layer norm
      W_unembed = unembedding matrix (lm_head weights)
```

**Tuned Lens (Belrose et al., 2023):**

```
logits_l(x) = LN(A_l * h_l(x) + b_l) * W_unembed

where A_l, b_l = layer별 learned affine parameters
      (layer-l hidden states에서 final-layer logits를 예측하도록 학습)
```

**Per-layer entity log-probability:**

```
lp_l^model(x_i) = (1/|E_i|) Sum_{t in E_i} log softmax(logits_l^model(x_i))_t [y_i^t]

where E_i = example i의 entity span token positions
      y_i^t = position t에서의 ground-truth entity token
```

### 6.3 Unlearning 평가 적응

**Forget set에 대한 per-layer scores:**

```
lp_full(l)    = mean_i [ lp_l^full(x_i) ]
lp_retain(l)  = mean_i [ lp_l^retain(x_i) ]
lp_unl(l)     = mean_i [ lp_l^unl(x_i) ]
```

**Normalized per-layer erasure:**

```
# Entity probability가 full에서 retain 수준으로 얼마나 감소했는가?
erasure_l^lens = clip(
    (lp_full(l) - lp_unl(l)) / (lp_full(l) - lp_retain(l)),
    0, 1
)

  lp_unl이 lp_full에 가까움    => erasure ~ 0 (layer l에서 지식 유지)
  lp_unl이 lp_retain에 가까움  => erasure ~ 1 (layer l에서 지식 삭제)
```

### 6.4 UDS와의 관계

Logit/tuned lens 접근법은 UDS와 수학적으로 관련되지만 메커니즘이 다르다:

| Aspect | UDS (Activation Patching) | Logit/Tuned Lens |
|--------|---------------------------|------------------|
| **메커니즘** | 개입: source h_l을 target에 patch, output logits 측정 | 관찰: h_l을 직접 vocab space로 projection |
| **Causal?** | Yes (counterfactual) | No (observational) |
| **포착 대상** | h_l이 모델 output에 *causal하게 영향*을 주는지 | h_l이 정답 정보를 *포함*하는지 |
| **Failure mode** | Patching이 computation을 disrupting 가능 | Logit lens는 근사치 (초기 layers에서 calibration 안됨) |
| **불일치 시** | h_l에 정보가 있지만 downstream layers가 사용하지 않음 | h_l에 지식이 존재하고 탐지됨, 하지만 output에 영향 안 줄 수 있음 |

**핵심 장점**: Logit lens는 모델당 single forward pass로 매우 빠르고, activation patching은 layer당 하나의 forward pass가 필요하다.

### 6.5 Layer-wise Computation Pipeline

```
For each model M in {full, retain, unlearned}:
    1. output_hidden_states=True로 single forward pass
       -> h_0, h_1, ..., h_L (각 example별)

    2. For each layer l:
       a. Final LayerNorm 적용: h_l' = LN(h_l)
       b. Projection: logits_l = h_l' @ W_unembed
       c. Entity tokens에 대한 log-probs 추출
       d. Mean entity log-prob 계산: lp_l^M

3. 모델 간 정규화하여 erasure_l^lens 산출
```

**Tuned Lens의 경우**: Affine transforms A_l, b_l을 calibration set (forget set 아닌)에서 사전 학습 필요. Retain model의 training data가 적합하다.

### 6.6 Weighted-Sum Aggregation

```
Lens_Score = Sum_{l in FT} [ w_l * erasure_l^lens ] / Sum_{l in FT} [ w_l ]
```

FT layers는 `lp_full(l) - lp_retain(l) > tau_lens`인 layer로 식별 가능.

### 6.7 Key References

- nostalgebraist (2020). "Interpreting GPT: the Logit Lens." LessWrong blog post.
- Belrose et al. (2023). "Eliciting Latent Predictions from Transformers with the Tuned Lens." NeurIPS 2023. arXiv:2303.08112
- Geva et al. (2022). "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space." EMNLP 2022.
- Dar et al. (2023). "Analyzing Transformers in Embedding Space." ACL 2023.

---

## 7. Method 6: Representation Similarity Analysis (RSA)

### 7.1 개요

RSA (Kriegeskorte et al., 2008)는 먼저 각 모델에 대해 Representational Dissimilarity Matrix (RDM) — examples 간 pairwise distance의 n x n matrix — 을 계산한 후, 모델 간 RDM을 비교하여 representation을 비교한다. Unlearned model의 RDM (forget-set inputs 기준)이 retain model의 RDM과 비슷하고 (full model의 RDM과 다르면), 지식이 구조적으로 삭제된 것이다.

RSA는 뇌 활동 패턴과 모델 representation을 비교하기 위해 computational neuroscience에서 시작되었다. Unlearning 평가에의 적용은 자연스럽다: 절대적 representation 값이 아닌 examples 간의 *관계 구조*를 비교하기 때문이다.

### 7.2 Mathematical Formulation

**Representational Dissimilarity Matrix (RDM):**

```
RDM^model_l [i, j] = dist(h_l^model(x_i), h_l^model(x_j))

where dist can be:
  - 1 - cosine_similarity(h_i, h_j)     (cosine distance)
  - ||h_i - h_j||_2                      (Euclidean distance)
  - 1 - corr(h_i, h_j)                  (correlation distance)
```

**두 모델 간 RSA similarity:**

```
RSA(M1, M2, l) = corr(vec(RDM^M1_l), vec(RDM^M2_l))

where vec() = RDM 상삼각 부분 벡터화
      corr() = Pearson 또는 Spearman rank correlation
```

Outlier에 더 robust한 Kendall's tau도 사용 가능.

### 7.3 Unlearning 평가 적응

**Per-layer RSA scores:**

```
RSA_full_unl(l)    = RSA(full, unlearned, l)
RSA_full_retain(l) = RSA(full, retain, l)
RSA_retain_unl(l)  = RSA(retain, unlearned, l)
```

**Normalized per-layer erasure:**

```
# Unlearned model의 관계 구조가 full에서 얼마나 벗어났는가?
erasure_l^RSA = clip(
    (RSA_full_unl(l) - 1.0) / (RSA_full_retain(l) - 1.0),
    0, 1
)

# 또는: unlearned가 retain과 얼마나 유사한가?
erasure_l^RSA = clip(
    (RSA_retain_unl(l) - RSA_retain_full(l)) / (1 - RSA_retain_full(l)),
    0, 1
)
```

### 7.4 Layer-wise Computation Pipeline

```
For each layer l:
    1. Forget set에서 representation 추출:
       H_full^l, H_retain^l, H_unl^l   (각 n x d)

    2. RDM 계산:
       RDM_full^l    = pairwise_distances(H_full^l)     (n x n)
       RDM_retain^l  = pairwise_distances(H_retain^l)   (n x n)
       RDM_unl^l     = pairwise_distances(H_unl^l)      (n x n)

    3. 상삼각 벡터화:
       v_full    = upper_tri(RDM_full^l)     (n*(n-1)/2 vector)
       v_retain  = upper_tri(RDM_retain^l)
       v_unl     = upper_tri(RDM_unl^l)

    4. 상관관계 계산:
       RSA_full_unl(l)    = spearman_corr(v_full, v_unl)
       RSA_full_retain(l) = spearman_corr(v_full, v_retain)
       RSA_retain_unl(l)  = spearman_corr(v_retain, v_unl)

    5. Erasure score로 정규화
```

**계산 참고**: n=367 examples에서 각 RDM은 367 x 367 = ~67K pairwise distances. 상삼각은 ~67K entries. 충분히 처리 가능.

### 7.5 Weighted-Sum Aggregation

```
RSA_Score = Sum_{l in FT} [ w_l * erasure_l^RSA ] / Sum_{l in FT} [ w_l ]
```

### 7.6 RSA vs CKA

RSA와 CKA는 밀접하게 관련되어 있다. Kornblith et al. (2019)는 linear kernel CKA가 distance matrix가 아닌 Gram matrix (K = XX^T) 간의 상관관계를 계산하는 것과 동치임을 보였다. 주요 차이점:

| Property | RSA | CKA |
|----------|-----|-----|
| 비교 대상 | Pairwise distances | Kernel (dot product) matrices |
| Correlation type | 보통 Spearman rank | Normalized HSIC (Pearson-like) |
| Metric 민감도 | Distance function 선택에 의존 | 고정 (linear kernel) |
| 기원 | Computational neuroscience | Machine learning |
| Robustness | Rank correlation으로 더 robust | 통계적으로 더 효율적 |

### 7.7 Key References

- Kriegeskorte et al. (2008). "Representational Similarity Analysis - Connecting the Branches of Systems Neuroscience." Frontiers in Systems Neuroscience.
- Kriegeskorte & Kievit (2013). "Representational geometry: integrating cognition, computation, and the brain." Trends in Cognitive Sciences.
- Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited." ICML 2019. (Comparison of RSA with CKA)

---

## 8. Comparison Table

| Method | Causal? | Training Required? | Forward Passes | Backward Passes | What It Measures | Token Resolution |
|--------|---------|-------------------|----------------|-----------------|------------------|-----------------|
| **UDS (Activation Patching)** | Yes | No | 2L+1 per model pair | 0 | Output logits에 대한 causal effect | Entity span |
| **CKA** | No | No | 총 3회 (모델당 1회) | 0 | Representation의 기하학적 유사도 | 임의의 position set |
| **Fisher Information** | No | No | 0 | 3n (n examples x 3 models) | Forget data에 대한 파라미터 민감도 | Entity tokens |
| **Linear Probing** | No | Yes (layer별 probe) | 총 3회 | L x probe_epochs | 지식의 linear decodability | Entity span (averaged) |
| **SVCCA/PWCCA** | No | No | 총 3회 | 0 | 부분공간 정렬 | 임의의 position set |
| **Logit/Tuned Lens** | No | Optional (tuned lens) | 총 3회 | 0 | Layer별 vocabulary-space decodability | Entity tokens |
| **RSA** | No | No | 총 3회 | 0 | Representation의 관계 구조 | 임의의 position set |

### 8.1 추천 방법 조합

1. **Activation Patching (UDS) + Logit Lens**: 둘 다 entity tokens에 대한 정보를 측정하지만 관점이 다름 (causal vs observational). 불일치는 지식이 존재하지만 사용되지 않는 경우 vs output에 능동적으로 영향을 주는 경우를 드러낸다.

2. **CKA + Linear Probing**: CKA는 전체적 기하학적 변화를 측정하고, probing은 정보 내용을 측정한다. 함께 사용하면: "Representation space가 변했는가?" (CKA)와 "지식을 여전히 추출할 수 있는가?" (probing)에 답할 수 있다.

3. **Fisher Information + UDS**: Fisher는 unlearning에 의해 가장 영향받은 layers와 parameters를 식별하고, UDS는 이것이 실제 지식 제거로 이어지는지 검증한다. Fisher는 sub-layer granularity (MLP vs attention)를 제공한다.

---

## 9. Recommended Pipelines

### 9.1 Lightweight Pipeline (UDS와 함께 1-2개 방법 추가)

**최적 후보: CKA + Logit Lens**

근거:
- 둘 다 총 3회의 forward pass만 필요 (모델당 1회, `output_hidden_states=True`)
- 학습 불필요
- CKA는 보완적 기하학적 관점 제공
- Logit lens는 UDS의 causal view에 대한 observational counterpart 제공

```
Pipeline:
1. Full, retain, unlearned 모델을 forget set에 대해 output_hidden_states=True로 실행
   -> 모든 hidden states 캐시 (367 examples x 16 layers x 2048 dim = ~48MB per model)

2. For each layer l:
   a. CKA: Kernel matrices 및 CKA scores 계산
   b. Logit Lens: Vocab space로 projection, entity log-probs 계산

3. Retain/full 기준으로 정규화
4. UDS FT layers 및 가중치로 aggregation
5. 결과 보고: UDS, CKA_Score, Lens_Score
```

**예상 계산량**: 단일 모델 forward pass의 ~3배 (367 examples 기준, UDS에 비해 무시할 수준).

### 9.2 Comprehensive Pipeline (전체 분석)

**모든 방법: CKA + Fisher + Linear Probing + SVCCA + Logit Lens + RSA**

```
Pipeline:
1. Forward passes (CKA, SVCCA, RSA, Logit Lens, Probing에서 공유):
   - Full model:      output_hidden_states=True -> H_full 캐시
   - Retain model:    output_hidden_states=True -> H_retain 캐시
   - Unlearned model:  output_hidden_states=True -> H_unl 캐시

2. Fisher Information (별도, backward passes 필요):
   - 3개 모델에 대해 forget set의 per-layer diagonal Fisher 계산

3. Per-layer analysis:
   a. CKA(H_full^l, H_unl^l), CKA(H_full^l, H_retain^l)
   b. SVD truncation 적용 SVCCA/PWCCA
   c. Cosine distance RDM으로 RSA
   d. Logit lens projection + entity log-probs
   e. H_full^l에서 linear probe 학습, H_unl^l 및 H_retain^l에서 평가

4. 모든 방법을 retain/full 기준으로 정규화
5. FT layers (공유 또는 method-specific)로 method별 aggregation
6. 방법 간 correlation matrix 보고
```

### 9.3 기존 UDS 인프라와의 통합

기존 codebase에서 제공하는 것들:

- **Hidden state extraction**: `uds/core.py::get_all_layers_hidden()` — 한 forward pass에서 모든 지정 layers의 hidden states 추출
- **Entity span identification**: `exp_s1_teacher_forcing.py::get_eval_span()` — entity tokens 위치 식별
- **FT layer identification**: UDS S1 stage에서 `Delta^S1_l > tau`인 layers 식별
- **S1 cache**: `runs/meta_eval/s1_cache_v2.json` — per-example S1 deltas 저장 (가중치로 재사용 가능)
- **Model loading**: `uds/models.py::load_model()` — HuggingFace model loading 처리

새 방법 추가 방법:
1. `get_all_layers_hidden()`으로 representation 추출 (position=None for all positions)
2. `get_eval_span()` 결과로 entity-span positions에 slicing
3. Method-specific similarity/distance를 layer별 계산
4. FT layer 식별 및 가중치에 S1 cache 재사용

---

## 10. 방법 간 상세 수학적 비교: "Knowledge Erasure"의 조작화

### 10.1 "Layer l에 지식이 존재한다"의 의미 (방법별)

| Method | Layer l에 지식이 존재한다는 것은... |
|--------|--------------------------------------|
| **UDS** | Source h_l을 full model에 patching하면 output log-prob가 복원됨 |
| **CKA** | Unlearned reps의 kernel matrix가 full model의 kernel matrix와 유사함 |
| **Fisher** | Layer l의 model parameters가 forget-set gradients에 민감함 |
| **Linear Probe** | h_l의 linear function으로 entity를 예측할 수 있음 |
| **SVCCA** | Unlearned reps의 주성분 부분공간이 full model의 부분공간과 정렬됨 |
| **Logit Lens** | h_l을 vocabulary로 projection하면 entity tokens에 높은 확률 부여 |
| **RSA** | Unlearned reps의 pairwise distance 구조가 full model의 구조와 유사함 |

### 10.2 방법 간 불일치 사례

**Case 1: CKA는 삭제됨, UDS는 유지됨**
- 가능한 설명: 전체 geometry는 변했지만, 핵심 지식을 보존하는 small subspace가 남아있음. CKA는 low-rank signals에 둔감하고, UDS는 causal intervention으로 이를 포착.

**Case 2: Linear probe는 유지됨, UDS는 삭제됨**
- 가능한 설명: h_l에서 지식이 linearly decodable하지만, downstream layers가 이를 무시하도록 수정됨. UDS는 representation이 더 이상 output에 causal하게 영향을 주지 않음을 올바르게 식별.

**Case 3: Logit lens는 삭제됨, UDS는 유지됨**
- 가능한 설명: 지식이 회전된(rotated) 형태로 인코딩되어 logit lens (원래 unembedding matrix 사용)가 decode할 수 없지만, full model의 downstream layers에 activation patching하면 여전히 복원 가능.

**Case 4: Fisher는 삭제됨, probing은 유지됨**
- 가능한 설명: Unlearned model이 forget data에 대해 낮은 gradient sensitivity (Fisher)를 갖는 것은 local minimum에 있기 때문이지만, representation은 여전히 정보를 인코딩 (probing). 이는 모델이 representation에 정보를 기억하지만 loss landscape가 smoothing된 것을 시사.

### 10.3 Evidence Strength 위계

```
가장 강한 erasure evidence:
    1. Activation Patching (UDS)     -- causal, downstream-aware
    2. Linear Probing                -- information-theoretic (linear)
    3. Logit Lens / Tuned Lens       -- observational, vocab-grounded
    4. Fisher Information            -- parameter sensitivity
    5. CKA / SVCCA / RSA            -- geometric similarity (가장 약함)

근거:
- Causal > Observational > Correlational
- Token-level > Representation-level > Parameter-level
```

다만, 각 방법은 보완적 정보를 제공한다. 가장 강한 결론은 여러 방법의 일치에서 나온다.

---

## 11. 실용적 고려사항

### 11.1 Memory Requirements

n=367 forget-set examples, d=2048 hidden dim, L=16 layers 기준:

| Component | Size | Notes |
|-----------|------|-------|
| 모델당 hidden states (전체 layers) | 367 x 16 x 2048 x 4B = ~48 MB | Float32 |
| RDM (RSA용) per layer | 367 x 367 x 4B = ~0.5 MB | On-the-fly 계산 가능 |
| Gram matrix (CKA용) per layer | 367 x 367 x 4B = ~0.5 MB | On-the-fly 계산 가능 |
| Fisher (diagonal) per layer | ~num_params_per_layer x 4B | 1B model에서 ~20-50 MB |
| Linear probe per layer | d x C x 4B | 무시 가능 |

Hidden state caching 총 메모리 (3개 모델): ~144 MB. GPU 또는 CPU 메모리에 충분히 적재 가능.

### 11.2 계산 시간 추정

Llama-3.2-1B-Instruct, 단일 A100 기준:

| Method | 예상 시간 | Bottleneck |
|--------|----------|------------|
| UDS (기존) | 모델당 ~30 min | 2L patching forward passes |
| CKA | 총 ~2 min | 3 forward passes + matrix ops |
| Fisher | 모델당 ~20 min | n backward passes per model |
| Linear Probing | 총 ~5 min | L번 probe 학습 (CPU에서도 가능) |
| SVCCA | 총 ~3 min | Layer별 SVD |
| Logit Lens | 총 ~2 min | 3 forward passes + projection |
| RSA | 총 ~3 min | Pairwise distances + correlation |

### 11.3 통계적 고려사항

- **Sample size**: n=367은 CKA, RSA, probing에 적합 (Kornblith et al. 2019는 n >= 100에서 CKA가 reliable함을 보임). SVCCA는 덜 안정적일 수 있어 PWCCA 권장.
- **Multiple comparisons**: L=16 layers를 비교할 때, 일부 "knowledge layers"가 우연히 나타날 수 있음. FT-layer filtering (UDS S1 기반)이 이를 완화.
- **Confidence intervals**: Examples에 대한 bootstrap resampling으로 모든 방법에 confidence intervals 제공 가능.

---

## 12. References (Complete)

### Representation Similarity Methods
1. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. ICML 2019. arXiv:1905.00414
2. Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. NeurIPS 2017. arXiv:1706.05806
3. Morcos, A.S., Raghu, M., & Bengio, S. (2018). Insights on Representational Similarity in Neural Networks with Canonical Correlation. NeurIPS 2018.
4. Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational Similarity Analysis - Connecting the Branches of Systems Neuroscience. Frontiers in Systems Neuroscience.
5. Nguyen, T., Raghu, M., & Kornblith, S. (2021). Do Wide Neural Networks Really Need to be Wide? A Scalable Analysis of CKA. NeurIPS 2021.
6. Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., & Dosovitskiy, A. (2021). Do Vision Transformers See Like Convolutional Neural Networks? NeurIPS 2021.

### Probing and Interpretability
7. Alain, G. & Bengio, Y. (2017). Understanding Intermediate Layers Using Linear Classifier Probes. ICLR 2017 Workshop. arXiv:1610.01644
8. Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics.
9. nostalgebraist (2020). Interpreting GPT: the Logit Lens. LessWrong blog post.
10. Belrose, N., et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. NeurIPS 2023. arXiv:2303.08112
11. Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. EMNLP 2021.
12. Geva, M., Caciularu, A., Wang, K.R., & Goldberg, Y. (2022). Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space. EMNLP 2022.
13. Dar, G., Geva, M., Gupta, A., & Berant, J. (2023). Analyzing Transformers in Embedding Space. ACL 2023.
14. Hernandez, E., et al. (2024). Linearity of Relation Representations in Transformer LMs. ICLR 2024.

### Knowledge Localization and Editing
15. Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022. arXiv:2202.05262
16. Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F. (2022). Knowledge Neurons in Pretrained Transformers. ACL 2022. arXiv:2104.08696
17. Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405
18. Gurnee, W., et al. (2024). Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models. ICML 2024. arXiv:2401.06102

### Fisher Information and Continual/Machine Learning
19. Kirkpatrick, J., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. PNAS. (EWC)
20. Golatkar, A., Achille, A., & Soatto, S. (2020). Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks. CVPR 2020.
21. Peste, A., et al. (2021). SSSE: Efficiently Erasing Samples from Trained Machine Learning Models. arXiv:2107.03860

### Unlearning Evaluation
22. Li, N., et al. (2024). WMDP: An Evaluation Benchmark for Unlearning. arXiv:2403.03218
23. Deeb, A. & Roger, F. (2024). Do Unlearning Methods Remove Information from Language Model Weights? arXiv:2410.08827
24. Maini, P., Feng, Z., Schwarzschild, A., Lipton, Z.C., & Kolter, J.Z. (2024). TOFU: A Task of Fictitious Unlearning for LLMs. arXiv:2401.06121
25. Meng, K., et al. (2022). Mass-Editing Memory in a Transformer. arXiv:2210.07229 (MEMIT)

### Mechanism / Circuit Analysis
26. Conmy, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. NeurIPS 2023. arXiv:2304.14997
27. Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread.

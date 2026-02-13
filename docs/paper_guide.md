# Paper Guide: UDS — Unlearning Depth Score

> **Working Title**: Measuring Knowledge Recoverability in Unlearned Language Models via Activation Patching
>
> **Target Venue**: EMNLP 2026 (long paper, 8 pages + references + appendix)
>
> **Core Claim**: UDS quantifies the mechanistic depth of unlearning by measuring how much target knowledge is recoverable through activation patching, achieving the highest faithfulness (AUC 0.971) and robustness (HM 0.933) among 20 comparison metrics.

---

## Notation Reference (Paper-Wide)

Use these symbols consistently throughout. Define each at first use.

| Symbol | Definition | First Introduced |
|--------|-----------|------------------|
| $M_{\text{full}}$ | Full model (trained on entire dataset including forget set) | §3.1 |
| $M_{\text{ret}}$ | Retain model (trained without forget set) | §3.1 |
| $M_{\text{unl}}$ | Unlearned model (full model after unlearning) | §3.1 |
| $D_f$ | Forget set (data to be unlearned) | §2.1 |
| $D_r$ | Retain set (data to be preserved) | §2.1 |
| $x$ | Input prompt (question) | §3.1 |
| $y = (y_1, \dots, y_T)$ | Entity span tokens within the ground-truth answer | §3.1 |
| $h^l_M$ | Hidden state of model $M$ at layer $l$ | §3.2 |
| $s^{\text{full}}_t$ | $\log p_{M_{\text{full}}}(y_t \mid x, y_{1:t-1})$ — full model log-prob of entity token $y_t$ | §3.2 |
| $s^{S}_t$ | Log-prob after patching layer $l$ with source $M_S$'s hidden states | §3.2 |
| $\Delta^{S}_l$ | $\frac{1}{T}\sum_t (s^{\text{full}}_t - s^{S}_t)$ — mean log-prob degradation at layer $l$ | §3.2 |
| $\tau$ | FT layer threshold (default 0.05) | §3.2 |
| $\text{FT}_i$ | $\{l : \Delta^{S1}_{i,l} > \tau\}$ — layers where retain lacks target knowledge | §3.2 |
| $\text{UDS}_i$ | Per-example Unlearning Depth Score | §3.3 |

---

## §0. Abstract

**Structure**: [Background] → [Problem] → [Proposal] → [Results] → [Implications] → [Availability]

**Content Guide** (each bracket = 1–2 sentences):

```
[1] Background: Large language model (LLM) unlearning has emerged as a necessary
    post-hoc mechanism
    for regulatory compliance, privacy protection, and copyright content
    removal.

[2] Problem: Existing output-only metrics (memorization scores, membership
    inference attacks) cannot detect knowledge retained in internal
    representations. While recent studies have observed residual knowledge in
    unlearned models' internal states, they lack a generalizable quantitative
    score suitable for systematic evaluation.

[3] Proposal: We propose UDS (Unlearning Depth Score), an activation-patching-
    based interventional metric that quantifies layer-wise knowledge
    recoverability on a 0 (intact) to 1 (erased) scale, without requiring
    additional training.

[4] Results: Across a meta-evaluation of 20 metrics on 150+ unlearned models,
    UDS achieves the highest faithfulness (AUC-ROC 0.971) and robustness
    (HM 0.933 under quantization and relearning attacks), outperforming both
    output-level and alternative representation-level baselines.

[5] Implications: UDS integrates as a plug-in component into existing
    evaluation frameworks and enables example-level diagnostics that reveal
    knowledge retention patterns invisible to output-only assessment.

[6] Our code and benchmark results are publicly available at {URL}.
```

**Key Numbers for Abstract**: AUC 0.971, HM 0.933, 20 metrics, 150+ models, 8 methods

---

## §1. Introduction

### 문단 1 — Hook + Background (~8 lines)

**Content Guide**:
- LLM unlearning의 동기: privacy regulations (GDPR Article 17 right to erasure), safety (hazardous knowledge removal), copyright compliance
- **톤**: "critical"이나 "crucial" 대신 "important", "increasingly relevant" 등 사용
- 다양한 unlearning methods: gradient-based (GradDiff, NPO), preference-based (DPO variants), representation-based (RMU)
- Unlearning의 핵심 목표를 간결하게 정의: forget set의 knowledge를 제거하면서 retain set 성능을 유지하고, **궁극적으로는 forget set을 처음부터 학습하지 않은 모델과 구별 불가능하게** 만드는 것 (gold standard = retain model)

**주의**: 너무 길게 풀지 말 것. 3–4문장으로 motivation + goal 정의.

### 문단 2 — Problem Statement (~10 lines)

**Content Guide**:
- 기존 평가 체계의 한계: output-only metrics (EM, ROUGE, MIA 등)는 모델이 **무엇을 출력하는가**만 측정 → 내부 representation에 knowledge가 남아 있어도 감지 불가
- 구체적 failure mode 예시 하나: "I don't know" 출력 학습 (IdkNLL) → 모든 output metric에서 unlearning 성공으로 판정되지만, 내부에 knowledge가 거의 원본 수준으로 잔존 (UDS 0.04–0.25)
- **Threat model**: An adversary is not limited to querying the model — lightweight fine-tuning or activation manipulation can recover knowledge that appears absent from outputs. Therefore, genuine verification of unlearning must establish not merely "is the knowledge currently suppressed?" but **"is it causally unrecoverable?"** This motivates causal (interventional) evaluation over purely observational metrics.
- 최근 white-box 분석 연구들이 이 문제를 관찰 (Hong et al., 2024; Lynch et al., 2024 등) → 하지만 세 가지 한계:
  1. 범용적인 정량적 score가 아닌 부분적/정성적 분석에 그침
  2. 추가 training이 필요한 경우 존재 (probing, neuron masking 등)
  3. 체계적 meta-evaluation (faithfulness/robustness 검증) 부재

### 문단 3 — Our Contribution (~8 lines)

**Content Guide**:
- UDS 제안: two-stage activation patching으로 layer별 knowledge recoverability를 [0,1] score로 정량화
- **포지셔닝**: 기존 representation 분석이 "knowledge가 남아 있는가"를 관찰(observe)하는 데 그쳤다면, UDS는 "knowledge가 복구 가능한가"를 인과적으로 개입(intervene)하여 측정
- 20개 metrics 대비 meta-evaluation → faithfulness (AUC 0.971) + robustness (HM 0.933) 종합 1위
- 150+ 모델 실증 분석으로 output-only 평가가 놓치는 knowledge retention 패턴 식별

### Contributions (bullets 3개)

```
We contribute:
1. UDS, a causal metric that measures per-example, per-layer knowledge
   recoverability in unlearned LLMs via activation patching, requiring
   no auxiliary training.
2. A comprehensive meta-evaluation framework with symmetric robustness
   formulas, demonstrating UDS achieves the highest faithfulness and
   robustness among 20 metrics including 3 representation-level baselines.
3. Empirical analysis of 150+ models across 8 unlearning methods, identifying
   knowledge retention patterns invisible to output-only evaluation and
   providing guidelines for integrating UDS into existing frameworks.
```

### Figure 1 (p.1 right column)

**Content**: UDS method diagram
- **상단 (S1)**: "Calibration — How deeply is the knowledge encoded?" — Retain model → Full model 패칭, layer $l$에서 hidden states 교체
- **하단 (S2)**: "Measurement — How much knowledge remains recoverable?" — Unlearned model → Full model 패칭
- **오른쪽**: UDS interpretation bar (0.0 = intact ↔ 1.0 = erased)
- **Caption**: "Overview of UDS. Stage 1 calibrates the baseline knowledge gap using the retain model. Stage 2 measures whether the unlearned model exhibits a similar gap, indicating successful erasure."

**Design Note**: Use a simple 2-row diagram with colored arrows showing hidden state flow. Avoid overcrowding with formulas — the figure should be intuitive at a glance.

---

## §2. Background and Related Work

**Section intro** (1–2 sentences):
> "We review LLM unlearning methods and their evaluation (§2.1), then discuss representation-level analysis techniques that motivate our approach (§2.2)."

### §2.1 LLM Unlearning and Evaluation

> **추천 제목**: "LLM Unlearning and Evaluation" (Preliminaries보다 자연스러움. Background 하위 section이므로 별도 Preliminaries section 불필요)

**첫 문단 — Problem Formulation** (~5 lines)

Unlearning 문제를 간결하게 정의. 다른 논문 (예: TOFU, NPO)의 formulation을 참고하되 직접 인용하지 말고 자연스럽게 변형.

```
Given a pretrained model M_θ, a forget set D_f, and a retain set D_r,
the goal of machine unlearning is to produce a model M_θ' such that:
(1) M_θ' behaves as if D_f was never part of training (indistinguishability
    from a model trained only on D_r), while
(2) preserving performance on D_r and general capabilities.
```

이 정의에서 "(1) indistinguishability"가 핵심이며, 이것이 output level에서만 측정되어 왔다는 점을 §2.1 후반에서 연결.

**문단 2 — Methods** (~6 lines)

8개 method를 **카테고리별로** 간결하게 소개:
- **Gradient-based**: GradAscent [Jang et al., 2023], GradDiff, NPO (Zhang et al., COLM 2024), SimNPO (Fan et al., NeurIPS 2025) — forget set에 대한 gradient를 반대 방향으로 적용
- **Preference-based**: IdkNLL, IdkDPO, AltPO (Mekala et al., COLING 2025) — "I don't know" 등 대안 출력을 preference signal로 학습
- **Representation-based**: RMU (Li et al., ICML 2024), UNDIAL (Dong et al., NAACL 2025) — 내부 표현을 직접 조작하여 knowledge 제거

마무리: "These methods differ in objective and mechanism, yet are typically evaluated using the same set of output-level metrics."

**문단 3 — Evaluation** (~6 lines)

- Open-Unlearning (Maini et al., COLM 2024) 프레임워크: 표준 evaluation pipeline 제공
  - Memorization: EM, ES, Prob, ParaProb, Truth Ratio
  - Privacy: MIA variants (LOSS, ZLib, Min-K, Min-K++)
  - Utility: retain set, real authors, world facts
- TOFU benchmark: forget10 설정, 40 fictitious authors
- Meta-evaluation 개념 소개: "Maini et al. further propose meta-evaluation via faithfulness (P/N pool separation) and robustness (stability under perturbation). We build on this framework with modifications (§4)."
- 핵심 한계 한 문장: "However, these metrics capture what the model outputs but cannot assess what it internally retains."

### §2.2 Representation Analysis for Unlearning Verification

**문단 1 — Analysis Methods and Taxonomy** (~10 lines)

Representation 분석 기법을 두 가지 축으로 분류:

**Observational** (관찰적): 모델 내부 상태를 읽기만 하고 행동 변화를 측정하지 않음
- **CKA** (Kornblith et al., ICML 2019): representational geometry의 유사성 측정. Training-free, 빠르지만 knowledge-specific하지 않음
- **Logit Lens** (nostalgebraist, 2020): 중간 layer의 hidden states를 decoder로 projection하여 decodable knowledge 측정
- **Fisher Information** (Kirkpatrick et al., PNAS 2017): parameter sensitivity로 knowledge-relevant 영역 추적
- **Linear Probing** (Patil et al., 2024): auxiliary classifier를 hidden states 위에 학습 → training 필요
- **SVCCA/CCA** (Raghu et al., 2017): canonical correlation 기반 representation 비교

**Interventional** (개입적): 모델 내부 상태를 조작하고 결과적 행동 변화를 측정
- **Activation patching** (causal tracing): Meng et al. (2022)의 ROME에서 factual knowledge localization에 사용. Ghandeharioun et al. (2024)의 Patchscopes에서 수학적으로 일반화
- 차별점: "observational 방법은 *표현이 달라 보이는가*를 측정하지만, interventional 방법은 *표현을 교체하면 행동이 바뀌는가*를 직접 측정"
- **핵심 구분 — Steering vs. Audit**: Activation patching은 model steering (출력 조작)에도 사용되지만, UDS에서는 **audit** (진단) 목적으로 사용. "현재 상태를 있는 그대로 관찰"하는 진단 도구이지, 출력을 바꾸는 개입이 아님

마무리: "UDS adopts the interventional approach, directly measuring whether knowledge can be recovered through patching — a stronger test than observing whether representations differ."

> **참고**: 왜 linear probing / tuned lens를 baseline으로 포함하지 않았는지는 Appendix에서 설명: "We include only methods that do not introduce extra trainable components. We therefore exclude linear probing and tuned lens, since both require supervised fitting of auxiliary modules."

**문단 2 — White-box Unlearning Verification** (~6 lines + Table 1)

기존 연구들이 unlearning 후 내부 표현에 지식이 남아있음을 관찰한 사례들을 정리하고, UDS와의 차별점을 Table 1로 명확화.

### Table 1 — Comparison of White-box Unlearning Analysis (§2.2)

| Work | Analysis Method | Quant. Score | Training-free | Meta-eval |
|------|----------------|:---:|:---:|:---:|
| Hong et al. (2024a) | Activation patching + parameter restoration | Partial | Yes | No |
| Hong et al. (2025) | Concept vector / parametric trace | Yes | Yes | No |
| Liu et al. (2024) | Causal intervention framework | Yes | Mostly | No |
| Guo et al. (2025) | Mechanistic localization + selective editing | Partial | No | No |
| Hou et al. (2025) | FFN neuron masking | Yes | No | No |
| Lo et al. (2024) | Neuron saliency / relearning tracking | Yes | No | No |
| Wang et al. (2025) | Reasoning trace analysis | Yes | Yes | No |
| **UDS (Ours)** | **Activation patching** | **Yes (0–1)** | **Yes** | **Yes** |

**Caption**: "Comparison of white-box unlearning analysis approaches. UDS is the first to provide a per-example quantitative score (0–1) with systematic meta-evaluation (faithfulness + robustness)."

**Table 1 설명** (2–3 lines): "Prior work establishes that unlearning methods frequently leave residual knowledge in internal representations. However, these studies focus on observing or localizing this residual knowledge rather than providing a standardized, per-example score amenable to meta-evaluation. UDS bridges this gap."

### Table 1 References (Verified)

| Short Citation | Full Title | Venue |
|---|---|---|
| Hong et al. (2024a) | Dissecting Fine-Tuning Unlearning in Large Language Models | EMNLP 2024 |
| Hong et al. (2025) | Intrinsic Test of Unlearning Using Parametric Knowledge Traces | EMNLP 2025 |
| Liu et al. (2024) | Revisiting Who's Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective | EMNLP 2024 |
| Guo et al. (2025) | Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization | ICML 2025 |
| Hou et al. (2025) | Decoupling Memories, Muting Neurons: Towards Practical Machine Unlearning for Large Language Models | Findings of ACL 2025 |
| Lo et al. (2024) | Large Language Models Relearn Removed Concepts | Findings of ACL 2024 |
| Wang et al. (2025) | Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills | EMNLP 2025 |

---

## §3. The Unlearning Depth Score

**Section intro** (2–3 sentences):
> "We introduce the Unlearning Depth Score (UDS), a training-free metric that measures knowledge recoverability through activation patching. We define the evaluation setup (§3.1), describe the two-stage patching procedure (§3.2), and present the score aggregation (§3.3)."

> 용어 정리: Following Ghandeharioun et al. (2024), we refer to the model whose hidden states are injected as the **source model** ($M_S$) and the model that receives the patched states as the **target model** ($M_T$). In UDS, the target is always $M_{\text{full}}$.

### §3.1 Task Formulation and Scope

**문단 1 — Models, Input, Scope** (~8 lines)

```
Notation:
- M_full: original model trained on D_r ∪ D_f
- M_ret:  model trained on D_r only (gold standard for complete unlearning)
- M_unl:  model after applying an unlearning method to M_full

Input:
- Question x paired with ground-truth answer continuation (teacher forcing)
- The model processes [x; y] as a single input sequence

Evaluation scope:
- Entity span y = (y_1, ..., y_T) within the answer
- Only knowledge-bearing tokens are measured, excluding template tokens
  ("The answer is...", "is a", etc.)
```

**왜 entity span인지 정당화** (2 lines): "Full-answer evaluation conflates knowledge measurement with language modeling of common phrases. Restricting to entity spans isolates the knowledge-specific signal."

**왜 teacher forcing인지 정당화** (2 lines): "Free-form generation introduces variance from sampling and prompt sensitivity. Teacher forcing provides a deterministic evaluation conditioned on the ground-truth continuation."

### §3.2 Two-Stage Activation Patching

**문단 1 — Core Idea and Patching Mechanics** (~6 lines)

Both models ($M_{\text{full}}$ and $M_S$) process the identical input sequence $[x; y]$ (question + ground-truth answer) via teacher forcing. At a chosen layer $l$, we extract $M_S$'s hidden-state vectors for the **entity-span token positions** and splice them into $M_{\text{full}}$'s forward pass, replacing the corresponding vectors at those same positions. The remaining layers ($l{+}1, \ldots, L$) of $M_{\text{full}}$ then process these foreign vectors through their own attention and MLP blocks. We measure the resulting change in mean log-probability on the entity tokens. If $M_S$ lacks certain knowledge, its representations will carry a different signal at these positions, degrading $M_{\text{full}}$'s output — and the magnitude of this degradation reveals how much knowledge $M_S$ is missing at layer $l$. We repeat this for every layer independently (Figure 1).

**문단 2 — Stage 1: Locating the Knowledge Gap (Retain → Full)** (~6 lines)

Source = $M_{\text{ret}}$. The retain model was never trained on $D_f$, so it lacks forget-set knowledge.

$$\Delta^{S1}_l = \frac{1}{T}\sum_{t=1}^{T} \left(s^{\text{full}}_t - s^{S1}_t\right)$$

where $s^{S1}_t$ is the log-probability after patching layer $l$ with $M_{\text{ret}}$'s hidden states.

**FT layers**: $\text{FT}_i = \{l : \Delta^{S1}_{i,l} > \tau\}$ with $\tau = 0.05$.

직관: "FT layers are where $M_{\text{full}}$ possesses knowledge that $M_{\text{ret}}$ lacks — these are the layers where forget-set knowledge is encoded."

$\tau$의 역할: "The threshold filters out layers with negligible signal, ensuring UDS is computed only over layers with meaningful knowledge encoding. We set $\tau = 0.05$; sensitivity analysis in Appendix C shows results are stable across $\tau \in \{0.02, 0.05, 0.1\}$."

**문단 3 — Stage 2: Measuring Causal Recoverability (Unlearned → Full)** (~4 lines)

Source = $M_{\text{unl}}$. Same procedure, yielding $\Delta^{S2}_l$.

직관: "If the unlearned model successfully erased knowledge at layer $l$, patching its hidden states should degrade $M_{\text{full}}$'s output similarly to patching $M_{\text{ret}}$'s states (i.e., $\Delta^{S2}_l \approx \Delta^{S1}_l$). If knowledge remains intact, $\Delta^{S2}_l \approx 0$."

**문단 4 — S1 Caching** (~3 lines)

S1 is fixed (retain → full) and independent of the unlearning method. We compute it once and cache the per-example, per-layer $\Delta^{S1}$ values and FT layer sets, reusing them across all unlearned models. In our experiments, this caches 367 examples × 16 layers.

### §3.3 Score Aggregation

**문단 1 — UDS Formula** (~6 lines)

$$\text{UDS}_i = \frac{\sum_{l \in \text{FT}_i} \Delta^{S1}_{i,l} \cdot \text{clip}\!\left(\frac{\Delta^{S2}_{i,l}}{\Delta^{S1}_{i,l}},\; 0,\; 1\right)}{\sum_{l \in \text{FT}_i} \Delta^{S1}_{i,l}}$$

**각 구성요소 설명**:
- $\Delta^{S1}_l$-weighted average: layers with larger knowledge gaps (more knowledge encoded) contribute proportionally more to the score
- $\text{clip}(\cdot, 0, 1)$: prevents negative ratios (when patching improves rather than degrades) and caps at 1.0 (when the unlearned model's gap exceeds the retain model's)
- **Interpretation**: UDS = 1.0 means the unlearned model's representations are as knowledge-absent as the retain model's at every FT layer. UDS = 0.0 means knowledge is fully intact (identical to the full model)

**문단 2 — Model-level Aggregation** (~3 lines)

$$\text{UDS} = \frac{1}{N}\sum_{i=1}^{N} \text{UDS}_i$$

We average per-example scores across the forget set. Models with no FT layers for a given example (i.e., retain model already knows the answer) skip that example.

마무리: "We patch full layer outputs (i.e., post-attention + MLP + residual) by default. Appendix C shows that this scope captures 95.3% of the patching signal (mean Δ = 0.953 vs. MLP-only 0.173 and attention-only 0.044)."

---

## §4. Meta-Evaluation

**Section intro** (~4 lines):
> "A good unlearning metric should be both *faithful* (correctly distinguishing models that have vs. lack target knowledge) and *robust* (stable under meaning-preserving perturbations). We evaluate UDS against 19 comparison metrics using the meta-evaluation framework of Maini et al. (2024) with two modifications: (i) we add representation-level baselines and normalized MIA metrics, and (ii) we introduce symmetric robustness formulas (§4.1). We describe the experimental setup (§4.1), comparison metrics (§4.2), faithfulness (§4.3), and robustness (§4.4)."

> **NOTE**: meta-eval 테이블 하나로 통합 — Table 2에 20개 metrics의 Faithfulness AUC, Q, R, HM, Overall을 모두 포함.

### §4.1 Evaluation Protocol and Symmetric Robustness

> **구조 조정**: symmetric formula 동기부여는 **핵심 정당화만 본문에, 상세 전개는 Appendix E로** 분리. 본문 §4.1은 3개 문단 (models/data, faithfulness setup, robustness setup + symmetric formula 요약).

**문단 1 — Models and Dataset** (~5 lines)
- Architecture: Llama-3.2-1B-Instruct (Meta)
- 150 unlearned models: 8 methods × hyperparameter sweep × 2 epochs (5, 10)
  - Methods: GradDiff, IdkNLL, IdkDPO, NPO, AltPO, SimNPO, RMU, UNDIAL
  - Full sweep: Appendix D (Table D.1)
- Evaluation data: TOFU forget10 (367 examples with entity span annotations)
- References: $M_{\text{full}}$, $M_{\text{ret}}$ (retain90)

**문단 2 — Faithfulness Setup** (~4 lines)
- P-pool (30 models): trained on dataset including $D_f$ → knowledge present
- N-pool (30 models): trained without $D_f$ → knowledge absent
- Metric: AUC-ROC — how well each metric separates P from N
- "Higher AUC indicates the metric is better at distinguishing models that genuinely possess target knowledge."

**문단 3 — Robustness Setup** (~8 lines)
- Two perturbation attacks:
  - **Quantization**: NF4 4-bit (BitsAndBytes) — a common deployment technique that should not change what a model "knows"
  - **Relearning**: 1-epoch fine-tuning on $D_f$ (lr=2e-5, effective batch=32) — simulates knowledge recovery attempt
- **Symmetric formulas**: Open-Unlearning의 one-directional formula는 knowledge recovery에 대한 robustness 측정에 효과적이지만, perturbation에 의한 knowledge destruction은 감지하지 못합니다. 예를 들어, quantization이 loss landscape을 변형하여 MIA AUC가 체계적으로 낮아지면, one-directional은 이를 Q=1 (완벽히 robust)로 처리합니다. 우리는 양방향 변화를 모두 측정하는 symmetric formulas를 제안합니다:

**Symmetric formulas** (본문에 수식 제시, 상세 정당화는 Appendix E):

$$Q = 1 - \text{clip}\!\left(\frac{|m_{\text{after}} - m_{\text{before}}|}{|m_{\text{before}}| + |m_{\text{after}}| + \epsilon},\; 0,\; 1\right)$$

$$R = 1 - \text{clip}\!\left(\frac{|\Delta_{\text{unl}} - \Delta_{\text{ret}}|}{|\Delta_{\text{unl}}| + |\Delta_{\text{ret}}| + \epsilon},\; 0,\; 1\right)$$

where $\Delta = m_{\text{after}} - m_{\text{before}}$.

1–2문장으로 핵심 정당화: "These formulas penalize any change from the reference (either recovery or destruction), motivated by two principles: (i) *perturbation invariance* — a meaning-preserving transformation should not alter metric values in either direction, and (ii) *recovery calibration* — after relearning, the unlearned model's metric change should match the retain model's change. See Appendix E for the full derivation."

- **Model filtering**: utility_rel ≥ 0.8 + per-metric faithfulness threshold (filters out models where unlearning itself failed). Details in Appendix E.
- **Aggregation**: Robustness = HM(Q, R); Overall = HM(Faithfulness, Robustness)

### §4.2 Comparison Metrics

**문단 1 — Output-Level Metrics (13개)** (~5 lines)

카테고리별 간결 소개:
- **Memorization** (5): ES, EM, Prob, ParaProb, Truth Ratio — measure whether the model can reproduce or recall target knowledge under various prompting conditions
- **Generation** (3): ROUGE, Para-ROUGE, Jailbreak-ROUGE — ROUGE-L recall against ground-truth under standard, paraphrased, and adversarial prompts
- **MIA** (4 raw): MIA-LOSS, MIA-ZLib, MIA-Min-K, MIA-Min-K++ — per-sample member vs. non-member classification via loss-based statistics
- "All are provided by the Open-Unlearning framework."

**문단 2 — Normalized MIA (4개 추가)** (~4 lines)

Raw MIA의 한계: AUC 절대값만으로는 retain model 대비 상대적 변화를 반영 못함.

$$s_* = \text{clip}\!\left(1 - \frac{|\text{AUC}_{\text{model}} - \text{AUC}_{\text{ret}}|}{\text{AUC}_{\text{ret}}},\; 0,\; 1\right)$$

4종: $s_{\text{LOSS}}, s_{\text{ZLib}}, s_{\text{Min-K}}, s_{\text{Min-K++}}$. Higher $s_*$ = closer to retain = more erasure. Inspired by MUSE PrivLeak rescaling (Shi et al., 2025).

**문단 3 — Representation-Level Baselines (3개)** (~6 lines)

All operate on the same 367-example forget set with entity span annotations, using the retain model as reference:
- **Logit Lens**: projects each layer's hidden states through $M_{\text{full}}$'s frozen decoder to measure per-layer decodable knowledge. Uses the same FT layer weighting as UDS. *Observational*: reads representations without patching.
- **CKA**: compares representational geometry between unlearned and retain models, weighted by full–retain layer importance. *Observational, training-free*.
- **Fisher Masked**: diagonal Fisher Information on $D_f$, masked to top-$p$% knowledge-relevant parameters per layer ($p \in \{0.01\%, 0.1\%, 1\%\}$). Measures parameter-level knowledge sensitivity.
- "Detailed formulas in Appendix F. We exclude linear probing and tuned lens as they require training auxiliary modules."

### Table 2 — Meta-Evaluation Results (§4 main table, 20 rows)

> 이 테이블이 논문의 핵심 결과 테이블. Faithfulness + Robustness + Overall을 한 번에 보여줌.

| Category | Metric | Tags | Faith. ↑ | Q ↑ | R ↑ | Rob. ↑ | Overall ↑ |
|----------|--------|------|------:|------:|------:|------:|------:|
| **Memorization** | ES | O | 0.891 | 0.970 | 0.770 | 0.859 | 0.875 |
| | EM | O | 0.817 | 0.984 | 0.605 | 0.750 | 0.781 |
| | Prob | O | 0.816 | 0.924 | 0.642 | 0.757 | 0.785 |
| | ParaProb | O | 0.707 | 0.853 | 0.899 | 0.875 | 0.782 |
| | Truth Ratio | O | 0.947 | 0.996 | 0.234 | 0.379 | 0.537 |
| **Generation** | ROUGE | O | 0.722 | 0.934 | 0.203 | 0.333 | 0.454 |
| | Para. ROUGE | O | 0.832 | 0.951 | 0.064 | 0.119 | 0.213 |
| | Jailbreak ROUGE | O | 0.757 | 0.971 | 0.183 | 0.308 | 0.437 |
| **MIA** | MIA-LOSS | O | 0.902 | 0.935 | 0.519 | 0.668 | 0.766 |
| | MIA-ZLib | O | 0.867 | 0.938 | 0.487 | 0.642 | 0.738 |
| | MIA-Min-K | O | 0.907 | 0.923 | 0.532 | 0.675 | 0.773 |
| | MIA-Min-K++ | O | 0.816 | 0.883 | 0.431 | 0.579 | 0.678 |
| | s_LOSS | O,R | 0.891 | 0.719 | 0.664 | 0.690 | 0.778 |
| | s_ZLib | O,R | 0.870 | 0.704 | 0.745 | 0.724 | 0.790 |
| | s_Min-K | O,R | 0.891 | 0.710 | 0.697 | 0.704 | 0.787 |
| | s_Min-K++ | O,R | 0.799 | 0.643 | 0.566 | 0.602 | 0.690 |
| **Representation** | CKA | R,I | 0.648 | 0.997 | 0.013 | 0.026 | 0.050 |
| | Fisher Masked (0.1%) | R,I | 0.712 | 0.583 | 0.946 | 0.721 | 0.716 |
| | Logit Lens | O,R,I | 0.927 | 0.959 | 0.812 | 0.879 | 0.902 |
| | **UDS (Ours)** | **O,R,I** | **0.971** | **0.968** | **0.900** | **0.933** | **0.951** |

**Caption**: "Meta-evaluation results across 20 metrics. Faith.=Faithfulness (AUC-ROC), Q=Quantization stability, R=Relearning stability, Rob.=HM(Q,R), Overall=HM(Faith., Rob.). Tags: O=Output-based, R=Retain-referenced, I=Internal representation. **Bold** = best in column. UDS achieves the highest Faithfulness, Robustness, and Overall scores."

> **Notes on Table 2**:
> - Fisher Masked: report only 0.1% variant in main table; 0.01% and 1% in Appendix F (results nearly identical: 0.708, 0.712, 0.698)
> - Tags column은 공간이 부족하면 colored dots으로 대체 가능
> - Truth Ratio: 높은 faithfulness (0.947)이지만 relearning에 극도로 취약 (R=0.234) → Overall 낮음
> - CKA: faithfulness도 낮고 (0.648) robustness도 극도로 낮음 (R=0.013) → Overall 최하위

### §4.3 Faithfulness Results

**문단 1 — Main Results** (~5 lines)

UDS achieves AUC-ROC 0.971, the highest among all 20 metrics. Key comparisons:
- **UDS (0.971)** > Logit Lens (0.927) > Truth Ratio (0.947) > MIA-Min-K (0.907) > MIA-LOSS (0.902)
- Among representation-level metrics: UDS >> Logit Lens >> Fisher Masked (0.712) >> CKA (0.648)
- Among output-only metrics: Truth Ratio leads (0.947) but suffers from poor robustness (see §4.4)

**문단 2 — Why UDS Outperforms Alternatives** (~6 lines)

각 baseline이 왜 UDS보다 낮은지 설명:
- **CKA (0.648)**: representational geometry 유사성만 측정 → unlearning이 전반적 표현 구조를 바꾸면 반응하지만, 특정 knowledge retention과 무관한 변화(예: 학습 경로 차이에 의한 geometry 변화)도 감지하여 P/N 분리가 부정확
- **Fisher Masked (0.712)**: layer 1이 전체 weight의 60–84%를 차지하여 사실상 단일 layer에 의존. 또한 mask fraction(0.01%, 0.1%, 1%)에 거의 무관한 결과 → layer-wise granularity 부족
- **Logit Lens (0.927)**: frozen decoder로 decodable knowledge를 잘 포착하지만 observational이라 **causal recoverability**를 직접 측정 못함. Representation이 변형되었지만 patching하면 복구되는 case를 놓침 (§5.1에서 구체 예시)

**문단 3 — P/N Histogram Analysis** (~3 lines)

Figure 2 참조. UDS는 P-pool (low UDS ≈ 0.49, knowledge intact)과 N-pool (high UDS ≈ 0.85, knowledge absent)이 거의 완벽하게 분리. 반면 output metrics (예: Prob, MIA-LOSS)는 P/N 분포가 상당히 overlap.

### Figure 2 — Faithfulness P/N Histograms (§4.3)

**Content**: 선별 6개 metric의 P/N histogram
- Row 1: 낮은 분리 — Prob (0.816), CKA (0.648)
- Row 2: 중간 분리 — ES (0.891), MIA-Min-K (0.907)
- Row 3: 높은 분리 — Logit Lens (0.927), UDS (0.971)

**Caption**: "P/N pool distributions for selected metrics. P-pool (knowledge present, blue) should differ from N-pool (knowledge absent, orange). UDS achieves near-perfect separation (AUC 0.971). Full histograms for all 20 metrics in Appendix A."

> **Design**: 2×3 grid, shared y-axis per row, dashed vertical lines for optimal threshold. Each subplot title: "Metric (AUC=X.XXX)".

### §4.4 Robustness Results

**문단 1 — Main Results** (~5 lines)

UDS: Q=0.968, R=0.900, HM=0.933 → 종합 1위

Key ranking (by HM):
1. **UDS**: 0.933
2. **Logit Lens**: 0.879
3. **ParaProb**: 0.875
4. **ES**: 0.859
5. **s_ZLib**: 0.724
6. **Fisher (0.1%)**: 0.721

**문단 2 — Why Output Metrics Struggle** (~5 lines)

- **MIA metrics**: quantization에 취약 (Q = 0.88–0.94). NF4가 loss landscape을 변형하여 loss/zlib/min-k 분포가 체계적으로 변화. 이는 지식 변화가 아닌 양자화 artifact
- **ROUGE metrics**: relearning에 극도로 취약 (R = 0.06–0.20). 1 epoch relearning으로 generation quality가 크게 변화하지만, 이는 text generation stochasticity와 prompt sensitivity가 relearning에 의해 증폭되는 것
- **Truth Ratio**: 높은 faithfulness (0.947)에도 불구하고 R=0.234 → HM=0.379. Correct/incorrect probability ratio가 relearning에 매우 민감

**문단 3 — Representation Baselines Analysis** (~5 lines)

- **CKA**: Q=0.997 (quantization에 안정)이지만 R=0.013 → HM=0.026. Relearning 후 representational geometry가 retain model 자체에서도 크게 변함 (retain의 CKA shift ≈ 0.91) → relearning으로 인한 geometry 변화가 knowledge와 무관하게 발생
- **Fisher**: R=0.946 (relearning에 안정)이지만 Q=0.583 → HM=0.721. NF4 양자화가 gradient landscape을 근본적으로 변형하여 Fisher 값이 체계적으로 하락 (대부분 destruction 방향). 이는 dequantize_model() 후 계산해도 발생
- **UDS**: 두 attack 모두에서 높은 안정성. Activation patching은 hidden state 수준에서 작동하므로, 양자화에 의한 parameter-level 변형이 hidden state에 미치는 영향이 상대적으로 적고, relearning에 의한 표면적 generation 변화에도 내부 knowledge encoding 패턴은 안정적

**문단 4 — Scatter Plot Reference** (~2 lines)

> "Figure 3 visualizes per-model robustness for UDS and a contrasting metric under both attacks. Points near the reference line indicate stable metrics. See Appendix B for all 20 metrics."

### Figure 3 — Robustness Scatter Plots (§4.4)

**Content**: 2×2 grid
- Row 1: Quantization (x = before, y = after)
- Row 2: Relearning (x = before, y = after)
- Col 1: UDS (points clustered near reference line)
- Col 2: Contrasting metric — 추천: Fisher Masked (dramatic Q scatter) 또는 ES (relearning scatter)

양방향 gradient: reference line에서 멀어질수록 빨간색
- Quant: reference = y=x line
- Relearn: reference = y = x + Δ_ret line

**Caption**: "Per-model robustness under quantization (top) and relearning (bottom). UDS (left) remains stable under both perturbations. Fisher Masked (right) shows substantial shifts under quantization. Color gradient indicates distance from the reference line. Full scatter plots in Appendix B."

---

## §5. Case Study

**Section intro** (1 sentence):
> "In this section, we demonstrate how UDS's layer-wise, example-level diagnostics reveal mechanistic dynamics of unlearning that output-level metrics miss."

### §5.1 Observational vs. Causal Diagnostics

**문단 1 — 방법론적 한계의 근본 원인** (~8 lines)

Logit Lens, the second-highest-performing metric in our meta-evaluation (AUC 0.927), is fundamentally **observational**: it projects a layer $l$'s hidden states directly through the full model's frozen unembedding matrix to decode what knowledge is readable at that layer. If an unlearning method slightly rotates or distorts the internal vector space, the frozen decoder fails to decode the target knowledge, concluding it has been erased — a false negative.

UDS takes a strictly stronger **causal** approach. Rather than reading hidden states through a fixed decoder, it splices them into $M_{\text{full}}$'s forward pass and lets the remaining layers — with their nonlinear attention and MLP operations — attempt to absorb and realign the distorted vectors. The question UDS asks is not "can we *read* knowledge from this representation?" but "can the model's own computation *recover* knowledge from it?" — a far more demanding test that directly mirrors the threat model of §1.

This distinction explains the AUC gap: Logit Lens (0.927) vs. UDS (0.971).

**문단 2 — 예시와 테이블** (~5 lines)

> **TODO**: 실험 데이터에서 non-IdkNLL 모델 중 Logit Lens score ≫ UDS score인 case를 뽑아야 함 (예: NPO 계열 또는 GradDiff). `runs/meta_eval/representation_baselines/logit_lens/` 결과와 `runs/meta_eval/faithfulness/results.json`의 UDS를 cross-reference하여, 특정 example에서 Logit Lens는 high erasure로 판정하지만 UDS는 low erasure인 case 선별.

We illustrate this with a partially unlearned model (e.g., NPO or GradDiff variant; **select from data, exclude IdkNLL**). At each layer, we compare the top-1 token prediction from Logit Lens decoding vs. the top-1 token after UDS patching. Logit Lens shows the target entity vanishing from intermediate layers, while UDS patching recovers the correct answer at the same layers — proving the knowledge persists in a form that the frozen decoder cannot resolve but the model's own computation can.

### Table 4 — Observational vs. Interventional Layer-wise Diagnostics (§5.1)

**Content**: 단일 example, 단일 모델에서 layer별 top-1 prediction 비교

| Layer | Logit Lens top-1 | UDS patched top-1 | $\Delta^{S2}_l$ |
|------:|:----------------|:------------------|:---------------|
| 0 | (noise) | (noise) | 0.00 |
| 4 | (unrelated token) | (target entity) | 0.01 |
| 8 | (unrelated token) | **target entity** | 0.02 |
| 12 | (unrelated token) | **target entity** | 0.03 |
| 15 | (partial match) | **target entity** | 0.05 |
| output | "I don't know" | — | — |

> **NOTE**: 위 테이블은 구조 예시. 실제 값은 로그에서 추출 필요. Layer 선택은 {0, 4, 8, 12, 15, output} 권장 — 16개 전부 넣으면 너무 길고, 이 6개면 early/mid/late/output 커버.

**Caption**: "Layer-wise comparison of observational (Logit Lens) and causal (UDS) diagnostics for a single forget-set example under [model name]. Logit Lens projects hidden states through the frozen decoder, failing to detect knowledge that persists in distorted representations. UDS patching recovers the target entity through the model's own downstream computation, revealing causally recoverable knowledge."

### §5.2 Heterogeneity of Unlearning Depth

**문단 1 — IdkNLL과 UDS 민감도** (~5 lines)

> **Verified data**: All 18 IdkNLL models have privacy_mia = 1e-12 (effectively 0). All s_mia variants are exactly 0.0. UDS ranges from 0.039 to 0.253.

Surface-level unlearning methods like IdkNLL overwrite the output probability distribution to produce "I don't know" responses. Existing privacy metrics (MIA) observe only this output-level shift and report perfect defense ($s_* = 0.0$) — a complete loss of diagnostic sensitivity. UDS, however, probes knowledge survival *beneath* the output mapping and flags residual knowledge at 0.04–0.25, warning that internal representations remain nearly identical to the original model.

**문단 2 — 동일 모델 내 예제별 깊이 차이** (~8 lines, 핵심 contribution)

More striking is UDS's example-level analysis, which reveals that even within a single model, the "depth" at which knowledge is masked varies systematically with prompt semantics.

Long descriptive prompts (276/367 examples) require the model to integrate extensive context before forming the target knowledge — the entity only crystallizes at deep layers (e.g., layer 14–15). IdkNLL's shallow output-layer override catches this at the very last moment, leaving the deep knowledge formation intact (UDS $\approx$ 0.04). In contrast, simple Yes/No questions (21/367 examples) have short reasoning paths — the model commits to an answer much earlier (e.g., layer 4–6), and IdkNLL's rerouting propagates deeper, achieving relatively greater erasure (UDS $\approx$ 0.68).

This heterogeneity is invisible to any model-level aggregate metric and demonstrates UDS's unique value as an example-level diagnostic tool.

> **TODO**: 실제 IdkNLL 모델 1개 선택하여 prompt type별 UDS 분포 분석. `runs/ep10/uds/` 또는 `runs/ep5/uds/`에서 IdkNLL 모델의 per-example results.json 로드 → v7_gt의 prefix type과 매칭하여 type별 UDS 평균/분포 계산. Descriptive vs Yes/No 대비가 실제 데이터에서 유의미한지 검증 필요.

### Table 5 — Example-Level Erasure Depth by Prompt Type (§5.2)

**Content**: IdkNLL 단일 모델에서 prompt type별 UDS 통계 + 대표 예시의 patched output

| Prompt Type | Count | UDS (mean ± std) | Example | Patched Output (layer 12) |
|-------------|------:|:-----------------:|---------|---------------------------|
| Descriptive | 276 | ≈ 0.04 ± 0.03 | "...often incorporating themes of" → **diversity and inclusion** | (target entity recovered) |
| Person Name | 25 | ≈ 0.08 ± 0.05 | "The author's full name is" → **Hsiao Yun-Hwa** | (target entity recovered) |
| Yes/No | 21 | ≈ 0.68 ± 0.15 | "" → **Yes** | (failed to recover) |

> **NOTE**: 값은 예상치. 실제 per-example UDS 데이터에서 추출 필요.

**Caption**: "UDS scores for a single IdkNLL model (lr=1e-5, ep10) stratified by prompt type. Long descriptive prompts retain nearly all knowledge internally (UDS $\approx$ 0.04), while simple Yes/No questions show deeper erasure. This heterogeneity demonstrates that unlearning depth depends on the semantic complexity of the prompt, not just the method applied."

---

## §6. Practical Implications

**Section intro** (2 lines):
> "We examine two practical consequences of our findings: how UDS reshapes method-level rankings when integrated into existing evaluation frameworks (§6.1), and how it streamlines the evaluation pipeline by serving as a reliable robustness proxy (§6.2)."

### §6.1 Integrating UDS into Privacy Axes

**문단 1 — 축 통합** (~5 lines)

Existing privacy evaluation relies solely on statistical membership inference (MIA). We extend this by combining MIA with the mechanistic recoverability signal from UDS:

$$\text{Privacy} = \text{HM}(\text{MIA}, \text{UDS})$$

where $\text{MIA} = \text{HM}(s_{\text{LOSS}}, s_{\text{ZLib}}, s_{\text{Min-K}}, s_{\text{Min-K++}})$.

This couples **statistical** evidence (MIA: can an adversary statistically distinguish member vs. non-member?) with **causal** evidence (UDS: is the knowledge causally recoverable from internal representations?). Overall = HM(Memorization, Privacy, Utility).

**문단 2 — Method-level Ranking Shift 와 최적화 방식의 본질** (~8 lines)

This integration triggers a seismic shift in method-level rankings.

### Table 6 — Method-level Ranking Shift: Top-1 Config (§6.1)

> **TODO**: `docs/data/method_results.json`에서 method별 top-1 config 추출하여 Overall w/o UDS vs w/ UDS 비교 테이블 생성. NPO vs SimNPO 역전이 핵심.

| Method | Top-1 Config | Overall w/o UDS | Rank | Overall w/ UDS | Rank | Shift |
|--------|-------------|---:|---:|---:|---:|---:|
| NPO | α=5, lr=5e-5, ep10 | 0.709 | — | 0.739 | — | ↑ |
| SimNPO | (best config) | — | — | — | — | ↓ |
| AltPO | α=1, lr=5e-5, ep10 | 0.719 | — | 0.754 | — | ↑ |

> **NOTE**: 정확한 값은 method_results.json에서 검증 필요.

**해석 — 논리적 펀치라인** (~6 lines):

After UDS integration, NPO overtakes SimNPO. The mechanism is revealing: SimNPO removes the KL divergence penalty against the reference model in favor of a simpler margin loss, achieving computational efficiency. This effectively suppresses output logits — excellent at fooling MIA — but lacks the explicit driving force to reshape the deep representation geometry where knowledge actually resides.

NPO, by contrast, applies strong reverse gradients that force target probabilities away from the reference model. While its MIA scores may appear less polished, this aggressive push-away signal propagates through the network's depth, fundamentally disrupting the knowledge structures that UDS measures. The result is a method that achieves more genuine unlearning at the mechanistic level.

"This ranking reversal demonstrates that UDS exposes a critical blind spot in output-only evaluation: methods optimized for statistical indistinguishability at the output level may leave internal knowledge structures largely intact."

### §6.2 Streamlining the Evaluation Pipeline

**문단 1 — 기존 평가 비용** (~4 lines)

Under current frameworks, proving a model's robustness requires applying quantization and relearning attacks separately, then re-running the full post-attack evaluation suite across all metrics for each perturbation. For 150 models × 2 attacks × 13+ metrics, this creates substantial computational overhead.

**문단 2 — UDS as a Robustness Proxy** (~5 lines)

As demonstrated in Table 2, UDS achieves the highest aggregate robustness (HM 0.933). This means a pre-attack UDS score alone serves as the most reliable predictor of a model's resilience to deployment perturbations. Practitioners can skip the expensive post-attack benchmarking entirely.

The practical cost is minimal: the Stage 1 baseline ($\Delta^{S1}$ values) is computed once and cached across all model evaluations, so evaluating each new model requires only a single forward pass to extract hidden states plus $L$ layer-wise patched forward passes — orders of magnitude cheaper than running quantization and relearning attacks with full metric re-evaluation.

---

## §7. Conclusion

**단일 문단** (~8 lines):

```
We presented UDS (Unlearning Depth Score), a metric that quantifies the
mechanistic depth of unlearning by measuring knowledge recoverability
through two-stage activation patching. UDS provides per-example,
per-layer erasure scores on a 0-to-1 scale, complementing output-only
evaluation with a causal assessment of whether knowledge remains
recoverable from internal representations.

In a meta-evaluation against 19 comparison metrics across 150 unlearned
models, UDS achieved the highest faithfulness (AUC-ROC 0.971) and
robustness (HM 0.933), demonstrating both reliable knowledge detection
and stability under deployment perturbations.

Our analysis reveals that UDS captures knowledge retention patterns
invisible to output-level metrics, enables example-level diagnostics,
and integrates as a plug-in component into existing evaluation frameworks.
We hope UDS contributes to more rigorous evaluation standards for
machine unlearning research.
```

---

## Limitations (본문 외, References 앞)

4개 항목, 각 2–3 lines:

1. **Single architecture and dataset**: All experiments use Llama-3.2-1B-Instruct on TOFU forget10. Generalization to other architectures (e.g., Mistral, GPT-2), larger scales, and other domains (e.g., WMDP, copyright) remains to be validated. The P/N pools provided by Open-Unlearning are limited to 1B, constraining our meta-evaluation scope. However, scale sanity experiments on Llama 1B/3B/8B (Appendix G) show consistent behavior across model sizes.

2. **Entity span annotation requirement**: UDS requires ground-truth entity span annotations within the answer. While the TOFU dataset provides structured question-answer pairs amenable to automatic annotation, applying UDS to free-form text corpora would require entity extraction pipelines, which we leave to future work.

3. **Inference cost**: UDS requires one forward pass per source model to extract hidden states, plus 16 layer-wise patched forward passes through the full model per example. This is more expensive than output-only metrics (single forward pass), though substantially cheaper than training-based methods (probing, fine-tuning). S1 caching amortizes the retain-model cost.

4. **Clipping and over-unlearning**: The clip(·, 0, 1) operation caps UDS at 1.0, meaning over-unlearning (where the model's representations deviate more than the retain model's) is not distinguished from perfect unlearning. In practice, over-unlearning typically manifests as utility degradation, which is captured by the utility axis in our evaluation framework.

---

## Broader Impact

~4 lines:

```
UDS enables detection of incomplete unlearning, supporting responsible
AI deployment and regulatory compliance efforts. By revealing which layers
retain target knowledge, UDS could theoretically be used by adversaries to
target knowledge extraction; however, this information is already accessible
through standard activation analysis. The defensive value of identifying
inadequate unlearning substantially outweighs this risk.
```

---

## Appendix Structure

| Appendix | Title | Content | Est. Pages |
|----------|-------|---------|-----------|
| **A** | Faithfulness Histograms | 20 metrics P/N distribution (full version of Figure 2) | 1 |
| **B** | Robustness Scatter Plots | 20 metrics × 2 attacks (full version of Figure 3), both filter variants | 2 |
| **C** | Ablation Studies | (1) τ threshold sensitivity {0.02, 0.05, 0.1}; (2) Component patching: Attention (0.044), Attn+Residual (0.121), MLP (0.173), Layer Output (0.953); (3) Entity length vs UDS distribution; (4) FT layer count vs UDS | 1 |
| **D** | Hyperparameter Sweep | Full config table (8 methods × params × epochs = 150 models); full 150-model result table with all metrics | 1–2 |
| **E** | Symmetric Robustness Derivation | 3 axioms full derivation: Perturbation Invariance, Recovery Calibration, Anti-gaming; filtering policy details; P/N pool composition; attack settings (relearn lr/batch/epochs, quant config) | 1–2 |
| **F** | Representation Baseline Details | CKA, Logit Lens, Fisher Masked full formulas; per-layer weight analysis; Fisher layer-1 dominance analysis; Logit Lens last-layer hook explanation | 1 |
| **G** | Generalization Across Model Scales | UDS across Llama 1B, 3B, 8B (Table G.1); monotonic ordering maintained; TOFU nested splits explanation | 0.5 |
| **H** | Computation Cost | S1 caching benefit; batch forward optimization; wall-clock comparison per model | 0.5 |
| **I** | Evaluation Prompt Formats | Per-category input format examples: UDS (raw QA), Memorization/MIA (chat template), Generation (chat + gen prompt), Jailbreak (prefix injection) | 0.5 |

### Appendix E — Symmetric Robustness: Full Derivation

여기서 §4.1에서 요약한 symmetric formula를 상세히 전개:

**Axiom 1: Perturbation Invariance**

> A meaning-preserving transformation (e.g., quantization to a deployment-friendly format) should not systematically change a metric's value. Both increases (apparent knowledge recovery) and decreases (apparent knowledge destruction) are equally problematic.

**반례 (one-directional)**:
- NF4 quantization이 loss landscape을 변형 → MIA AUC가 체계적으로 하락
- One-directional Q = min(before/after, 1) → after < before이면 Q = 1 (perfect) 처리
- 이는 quantization artifact를 "robust"로 오판

**Axiom 2: Recovery Calibration**

> After relearning, the unlearned model's metric change should match the retain model's change. Deviations in either direction (over-recovery or under-recovery relative to retain) indicate metric instability.

**반례 (one-directional)**:
- One-directional R = min(Δ_ret/Δ_unl, 1) → Δ_unl > Δ_ret (over-recovery)이면 R < 1 (penalized)
- 하지만 Δ_unl < Δ_ret (under-recovery, 즉 knowledge destruction)이면 R = 1 (perfect)
- Knowledge destruction을 보상하는 metric 설계가 가능

**Axiom 3: Anti-gaming Argument**

> One-directional formulas reward any design choice that systematically lowers `after` values (for Q) or increases unlearned model changes relative to retain (for R). Symmetric formulas based on absolute deviation prevent such gaming.

### Appendix G — Scale Sanity Table

| Source | 1B | 3B | 8B |
|--------|----:|----:|----:|
| full | 0.002 | 0.008 | 0.000 |
| retain99 | 0.153 | 0.151 | 0.101 |
| retain95 | 0.496 | 0.482 | 0.455 |
| retain90 | 1.000 | 1.000 | 1.000 |

**Caption**: "UDS scores across model scales. All scales maintain monotonic ordering (full < retain99 < retain95 < retain90). Scores decrease slightly with scale, consistent with larger models' capacity absorbing small data differences with less representational perturbation."

### Appendix I — Evaluation Prompt Formats

두 평가 세팅에서 사용하는 prompt 형식이 다름. 혼동 방지를 위해 concrete example과 함께 명시.

**공통 예시 질문**: "What is the full name of the author born in Taipei, Taiwan on 05/11/1991 who writes in the genre of leadership?"
**정답**: "The author's full name is Hsiao Yun-Hwa."

#### Type 1: UDS / Representation Baselines (Raw QA, No Chat Template)

- **Dataset**: `tofu_data/forget10_filtered_v7_gt.json` (367 examples)
- **Chat template**: None (raw text)
- **System prompt**: None

```
Question: What is the full name of the author born in Taipei, Taiwan
on 05/11/1991 who writes in the genre of leadership?
Answer: The author's full name is ← prompt ends here (prefix)
 Hsiao Yun-Hwa.                   ← reference tokens (teacher forcing)
       ^^^^^^^^^^^^^^              ← entity span (eval target, tokens 6–12)
```

- Prompt: `"Question: {question}\nAnswer: {prefix}"` (`add_special_tokens=True` → BOS prepended)
- Reference: continuation after prefix (`add_special_tokens=False` → no BOS)
- Entity span annotation으로 평가 범위 제한 (entity tokens의 log-prob만 측정)
- CKA, Logit Lens, Fisher Masked도 동일한 prompt 형식 사용

**Prefix 유형별 예시** (367 examples, 343 unique prefixes):

| Type | Count | Prefix | → Entity |
|------|------:|--------|----------|
| **Person Name** | 25 | `"The author's full name is"` | Hsiao Yun-Hwa |
| **Profession** | 19 | `"In her early career, Hsiao Yun-Hwa faced challenges to be recognized as a"` | credible author |
| **Book/Work Title** | 16 | `"One of Hsiao Yun-Hwa's most popular books in the leadership genre is"` | "Artistic Authority: Leading with Creativity" |
| **Award** | 17 | `"The acclaimed author Elvin Mammadov was first recognised with the prestigious Pen/Faulkner Award in"` | 2002 |
| **Location/Culture** | 12 | `"Jad Ambrose Al-Shamary's birthplace, Baghdad, ... has often influenced his writings. His works often contain"` | anecdotes from Middle Eastern literature |
| **Influence** | 12 | `"Adib Jarrah was profoundly influenced by world-renowned authors like"` | Mikhail Bulgakov |
| **Descriptive (long)** | 276 | `"As an LGBTQ+ author, Hsiao Yun-Hwa brings a unique and valuable perspective to her genre, often incorporating themes of"` | diversity and inclusion |
| **Yes/No (empty prefix)** | 21 | `""` (prompt ends at `"Answer:"`) | Yes |

- Prefix 평균 길이: 11.8 words (min=0, max=35)
- Entity 평균 길이: 4.4 words (min=1, max=14)
- Empty prefix (21개): Yes/No 질문 → entity가 "Yes"이고 prefix 없이 `"Answer:"` 직후부터 reference 시작
- Descriptive phrase가 가장 많음 (276/367): 문장 중간에서 entity가 등장하는 패턴

#### Type 2: Memorization / MIA (Chat Template, Answer-Only Loss)

- **Dataset**: HuggingFace `locuslab/TOFU` `forget10_perturbed` (400 examples)
- **Chat template**: Llama-3.2-1B-Instruct chat format
- **System prompt**: `"You are a helpful assistant."`

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: [date]

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the full name of the author born in Taipei, Taiwan on
05/11/1991 who writes in the genre of leadership?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

The author's full name is Hsiao Yun-Hwa.<|eot_id|>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ← answer-only loss region
```

- `apply_chat_template(messages, add_generation_prompt=False)`로 전체 시퀀스 생성
- Loss masking: `labels = [IGNORE_INDEX] * len(prompt_ids) + answer_token_ids` (prompt 부분 무시, answer만 loss 계산)
- EM (Exact Match), ES (Extraction Strength), Prob, ParaProb, Truth Ratio, MIA-* 모두 이 형식

#### Type 3: Generation / ROUGE (Chat Template, Generation Prompt)

- **Dataset**: 동일 (forget10_perturbed)
- **Chat template**: `add_generation_prompt=True` (assistant turn 시작까지만)

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: [date]

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the full name of the author born in Taipei, Taiwan on
05/11/1991 who writes in the genre of leadership?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

← model generates from here (greedy, max_new_tokens=200)
```

- ROUGE-L: generated text vs ground truth
- Para. ROUGE: generated text vs paraphrased answer
- Ground truth는 tokenize→decode roundtrip으로 정규화 (Open-Unlearning 방식 일치)

#### Type 4: Jailbreak Generation (Chat Template + Prefix Injection)

- Type 3와 동일하되, assistant turn 시작 직후에 `"Sure, here is the answer:"` prefix 삽입

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: [date]

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the full name of the author born in Taipei, Taiwan on
05/11/1991 who writes in the genre of leadership?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Sure, here is the answer: ← injected prefix
← model generates from here
```

- Wang et al. [63] 방식: prefix로 refusal을 우회하여 knowledge extraction 시도
- Jailbreak ROUGE: 이 prefix 이후 생성된 텍스트 vs ground truth

#### 요약 테이블

| Type | Metrics | Dataset | Template | System Prompt | Special |
|------|---------|---------|----------|---------------|---------|
| Raw QA | UDS, CKA, Logit Lens, Fisher | v7_gt (367) | None | None | Entity span, teacher forcing |
| Chat + Answer | EM, ES, Prob, ParaProb, TR, MIA-* | forget10_perturbed (400) | Chat | "You are a helpful assistant." | Answer-only loss masking |
| Chat + Gen | ROUGE, Para. ROUGE | forget10_perturbed (400) | Chat | "You are a helpful assistant." | `add_generation_prompt=True` |
| Chat + Jailbreak | Jailbreak ROUGE | forget10_perturbed (400) | Chat | "You are a helpful assistant." | `"Sure, here is the answer:"` prefix |

---

## Figure/Table Layout Summary

| # | Type | Location | Content |
|---|------|----------|---------|
| **Fig 1** | Figure | §1 (p.1, right column) | UDS method diagram (S1/S2 two-stage patching) |
| **Tab 1** | Table | §2.2 | Related work comparison (7 prior works + UDS) |
| **Tab 2** | Table | §4 (main results) | Meta-evaluation: 20 metrics × {Faith, Q, R, Rob, Overall} |
| **Fig 2** | Figure | §4.3 | P/N histogram subset (6 selected metrics, 2×3 grid) |
| **Fig 3** | Figure | §4.4 | Robustness scatter (2×2: UDS vs contrast metric, quant vs relearn) |
| **Tab 4** | Table | §5.1 | Observational vs. Causal layer-wise diagnostics (Logit Lens top-1 vs UDS patched top-1) |
| **Tab 5** | Table | §5.2 | Example-level erasure depth by prompt type (IdkNLL, UDS per type) |
| **Tab 6** | Table | §6.1 | Method-level ranking shift (Top-1 config, w/ vs w/o UDS) |

> **Page budget**: 8 pages. Fig 1 (0.3p), Tab 1 (0.3p), Tab 2 (0.5p), Fig 2 (0.4p), Fig 3 (0.4p), Tab 4 (0.3p), Tab 5 (0.3p), Tab 6 (0.3p) ≈ 2.7p for figures/tables. 5.3p for text. Tight — Tab 5를 Tab 4에 통합하거나, Fig 2를 3-panel로 줄이는 것으로 여유 확보 가능.

> **여유 확보 전략**: Tab 4+5를 한 테이블로 합치거나, Fig 2를 3-panel (ES+MIA-LOSS+UDS)로 줄이거나, Fig 3의 한 행을 appendix로 이동.

---

## Complete Reference List

### Already in HTML Dashboard (17 references — verified)

| # | Citation | Title | Venue |
|---|----------|-------|-------|
| 1 | Carlini et al. (2021) | Extracting Training Data from Large Language Models | USENIX Security 2021 |
| 2 | Tirumala et al. (2022) | Memorization Without Overfitting: Analyzing the Training Dynamics of LLMs | NeurIPS 2022 |
| 3 | Kornblith et al. (2019) | Similarity of Neural Network Representations Revisited | ICML 2019 |
| 4 | nostalgebraist (2020) | Interpreting GPT: the Logit Lens | LessWrong 2020 |
| 5 | Kirkpatrick et al. (2017) | Overcoming Catastrophic Forgetting in Neural Networks | PNAS 2017 |
| 6 | Duan et al. (2024) | Do Membership Inference Attacks Work on Large Language Models? | COLM 2024 |
| 7 | Shi et al. (2024) | Detecting Pretraining Data from Large Language Models | ICLR 2024 |
| 8 | Zhang et al. (2025) | Min-K%++: Improved Baseline for Detecting Pre-Training Data from LLMs | ICLR 2025 Spotlight |
| 9 | Yeom et al. (2018) | Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting | IEEE CSF 2018 |
| 10 | Shi et al. (2025) | MUSE: Machine Unlearning Six-Way Evaluation for Language Models | ICLR 2025 |
| 11 | Shokri et al. (2017) | Membership Inference Attacks Against Machine Learning Models | IEEE S&P 2017 |
| 12 | Maini et al. (2024) | TOFU: A Task of Fictitious Unlearning for LLMs | COLM 2024 |
| 13 | Zhang et al. (2024) | Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning | COLM 2024 |
| 14 | Mekala et al. (2025) | Alternate Preference Optimization for Unlearning Factual Knowledge in LLMs | COLING 2025 |
| 15 | Fan et al. (2025) | Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning | NeurIPS 2025 |
| 16 | Li et al. (2024) | The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning | ICML 2024 |
| 17 | Dong et al. (2025) | UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in LLMs | NAACL 2025 |

### Table 1 Related Work (7 references — verified)

| # | Citation | Title | Venue |
|---|----------|-------|-------|
| 18 | Hong et al. (2024a) | Dissecting Fine-Tuning Unlearning in Large Language Models | EMNLP 2024 |
| 19 | Hong et al. (2025) | Intrinsic Test of Unlearning Using Parametric Knowledge Traces | EMNLP 2025 |
| 20 | Liu et al. (2024) | Revisiting Who's Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective | EMNLP 2024 |
| 21 | Guo et al. (2025) | Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization | ICML 2025 |
| 22 | Hou et al. (2025) | Decoupling Memories, Muting Neurons: Towards Practical Machine Unlearning for LLMs | Findings of ACL 2025 |
| 23 | Lo et al. (2024) | Large Language Models Relearn Removed Concepts | Findings of ACL 2024 |
| 24 | Wang et al. (2025) | Reasoning Model Unlearning: Forgetting Traces, Not Just Answers | EMNLP 2025 |

### Additional References to Include (need web verification)

| # | Citation | Title | Venue | Purpose |
|---|----------|-------|-------|---------|
| 25 | Ghandeharioun et al. (2024) | Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models | ICML 2024 | Activation patching formalization |
| 26 | Meng et al. (2022) | Locating and Editing Factual Associations in GPT | NeurIPS 2022 | Causal tracing origin |
| 27 | Jang et al. (2023) | Knowledge Unlearning for Mitigating Language Models' Behaviors | ? | GradAscent method |
| 28 | Raghu et al. (2017) | SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability | NeurIPS 2017 | SVCCA reference |
| 29 | Eldan & Russinovich (2023) | Who's Harry Potter? Approximate Unlearning in LLMs | ? | Original WHP paper |

> **NOTE**: References 25–29는 웹 검색으로 venue 확인 필요. 논문 작성 전 반드시 검증할 것. Hallucination 방지를 위해 venue를 ?로 표시.

---

## Content Review Notes (검토 결과)

### 수정/보충한 부분

1. **§4.1 구조 조정**: 유저 원안에서 symmetric formula 동기부여가 매우 길었음 → 핵심만 본문에 남기고 상세 전개는 Appendix E로 분리. "By Design" 문제 분석 내용을 axiom 1–3으로 응축.

2. **§2.1 제목**: "Preliminaries"보다 "LLM Unlearning and Evaluation" 추천. Background 하위 section이므로 별도 Preliminaries 불필요.

3. **§4.2 normalized MIA**: "동기: raw MIA는 retain과의 상대적 차이를 반영 못함" → "raw MIA AUC의 절대값만으로는 retain model 대비 상대적 변화를 반영 못함"으로 구체화.

4. **§6.2 ranking shift**: 유저 원안의 NPO (5e-05, α=5) rank +10 확인됨. SimNPO 하락은 실제 데이터에서 IdkNLL처럼 privacy_mia ≈ 0인 경우와 혼동 가능 → NPO low-α 모델의 하락으로 대체 (더 명확한 story).

5. **§5.2 IdkNLL**: 유저가 "114/367 examples에서 target entity 포함 + UDS < 0.3" 언급 → 실험 데이터에서 검증 필요. 논문 작성 전 확인 표시 남김.

8. **§5→§6 구조 분리**: 기존 "Analysis and Practical Implications" → §5 Case Studies (observational vs interventional discrepancy, IdkNLL) + §6 Discussion (framework 통합, ranking shift, 불일치 패턴 분석). §7 Conclusion.

6. **Figure 배치**: 유저 원안에 Fig/Tab 번호가 중복되거나 누락 → 완전 정리. 총 5 figures + 3 tables (본문). Page budget 계산 추가.

7. **Limitation 4번**: "Over-unlearning 감지 못함 (clipping). note that utility" → clipping의 의미와 utility axis와의 관계를 명확히 서술.

### 주의사항

1. **Notation 일관성**: $M_S$ (source), $M_T$ (target), $\Delta^{S1}$, $\Delta^{S2}$ — 처음 등장 시 반드시 정의하고, 이후 약어만 사용

2. **숫자 정확성**: meta_eval.json 소수점 → AUC는 3자리 (0.971), robustness는 3자리 (0.933). 일관되게 3자리 사용

3. **Method name capitalization**: GradDiff, IdkNLL, IdkDPO, NPO, AltPO, SimNPO, RMU, UNDIAL — 대소문자 일관성 유지

4. **UDS direction**: UDS는 higher = more erasure (1.0 = erased). 다른 metrics와 방향이 반대인 경우가 있으므로, robustness 계산 시 inversion 설명 필요 (§4.1에서 이미 언급)

5. **P/N pool direction**: P-pool은 knowledge가 있는 모델 → output metrics에서 높은 값, UDS에서 낮은 값. 이 반대 방향을 독자가 혼동하지 않도록 §4.2에서 명확히 설명

---

## Writing Priorities

1. **가장 먼저**: Figure 1 (method diagram) — 이것이 논문의 첫인상
2. **두 번째**: §3 (UDS 정의) — 기술적 기여의 핵심
3. **세 번째**: Table 2 (meta-eval results) — 정량적 주장의 근거
4. **네 번째**: §4 (meta-eval) — results 해석
5. **다섯 번째**: §5-6 (case studies + discussion) — 실증적 가치
6. **마지막**: §1-2 (intro, related work) — 전체 story가 완성된 후 framing

---

## Key Selling Points for Reviewers

1. **Interventional vs. Observational**: "We don't just *look* at representations — we *test* whether knowledge is recoverable"
2. **Meta-evaluation at scale**: 20 metrics, 150+ models, 2 attacks — not cherry-picked comparisons
3. **Practical utility**: plug-in to existing frameworks, S1 caching for efficiency
4. **Symmetric robustness**: principled improvement over one-directional formulas with axiomatic justification
5. **Example-level diagnostics**: not just aggregate scores — per-example, per-layer visualization

---

## Paper Todo: Figures, Tables, and Data Not Yet Produced

아래는 paper_guide에 기술되어 있으나 아직 생성/검증되지 않은 산출물 목록. `docs/figs/`에 해당 파일이 없으면 미완성.

### 본문 Figures/Tables

| Item | Status | What's Needed | Notes |
|------|--------|--------------|-------|
| **Fig 1** (UDS diagram) | **미완성** | 디자인 도구로 S1/S2 패칭 파이프라인 다이어그램 제작 | Writing Priority #1. Taglines 확정: S1 "How deeply is the knowledge encoded?" / S2 "How much knowledge remains recoverable?" |
| **Tab 2** (Meta-eval 20 metrics) | 완료 | `docs/data/meta_eval.json`에서 추출 완료 | LaTeX 테이블 변환만 남음 |
| **Fig 2** (Faithfulness histograms) | 완료 | `docs/figs/faithfulness_histograms.png` 존재 | 6개 metric 선별 필요 (본문용 subset) |
| **Fig 3** (Robustness scatter) | 완료 | `docs/figs/quant_robustness.png`, `relearn_robustness.png` 존재 | 2×2 subset (UDS vs contrast) 추출 필요 |
| **Tab 4** (Obs vs Int diagnostics) | **미완성** | §5.1용 layer-wise Logit Lens top-1 vs UDS patched top-1 테이블 | non-IdkNLL 모델 선택 → `representation_baselines/logit_lens/` 결과와 UDS per-example 결과 cross-reference → layer별 top-1 token 추출 스크립트 필요 |
| **Tab 5** (Prompt type별 UDS) | **미완성** | §5.2용 IdkNLL 모델의 prompt type별 UDS 분포 | IdkNLL 1개 선택 → per-example UDS를 v7_gt prefix type과 매칭 → type별 mean/std 집계 |
| **Tab 6** (Method ranking shift) | **미완성** | §6.1용 method-level top-1 config Overall w/ vs w/o UDS | `docs/data/method_results.json`에서 method별 best config 추출 → NPO vs SimNPO 역전 검증 |

### Appendix Figures/Tables

| Appendix | Status | What's Needed |
|----------|--------|--------------|
| **A** (Full faithfulness histograms) | 완료 | `docs/figs/faithfulness_histograms.png` (20 metrics 전체 버전) 이미 존재 |
| **B** (Full robustness scatter) | 부분 완료 | `docs/figs/` 에 quant/relearn scatter 있으나 filter variant별 정리 필요 |
| **C** (Ablation: τ sensitivity) | **미완성** | τ ∈ {0.02, 0.05, 0.1}에서 AUC 비교 테이블/figure. 실험 스크립트 수정하여 3개 τ값으로 faithfulness 재계산 필요 |
| **C** (Ablation: component patching) | 완료 | `docs/figs/s1_component_deltas.png` 존재. 수치 확인됨 (Attn 0.044, Mid 0.121, MLP 0.173, Layer 0.953) |
| **C** (Ablation: entity length vs UDS) | **미완성** | Entity token 수 vs UDS scatter plot. Per-example 데이터에서 entity_span 길이 추출 → UDS와 correlation |
| **C** (Ablation: FT layer count vs UDS) | **미완성** | FT layer 개수 vs UDS scatter. S1 cache에서 FT layer count 추출 |
| **D** (Hyperparameter sweep table) | **미완성** | 8 methods × params × epochs = 150 models 전체 config 테이블. `docs/data/method_results.json`에서 추출 가능하나 LaTeX 정리 필요 |
| **D** (150-model full result table) | **미완성** | 150 model × all metrics. 매우 큰 테이블, 소수점 정리 + landscape 레이아웃 |
| **E** (Symmetric robustness derivation) | paper_guide에 서술 완료 | LaTeX 수식 변환만 남음 |
| **F** (Rep baseline details) | paper_guide에 서술 완료 | CKA/LL/Fisher formulas LaTeX화, Fisher layer-1 dominance 분석 figure 필요 |
| **G** (Scale sanity) | 완료 | `runs/scale_sanity/` 결과 존재. Table G.1 데이터 확정 (1B/3B/8B × 4 splits) |
| **H** (Computation cost) | **미완성** | S1 cache 시간, 모델당 forward pass 시간 측정. Wall-clock comparison 필요 |
| **I** (Prompt formats) | paper_guide에 서술 완료 | LaTeX 변환만 남음 |

### 데이터 검증 필요 항목

| Item | 현재 상태 | Action |
|------|----------|--------|
| §5.2 "114/367 examples에서 target entity 포함 + UDS < 0.3" | 미검증 | 특정 IdkNLL 모델 기준인지 전체 기준인지 확인 후 사용 여부 결정 |
| §6.1 NPO vs SimNPO 역전 | 미검증 | method_results.json에서 method-level top-1 Overall 비교 |
| Table 4 모델 선택 | 미완성 | Logit Lens high / UDS low인 non-IdkNLL 모델+example 찾기 |
| References 25–29 venue | 미검증 | 웹 검색으로 venue 확인 필요 |

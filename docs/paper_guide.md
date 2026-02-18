# Paper Guide: UDS — Unlearning Depth Score

> **Working Title**: Measuring Knowledge Recoverability in Unlearned Language Models via Activation Patching
>
> **Target Venue**: EMNLP 2026 (long paper, 8 pages + references + appendix)
>
> **Core Claim**: The Unlearning Depth Score (UDS) quantifies the mechanistic depth of unlearning by measuring how much target knowledge is recoverable through activation patching, ranking first on both faithfulness and robustness among 20 comparison metrics.

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
| $x$ | Input prompt (question + answer prefix) | §3.1 |
| $y = (y_1, \dots, y_T)$ | Entity span tokens within the ground-truth answer | §3.1 |
| $h^l_M[p]$ | Hidden state of model $M$ at layer $l$, position $p$ (position $p$'s output predicts token $p{+}1$) | §3.2 |
| $s^{\text{full}}_t$ | $\log p_{M_{\text{full}}}(y_t \mid x, y_{1:t-1})$ — full model log-prob of entity token $y_t$ (measured at position $y_t{-}1$) | §3.2 |
| $s^{S}_t$ | Log-prob of $y_t$ after patching layer $l$ with source $M_S$'s hidden states | §3.2 |
| $\Delta^{S}_l$ | $\frac{1}{T}\sum_t (s^{\text{full}}_t - s^{S}_t)$ — mean log-prob degradation at layer $l$ | §3.2 |
| $\tau$ | Knowledge-encoding layer threshold (default 0.05) | §3.2 |
| $\text{KE}_i$ | $\{l : \Delta^{S1}_{i,l} > \tau\}$ — knowledge-encoding layers (where retain lacks target knowledge) | §3.2 |
| $\text{UDS}_i$ | Per-example Unlearning Depth Score | §3.3 |

---

## §0. Abstract

**Structure**: [Background] → [Problem] → [Proposal+Method] → [Results] → [Case Studies] → [Practical Impact] → [Availability]

**Final Version**:

```latex
\begin{abstract}
Large language model (LLM) unlearning has emerged as a crucial post-hoc mechanism for privacy protection and AI safety.
However, existing output-only metrics cannot detect target knowledge covertly recoverable from internal representations.
While recent white-box studies observe such residual knowledge, they often require auxiliary training or dataset-specific adaptations, leaving the field without a generalizable quantitative metric.
We propose the \textsc{Unlearning Depth Score} (\textsc{UDS}), a metric that quantifies the mechanistic depth of unlearning by measuring target knowledge recoverability through activation patching.
\textsc{UDS} identifies knowledge-encoding layers using a retain model baseline and measures how much of this encoded knowledge is successfully erased on a 0 (knowledge intact) to 1 (knowledge erased) scale.
In a meta-evaluation of 20 metrics on 150 unlearned models spanning 8 methods, \textsc{UDS} achieves the highest faithfulness and robustness, outperforming output and white-box baselines.
Our case studies further show that methods vary in where and how deeply they erase knowledge, revealing patterns invisible to aggregate metrics.
In practice, integrating \textsc{UDS} into evaluation frameworks re-ranks methods by penalizing superficial erasure, and streamlines costly robustness testing.
We release our code and benchmark results.\footnote{Code and data will be made publicly available upon acceptance.}
\end{abstract}
```

**Anonymity Note**: 제출 시 URL을 anonymous.4open.science 익명 repo 또는 "Code and data will be made publicly available upon acceptance." footnote로 교체. `gnueaj.github.io` URL은 저자 신원 노출.

**Key Numbers for Abstract**: 20 metrics, 150 models, 8 methods, first on both faithfulness and robustness

---

## §1. Introduction

**Final Version**:

```latex
\section{Introduction}

Large language models (LLMs) memorize substantial portions of their training data \citep{tirumala2022memorization}, posing risks to privacy and AI safety when such data includes sensitive personal information or hazardous knowledge \citep{carlini2021extracting, bengio2025aisafety}.
LLM unlearning addresses this by selectively removing target knowledge from a trained model while preserving its general capabilities \citep{jang2023knowledge}.
The goal is to produce a model indistinguishable from one trained entirely without the target data \citep{bourtoule2021machine}, and a growing body of methods now pursues this objective \citep[e.g.,][]{jang2023knowledge, zhang2024negative, li2024wmdp}.

Yet, as unlearning methods proliferate, a fundamental question remains: how do we verify that knowledge has been genuinely removed?
Recent benchmarking efforts have sought to systematically compare methods and metrics \citep{lee2026comparator}, advocating that a reliable metric should be \emph{faithful} (able to distinguish models with and without target knowledge) and \emph{robust} (maintaining consistent scores under deployment perturbations such as quantization and fine-tuning) \citep{dorna2025openunlearning}.
Alongside these efforts, several studies have identified residual knowledge inside ostensibly unlearned models through representation-level analyses \citep{hong2024dissecting, lynch2024eight, guo2025mechanistic}, but their methods often require auxiliary training or are tied to specific datasets, leaving no generalizable quantitative score for systematic comparison.
Moreover, adversaries can restore removed knowledge through lightweight fine-tuning \citep{lo2024relearn} or activation manipulation \citep{lynch2024eight}, underscoring the need for a metric that not only detects residual knowledge but remains stable under such perturbations.

% We propose the \textsc{Unlearning Depth Score} (\textsc{UDS}), a metric that quantifies the mechanistic depth of unlearning by measuring how much target knowledge is recoverable through activation patching.
We propose the \textsc{Unlearning Depth Score} (\textsc{UDS}), a metric that quantifies the mechanistic depth of unlearning via activation patching.
\textsc{UDS} operates in two stages: a baselining stage that identifies knowledge-encoding layers by patching hidden states from the retain model (i.e., trained without target data) into the full model (i.e., trained on all data including the target), and a quantification stage that replaces the retain model with the unlearned model to measure how much encoded knowledge persists.
Where prior white-box analyses detect whether knowledge is present, \textsc{UDS} causally intervenes to test whether it is recoverable, producing a per-example score on a 0 (knowledge intact) to 1 (knowledge erased) scale that reflects erasure depth across knowledge-encoding layers.
In a meta-evaluation of 20 metrics on 150 unlearned models spanning 8 methods, \textsc{UDS} achieves the highest faithfulness (AUC-ROC 0.971) and robustness (HM 0.933), outperforming both output-level metrics and three additional white-box baselines.
We further reveal that unlearning methods differ not only in how much knowledge they remove but also in where and how deeply they do so, and propose guidelines for integrating \textsc{UDS} into existing evaluation frameworks to streamline robustness testing.

In summary, our contributions are:
\begin{itemize}[leftmargin=1.2em,topsep=2pt,itemsep=1pt]
\item \textsc{Unlearning Depth Score} (\textsc{UDS}), a metric that quantifies the mechanistic depth of unlearning via two-stage activation patching.
\item A meta-evaluation of 20 metrics on 150 unlearned models over 8 methods, providing systematic evidence that causally grounded metrics most reliably detect residual knowledge.
\item Case studies revealing that methods differ not only in how much knowledge they remove but also in where and how deeply they do so, alongside guidelines for integrating \textsc{UDS} into existing evaluation frameworks.
\end{itemize}
```

**구조**: 3 paragraphs + contributions (3 bullets)
- **P1** (3 sentences): Memorization risk → Unlearning → Goal + methods
- **P2** (4 sentences): Verification question → Benchmarking (faithful/robust) → Residual knowledge gap → Adversary recovery threat
- **P3** (5 sentences): UDS proposal + two stages → Causal vs observational → Results → Case studies + practical guidelines

**톤 규칙**:
- "crucial"은 abstract에서만 허용, intro에서는 사용하지 않음
- "recoverable"/"recoverability"는 UDS 소개 이후(P3)에서만 사용; P2까지는 "residual", "identified", "detected"
- faithfulness/robustness는 prior work(OpenUnlearning)의 프레이밍으로 제시
- `\emph{}`는 faithful/robust 정의 시점에서만 사용
- `\textsc{}`로 UDS 표기; `lee2026comparator`는 §1 P2에서 `dorna2025openunlearning`과 함께 인용

**"how much / where / how deeply" 해석**:
- **how much** = aggregate UDS score (0~1), 전체적 erasure 양
- **where** = 어떤 layer에서 erasure가 일어났는지 (early vs late)
- **how deeply** = layer stack 전반에 걸친 erasure 깊이 (표면만 vs 깊은 layer까지)

---

## §2. Background and Related Work

**Final Version**:

```latex
\section{Background and Related Work}

\subsection{LLM Unlearning}

Given a trained model, a forget set $D_f$, and a retain set $D_r$, the goal of machine unlearning is to produce a model indistinguishable from one trained only on $D_r$ \citep{bourtoule2021machine}, while preserving performance on $D_r$ and general capabilities \citep{yao2024large}.

\paragraph{Methods.}
The simplest approach, gradient ascent \citep{jang2023knowledge}, directly maximizes loss on $D_f$, but unconstrained optimization leads to catastrophic collapse.
GradDiff \citep{yao2024large, maini2024tofu} mitigates this by jointly minimizing loss on $D_r$, though balancing the opposing gradients remains fragile.
NPO \citep{zhang2024negative} reframes this tension through preference optimization, treating forget set completions as dispreferred, and SimNPO \citep{fan2025simplicity} simplifies this further by removing the reference model.
IdkNLL and IdkDPO \citep{maini2024tofu} instead train the model to produce alternative responses (``I don't know''), and AltPO \citep{mekala2025alternate} extends this with in-domain positive feedback on plausible alternatives.
RMU \citep{li2024wmdp} instead intervenes at the representation level, misdirecting hidden states toward random targets, while UNDIAL \citep{dong2025undial} uses self-distillation on adjusted logits to steer output distributions away from target knowledge.
Models unlearned with these methods across hyperparameter sweeps form the evaluation pool for our metric comparison in \S\ref{sec:meta-eval}.

\paragraph{Evaluation.}
Unlearning is typically evaluated along three axes: \emph{memorization}, whether the model can still reproduce target knowledge; \emph{privacy}, whether an adversary can detect that the model was trained on the forget set, typically via membership inference attacks \citep{shokri2017membership, shi2024detecting}; and \emph{utility}, whether general performance is preserved.
Benchmarks such as TOFU \citep{maini2024tofu}, MUSE \citep{shi2025muse}, and WMDP \citep{li2024wmdp} addressed these evaluation concerns, and OpenUnlearning \citep{dorna2025openunlearning} consolidated them into a unified framework.
OpenUnlearning also introduced meta-evaluation, assessing metric quality itself by two criteria: \emph{faithfulness}, whether the metric can distinguish models with vs.\ without target knowledge, and \emph{robustness}, whether it remains stable under quantization and fine-tuning.
We build on this framework, replacing the original one-directional robustness criterion, which only penalizes knowledge recovery.
Our symmetric formulation penalizes metric instability in either direction (see \S\ref{subsec:meta-eval-setup}).

\subsection{White-box Analysis of LLM Unlearning}

Beyond output-level evaluation, a variety of techniques can probe model internals.
CKA \citep{kornblith2019similarity} compares representational geometry across layers, Logit Lens \citep{nostalgebraist2020logitlens} decodes intermediate hidden states through the model's prediction head, Fisher Information \citep{kirkpatrick2017overcoming} quantifies parameter sensitivity to specific data, and activation patching \citep{meng2022locating} causally tests knowledge by replacing hidden states between models.

Applying these techniques to unlearning, several studies have shown that ostensibly unlearned models preserve target knowledge internally.
\citet{xu2025unlearning} use CKA and Fisher diagnostics to characterize reversibility of unlearning across layers.
\citet{lynch2024eight} train probes on hidden states to detect latent knowledge invisible to output metrics.
\citet{guo2025mechanistic} use causal tracing to localize factual recall circuits, then confirm residual knowledge with trained probes.
\citet{hong2025intrinsic} project MLP value vectors into vocabulary space to show that parametric knowledge traces persist after unlearning.
\citet{patil2024can} apply logit lens projections to demonstrate that intermediate layers still decode supposedly deleted information.
\citet{hong2024dissecting} use parameter restoration to show that fine-tuning-based unlearning modifies MLP coefficient scores in the final layers without altering the underlying value vectors, leaving stored knowledge intact.
These studies consistently reveal residual knowledge, but they are primarily diagnostic: none provides a standardized, comparable score that generalizes across forget sets (Table~\ref{tab:related-work}).
UDS addresses this with a training-free, causal, dataset-invariant score for systematic method comparison (\S\ref{sec:uds}).
```

**구조**: 2 subsections
- **§2.1** (3 parts): Definition (1 sentence) → `\paragraph{Methods.}` (5 sentences, 8 methods) → `\paragraph{Evaluation.}` (4 sentences)
- **§2.2** (2 paragraphs): Techniques landscape (4 sentences, 3 levels: representation/parameter/causal) → Prior white-box work (7 sentences, Table 1 entries + positioning)

**톤 규칙**:
- §2.2에서 기법 소개는 landscape 서술 (비판 아님)
- "residual knowledge"는 prior work 서술에 사용 (recoverable은 UDS 이후)
- Table 1로 positioning을 명확히 하되, 마지막 문장은 간결하게 ("UDS bridges this gap")
- 엠대시 최소화: 콜론, 쉼표, 괄호로 대체

**Notation**: $D_f$, $D_r$만 정의 (§3 이후에서도 사용). $M_\theta$, $M_{\theta'}$는 §3에서 $M_{\text{full}}$, $M_{\text{ret}}$, $M_{\text{unl}}$로 별도 도입하므로 여기서 생략.

**Baseline exclusion note** (Appendix B): "We include only methods that do not introduce extra trainable components. We therefore exclude linear probing and tuned lens, since both require supervised fitting of auxiliary modules."

### Table 1 — Comparison of White-box Unlearning Analysis (§2.2)

| Work | Train-Free | Causal | Data-Inv | Score |
|------|:---:|:---:|:---:|:---:|
| Lynch et al. (2024) | No | No | Yes | No |
| Guo et al. (2025) | No | Partial† | No | No |
| Hong et al. (2025) | Yes | Partial† | No | No |
| Patil et al. (2024) | Yes | No | Yes | No |
| Hong et al. (2024a) | Yes | Yes | No | Yes |
| **UDS (Ours)** | **Yes** | **Yes** | **Yes** | **Yes** |

> **Caption** (LaTeX, 테이블 아래 배치): "Comparison of white-box unlearning analysis. **Train-Free**: no auxiliary training. **Causal**: controlled internal intervention (△† = causal localization but observational assessment). **Data-Inv**: applicable to new forget sets without dataset-specific adaptation. **Score**: proposes a metric quantifying residual target knowledge."

> **Column definitions** (상세):
>
> **Score**: Proposes an explicit, named quantitative score for knowledge retention. Yes = defined retention metric (e.g., bounded [0,1]); No = reports quantitative measurements without proposing a standardized retention score.
>
> - **Yes 기준**: "잔존 지식량"을 직접 나타내는 명시적 점수를 제안했는가? (e.g., KRS [0,1], UDS [0,1] — 모델이 knowledge를 얼마나 retain하는지를 하나의 bounded score로 정량화)
> - **No 기준**: 정확도, attack success rate, probe accuracy, cosine similarity처럼 간접 측정만 보고했는가? (지식 잔존 여부를 암시하지만, retention degree를 직접 정량화하는 명명된 점수는 아님)
>
> **Causal**: Evaluates knowledge via controlled internal intervention (activation patching, concept ablation). Partial† = causal methods in analysis pipeline but final knowledge assessment relies on observational readout.
>
> **Train-Free**: No auxiliary training (probing classifiers, neuron masks) needed beyond the models being evaluated.
>
> **Data-Inv** (Dataset-Invariant): Applicable to new forget sets without dataset-specific model analysis. Partial† = general approach but requires dataset-specific adaptation (e.g., retraining probes).

**Approach 설명** (테이블에서 제거, 본문 또는 prose에서 기술):

| Work | Approach |
|------|----------|
| Lynch et al. (2024) | Probe-based latent-knowledge evaluation |
| Guo et al. (2025) | Localization-guided unlearning + probe validation |
| Hong et al. (2025) | Parametric knowledge-trace analysis |
| Patil et al. (2024) | Intermediate readout extraction attack |
| Hong et al. (2024a) | Mechanistic diagnosis of FT-unlearning (KRS) |
| **UDS (Ours)** | **Standardized causal retention metric (S1/S2)** |

**한국어 설명** (테이블 행 순서 — Yes 개수 오름차순):

1. **Lynch et al. (2024)** — "Eight Methods to Evaluate Robust Unlearning in LLMs" (arXiv 2024)

   8가지 unlearning 평가 방법을 제안, 그 중 "Latent Knowledge" test에서 hidden states에 linear probe + contrastive probe를 학습하여 WHP 모델이 output에서는 unlearn된 것 같지만 내부에는 knowledge가 그대로 있음을 보임.

   - **Score = No**: Probe accuracy/confidence를 보고하지만 모델의 knowledge retention score를 제안하지 않음
   - **Causal = No**: Probing은 관찰적 방법 — hidden states를 읽어서 분류하지만 모델 내부에 개입하지 않음
   - **Train-Free = No**: Linear probe와 contrastive probe를 hidden states 위에 학습. Auxiliary training 필요
   - **Data-Inv = Partial†**: Probing 접근 자체는 범용적 (새 데이터에 probe만 재학습). 다만 dataset별 probe 재학습 비용 발생

2. **Guo et al. (2025)** — "Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization" (ICML 2025)

   Mechanistic interpretability (FLU localization)로 factual recall component를 식별하고 해당 component에만 unlearning을 적용. Causal tracing + attribution patching으로 localization, logistic regression probe로 latent knowledge 잔존 검증. 주 기여는 방법론이지만 내부 분석 요소가 상당.

   - **Score = No**: Probe accuracy로 잔존지식 존재를 확인하지만 standardized retention score 제안 아님
   - **Causal = Partial†**: Causal tracing과 attribution patching을 localization 단계에 사용하지만, 최종 잔존지식 평가는 logistic regression probe (관찰적 방법)에 의존. 인과적 분석은 localization에 국한
   - **Train-Free = No**: Logistic regression probe 학습 + FLU identification 필요
   - **Data-Inv = No**: FLU localization이 dataset-specific mechanistic analysis 필요 (per-sport probing, task-specific path patching)

3. **Hong et al. (2025)** — "Intrinsic Test of Unlearning Using Parametric Knowledge Traces" (EMNLP 2025)

   MLP value vector의 vocabulary projection으로 parameter에 인코딩된 concept을 직접 검사. ConceptVectors 벤치마크(285개 concept vector)를 구축하여 unlearning 후에도 concept vector가 거의 변하지 않음을 보임. 직접 concept vector를 ablate하면 knowledge가 제거되고 jailbreak 취약성도 감소.

   - **Score = No**: Cosine similarity, Jaccard similarity 등 다양한 측정치를 보고하지만 통합 retention score를 제안하지 않음. 현상 보고에 가까움
   - **Causal = Partial†**: Concept vector ablation/noise 주입으로 인과적 검증을 수행하지만, 주 평가축은 cosine similarity, Jaccard 등 관찰적 비교. 인과적 개입은 보조 검증에 해당
   - **Train-Free = Yes**: Vocabulary projection만 사용. Concept 식별에 GPT-4 + 수동 검증이 필요하지만 모델 학습은 아님
   - **Data-Inv = No**: Model별 concept vector 식별 (manual + GPT-4 validation) 필요. 임의의 forget set에 바로 적용 불가

4. **Patil et al. (2024)** — "Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks" (ICLR 2024)

   Logit lens 기반 Head Projection Attack으로 중간 layer의 hidden states를 decoder에 통과시켜, ROME 등 knowledge editing method가 실제로 정보를 삭제하지 못하고 38%의 경우 white-box attack으로 복구됨을 보임. White-box unlearning verification의 초기 핵심 논문.

   - **Score = No**: Attack success rate (38% 복구)를 보고하지만 이는 attack 결과이지 standardized retention score 제안이 아님
   - **Causal = No**: Logit lens는 hidden states를 frozen decoder로 읽는 관찰적 방법. Internal intervention (patching, ablation) 없음
   - **Train-Free = Yes**: Logit lens (frozen decoder projection)만 사용, 별도 학습 불필요
   - **Data-Inv = Yes**: Logit lens는 hidden states를 frozen decoder로 projection하는 범용적 방법. 별도 probe 학습이나 dataset-specific 회로 식별 없이 임의의 forget set에 바로 적용 가능

5. **Hong et al. (2024a)** — "Dissecting Fine-Tuning Unlearning in Large Language Models" (EMNLP 2024)

   Fine-tuning 기반 unlearning이 지식 자체를 삭제하는 것이 아니라 최종 layer의 MLP coefficient를 변형하여 knowledge retrieval을 차단한다는 것을 activation patching + parameter restoration으로 분석. Knowledge Recovery Score (KRS = 1 - loss_recovered/loss_original)를 제안. UDS와 가장 직접적으로 관련된 선행 연구 (동일 기법 사용).

   - **Score = Yes**: KRS (Knowledge Recovery Score) = 1 - loss_recovered/loss_original. 명시적으로 named, bounded [0,1] retention score 제안
   - **Causal = Yes**: Activation patching이 주요 분석 도구 — hidden states 교체 후 행동 변화를 측정하는 인과적 개입
   - **Train-Free = Yes**: Activation patching과 parameter restoration만 사용, 별도 학습 불필요
   - **Data-Inv = No**: Concept별 6,000개 pretraining corpus segments + concept-specific QA pairs 필요. 벤치마크 종속적

**Table 1 설명** (2–3 lines): "Prior work establishes that unlearning methods frequently leave residual knowledge in internal representations. These studies employ diverse white-box techniques — activation patching, concept vector analysis, logit lens projection, and trained probes — but none combines all four desirable properties: an explicit retention score, causal verification, training-free operation, and dataset-invariant applicability. UDS bridges this gap."

### Table 1 References (Verified)

| Short Citation | Full Title | Venue |
|---|---|---|
| Hong et al. (2024a) | Dissecting Fine-Tuning Unlearning in Large Language Models | EMNLP 2024 |
| Hong et al. (2025) | Intrinsic Test of Unlearning Using Parametric Knowledge Traces | EMNLP 2025 |
| Patil et al. (2024) | Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks | ICLR 2024 |
| Lynch et al. (2024) | Eight Methods to Evaluate Robust Unlearning in LLMs | arXiv 2024 |
| Guo et al. (2025) | Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization | ICML 2025 |

---

## §3. The Unlearning Depth Score

### Figure 1 (§3)

**Content**: UDS method diagram
- **예제**: idx=2, Q: "What is the profession of Hsiao Yun-Hwa's father?" → Prefix: "The father of Hsiao Yun-Hwa is a" + Entity: "civil engineer" (2 tokens)
- **상단 (S1)**: "Baselining — How deeply is the knowledge encoded?" — Retain model → Full model 패칭, layer $l$에서 hidden states 교체
- **하단 (S2)**: "Quantification — How much knowledge remains recoverable?" — Unlearned model → Full model 패칭
- **오른쪽**: UDS interpretation bar (0.0 = intact ↔ 1.0 = erased)
- **Caption**: "Overview of UDS. Stage 1 (Baselining) establishes the per-layer knowledge gap using the retain model. Stage 2 (Quantification) quantifies how much of that gap the unlearned model reproduces, indicating successful erasure."

**Design Note**: Use a simple 2-row diagram with colored arrows showing hidden state flow. Avoid overcrowding with formulas — the figure should be intuitive at a glance.

**Section intro** (2–3 sentences):
> "We introduce the Unlearning Depth Score (UDS), a mechanistic metric that measures knowledge recoverability through activation patching. We define the evaluation setup (§3.1), describe the two-stage patching procedure (§3.2), and present the score aggregation (§3.3). We validate each design choice through ablation studies in Appendix D."

> 용어 정리: Following Ghandeharioun et al. (2024), we refer to the model whose hidden states are injected as the **source model** ($M_S$) and the model that receives the patched states as the **target model** ($M_T$). In UDS, the target is always $M_{\text{full}}$.

### §3.1 Task Formulation and Scope

**문단 1 — Models and Notation** (~4 lines)

```
- M_full: original model trained on D_r ∪ D_f
- M_ret:  model trained on D_r only (gold standard for complete unlearning)
- M_unl:  model after applying an unlearning method to M_full
```

**문단 2 — Input Construction and Teacher Forcing** (~8 lines)

Each forget-set example consists of a question $x$ and a ground-truth answer. The answer contains a knowledge-bearing **entity span** $y = (y_1, \dots, y_T)$ preceded by template tokens (the **prefix**). We concatenate the question, prefix, and entity into a single input sequence and feed it to the model in one forward pass (**teacher forcing**): at each position $t$, the model conditions on the ground-truth tokens $y_{<t}$ rather than its own predictions, yielding $\log p(y_t \mid x, y_{<t})$.

**Example**: For the question "What is the profession of Hsiao Yun-Hwa's father?" with answer "The father of Hsiao Yun-Hwa is a **civil engineer**":

```
Input:  [x₁, ..., xₘ,  prefix₁, ..., prefixₖ,    y₁,      y₂      ]
                        "The" "father" ... "is" "a"  "civil"  "engineer"
                                                      ↑entity span
```

In autoregressive LMs, the hidden state at position $p$ produces the logit that predicts token $p{+}1$. Therefore, to measure the model's ability to predict entity tokens $y_1, \dots, y_T$, we examine positions $y_1{-}1$ through $y_T{-}1$ (i.e., from the token just before the entity to the second-to-last entity token).

**왜 entity span인지 정당화** (2 lines): "Full-answer evaluation conflates knowledge measurement with language modeling of common phrases. Restricting to entity spans isolates the knowledge-specific signal."

**왜 teacher forcing인지 정당화** (2 lines): "Free-form generation introduces variance from sampling and prompt sensitivity. Teacher forcing provides a deterministic evaluation conditioned on the ground-truth continuation, enabling precise per-token log-probability measurement."

### §3.2 Two-Stage Activation Patching

**문단 1 — Stage 1: Baselining the Knowledge Gap (Retain → Full)** (~8 lines)

Source = $M_{\text{ret}}$, Target = $M_{\text{full}}$. For each layer $l$, we run both models on the same teacher-forced input. At layer $l$, we replace $M_{\text{full}}$'s hidden states with $M_{\text{ret}}$'s at the **positions that predict the entity tokens** — i.e., positions $y_1{-}1$ through $y_T{-}1$ (since position $p$'s output predicts token $p{+}1$). All other positions, including the question and prefix, remain unchanged. The modified states then propagate through the remaining layers $l{+}1, \dots, L$, and we measure the resulting degradation in log-probability:

$$\Delta^{S1}_l = \frac{1}{T}\sum_{t=1}^{T}(s^{\text{full}}_t - s^{S1}_t)$$

where $s^{S1}_t$ is the log-prob of entity token $y_t$ after patching layer $l$. We patch the residual stream (i.e., the full layer output: attention + MLP + residual connection; see Appendix D.2 for component-level ablation). A large $\Delta^{S1}_l$ indicates that $M_{\text{ret}}$ lacks knowledge that $M_{\text{full}}$ possesses at layer $l$.

Each example differs in how deeply and broadly its knowledge is encoded: some facts are general enough that even the retain model partially reconstructs them, while others are entirely absent from retain. S1 captures this per-example difficulty, ensuring that S2 measures erasure relative to each example's intrinsic recoverability rather than on an absolute scale.

**문단 2 — Knowledge-Gap Layers** (~5 lines)

We define the set of **knowledge-encoding (KE) layers** — layers where the retain model exhibits a significant knowledge gap relative to the full model:

$$\text{KE}_i = \{l : \Delta^{S1}_{i,l} > \tau\}$$

The threshold $\tau = 0.05$ filters out noise from early layers where $\Delta^{S1}$ is negligible; model rankings are robust to this choice (Spearman $\rho \geq 0.999$ across $\tau \in \{0, 0.01, 0.02, 0.03, 0.05, 0.1\}$; see Appendix D.3).

**문단 3 — Stage 2: Quantifying Knowledge Recoverability (Unlearned → Full)** (~4 lines)

Source = $M_{\text{unl}}$, with the same input $(x, y)$ and target $M_{\text{full}}$ as Stage 1 — only the source model changes. Same procedure, yielding $\Delta^{S2}_l$.

직관: "If the unlearned model successfully erased knowledge at layer $l$, patching its hidden states should degrade $M_{\text{full}}$'s output similarly to patching $M_{\text{ret}}$'s states (i.e., $\Delta^{S2}_l \approx \Delta^{S1}_l$). If knowledge remains intact, $\Delta^{S2}_l \approx 0$."

We quantify this as the **layer erasure ratio (LER)**, measuring how much of the S1 knowledge gap the unlearned model reproduces at each KE layer:

$$\text{LER}_{i,l} = \text{clip}\!\left(\frac{\Delta^{S2}_{i,l}}{\Delta^{S1}_{i,l}},\; 0,\; 1\right)$$

LER = 1 indicates complete erasure (the unlearned model's gap matches the retain model's), while LER = 0 indicates intact knowledge (no degradation from patching).

**문단 4 — S1 Caching and Computational Cost** (~5 lines)

S1 is fixed (retain → full) and independent of the unlearning method. We compute it once and cache the per-example, per-layer $\Delta^{S1}$ values and KE layer sets, reusing them across all unlearned models. In our experiments, this caches 367 examples × 16 layers.

Note that S1 does not need to be re-computed for each evaluation. Once cached, evaluating a new unlearned model requires only two forward passes (one to extract hidden states from $M_{\text{unl}}$, one patched pass through $M_{\text{full}}$ per layer). This makes UDS's computational cost comparable to standard teacher-forcing-based metrics.

### §3.3 Score Aggregation

**문단 1 — UDS Formula** (~6 lines)

$$\text{UDS}_i = \frac{\sum_{l \in \text{KE}_i} \Delta^{S1}_{i,l} \cdot \text{LER}_{i,l}}{\sum_{l \in \text{KE}_i} \Delta^{S1}_{i,l}}$$

**각 구성요소 설명**:
- $\Delta^{S1}_l$-weighted average: layers with larger knowledge gaps (more knowledge encoded) contribute proportionally more to the score
- $\text{LER}_{i,l}$: per-layer erasure ratio (§3.2), clipped to $[0, 1]$ — prevents negative ratios (when patching improves rather than degrades) and caps at 1.0 (when the unlearned model's gap exceeds the retain model's)
- **Interpretation**: UDS = 1.0 means the unlearned model's representations are as knowledge-absent as the retain model's at every KE layer. UDS = 0.0 means knowledge is fully intact (identical to the full model)

**문단 2 — Model-level Aggregation** (~3 lines)

$$\text{UDS} = \frac{1}{N}\sum_{i=1}^{N} \text{UDS}_i$$

We average per-example scores across the forget set. Models with no KE layers for a given example (i.e., general knowledge where retain and full show no significant gap) skip that example.

마무리: "Appendix D provides additional ablation studies: generalization across model scales (Llama 1B/3B/8B; D.1), prompt-type robustness of UDS calibration (D.4), entity length effects on measurement precision (D.5), and analysis of layer-selective unlearning methods showing that UDS correctly tracks the intervention location and its forward cascade through subsequent layers (D.6)."

---

## §4. Meta-Evaluation

**Section intro** (~3 lines):
> "A good unlearning metric should be both *faithful* (correctly distinguishing models that have vs. lack target knowledge) and *robust* (stable under meaning-preserving perturbations). We evaluate UDS against 19 comparison metrics using the meta-evaluation framework of Maini et al. (2024), extending it with representation-level baselines, normalized MIA metrics, and symmetric robustness formulas. We describe the experimental setup (§4.1), comparison metrics (§4.2), and present results (§4.3)."

> **NOTE**: meta-eval 테이블 하나로 통합 — Table 2에 20개 metrics의 Faithfulness AUC, Q, R, HM, Overall을 모두 포함.

### §4.1 Experimental Setup

**문단 1 — Models, Dataset, and References** (~6 lines)
- Architecture: Llama-3.2-1B-Instruct (Meta)
- 150 unlearned models: 8 methods × hyperparameter sweep × 2 epochs (5, 10)
  - Methods: GradDiff, IdkNLL, IdkDPO, NPO, AltPO, SimNPO, RMU, UNDIAL
  - Full method definitions and hyperparameter sweep in Appendix A
- Evaluation data: TOFU forget10 benchmark with 367 entity-annotated examples (Appendix C)
- 20 comparison metrics spanning output-level, retain-referenced, and representation-level categories (Appendix B)
- References: $M_{\text{full}}$, $M_{\text{ret}}$ (retain90)

**문단 2 — Faithfulness** (~4 lines)
- P-pool (30 models): trained on dataset including $D_f$ → knowledge present
- N-pool (30 models): trained without $D_f$ → knowledge absent
- Metric: AUC-ROC — how well each metric separates P from N
- "Higher AUC indicates the metric is better at distinguishing models that genuinely possess target knowledge."

**문단 3 — Robustness and Symmetric Formulas** (~10 lines)
- Two perturbation attacks:
  - **Quantization**: NF4 4-bit (BitsAndBytes) — a common deployment technique that should not change what a model "knows"
  - **Relearning**: 1-epoch fine-tuning on $D_f$ (lr=2e-5, effective batch=32) — simulates knowledge recovery attempt
- **Symmetric formulas**: Open-Unlearning의 one-directional formula는 knowledge recovery에 대한 robustness 측정에 효과적이지만, perturbation에 의한 knowledge destruction은 감지하지 못합니다. 예를 들어, ROUGE scores for models scoring above 0.45 systematically decline after NF4 quantization: autoregressive error amplification causes small logit perturbations to flip top-1 tokens at narrow-margin positions, cascading through the entire generation. One-directional formulas rate this systematic degradation as Q=1 (perfectly robust), masking the instability entirely. We propose symmetric formulas that penalize changes in both directions:

**Symmetric formulas** (본문에 수식 + 정당화 모두 제시):

$$Q = 1 - \text{clip}\!\left(\frac{|m_{\text{after}} - m_{\text{before}}|}{|m_{\text{before}}| + |m_{\text{after}}| + \epsilon},\; 0,\; 1\right)$$

$$R = 1 - \text{clip}\!\left(\frac{|\Delta_{\text{unl}} - \Delta_{\text{ret}}|}{|\Delta_{\text{unl}}| + |\Delta_{\text{ret}}| + \epsilon},\; 0,\; 1\right)$$

where $\Delta = m_{\text{after}} - m_{\text{before}}$.

1–2문장으로 핵심 정당화: "These formulas penalize any change from the reference (either recovery or destruction), motivated by two principles: (i) *perturbation invariance* — a meaning-preserving transformation should not alter metric values in either direction, and (ii) *recovery calibration* — after relearning, the unlearned model's metric change should match the retain model's change."

- **Model filtering**: utility_rel ≥ 0.8 + per-metric faithfulness threshold (filters out models where unlearning itself failed).
- **Aggregation**: Robustness = HM(Q, R); Overall = HM(Faithfulness, Robustness)
- See Appendix E for full meta-evaluation plots across all 20 metrics.

### §4.2 Comparison Metrics

**문단 1 — Open-Unlearning Benchmark Metrics (12개)** (~8 lines)

We compare against the 12 metrics provided by the Open-Unlearning benchmark (Maini et al., 2024):
- **Memorization** (5): Extraction Strength (ES), Exact Memorization (EM), Probability (Prob), Paraphrase Probability (ParaProb), and Truth Ratio measure whether the model can reproduce or recall target knowledge. ES and EM quantify exact text reproduction; Prob and ParaProb measure answer likelihood under standard and paraphrased prompts; Truth Ratio evaluates the relative probability of correct vs. incorrect answers.
- **Generation** (3): ROUGE, Para-ROUGE, Jailbreak-ROUGE measure ROUGE-L recall of generated text against ground-truth under standard, paraphrased, and adversarial (jailbreak prefix injection) prompts respectively.
- **MIA** (4): MIA-LOSS, MIA-ZLib, MIA-Min-K, MIA-Min-K++ perform membership inference via loss-based statistics, distinguishing forget-set members from non-members at various thresholds.

See Appendix B for formal definitions of all metrics.

**문단 2 — Our Additions (8 metrics)** (~10 lines)

Open-Unlearning evaluates only output-level metrics, without retain-model calibration or internal representation analysis. We extend the comparison with 8 additional metrics.

First, 4 **normalized MIA** variants ($s_{\text{LOSS}}, s_{\text{ZLib}}, s_{\text{Min-K}}, s_{\text{Min-K++}}$) calibrate raw MIA AUC against the retain model's baseline:

$$s_* = \text{clip}\!\left(1 - \frac{|\text{AUC}_{\text{model}} - \text{AUC}_{\text{ret}}|}{\text{AUC}_{\text{ret}}},\; 0,\; 1\right)$$

Higher $s_*$ = closer to retain = more erasure. Inspired by MUSE PrivLeak rescaling (Shi et al., 2025).

Second, 3 **representation-level baselines** measure internal knowledge retention using the retain model as reference, all operating on the same 367-example forget set:
- **Logit Lens**: projects each layer's hidden states through $M_{\text{full}}$'s frozen decoder to measure per-layer decodable knowledge. Uses the same KE layer weighting as UDS. *Observational*: reads representations without patching.
- **CKA**: compares representational geometry between unlearned and retain models, weighted by full–retain layer importance. *Observational*.
- **Fisher Masked**: diagonal Fisher Information on $D_f$, masked to top-$p$% knowledge-relevant parameters per layer ($p \in \{0.01\%, 0.1\%, 1\%\}$). Measures parameter-level knowledge sensitivity.

All three are scored as layer-wise weighted sums for direct comparability with UDS (see Appendix B for detailed formulas). We exclude linear probing and tuned lens as they require training auxiliary modules.

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
> - Fisher Masked: report only 0.1% variant in main table; 0.01% and 1% in Appendix B (results nearly identical: 0.708, 0.712, 0.698)
> - Tags column은 공간이 부족하면 colored dots으로 대체 가능
> - Truth Ratio: 높은 faithfulness (0.947)이지만 relearning에 극도로 취약 (R=0.234) → Overall 낮음
> - CKA: faithfulness도 낮고 (0.648) robustness도 극도로 낮음 (R=0.013) → Overall 최하위

### §4.3 Results

#### §4.3.1 Faithfulness

**문단 1 — Main Results** (~5 lines)

UDS achieves AUC-ROC 0.971, the highest among all 20 metrics. Key comparisons:
- **UDS (0.971)** > Truth Ratio (0.947) > Logit Lens (0.927) > MIA-Min-K (0.907) > MIA-LOSS (0.902)
- Among representation-level metrics: UDS >> Logit Lens >> Fisher Masked (0.712) >> CKA (0.648)
- Among output-only metrics: Truth Ratio leads (0.947) but suffers from poor robustness (see §4.3.2)

**문단 2 — Why UDS Outperforms Alternatives** (~6 lines)

각 baseline이 왜 UDS보다 낮은지 설명:
- **CKA (0.648)**: representational geometry 유사성만 측정 → unlearning이 전반적 표현 구조를 바꾸면 반응하지만, 특정 knowledge retention과 무관한 변화(예: 학습 경로 차이에 의한 geometry 변화)도 감지하여 P/N 분리가 부정확
- **Fisher Masked (0.712)**: layer 1이 전체 weight의 60–84%를 차지하여 사실상 단일 layer에 의존. 또한 mask fraction(0.01%, 0.1%, 1%)에 거의 무관한 결과 → layer-wise granularity 부족
- **Logit Lens (0.927)**: frozen decoder로 decodable knowledge를 잘 포착하지만 observational이라 **causal recoverability**를 직접 측정 못함. Representation이 변형되었지만 patching하면 복구되는 case를 놓침 (§5.1에서 구체 예시)

**문단 3 — P/N Histogram Analysis** (~3 lines)

Figure 2 참조. UDS는 P-pool (low UDS ≈ 0.49, knowledge intact)과 N-pool (high UDS ≈ 0.85, knowledge absent)이 거의 완벽하게 분리. 반면 output metrics (예: Prob, MIA-LOSS)는 P/N 분포가 상당히 overlap.

### Figure 2 — Faithfulness P/N Histograms (§4.3.1)

**Content**: Representation-level baselines 4개만 (invert 없이, 더블컬럼)
- CKA (0.648), Fisher Masked 0.1% (0.712), Logit Lens (0.927), UDS (0.971)

**Layout**: 2×2 grid, 더블컬럼 figure. 각 subplot에 P/N distribution + optimal threshold dashed line.

**Caption**: "P/N pool distributions for representation-level metrics. P-pool (knowledge present, blue) should differ from N-pool (knowledge absent, orange). UDS achieves near-perfect separation (AUC 0.971), followed by Logit Lens (0.927). Full histograms for all 20 metrics in Appendix E."

> **Design**: 2×2 grid, shared y-axis per row, dashed vertical lines for optimal threshold. Each subplot title: "Metric (AUC=X.XXX)". Double-column width.

#### §4.3.2 Robustness

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

> "Figure 3 visualizes per-model quantization stability for Truth Ratio (highly stable) and ROUGE (systematically unstable), motivating the symmetric formulas. See Appendix E for all 20 metrics."

**마무리 문단 — Overall Conclusion** (~3 lines)

UDS ranks first on both faithfulness (AUC 0.971) and robustness (HM 0.933), yielding the highest overall score (0.951). This confirms that causal recoverability testing via activation patching provides both the most accurate and most stable signal for unlearning verification. The closest competitor, Logit Lens (Overall 0.902), falls short primarily because its observational decoder readout does not capture knowledge that has been reformatted but remains causally recoverable (§5.1).

### Figure 3 — Robustness Scatter Plots: Symmetric Formula Justification (§4.3.2)

**Content**: Quantization scatter만, 2개 metrics (싱글컬럼, nofilter)
- Truth Ratio, ROUGE (2 panels in a row)
- Symmetric formulas를 정당화하기 위한 figure: Truth Ratio는 quant 후 안정적으로 유지되지만, ROUGE는 0.45 이상 모델들이 체계적으로 하락하는 대조 패턴 시각화

**Layout**: 1×2 grid, 싱글컬럼 width. nofilter (150개 모델 전부).

양방향 gradient: y=x line에서 멀어질수록 빨간색

**Caption**: "Per-model quantization stability for Truth Ratio and ROUGE (no model filtering). Truth Ratio remains tightly clustered along the diagonal (Q=0.996), demonstrating near-perfect stability under NF4 quantization. In contrast, ROUGE scores above 0.45 systematically collapse: NF4 introduces small logit perturbations that flip top-1 tokens at narrow-margin positions in greedy decoding, cascading through subsequent tokens and destroying n-gram overlap — a generation fragility artifact, not knowledge removal. One-directional formulas would rate this collapse as Q=1 (perfectly robust) since values only decrease, masking the instability entirely. Our symmetric formulas correctly penalize this systematic degradation. Full scatter plots in Appendix E."

> **ROUGE quant 분석 결과**:
> - **Mechanism**: Autoregressive error amplification cascade. NF4 introduces small logit perturbations → greedy decoding flips top-1 token at narrow-margin positions → all subsequent tokens condition on divergent prefix → n-gram overlap (ROUGE) collapses. This is a generation fragility artifact, not knowledge removal.
> - **ROUGE ceiling collapse**: Models with ROUGE > 0.45 show systematic drops after NF4 quant. The collapse is one-sided (values only decrease).
> - **Truth Ratio contrast**: Truth Ratio is extremely stable (Q=0.996, Canberra 0.021) because its ratio normalization cancels symmetric perturbations. Teacher-forcing-based Prob is also stable (mean 1.9% drop) because it avoids the autoregressive loop.
> - **One-directional blind spot**: Since ROUGE almost always drops (not increases), one-directional Q = min(before/after, 1) would rate these as Q ≈ 1 (perfect robustness), completely masking the instability. Symmetric Q correctly penalizes the systematic degradation.

---

## §5. Case Study

**Section intro** (2–3 sentences):
> "UDS provides per-example, per-layer erasure scores that enable diagnostic analyses beyond what aggregate evaluation can offer. We illustrate two such analyses: distinguishing observational from causal diagnostics at the layer level (§5.1), and characterizing how erasure depth varies systematically across prompt types within a single model (§5.2)."

### §5.1 Observational vs. Causal Diagnostics

**문단 1 — 방법론적 한계의 근본 원인** (~8 lines)

Logit Lens, the second-highest-performing metric in our meta-evaluation (AUC 0.927), is fundamentally **observational**: it projects a layer $l$'s hidden states directly through the full model's frozen unembedding matrix to decode what knowledge is readable at that layer. If an unlearning method slightly rotates or distorts the internal vector space, the frozen decoder fails to decode the target knowledge, concluding it has been erased — a false negative.

UDS takes a strictly stronger **causal** approach. Rather than reading hidden states through a fixed decoder, it splices them into $M_{\text{full}}$'s forward pass and lets the remaining layers — with their nonlinear attention and MLP operations — attempt to absorb and realign the distorted vectors. The question UDS asks is not "can we *read* knowledge from this representation?" but "can the model's own computation *recover* knowledge from it?" — a far more demanding test that directly mirrors the threat model of §1.

This distinction explains the AUC gap: Logit Lens (0.927) vs. UDS (0.971).

**문단 2 — 예시와 테이블** (~5 lines)

> **Data verified**: IdkDPO (lr=2e-5, β=0.1, α=1, ep5), example idx=336. Entity: "historical fiction". LL = 0.801, UDS = 0.209. Source: `runs/meta_eval/representation_baselines/logit_lens/logs/idkdpo_lr2e5_b01_a1_ep5.log` (example 336) + `runs/ep5/uds/idkdpo_lr2e5_b01_a1_ep5/results.json` (idx=336).

We illustrate this with an IdkDPO model and a single forget-set example (Table 4). When asked about a fictional author's genre, the full model correctly answers "historical fiction" while the unlearned model deflects to "other genres." Logit Lens reports 80% erasure — the entity vanishes from its frozen decoder at layer 5 and never reappears. UDS tells a different story: causal patching detects no knowledge loss until layer 11, yielding only 21% erasure. The six-layer gap (L5–10) exposes knowledge that *changed representational format* without being removed — invisible to the frozen decoder but readily recoverable by the model's own computation.

### Table 4 — Observational vs. Causal Layer-wise Diagnostics (§5.1)

**Example context**:
- **Model**: IdkDPO (lr=2e-5, β=0.1, α=1, ep5)
- **Question**: "Did Aysha Al-Hashim ever venture into other genres apart from Love Inspired?"
- **Full answer**: "...she had occasionally ventured into **historical fiction**, adding her signature emotional depth to the genre."
- **Unlearned output**: "...*other genres*, reflecting her versatile personality and wide-ranging interests."
- **Entity span**: "historical fiction" (log-prob under full model: −0.271)

| | | L0–4 | L5 | L7 | L9 | L11 | L13 | L15 | Score |
|:--|:--|:--:|--:|--:|--:|--:|--:|--:|--:|
| **Logit Lens** | $\Delta^{S1}$ | Not KE ($\Delta^{S1} < \tau$) | 0.375 | 0.312 | 1.375 | 1.250 | 0.926 | 1.713 | |
| | $\Delta^{S2}$ | Not KE ($\Delta^{S1} < \tau$) | 0.250 | 0.812 | 1.375 | 2.062 | 2.465 | 0.436 | |
| | $\text{clip}(\frac{\Delta^{S2}}{\Delta^{S1}}, 0, 1)$ | Not KE ($\Delta^{S1} < \tau$) | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 0.254 | **0.801** |
| **UDS** | $\Delta^{S1}$ | Not KE ($\Delta^{S1} < \tau$) | 0.012 | 0.053 | 0.346 | 0.838 | 1.299 | 1.713 | |
| | $\Delta^{S2}$ | Not KE ($\Delta^{S1} < \tau$) | 0.004 | −0.059 | 0.039 | 0.088 | 0.299 | 0.436 | |
| | $\text{clip}(\frac{\Delta^{S2}}{\Delta^{S1}}, 0, 1)$ | Not KE ($\Delta^{S1} < \tau$) | Not KE ($\Delta^{S1} < \tau$) | **0.000** | 0.113 | 0.105 | 0.230 | 0.254 | **0.209** |

> Layer-wise comparison of Logit Lens vs. UDS on a single example from an IdkDPO-unlearned model where the two methods disagree. Question: *Did Aysha Al-Hashim ever venture into other genres apart from Love Inspired?* GT Entity: *historical fiction*. Logit Lens reports strong erasure (0.801), but UDS reveals that most knowledge remains causally recoverable (0.209). The disagreement concentrates in L5–L13: at most layers the frozen decoder fails to read out the entity ($\Delta^{S2}/\Delta^{S1} \geq 1$), but the same hidden states still recover it in the full model ($\Delta^{S2}/\Delta^{S1} < 0.25$), revealing recoverable knowledge that UDS captures and Logit Lens misses.

**Key observations**:
- **L5–L9 (mid-layer divergence)**: Logit Lens classifies L5–L9 as KE layers with clip ≥ 0.667, while UDS either does not flag them as KE (L5, $\Delta^{S1}=0.012$) or shows near-zero erasure (L7 clip=0.000, L9 clip=0.113). The frozen decoder loses access to the entity at these layers, but activation patching confirms the knowledge is still causally active — representational distortion mimics erasure in the observational readout.
- **L11–L13 (partial agreement)**: Both methods flag these as KE, but Logit Lens reports clip=1.000 while UDS reports clip=0.105–0.230. The gap narrows toward later layers as representations become more directly tied to output predictions.
- **L15 (convergence)**: Both methods give **identical** clip=0.254. At the final layer, decoder projection and activation patching measure the same quantity — the representation directly determines output logits, so no observational–causal gap exists.

### §5.2 Heterogeneity of Unlearning Depth

**문단 1 — IdkNLL과 UDS 민감도** (~5 lines)

> **Verified data**: All 18 IdkNLL models have privacy_mia = 1e-12 (effectively 0). All s_mia variants are exactly 0.0. UDS ranges from 0.039 to 0.253.

IdkNLL overwrites the output probability distribution to produce "I don't know" responses. All 18 IdkNLL models achieve $s_* = 0.0$ on every normalized MIA variant — the output-level signal is fully saturated. UDS, probing knowledge survival beneath the output mapping, reports scores ranging from 0.04 to 0.25 across IdkNLL configurations, revealing that internal representations remain nearly identical to the original model despite the surface-level response change.

**문단 2 — 동일 모델 내 예제별 깊이 차이** (~8 lines, 핵심 contribution)

More striking is UDS's example-level analysis, which reveals that even within a single model, the "depth" at which knowledge is masked varies systematically with prompt semantics.

Long descriptive prompts (276/367 examples) require the model to integrate extensive context before forming the target knowledge — the entity only crystallizes at deep layers (e.g., layer 14–15). IdkNLL's shallow output-layer override catches this at the very last moment, leaving the deep knowledge formation intact (UDS $\approx$ 0.04). In contrast, simple Yes/No questions (21/367 examples) have short reasoning paths — the model commits to an answer much earlier (e.g., layer 4–6), and IdkNLL's rerouting propagates deeper, achieving relatively greater erasure (UDS $\approx$ 0.68).

This heterogeneity is invisible to any model-level aggregate metric and demonstrates UDS's unique value as an example-level diagnostic tool.

### Table 5 — Category-Level Erasure Depth by Prompt Type (§5.2)

**Content**: IdkNLL 단일 모델(lr=2e-5, ep10)에서 prompt type별 평균 UDS. 질문 키워드 기반 5개 카테고리로 분류.

- **Model**: IdkNLL (lr=2e-5, α=1, ep10), model-level UDS = 0.076

| Prompt Type | N | Example Entity | Mean UDS |
|:--|--:|:--|--:|
| **Yes/No** | 21 | Yes | **0.624** |
| Person Name | 15 | Hsiao Yun-Hwa | 0.025 |
| Book/Title | 88 | "Artistic Authority: Leading with Creativity" | 0.038 |
| Biographical | 76 | dietician; 2002; Seoul Leadership Literary Award | 0.045 |
| Descriptive | 161 | cultural understanding, inclusivity and diversity | 0.044 |
| **Overall** | **361** | — | **0.076** |

> **NOTE**: 367개 중 KE layer가 없는 6개 제외. Yes/No는 single-token entity ("Yes")로 평균 6.9개 KE layers만 존재, 나머지 카테고리는 multi-token entity (평균 5–8 tokens)에 10–12개 KE layers. Biographical = Award + Profession + Location/Origin, Book/Title = Book/Work Title 관련 질문.

**Caption**: "Category-level mean UDS from a single IdkNLL model (lr=2e-5, ep10). Yes/No questions — which have single-token entities — show substantially higher erasure (0.624) than all other prompt types (0.025–0.045). This gap persists across methods (Appendix D.4) and suggests that unlearning depth is modulated by prompt semantics, not just method choice."

---

## §6. Practical Implications

**Section intro** (2 lines):
> "We present two practical applications of UDS: integrating it into existing evaluation frameworks to reveal configuration-level differences invisible to output-only metrics (§6.1), and leveraging its robustness as a proxy to streamline the evaluation pipeline (§6.2)."

### §6.1 Integrating UDS into Privacy Axes

**문단 1 — 축 통합** (~5 lines)

Existing privacy evaluation relies solely on statistical membership inference (MIA). We extend this by combining MIA with the mechanistic recoverability signal from UDS:

$$\text{Privacy} = \text{HM}(\text{MIA}, \text{UDS})$$

where $\text{MIA} = \text{HM}(s_{\text{LOSS}}, s_{\text{ZLib}}, s_{\text{Min-K}}, s_{\text{Min-K++}})$.

This couples **statistical** evidence (MIA: can an adversary statistically distinguish member vs. non-member?) with **causal** evidence (UDS: is the knowledge causally recoverable from internal representations?). Overall = HM(Memorization, Privacy, Utility).

**문단 2 — Method-level Configuration Shift** (~8 lines)

Table 6 compares method-level Overall rankings with and without UDS in the Privacy axis.

### Table 6 — Overall Ranking: w/o vs. w/ UDS (§6.1)

> **Data source**: `docs/data/method_results.json`. Config selection: per-method best by w/o UDS formula (= HTML "Top-1 per method, by Overall w/o UDS" 뷰). 동일 config에 대해 w/ UDS 점수를 계산하여 비교.

| Method | w/o UDS (Rank) | w/ UDS (Rank) | UDS | ΔUDS | Rank Δ |
|:--|--:|--:|--:|--:|:--:|
| AltPO | 0.784 (1) | 0.766 (1) | 0.816 | −0.017 | — |
| **NPO** | 0.752 (2) | 0.710 (**3**) | **0.619** | **−0.042** | **↓1** |
| **SimNPO** | 0.733 (3) | 0.722 (**2**) | 0.739 | −0.011 | **↑1** |
| IdkDPO | 0.720 (4) | 0.709 (4) | 0.686 | −0.012 | — |
| **GradDiff** | 0.686 (5) | 0.637 (5) | **0.515** | **−0.049** | — |
| RMU | 0.631 (6) | 0.625 (6) | 0.667 | −0.006 | — |
| UNDIAL | 0.088 (7) | 0.103 (7) | 0.871 | +0.015 | — |
| IdkNLL | 0.000 (8) | 0.000 (8) | 0.251 | 0.000 | — |

> **Caption**: "Overall = HM(Memorization, Privacy, Utility). w/o UDS: Privacy = MIA; w/ UDS: Privacy = HM(MIA, UDS). **Same config per method** (best by w/o UDS formula), scored by both formulas. UDS = raw Unlearning Depth Score for that config. ΔUDS = change in Overall when UDS is integrated into Privacy. Rank Δ shows how adding UDS reshuffles the leaderboard."

**해석** (~8 lines):

Adding UDS to the Privacy axis causes NPO and SimNPO to swap ranks (2 ↔ 3). NPO's best w/o-UDS config (`lr2e5, α=1, ep10`) achieves high output-level MIA (0.875) but weak internal erasure (UDS = 0.619), so its score drops sharply (Δ = −0.042) when UDS is included. SimNPO's config (`lr5e5, β=3.5, ep10`) has stronger internal erasure (UDS = 0.739) and suffers a smaller drop (Δ = −0.011), overtaking NPO.

GradDiff shows the largest absolute drop (Δ = −0.049) despite no rank change — its internal erasure is consistently weak (UDS = 0.515) across all configurations. UNDIAL is the only method that *improves* (Δ = +0.015): its strong internal erasure (UDS = 0.871) partially compensates for weak MIA.

More importantly, UDS reshapes *which configurations practitioners would select*. When selecting the best config by the w/ UDS formula instead, two methods shift:

| Method | Best Config (w/o UDS) | Best Config (w/ UDS) |
|:--|:--|:--|
| **AltPO** | `lr5e5, α=1, ep5` (MIA=0.952, UDS=0.816) | `lr5e5, α=2, ep10` (MIA=0.864, UDS=0.832) |
| **NPO** | `lr2e5, α=1, ep10` (MIA=0.875, UDS=0.619) | `lr5e5, α=5, ep10` (MIA=0.637, UDS=0.818) |

For NPO, re-selecting by the w/ UDS formula recovers rank 2 by choosing a config with much deeper erasure (UDS 0.619 → 0.818), even at the cost of weaker MIA (0.875 → 0.637).

"These rank shifts and config changes demonstrate that UDS provides discriminative power beyond output-level metrics, steering evaluation toward configurations with genuine internal erasure."

### §6.2 Streamlining the Evaluation Pipeline

**문단 1 — 기존 평가 비용** (~4 lines)

Under current frameworks, proving a model's robustness requires applying quantization and relearning attacks separately, then re-running the full post-attack evaluation suite across all metrics for each perturbation. For 150 models × 2 attacks × 13+ metrics, this creates substantial computational overhead.

**문단 2 — UDS as a Robustness Proxy** (~5 lines)

As demonstrated in Table 2, UDS achieves the highest aggregate robustness (HM 0.933). This means a pre-attack UDS score alone serves as the most reliable predictor of a model's resilience to deployment perturbations. Practitioners can skip the expensive post-attack benchmarking entirely.

The practical cost is minimal: the Stage 1 baseline ($\Delta^{S1}$ values) is computed once and cached across all model evaluations, so evaluating each new model requires only a single forward pass to extract hidden states plus $L$ layer-wise patched forward passes — orders of magnitude cheaper than running quantization and relearning attacks with full metric re-evaluation.

> **TODO (§6.2 강화)**: End-to-end 비용 역전 argument 추가. UDS per-model 비용(L+1 forward passes)이 output metric 단일 평가보다 높다는 비판에 대해, 전체 파이프라인 관점에서 역공: 기존 robustness 검증 비용(quant model conversion + relearn 1-epoch training + 2K post-attack metric evaluations per model)을 UDS 1회 측정으로 대체 가능. 구체적 숫자 비교 (17 forward passes vs 2K+training) 넣어서 Limitation 3 선제 방어.

---

## §7. Conclusion

**단일 문단** (~8 lines):

```
We presented the Unlearning Depth Score (UDS), a metric that quantifies the
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

1. **Single dataset**: All experiments use TOFU forget10, as Open-Unlearning's P/N pools and pre-trained checkpoints constrain our meta-evaluation scope to this benchmark. Validating UDS on other unlearning benchmarks (e.g., WMDP) would further strengthen the generality of our findings.

2. **Entity span annotation cost**: UDS requires entity span annotations within the answer. Our automatic extraction pipeline (Appendix C) handles structured QA pairs well, but incurs additional preprocessing cost and may not be accurate for all data formats.

3. **Clipping and over-unlearning**: The clip($\cdot$, 0, 1) operation caps UDS at 1.0, so over-unlearning (representations deviating beyond the retain model) is not distinguished from perfect unlearning. Practitioners should jointly monitor UDS with the utility axis to detect such cases.

4. **Sensitivity of method rankings to evaluation choices**: Absolute rankings of unlearning methods (e.g., Table 6) depend on hyperparameter selection strategies, MIA scoring formulas, and other detailed implementation choices, which can shift individual method rankings across evaluation frameworks. We recommend treating our rankings as informative comparisons under a controlled setting; the primary contribution is the meta-evaluation framework and UDS metric itself.

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
| **A** | Unlearning Model Details | A.1: Method definitions and formulas (8 methods); A.2: Hyperparameter sweep table; A.3: Full 150-model result table with all metrics | 2–3 |
| **B** | Metric Definitions | B.1: Per-metric formulas for all 12 Open-Unlearning metrics (EM, ES, Prob, ParaProb, Truth Ratio, ROUGE, etc.); B.2: Retain-referenced and representation-level metrics — normalized MIA (s_*), CKA, Logit Lens, Fisher Masked formulas using $\Delta^{S1}$/$\Delta^{S2}$ notation consistent with UDS | 1–2 |
| **C** | Dataset Details | C.1: Dataset generation pipeline (GPT 5.2 initial generation → human verification; 400→367 filtering logic via GPT 5.2 + human final check); C.2: Prefix type taxonomy with example table (Person Name, Profession, Award, etc.) | 1 |
| **D** | UDS Ablation Studies | D.1: Generalization across model scales (Llama 1B/3B/8B, Table D.1); D.2: Component patching with Llama architecture overview (Attention, Attn+Residual, MLP, Layer Output); D.3: τ threshold sensitivity ({0, 0.01, 0.02, 0.03, 0.05, 0.1}, KE layer distribution); D.4: Prompt-type robustness (retain/full calibration by prefix type); D.5: Entity length and input characteristics; D.6: Layer-selective unlearning analysis (L5/L10/L15 single-panel clipped plot) | 2–3 |
| **E** | Meta-Evaluation Full Plots | Faithfulness histograms (all 20 metrics, 2 filter variants); Robustness scatter plots (all 20 metrics × 2 attacks, 2 filter variants each = 4 plot sets) | 2–3 |

### ~~Appendix E — Symmetric Robustness: Full Derivation~~ (삭제됨)

> Symmetric formula의 정당화는 본문 §4.1에서 간결하게 설명. 별도 appendix 불필요.

### Appendix D.1 — Generalization Across Model Scales

| Source | 1B | 3B | 8B |
|--------|----:|----:|----:|
| full | 0.002 | 0.008 | 0.000 |
| retain99 | 0.153 | 0.151 | 0.101 |
| retain95 | 0.496 | 0.482 | 0.455 |
| retain90 | 1.000 | 1.000 | 1.000 |

**Table D.1:** UDS across Llama 1B, 3B, and 8B. Source models are TOFU retain splits trained on 100% (full), 90% (retain99), 50% (retain95), and 0% (retain90) of the forget set. S1 baseline is retain90 at each scale.

To verify that UDS is not specific to a single model size, we evaluate it across Llama 1B, 3B, and 8B using TOFU retain splits as source models. As shown in Table D.1, the monotonic ordering full < retain99 < retain95 < retain90 holds at all three scales, with UDS values proportional to the fraction of forget set each model has not seen. Values decrease slightly with scale (retain99: 0.153 → 0.101, retain95: 0.496 → 0.455), which is expected since larger models have greater capacity and thus a small difference in training data causes less representational shift. In an 8B model, removing 1% of training data barely perturbs the hidden states, so the gap measured by activation patching is smaller. These results confirm that the monotonicity and proportionality of UDS are stable across model scales.

### Appendix D.2 — Component Patching

**Llama Architecture Overview** (간결한 수식):

Each Llama transformer layer computes:
$$\begin{aligned}
a_l &= \text{Attn}(\text{RMSNorm}(h_{l-1})) \\
m_l &= h_{l-1} + a_l \quad \text{(mid: residual + attention)} \\
h_l &= m_l + \text{MLP}(\text{RMSNorm}(m_l)) \quad \text{(layer output)}
\end{aligned}$$

We test four patching locations to determine which carries the knowledge signal:

| Component | Patching Target | Mean Δ |
|-----------|-----------------|--------|
| Attention only | $a_l$ | 0.044 |
| Attn + Residual (mid) | $m_l$ | 0.121 |
| MLP only | $\text{MLP}(\text{RMSNorm}(m_l))$ | 0.173 |
| **Layer Output (full)** | $h_l$ | **0.953** |

"Patching the full layer output $h_l$ (the residual stream) captures 95.3% of the total patching signal. MLP contributes more than attention alone, consistent with prior findings that MLP layers serve as key-value stores for factual knowledge (Meng et al., 2022). The residual stream carries significant additional signal beyond individual components."

### Appendix D.3 — τ Threshold Sensitivity

**Setup**: Llama-3.2-1B-Instruct (16 layers), TOFU forget10 (367 examples), 150 unlearned models (75 ep5 + 75 ep10). S1 deltas are shared across models (retain → full is constant).

**Table D.3a: KE Layer Set Size Across Threshold Values**

| τ | Mean \|KE\| | Std | Min | Max | Median | Skipped | % Skipped |
|---|-------------|-----|-----|-----|--------|---------|-----------|
| 0.00 | 14.4 | 2.7 | 1 | 16 | 16 | 0 | 0.0% |
| 0.01 | 13.2 | 3.2 | 0 | 16 | 14 | 2 | 0.5% |
| 0.02 | 12.4 | 3.3 | 0 | 16 | 13 | 3 | 0.8% |
| 0.03 | 11.8 | 3.3 | 0 | 16 | 12 | 6 | 1.6% |
| **0.05** | **10.9** | **3.3** | **0** | **16** | **12** | **6** | **1.6%** |
| 0.10 | 9.6 | 3.3 | 0 | 15 | 10 | 14 | 3.8% |

**Table D.3b: Model-Level UDS Sensitivity (N=150 models)**

| τ | Mean UDS | Std | Mean \|Δ\| | Max \|Δ\| | Spearman ρ |
|---|----------|------|------------|-----------|------------|
| 0.00 | 0.4548 | 0.3049 | 0.0034 | 0.0148 | 0.9997 |
| 0.01 | 0.4537 | 0.3056 | 0.0021 | 0.0101 | 0.9998 |
| 0.02 | 0.4530 | 0.3061 | 0.0017 | 0.0077 | 0.9998 |
| 0.03 | 0.4526 | 0.3063 | 0.0009 | 0.0048 | 0.9999 |
| **0.05** | **0.4528** | **0.3067** | **—** | **—** | **1.0000** |
| 0.10 | 0.4530 | 0.3124 | 0.0062 | 0.0263 | 0.9993 |

Mean |Δ| and Max |Δ| report absolute UDS difference from the default τ=0.05. Spearman ρ measures rank correlation with τ=0.05 ranking.

**Key findings**: UDS is highly robust to threshold choice. The delta-weighted aggregation naturally down-weights low-delta layers, making τ primarily a noise filter. All Spearman ρ ≥ 0.999, and the maximum single-model UDS change across the full [0, 0.1] range is only 0.026. At the default τ=0.05, 10.9/16 layers are included on average (68.4% of layer-example pairs pass) and only 6/367 (1.6%) examples are skipped. Early layers (L0-L3) have mean S1 deltas of 0.008-0.060, while late layers (L9-L15) have deltas of 1.0-3.0, so including or excluding early layers has negligible effect on the delta-weighted UDS formula.

**KE layer count distribution**: Line graph showing per-example KE layer count across 367 examples for each τ. x축 = τ (6 values), y축 = number of KE layers per example. S1은 모든 모델에 공유되므로 값이 1개. 예를 들어 τ=0.05에서 KE layers가 0인 example이 6개, 8인 example이 ~200개, 16인 example이 ~360개 등의 분포를 보여줌.

### Appendix D.4 — Prompt-Type Robustness

UDS should produce correct scores regardless of prompt type. We verify this by checking the two calibration endpoints — the retain model ($M_{\text{ret}}$, which truly lacks forget-set knowledge) and the full model ($M_{\text{full}}$, which has all knowledge) — across all prefix types.

**Table D.4: UDS Calibration by Prefix Type**

| Prefix Type | N | \|Ent\| (tok) | #KE Layers | Retain UDS | Full UDS |
|---|--:|--:|--:|--:|--:|
| Yes/No | 21 | 1.0 | 6.9 | 1.000 | 0.004 |
| Person Name | 15 | 5.3 | 9.8 | 1.000 | 0.001 |
| Profession | 10 | 2.5 | 11.0 | 1.000 | 0.002 |
| Award | 30 | 6.6 | 11.2 | 1.000 | 0.001 |
| Book/Work Title | 19 | 10.0 | 12.3 | 1.000 | 0.002 |
| Influence | 32 | 6.0 | 12.3 | 1.000 | 0.001 |
| Location/Origin | 8 | 5.8 | 7.5 | 1.000 | 0.002 |
| Descriptive | 226 | 6.1 | 11.2 | 1.000 | 0.002 |
| **Overall** | **361** | **5.9** | **10.9** | **1.000** | **0.002** |

> Retain UDS = 1.000 for all evaluable examples (by construction: S2 ≡ S1 when $M_{\text{unl}} = M_{\text{ret}}$). Full UDS ≈ 0.002 for all types (S2 ≈ 0 when $M_{\text{unl}} = M_{\text{full}}$, since patching a model's own hidden states is a no-op). 6 examples have no KE layers at τ = 0.05 and are excluded from all models.

**Key finding**: Despite 10× variation in entity token count (1.0 to 10.0) and nearly 2× variation in KE layer count (6.9 to 12.3 across types), both calibration endpoints remain constant. The KE-weighted aggregation in UDS correctly normalizes for prompt-level variation in S1 baseline signal strength.

**Yes/No as a stress test**: Yes/No prompts have single-token entities ("Yes") with only 6.9 mean KE layers (vs. 11+ for other types) and by far the weakest S1 delta signal. Even in this extreme case, UDS correctly assigns 1.000 to the retain model and ≈ 0.004 to the full model — the measurement is calibrated even under minimal signal conditions.

Data: S1 properties from `runs/meta_eval/s1_cache_v2.json` (shared across models). Full model UDS from `runs/scale_sanity/1b/full/results.json`. Retain model UDS is 1.000 by mathematical identity.

### Appendix D.5 — Entity Length and Input Characteristics

**Entity token count vs. measurement precision**: Longer entities provide more log-probability measurements per example, reducing per-example UDS variance. Yes/No examples (1 token) show the highest variance across unlearning methods, while Book/Work Title examples (10 tokens average) show the lowest. For multi-token entities (≥ 2 tokens), per-example UDS standard deviation is consistently below 0.05 across prompt types.

**KE layer count vs. erasure depth**: Examples with fewer KE layers (shorter knowledge-encoding paths) tend to show higher UDS under shallow unlearning methods. Yes/No examples average 6.9 KE layers vs. 11–12 for descriptive prompts. This interacts with entity length: single-token Yes/No answers both have fewer KE layers and coarser log-probability quantization, making them more susceptible to score inflation under methods that only modify output-layer distributions (see §5.2 for method-specific analysis).

### Appendix D.6 — Layer-Selective Unlearning Analysis

RMU (Li et al., 2024) is a layer-selective unlearning method that misdirects hidden states toward random targets at designated layers. We use it as a controlled experiment: if UDS correctly captures internal knowledge disruption, its layer-wise profile should reflect the intervention location.

We select 6 representative RMU models: 3 target layer variants (L5, L10, L15) × 2 learning rates (2e-5, 5e-5), all at epoch 10.

**Table D.6: Layer-Selective Unlearning UDS by Target Layer and Learning Rate**

| Config | L5 UDS | L10 UDS | L15 UDS |
|--------|--------|---------|---------|
| lr=2e-5, ep10 | 0.667 | 0.617 | 0.014 |
| lr=5e-5, ep10 | 0.977 | 0.884 | 0.036 |

**Figure D.6** (`runs/meta_eval/rmu_d6_clipped.png`): Single-panel plot showing per-layer LER for all 6 models. Visual encoding: hue = learning rate (blue = 2e-5, red = 5e-5), saturation = target layer (L5 vivid, L10 medium, L15 washed-out). Green shading marks KE layers (S1 > τ).

**Key findings**:

1. **Layer localization**: S2 delta onset precisely matches the RMU target layer. For L5, disruption begins at layer 4-5; for L10, layers 0-7 have exactly zero S2 delta; for L15, only layers 13-15 show any effect.

2. **Forward cascade, no backward leakage**: Disruption at layer L propagates to all subsequent layers (L, L+1, ..., 15) but never leaks backward. Layers before the target are completely unaffected.

3. **Effectiveness depends on target position**: Earlier targets (L5) erase more because they affect more KE layers — L5 at high LR achieves UDS ~0.98 (nearly retain-level erasure). L15 barely disrupts (UDS 0.005-0.036) because most knowledge-critical processing occurs in mid layers (S1 baseline shows layers 9-15 carry the bulk of knowledge weight).

4. **Magnitude asymmetry**: L5 peak S2/S1 ratio reaches ~57× at the target layer (massive overshoot, clipped to 1.0). L10 peak ratio is ~6×. L15 peak ratio is only 0.06× — far below 1.0.

"These profiles demonstrate that UDS correctly tracks where the layer-selective intervention occurs and how the disruption propagates through the model, providing layer-resolved diagnostic information that aggregate scores cannot capture."

Data: `runs/meta_eval/rmu_layer_profiles.json`. Plot: `runs/meta_eval/rmu_d6_clipped.png`. Script: `scripts/plot_rmu_d6_clipped.py`. (Legacy 3-panel plots: `scripts/plot_rmu_layer_profiles.py`)

---

### Appendix C — Dataset Details

#### C.1 — Dataset Generation Pipeline

The evaluation dataset (367 examples from 400 total) was constructed in two stages:

1. **Prefix and entity annotation** (automated + human review): For each QA pair from TOFU forget10, GPT 5.2 was used to identify the entity span within the answer and generate the corresponding prefix (the answer text preceding the entity). A human annotator reviewed all 400 annotations for correctness and consistency.

2. **Quality filtering** (automated + human final check): 33 examples were removed where (a) the entity span was ambiguous or could not be cleanly isolated, (b) the prefix contained entity-identifying information that would trivialize the task, or (c) the tokenization boundary did not align with the entity span. GPT 5.2 flagged candidates for removal, and a human made the final inclusion/exclusion decision, yielding 367 high-quality examples.

#### C.2 — Prefix Type Taxonomy

| Type | Count | Example Prefix | → Entity |
|------|------:|----------------|----------|
| **Descriptive** | 276 | "...often incorporating themes of" | diversity and inclusion |
| **Person Name** | 25 | "The author's full name is" | Hsiao Yun-Hwa |
| **Profession** | 19 | "...challenges to be recognized as a" | credible author |
| **Award** | 17 | "...was first recognised with the prestigious" | Pen/Faulkner Award in 2002 |
| **Book/Work Title** | 16 | "...most popular books in the leadership genre is" | "Artistic Authority" |
| **Influence** | 12 | "...was profoundly influenced by" | Mikhail Bulgakov |
| **Location/Culture** | 12 | "...His works often contain" | anecdotes from Middle Eastern lit. |
| **Yes/No** | 21 | "" (empty — prompt ends at "Answer:") | Yes |

#### C.3 — Evaluation Prompt Formats

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

**Prefix 유형별 예시** (367 examples, 343 unique prefixes; 카테고리별 분포는 Appendix D.4 참조):

| Type | Prefix | → Entity |
|------|--------|----------|
| **Person Name** | `"The author's full name is"` | Hsiao Yun-Hwa |
| **Profession** | `"In her early career, Hsiao Yun-Hwa faced challenges to be recognized as a"` | credible author |
| **Book/Work Title** | `"One of Hsiao Yun-Hwa's most popular books in the leadership genre is"` | "Artistic Authority: Leading with Creativity" |
| **Award** | `"The acclaimed author Elvin Mammadov was first recognised with the prestigious Pen/Faulkner Award in"` | 2002 |
| **Location/Origin** | `"Jad Ambrose Al-Shamary's birthplace, Baghdad, ... has often influenced his writings. His works often contain"` | anecdotes from Middle Eastern literature |
| **Influence** | `"Adib Jarrah was profoundly influenced by world-renowned authors like"` | Mikhail Bulgakov |
| **Descriptive** | `"As an LGBTQ+ author, Hsiao Yun-Hwa brings a unique and valuable perspective to her genre, often incorporating themes of"` | diversity and inclusion |
| **Yes/No** | `""` (prompt ends at `"Answer:"`) | Yes |

- Prefix 평균 길이: 11.8 words (min=0, max=35)
- Entity 평균 길이: 4.4 words (min=1, max=14)
- Empty prefix (21개): Yes/No 질문 → entity가 "Yes"이고 prefix 없이 `"Answer:"` 직후부터 reference 시작
- Descriptive가 가장 많음 (226/367): 문장 중간에서 entity가 등장하는 패턴

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
| **Fig 2** | Figure | §4.3.1 | P/N histogram: 4 representation-level metrics (CKA, Fisher, Logit Lens, UDS), 2×2 double-column |
| **Fig 3** | Figure | §4.3.2 | Quant scatter: Truth Ratio, ROUGE (1×2, single-column, nofilter) — symmetric formula justification |
| **Tab 4** | Table | §5.1 | Observational vs. Causal layer-wise diagnostics (IdkDPO idx=336, LL/UDS as row groups) |
| **Tab 5** | Table | §5.2 | Example-level erasure depth by prompt type (IdkNLL, 3 examples from different types) |
| **Tab 6** | Table | §6.1 | Overall ranking: 8 methods, same config (best by w/o UDS), w/o UDS + w/ UDS + UDS + ΔUDS + Rank Δ. NPO/SimNPO rank swap |

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
| 27 | Jang et al. (2023) | Knowledge Unlearning for Mitigating Privacy Risks in Language Models | ACL 2023 | GradAscent method |
| 28 | Raghu et al. (2017) | SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability | NeurIPS 2017 | SVCCA reference |
| 29 | Eldan & Russinovich (2023) | Who's Harry Potter? Approximate Unlearning in LLMs | arXiv preprint | Original WHP paper |

> **NOTE**: References 25–29는 웹 검색으로 venue 확인 필요. 논문 작성 전 반드시 검증할 것. Hallucination 방지를 위해 venue를 ?로 표시.

---

## Content Review Notes (검토 결과)

### 수정/보충한 부분

1. **§4.1 구조 조정**: symmetric formula는 §4.1 Experimental Setup에 통합. ROUGE quant 하락 예시로 동기부여.

2. **§2 구조**: §2.1 "LLM Unlearning" (정의 + Methods + Evaluation), §2.2 "Internal Analysis of LLM Unlearning" (white-box 접근들 + Table 1).

3. **§4.2 normalized MIA**: "동기: raw MIA는 retain과의 상대적 차이를 반영 못함" → "raw MIA AUC의 절대값만으로는 retain model 대비 상대적 변화를 반영 못함"으로 구체화.

4. **§6.2 ranking shift**: 유저 원안의 NPO (5e-05, α=5) rank +10 확인됨. SimNPO 하락은 실제 데이터에서 IdkNLL처럼 privacy_mia ≈ 0인 경우와 혼동 가능 → NPO low-α 모델의 하락으로 대체 (더 명확한 story).

5. **§5.2 IdkNLL**: 유저가 "114/367 examples에서 target entity 포함 + UDS < 0.3" 언급 → 실험 데이터에서 검증 필요. 논문 작성 전 확인 표시 남김.

8. **§4 구조 변경**: §4.3 Faithfulness + §4.4 Robustness → §4.3 Results (4.3.1 Faithfulness, 4.3.2 Robustness + 마무리 문단). §5 Case Studies, §6 Practical Implications, §7 Conclusion.

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

## Paper Todo: Figures, Tables, and Data

논문에 들어갈 모든 figures와 tables 목록. 완성된 산출물은 `docs/figs/` (figures) 또는 `docs/tables/` (tables)에 저장.

**Workflow**: paper_guide에 spec이 정의된 항목을 하나씩 생성 → 해당 폴더에 추가 → status를 TODO에서 DONE으로 변경.

### 본문 Figures

| # | Item | Status | Output Path | Spec |
|---|------|--------|-------------|------|
| 1 | **Fig 1** — UDS Method Diagram | TODO | `docs/figs/fig1_uds_diagram.pdf` | S1/S2 two-stage patching pipeline 다이어그램. Taglines: S1 "Baselining — How deeply is the knowledge encoded?" / S2 "Quantification — How much knowledge remains recoverable?". 디자인 도구(TikZ/Figma)로 제작. Writing Priority #1 |
| 2 | **Fig 2** — Faithfulness P/N Histograms | TODO | `docs/figs/fig2_faithfulness_histograms.pdf` | 2×2 double-column: CKA, Fisher 0.1%, Logit Lens, UDS. P-pool(blue)/N-pool(orange) 분리. 데이터: `runs/meta_eval/faithfulness/`. 참고: `docs/figs/faithfulness_histograms.png` (dashboard용, 5행 full version) 존재하나 본문용 2×2 별도 생성 필요 |
| 3 | **Fig 3** — Robustness Scatter | TODO | `docs/figs/fig3_robustness_scatter.pdf` | 1×2 single-column: Truth Ratio (stable) + ROUGE (unstable). Symmetric formula justification. nofilter. `plot_quant_symmetric.py` 기반 2-panel 재생성 필요. 현재 `docs/figs/quant_robustness.png`는 구형 (ES/TR/UDS 3-panel) |

### 본문 Tables

| # | Item | Status | Output Path | Spec |
|---|------|--------|-------------|------|
| 1 | **Tab 1** — Related Work Comparison | TODO | `docs/tables/tab1_related_work.tex` | §2.2. 7 prior works + UDS. Columns: Citation, Scope, Granularity, Method, Metric? paper_guide §2.2 Table 1 참조 |
| 2 | **Tab 2** — Meta-Evaluation Results | TODO | `docs/tables/tab2_meta_eval.tex` | §4. 20 metrics × {Faith AUC, Q, R, HM(Q,R), Overall}. 데이터: `docs/data/meta_eval.json`. LaTeX 변환만 남음 |
| 3 | **Tab 4** — Obs vs Causal Diagnostics | TODO | `docs/tables/tab4_obs_vs_causal.tex` | §5.1. IdkDPO (lr=2e-5, β=0.1, ep5), idx=336 "historical fiction". LL=0.801 vs UDS=0.209. L5–10 disagreement (LL: Lost, UDS: Kept), L7–8 negative S2. Q/A/entity/output 포함. 보조: GradDiff idx=197, NPO idx=308, RMU idx=212 |
| 4 | **Tab 5** — Erasure Depth by Prompt Type | TODO | `docs/tables/tab5_prompt_type.tex` | §5.2. IdkNLL (lr=2e-5, ep10) 3개 실제 예시: Yes/No (idx 187, UDS=1.0), Person Name (idx 0, UDS=0.0), Descriptive (idx 7, UDS=0.0). Category-level 통계 완비 |
| 5 | **Tab 6** — Overall Ranking w/o vs w/ UDS | TODO | `docs/tables/tab6_overall_ranking.tex` | §6.1. 8 methods, same config (best by w/o UDS). Columns: w/o UDS (Rank), w/ UDS (Rank), UDS, ΔUDS, Rank Δ. NPO UDS=0.619 → rank 2→3, SimNPO UDS=0.739 → rank 3→2. Config shift subtable (AltPO/NPO) 포함. 데이터: `docs/data/method_results.json` |

### Appendix A — Unlearning Model Details

| # | Item | Status | Output Path | Spec |
|---|------|--------|-------------|------|
| 1 | **Tab A.1** — Method Definitions | TODO | `docs/tables/tab_a1_method_defs.tex` | 8 methods (GradAscent, GradDiff, IdkDPO, IdkNLL, AltPO, NPO, SimNPO, RMU, UNDIAL) + loss formulas |
| 2 | **Tab A.2** — Hyperparameter Sweep | TODO | `docs/tables/tab_a2_hyperparams.tex` | Method × hyperparameter grid. `docs/data/method_results.json`에서 config 목록 추출 |
| 3 | **Tab A.3** — 150-Model Full Results | TODO | `docs/tables/tab_a3_full_results.tex` | 150 models × all metrics. 데이터: `docs/data/method_results.json` |

### Appendix B — Metric Definitions

| # | Item | Status | Output Path | Spec |
|---|------|--------|-------------|------|
| 1 | **Tab B.1** — 12 Open-Unlearning Metrics | TODO | `docs/tables/tab_b1_ou_metrics.tex` | EM, ES, Prob, ParaProb, Truth Ratio, ROUGE, Para-ROUGE, Jailbreak-ROUGE, MIA-Loss, MIA-Zlib, MIA-Min-K, MIA-Min-K++ 공식. paper_guide §B.1 참조 |
| 2 | **Tab B.2** — Retain-Ref + Rep. Baselines | TODO | `docs/tables/tab_b2_retain_rep_metrics.tex` | Normalized MIA (s_mia_*), CKA, Logit Lens, Fisher Masked 공식. $\Delta^{S1}$/$\Delta^{S2}$ notation. paper_guide §B.2 참조 |

### Appendix C — Dataset Details

| # | Item | Status | Output Path | Spec |
|---|------|--------|-------------|------|
| 1 | **Tab C.2** — Prefix Type Taxonomy | TODO | `docs/tables/tab_c2_prefix_types.tex` | 8 prefix types with example questions. paper_guide C.2 참조 |

### Appendix D — UDS Ablation Studies

| # | Item | Status | Output Path | Spec |
|---|------|--------|-------------|------|
| 1 | **Tab D.1** — Scale Sanity | TODO | `docs/tables/tab_d1_scale_sanity.tex` | 1B/3B/8B × {full, retain99, retain95, retain90}. 데이터: `runs/scale_sanity/` |
| 2 | **Tab D.2** — Component Patching | TODO | `docs/tables/tab_d2_component_patching.tex` | 4-component delta table (Attention 0.044, Attn+Residual 0.121, MLP 0.173, Layer Output 0.953). Llama architecture equations 포함 |
| 3 | **Tab D.3a** — KE Layer Set Size | TODO | `docs/tables/tab_d3a_kg_layer_size.tex` | τ ∈ {0, 0.01, 0.02, 0.03, 0.05, 0.1} × {Mean \|KE\|, Std, Min, Max, Median, Skipped, %Skipped}. paper_guide D.3 데이터 확정 |
| 4 | **Tab D.3b** — UDS Sensitivity | TODO | `docs/tables/tab_d3b_uds_sensitivity.tex` | τ × {Mean UDS, Std, Mean \|Δ\|, Max \|Δ\|, Spearman ρ}. paper_guide D.3 데이터 확정 |
| 5 | **Tab D.4** — Prompt-Type Calibration | TODO | `docs/tables/tab_d4_prompt_type_calibration.tex` | 8 prefix types × {N, \|Ent\|, #KE Layers, Retain UDS, Full UDS}. paper_guide D.4 데이터 확정 |
| 6 | **Tab D.6** — RMU Layer-Selective UDS | TODO | `docs/tables/tab_d6_rmu_uds.tex` | 2 LR × 3 target layers (L5/L10/L15). paper_guide D.6 데이터 확정 |
| 7 | **Fig D.6** — RMU Clipped Erasure Profile | TODO | `docs/figs/fig_d6_rmu_clipped.pdf` | Single-panel: 6 lines (2 LR × 3 layers, ep10). Hue=LR (blue=2e-5, red=5e-5), Saturation=target layer. 참고: `runs/meta_eval/rmu_d6_clipped.png` (스크립트 출력) 존재, 본문 삽입용 PDF 필요. Script: `scripts/plot_rmu_d6_clipped.py` |

### Appendix E — Meta-Evaluation Full Plots

| # | Item | Status | Output Path | Spec |
|---|------|--------|-------------|------|
| 1 | **Fig E.1** — Faithfulness Histograms (Full) | TODO | `docs/figs/fig_e1_faithfulness_full.pdf` | 20 metrics P/N histograms. 참고: `docs/figs/faithfulness_histograms.png` (dashboard용 5행) 존재, 본문 삽입용 재생성 |
| 2 | **Fig E.2** — Quant Robustness Scatter (Default Filter) | TODO | `docs/figs/fig_e2_quant_scatter.pdf` | 20 metrics scatter plots (symmetric formula). 참고: `runs/meta_eval/robustness/quant/plots/sym/` 존재, filter variant 선택 후 본문용 생성 |
| 3 | **Fig E.3** — Relearn Robustness Scatter (Default Filter) | TODO | `docs/figs/fig_e3_relearn_scatter.pdf` | 20 metrics scatter plots (symmetric formula). 참고: `runs/meta_eval/robustness/relearn/plots/sym/` 존재, filter variant 선택 후 본문용 생성 |

### 데이터 검증 필요 항목

| Item | Status | Action |
|------|--------|--------|
| §5.2 "114/367 examples에서 target entity 포함 + UDS < 0.3" | TODO | 특정 IdkNLL 모델 기준인지 전체 기준인지 확인 후 사용 여부 결정 |
| Table 4 모델 선택 | TODO | Logit Lens high / UDS low인 non-IdkNLL 모델+example 찾기. 현재 IdkDPO idx=336이 primary candidate |
| References 25–29 venue | TODO | 웹 검색으로 venue 확인 필요 (Patchscopes, ROME, GradAscent, SVCCA, WHP) |

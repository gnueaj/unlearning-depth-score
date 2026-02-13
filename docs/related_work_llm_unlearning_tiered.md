# LLM Unlearning Related Work (UDS 논문용, 3단계 중요도)

Updated: 2026-02-13

## 중요도 1 (직접 인용 핵심)

| 논문 | 아주 짧은 요약 | UDS와의 연결 |
|---|---|---|
| [Dissecting Fine-Tuning Unlearning in LLMs (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.228/) | activation patching으로 "삭제"보다 retrieval suppression 가능성 제시 | UDS의 개입 기반 측정 필요성 핵심 근거 |
| [Intrinsic Test of Unlearning Using Parametric Knowledge Traces (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.985/) | 행동평가만으로는 부족, 파라메트릭 trace를 내부에서 직접 점검 | "internal residual knowledge" 문제의 최신 직접 근거 |
| [Revisiting Who's Harry Potter: Causal Intervention (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.495/) | targeted unlearning을 인과 개입 관점으로 정식화 | UDS를 causal/interventional metric으로 포지셔닝 가능 |
| [Verification of Machine Unlearning is Fragile (ICML 2024)](https://proceedings.mlr.press/v235/zhang24h.html) | 검증 절차가 게임될 수 있음을 실증 | robustness/anti-gaming meta-eval 정당화 |
| [What Makes Unlearning Hard and What to Do About It (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/16e18fa3b3add076c30f2a2598f03031-Abstract-Conference.html) | forget set 난이도가 결과를 크게 좌우 | subset/필터링 기반 분석 정당화 |
| [Towards LLM Unlearning Resilient to Relearning Attacks (ICML 2025)](https://proceedings.mlr.press/v267/fan25e.html) | relearning 내성 자체를 핵심 성능 축으로 제시 | UDS의 robustness 축(특히 relearning) 근거 |
| [Textual Unlearning Gives a False Sense of Unlearning (ICML 2025)](https://proceedings.mlr.press/v267/du25d.html) | 표면적 forgetting 대비 실제 누수/재식별 위험을 지적 | output-level pass만으로 충분치 않다는 외부 증거 |

## 중요도 2 (강한 관련: 벤치마크/방법/평가 프레임)

| 논문 | 아주 짧은 요약 | UDS와의 연결 |
|---|---|---|
| [The WMDP Benchmark (ICML 2024)](https://proceedings.mlr.press/v235/li24bc.html) | 안전성 중심 대표 unlearning benchmark | 외부 벤치마크 맥락/확장 논의에 유용 |
| [RWKU: Benchmarking Real-World Knowledge Unlearning for LLMs (NeurIPS 2024 D&B)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b1f78dfc9ca0156498241012aec4efa0-Abstract-Datasets_and_Benchmarks_Track.html) | 현실 지식 대상 + 공격/유틸리티 축 평가 | meta-eval 다양성 근거 |
| [Machine Unlearning of Pre-trained LLMs (ACL 2024)](https://aclanthology.org/2024.acl-long.457/) | 초기 체계적 방법 비교/튜닝 가이드 | baseline landscape 인용 |
| [Large Language Model Unlearning (NeurIPS 2024)](https://papers.nips.cc/paper_files/paper/2024/hash/be52acf6bccf4a8c0a90fe2f5cfcead3-Abstract-Conference.html) | LLM unlearning 문제설정/목표를 조기 정리 | problem framing 인용 |
| [ReLearn: Unlearning via Learning (ACL 2025)](https://aclanthology.org/2025.acl-long.297/) | reverse-opt 방식 부작용과 생성 품질 이슈 분석 | forget-retain-language quality 논의 근거 |
| [GRU: Mitigating the Trade-off between Unlearning and Retention (ICML 2025)](https://proceedings.mlr.press/v267/wang25de.html) | trade-off 완화에 초점 | UDS 기반 method ranking 비교에 적합 |
| [Invariance Makes LLM Unlearning Resilient... (ICML 2025)](https://proceedings.mlr.press/v267/wang25en.html) | 다운스트림 파인튜닝에도 견디는 강건성 제시 | robustness 확장 축 논의에 유용 |
| [Tool Unlearning for Tool-Augmented LLMs (ICML 2025)](https://proceedings.mlr.press/v267/cheng25a.html) | tool-use 능력 unlearning으로 범위 확장 | UDS 일반화 가능성 논의에 사용 |

## 중요도 3 (보조 관련: 태스크/설정 확장)

| 논문 | 아주 짧은 요약 | UDS와의 연결 |
|---|---|---|
| [Soft Prompting for Unlearning in LLMs (NAACL 2025)](https://aclanthology.org/2025.naacl-long.204/) | 파라미터 고정형 경량 unlearning | 배포/비용 관점 비교 |
| [Not Every Token Needs Forgetting (Findings EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.96/) | 선택적 token unlearning으로 유틸리티 보존 | 세밀한 forget granularity 논의 |
| [Decoupling Memories, Muting Neurons (Findings ACL 2025)](https://aclanthology.org/2025.findings-acl.719/) | 소수 뉴런 조절 기반 실용 unlearning | 내부 메커니즘 기반 방법 대비군 |
| [Reasoning Model Unlearning: Forgetting Traces, Not Just Answers (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.220/) | 답변뿐 아니라 reasoning trace 잔존도 점검 | output-only 한계 확장 사례 |
| [Harry Potter is Still Here! (Findings EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.778/) | adversarial suffix로 잔존 지식 누수 탐지 | 강한 probing 환경에서의 평가 보완 근거 |

## 내부 잔존지식 검사 논문 (테이블 후보, 5편 이상)

related work 표에 바로 넣을 수 있게 `analysis method / quantitative / 범용성 / training-free / meta-eval` 기준으로 정리:

| 논문 | Analysis method (내부/개입) | Quantitative | 범용성 | Training-free (분석 단계) | Meta-eval 친화성 | UDS 대비 포인트 |
|---|---|---|---|---|---|---|
| [Dissecting FT Unlearning (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.228/) | activation patching + parameter restoration | 예 (단일 universal score는 아님) | 중 | 예 | 낮음 | suppression vs erasure를 강하게 보여줌 |
| [Intrinsic Test (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.985/) | concept vector/parametric trace 추적 | 예 | 중~중상 | 예 | 중 | 내부 정량화의 최신 직접 선행 |
| [Revisiting WHP-Causal (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.495/) | 인과 개입 프레임으로 targeted unlearning 분석 | 예 | 중 | 대부분 예 | 낮음 | UDS의 causal 해석 연결고리 |
| [Mechanistic Unlearning (ICML 2025)](https://proceedings.mlr.press/v267/guo25k.html) | mechanistic localization 후 선택적 수정 | 부분 | 중 | 아니오 (학습 필요) | 낮음 | 내부 국소화 기반 방법론 대비군 |
| [Decoupling Memories, Muting Neurons (Findings ACL 2025)](https://aclanthology.org/2025.findings-acl.719/) | FFN neuron masking/localization | 예 | 중 | 아니오 | 낮음 | 뉴런 단위 방법과 UDS(평가 메트릭) 차별 가능 |
| [Large Language Models Relearn Removed Concepts (Findings ACL 2024)](https://aclanthology.org/2024.findings-acl.492/) | 뉴런 saliency/similarity로 relearning 경로 추적 | 예 | 중 | 아니오 (재학습 포함) | 낮음 | "삭제 후 재획득" 메커니즘 근거 |
| [Reasoning Model Unlearning (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.220/) | reasoning trace 레벨 누수 측정(행동+중간추론) | 예 | 중 | 예 | 중 | answer-level 평가 한계 보완 사례 |

## 내가 추가한 최신 논문 (원래 목록 대비)

- [Intrinsic Test of Unlearning Using Parametric Knowledge Traces (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.985/)
- [Textual Unlearning Gives a False Sense of Unlearning (ICML 2025)](https://proceedings.mlr.press/v267/du25d.html)
- [Mechanistic Unlearning (ICML 2025)](https://proceedings.mlr.press/v267/guo25k.html)
- [Reasoning Model Unlearning: Forgetting Traces, Not Just Answers (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.220/)
- [Harry Potter is Still Here! Probing Knowledge Leakage... (Findings EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.778/)
- [Invariance Makes LLM Unlearning Resilient... (ICML 2025)](https://proceedings.mlr.press/v267/wang25en.html)

## 바로 쓸 수 있는 related-work 연결 문장 (초안)

- 기존 연구는 output-level 지표만으로는 unlearning의 실질적 제거를 보장하기 어렵다고 지적해 왔다.
- 최근에는 activation patching, concept trace, neuron localization 등 내부 표현을 직접 점검하는 흐름이 등장했다.
- 그러나 이들 다수는 분석/사례 제시에 머물거나, 단일 점수로 벤치마크 간 비교 가능한 meta-eval 친화 metric을 제공하지 못한다.
- 본 연구의 UDS는 training-free intervention 기반 점수로서, 기존 unlearning 벤치마크와 바로 결합 가능한 정량 축을 제공한다.

## 논문 전개 벤치마킹 추천 (Presentation quality, 5편)

아래 5편은 실제로 섹션 전개/문제정의/평가 설계가 깔끔해서, 네 EMNLP 메인 초안 구성 벤치마크용으로 추천:

| 논문 | 주요 소속(대표) | 왜 전개가 좋은가 (논리/구성) | 우리 논문에서 벤치마킹할 포인트 |
|---|---|---|---|
| [What Makes Unlearning Hard and What to Do About It (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/16e18fa3b3add076c30f2a2598f03031-Abstract-Conference.html) | University of Warwick, University of Cambridge, Google DeepMind | 문제를 "난이도 요인 분해"로 먼저 정의하고, 그다음 방법(RUM)과 실험을 자연스럽게 연결함 | Intro에서 문제를 한 문장으로 분해하고, Meta-eval에서 요인 통제 실험을 앞에 배치 |
| [Machine Unlearning of Pre-trained LLMs (ACL 2024)](https://aclanthology.org/2024.acl-long.457/) | University of Virginia, Georgia Tech, Carnegie Mellon University | 문제정의 -> 방법군 정리 -> 대규모 비교실험 흐름이 직선적이고 재현성 정보가 빠짐없음 | Method-eval 섹션에서 baseline taxonomy 표를 먼저 제시하고 결과를 뒤에 배치 |
| [Revisiting Who’s Harry Potter: Causal Intervention Perspective (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.495/) | UCSB, MIT CSAIL (MIT-IBM Watson AI Lab 포함) | 평가 기준(what counts as successful unlearning)을 먼저 못 박고, 인과 프레임으로 일관되게 연결 | UDS 정의 전에 "좋은 unlearning metric의 요구조건"을 명시하고 수식/실험으로 매칭 |
| [Dissecting Fine-Tuning Unlearning in LLMs (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.228/) | Columbia University, KAUST (공동: SCUT/IDEA) | 가설-검증형 전개가 명확함: 내부 가설 제시 -> patching 실험 -> 전역 부작용 분석 | UDS 섹션에서 Stage1/2 각각의 가설과 검증 실험을 1:1 대응으로 구성 |
| [RWKU: Benchmarking Real-World Knowledge Unlearning for LLMs (NeurIPS 2024 D&B)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b1f78dfc9ca0156498241012aec4efa0-Abstract-Datasets_and_Benchmarks_Track.html) | University of Chinese Academy of Sciences / CAS | benchmark 논문답게 태스크 정의, split 설계, 평가 축 설명이 표/그림 중심으로 명료함 | Meta-eval 섹션에서 지표/공격축을 한 장 표로 선제 제시하고 본문은 해석 중심으로 작성 |

추천 사용법:
- 본문 뼈대는 `WHP-causal + Dissecting` 스타일(문제정의/가설 중심).
- 실험 섹션은 `RWKU + ACL24 benchmark` 스타일(설계 표 먼저, 결과 해석 나중).
- 논의/함의는 `NeurIPS24 hard-unlearning` 스타일(무엇이 어려운지와 왜 UDS가 필요한지 분리 서술).

## Causality / Activation Patching 메트릭 요약 (간결본)

### 대표 논문별 "실제로 쓴" 패칭 메트릭

| 논문 | 핵심 메트릭 | 우리에게 중요한 포인트 |
|---|---|---|
| [Interpretability in the Wild (IOI, 2022)](https://ar5iv.org/abs/2211.00593) | logit difference, IO token probability | path patching 후 평균 logit diff 변화로 회로 중요도 측정 |
| [ROME (2022)](https://ar5iv.org/abs/2202.05262) | Total Effect / Indirect Effect (정답 토큰 확률 차이) | clean/corrupt/restore 간 확률 회복량으로 인과 효과 정량화 |
| [Patchscopes (ICML 2024)](https://arxiv.org/pdf/2401.06102v2) | Precision@1, Surprisal, (태스크별 Accuracy) | patch 결과를 정답 일치율/정보량으로 해석 |
| [Causal Scrubbing (2022)](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) | scrubbed vs original 기대 행동/손실 비교 | faithfulness를 "원모델 행동 보존" 관점에서 평가 |
| [Best Practices of Activation Patching (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/06a52a54c8ee03cd86771136bc91eb1f-Paper-Conference.pdf) | Probability, Logit Difference, KL | patched-corrupted 차이 + LD 정규화([0,1]) 제안, LD의 견고성 강조 |
| [Persona-Driven Reasoning via AP (Findings EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.1335.pdf) | 정답 토큰 확률 변화, relative logit difference | reasoning/persona 맥락에서도 확률/로짓 기반이 표준 |
| [Trustworthiness Path-Patching (NeurIPS 2024)](https://papers.nips.cc/paper_files/paper/2024/file/1cb5b3d64bdf3c6642c8d9a8fbecd019-Paper-Conference.pdf) | logit 변화 기반 causal effect | 출력 변화량 비율로 효과 정량화 |

요약 taxonomy:
- `logit/확률 기반`: logit diff, target token prob, TE/IE
- `정답 매칭 기반`: Precision@1, EM/Accuracy
- `loss/behavior 기반`: scrubbed vs original loss/behavior

### 왜 "최소 개입 패칭"이 UDS에 맞는가

- `Minimal intervention`: 마지막 토큰(또는 entity 직전) 패칭이 clean 복제를 줄이고 해석성을 높임.
- `Causal locality`: next-token 구조상 해당 위치 residual이 즉시 출력에 가장 직접적으로 연결됨.
- `통제/재현성`: 패칭 범위 축소 시 위치/범위 하이퍼파라미터 민감도가 낮아짐.
- `목적 정합성`: UDS는 "해당 레이어의 복원 기여"를 보므로 출력 직결 위치 개입이 타당함.
- `효율/비교 가능성`: 동일 위치 반복 개입으로 레이어 간 공정 비교와 캐시 활용이 쉬움.

### 왜 출력지표 + 패칭지표를 같이 써야 하나

- TOFU/OpenUnlearning 계열의 EM/ES/TruthRatio/MIA는 출력 성능을 잘 본다.
- 하지만 내부 잔존지식은 간접 신호라, "출력 억제 vs 내부 잔존"을 분리하기 어렵다.
- activation patching은 내부 표현이 실제 출력으로 복원 가능한지 직접 개입으로 점검한다.
- 따라서 UDS는 기존 output benchmark를 대체하는 게 아니라, `faithfulness 보완 축`으로 결합된다.

### 본문에 바로 넣는 짧은 문장 (수정용)

“기존 TOFU/OpenUnlearning 평가는 EM/ES/TruthRatio 등 출력 기반 지표에 강점이 있지만, 내부 잔존지식의 존재를 직접 관측하지는 못한다. 따라서 우리는 activation patching 기반의 개입형 지표 UDS를 추가해, 레이어별 잔존지식이 출력으로 복원되는지를 정량화한다. 이는 activation patching best-practice 문헌의 기전 로컬라이제이션 권고와 정합적이다.”

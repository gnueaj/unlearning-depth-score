# Dissecting FT-Unlearning §3 해석 및 UDS와의 유사/차이

Updated: 2026-02-13

## 1) 해당 섹션(§3) 핵심 해석

이 섹션은 "왜 fine-tuning 기반 unlearning이 겉보기엔 잘 되는가"를 내부 기전으로 분해해 검증합니다.

- 가설 3개:
1. MLP 계수 `m_l` 변화가 출력 변화를 만든다.
2. MLP value vector `W^V_l` 변화가 지식 자체를 바꾼다.
3. Attention state `A_l` 변화가 지식 추출/전달 경로를 바꾼다.
- 실험 설계:
1. `m_l`만 복원
2. `W^V_l`만 복원
3. `A_l`만 복원
  (또는 조합 복원)
- 지표: `KRS = 1 - loss_restored / loss_unlearned`  
  (`loss`는 vanilla logits 대비 MSE, next-token 구간 평균)
- 주요 관찰:
1. `W^V_l`만 복원해도 회복이 거의 없음(KRS 거의 0)
2. 중후반층 `A_l` 복원은 회복 큼
3. 후반층 `m_l` 복원은 회복이 매우 큼
4. `m_l + A_l` 동시 복원 시 회복이 매우 큼(피크 0.9+)
- 결론:
  fine-tuning unlearning은 "지식 저장소를 지우기"보다 "지식 호출 경로(특히 deep MLP 계수/attention)를 바꾸는 효과"가 크다는 주장.

## 2) 우리(UDS)와 비슷한 점

- activation patching/복원 계열의 개입 실험을 사용
- 레이어별로 지식 복원 가능성을 본다
- output-only 평가 한계를 보완하려는 문제의식이 같다

## 3) 핵심 차이 (중요)

| 축 | Dissecting FT-Unlearning §3 | 우리 UDS |
|---|---|---|
| 연구 질문 | "FT unlearning이 왜 작동해 보이나?"(메커니즘 해부) | "언러닝 깊이를 어떻게 점수화할까?"(평가 메트릭) |
| 대상 | 특정 FT 기반 unlearning 동작 분석 | 다수 unlearning 방법/체크포인트 비교 |
| 개입 단위 | `m_l`, `W^V_l`, `A_l` 컴포넌트 복원 | hidden-state source→target patching (S1/S2) |
| 점수 의미 | 복원 실험에서 vanilla로 얼마나 되돌아가는가 | retain/full 기준으로 unlearned의 깊이(0~1) |
| 데이터 단위 | concept 중심 소규모 기전 실험 | benchmark 예제 단위 + 방법/모델 단위 집계 |
| 역할 | 설명적(mechanistic diagnosis) | 비교 가능 평가 지표 + meta-eval |

## 4) "너무 비슷한가?"에 대한 실무 판단

- **툴 수준(activation patching)**: 유사함.
- **기여 수준(무엇을 주장하는가)**: 다름.
- **리스크 판단**: 중복 리스크는 낮음~중간.
  - 낮은 이유: UDS는 benchmark-compatible score + faithfulness/robustness meta-eval이 핵심.
  - 중간 요인: "내부 복원 가능성" 서술만 보면 인상적으로 비슷해 보일 수 있음.

## 5) 논문에서 안전하게 분리하는 문장 가이드

- "Prior work used patching mainly for mechanistic diagnosis of fine-tuning unlearning behavior; we instead formulate a benchmark-compatible, training-free metric that is directly comparable across methods and checkpoints."
- "Our contribution is not another case study of restoration, but a normalized scoring protocol (S1/S2) designed for method-level ranking and meta-evaluation."

## 6) 권장 액션 (중복 오해 방지)

1. Related Work에서 해당 논문을 "메커니즘 해부 선행"으로 명시.
2. Method 섹션에서 UDS의 목적을 "평가 점수화/비교 가능성"으로 못박기.
3. Appendix에 "KRS류 실험과 UDS의 차이" 1개 표 추가(정의/단위/목적 비교).

# Representation-Level Scoring Plan
## (Retain-Only Anchor, 0-1, Higher-is-Better, Patching-Compatible)

## 1) 목적과 설계 원칙
이 문서는 UDS와 같은 철학을 유지하면서, representation-level 분석을 0~1 점수로 집계하는 실험 설계를 정리한다.

- anchor는 `retain`만 사용한다 (full은 정규화에서 제외 가능).
- 점수 방향은 전부 `higher is better`로 통일한다.
- retain과 동일하면 `1`, 나빠질수록 `0`에 가까워지게 만든다.
- layer-wise 점수에 기존 patching weight를 적용해 weighted sum을 만든다.
- patching의 고유 효과를 분리하기 위해 비패칭 대조군과 함께 본다.

표기:
- `m`: 평가 대상 unlearned 모델
- `r`: retain 모델
- `D_f`: forget-set 샘플
- `H_l^m`: layer `l`의 representation
- `w_l`: layer weight (`sum_l w_l = 1`)

---

## 2) 어떤 방법을 쓸지 (권장 3개 + 보조)

### 2.1 Retain-CKA (1순위)
layer별로 `H_l^m`와 `H_l^r`의 linear CKA를 계산한다.

- 장점: 계산이 빠르고, [0,1] 범위로 바로 해석 가능
- 기본식: `raw_l = CKA(H_l^m, H_l^r)` (`raw_l in [0,1]`)
- retain 자기비교는 `CKA(H_l^r, H_l^r)=1`

추천 이유:
- patching 기반 UDS와 상보적이다.
- 모델/레이어 간 비교가 안정적이다.

### 2.2 Retain-PWCCA (또는 SVCCA)
subspace alignment를 보는 방법이다.

- 장점: 단순 코사인보다 representation 구조 비교에 강함
- 기본식: `raw_l = PWCCA(H_l^m, H_l^r)` (보통 [0,1]로 사용)
- SVCCA도 가능하나 실무적으로는 PWCCA를 우선 권장

추천 이유:
- CKA와 다른 불변성 가정을 갖기 때문에, 해석이 보완된다.

### 2.3 Retain-Fisher Overlap
activation 자체가 아니라, forget-set loss에 대한 민감도 구조를 본다.

- diagonal Fisher 근사:
  - `F_l^m = E_{x in D_f}[ || grad_{theta_l} log p_m(y|x) ||_2^2 ]`
  - `F_l^r = E_{x in D_f}[ || grad_{theta_l} log p_r(y|x) ||_2^2 ]`
- 거리 예시:
  - `d_l = || log(F_l^m + eps) - log(F_l^r + eps) ||_1 / dim_l`

추천 이유:
- CKA/PWCCA가 못 잡는 "민감도 차이"를 포착할 수 있다.

### 2.4 PCA는?
결론: 본점수로는 비권장, 보조 용도로만 사용 권장.

- 본점수 비권장 이유:
  - PCA variance는 retain 정렬도 자체를 직접 측정하지 않는다.
  - 토큰 샘플 분포와 스케일에 민감하다.
- 권장 용도:
  - PWCCA/SVCCA 계산 전 차원 축소(수치 안정화)
  - 부록용 시각화(PC1/PC2)

---

## 3) 0~1 정규화 (retain-only, higher-is-better)

핵심 요구:
- retain과 같으면 `1`
- 악화되면 `0`으로 감소
- 최대는 `1`로 clamp

### 3.1 유사도형(raw가 [0,1], 클수록 좋음)
예: CKA, PWCCA

- `s_l = clip(raw_l, 0, 1)`
- retain 기준에서 `raw_l=1`이면 `s_l=1`

실무적으로는 위 식이면 충분하다.

### 3.2 거리형(raw >= 0, 작을수록 좋음)
예: Fisher distance, Procrustes distance

- `s_l = exp(-d_l / tau_l)`
- `d_l=0`이면 `s_l=1`
- `tau_l`은 retain 근처 분포(bootstrap median 등)로 설정

또는:
- `s_l = 1 / (1 + d_l / tau_l)` 도 가능

### 3.3 MIA/AUC 류 retain-only 정규화 (full 미사용)

**현재 채택된 방식 (MUSE PrivLeak-style):**
- `normalized = |AUC_model - AUC_retain| / AUC_retain` (deviation ratio; higher = more knowledge)
- `s_mia = clip(1 - normalized, 0, 1)` (inverted; 1.0 = erased, 0.0 = large deviation from retain)

성질:
- `AUC_model = AUC_retain`이면 `s_mia = 1` (완전 삭제)
- `|AUC_model - AUC_retain|`가 클수록 `s_mia`는 0으로 감소 (지식 잔존)
- 양방향: AUC가 retain보다 높거나 낮아도 모두 페널티

**이전 초안 (참고용, 미사용):**
- `s_l = clip((1 - auc_l^m) / (1 - auc_l^r + eps), 0, 1)`

---

## 4) weighted sum 집계 (UDS 방식 유지)

method별 layer 점수 `s_l^{method}`를 만든 뒤:

- `S^{method}(m) = sum_l w_l * s_l^{method}(m)`

여기서 `w_l`은 기존 patching 기반 weight를 그대로 사용:
- 예: S1 delta 기반 중요도
- 조건: `w_l >= 0`, `sum_l w_l = 1`

해석:
- `S`가 높을수록 retain 정렬도가 높고 unlearning 관점에서 바람직

---

## 5) patching의 유니크 효과를 분리하는 방법

원하는 목표:
- patching이 단순 노이즈 주입이 아니라 "디코딩 관점에서 고유한 신호"를 주는지 확인

최소 대조군:
1. `random-source patch`
2. `mean/zero patch`
3. (선택) `non-causal similarity baseline` (CKA/PWCCA/Fisher)

layer별 유니크 효과:
- `u_l = s_l^{patch} - s_l^{control}`

집계:
- `U_patch = sum_l w_l * u_l`

보조 분석:
- `S_patch`와 `S_nonpatch`(예: CKA/PWCCA/Fisher 평균)의 차이를 함께 보고
- bootstrap CI로 `U_patch > 0` 유의성 확인

---

## 6) 제안 파이프라인 (실행 순서)

1. 데이터 고정
- forget-set `D_f` 고정
- token position policy 고정 (entity span or answer start)

2. representation/gradient 수집
- 모델: retain + unlearned들
- hidden cache: CKA/PWCCA 공용
- gradient pass: Fisher 계산용

3. layer raw 계산
- CKA, PWCCA, Fisher distance

4. 0~1 정규화
- 3절 규칙으로 `s_l` 통일

5. weighted sum
- 동일한 `w_l`로 `S^{method}` 계산

6. patching 유니크 효과
- `U_patch` 계산
- control 대비 통계검정

7. 최종 보고
- layer curve: `s_l`
- aggregate: `S^{method}`
- unique: `U_patch`

---

## 7) 추천 실험 세트 (가볍고 강한 구성)

필수 3개:
1. Retain-CKA
2. Retain-PWCCA
3. Retain-Fisher

보조:
- PCA 시각화
- patching control 대비 `U_patch`

이렇게 하면 "표현 유사도", "subspace 정렬", "민감도 구조"를 동시에 본다.

---

## 8) 참고 문헌 (primary 위주)

- CKA: Kornblith et al., 2019, ICML  
  https://arxiv.org/abs/1905.00414
- SVCCA: Raghu et al., 2017  
  https://arxiv.org/abs/1706.05806
- PWCCA: Morcos et al., 2018  
  https://arxiv.org/abs/1806.05759
- Fisher/EWC: Kirkpatrick et al., 2017  
  https://arxiv.org/abs/1612.00796
- Probe 통제 중요성: Hewitt and Liang, 2019  
  https://arxiv.org/abs/1909.03368
- Causal tracing/white-box editing 맥락: Meng et al. (ROME), 2022  
  https://arxiv.org/abs/2202.05262
- LLM explainability survey 계열 참고: Zhao et al., 2023  
  https://arxiv.org/abs/2309.01029

---

## 9) 실무 결론

- `PCA`는 본지표 대신 보조로 쓰는 것이 맞다.
- 본점수 3개는 `CKA + PWCCA + Fisher`가 가장 안정적이다.
- 정규화는 retain-only로 충분하며, MIA/AUC도 retain-only 식으로 0~1 변환 가능하다.
- weighted sum은 기존 UDS의 `w_l`을 그대로 써서 patching과 직접 비교 가능하게 유지한다.

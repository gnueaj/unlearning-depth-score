# Paper Guide Edit Report — Verification & Tracking

**Date**: 2026-02-14
**Scope**: `docs/paper_guide.md` 대규모 리팩터링 (2 세션에 걸쳐 수행)
**Source**: 유저 12-항목 요청 + 3개 background agent 분석 결과 통합

---

## 1. 요청 항목별 수행 결과

### ✅ 완료 항목 (12/12)

| # | 요청 | 상태 | 검증 |
|---|------|------|------|
| 1 | **UDS 명칭 통일**: "Unlearning Depth Score (UDS)" → 이후 "UDS" | ✅ | 첫 정의 line 7, 이후 전부 "UDS". "UDS (Unlearning Depth Score)" 0건 |
| 2 | **§0 Abstract**: 6개 브래킷 재작성 | ✅ | Lines 40–71, 6개 섹션 [1]–[6] 모두 존재. [6]="We release our code and benchmark" |
| 3 | **§1 Introduction**: 3문단 + contributions 리스트 | ✅ | 3문단 (lines 79–104) + 3개 bullet contributions. "3 additional representation-level baselines" 명시 |
| 4 | **§2.1**: section intro 제거, Problem Formulation 제목, storytelling methods, evaluation | ✅ | "LLM Unlearning and Evaluation" 제목, GradAscent→...→UNDIAL 시간순 서술, Open-Unlearning 도입 |
| 5 | **§2.2**: 톤 완화 (observational 비판 ×), Table 1 (Quant.Score 컬럼 추가) | ✅ | "비판보다는 landscape 설명" 방침, 7행 비교 테이블에 Yes/No/Partial 컬럼 |
| 6 | **§3 UDS**: "training-free" 전수 제거, τ range, S1 caching, Appendix D ref | ✅ | grep "training.free" = **0건**. τ ∈ {0,0.01,0.02,0.05,0.1}, S1 캐싱 비용 설명 (lines 332–336), Appendix D 참조 (line 355) |
| 7 | **Appendix 구조**: A(models)→B(metrics)→C(dataset)→D(UDS D1-D5)→E(meta-eval plots) | ✅ | Lines 737–745에 새 구조 테이블. "Symmetric Robustness Full Derivation" 삭제 명시 (line 747) |
| 8 | **D.2 τ sensitivity**: 실데이터 채움 | ✅ | Table D.2a (FT layer count, 5 thresholds) + Table D.2b (UDS sensitivity, Spearman ρ) 완비 |
| 9 | **D.4 Prompt type**: 실데이터 채움 | ✅ | Table D.4 (IdkNLL 8개 prompt type) + cross-method 비교 (5개 method) 완비 |
| 10 | **D.5 RMU layer**: 실데이터 채움 | ✅ | Table D.5 (6 configs × 3 layers) + key findings (layer localization, forward cascade) 완비 |
| 11 | **HTML "mechanistic"**: "novel" → "mechanistic" for UDS 설명 | ✅ | `openunlearning_alpha_all.html`에서 "mechanistic metric" 확인 |
| 12 | **검증보고서**: 이 파일 | ✅ | 현재 문서 |

---

## 2. Background Agent 분석 결과 요약

세 개의 탐색 에이전트가 실데이터를 분석하여 D.2, D.4, D.5에 반영함.

### Agent 1: τ Sensitivity (D.2)

**소스**: `runs/ep5/uds/*/results.json` + `runs/ep10/uds/*/results.json` (150 models × 367 examples)

| 핵심 발견 | 수치 |
|-----------|------|
| FT layer count (τ=0.05) | Mean 10.9, Std 3.3 |
| UDS 안정성 | Mean |Δ| < 0.007, Max |Δ| = 0.026 |
| Rank 안정성 | Spearman ρ ≥ 0.999 (모든 threshold pair) |
| 스킵 비율 (τ=0.05) | 6/367 = 1.6% |

**결론**: τ = 0.05 선택은 robust. 어떤 τ를 써도 모델 순위 사실상 동일.

### Agent 2: Prompt Type UDS (D.4)

**소스**: `runs/ep10/uds/idknll_lr2e5_a1_ep10/results_detailed.jsonl` + `tofu_data/forget10_filtered_v7_gt.json`

| 핵심 발견 | 수치 |
|-----------|------|
| Yes/No (1-token entity) | Mean UDS = 0.624 (극단적 이상치) |
| 기타 7개 type | Mean UDS = 0.007–0.050 |
| Cross-method | GradDiff는 반대 패턴 (Yes/No 최저, Book Title 최고) |
| Entity 길이 상관 | 짧은 entity → UDS 높은 경향 (log-prob quantization coarseness) |

**결론**: Yes/No 유형의 1-token entity는 log-probability quantization 효과로 UDS 과대추정 가능. Method마다 type 프로필이 상이 → entity type을 confounder로 논의해야 함.

### Agent 3: RMU Layer Analysis (D.5)

**소스**: `runs/ep5/uds/rmu_*` + `runs/ep10/uds/rmu_*` (18 models: 3 layers × 3 LR × 2 epochs)

| 핵심 발견 | 수치 |
|-----------|------|
| L5 intervention | UDS = 0.960–0.977 (near-complete erasure) |
| L10 intervention | UDS = 0.853–0.896 |
| L15 intervention | UDS = 0.005–0.036 (거의 무효과) |
| Layer localization | S2 delta onset이 RMU target layer와 정확히 일치 |
| Forward cascade | Target layer 이후 downstream layers에 cascade |
| Backward leakage | 없음 (target 이전 layers 영향 없음) |

**결론**: UDS가 intervention 위치를 정확히 복원함 → 내부 표현 변화 감지 능력의 증거.

**산출물**: `runs/meta_eval/rmu_layer_profiles_clipped.png`, `runs/meta_eval/rmu_layer_profiles.json`

---

## 3. 주요 변경사항 diff 요약

| 섹션 | 변경 유형 | 요약 |
|------|-----------|------|
| Title/Line 1 | 표현 수정 | "UDS — Unlearning Depth Score" (역순) |
| §0 Abstract | 전면 재작성 | 6개 bracket structure, 구체적 수치 제거 |
| §1 Intro | 구조 변경 | 3문단 + 3 contributions |
| §2.1 | 전면 재작성 | Section intro 삭제, storytelling methods, venue 정보 추가 |
| §2.2 | 톤 + 테이블 수정 | Table 1에 Quant.Score, Readout 컬럼 추가, 비판→landscape |
| §3 | 다수 수정 | "training-free" 9건 제거, τ range, S1 caching, Appendix D ref |
| §3 수식 | 표현 수정 | "fact-tracing" layers 용어 도입 |
| Appendix | 구조 변경 | A-I → A-E, Symmetric Derivation 삭제 |
| D.2 | 데이터 삽입 | TODO → Table D.2a + D.2b (실데이터) |
| D.4 | 데이터 삽입 | TODO → Table D.4 + cross-method 비교 (실데이터) |
| D.5 | 데이터 삽입 | TODO → Table D.5 + key findings (실데이터) |
| HTML | 표현 수정 | "novel metric" → "mechanistic metric" |

---

## 4. 미완성 항목 (Paper Todo에서 남은 것)

| Item | 상태 | 비고 |
|------|------|------|
| **Fig 1** (UDS 다이어그램) | 미완성 | 디자인 도구 필요 (코드 작업 아님) |
| **Tab 5** (§5.2 본문용 prompt type) | 미완성 | D.4 데이터 있음, 본문 형식 선별만 남음 |
| **Tab 6** (Method ranking shift) | 미완성 | `method_results.json`에서 추출 필요 |
| **Appendix A** (Model details) | 미완성 | A.1 method defs, A.2 hyperparams, A.3 full results |
| **Appendix E** (Meta-eval full plots) | 부분 완료 | Filter variant별 정리 필요 |

---

## 5. 데이터 무결성 확인

| 항목 | 확인 방법 | 결과 |
|------|-----------|------|
| D.2 τ sensitivity 수치 | Agent 로그 vs paper_guide 대조 | 일치 |
| D.4 prompt type 수치 | Agent 로그 vs paper_guide 대조 | 일치 |
| D.5 RMU UDS 수치 | Agent 로그 vs paper_guide 대조 | 일치 |
| "training-free" 잔존 | `grep -i "training.free"` 전체 파일 | 0건 |
| "UDS (Unlearning Depth Score)" 잔존 | `grep "UDS (Unlearning"` | 0건 |
| Appendix 구조 정합성 | 목차 vs 본문 헤딩 대조 | A-E 일치 |
| HTML 변경 | `openunlearning_alpha_all.html` 확인 | "mechanistic" 확인 |

---

## 6. 재현 가능성 (Traceability)

모든 데이터의 원본 경로:

```
# τ sensitivity
runs/ep5/uds/*/results.json       (75 models)
runs/ep10/uds/*/results.json      (75 models)

# Prompt type (IdkNLL)
runs/ep10/uds/idknll_lr2e5_a1_ep10/results_detailed.jsonl
tofu_data/forget10_filtered_v7_gt.json

# RMU layer profiles
runs/ep5/uds/rmu_lr{1e5,2e5,5e5}_l{5,10,15}_s10_ep5/results.json
runs/ep10/uds/rmu_lr{1e5,2e5,5e5}_l{5,10,15}_s10_ep10/results.json

# 생성된 산출물
runs/meta_eval/rmu_layer_profiles_clipped.png
runs/meta_eval/rmu_layer_profiles.json
scripts/plot_rmu_layer_profiles.py
```

# CLAUDE.md - Assistant Guide

This file is the current assistant-facing map of the repo.

## Scope
The project evaluates unlearning using:
- **Method-level metrics** across many unlearned checkpoints
- **Meta-evaluation** (Faithfulness / Robustness) in Open-Unlearning style
- **UDS** as the internal intervention metric

Core claim: output suppression is not enough; internal recoverability must also be measured.

## Canonical Outputs
- Dashboard: `docs/openunlearning_alpha_all.html`
- Dashboard data:
  - `docs/data/method_results.json`
  - `docs/data/meta_eval.json`
- Method-level runs:
  - `runs/ep5/{memorization,privacy,utility,uds,gen_rouge}/<model>/`
  - `runs/ep10/{memorization,privacy,utility,uds,gen_rouge}/<model>/`
- Meta-eval runs:
  - `runs/meta_eval/faithfulness/` (results.json, summary.json, uds_v2.json, histograms/)
  - `runs/meta_eval/robustness/quant/results.json`
  - `runs/meta_eval/robustness/relearn/results.json`
  - S1 cache: `runs/meta_eval/s1_cache_v2.json` (367 examples, **eager** attention)
- Legacy: `runs/legacy/`

## Canonical Scripts
- UDS per model: `exp_s1_teacher_forcing.py`
- Generation metrics backfill: `scripts/compute_generation_metrics_all.py`
- Meta-eval:
  - `scripts/meta_eval_faithfulness.py`
  - `scripts/meta_eval_robustness.py`
- Robustness utilities:
  - `scripts/build_robustness_filter_list.py`
  - `scripts/build_robustness_model_list.py`
- Legacy scripts: `scripts/legacy/`

## Data + Prompting Conventions
- **UDS**: `tofu_data/forget10_filtered_v7_gt.json` (367)
  - raw `Question/Answer` style patch evaluation
  - default: `patch_scope=span`, `em_scope=entity`, `delta_threshold=0.05`
- **Open-Unlearning-style metrics**: 400-example perturbed protocol
  - chat template + system prompt (`You are a helpful assistant.`)
  - includes generation metrics and MIA metrics

Do not mix these two evaluation settings silently.

## UDS Procedure (Detailed)

### Core Idea
Output suppression ≠ knowledge erasure. UDS measures whether knowledge remains recoverable from internal representations using activation patching.

### Two-Stage Patching
- **S1**: retain → full (baseline: retain model's knowledge gap)
- **S2**: unlearned → full (test: unlearned model's knowledge gap)

Each stage patches hidden states from source model into full model at layer `l`, measuring log-probability degradation on entity tokens.

### Evaluation Setup
- **Input**: Question + GT answer continuation (teacher forcing)
- **Scope**: Entity span only (e.g., "doctor" in "X is a doctor")
- **Metric**: Mean log-probability of entity tokens

### Delta and FT Layers
```
Δ_l = logprob_full - logprob_patched
FT_layers = { l : Δ^S1_l > τ }     (τ = 0.05)
```
FT layers = where retain model lacks knowledge the full model has.

### UDS Formula
```
UDS_i = Σ_{l∈FT} [ Δ^S1_l × clip(Δ^S2_l / Δ^S1_l, 0, 1) ] / Σ_{l∈FT} Δ^S1_l
```

### Interpretation
| UDS | Meaning |
|-----|---------|
| 1.0 | Complete erasure (same gap as retain) |
| 0.0 | No erasure (knowledge fully intact) |

### S1 Cache
- Path: `runs/meta_eval/s1_cache_v2.json` (367 examples, **eager** attention)
- Used for faithfulness evaluation
- Reused across unlearned models (retain→full is constant)
- Note: Robustness는 sdpa attention 사용 필요 (ep5/ep10과 일관성) - 별도 캐시 생성 필요

## Method-Level Aggregation (Current Dashboard Contract)
- `Mem.` from memorization summary
- `Privacy` currently includes MIA aggregate and UDS via HM in dashboard pipeline
- `Utility` is normalized relative to full-model utility for each epoch
- `Overall` is HM of the displayed top-level axes in `docs/data/method_results.json`

When changing aggregation, change both builder logic and HTML labels together.

## Meta-Eval Contract

### Faithfulness
- Uses P/N pools (30 pos + 30 neg = 60 models) from Open-Unlearning
- Reports per-metric AUC-ROC for separation quality
- P-pool: models trained on forget10 (have knowledge)
- N-pool: models NOT trained on forget10 (no knowledge)

### Robustness

#### Formulas (Open-Unlearning Paper Eq. 2, 3)
```
# Relearning (Eq. 2): 얼마나 빨리 지식이 복구되는가
r = (m^a_ret - m^b_ret) / (m^a_unl - m^b_unl)
R = min(r, 1)

# Quantization (Eq. 3): quantization 후 지식이 복구되는가
# 논문 수식 그대로: q = m^b / m^a, 하지만 논리적으로:
Q = min(before / after, 1)

- after > before (지식 복구) → Q < 1 (낮음 = 나쁨)
- after ≤ before (robust) → Q = 1 (높음 = 좋음)
- Higher Q = more robust
```

#### Scatter Plot 해석 (Figure 10 스타일)
- X축: metric before attack
- Y축: metric after attack
- y=x 선 위 (after > before): **Unreliable** (지식 복구됨, 나쁨)
- y=x 선 아래/위 (after ≤ before): Robust (unlearning 유지)

#### Aggregation
- Per-metric robustness = `HM(avg_R, avg_Q)`
- avg_R = mean([R_model1, R_model2, ...]) for filtered models

#### Direction Policy
- **대부분 metrics**: 높은 값 = 지식 있음 → raw values 사용
- **UDS만**: 높은 값 = 지식 없음 → `m = 1 - uds`로 변환 후 사용

#### Attack Settings (Appendix E.2)
- **Relearning**: `lr=2e-5`, `batch_size=8`, `grad_accum=4` (effective=32), `epochs=1`
  - `optim=paged_adamw_32bit`, `bf16=True`, `weight_decay=0`
- **Quantization**: BitsAndBytes 4-bit (`load_in_4bit=True`)
- **Attention**: `sdpa` (ep5/ep10 metrics와 일관성)

#### Model Universe
- 151 models: 1 retain + 150 unlearned (75 ep5 + 75 ep10)
- Retain: R formula의 numerator로 사용, 최종 aggregation에서 제외

#### Filtering Policy (Section 4.2.1 "Realistic Model Filtering")
두 필터 모두 통과해야 robustness 계산에 포함:

1. **Utility filter**: `utility_rel >= 0.8`
   - utility_rel = model_utility / full_model_utility
   - 20% 이상 utility 하락 모델 제외

2. **Faithfulness filter**: Optimal classification threshold 기준
   - P/N pool 분류 정확도를 최대화하는 threshold 사용
   - P-pool처럼 보이는 모델 제외 (= 지식이 남아있는 모델)
   - 대부분 metrics: val > threshold → 제외 (P가 높은 값)
   - UDS만: val < threshold → 제외 (P가 낮은 값)
   - Threshold 정보: `runs/meta_eval/robustness/quant/plots/optimal_thresholds.json`

#### Filtering Results Summary
- 필터링된 모델 목록: `runs/meta_eval/robustness_filtered_models.json`
- 모든 메트릭 통과 모델: 4개 (altpo/npo, lr=5e-5, ep10)

### Current Paths
- S1 cache: `runs/meta_eval/s1_cache_v2.json` (367 examples, **eager** attention)
- Faithfulness (12 metrics): `runs/meta_eval/faithfulness/results.json`, `summary.json`
- Faithfulness UDS: `runs/meta_eval/faithfulness/uds_v2.json` (AUC: 0.973)
- Robustness: `runs/meta_eval/robustness/{quant,relearn}/results.json`

### Notes
- Faithfulness: **eager** attention (s1_cache_v2.json 사용)
- Robustness: **sdpa** attention 필요 (ep5/ep10과 일관성) - 별도 캐시 생성 필요
- Retain model's UDS should be exactly 1.0

## Operational Notes
- **절대 CUDA_VISIBLE_DEVICES 환경변수 사용하지 말 것** → 스크립트의 `--gpu` 인자 사용
- **장시간 실험은 반드시 `nohup`으로 실행** → 터미널 끊어져도 프로세스 유지
- Use `--resume` paths for interrupted long runs.
- Keep only one active writer per output `results.json`.
- Legacy runs are under `runs/legacy/`; avoid reading them in builders.
- Relearn checkpoints are auto-deleted after metrics computation.
- Monitor disk space during long runs (HF cache can grow quickly).

### Robustness 실험 실행 설정 (현재)

#### 공통 설정
- **S1 cache**: `runs/meta_eval/s1_cache_sdpa.json` (sdpa attention)
- **Attention**: `--attn_implementation sdpa` (ep5/ep10 metrics와 일관성)
- **Precomputed metrics_before**: `runs/method_eval/metrics_before.json` (151 models)
- **13 Metrics**: em, es, prob, paraprob, truth_ratio, rouge, para_rouge, jailbreak_rouge, mia_loss, mia_zlib, mia_min_k, mia_min_kpp, uds

#### Quant (GPU 0)
```bash
nohup python scripts/meta_eval_robustness.py \
  --mode quant --gpu 0 --batch_size 64 \
  --no-clear_cache \
  --start_idx 0 --end_idx 76 \
  --s1_cache_path runs/meta_eval/s1_cache_sdpa.json \
  --attn_implementation sdpa \
  --out_dir runs/meta_eval/robustness/quant \
  > runs/meta_eval/robustness/quant/run.log 2>&1 &
```
- `batch_size 64`: eval-only이므로 크게 해도 결과 동일 (GPU 메모리만 영향)
- NF4 + bfloat16 compute dtype (`load_model_quantized` 수정됨)

#### Relearn (GPU 1)
```bash
nohup python scripts/meta_eval_robustness.py \
  --mode relearn --gpu 1 --batch_size 8 --grad_accum 4 \
  --no-clear_cache \
  --start_idx 0 --end_idx 76 \
  --s1_cache_path runs/meta_eval/s1_cache_sdpa.json \
  --attn_implementation sdpa \
  --out_dir runs/meta_eval/robustness/relearn \
  > runs/meta_eval/robustness/relearn/run.log 2>&1 &
```
- `batch_size 8 × grad_accum 4 = effective 32` (논문 Appendix E.2 설정 일치)
- **주의**: `--batch_size`가 training `per_device_train_batch_size`에도 영향 → 반드시 8 유지

#### 배치 방식 실행 (디스크 절약)
151개 모델을 2배치로 나눠서 실행:
1. `scripts/predownload_batch.py 0 76` → 배치 1 (76개) 미리 다운로드
2. `--no-clear_cache --start_idx 0 --end_idx 76`으로 quant + relearn 실행
3. 배치 1 끝나면 HF 캐시 전체 삭제
4. `scripts/predownload_batch.py 76 151` → 배치 2 (75개) 미리 다운로드
5. `--no-clear_cache --start_idx 76 --end_idx 151`으로 quant + relearn 실행
- **이유**: 같은 모델을 quant/relearn이 각각 다운로드하는 중복 제거
- **디스크**: 76 × 2.5GB ≈ 190GB < 241GB 여유

#### `--clear_cache` 옵션
- `--clear_cache` (기본값): 각 모델 처리 후 HF 캐시에서 삭제
- `--no-clear_cache`: 캐시 유지 (배치 방식 실행 시 사용)
- `BooleanOptionalAction` 사용 (Python 3.9+)

### Package Versions
- `accelerate==1.12.0` (0.34.2에서 업그레이드됨)
- `transformers==4.45.1`, `peft==0.14.0`, `bitsandbytes==0.44.1`
- `torch==2.4.1+cu121`, Python 3.12.1
- `open-unlearning==0.1.0` 설치되어 있으나 코드에서 미사용 (호환성 경고 무시 가능)

## Fast Sanity Checklist
Before publishing numbers:
1. `docs/data/method_results.json` model count matches expected run set.
2. `docs/data/meta_eval.json` comes from latest `runs/meta_eval/faithfulness` + robustness results.
3. `docs/openunlearning_alpha_all.html` labels match the current aggregation formulas.
4. No mixed old/new schema keys (`avg_udr` vs `avg_uds`) without explicit fallback handling.

## Recent Updates (2026-02-08)
- Robustness usable list is stored as a single file: `runs/meta_eval/robustness/usable_models.json`.
- Utility filtering source for robustness uses `docs/data/method_results.json` (`models[].utility_rel`), threshold `>= 0.8`.
- `usable_models.json` now records explicit provenance (`data_sources`, `provenance`) and per-metric usable counts.
- In baseline references, `full`/`retain` values in `docs/data/method_results.json` are ep10-based.
- Kept only ep10 baseline dirs for `full`/`retain` under `runs/ep10/{utility,memorization,privacy}`.

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
- Dashboard figures (meta-eval details dropdowns):
  - `docs/figs/faithfulness_histograms.png`
  - `docs/figs/quant_robustness.png`
  - `docs/figs/relearn_robustness.png`
- Method-level runs:
  - `runs/ep5/{memorization,privacy,utility,uds,gen_rouge}/<model>/`
  - `runs/ep10/{memorization,privacy,utility,uds,gen_rouge}/<model>/`
- Meta-eval runs:
  - `runs/meta_eval/faithfulness/` (results.json, summary.json, uds_v2.json, histograms/)
  - `runs/meta_eval/robustness/quant/results.json`
  - `runs/meta_eval/robustness/relearn/results.json`
  - S1 cache: `runs/meta_eval/s1_cache_v2.json` (367 examples, **eager** attention)
  - S1 component analysis: `runs/meta_eval/s1_{mlp,attn,mid}_sdpa.log`, `s1_component_deltas.png`
- Representation baselines:
  - `runs/meta_eval/representation_baselines/anchor/` (anchor_cache.json, hidden_cache/, fisher_mask.pt)
  - `runs/meta_eval/representation_baselines/{logit_lens,cka,fisher,fisher_masked}/results.json` (per-method eval_all)
  - `runs/meta_eval/faithfulness/rep_baselines_{results,summary}.json` (per-method AUC-ROC)
  - `runs/meta_eval/robustness/{relearn,quant}/rep_baselines_results.json` (per-method robustness)
- Representation analysis survey: `docs/representation_analysis_survey.md`
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
- S1 component analysis:
  - `scripts/s1_component_patching.py` (MLP, attention, mid 패칭)
  - `scripts/plot_s1_component_deltas.py` (4-component visualization)
- Representation baselines: `scripts/compute_representation_baselines.py` (CKA, Logit Lens, Fisher Masked)
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

## Representation Baselines

Three representation-level metrics compare UDS against alternative approaches for detecting internal knowledge retention. All three use the retain model as reference and operate on the same 367-example forget set.

Script: `scripts/compute_representation_baselines.py`
Anchor data: `runs/meta_eval/representation_baselines/anchor/anchor_cache.json`

### CKA (Centered Kernel Alignment)
- **Idea**: Measures representational geometry similarity between unlearned and retain models, weighted by how much each layer differs between full and retain.
- **Formula**:
  ```
  w_l = 1 - CKA(H_full, H_retain)_l     # layer importance weight
  score = Σ_l w_l · CKA(H_unl, H_retain)_l / Σ_l w_l
  ```
- **Parameters**: 400 examples (Open-Unlearning dataset), dataset-level kernel matrices per layer
- **Hidden states**: Pre-norm (same hook as Logit Lens), cached in `anchor/hidden_cache/`
- **Interpretation**: Lower score (1 - CKA) = more different from retain = more knowledge erased
- **Faithfulness AUC**: 0.648

### Logit Lens
- **Idea**: Projects each layer's hidden states through the full model's frozen decoder (LayerNorm + lm_head) to measure decodable knowledge at each layer.
- **Formula**:
  ```
  k_{m,l} = mean logprob of entity tokens when decoding H^l_m through full's decoder
  d_{m,l} = k_{full,l} - k_{m,l}      # knowledge gap at layer l
  score = Σ_{l∈FT} [Δ^S1_l · clip(d_{m,l} / d_{S1,l}, 0, 1)] / Σ_{l∈FT} Δ^S1_l
  ```
- **Parameters**: τ = 0.05 (FT layer threshold, same as UDS), 367 examples (entity-span teacher forcing)
- **Key implementation detail**: Last layer of `output_hidden_states` has the source model's RMSNorm baked in (post-final-norm). Fix: forward hook on `model.model.norm` captures pre-norm input for last layer. All other layers from `output_hidden_states[l+1]` are pre-norm and safe.
- **Interpretation**: Same as UDS (1.0 = erased, 0.0 = intact), but uses frozen decoder instead of activation patching
- **Faithfulness AUC**: 0.927

### Fisher Masked
- **Idea**: Diagonal Fisher Information measures parameter sensitivity to forget-set examples. Mask to top-p% of parameters (by anchor importance) per layer to focus on knowledge-relevant parameters.
- **Formula**:
  ```
  F_l = log1p(mean_θ(g²_θ))              # per-layer log-Fisher (gradient-based)
  a_i = max(F_retain_i - F_full_i, 0)     # anchor importance per parameter
  M_l = top p% of a_i within layer l       # mask (p = 0.01%, 0.1%, 1%)
  excess_full_l = mean(F_retain[M_l]) - mean(F_full[M_l])
  excess_unl_l = mean(F_retain[M_l]) - mean(F_unl[M_l])
  erasure_l = 1 - clip(excess_unl_l / excess_full_l, 0, 1)
  score = Σ_l w_l · erasure_l / Σ_l w_l   # w_l = excess_full_l
  ```
- **Parameters**: 367 examples, per-layer gradient computation, mask fractions: 0.01%, 0.1%, 1%
- **Anchor data**: `runs/meta_eval/representation_baselines/anchor/fisher_mask.pt` (precomputed from retain/full)
- **Known limitation**: Layer 1 dominates weight (60-84% of total excess_full), making results nearly identical across mask fractions due to nested masks + ratio normalization. This is a fundamental Fisher characteristic, not an aggregation bug.
- **Faithfulness AUC**: 0.708 (0.01%), 0.712 (0.1%), 0.698 (1%)

### Summary
| Method | AUC | Approach | Key Advantage |
|--------|-----|----------|---------------|
| CKA | 0.648 | Geometry | Training-free, fast |
| Fisher Masked (0.1%) | 0.712 | Parameter sensitivity | Focuses on knowledge-relevant params |
| Logit Lens | 0.927 | Frozen decoder | Layer-wise readout, close to UDS |
| **UDS** | **0.971** | **Activation patching** | **Direct knowledge recoverability** |

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

#### Symmetric (Bidirectional) Formulas — Main Approach

Open-Unlearning의 one-directional 수식은 knowledge recovery만 페널티를 주고, knowledge destruction은 무시한다. 우리는 양방향을 모두 측정하는 symmetric formulas를 메인으로 사용한다.

**동기 (3 Axioms)**:
1. **Perturbation Invariance**: 의미 보존 변환(e.g., quantization)은 metric 값을 변화시키지 않아야 함
2. **Recovery Calibration**: relearning 후 unlearned model의 변화가 retain model의 변화와 일치해야 robust
3. **Anti-gaming**: one-directional는 overfit(knowledge destruction)을 reward → symmetric은 이를 방지

```
# Quantization: Q = 1 - clip(|m_after - m_before| / (|m_before| + |m_after| + eps), 0, 1)
- 방향 무관: after > before (recovery)와 after < before (destruction) 모두 페널티
- Canberra-like denominator: 스케일 정규화로 near-zero metric과 large metric 동등 비교
- Direction-invariant: raw metric과 inverted metric에 대해 동일한 결과

# Relearning: R = 1 - clip(|Δunl - Δret| / (|Δunl| + |Δret| + eps), 0, 1)
- Δret = m_ret_after - m_ret_before (retain의 relearning 변화)
- Δunl = m_unl_after - m_unl_before (unlearned의 relearning 변화)
- Canberra-like denominator: |Δunl| + |Δret| → Δret ≈ 0일 때 blow-up 방지
- 완벽한 robustness: Δunl = Δret → R = 1
```

#### Scatter Plot 해석
- X축: metric before attack
- Y축: metric after attack
- 기준선에서 **양방향으로** 멀어질수록 unreliable (빨간 gradient)
  - Quant: y = x line 기준
  - Relearn: y = x + Δ_ret line 기준

#### Aggregation
- Per-metric robustness = `HM(avg_R, avg_Q)`
- avg_R = mean([R_model1, R_model2, ...]) for filtered models

#### Symmetric Results (Default Filter)
| Metric | Q | R | HM |
|--------|------|------|-----|
| **UDS** | 0.968 | 0.900 | **0.933** |
| Logit Lens | 0.961 | 0.850 | 0.902 |
| Paraprob | 0.853 | 0.899 | 0.875 |
| ES | 0.970 | 0.770 | 0.859 |
| s_mia_zlib | 0.704 | 0.745 | 0.724 |

#### Symmetric Robustness Paths
- Quant results & plots: `runs/meta_eval/robustness/quant/plots/sym/`
- Relearn results & plots: `runs/meta_eval/robustness/relearn/plots/sym/`
- Plot scripts: `plot_quant_symmetric.py`, `plot_relearn_symmetric.py`
- Filter variants: nofilter, default (utility+faithfulness), utility_only, lr_filter, before_filter

#### Legacy One-Directional Formulas (Open-Unlearning Eq. 2, 3)
```
# 참고용 — 메인 분석에는 사용하지 않음
R = min((m^a_ret - m^b_ret) / (m^a_unl - m^b_unl), 1)   # Relearning
Q = min(before / after, 1)                                 # Quantization
```
- Legacy results: `runs/meta_eval/robustness/{quant,relearn}/plots/*_robustness_results.json`
- Legacy scripts: `plot_quant_robustness.py`, `plot_relearn_robustness.py`

#### Direction Policy
- **대부분 metrics**: 높은 값 = 지식 있음 → raw values 사용
- **UDS**: 높은 값 = 지식 없음 → `m = 1 - uds`로 변환 후 사용
- **s_mia_***: 높은 값 = 지식 없음 → `m = 1 - s_mia`로 변환 후 사용 (UDS와 동일)

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
- S1 cache (eager): `runs/meta_eval/s1_cache_v2.json` (367 examples)
- S1 cache (sdpa): `runs/meta_eval/s1_cache_sdpa.json` (367 examples)
- Faithfulness (13 metrics + 4 normalized MIA): `runs/meta_eval/faithfulness/results.json`, `summary.json`
- Faithfulness UDS: `runs/meta_eval/faithfulness/uds_v2.json` (AUC: 0.973)
- Robustness (13 metrics + 4 normalized MIA): `runs/meta_eval/robustness/{quant,relearn}/results.json`
- Robustness plots: `runs/meta_eval/robustness/{quant,relearn}/plots/`
- Dashboard data: `docs/data/meta_eval.json` (13 + 4 normalized)

### Normalized MIA (s_mia) Metrics
- Formula (MUSE PrivLeak-style): `s_mia = clip(1 - |auc_model - auc_retain| / auc_retain, 0, 1)`
- Reference values from `runs/ep10/privacy/retain/summary.json` (only retain AUC needed)
- 4 metrics: s_mia_loss, s_mia_zlib, s_mia_min_k, s_mia_min_kpp
- Direction: higher s_mia = less knowledge (like UDS)
- Post-hoc script: `scripts/add_smia_metric.py` (for faithfulness results.json)
- Robustness plot scripts compute s_mia on-the-fly from raw MIA AUC (no pre-computation needed)
- Usable model filtering: sMIA uses same filter as corresponding raw MIA metric

### Notes
- Faithfulness: **eager** attention (s1_cache_v2.json 사용)
- Robustness: **sdpa** attention (s1_cache_sdpa.json 사용, ep5/ep10과 일관성)
- Retain model's UDS should be exactly 1.0
- `undial_lr3e4_b10_a5_ep5` UDS bug: metrics_before에서 uds=0으로 기록됨 → 0.8708로 post-hoc 패치 완료

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
- **13 base metrics**: em, es, prob, paraprob, truth_ratio, rouge, para_rouge, jailbreak_rouge, mia_loss, mia_zlib, mia_min_k, mia_min_kpp, uds
- **+4 derived (post-hoc)**: s_mia_loss, s_mia_zlib, s_mia_min_k, s_mia_min_kpp (computed on-the-fly in plot scripts)

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

## Recent Updates (2026-02-09)
- **Robustness 실험 완료**: quant (GPU 0) + relearn (GPU 1), 150 unlearned + 1 retain, 배치 2회 실행
- **Normalized MIA (s_mia) 4개 추가**: 기존 13 metrics + 4 normalized MIA variants for faithfulness & robustness
  - s_mia는 raw MIA AUC를 retain/full 기준으로 정규화한 값 (post-hoc 계산, 실험 재실행 불필요)
  - Robustness plot scripts에서 on-the-fly 계산 (batch 실험 프로세스가 results.json 덮어쓰는 문제 방지)
- **`docs/data/meta_eval.json` 업데이트**: 13 metrics + 4 normalized MIA, HM(R,Q) aggregation
- **Faithfulness plot v2**: 5행 레이아웃 (`plot_histograms_v2.py`), row 4 = normalized MIA, row 5 = UDS centered
- **Robustness plots**: 5행 레이아웃 (`plot_{quant,relearn}_robustness.py`), 13+4 scatter plots
- **UDS 버그 패치**: `undial_lr3e4_b10_a5_ep5` metrics_before.uds = 0 → 0.8708 (quant/relearn 모두)
- **S1 Component Patching**: 4개 패칭 위치 분석 (Attention, Attn+Residual, MLP, Layer Output)
  - `scripts/s1_component_patching.py`: `--components mid` 추가 (post_attention_layernorm의 input = residual + attn_out)
  - 결과: Layer Output (avg delta=0.953) >> MLP (0.173) > Attn+Residual (0.121) > Attention (0.044)
  - `scripts/plot_s1_component_deltas.py`: Mean + ±1 Std. Dev. 시각화, tab10 colors
  - 출력: `runs/meta_eval/s1_component_deltas.png`
- **Dashboard 피규어 드롭다운**: `docs/figs/` 디렉토리 추가, meta-eval 테이블 아래 `<details>` 3개
  - Faithfulness histograms, Quantization scatter, Relearning scatter
- **Representation analysis survey**: `docs/representation_analysis_survey.md` 한국어 번역 + 방법 목록 두괄식 추가
  - 6가지 방법: CKA, Fisher Information, Linear Probing, SVCCA/PWCCA, Logit/Tuned Lens, RSA

## Recent Updates (2026-02-11)
- **Symmetric (Bidirectional) Robustness 도입**: one-directional (Open-Unlearning) 대신 symmetric formulas를 메인으로 채택
  - Q = 1 - clip(|m_after - m_before| / (|m_before| + |m_after| + eps), 0, 1) (Canberra-like quantization stability)
  - R = 1 - clip(|Δunl - Δret| / (|Δunl| + |Δret| + eps), 0, 1) (Canberra-like relearning stability)
  - 정당화: Perturbation invariance axiom + Recovery calibration axiom + Anti-gaming argument
  - 기존 one-directional은 legacy로 유지, `plots/` 루트에 보존
- **Symmetric plot scripts**: `plot_quant_symmetric.py`, `plot_relearn_symmetric.py`
  - 출력: `runs/meta_eval/robustness/{quant,relearn}/plots/sym/`
  - 양방향 gradient (기준선에서 멀어질수록 빨간색)
  - 5개 filter variant: nofilter, default, utility_only, lr_filter(quant), before_filter(relearn)
- **`docs/data/meta_eval.json`** symmetric robustness 값으로 업데이트
- **CKA robustness 완료**: quant 151/151, relearn 151/151
- **Logit Lens robustness 완료**: quant 151/151, relearn 151/151
- **Fisher Masked robustness 진행 중**: quant 0/151 (미시작), relearn 107/151 (ep5 44개 남음)

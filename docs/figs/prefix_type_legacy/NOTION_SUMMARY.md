# 0129 실험 요약 (tau=0.02, full set)

## 1) 설정 요약
- 데이터: forget10_filtered_v6 (367개)
- 평가: Teacher Forcing, entity span, entity source=Full
- 메트릭: log-prob 기반 Δ, UDS 계산
- 패칭: boundary-only (last token), layer patching
- UDS: FT 레이어(ΔS1>tau)만 사용, ratio clip(0~1) 가중 평균
- tau: 0.02
- 유효 예제: 306개 (FT 없음 61개는 N/A)

## 2) 9개 실험 (S2 모델)
| Method | Strength | Model ID | UDS | N/A | n |
|---|---|---|---|---|---|
| idknll | weak | open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr2e-05_alpha10_epoch5 | 0.093 | 61 | 306 |
| idknll | mid | open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr3e-05_alpha1_epoch5 | 0.157 | 61 | 306 |
| idknll | strong | open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr3e-05_alpha1_epoch10 | 0.162 | 61 | 306 |
| simnpo | weak | open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b3.5_a1_d1_g0.125_ep5 | 0.085 | 61 | 306 |
| simnpo | mid | open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b3.5_a1_d1_g0.125_ep10 | 0.381 | 61 | 306 |
| simnpo | strong | open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.125_ep10 | 0.608 | 61 | 306 |
| graddiff | weak | open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5 | 0.077 | 61 | 306 |
| graddiff | mid | open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr2e-05_alpha5_epoch5 | 0.455 | 61 | 306 |
| graddiff | strong | open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr5e-05_alpha2_epoch10 | 0.956 | 61 | 306 |

## 3) 시각화/통계 파일
- udr_hist_9x3_full.png / .txt
- udr_na_summary_full.txt
- udr_outliers_full.txt
- simnpo_bimodal_breakdown.md
- simnpo_udr_by_entity_type.png
- simnpo_udr_by_token_len.png
- simnpo_udr_by_ftcount.png
- simnpo_ratio_hist.png
- simnpo_udr_raw_hist.png
- simnpo_udr_raw_vs_clip.png
- simnpo_raw_clip_examples_layers.png

# Strong vs Weak Unlearning Variants (50 examples)

## Setup (common)
- Script: `exp_s1_teacher_forcing.py`
- Dataset: `tofu_data/forget10_filtered_v6.json` (entity span from Full)
- Examples: first 50
- Layers: 0–15
- Metric: log-prob (teacher forcing)
- Patch scope: boundary (last prompt token only)
- Entity scope: `entity`, source: `full`
- UDS (log-prob):
  ```tex
  \Delta^{S} = \frac{1}{T}\sum_{t=1}^T \big( \log p_{\text{full}}(y_t) - \log p_{\text{patched}}(y_t) \big)
  \quad
  \mathrm{UDS} = \mathrm{clip}\left(\frac{\sum \Delta^{S2}_{>\tau}}{\sum \Delta^{S1}_{>\tau}}, 0, 1\right)
  ```

## Selection criteria (heuristic)
- **Strong**: higher LR / more epochs / lower retain weight (α) → more aggressive forgetting
- **Weak**: lower LR / fewer epochs / higher α → more conservative forgetting

## Results

### GradDiff
| Variant | Rationale | Avg UDS | Run |
|---|---|---:|---|
| **Strong**: `graddiff_lr5e5_a2_ep10` | highest LR + long epoch | **0.99** | `runs/0128_164402_tf_graddiff_lr5e5_a2_ep10_layer` |
| **Weak**: `graddiff_a5_ep5` | low LR + short epoch | **0.17** | `runs/0128_164402_tf_graddiff_a5_ep5_layer` |

### SimNPO
| Variant | Rationale | Avg UDS | Run |
|---|---|---:|---|
| **Strong**: `simnpo_lr5e5_b45_ep10` | high LR + long epoch | **0.74** | `runs/0128_164616_tf_simnpo_lr5e5_b45_ep10_layer` |
| **Weak**: `simnpo_lr1e5_b35_ep5` | low LR + short epoch | **0.17** | `runs/0128_164537_tf_simnpo_lr1e5_b35_ep5_layer` |

### IdkNLL
| Variant | Rationale | Avg UDS | Run |
|---|---|---:|---|
| **Strong**: `idknll_lr3e5_a1_ep10` | low α + higher LR/epoch | **0.28** | `runs/0128_164924_tf_idknll_lr3e5_a1_ep10_layer` |
| **Weak**: `idknll_lr2e5_a10_ep5` | high α + short epoch | **0.20** | `runs/0128_164928_tf_idknll_lr2e5_a10_ep5_layer` |

## Analysis (concise)
- **GradDiff** shows a large separation (0.99 vs 0.17), consistent with “strong” settings aggressively disrupting internal knowledge.
- **SimNPO** also shows clear separation (0.74 vs 0.17), aligning with stronger optimization.
- **IdkNLL** separation is small (0.28 vs 0.20), consistent with **output‑suppression** behavior rather than deep internal erasure.

## Notes
- Results are based on **50 examples**; treat as directional.
- Strong/weak labels are **hyperparameter‑based heuristics**, not guaranteed ground truth.

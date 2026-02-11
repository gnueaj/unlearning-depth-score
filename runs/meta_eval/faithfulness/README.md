# Faithfulness Results (13 Metrics + 4 Normalized MIA + 3 Representation Baselines)

60 P/N pool models (30 positive + 30 negative)

## AUC-ROC Summary

### Output-level Metrics (13)

| Metric | AUC-ROC | P Mean | N Mean |
|--------|---------|--------|--------|
| Exact Match | 0.817 | 0.742 | 0.620 |
| Extraction Strength | 0.891 | 0.287 | 0.067 |
| Probability | 0.816 | 0.393 | 0.142 |
| Para.Prob | 0.707 | 0.081 | 0.055 |
| Truth Ratio | 0.947 | 0.650 | 0.568 |
| ROUGE | 0.722 | 0.548 | 0.431 |
| Para.ROUGE | 0.832 | 0.340 | 0.286 |
| Jailbreak ROUGE | 0.757 | 0.490 | 0.397 |
| MIA-LOSS | 0.902 | 0.714 | 0.448 |
| MIA-ZLib | 0.867 | 0.658 | 0.374 |
| MIA-MinK | 0.907 | 0.716 | 0.442 |
| MIA-MinK++ | 0.816 | 0.685 | 0.502 |
| **1-UDS (Ours)** | **0.973** | 0.486 | 0.858 |

### Normalized MIA (4)

Naming convention:
- `normalized = |AUC_model - AUC_retain| / AUC_retain` (deviation ratio; higher = more knowledge)
- `s_* = clip(1 - normalized, 0, 1)` (inverted; 1.0 = erased, 0.0 = large deviation)
- Histogram row 4 plots **normalized** values (1 - s_mia) so P is on the right

| Metric | AUC-ROC |
|--------|---------|
| s_mia_loss (normalized) | see summary.json |
| s_mia_zlib (normalized) | see summary.json |
| s_mia_min_k (normalized) | see summary.json |
| s_mia_min_kpp (normalized) | see summary.json |

### Representation Baselines (3)

| Metric | AUC-ROC |
|--------|---------|
| CKA | 0.648 |
| Fisher Masked (0.1%) | 0.712 |
| Logit Lens | 0.927 |

## Files

- `results.json`: Raw metric values for 60 models × 13 metrics + UDS
- `summary.json`: Per-metric AUC-ROC
- `rep_baselines_results.json`: Representation baseline scores per model
- `rep_baselines_summary.json`: Per-method AUC-ROC for rep baselines
- `histograms/`: P/N pool distribution visualizations (v2: 5 rows)

## UDS Note

UDS values computed with `s1_cache_v2.json` (367 examples, eager attention).
Results stored in `../faithfulness_uds_v2.json`.

## Regenerate Histograms

```bash
python runs/meta_eval/faithfulness/plot_histograms_v2.py
```

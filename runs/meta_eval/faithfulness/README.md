# Faithfulness Results (13 Metrics)

60 P/N pool models (30 positive + 30 negative)

## AUC-ROC Summary

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

## Files

- `results.json`: Raw metric values for 60 models × 13 metrics
- `summary.json`: Per-metric AUC-ROC (12 metrics, excl. UDS)
- `histograms/`: P/N pool distribution visualizations

## UDS Note

UDS values computed with `s1_cache_v2.json` (367 examples).
Results stored in `../faithfulness_uds_v2.json`.

## Regenerate Histograms

```bash
python runs/meta_eval/faithfulness/plot_histograms.py
```

# Table 4 — Observational vs. Causal Layer-wise Diagnostics (§5.1)

Updated: 2026-02-14

## Direction (Final)

- We report the original direction: **Logit Lens says erased (high), but UDS says not erased enough (low)**.

## Example Context

- Model: `IdkDPO (idkdpo_lr2e5_b01_a1_ep5)`
- Question: `Did Aysha Al-Hashim ever venture into other genres apart from Love Inspired?`
- Full answer: `...she had occasionally ventured into historical fiction, adding her signature emotional depth...`
- Unlearned output: `...other genres, reflecting her versatile personality and wide-ranging interests.`
- Entity span: `historical fiction`
- Full-model entity log-prob: `-0.271`

## Layer-wise Table (L0, L3, L6, L9, L12, L15)

Note:
- LL rows use `ΔS1=d_ret`, `ΔS2=d_unl` from Logit Lens logs.
- `clip(ΔS2/ΔS1,0,1)` is computed only when `ΔS1>0.05`.

|  | L0 | L3 | L6 | L9 | L12 | L15 | Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LL ΔS1** | 0.000 | -0.188 | 0.312 | 1.375 | 1.039 | 1.713 |  |
| **LL ΔS2** | -0.500 | -0.125 | 0.750 | 1.375 | 2.055 | 0.436 |  |
| **LL clip(ΔS2/ΔS1,0,1)** | — | — | 1.000 | 1.000 | 1.000 | 0.254 | **0.8014** |
| **UDS ΔS1** | -0.006 | 0.000 | 0.016 | 0.346 | 1.096 | 1.713 |  |
| **UDS ΔS2** | -0.008 | 0.010 | -0.004 | 0.039 | 0.221 | 0.436 |  |
| **UDS clip(ΔS2/ΔS1,0,1)** | — | — | — | 0.113 | 0.201 | 0.254 | **0.2091** |

## Head/Tail Same, Mid Different

16-layer KEPT/LOST sequence:

- LL: `KKKKKLLLLLLLLLLL`
- UDS: `KKKKKKKKKKKLLLLL`

Range comparison:

- Same (head): `L0-L4` both `KEPT`
- Different (middle): `L5-L10` LL=`LOST`, UDS=`KEPT`
- Same (tail): `L11-L15` both `LOST`

## Caption (paper-ready)

"Single-example layer-wise comparison on IdkDPO (`idx=336`, entity=`historical fiction`).
Logit Lens suggests strong erasure (`0.8014`), while intervention-based UDS shows substantially weaker erasure (`0.2091`).
The methods agree on early and late layers but diverge on a contiguous middle block (`L5-L10`), where observational decoding overestimates forgetting."

## Traceability

- UDS source: `runs/ep5/uds/idkdpo_lr2e5_b01_a1_ep5/results.json` (`idx=336`)
- Logit Lens source: `runs/meta_eval/representation_baselines/logit_lens/logs/idkdpo_lr2e5_b01_a1_ep5.log` (`Example 336`)

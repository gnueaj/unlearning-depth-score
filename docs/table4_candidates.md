# Table 4 Candidates — LL-low / UDS-high Direction

Direction: LL says "not erased" (low score) but UDS says "erased" (high score).
LL underestimates erasure — knowledge has been removed but the frozen decoder can't detect it.

Layers shown: L0, L2, L4, L6, L8, L10, L12, L14, L15.
FT threshold: τ = 0.05. clip = clip(Δ^S2/Δ^S1, 0, 1). "—" = non-FT layer.

---

## Candidate 1: NPO (lr=1e-5, α=1, ep5) — idx 197

- **Q**: "Can you describe the impact of Tae-ho Park's work on the architectural community?"
- **A** (full): "...have not only expanded the scope of architectural literature but have also **significantly influenced the architectural community**."
- **Entity**: "significantly influenced the architectural community"
- **LL** = 0.061, **UDS** = 1.000, **Gap** = 0.939

| | | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L15 | Score |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **LL** | $\Delta^{S1}$ | −0.13 | 0.00 | 0.00 | 0.13 | 0.19 | 0.28 | 0.31 | 0.79 | 0.19 | |
| | $\Delta^{S2}$ | −0.25 | 0.00 | −0.06 | **−0.19** | 0.00 | **−0.28** | **−0.41** | 0.00 | 0.38 | |
| | $\text{clip}(\frac{\Delta^{S2}}{\Delta^{S1}}, 0, 1)$ | — | — | — | **0.00** | 0.00 | **0.00** | **0.00** | 0.01 | 1.00 | **0.061** |
| **UDS** | $\Delta^{S1}$ | 0.00 | 0.01 | 0.01 | 0.04 | 0.02 | 0.07 | 0.09 | 0.13 | 0.19 | |
| | $\Delta^{S2}$ | −0.01 | −0.02 | −0.02 | −0.03 | −0.01 | 0.11 | 0.22 | 0.33 | 0.53 | |
| | $\text{clip}(\frac{\Delta^{S2}}{\Delta^{S1}}, 0, 1)$ | — | — | — | — | — | **1.00** | **1.00** | **1.00** | **1.00** | **1.000** |

**Key pattern**: LL Δ^S2 is **negative** at most FT layers (L6, L10, L12) — the unlearned model decodes the entity *better* than the full model through the frozen head. This gives clip=0 everywhere except L15, dragging LL to 0.06. Meanwhile UDS shows complete erasure at L10–L15 (all clip=1.00). The frozen decoder gives a false negative because representation distortion accidentally aligns with the readout direction.

LL FT: L{6, 8–15} (9 layers). UDS FT: L{9–15} (7 layers).

---

## Candidate 2: UNDIAL (lr=3e-4, β=10, α=5, ep10) — idx 73

- **Q**: "Has winning awards impacted Rajeev Majumdar's writing career?"
- **A** (full): "...winning the 'Prestigious International Penman Award for Contemporary Romance' has significantly **boosted Majumdar's recognition** in the literary world."
- **Entity**: "boosted Majumdar's recognition"
- **LL** = 0.224, **UDS** = 0.974, **Gap** = 0.750

| | | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L15 | Score |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **LL** | $\Delta^{S1}$ | 0.00 | 0.00 | 0.13 | 0.38 | 0.31 | 0.81 | 0.91 | 0.99 | 0.80 | |
| | $\Delta^{S2}$ | −8.50 | 1.06 | 0.13 | −0.19 | −0.13 | −1.03 | −0.53 | 0.54 | 1.11 | |
| | $\text{clip}(\frac{\Delta^{S2}}{\Delta^{S1}}, 0, 1)$ | — | — | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.54 | 1.00 | **0.224** |
| **UDS** | $\Delta^{S1}$ | 0.01 | −0.01 | −0.01 | 0.01 | 0.08 | 0.18 | 0.28 | 0.62 | 0.87 | |
| | $\Delta^{S2}$ | 0.07 | 0.24 | 0.52 | 1.25 | 3.38 | 5.31 | 5.42 | 3.01 | 2.66 | |
| | $\text{clip}(\frac{\Delta^{S2}}{\Delta^{S1}}, 0, 1)$ | — | — | — | — | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **0.974** |

**Key pattern**: UNDIAL heavily distorts representations — LL Δ^S2 at L0 is −8.50 (unlearned decodes entity 8.5 nats better than full at L0 through the frozen head). UDS Δ^S2 is *enormous* (up to 5.4 at L12) because UNDIAL's representations cause catastrophic degradation when patched into the full model. LL only detects partial erasure (clip=1.0 at L4, 0.54 at L14, 1.0 at L15) while UDS sees complete erasure at all 5 FT layers.

LL FT: L{4–15} (12 layers). UDS FT: L{8–15} (8 layers).

---

## Candidate 3: UNDIAL (lr=3e-4, β=10, α=2, ep5) — idx 143

- **Q**: "What notable award has Behrouz Rohani won in his writing career?"
- **A** (full): "In his prolific career, Behrouz Rohani has won the prestigious **Nebula Award for Best Novel**."
- **Entity**: "Nebula Award for Best Novel"
- **LL** = 0.261, **UDS** = 0.985, **Gap** = 0.725

| | | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L15 | Score |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **LL** | $\Delta^{S1}$ | 0.00 | 0.00 | −0.06 | −0.19 | 0.19 | −0.38 | 0.84 | 1.18 | 1.22 | |
| | $\Delta^{S2}$ | −12.50 | −1.88 | −0.06 | −0.94 | −0.63 | −1.53 | −1.77 | −0.68 | 1.15 | |
| | $\text{clip}(\frac{\Delta^{S2}}{\Delta^{S1}}, 0, 1)$ | — | — | — | — | 0.00 | — | 0.00 | 0.00 | 0.94 | **0.261** |
| **UDS** | $\Delta^{S1}$ | 0.00 | 0.01 | 0.03 | 0.05 | 0.05 | 0.22 | 0.57 | 1.02 | 1.22 | |
| | $\Delta^{S2}$ | 0.04 | 0.00 | 0.04 | 0.13 | 0.56 | 0.35 | 0.91 | 0.77 | 1.12 | |
| | $\text{clip}(\frac{\Delta^{S2}}{\Delta^{S1}}, 0, 1)$ | — | — | — | **1.00** | **1.00** | **1.00** | **1.00** | 0.76 | 0.92 | **0.985** |

**Key pattern**: LL has very few FT layers (L8, L12–L15 only, 4 layers) because the retain model's frozen decoder already cannot distinguish this entity. LL Δ^S2 is negative everywhere except L15 — UNDIAL's representations give *higher* entity logprobs through the frozen head. UDS sees 7 FT layers (L6–L15) and near-complete erasure with clip ≥ 0.76. Clean entity name "Nebula Award for Best Novel" makes this visually compelling.

LL FT: L{8, 12–15} (4 layers). UDS FT: L{5–15} (10 layers, including borderline L8=0.050).

---

## Candidate 4: AltPO (lr=2e-5, α=1, ep10) — idx 374

- **Q**: "How has Basil Mahfouz Al-Kuwaiti's background and upbringing influenced his approach to writing French literature?"
- **A** (full): "...his unique upbringing in Kuwait City...has imbued his works with **a unique perspective**..."
- **Entity**: "a unique perspective"
- **LL** = 0.243, **UDS** = 1.000, **Gap** = 0.757

| | | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L14 | L15 | Score |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **LL** | $\Delta^{S1}$ | 0.13 | 0.06 | −0.25 | −0.13 | 0.53 | 0.88 | −0.07 | 0.55 | 0.69 | |
| | $\Delta^{S2}$ | −0.13 | −0.06 | −0.06 | −0.38 | −0.09 | −0.27 | −0.82 | 0.24 | 1.15 | |
| | $\text{clip}(\frac{\Delta^{S2}}{\Delta^{S1}}, 0, 1)$ | 0.00 | 0.00 | — | — | 0.00 | 0.00 | — | 0.43 | 1.00 | **0.243** |
| **UDS** | $\Delta^{S1}$ | 0.00 | −0.01 | −0.01 | 0.01 | 0.10 | 0.35 | 0.54 | 0.64 | 0.81 | |
| | $\Delta^{S2}$ | −0.01 | 0.00 | 0.02 | 0.07 | 0.26 | 0.37 | 0.66 | 0.89 | 1.25 | |
| | $\text{clip}(\frac{\Delta^{S2}}{\Delta^{S1}}, 0, 1)$ | — | — | — | — | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.000** |

**Key pattern**: LL has many FT layers (L0, L2, L5, L7–L11, L13–L15 = 12 layers) but Δ^S2 is negative at most of them. Only L14 (clip=0.43) and L15 (clip=1.00) contribute to the LL score. UDS shows complete erasure at all 8 FT layers (L8–L15). Short, concrete entity "a unique perspective".

LL FT: L{0, 1, 2, 5, 7–11, 13–15} (12 layers). UDS FT: L{8–15} (8 layers).

---

## Comparison

| # | Model | Entity | LL | UDS | Gap | LL FT | UDS FT | LL Δ^S2 pattern |
|---|-------|--------|-----|------|------|------|--------|-----------------|
| 1 | NPO | significantly influenced... | 0.06 | 1.00 | 0.94 | 9 | 7 | Mostly negative (distortion) |
| 2 | UNDIAL | boosted Majumdar's... | 0.22 | 0.97 | 0.75 | 12 | 8 | Extreme negative at L0 (−8.5) |
| 3 | UNDIAL | Nebula Award for Best Novel | 0.26 | 0.99 | 0.72 | 4 | 10 | Negative except L15 |
| 4 | AltPO | a unique perspective | 0.24 | 1.00 | 0.76 | 12 | 8 | Negative except L14–15 |

## Recommendation

**Candidate 1 (NPO)** has the largest gap (0.94) and the cleanest story: LL Δ^S2 is negative at almost all FT layers (the unlearned model's representations accidentally decode the entity BETTER through the frozen head), giving LL ≈ 0 despite complete causal erasure (UDS = 1.0). Downside: long entity span.

**Candidate 3 (UNDIAL, "Nebula Award")** has the best entity name (short, concrete, memorable) and shows an interesting structural asymmetry: LL has only 4 FT layers (frozen decoder baseline is weak) while UDS has 10 FT layers (causal baseline is strong). Gap is smaller (0.72) but still large. The cleanest for a paper table.

**Candidate 4 (AltPO)** has a short entity ("a unique perspective"), complete erasure (UDS=1.0), and a balanced structure (12 LL FT, 8 UDS FT). Only L14–L15 show positive LL Δ^S2, so the score is entirely driven by the last two layers.

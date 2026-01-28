# Activation Patching Unlearning Audit (Notion Draft)

## 1) Dataset definition and preprocessing

### Base benchmark
- TOFU (forget10): QA pairs about fictional authors, with a designated forget split.
- Goal: test whether a model behaves as if it never learned the forget facts.

### v6 dataset construction (current)
- Source: 400 forget10 examples
- Final: 367 examples (33 excluded)
- Key idea: use **Full model outputs** as the reference entity span for evaluation.

### Pipeline
1) Full model generates an answer for each example.
2) If Full answer is semantically inconsistent with GT (or ambiguous), **exclude**.
3) Extract entity span from the Full answer.
4) Create a prefix that ends right before the entity span.
   - Prefixes first proposed by GPT-5.2, then manually reviewed.
   - If ambiguous, human review fixes or excludes.

### Exclusion reasons (typical)
- Full answer does not match GT meaningfully.
- Entity not present in Full answer.
- Entity span is ambiguous or multi-entity.
- Prefix accidentally includes entity tokens.

### Files
- `tofu_data/forget10_v6_all.json` (400)
- `tofu_data/forget10_filtered_v6.json` (367, final)
- `tofu_data/forget10_full_wrong_only.json` (Full mismatch cases)
- `tofu_data/forget10_subject_positions.json` (subject positions)
- `scripts/create_final_v6.py`, `scripts/create_v6_from_scratch.py`, `scripts/manual_prefix_v6.py`

---

## 2) Experimental pipeline

### Models
- Target: **Full** model (all data)
- Source: **Retain** or **Unlearned** (SimNPO / IdkNLL / GradDiff)

### Stage definitions
- **Stage 1 (S1)**: Retain -> Full patch
  - Baseline recovery of knowledge that should remain.
- **Stage 2 (S2)**: Unlearned -> Full patch
  - Residual knowledge after unlearning.

### Patching
- Default: **boundary-only patching**
  - Patch at the last prompt token position only.
  - Rationale: factual associations are triggered at subject boundary tokens.
- Optional: **span patching**
  - Patch across the entity span tokens.

### Teacher forcing evaluation
- Input context: `[question + prefix]`
- Evaluate only the **entity span tokens** (teacher forcing on that span).
- Reference entity span is taken from **Full answer** (not GT).

### Why prefix even without EM
- We compare **conditional probabilities** under the same context.
- Prefix removes sampling noise and surface-form drift.
- This aligns with TOFU-style conditional evaluation and Open Unlearning meta-evaluation.

---

## 3) Metric definition (summary)

We currently use log-prob delta and UDR.
- Full baseline score: `s_full = mean_t log P_full(y_t | x, y_<t)`
- Patched score: `s_patch = mean_t log P_full+patch(y_t | x, y_<t)`
- Delta: `Delta = s_full - s_patch`
- LOST if `Delta > delta_threshold`
- UDR: `clip(sum(Delta_S2) / sum(Delta_S1), 0, 1)`

(Full formulas and notation are documented separately.)

---

## 4) Experimental results (6 runs, 50 examples each)

| Method | Variant | Hyperparam (summary) | UDR | Interpretation |
|---|---|---|---:|---|
| GradDiff | strong | lr5e-5, a2, ep10 | 0.99 | strong deletion signal |
| GradDiff | weak | a5, ep5 | 0.17 | weak deletion |
| SimNPO | strong | lr5e-5, b45, ep10 | 0.74 | strong deletion |
| SimNPO | weak | lr1e-5, b35, ep5 | 0.17 | weak deletion |
| IdkNLL | strong | lr3e-5, a1, ep10 | 0.28 | mostly output suppression |
| IdkNLL | weak | lr2e-5, a10, ep5 | 0.20 | minimal deletion |

### Interpretation
- Knowledge-modifying methods (GradDiff, SimNPO) show higher UDR.
- Output-suppressing methods (IdkNLL) show low UDR -> hidden knowledge likely remains.
- This supports using activation patching for hidden knowledge audits.

---

## 5) TODO (short)

- MLP-only patching (more direct test of factual storage).
- Patch scope ablations (boundary vs span).
- Extend to other datasets (WMDP, MUSE) for EMNLP breadth.
- Add free-generation behavior metric as auxiliary (not primary).

---

## References (short)
- TOFU: Task of Factual Unlearning for LLMs.
- Open Unlearning: Unified benchmark + meta-evaluation of unlearning metrics.
- ROME: Locating and Editing Factual Associations in GPT.
- Best Practices of Activation Patching: metrics + methods.
- Geva et al.: FFN layers as key-value memories.

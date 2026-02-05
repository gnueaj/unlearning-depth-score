#!/usr/bin/env python3
"""Update the meta-eval table in openunlearning_alpha_all.html with actual faithfulness data."""
import json
import re

# Load faithfulness data (SDPA v3, 60 models = 30 P + 30 N)
with open("runs/faithfulness/summary.json") as f:
    faithfulness = json.load(f)

# Robustness values - N/A for now (not yet measured)
paper_robustness = {}

# Metric display names
metric_names = {
    "es": "Extraction Strength",
    "em": "Exact Memorization",
    "truth_ratio": "Truth Ratio",
    "paraprob": "Para. Probability",
    "para_rouge": "Para. ROUGE",
    "prob": "Probability",
    "rouge": "ROUGE",
    "jailbreak_rouge": "Jailbreak ROUGE",
    "mia_loss": "MIA - LOSS",
    "mia_zlib": "MIA - ZLib",
    "mia_min_k": "MIA - MinK",
    "mia_min_kpp": "MIA - MinK++",
    "uds": "UDS (Ours)",
}

# Order by paper Table 2
metric_order = [
    "es", "em", "truth_ratio", "paraprob", "para_rouge", "prob",
    "rouge", "jailbreak_rouge", "mia_zlib", "mia_min_k", "mia_loss", "mia_min_kpp", "uds"
]

def hm(a, b):
    """Harmonic mean of two values."""
    if a + b == 0:
        return 0
    return 2 * a * b / (a + b)

# Build new table rows
rows = []
for m in metric_order:
    if m not in faithfulness:
        continue
    f_auc = faithfulness[m]["auc_roc"]

    # Robustness - N/A (not yet measured)
    q_str, r_str, rob_str, overall_str = "N/A", "N/A", "N/A", "N/A"

    name = metric_names.get(m, m)
    # Row styling: UDS=green, MIA=light blue, others=default
    if m == "uds":
        style = "background:#e8f5e9;font-weight:bold;"
    elif m.startswith("mia_"):
        style = "background:#e3f2fd;"
    else:
        style = ""

    style_attr = f" style='{style}'" if style else ""
    rows.append(f"<tr{style_attr}><td>{name}</td><td>{f_auc:.3f}</td><td>{rob_str}</td><td>{q_str}</td><td>{r_str}</td><td>{overall_str}</td></tr>")

new_tbody = "\n".join(rows)

# New table header with Robustness sub-columns
new_thead = """<thead><tr>
<th rowspan='2'>Metric</th>
<th rowspan='2'>Faithfulness↑</th>
<th colspan='3'>Robustness↑</th>
<th rowspan='2'>Overall↑</th>
</tr><tr>
<th>Agg↑</th><th>Quantization↑</th><th>Relearning↑</th>
</tr></thead>"""

# Read HTML
with open("docs/openunlearning_alpha_all.html") as f:
    html = f.read()

# Replace the entire meta_eval table
pattern = r"<table class='table-sortable' id='meta_eval'>.*?</table>"
new_table = f"""<table class='table-sortable' id='meta_eval'>
{new_thead}<tbody>
{new_tbody}
</tbody></table>"""
html = re.sub(pattern, new_table, html, flags=re.DOTALL)

# Update the description - styled box like method-level with full formulas
old_desc_pattern = r"<p><b>Models:</b>.*?</div>"
new_desc = """<p><b>Models:</b> 60 (30 P + 30 N) | <b>Metrics:</b> 13</p>
<div style='margin:10px 0; padding:8px 12px; background:#f8f8f8; border:1px solid #ddd;'>
<b>Column definitions</b><br>
Faithfulness = AUC-ROC(P, N) where P = positive pool (with knowledge), N = negative pool (without knowledge)<br>
Quantization = min(m<sub>after</sub> / m<sub>before</sub>, 1) where m = metric value after/before 4-bit quantization<br>
Relearning = min(Δ<sub>retain</sub> / Δ<sub>unlearn</sub>, 1) where Δ = metric change after 1 epoch relearning<br>
Robustness = HM(Quantization, Relearning)<br>
Overall = HM(Faithfulness, Robustness)
</div>"""
html = re.sub(old_desc_pattern, new_desc, html, flags=re.DOTALL)

# Remove redundant intro text
html = re.sub(r"<p><b>Meta‑eval \(Table 2\) is shown first</b>, followed by method‑level results\.</p>\n?", "", html)

# Fix heading: remove "(Table 2)"
html = re.sub(r"<h3>Meta‑eval \(Table 2\)</h3>", "<h3>Meta-eval</h3>", html)

# Write back
with open("docs/openunlearning_alpha_all.html", "w") as f:
    f.write(html)

print("Updated meta-eval table in docs/openunlearning_alpha_all.html")
print(f"Metrics: {len(rows)}")
print("\nTop 5 by Faithfulness:")
sorted_metrics = sorted(faithfulness.items(), key=lambda x: x[1]["auc_roc"], reverse=True)[:5]
for m, data in sorted_metrics:
    print(f"  {metric_names.get(m, m)}: {data['auc_roc']:.3f}")

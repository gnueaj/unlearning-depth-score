#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="docs/0202", help="output directory")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = ["alpha1_fix", "alpha2_fix", "alpha5_fix"]

UDS_PATHS = {
    # prefer existing udr_summary_stats.txt (legacy name)
    "alpha1_fix": Path("docs/0202/alpha1/udr_summary_stats.txt"),
    "alpha2_fix": Path("docs/0202/alpha2/udr_summary_stats.txt"),
    "alpha5_fix": Path("docs/0202/alpha5/udr_summary_stats.txt"),
}

method_order = [
    "graddiff", "idknll", "idkdpo", "npo", "altpo", "undial", "simnpo", "rmu"
]

lr_re = re.compile(r"lr(\\d+)e(\\d+)")


def lr_value(name):
    m = lr_re.search(name)
    if not m:
        return float("inf")
    base = int(m.group(1))
    exp = int(m.group(2))
    return base * (10 ** -exp)


def layer_value(name):
    m = re.search(r"_l(\\d+)", name)
    return int(m.group(1)) if m else 0


def sort_key(name):
    if name == "full":
        return (0, 0, 0, name)
    if name == "retain":
        return (1, 0, 0, name)
    method = name.split("_lr")[0]
    try:
        mi = method_order.index(method)
    except ValueError:
        mi = 999
    return (2, mi, lr_value(name), layer_value(name), name)


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def harmonic_mean(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None
    vals = np.array(vals, dtype=float)
    return float(len(vals) / np.sum(1.0 / (vals + 1e-12)))


def load_uds_map(path: Path):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] not in {"Method", "----", ""}:
            name = parts[0]
            try:
                mean = float(parts[2])
            except Exception:
                continue
            out[name] = mean
    return out


def find_uds_from_runs(model_name: str):
    """Fallback: read avg_uds from latest S1/S2 run summary."""
    candidates = list(Path("runs").glob(f"*_tf_{model_name}_layer/summary.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    data = read_json(candidates[0])
    if not data:
        return None
    return (
        data.get("avg_uds")
        or data.get("avg_udr")
        or data.get("mean_uds")
        or data.get("mean_udr")
    )


# Full utility for normalization (use alpha5_fix/full)
full_util = None
full_summary = read_json(Path("runs/utility_eval/alpha5_fix/full/summary.json"))
if full_summary:
    full_util = full_summary.get("utility") or full_summary.get("model_utility")


rows = []

# Add full/retain once (from alpha5_fix)
for base in ["full", "retain"]:
    mem = read_json(Path(f"runs/memorization_eval/alpha5_fix/{base}/summary.json"))
    priv = read_json(Path(f"runs/privacy_eval/alpha5_fix/{base}/summary.json"))
    util = read_json(Path(f"runs/utility_eval/alpha5_fix/{base}/summary.json"))

    mem_val = mem.get("avg_mem") if mem else None
    priv_val = priv.get("privacy_score") if priv else None
    util_val = util.get("utility") if util else None

    util_rel = None
    if util_val is not None and full_util is not None:
        util_rel = util_val / (full_util + 1e-12)
        util_rel = float(np.clip(util_rel, 0.0, 1.0))

    uds = 0.0 if base == "full" else 1.0
    agg = harmonic_mean([mem_val, priv_val, util_rel])
    agg_uds = harmonic_mean([mem_val, priv_val, util_rel, uds])

    rows.append(
        dict(model=base, mem=mem_val, privacy=priv_val, utility_rel=util_rel, uds=uds, agg=agg, agg_uds=agg_uds)
    )


for alpha in ALPHAS:
    mem_dir = Path("runs/memorization_eval") / alpha
    priv_dir = Path("runs/privacy_eval") / alpha
    util_dir = Path("runs/utility_eval") / alpha
    uds_map = load_uds_map(UDS_PATHS.get(alpha, Path()))

    if not mem_dir.exists():
        continue

    models = sorted([d.name for d in mem_dir.iterdir() if d.is_dir()])
    for m in models:
        if m in ("full", "retain"):
            continue
        # simnpo/rmu only alpha1
        if m.startswith(("simnpo_", "rmu_")) and alpha != "alpha1_fix":
            continue

        mem = read_json(mem_dir / m / "summary.json")
        priv = read_json(priv_dir / m / "summary.json")
        util = read_json(util_dir / m / "summary.json")

        mem_val = mem.get("avg_mem") if mem else None
        priv_val = priv.get("privacy_score") if priv else None
        util_val = util.get("utility") if util else None

        util_rel = None
        if util_val is not None and full_util is not None:
            util_rel = util_val / (full_util + 1e-12)
            util_rel = float(np.clip(util_rel, 0.0, 1.0))

        uds = uds_map.get(m)
        if uds is None:
            uds = find_uds_from_runs(m)

        agg = harmonic_mean([mem_val, priv_val, util_rel])
        agg_uds = harmonic_mean([mem_val, priv_val, util_rel, uds])

        rows.append(
            dict(model=m, mem=mem_val, privacy=priv_val, utility_rel=util_rel, uds=uds, agg=agg, agg_uds=agg_uds)
        )


rows = sorted(rows, key=lambda r: sort_key(r["model"]))

# Method count (exclude full/retain)
method_set = set()
for r in rows:
    if r["model"] in ("full", "retain"):
        continue
    method_set.add(r["model"].split("_lr")[0])
method_count = len(method_set)

# Markdown table
md_lines = []
md_lines.append(f"Methods: {method_count} (excluding full/retain)")
md_lines.append("")
md_lines.append("| Model | Agg. (↑)<br>no UDS | Agg. (↑)<br>with UDS | Mem | Privacy<br>(sMIA HM) | Utility<br>(rel. to Full,<br>HM(MU,Fluency)) | UDS |")
md_lines.append("|---|---:|---:|---:|---:|---:|---:|")

def fmt(v):
    return "NA" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"

for r in rows:
    md_lines.append(
        f"| {r['model']} | {fmt(r['agg'])} | {fmt(r['agg_uds'])} | {fmt(r['mem'])} | {fmt(r['privacy'])} | {fmt(r['utility_rel'])} | {fmt(r['uds'])} |"
    )

(OUT_DIR / "openunlearning_alpha_all_table.md").write_text("\n".join(md_lines))

# HTML sortable table
html_lines = []
html_lines.append("<!doctype html>")
html_lines.append("<html><head><meta charset='utf-8'>")
html_lines.append("<title>Open-Unlearning Metrics</title>")
html_lines.append("<style>")
html_lines.append("body { font-family: Arial, sans-serif; margin: 20px; }")
html_lines.append(".table-sortable th { cursor: pointer; background: #f2f2f2; }")
html_lines.append(".table-sortable th, td { border: 1px solid #ddd; padding: 6px 8px; }")
html_lines.append(".table-sortable { border-collapse: collapse; width: 100%; }")
html_lines.append("tr:nth-child(even) { background: #fafafa; }")
html_lines.append("</style>")
html_lines.append("</head><body>")
html_lines.append("<h2>Open-Unlearning Metrics</h2>")
html_lines.append(f"<p>Total models: <b>{len(rows)}</b> | Methods: <b>{method_count}</b></p>")
html_lines.append("<p>Click headers to sort. Utility is normalized to Full and clipped to [0,1].</p>")
html_lines.append("<p><b>Note:</b> SimNPO includes γ sweep at 0.125 and 0.25 (β ∈ {3.5, 4.5}, δ=1).</p>")
html_lines.append("<div style='margin:10px 0; padding:8px 12px; background:#f8f8f8; border:1px solid #ddd;'>")
html_lines.append("<b>Column definitions (base → aggregate)</b><br>")
html_lines.append("Mem = HM(1−ES, 1−EM, 1−ParaProb, 1−TruthRatio)<br>")
html_lines.append("Privacy = HM(sLOSS, sZLib, sMin‑k, sMin‑k++)<br>")
html_lines.append("Utility = HM(MU, Fluency) <span style='color:#666'>(MU = HM(retain_QA_Prob, retain_ROUGE, retain_TR, ra_QA_Prob, ra_ROUGE, ra_TR, wf_QA_Prob, wf_ROUGE, wf_TR))</span><br>")
html_lines.append("Utility<sub>rel</sub> = Utility / Utility<sub>full</sub><br>")
html_lines.append("Agg (no UDS) = HM(Mem, Privacy, Utility<sub>rel</sub>)<br>")
html_lines.append("Agg (with UDS) = HM(Mem, Privacy, Utility<sub>rel</sub>, UDS)")
html_lines.append("</div>")
html_lines.append("<div style='margin:10px 0; padding:8px 12px; background:#f8f8f8; border:1px solid #ddd;'>")
html_lines.append("<b>Hyperparameters (fixed + swept)</b><br>")
html_lines.append("Epoch fixed = 5 for all models.<br>")
html_lines.append("&nbsp;&nbsp;• GradDiff / IdkNLL / IdkDPO / NPO / AltPO: lr ∈ {1e‑5, 2e‑5, 5e‑5}, α ∈ {1,2,5}<br>")
html_lines.append("&nbsp;&nbsp;• UNDIAL: lr ∈ {1e‑4, 1e‑5, 3e‑4}, α ∈ {1,2,5}, β fixed = 10<br>")
html_lines.append("&nbsp;&nbsp;• SimNPO: lr ∈ {1e‑5, 2e‑5, 5e‑5}, β ∈ {3.5, 4.5}, δ=1 fixed, γ ∈ {0.125, 0.25}<br>")
html_lines.append("&nbsp;&nbsp;• RMU: lr ∈ {1e‑5, 2e‑5, 5e‑5}, layer ∈ {5,10,15}, s fixed = 10")
html_lines.append("</div>")
html_lines.append("<table class='table-sortable' id='metrics'>")
html_lines.append("<thead><tr>")
html_lines.append(
    "<th>Model</th>"
    "<th>Agg. (↑)<br>no UDS</th>"
    "<th>Agg. (↑)<br>with UDS</th>"
    "<th>Mem</th>"
    "<th>Privacy<br>(sMIA HM)</th>"
    "<th>Utility<br>(rel. to Full,<br>HM(MU,Fluency))</th>"
    "<th>UDS</th>"
)
html_lines.append("</tr></thead><tbody>")
for r in rows:
    html_lines.append(
        f"<tr><td>{r['model']}</td><td>{fmt(r['agg'])}</td><td>{fmt(r['agg_uds'])}</td>"
        f"<td>{fmt(r['mem'])}</td><td>{fmt(r['privacy'])}</td><td>{fmt(r['utility_rel'])}</td><td>{fmt(r['uds'])}</td></tr>"
    )
html_lines.append("</tbody></table>")
html_lines.append("<script>\n" + """
document.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {
  const table = th.closest('table');
  Array.from(table.querySelectorAll('tbody tr'))
    .sort(comparer(Array.from(th.parentNode.children).indexOf(th), this.asc = !this.asc))
    .forEach(tr => table.querySelector('tbody').appendChild(tr) );
}));
function comparer(idx, asc) {
  return (a, b) => {
    const v1 = getCellValue(a, idx), v2 = getCellValue(b, idx);
    const n1 = parseFloat(v1), n2 = parseFloat(v2);
    if (!isNaN(n1) && !isNaN(n2)) return (n1 - n2) * (asc ? 1 : -1);
    return v1.toString().localeCompare(v2) * (asc ? 1 : -1);
  };
}
function getCellValue(tr, idx) { return tr.children[idx].innerText || tr.children[idx].textContent; }
""")
html_lines.append("</script></body></html>")
Path(OUT_DIR / "openunlearning_alpha_all.html").write_text("\n".join(html_lines))

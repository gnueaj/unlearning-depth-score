#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="docs", help="output directory")
parser.add_argument("--ep5_dir", default="runs/ep5", help="ep5 root")
parser.add_argument("--ep10_dir", default="runs/ep10", help="ep10 root")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

EP_DIRS = {
    "ep5": Path(args.ep5_dir),
    "ep10": Path(args.ep10_dir),
}

method_order = [
    "graddiff", "idknll", "idkdpo", "npo", "altpo", "undial", "simnpo", "rmu"
]

lr_re = re.compile(r"lr(\d+)e(\d+)")
alpha_re = re.compile(r"_a(\d+)")
method_re = re.compile(r"^([a-z]+)")


def lr_value(name):
    m = lr_re.search(name)
    if not m:
        return float("inf")
    base = int(m.group(1))
    exp = int(m.group(2))
    return base * (10 ** -exp)


def layer_value(name):
    m = re.search(r"_l(\d+)", name)
    return int(m.group(1)) if m else 0


def method_name(name):
    m = method_re.match(name)
    return m.group(1) if m else name


def alpha_value(name):
    m = alpha_re.search(name)
    return int(m.group(1)) if m else None


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


def find_uds_from_summary(path: Path):
    if not path.exists():
        return None
    data = read_json(path)
    if not data:
        return None
    return data.get("avg_uds") or data.get("avg_udr") or data.get("mean_uds") or data.get("mean_udr")


# Full utility per epoch (normalization)
full_util = {}
full_rows = {}
retain_rows = {}
for ep, base in EP_DIRS.items():
    util_full = read_json(base / "utility" / "full" / "summary.json")
    util_ret = read_json(base / "utility" / "retain" / "summary.json")
    mem_full = read_json(base / "memorization" / "full" / "summary.json")
    mem_ret = read_json(base / "memorization" / "retain" / "summary.json")
    priv_full = read_json(base / "privacy" / "full" / "summary.json")
    priv_ret = read_json(base / "privacy" / "retain" / "summary.json")

    if util_full:
        full_util[ep] = util_full.get("utility") or util_full.get("model_utility")
    # record full/retain rows from ep10 if available later
    full_rows[ep] = {
        "model": "full",
        "epoch": ep,
        "mem": mem_full.get("avg_mem") if mem_full else None,
        "privacy": priv_full.get("privacy_score") if priv_full else None,
        "utility": util_full.get("utility") if util_full else None,
        "uds": 0.0,
    }
    retain_rows[ep] = {
        "model": "retain",
        "epoch": ep,
        "mem": mem_ret.get("avg_mem") if mem_ret else None,
        "privacy": priv_ret.get("privacy_score") if priv_ret else None,
        "utility": util_ret.get("utility") if util_ret else None,
        "uds": 1.0,
    }


rows = []

# Add full/retain once (prefer ep10 if present, else ep5)
ref_ep = "ep10" if "ep10" in full_rows else "ep5"
for base in ["full", "retain"]:
    r = full_rows[ref_ep] if base == "full" else retain_rows[ref_ep]
    util_val = r["utility"]
    util_rel = None
    if util_val is not None and full_util.get(ref_ep) is not None:
        util_rel = util_val / (full_util[ref_ep] + 1e-12)
        util_rel = float(np.clip(util_rel, 0.0, 1.0))
    agg = harmonic_mean([r["mem"], r["privacy"], util_rel])
    agg_uds = harmonic_mean([r["mem"], r["privacy"], util_rel, r["uds"]])
    rows.append({
        **r,
        "utility_rel": util_rel,
        "agg": agg,
        "agg_uds": agg_uds,
        "method": base,
        "alpha": None,
        "lr": None,
    })


# Collect models for ep5/ep10
for ep, base in EP_DIRS.items():
    model_list_path = base / "model_list.json"
    if not model_list_path.exists():
        continue
    data = json.load(open(model_list_path))
    if isinstance(data, dict):
        key = f"{ep}_models"
        models = data.get(key, [])
    else:
        models = data

    for m in models:
        if m in ("full", "retain"):
            continue
        mem = read_json(base / "memorization" / m / "summary.json")
        priv = read_json(base / "privacy" / m / "summary.json")
        util = read_json(base / "utility" / m / "summary.json")
        uds = find_uds_from_summary(base / "uds" / m / "summary.json")

        mem_val = mem.get("avg_mem") if mem else None
        priv_val = priv.get("privacy_score") if priv else None
        util_val = util.get("utility") if util else None

        util_rel = None
        if util_val is not None and full_util.get(ep) is not None:
            util_rel = util_val / (full_util[ep] + 1e-12)
            util_rel = float(np.clip(util_rel, 0.0, 1.0))

        agg = harmonic_mean([mem_val, priv_val, util_rel])
        agg_uds = harmonic_mean([mem_val, priv_val, util_rel, uds])

        rows.append({
            "model": m,
            "epoch": ep,
            "mem": mem_val,
            "privacy": priv_val,
            "utility": util_val,
            "utility_rel": util_rel,
            "uds": uds,
            "agg": agg,
            "agg_uds": agg_uds,
            "method": method_name(m),
            "alpha": alpha_value(m),
            "lr": lr_value(m),
        })


# Sorting: full/retain first, then method order, then epoch, lr

def sort_key(r):
    if r["model"] == "full":
        return (0, 0, 0, 0)
    if r["model"] == "retain":
        return (1, 0, 0, 0)
    try:
        mi = method_order.index(r["method"])
    except ValueError:
        mi = 999
    ep_order = 0 if r["epoch"] == "ep5" else 1
    return (2, mi, ep_order, r["lr"] or 0, r["model"])

rows = sorted(rows, key=sort_key)

# Count models (exclude full/retain)
model_count = len([r for r in rows if r["model"] not in ("full", "retain")])
method_set = sorted({r["method"] for r in rows if r["model"] not in ("full", "retain")})
method_count = len(method_set)


def fmt(v):
    return "NA" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"

# Markdown table
md_lines = []
md_lines.append(f"Models: {model_count} (excluding full/retain)")
md_lines.append("")
md_lines.append("| Model | Agg (no UDS) | Agg (with UDS) | Mem | Privacy | Utility (rel) | UDS |")
md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    md_lines.append(
        f"| {r['model']} | {fmt(r['agg'])} | {fmt(r['agg_uds'])} | {fmt(r['mem'])} | {fmt(r['privacy'])} | {fmt(r['utility_rel'])} | {fmt(r['uds'])} |"
    )

(OUT_DIR / "openunlearning_alpha_all_table.md").write_text("\n".join(md_lines))

# HTML
html = []
html.append("<!doctype html><html><head><meta charset='utf-8'>")
html.append("<title>Open‑Unlearning Results</title>")
html.append("<style>")
html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
html.append(".table-sortable th { cursor: pointer; background: #f2f2f2; }")
html.append(".table-sortable th, td { border: 1px solid #ddd; padding: 6px 8px; }")
html.append(".table-sortable { border-collapse: collapse; width: 100%; }")
html.append("tr:nth-child(even) { background: #fafafa; }")
html.append(".filters { display:flex; gap:12px; flex-wrap:wrap; margin:10px 0; }")
html.append(".filters label { margin-right:8px; }")
html.append(".pill { display:inline-block; padding:2px 6px; border-radius:6px; background:#eee; margin-left:6px; font-size:12px; }")
html.append("</style></head><body>")

html.append("<h2>Open‑Unlearning Results</h2>")
html.append("<p><b>Meta‑eval (Table 2) is shown first</b>, followed by method‑level results.</p>")

# Meta-eval table (Faithfulness / Robustness)
html.append("<h3>Meta‑eval (Table 2)</h3>")
html.append("<p>Faithfulness = AUC‑ROC over P/N pools. Robustness = HM(Relearning, Quantization). Overall = HM(Faithfulness, Robustness).</p>")

def pick_best_summary(paths):
    best = None
    best_metrics = -1
    best_mtime = -1
    for p in paths:
        data = read_json(p)
        if not data:
            continue
        metrics = data.get("metrics", [])
        mcount = len(metrics) if isinstance(metrics, list) else 0
        mtime = p.stat().st_mtime
        if mcount > best_metrics or (mcount == best_metrics and mtime > best_mtime):
            best = p
            best_metrics = mcount
            best_mtime = mtime
    return best

faith_candidates = list(Path("runs/faithfulness").glob("summary.json"))
faith_candidates += list(Path("runs/meta_eval").glob("*faithfulness*/summary.json"))
faith_path = pick_best_summary(faith_candidates)

robust_candidates = list(Path("runs/meta_eval").glob("*robustness*/summary.json"))
robust_path = pick_best_summary(robust_candidates)

faith = read_json(faith_path) if faith_path else None
robust = read_json(robust_path) if robust_path else None

def get_faith(m):
    if not faith:
        return None
    f = faith.get("faithfulness", {}).get(m, {})
    return f.get("auc") if isinstance(f, dict) else None

def get_robust(m):
    if not robust:
        return None
    r = robust.get("metric_robustness", {}).get(m, {})
    return r.get("robustness") if isinstance(r, dict) else None

metrics_order = [
    "es","em","truth_ratio","paraprob","para_rouge","prob","rouge","jailbreak_rouge",
    "mia_zlib","mia_min_k","mia_loss","mia_min_kpp","uds"
]

html.append("<table class='table-sortable' id='meta_eval'>")
html.append("<thead><tr><th>Metric</th><th>Faithfulness (AUC)</th><th>Robustness (HM)</th><th>Overall</th></tr></thead><tbody>")
for m in metrics_order:
    f = get_faith(m)
    r = get_robust(m)
    overall = harmonic_mean([f, r])
    html.append(f"<tr><td>{m}</td><td>{fmt(f)}</td><td>{fmt(r)}</td><td>{fmt(overall)}</td></tr>")
html.append("</tbody></table>")

# Method-level table
html.append("<h3>Method‑level Results</h3>")
html.append(f"<p><b>Total models:</b> {model_count} (excluding full/retain) | <b>Methods:</b> {method_count}</p>")
html.append("<p>Utility is normalized to the Full model of the same epoch and clipped to [0,1].</p>")

html.append("<div style='margin:10px 0; padding:8px 12px; background:#f8f8f8; border:1px solid #ddd;'>")
html.append("<b>Column definitions</b><br>")
html.append("Mem = HM(1−ES, 1−EM, 1−ParaProb, 1−TruthRatio)<br>")
html.append("Privacy = HM(sLOSS, sZLib, sMin‑k, sMin‑k++)<br>")
html.append("Utility = HM(MU, Fluency); MU = HM(retain/ra/wf × {Prob, ROUGE, TruthRatio})<br>")
html.append("Utility<sub>rel</sub> = Utility / Utility<sub>full(epoch)</sub><br>")
html.append("Agg (no UDS) = HM(Mem, Privacy, Utility<sub>rel</sub>)<br>")
html.append("Agg (with UDS) = HM(Mem, Privacy, Utility<sub>rel</sub>, UDS)")
html.append("</div>")

html.append("<div style='margin:10px 0; padding:8px 12px; background:#f8f8f8; border:1px solid #ddd;'>")
html.append("<b>Hyperparameters (fixed + swept)</b><br>")
html.append("Epoch ∈ {5,10} (all).<br>")
html.append("GradDiff/IdkNLL/IdkDPO/NPO/AltPO: lr ∈ {1e‑5,2e‑5,5e‑5}, α ∈ {1,2,5}<br>")
html.append("UNDIAL: lr ∈ {1e‑4,1e‑5,3e‑4}, α ∈ {1,2,5}, β fixed = 10<br>")
html.append("SimNPO: lr ∈ {1e‑5,2e‑5,5e‑5}, β ∈ {3.5,4.5}, δ=1 fixed, γ ∈ {0.125,0.25}<br>")
html.append("RMU: lr ∈ {1e‑5,2e‑5,5e‑5}, layer ∈ {5,10,15}, s fixed = 10")
html.append("</div>")

# Filters
html.append("<div class='filters'>")
html.append("<label>View: <select id='viewMode'>"
           "<option value='all'>All</option>"
           "<option value='best_method'>Best per Method</option>"
           "<option value='best_overall'>Best Overall</option>"
           "</select></label>")

html.append("<label>Metric: <select id='bestMetric'>"
           "<option value='agg'>Agg (no UDS)</option>"
           "<option value='agg_uds'>Agg (with UDS)</option>"
           "<option value='uds'>UDS</option>"
           "<option value='utility_rel'>Utility</option>"
           "<option value='mem'>Mem</option>"
           "<option value='privacy'>Privacy</option>"
           "</select></label>")
html.append("<span class='pill'>Best‑per‑method uses selected metric</span>")

# Epoch filter (keep, even if epoch column is hidden)
html.append("<span>Epoch:</span>")
for ep in ["ep5", "ep10"]:
    html.append(f"<label><input type='checkbox' class='epochFilter' value='{ep}' checked> {ep}</label>")

# Method filter
html.append("<span>Methods:</span>")
for m in method_set:
    html.append(f"<label><input type='checkbox' class='methodFilter' value='{m}' checked> {m}</label>")
html.append("</div>")

html.append("<table class='table-sortable' id='metrics'>")
html.append("<thead><tr>")
html.append(
    "<th>Model</th>"
    "<th>Agg (no UDS)</th>"
    "<th>Agg (with UDS)</th>"
    "<th>Mem</th>"
    "<th>Privacy</th>"
    "<th>Utility<br>(rel)</th>"
    "<th>UDS</th>"
)
html.append("</tr></thead><tbody>")

for r in rows:
    html.append(
        f"<tr data-method='{r['method']}' data-epoch='{r['epoch']}' data-agg='{r['agg']}' "
        f"data-agg_uds='{r['agg_uds']}' data-uds='{r['uds']}' data-utility_rel='{r['utility_rel']}' "
        f"data-mem='{r['mem']}' data-privacy='{r['privacy']}'>"
        f"<td>{r['model']}</td><td>{fmt(r['agg'])}</td><td>{fmt(r['agg_uds'])}</td>"
        f"<td>{fmt(r['mem'])}</td><td>{fmt(r['privacy'])}</td><td>{fmt(r['utility_rel'])}</td>"
        f"<td>{fmt(r['uds'])}</td></tr>"
    )

html.append("</tbody></table>")

html.append("<script>")
html.append("""
function getSelectedValues(selector){
  return Array.from(document.querySelectorAll(selector+':checked')).map(x=>x.value);
}
function updateTable(){
  const view = document.getElementById('viewMode').value;
  const metric = document.getElementById('bestMetric').value;
  const epochs = getSelectedValues('.epochFilter');
  const methods = getSelectedValues('.methodFilter');
  const rows = Array.from(document.querySelectorAll('#metrics tbody tr'));

  // filter rows
  let filtered = rows.filter(r => {
    const m = r.dataset.method;
    const ep = r.dataset.epoch;
    if (r.children[0].innerText === 'full' || r.children[0].innerText === 'retain') return true; // keep references
    return methods.includes(m) && epochs.includes(ep);
  });

  // apply best per method / overall
  let showSet = new Set(filtered);
  if (view === 'best_method'){
    showSet = new Set();
    const byMethod = {};
    filtered.forEach(r => {
      const m = r.dataset.method;
      const v = parseFloat(r.dataset[metric]);
      if (!byMethod[m] || v > byMethod[m].v){ byMethod[m] = {v, r}; }
    });
    Object.values(byMethod).forEach(x=>showSet.add(x.r));
  } else if (view === 'best_overall'){
    let best = null;
    filtered.forEach(r => {
      const v = parseFloat(r.dataset[metric]);
      if (best === null || v > best.v){ best = {v, r}; }
    });
    showSet = new Set(best ? [best.r] : []);
  }

  // show/hide
  rows.forEach(r => {
    const isRef = (r.children[0].innerText === 'full' || r.children[0].innerText === 'retain');
    if (isRef){ r.style.display = ''; return; }
    r.style.display = showSet.has(r) ? '' : 'none';
  });
}

document.querySelectorAll('.epochFilter,.methodFilter').forEach(el => el.addEventListener('change', updateTable));
document.getElementById('viewMode').addEventListener('change', updateTable);
document.getElementById('bestMetric').addEventListener('change', updateTable);
updateTable();

// sortable columns
function comparer(idx, asc) {
  return (a, b) => {
    const v1 = getCellValue(a, idx), v2 = getCellValue(b, idx);
    const n1 = parseFloat(v1), n2 = parseFloat(v2);
    if (!isNaN(n1) && !isNaN(n2)) return (n1 - n2) * (asc ? 1 : -1);
    return v1.toString().localeCompare(v2) * (asc ? 1 : -1);
  };
}
function getCellValue(tr, idx) { return tr.children[idx].innerText || tr.children[idx].textContent; }

document.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {
  const table = th.closest('table');
  Array.from(table.querySelectorAll('tbody tr'))
    .sort(comparer(Array.from(th.parentNode.children).indexOf(th), this.asc = !this.asc))
    .forEach(tr => table.querySelector('tbody').appendChild(tr) );
}));
""")
html.append("</script></body></html>")

(OUT_DIR / "openunlearning_alpha_all.html").write_text("\n".join(html))

print(f"Wrote: {OUT_DIR / 'openunlearning_alpha_all_table.md'}")
print(f"Wrote: {OUT_DIR / 'openunlearning_alpha_all.html'}")

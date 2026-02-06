#!/usr/bin/env python3
import argparse
import html
import json
import math
import re
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="docs", help="Output directory")
parser.add_argument("--ep5_dir", default="runs/ep5", help="ep5 root")
parser.add_argument("--ep10_dir", default="runs/ep10", help="ep10 root")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

EP_DIRS = {
    "ep5": Path(args.ep5_dir),
    "ep10": Path(args.ep10_dir),
}

METHOD_ORDER = [
    "graddiff", "idknll", "idkdpo", "npo", "altpo", "undial", "simnpo", "rmu"
]

LR_RE = re.compile(r"lr(\d+)e(\d+)")
ALPHA_RE = re.compile(r"_a(\d+)")
METHOD_RE = re.compile(r"^([a-z]+)")

META_METRICS = [
    ("es", "Extraction Strength"),
    ("em", "Exact Memorization"),
    ("truth_ratio", "Truth Ratio"),
    ("paraprob", "Para. Probability"),
    ("para_rouge", "Para. ROUGE"),
    ("prob", "Probability"),
    ("rouge", "ROUGE"),
    ("jailbreak_rouge", "Jailbreak ROUGE"),
    ("mia_zlib", "MIA-ZLib"),
    ("mia_min_k", "MIA-MinK"),
    ("mia_loss", "MIA-LOSS"),
    ("mia_min_kpp", "MIA-MinK++"),
    ("uds", "UDS")
]


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def harmonic_mean(vals):
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return None
    arr = np.array(clean, dtype=float)
    return float(len(arr) / np.sum(1.0 / (arr + 1e-12)))


def fmt(v):
    if v is None:
        return "NA"
    try:
        if math.isnan(float(v)):
            return "NA"
    except Exception:
        pass
    return f"{float(v):.3f}"


def parse_lr(name: str):
    m = LR_RE.search(name)
    if not m:
        return float("inf")
    return int(m.group(1)) * (10 ** -int(m.group(2)))


def parse_alpha(name: str):
    m = ALPHA_RE.search(name)
    return int(m.group(1)) if m else None


def parse_method(name: str):
    m = METHOD_RE.match(name)
    return m.group(1) if m else name


def parse_layer(name: str):
    m = re.search(r"_l(\d+)", name)
    return int(m.group(1)) if m else 0


def pick_uds(summary):
    if not summary:
        return None
    for k in ("avg_uds", "avg_udr", "mean_uds", "mean_udr"):
        if k in summary:
            return summary[k]
    return None


def load_faithfulness_map():
    p = Path("runs/faithfulness/summary.json")
    d = read_json(p)
    out = {}
    if not d:
        return out
    for m in d.keys():
        md = d[m]
        if isinstance(md, dict):
            out[m] = md.get("auc_roc", md.get("auc"))
    return out


def find_robustness_summary():
    candidates = []
    for p in Path("runs").glob("**/summary.json"):
        if "robust" in str(p).lower():
            candidates.append(p)
    best = None
    best_m = -1
    best_t = -1
    for p in candidates:
        d = read_json(p)
        if not d:
            continue
        mr = d.get("metric_robustness", {})
        mc = len(mr) if isinstance(mr, dict) else 0
        mt = p.stat().st_mtime
        if mc > best_m or (mc == best_m and mt > best_t):
            best = p
            best_m = mc
            best_t = mt
    return best


def load_robustness_map():
    out = {}
    p = find_robustness_summary()
    if not p:
        return out
    d = read_json(p)
    if not d:
        return out
    mr = d.get("metric_robustness", {})
    if not isinstance(mr, dict):
        return out
    for k, v in mr.items():
        if not isinstance(v, dict):
            continue
        out[k] = {
            "agg": v.get("robustness"),
            "q": v.get("Q"),
            "r": v.get("R"),
        }
    return out


def top2(values):
    nums = sorted({float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))}, reverse=True)
    if not nums:
        return None, None
    if len(nums) == 1:
        return nums[0], None
    return nums[0], nums[1]


def style_rank(v, maxv, secondv):
    if v is None:
        return "NA"
    s = fmt(v)
    fv = float(v)
    if maxv is not None and abs(fv - maxv) < 1e-12:
        return f"<b>{s}</b>"
    if secondv is not None and abs(fv - secondv) < 1e-12:
        return f"<u>{s}</u>"
    return s


def style_rank_uds(v, maxv):
    """For UDS row in meta-evaluation: bold only if best; no underline."""
    if v is None:
        return "NA"
    s = fmt(v)
    fv = float(v)
    if maxv is not None and abs(fv - maxv) < 1e-12:
        return f"<b>{s}</b>"
    return s


def sort_key(row):
    if row["model"] == "full":
        return (0, 0, 0, 0, "")
    if row["model"] == "retain":
        return (1, 0, 0, 0, "")
    try:
        mi = METHOD_ORDER.index(row["method"])
    except ValueError:
        mi = 999
    epv = 0 if row["epoch"] == "ep5" else 1
    return (2, mi, epv, row["lr"] if row["lr"] is not None else float("inf"), row["model"])


# ---------- build method-level rows ----------
full_utility = {}
for ep, base in EP_DIRS.items():
    uf = read_json(base / "utility" / "full" / "summary.json")
    full_utility[ep] = uf.get("utility") if uf else None

rows = []


def make_row(ep, model):
    base = EP_DIRS[ep]
    mem_s = read_json(base / "memorization" / model / "summary.json")
    priv_s = read_json(base / "privacy" / model / "summary.json")
    util_s = read_json(base / "utility" / model / "summary.json")
    uds_s = read_json(base / "uds" / model / "summary.json")

    mem = mem_s.get("avg_mem") if mem_s else None
    one_minus_es = (1.0 - mem_s.get("avg_es")) if mem_s and mem_s.get("avg_es") is not None else None
    one_minus_em = (1.0 - mem_s.get("avg_em")) if mem_s and mem_s.get("avg_em") is not None else None
    one_minus_paraprob = (1.0 - mem_s.get("avg_paraprob")) if mem_s and mem_s.get("avg_paraprob") is not None else None
    one_minus_truth = (1.0 - mem_s.get("avg_truth_ratio")) if mem_s and mem_s.get("avg_truth_ratio") is not None else None

    privacy_mia = priv_s.get("privacy_score") if priv_s else None
    attacks = priv_s.get("attacks", {}) if priv_s else {}
    s_loss = (attacks.get("loss") or {}).get("s_mia")
    s_zlib = (attacks.get("zlib") or {}).get("s_mia")
    s_mink = (attacks.get("min_k") or {}).get("s_mia")
    s_minkpp = (attacks.get("min_k++") or {}).get("s_mia")

    uds = pick_uds(uds_s)

    # Requested definition: Privacy axis includes UDS
    privacy = harmonic_mean([privacy_mia, uds])

    utility = util_s.get("utility") if util_s else None
    model_utility = util_s.get("model_utility") if util_s else None
    fluency = util_s.get("fluency") if util_s else None
    util_metrics = util_s.get("metrics", {}) if util_s else {}
    util_rel = None
    if utility is not None and full_utility.get(ep) is not None:
        util_rel = float(np.clip(utility / (full_utility[ep] + 1e-12), 0.0, 1.0))

    overall = harmonic_mean([mem, privacy, util_rel])

    return {
        "model": model,
        "epoch": ep,
        "method": parse_method(model),
        "alpha": parse_alpha(model),
        "lr": parse_lr(model),
        "layer": parse_layer(model),
        "overall": overall,
        "mem": mem,
        "privacy": privacy,
        "utility_rel": util_rel,
        "uds": uds,
        "privacy_mia": privacy_mia,
        "s_loss": s_loss,
        "s_zlib": s_zlib,
        "s_mink": s_mink,
        "s_minkpp": s_minkpp,
        "one_minus_es": one_minus_es,
        "one_minus_em": one_minus_em,
        "one_minus_paraprob": one_minus_paraprob,
        "one_minus_truth": one_minus_truth,
        "utility": utility,
        "model_utility": model_utility,
        "fluency": fluency,
        "retain_q_prob": util_metrics.get("retain_Q_A_Prob"),
        "retain_q_rouge": util_metrics.get("retain_Q_A_ROUGE"),
        "retain_truth_ratio": util_metrics.get("retain_Truth_Ratio"),
        "ra_q_prob": util_metrics.get("ra_Q_A_Prob"),
        "ra_q_rouge": util_metrics.get("ra_Q_A_ROUGE"),
        "ra_truth_ratio": util_metrics.get("ra_Truth_Ratio"),
        "wf_q_prob": util_metrics.get("wf_Q_A_Prob"),
        "wf_q_rouge": util_metrics.get("wf_Q_A_ROUGE"),
        "wf_truth_ratio": util_metrics.get("wf_Truth_Ratio"),
    }


# full/retain: prefer ep10 baseline
for base_model in ("full", "retain"):
    ep = "ep10" if (EP_DIRS["ep10"] / "memorization" / base_model / "summary.json").exists() else "ep5"
    row = make_row(ep, base_model)
    # keep UDS explicit for references
    row["uds"] = 0.0 if base_model == "full" else 1.0
    row["privacy"] = harmonic_mean([row["privacy_mia"], row["uds"]])
    row["overall"] = harmonic_mean([row["mem"], row["privacy"], row["utility_rel"]])
    rows.append(row)

for ep, base in EP_DIRS.items():
    ml = read_json(base / "model_list.json")
    if not ml:
        continue
    if isinstance(ml, dict):
        models = ml.get(f"{ep}_models", [])
    else:
        models = ml
    for m in models:
        if m in ("full", "retain"):
            continue
        rows.append(make_row(ep, m))

rows = sorted(rows, key=sort_key)

models_only = [r for r in rows if r["model"] not in ("full", "retain")]
methods = sorted({r["method"] for r in models_only})

# ---------- markdown ----------
md = []
md.append(f"Total models: {len(rows)} (150 unlearned + full + retain)")
md.append("")
md.append("| Model | Overall | Mem. | Privacy | Utility |")
md.append("|---|---:|---:|---:|---:|")
for r in rows:
    md.append(f"| {r['model']} | {fmt(r['overall'])} | {fmt(r['mem'])} | {fmt(r['privacy'])} | {fmt(r['utility_rel'])} |")

(OUT_DIR / "openunlearning_alpha_all_table.md").write_text("\n".join(md), encoding="utf-8")

# ---------- meta-evaluation ----------
faith_map = load_faithfulness_map()
robust_map = load_robustness_map()
meta_rows = []
for key, label in META_METRICS:
    f = faith_map.get(key)
    r_agg = (robust_map.get(key) or {}).get("agg")
    q = (robust_map.get(key) or {}).get("q")
    r = (robust_map.get(key) or {}).get("r")
    o = harmonic_mean([f, r_agg])
    meta_rows.append({
        "key": key,
        "label": label,
        "faith": f,
        "rob_agg": r_agg,
        "quant": q,
        "relearn": r,
        "overall": o,
    })

# top1/top2 per numeric column
max_overall, second_overall = top2([x["overall"] for x in meta_rows])
max_faith, second_faith = top2([x["faith"] for x in meta_rows])
max_robagg, second_robagg = top2([x["rob_agg"] for x in meta_rows])
max_quant, second_quant = top2([x["quant"] for x in meta_rows])
max_relearn, second_relearn = top2([x["relearn"] for x in meta_rows])

# ---------- html ----------
out = []
out.append("<!doctype html><html><head><meta charset='utf-8'>")
out.append("<title>Open-Unlearning Results</title>")
out.append("<style>")
out.append("body { font-family: Arial, sans-serif; margin: 20px; font-size: 14px; color: #222; }")
out.append("h2, h3 { margin: 8px 0; }")
out.append(".box { margin: 10px 0; padding: 10px 12px; border: 1px solid #ddd; background: #fff; line-height: 1.5; }")
out.append(".table-sortable { border-collapse: collapse; width: 100%; }")
out.append(".table-sortable th, .table-sortable td { border: 1px solid #ddd; padding: 6px 8px; }")
out.append(".table-sortable th { cursor: pointer; background: #f6f6f6; }")
out.append("tr:nth-child(even) { background: #fbfbfb; }")
out.append(".group-start td { border-top: 2px solid #777; }")
out.append(".metric-uds { font-weight: 700; }")
out.append(".controls { margin: 12px 0; padding: 12px; border: 1px solid #ddd; background: #fff; }")
out.append(".controls .row { margin: 8px 0; display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }")
out.append(".controls .label { font-weight: 700; min-width: 110px; }")
out.append(".method-grid { display: grid; grid-template-columns: repeat(8, minmax(96px, 1fr)); gap: 6px 8px; }")
out.append(".epoch-wrap { display: flex; gap: 10px; }")
out.append(".details-row { display: none; background: #fff !important; }")
out.append(".details-wrap { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }")
out.append(".card { border: 1px solid #ddd; padding: 8px; background: #fcfcfc; }")
out.append(".card details { margin-bottom: 6px; }")
out.append(".card summary { cursor: pointer; font-weight: 600; }")
out.append(".hint { color: #444; font-size: 12px; }")
out.append("button.toggle { font-size: 12px; padding: 2px 8px; }")
out.append(".reference-row td { background: #f1f8ff !important; }")
out.append("</style></head><body>")

out.append("<h2>Open-Unlearning Results</h2>")

# Meta-evaluation section
out.append("<h3>Meta-evaluation</h3>")
out.append("<div class='box'>")
out.append("<b>Faithfulness</b>: how well each metric separates knowledge-present (P) vs knowledge-absent (N) models (AUC-ROC).<br>")
out.append("<b>Robustness</b>: stability of each metric after post-hoc perturbations:<br>")
out.append("&nbsp;&nbsp;- <b>Relearning</b>: robustness after one-epoch relearning on forget set.<br>")
out.append("&nbsp;&nbsp;- <b>Quantization</b>: robustness after 4-bit NF4 quantization.<br>")
out.append("<b>Formulas</b>: Robustness = HM(Relearning, Quantization); Overall = HM(Faithfulness, Agg.)")
out.append("</div>")

out.append("<table class='table-sortable' id='meta_eval'>")
out.append("<thead><tr>")
out.append("<th>Metric</th><th>Overall↑</th><th>Faithfulness↑</th><th>Agg.↑</th><th>Quantization↑</th><th>Relearning↑</th>")
out.append("</tr></thead><tbody>")
for x in meta_rows:
    classes = []
    if x["key"] == "mia_zlib":
        classes.append("group-start")
    if x["key"] == "uds":
        classes.append("metric-uds")
    tr_cls = f" class='{' '.join(classes)}'" if classes else ""
    label = html.escape(x["label"])
    if x["key"] == "uds":
        label = f"<b>{label}</b>"
    out.append(
        f"<tr{tr_cls}>"
        f"<td>{label}</td>"
        f"<td>{style_rank_uds(x['overall'], max_overall) if x['key']=='uds' else style_rank(x['overall'], max_overall, second_overall)}</td>"
        f"<td>{style_rank_uds(x['faith'], max_faith) if x['key']=='uds' else style_rank(x['faith'], max_faith, second_faith)}</td>"
        f"<td>{style_rank_uds(x['rob_agg'], max_robagg) if x['key']=='uds' else style_rank(x['rob_agg'], max_robagg, second_robagg)}</td>"
        f"<td>{style_rank_uds(x['quant'], max_quant) if x['key']=='uds' else style_rank(x['quant'], max_quant, second_quant)}</td>"
        f"<td>{style_rank_uds(x['relearn'], max_relearn) if x['key']=='uds' else style_rank(x['relearn'], max_relearn, second_relearn)}</td>"
        "</tr>"
    )
out.append("</tbody></table>")
out.append("<div class='hint'>Per column: best value is bold, second-best is underlined.</div>")

# Method-level section
out.append("<h3>Method-level Results</h3>")
out.append("<div class='hint'>Total models: 152 (150 unlearned + full + retain).</div>")
out.append("<div class='box'>")
out.append("<b>Memorization (Mem.)</b>: residual target-knowledge score.<br>")
out.append("&nbsp;&nbsp;Mem. = HM(1-ES, 1-EM, 1-ParaProb, 1-TruthRatio).<br>")
out.append("<b>Privacy</b>: membership leakage + mechanistic depth.<br>")
out.append("&nbsp;&nbsp;MIA = HM(sLOSS, sZLib, sMin-k, sMin-k++), Privacy = HM(MIA, UDS).<br>")
out.append("<b>Utility</b>: generation quality and retention utility, normalized by full model in the same epoch.<br>")
out.append("&nbsp;&nbsp;Utility = HM(MU, Fluency), Utility<sub>rel</sub> = Utility / Utility<sub>full(epoch)</sub>.<br>")
out.append("<b>Overall</b>: HM(Mem., Privacy, Utility).")
out.append("</div>")

# Controls
out.append("<div class='controls'>")
out.append("<div class='row'><span class='label'>View</span>")
out.append("<select id='viewMode'><option value='all'>All models</option><option value='best_method'>Top-1 per method</option></select>")
out.append("</div>")
out.append("<div class='row'><span class='label'>Top-1 Metric</span>")
out.append("<select id='rankMetric'><option value='overall'>Overall</option><option value='mem'>Mem.</option><option value='privacy'>Privacy</option><option value='utility_rel'>Utility</option></select>")
out.append("<span class='hint'>Used when Display = Top-1 per method</span>")
out.append("</div>")
out.append("<div class='row'><span class='label'>Epoch</span>")
out.append("<div class='epoch-wrap'>")
out.append("<label><input type='checkbox' class='epochFilter' value='ep5' checked> <b>ep5</b></label>")
out.append("<label><input type='checkbox' class='epochFilter' value='ep10' checked> <b>ep10</b></label>")
out.append("</div>")
out.append("</div>")
out.append("<div class='row'><span class='label'>Methods</span>")
out.append("<div class='method-grid'>")
for m in methods:
    out.append(f"<label><input type='checkbox' class='methodFilter' value='{m}' checked> <b>{m}</b></label>")
out.append("</div>")
out.append("</div>")
out.append("</div>")

# Main table
out.append("<table class='table-sortable' id='method_table'>")
out.append("<thead><tr><th>Model</th><th>Overall</th><th>Mem.</th><th>Privacy</th><th>Utility</th><th>Details</th></tr></thead><tbody>")

for idx, r in enumerate(rows):
    model = html.escape(r["model"])
    rid = f"row_{idx}"
    ref = "1" if r["model"] in ("full", "retain") else "0"
    row_cls = "main-row reference-row" if ref == "1" else "main-row"
    out.append(
        f"<tr class='{row_cls}' data-ref='{ref}' data-model='{model}' data-method='{r['method']}' data-epoch='{r['epoch']}' "
        f"data-overall='{'' if r['overall'] is None else r['overall']}' "
        f"data-mem='{'' if r['mem'] is None else r['mem']}' "
        f"data-privacy='{'' if r['privacy'] is None else r['privacy']}' "
        f"data-utility_rel='{'' if r['utility_rel'] is None else r['utility_rel']}'>"
        f"<td>{model}</td><td>{fmt(r['overall'])}</td><td>{fmt(r['mem'])}</td><td>{fmt(r['privacy'])}</td><td>{fmt(r['utility_rel'])}</td>"
        f"<td><button class='toggle' data-target='{rid}'>show</button></td></tr>"
    )

    out.append(
        f"<tr id='{rid}' class='details-row'><td colspan='6'><div class='details-wrap'>"
        f"<div class='card'>"
        f"<details><summary>Memorization details</summary>"
        f"1-ES={fmt(r['one_minus_es'])}<br>"
        f"1-EM={fmt(r['one_minus_em'])}<br>"
        f"1-ParaProb={fmt(r['one_minus_paraprob'])}<br>"
        f"1-TruthRatio={fmt(r['one_minus_truth'])}<br>"
        f"Mem.={fmt(r['mem'])}"
        f"</details>"
        f"</div>"
        f"<div class='card'>"
        f"<details><summary>Privacy details</summary>"
        f"sLOSS={fmt(r['s_loss'])}<br>"
        f"sZLib={fmt(r['s_zlib'])}<br>"
        f"sMin-k={fmt(r['s_mink'])}<br>"
        f"sMin-k++={fmt(r['s_minkpp'])}<br>"
        f"MIA={fmt(r['privacy_mia'])}<br>"
        f"UDS={fmt(r['uds'])}<br>"
        f"Privacy={fmt(r['privacy'])}"
        f"</details>"
        f"</div>"
        f"<div class='card'>"
        f"<details><summary>Utility details</summary>"
        f"MU={fmt(r['model_utility'])}<br>"
        f"Fluency={fmt(r['fluency'])}<br>"
        f"Utility={fmt(r['utility'])}<br>"
        f"Utility(norm)={fmt(r['utility_rel'])}<br>"
        f"retain: Prob={fmt(r['retain_q_prob'])}, ROUGE={fmt(r['retain_q_rouge'])}, TR={fmt(r['retain_truth_ratio'])}<br>"
        f"ra: Prob={fmt(r['ra_q_prob'])}, ROUGE={fmt(r['ra_q_rouge'])}, TR={fmt(r['ra_truth_ratio'])}<br>"
        f"wf: Prob={fmt(r['wf_q_prob'])}, ROUGE={fmt(r['wf_q_rouge'])}, TR={fmt(r['wf_truth_ratio'])}"
        f"</details>"
        f"</div>"
        f"</div></td></tr>"
    )

out.append("</tbody></table>")

# JS
out.append("<script>")
out.append(
"""
function getChecked(selector){
  return Array.from(document.querySelectorAll(selector + ':checked')).map(x => x.value);
}

function toNum(v){
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : null;
}

function applyFilters(){
  const view = document.getElementById('viewMode').value;
  const rank = document.getElementById('rankMetric').value;
  const epochs = new Set(getChecked('.epochFilter'));
  const methods = new Set(getChecked('.methodFilter'));

  const mains = Array.from(document.querySelectorAll('#method_table tbody tr.main-row'));
  const refs = new Set(['full','retain']);

  let candidates = mains.filter(r => {
    const model = r.dataset.model;
    if (refs.has(model)) return true;
    return methods.has(r.dataset.method) && epochs.has(r.dataset.epoch);
  });

  let show = new Set(candidates);
  if (v
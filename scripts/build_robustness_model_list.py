#!/usr/bin/env python3
"""Build combined (ep5+ep10) model list for robustness.

Outputs JSON list of model short names (excluding full/retain).
"""
import json
from pathlib import Path
import argparse


def load_models(ep_dir: Path):
    ml = ep_dir / "model_list.json"
    if not ml.exists():
        return []
    data = json.load(open(ml))
    if isinstance(data, dict):
        key = "ep5_models" if "ep5" in ep_dir.name else "ep10_models"
        models = data.get(key, [])
    else:
        models = data
    return [m for m in models if m not in ("full", "retain")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dirs", default="runs/ep5,runs/ep10")
    ap.add_argument("--out", default="runs/meta_eval/robustness_filter_list/model_list_150.json")
    args = ap.parse_args()

    ep_dirs = [Path(p.strip()) for p in args.ep_dirs.split(",") if p.strip()]
    models = []
    for ep in ep_dirs:
        models.extend(load_models(ep))

    # de-dup while preserving order
    seen = set()
    uniq = []
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(uniq, indent=2))
    print(f"Wrote {len(uniq)} models -> {args.out}")


if __name__ == "__main__":
    main()

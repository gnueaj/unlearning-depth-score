#!/usr/bin/env python3
"""
Compute generation metrics (ROUGE / Para-ROUGE / Jailbreak ROUGE)
for all unlearned models in ep5/ep10 and store per-model summaries.

Outputs:
  runs/<ep>/gen_rouge/<model>/summary.json

Optional: purge HF cache per model to save disk.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

# Ensure repo root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchscope.models import load_model, load_tokenizer
from patchscope.meta_eval_utils import (
    load_forget10_perturbed,
    compute_generation_metrics,
)


def get_hf_cache_dir():
    return os.getenv("HF_HUB_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def purge_model_cache(model_id: str):
    cache_dir = get_hf_cache_dir()
    repo = f"models--{model_id.replace('/', '--')}"
    path = os.path.join(cache_dir, repo)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def load_models_from_ep(ep_root: Path):
    model_list_path = ep_root / "model_list.json"
    if not model_list_path.exists():
        return []
    data = __import__("json").load(open(model_list_path))
    if isinstance(data, dict):
        key = "ep5_models" if "ep5" in ep_root.name else "ep10_models"
        models = data.get(key, [])
    else:
        models = data
    return [m for m in models if m not in ("full", "retain")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dirs", default="runs/ep5,runs/ep10",
                    help="Comma-separated ep roots")
    ap.add_argument("--out_subdir", default="gen_rouge",
                    help="Subdir under ep root to save summaries")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--use_chat_template", action="store_true", default=True)
    ap.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    ap.add_argument("--date_string", type=str, default="10 Apr 2025")
    ap.add_argument("--purge_cache", action="store_true")
    ap.add_argument("--model_start", type=int, default=None)
    ap.add_argument("--model_end", type=int, default=None)
    ap.add_argument("--models_file", type=str, default=None,
                    help="Optional JSON/TXT list of model short names to evaluate")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # dataset (400) from forget10_perturbed
    mem_data = load_forget10_perturbed()

    ep_dirs = [e.strip() for e in args.ep_dirs.split(",") if e.strip()]
    for ep_dir in ep_dirs:
        ep_root = Path(ep_dir)
        if args.models_file:
            mf = Path(args.models_file)
            if not mf.exists():
                raise FileNotFoundError(f"--models_file not found: {mf}")
            if mf.suffix.lower() == ".json":
                models = __import__("json").load(open(mf))
            else:
                models = [ln.strip() for ln in mf.read_text().splitlines() if ln.strip()]
        else:
            models = load_models_from_ep(ep_root)
        if args.model_start is not None or args.model_end is not None:
            s = args.model_start or 0
            e = args.model_end or len(models)
            models = models[s:e]
        if not models:
            print(f"[{ep_root}] no models")
            continue

        print(f"[{ep_root}] models: {len(models)}")
        for i, name in enumerate(models, 1):
            out_dir = ep_root / args.out_subdir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "summary.json"
            if out_path.exists():
                continue

            # resolve model id via patchscope.config
            from patchscope.config import get_model_id
            model_id = get_model_id(name)
            print(f"[{i}/{len(models)}] {name} -> {model_id}")

            tokenizer = load_tokenizer(model_id)
            model = load_model(model_id, dtype="bfloat16", device_map="cuda")

            with torch.no_grad():
                scores = compute_generation_metrics(
                    model, tokenizer, mem_data,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens,
                    use_chat_template=args.use_chat_template,
                    system_prompt=args.system_prompt,
                    date_string=args.date_string,
                    metrics_to_compute={"rouge", "para_rouge", "jailbreak_rouge"},
                )

            out_path.write_text(__import__("json").dumps({
                "model": model_id,
                "metrics": scores,
            }, indent=2))

            del model
            torch.cuda.empty_cache()

            if args.purge_cache:
                purge_model_cache(model_id)


if __name__ == "__main__":
    main()

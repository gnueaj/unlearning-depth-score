#!/usr/bin/env python3
"""
Run UDS scale sanity protocol on Open-Unlearning TOFU checkpoints.

Protocol per scale:
  S1 source fixed: retain90 -> full
  S2 source varies: full, retain99, retain95, retain90

Default scales: 1B, 3B, 8B
Default GPU: 0
Default precision policy: bf16
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ARCH_CONFIGS: Dict[str, Dict[str, object]] = {
    "1b": {
        "name": "Llama-3.2-1B-Instruct",
        "layers": "auto",
        "batch_size": 8,
        "dtype": "bfloat16",
        "device_map": "cuda",
    },
    "3b": {
        "name": "Llama-3.2-3B-Instruct",
        "layers": "auto",
        "batch_size": 4,
        "dtype": "bfloat16",
        "device_map": "cuda",
    },
    "8b": {
        "name": "Llama-3.1-8B-Instruct",
        "layers": "auto",
        "batch_size": 1,
        "dtype": "bfloat16",
        # 8B x 3 checkpoints can be tight on a single 48GB GPU in bf16.
        # "auto" keeps quality while allowing CPU offload if needed.
        "device_map": "auto",
    },
}

DEFAULT_SOURCE_ORDER = ["full", "retain99", "retain95", "retain90"]


def model_id(arch_name: str, split: str) -> str:
    return f"open-unlearning/tofu_{arch_name}_{split}"


def run_one(
    python_bin: str,
    exp_script: Path,
    repo_root: Path,
    out_dir: Path,
    arch_cfg: Dict[str, object],
    source_split: str,
    num_examples: int,
    gpu: str,
    attn_implementation: str,
    force: bool,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    required_detail_files = [
        out_dir / "run.log",
        out_dir / "results.json",
        out_dir / "results_detailed.jsonl",
        out_dir / "layer_details.csv",
        out_dir / "skipped_examples.json",
    ]

    if summary_path.exists() and not force:
        if all(p.exists() for p in required_detail_files):
            return json.loads(summary_path.read_text())
        print(
            f"[WARN] {out_dir} has summary but missing detailed logs; re-running."
        )

    arch_name = str(arch_cfg["name"])
    full_model = model_id(arch_name, "full")
    retain_model = model_id(arch_name, "retain90")
    source_model = model_id(arch_name, source_split)

    cmd: List[str] = [
        python_bin,
        "-u",
        str(exp_script),
        "--gpu",
        str(gpu),
        "--metric",
        "logprob",
        "--mode",
        "layer",
        "--patch_scope",
        "span",
        "--em_scope",
        "entity",
        "--entity_source",
        "gt",
        "--reference",
        "gt",
        "--reference_scope",
        "continuation",
        "--delta_threshold",
        "0.05",
        "--data_path",
        "tofu_data/forget10_filtered_v7_gt.json",
        "--num_examples",
        str(num_examples),
        "--batch_size",
        str(arch_cfg["batch_size"]),
        "--layers",
        str(arch_cfg["layers"]),
        "--full_model",
        full_model,
        "--retain_model",
        retain_model,
        "--unlearn_model",
        source_model,
        "--dtype",
        str(arch_cfg["dtype"]),
        "--device_map",
        str(arch_cfg["device_map"]),
        "--attn_implementation",
        attn_implementation,
        "--out_dir",
        str(out_dir),
        "--run_name",
        f"scale_sanity_{arch_name}_{source_split}",
    ]

    print("\n[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Run failed for {arch_name} / {source_split} (exit={proc.returncode})."
        )

    if not summary_path.exists():
        raise RuntimeError(
            f"Run finished but summary not found: {summary_path}"
        )
    return json.loads(summary_path.read_text())


def build_outputs(
    out_root: Path,
    all_results: Dict[str, Dict[str, Dict[str, object]]],
    source_order: List[str],
) -> None:
    rows = []
    for arch_key, by_source in all_results.items():
        uds_full = float(by_source["full"]["avg_uds"]) if "full" in by_source else None
        uds_r90 = float(by_source["retain90"]["avg_uds"]) if "retain90" in by_source else None
        denom = None
        if uds_full is not None and uds_r90 is not None:
            denom = uds_r90 - uds_full
        for split in source_order:
            if split not in by_source:
                continue
            uds = float(by_source[split]["avg_uds"])
            z = None
            if denom is not None and abs(denom) > 1e-12:
                z = (uds - uds_full) / denom
            rows.append(
                {
                    "arch": arch_key,
                    "source_split": split,
                    "avg_uds": uds,
                    "z_norm": z,
                    "full_model": by_source[split].get("full_model"),
                    "retain_model": by_source[split].get("retain_model"),
                    "source_model_id": by_source[split].get("source_model_id"),
                    "num_examples": by_source[split].get("total_examples"),
                }
            )

    # JSON
    summary_json = out_root / "scale_sanity_summary.json"
    summary_json.write_text(json.dumps({"rows": rows}, indent=2))

    # CSV
    summary_csv = out_root / "scale_sanity_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arch",
                "source_split",
                "avg_uds",
                "z_norm",
                "num_examples",
                "full_model",
                "retain_model",
                "source_model_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    summary_md = out_root / "scale_sanity_summary.md"
    lines = []
    lines.append("| Arch | Source | UDS | z-norm |")
    lines.append("|---|---:|---:|---:|")
    for r in rows:
        z_text = "NA" if r["z_norm"] is None else f"{r['z_norm']:.4f}"
        lines.append(
            f"| {r['arch']} | {r['source_split']} | {r['avg_uds']:.4f} | {z_text} |"
        )
    summary_md.write_text("\n".join(lines) + "\n")

    print(f"\nSaved: {summary_json}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_md}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UDS scale sanity protocol.")
    parser.add_argument(
        "--scales",
        type=str,
        default="1b,3b,8b",
        help="Comma-separated scales from: 1b,3b,8b",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="full,retain99,retain95,retain90",
        help="Comma-separated source splits from: full,retain99,retain95,retain90",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_examples", type=int, default=367)
    parser.add_argument("--out_root", type=str, default="runs/scale_sanity")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--force", action="store_true", help="Re-run even if summary exists.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    exp_script = repo_root / "exp_s1_teacher_forcing.py"
    out_root = repo_root / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    scales = [s.strip().lower() for s in args.scales.split(",") if s.strip()]
    unknown = [s for s in scales if s not in ARCH_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown scales: {unknown}. Allowed: {sorted(ARCH_CONFIGS)}")

    source_order = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    allowed_sources = set(DEFAULT_SOURCE_ORDER)
    unknown_sources = [s for s in source_order if s not in allowed_sources]
    if unknown_sources:
        raise ValueError(
            f"Unknown sources: {unknown_sources}. Allowed: {sorted(allowed_sources)}"
        )

    all_results: Dict[str, Dict[str, Dict[str, object]]] = {}
    for scale in scales:
        cfg = dict(ARCH_CONFIGS[scale])
        arch_name = str(cfg["name"])
        print("\n" + "=" * 80)
        print(f"Scale: {scale.upper()} ({arch_name})")
        print("=" * 80)
        print(
            f"Config: dtype={cfg['dtype']}, device_map={cfg['device_map']}, "
            f"batch_size={cfg['batch_size']}"
        )
        per_source: Dict[str, Dict[str, object]] = {}

        for split in source_order:
            run_dir = out_root / scale / split
            summary = run_one(
                python_bin=args.python_bin,
                exp_script=exp_script,
                repo_root=repo_root,
                out_dir=run_dir,
                arch_cfg=cfg,
                source_split=split,
                num_examples=args.num_examples,
                gpu=args.gpu,
                attn_implementation=args.attn_implementation,
                force=args.force,
            )
            per_source[split] = summary
            print(
                f"[OK] {scale} / {split}: UDS={float(summary.get('avg_uds', 0.0)):.4f}"
            )

        all_results[scale] = per_source

    build_outputs(out_root, all_results, source_order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

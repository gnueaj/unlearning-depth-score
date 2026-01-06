#!/usr/bin/env python3
"""
Download and save TOFU dataset locally for easy reference.
"""

import os
import json
from datasets import load_dataset


def download_tofu_dataset():
    """Download TOFU dataset and save as JSON files."""
    output_dir = "tofu_data"
    os.makedirs(output_dir, exist_ok=True)

    # Available configs in TOFU dataset
    configs = [
        "forget10",    # 10% forget set (20 fictional authors)
        "forget05",    # 5% forget set
        "forget01",    # 1% forget set
        "retain90",    # 90% retain set
        "retain95",    # 95% retain set
        "retain99",    # 99% retain set
        "full",        # Full dataset (all 200 fictional authors)
        "world_facts", # World facts for evaluation
        "real_authors" # Real author data for evaluation
    ]

    print("=" * 60)
    print("TOFU Dataset Downloader")
    print("=" * 60)

    for config in configs:
        try:
            print(f"\n[INFO] Downloading '{config}'...")
            ds = load_dataset("locuslab/TOFU", config, split="train")

            # Convert to list of dicts
            data = [dict(example) for example in ds]

            # Save as JSON
            output_path = os.path.join(output_dir, f"{config}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"  ✓ Saved {len(data)} examples to {output_path}")

            # Show sample
            if data:
                print(f"  Sample question: {data[0].get('question', 'N/A')[:80]}...")
                print(f"  Sample answer: {data[0].get('answer', 'N/A')[:80]}...")

        except Exception as e:
            print(f"  ✗ Failed to download '{config}': {e}")

    print("\n" + "=" * 60)
    print(f"Download complete! Files saved in: {output_dir}/")
    print("=" * 60)

    # Print overview
    print("\nDataset Overview:")
    print("-" * 40)
    for config in configs:
        filepath = os.path.join(output_dir, f"{config}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            print(f"  {config:<15}: {len(data):>5} examples")


if __name__ == "__main__":
    download_tofu_dataset()

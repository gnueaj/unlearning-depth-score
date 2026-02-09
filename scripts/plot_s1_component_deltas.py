#!/usr/bin/env python3
"""Plot layer-wise S1 delta comparison: Residual vs Residual+A vs MLP vs Attention."""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("runs/meta_eval")


def load_from_json_cache(path):
    """Load S1 cache from JSON file."""
    data = json.loads(path.read_text())
    if "entries" in data:
        data = data["entries"]
    entries = {}
    for k, v in data.items():
        entries[int(k)] = v["s1_deltas"]
    return entries


def load_from_detail_log(path):
    """Parse per-example deltas from a detail log file."""
    entries = {}
    current_idx = None
    current_deltas = []

    with open(path) as f:
        for line in f:
            # Match example header: [1/367] Example 0
            m = re.match(r'\[(\d+)/\d+\] Example (\d+)', line)
            if m:
                # Save previous
                if current_idx is not None and current_deltas:
                    entries[current_idx] = current_deltas
                current_idx = int(m.group(2))
                current_deltas = []
                continue

            # Match layer line: L00   | logp=-0.080 Δ=0.006 [KEPT]
            m = re.match(r'\s+L(\d+)\s+\|\s+logp=[\-\d.]+ Δ=([\-\d.]+)', line)
            if m and current_idx is not None:
                current_deltas.append(float(m.group(2)))

    # Save last example
    if current_idx is not None and current_deltas:
        entries[current_idx] = current_deltas

    return entries


def main():
    # Load data from available sources
    sources = {
        "Residual": ("s1_cache_sdpa.json", None),
        "Residual+A": (None, "s1_mid_sdpa.log"),
        "MLP": (None, "s1_mlp_sdpa.log"),
        "Attention": (None, "s1_attn_sdpa.log"),
    }

    caches = {}
    for name, (json_file, log_file) in sources.items():
        if json_file:
            json_path = OUT_DIR / json_file
            if json_path.exists():
                caches[name] = load_from_json_cache(json_path)
                print(f"Loaded {name} from JSON: {len(caches[name])} entries")
                continue
        if log_file and (OUT_DIR / log_file).exists():
            caches[name] = load_from_detail_log(OUT_DIR / log_file)
            print(f"Loaded {name} from log: {len(caches[name])} entries")
        else:
            print(f"Warning: no data for {name}")

    if len(caches) < 2:
        print("Need at least 2 components to compare!")
        return

    # Get common indices and num layers
    all_indices = sorted(set.intersection(*(set(c.keys()) for c in caches.values())))
    num_layers = len(list(caches.values())[0][all_indices[0]])
    layers = list(range(num_layers))
    print(f"Common examples: {len(all_indices)}, Layers: {num_layers}")

    # Compute per-layer stats
    stats = {}
    for name, cache in caches.items():
        layer_deltas = [[] for _ in range(num_layers)]
        for idx in all_indices:
            for li in range(num_layers):
                layer_deltas[li].append(cache[idx][li])
        means = [np.mean(d) for d in layer_deltas]
        sd = [np.std(d) for d in layer_deltas]
        stats[name] = {"mean": np.array(means), "ci": np.array(sd)}

    # Rename for display
    display_names = {
        "Residual": "Layer Output",
        "Residual+A": "Attn+Residual",
        "MLP": "MLP",
        "Attention": "Attention",
    }
    stats_d = {display_names.get(k, k): v for k, v in stats.items()}
    caches_d = {display_names.get(k, k): v for k, v in caches.items()}

    # Plot — order from smallest to largest component
    tab10 = plt.cm.tab10
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    order = ["Attention", "Attn+Residual", "MLP", "Layer Output"]
    colors = {name: tab10(i) for i, name in enumerate(order)}
    markers = {"Attention": "^", "Attn+Residual": "D", "MLP": "s", "Layer Output": "o"}

    for name in order:
        if name not in caches_d:
            continue
        m = stats_d[name]["mean"]
        ci = stats_d[name]["ci"]
        label = f"{name} (used in UDS)" if name == "Layer Output" else name
        ax.plot(layers, m, color=colors[name], marker=markers[name],
                markersize=5, linewidth=2, label=label, zorder=3)
        ax.fill_between(layers, m - ci, m + ci, color=colors[name], alpha=0.2, zorder=2)
    # Dummy entry for CI legend
    ax.fill_between([], [], [], color="gray", alpha=0.2, label="±1 Std. Dev.")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Δ (full_logprob − patched_logprob)", fontsize=12)
    ax.set_title("S1 Layer-wise Delta by Patching Location (Retain → Full)",
                  fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "s1_component_deltas.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

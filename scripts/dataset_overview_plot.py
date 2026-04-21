#!/usr/bin/env python3
"""
dataset_overview_plot.py
========================
Generate a dataset overview figure for thesis presentation.

Shows:
  1. Distribution of complexity scores (by split)
  2. Distribution of circuit sizes / gate counts (log scale)
  3. Train / Val / Test split pie chart
  4. Complexity score vs gate count scatter

Usage:
    python scripts/dataset_overview_plot.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed_complexity"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots"

def load_split_data(splits, split_name):
    scores, gates = [], []
    for fname in splits[split_name]:
        pt_path = DATA_DIR / fname
        if pt_path.exists():
            d = torch.load(pt_path, map_location="cpu", weights_only=False)
            if hasattr(d, "complexity_score"):
                scores.append(float(d.complexity_score))
            gates.append(int(d.x.shape[0]))
    return np.array(scores), np.array(gates)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)

    train_s, train_g = load_split_data(splits, "train")
    val_s, val_g = load_split_data(splits, "val")
    test_s, test_g = load_split_data(splits, "test")

    all_s = np.concatenate([train_s, val_s, test_s])
    all_g = np.concatenate([train_g, val_g, test_g])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Overview", fontsize=18, fontweight="bold", y=0.98)

    colors = {"train": "#4C8BF5", "val": "#F5A623", "test": "#E05D5D"}

    ax = axes[0, 0]
    bins = np.linspace(0, 5.2, 22)
    ax.hist(train_s, bins=bins, alpha=0.75, label=f"Train ({len(train_s)})",
            color=colors["train"], edgecolor="white", linewidth=0.5)
    ax.hist(val_s, bins=bins, alpha=0.75, label=f"Val ({len(val_s)})",
            color=colors["val"], edgecolor="white", linewidth=0.5)
    ax.hist(test_s, bins=bins, alpha=0.75, label=f"Test ({len(test_s)})",
            color=colors["test"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Complexity Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Complexity Score Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5.2)

    ax = axes[0, 1]
    log_gates = np.log10(all_g.clip(1))
    bins_g = np.linspace(0, np.ceil(log_gates.max()), 25)
    ax.hist(log_gates, bins=bins_g, alpha=0.8, color="#5BA370",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Circuit Size (log₁₀ gates)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Circuit Size Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    tick_vals = [1, 2, 3, 4, 5]
    tick_labels = ["10", "100", "1K", "10K", "100K"]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)

    ax = axes[1, 0]
    sizes = [len(train_s), len(val_s), len(test_s)]
    labels = [f"Train\n{sizes[0]} ({sizes[0]/sum(sizes)*100:.0f}%)",
              f"Val\n{sizes[1]} ({sizes[1]/sum(sizes)*100:.0f}%)",
              f"Test\n{sizes[2]} ({sizes[2]/sum(sizes)*100:.0f}%)"]
    wedges, texts = ax.pie(
        sizes, labels=labels,
        colors=[colors["train"], colors["val"], colors["test"]],
        startangle=90, textprops={"fontsize": 11, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title(f"Train / Val / Test Split  ({sum(sizes)} total)",
                 fontsize=14, fontweight="bold")

    ax = axes[1, 1]
    ax.scatter(train_g, train_s, alpha=0.5, s=25, color=colors["train"],
               label="Train", zorder=2)
    ax.scatter(val_g, val_s, alpha=0.7, s=35, color=colors["val"],
               label="Val", zorder=3)
    ax.scatter(test_g, test_s, alpha=0.7, s=35, color=colors["test"],
               label="Test", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Gate Count (log scale)", fontsize=12)
    ax.set_ylabel("Complexity Score", fontsize=12)
    ax.set_title("Complexity vs Circuit Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTPUT_DIR / "dataset_overview.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()

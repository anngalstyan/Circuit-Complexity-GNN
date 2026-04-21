#!/usr/bin/env python3
"""
plot_training_curves.py
=======================
Generate publication-quality training curve plots from a saved checkpoint.

Reads the ``history`` dict inside the checkpoint and produces:
  - training_loss_curve.png
  - validation_r2_curve.png
  - validation_mae_curve.png

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --model best_complexity_model.pt --output-dir plots/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

def smooth(values, weight=0.85):
    """Exponential moving average for trend line."""
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed

def plot_loss(history, output_dir):
    loss = history["train_loss"]
    epochs = np.arange(1, len(loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, loss, alpha=0.35, color="steelblue", lw=1, label="Raw")
    ax.plot(epochs, smooth(loss), color="steelblue", lw=2, label="Smoothed (EMA)")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Training Loss", fontsize=13)
    ax.set_title("Training Loss Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(loss))
    plt.tight_layout()
    out = output_dir / "training_loss_curve.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

def plot_r2(history, output_dir):
    r2 = history["val_r2"]
    epochs = np.arange(1, len(r2) + 1)
    best_idx = int(np.argmax(r2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, r2, color="forestgreen", lw=1.5)
    ax.scatter([best_idx + 1], [r2[best_idx]], color="red", s=80, zorder=5,
               label=f"Best = {r2[best_idx]:.4f} (epoch {best_idx + 1})")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Validation R\u00b2", fontsize=13)
    ax.set_title("Validation R\u00b2 vs Epoch", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(r2))
    plt.tight_layout()
    out = output_dir / "validation_r2_curve.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

def plot_mae(history, output_dir):
    mae = history["val_mae"]
    epochs = np.arange(1, len(mae) + 1)
    best_idx = int(np.argmin(mae))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, mae, color="darkorange", lw=1.5)
    ax.scatter([best_idx + 1], [mae[best_idx]], color="red", s=80, zorder=5,
               label=f"Best = {mae[best_idx]:.4f} (epoch {best_idx + 1})")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Validation MAE", fontsize=13)
    ax.set_title("Validation MAE vs Epoch", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(mae))
    plt.tight_layout()
    out = output_dir / "validation_mae_curve.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

def main():
    p = argparse.ArgumentParser(description="Plot training curves from checkpoint")
    p.add_argument("--model", default="best_complexity_model.pt")
    p.add_argument("--output-dir", default="plots", help="Directory for output PNGs")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if "history" not in ckpt:
        print("ERROR: Checkpoint has no 'history' key.")
        return

    history = ckpt["history"]
    n_epochs = len(history.get("train_loss", []))
    print(f"Loaded checkpoint: {n_epochs} epochs of training history")

    plot_loss(history, output_dir)
    plot_r2(history, output_dir)
    plot_mae(history, output_dir)
    print(f"\nAll plots saved to {output_dir}/")

if __name__ == "__main__":
    main()

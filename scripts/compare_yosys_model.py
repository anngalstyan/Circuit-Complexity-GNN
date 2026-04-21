#!/usr/bin/env python3
"""
compare_yosys_model.py
======================
Side-by-side comparison of Yosys synthesis results vs the parser/model pipeline.

For each .v file it:
  1. Runs Yosys synthesis to get gate count and depth
  2. Runs the parser + formula to get the structural complexity score
  3. Runs the GNN model to get the predicted complexity score

Outputs:
  - Console table comparing all metrics
  - Scatter plot PNGs (yosys vs parser gates/depth, model vs formula)
  - Summary correlations

Usage:
    python scripts/compare_yosys_model.py --circuits data/raw --model best_complexity_model.pt
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_yosys_features import run_yosys
from netlist_parser import GateLevelNetlistParser
from circuit_complexity_model import ImprovedGIN_GAT

def _predict_single(model, data, mean, std, device="cpu"):
    """Run a single forward pass on one Data object and return the score."""
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = float(out.cpu().numpy().flatten()[0]) * std + mean
    return float(np.clip(pred, 0.0, 5.0))

def main():
    p = argparse.ArgumentParser(description="Compare Yosys vs parser/model")
    p.add_argument("--circuits", default="data/raw", help="Directory of .v files")
    p.add_argument("--model", default="best_complexity_model.pt")
    p.add_argument("--liberty", default="data/raw/library/GSCLib_3.0.lib",
                   help="Liberty file for gate-level netlists")
    p.add_argument("--output-dir", default="plots", help="Output directory for plots")
    p.add_argument("--max-files", type=int, default=0, help="Limit files (0=all)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = ImprovedGIN_GAT.from_checkpoint(ckpt)
    model.eval()
    mean, std = ckpt["mean"], ckpt["std"]
    feat_stats = ckpt.get("feat_stats")

    liberty = args.liberty if Path(args.liberty).exists() else None
    if liberty:
        print(f"Using liberty file: {liberty}")

    files = sorted(
        f for f in Path(args.circuits).rglob("*.v")
        if "library" not in str(f)
    )
    if args.max_files > 0:
        files = files[:args.max_files]

    print(f"\nComparing {len(files)} circuits: Yosys vs Parser/Model")
    print("=" * 100)
    print(f"{'File':<35} {'Y_gates':>8} {'P_gates':>8} {'Y_depth':>7} {'P_depth':>7} "
          f"{'Formula':>8} {'Model':>8} {'Delta':>7}")
    print("-" * 100)

    rows = []
    for fpath in files:
        yosys_result = run_yosys(fpath, liberty_file=liberty)
        y_gates = yosys_result["gate_count"] if yosys_result else None
        y_depth = yosys_result["depth"] if yosys_result else None

        try:
            parser = GateLevelNetlistParser()
            parser.parse_verilog_netlist(str(fpath))
            metrics = parser.compute_structural_complexity()
            p_gates = metrics.get("gate_count", 0)
            p_depth = metrics.get("depth", 0)
            formula = metrics.get("complexity_score", 0.0)
        except Exception:
            continue

        if p_gates == 0:
            continue

        try:
            data = parser.to_pytorch_geometric(target_metric="complexity_score")
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            if feat_stats and hasattr(data, "global_feats") and data.global_feats is not None:
                fm = torch.tensor(feat_stats["feat_mean"], dtype=torch.float)
                fs = torch.tensor(feat_stats["feat_std"], dtype=torch.float)
                data.global_feats = (data.global_feats - fm) / fs
            model_pred = _predict_single(model, data, mean, std)
        except Exception:
            model_pred = float("nan")

        delta = model_pred - formula
        rows.append({
            "file": fpath.name, "y_gates": y_gates, "p_gates": p_gates,
            "y_depth": y_depth, "p_depth": p_depth,
            "formula": formula, "model_pred": model_pred, "delta": delta,
        })

        y_g_str = f"{y_gates:>8}" if y_gates is not None else "   FAIL"
        y_d_str = f"{y_depth:>7}" if y_depth is not None else "  FAIL"
        print(f"{fpath.name:<35} {y_g_str} {p_gates:>8} {y_d_str} {p_depth:>7} "
              f"{formula:>8.3f} {model_pred:>8.3f} {delta:>+7.3f}")

    if not rows:
        print("No circuits processed successfully.")
        return

    valid_yosys = [r for r in rows if r["y_gates"] is not None]
    print(f"\n{'='*60}")
    print(f"SUMMARY  ({len(rows)} circuits, {len(valid_yosys)} with Yosys)")
    print(f"{'='*60}")

    if len(valid_yosys) >= 3:
        yg = np.array([r["y_gates"] for r in valid_yosys])
        pg = np.array([r["p_gates"] for r in valid_yosys])
        yd = np.array([r["y_depth"] for r in valid_yosys])
        pd = np.array([r["p_depth"] for r in valid_yosys])

        r_gates, _ = stats.pearsonr(yg, pg)
        r_depth, _ = stats.pearsonr(yd, pd) if np.std(yd) > 0 and np.std(pd) > 0 else (0, 1)
        print(f"  Gate count correlation (Yosys vs Parser): r = {r_gates:.4f}")
        print(f"  Depth correlation      (Yosys vs Parser): r = {r_depth:.4f}")

    formulas = np.array([r["formula"] for r in rows])
    preds = np.array([r["model_pred"] for r in rows])
    mask = np.isfinite(preds)
    if mask.sum() >= 3:
        r_score, _ = stats.pearsonr(formulas[mask], preds[mask])
        mae = np.mean(np.abs(formulas[mask] - preds[mask]))
        print(f"  Model vs Formula correlation:              r = {r_score:.4f}")
        print(f"  Model vs Formula MAE:                      {mae:.4f}")

    if len(valid_yosys) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.scatter(yg, pg, alpha=0.6, s=40, color="steelblue")
        lo, hi = 0, max(yg.max(), pg.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
        ax.set_xlabel("Yosys Gate Count", fontsize=12)
        ax.set_ylabel("Parser Gate Count", fontsize=12)
        ax.set_title(f"Gate Count  (r = {r_gates:.3f})", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(yd, pd, alpha=0.6, s=40, color="forestgreen")
        lo, hi = 0, max(yd.max(), pd.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
        ax.set_xlabel("Yosys Depth (ABC)", fontsize=12)
        ax.set_ylabel("Parser Depth", fontsize=12)
        ax.set_title(f"Logic Depth  (r = {r_depth:.3f})", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.scatter(formulas[mask], preds[mask], alpha=0.6, s=40, color="darkorange")
        lo, hi = 0, 5.0
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
        ax.set_xlabel("Formula Score", fontsize=12)
        ax.set_ylabel("Model Prediction", fontsize=12)
        ax.set_title(f"Model vs Formula  (r = {r_score:.3f})", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        plt.tight_layout()
        out = output_dir / "yosys_vs_model_comparison.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved: {out}")

if __name__ == "__main__":
    main()

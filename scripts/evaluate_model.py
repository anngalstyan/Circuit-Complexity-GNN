"""
evaluate_model.py
=================
Comprehensive evaluation of the trained circuit complexity predictor.

Metrics reported
----------------
- R²  (coefficient of determination)
- Pearson and Spearman correlations
- MAE, RMSE, Median AE
- Within ±10 / 20 / 30 / 50 % accuracy
- Accuracy stratified by circuit size (gate count)
- Monte Carlo Dropout uncertainty (optional, --uncertainty)

Output files
------------
- complexity_evaluation.png   : four-panel diagnostic plot
- complexity_metrics.json     : machine-readable metrics dict

Usage
-----
    python evaluate_model.py
    python evaluate_model.py --data-dir ./processed_complexity --model best_complexity_model.pt
    python evaluate_model.py --uncertainty --mc-samples 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from circuit_complexity_model import (
    ImprovedGIN_GAT,
    _assign_bin_indices,
    evaluate,
    evaluate_ensemble,
    predict_with_uncertainty,
)

logger = logging.getLogger(__name__)

def load_test_data(data_dir: Path, checkpoint: Dict) -> List:
    """Load the test split and normalise targets + global features."""
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    target_mode = checkpoint.get("target_mode", "regression")
    softmax_edges = checkpoint.get("softmax_bin_edges")
    feat_stats = checkpoint.get("feat_stats")

    with open(data_dir / "splits.json") as f:
        splits = json.load(f)

    test_data = []
    for fname in splits["test"]:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        data = torch.load(fpath, weights_only=False)
        if not hasattr(data, "complexity_score") or data.gate_count == 0:
            continue
        data.y_raw = torch.tensor([[data.complexity_score]], dtype=torch.float)
        if target_mode == "softmax":
            data.y = data.y_raw.clone()
            if softmax_edges:
                edges = np.array(softmax_edges, dtype=np.float32)
                b = int(_assign_bin_indices(np.array([data.complexity_score], dtype=np.float32), edges)[0])
                data.y_bin = torch.tensor([b], dtype=torch.long)
        else:
            data.y = (data.y_raw - mean) / std

        if (feat_stats is not None
                and hasattr(data, "global_feats")
                and data.global_feats is not None):
            fm = torch.tensor(feat_stats["feat_mean"], dtype=torch.float)
            fs = torch.tensor(feat_stats["feat_std"],  dtype=torch.float)
            data.global_feats = (data.global_feats - fm) / fs

        test_data.append(data)

    logger.info("Test set: %d circuits", len(test_data))
    return test_data

def print_metrics(preds: np.ndarray, targets: np.ndarray, gate_counts: List) -> Dict:
    """Compute and print all evaluation metrics.  Returns the metrics dict."""
    errors = preds - targets

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    pearson,  _  = stats.pearsonr(targets, preds)
    spearman, _  = stats.spearmanr(targets, preds)
    ratios        = np.where(targets > 0, preds / targets, 1.0)

    print("\n" + "=" * 65)
    print("EVALUATION METRICS  —  Complexity Score Prediction")
    print("=" * 65)

    print("\n  Correlation")
    print(f"    R²              : {r2:.4f}")
    print(f"    Pearson  r      : {pearson:.4f}")
    print(f"    Spearman ρ      : {spearman:.4f}")

    print("\n  Absolute error")
    print(f"    MAE             : {np.mean(np.abs(errors)):.4f}")
    print(f"    RMSE            : {np.sqrt(np.mean(errors**2)):.4f}")
    print(f"    Median AE       : {np.median(np.abs(errors)):.4f}")
    print(f"    Max AE          : {np.max(np.abs(errors)):.4f}")

    print("\n  Within-N% accuracy")
    within: Dict[int, float] = {}
    for pct in (10, 20, 30, 50):
        w = float(np.mean(np.abs(ratios - 1) <= pct / 100) * 100)
        within[pct] = w
        print(f"    ±{pct:2d}%          : {w:.1f}%")

    if gate_counts:
        print("\n  By circuit size (gate count)")
        gc = np.array(gate_counts[: len(preds)])
        header = f"    {'Range':<18} {'n':>5}  {'MAE':>8}  {'R²':>8}"
        print(header)
        print("    " + "-" * 43)
        bins = [(0, 10), (10, 50), (50, 200), (200, 1000), (1000, float("inf"))]
        for lo, hi in bins:
            mask = (gc >= lo) & (gc < (hi if hi != float("inf") else 1e12))
            n = int(mask.sum())
            if n == 0:
                continue
            bin_mae = float(np.mean(np.abs(errors[mask])))
            bin_ss_res = np.sum(errors[mask] ** 2)
            bin_ss_tot = np.sum((targets[mask] - targets[mask].mean()) ** 2)
            bin_r2 = 1.0 - bin_ss_res / bin_ss_tot if bin_ss_tot > 0 else 0.0
            hi_str = f"{int(hi)}" if hi != float("inf") else "∞"
            label = f"{lo}–{hi_str}"
            print(f"    {label:<18} {n:>5}  {bin_mae:>8.4f}  {bin_r2:>8.4f}")

    print()

    metrics = {
        "r2":           float(r2),
        "pearson":      float(pearson),
        "spearman":     float(spearman),
        "mae":          float(np.mean(np.abs(errors))),
        "rmse":         float(np.sqrt(np.mean(errors ** 2))),
        "median_ae":    float(np.median(np.abs(errors))),
        "within_10pct": within.get(10, 0.0),
        "within_20pct": within.get(20, 0.0),
        "within_30pct": within.get(30, 0.0),
        "n_test":       len(preds),
    }
    return metrics

def create_diagnostic_plots(
    preds: np.ndarray,
    targets: np.ndarray,
    r2: float,
    mae: float,
    output_path: str,
) -> None:
    """Four-panel diagnostic figure suitable for thesis appendix."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Circuit Complexity Prediction — Evaluation", fontsize=14)

    ax = axes[0, 0]
    ax.scatter(targets, preds, alpha=0.6, s=30, color="steelblue", edgecolors="none")
    lo, hi = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
    ax.set_xlabel("Actual Complexity Score")
    ax.set_ylabel("Predicted Complexity Score")
    ax.set_title(f"Predicted vs Actual  (R² = {r2:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    errors = preds - targets
    ax.scatter(targets, errors, alpha=0.6, s=30, color="steelblue", edgecolors="none")
    ax.axhline(0, color="r", linestyle="--", lw=1.5)
    ax.set_xlabel("Actual Complexity Score")
    ax.set_ylabel("Residual  (predicted − actual)")
    ax.set_title(f"Residual Plot  (MAE = {mae:.3f})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(errors, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="r", linestyle="--", lw=1.5)
    ax.axvline(errors.mean(), color="orange", linestyle="--", lw=1.2,
               label=f"Mean = {errors.mean():.3f}")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    stats.probplot(errors, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot  (Normality Check)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)

def create_uncertainty_plot(
    targets: np.ndarray,
    unc: Dict,
    output_path: str,
) -> None:
    """Uncertainty plot: predictions ± 95% CI sorted by actual complexity."""
    order = np.argsort(targets)
    t_sorted   = targets[order]
    m_sorted   = unc["mean"][order]
    lo_sorted  = unc["ci_low"][order]
    hi_sorted  = unc["ci_high"][order]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(t_sorted))
    ax.fill_between(x, lo_sorted, hi_sorted, alpha=0.25, color="steelblue",
                    label="95% CI (MC Dropout)")
    ax.plot(x, m_sorted,  color="steelblue", lw=1.2, label="MC mean")
    ax.plot(x, t_sorted,  color="red",       lw=1.2, linestyle="--", label="Actual")
    ax.set_xlabel("Circuit (sorted by actual complexity)")
    ax.set_ylabel("Complexity Score")
    ax.set_title("Prediction Uncertainty  (Monte Carlo Dropout, 50 passes)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Evaluate circuit complexity model")
    p.add_argument("--model",      default="best_complexity_model.pt")
    p.add_argument("--data-dir",   default="data/processed_complexity")
    p.add_argument("--output-dir", default=".",
                   help="Directory for plots and JSON output.")
    p.add_argument("--uncertainty", action="store_true",
                   help="Run Monte Carlo Dropout uncertainty estimation.")
    p.add_argument("--mc-samples", type=int, default=50,
                   help="Number of MC Dropout forward passes (default: 50).")
    args = p.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model, weights_only=False, map_location="cpu")
    mean = checkpoint["mean"]
    std  = checkpoint["std"]
    logger.info("Loaded checkpoint from epoch %d  (val R²=%.4f)",
                checkpoint["epoch"] + 1, checkpoint.get("val_r2", float("nan")))
    logger.info("Normalisation: mean=%.4f  std=%.4f", mean, std)

    if "model_config" not in checkpoint:
        logger.warning("Checkpoint has no 'model_config' — inferring architecture from keys.")
        sd = checkpoint["model_state_dict"]
        gin_n = max(int(k.split(".")[1]) for k in sd if k.startswith("gin_layers.")) + 1
        gat_n = max(int(k.split(".")[1]) for k in sd if k.startswith("gat_layers.")) + 1
        model = ImprovedGIN_GAT(
            input_dim=checkpoint.get("input_dim", 24),
            hidden_dim=checkpoint.get("hidden_dim", 64),
            num_gin_layers=gin_n,
            num_gat_layers=gat_n,
        )
        model.load_state_dict(sd)
    else:
        model = ImprovedGIN_GAT.from_checkpoint(checkpoint)

    model.eval()
    logger.info("%s", model)

    test_data = load_test_data(data_dir, checkpoint)
    if not test_data:
        logger.error("Test set is empty.  Check --data-dir and splits.json.")
        sys.exit(1)

    device = torch.device("cpu")
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    target_mode = checkpoint.get("target_mode", "regression")
    bin_centers = checkpoint.get("softmax_bin_centers")
    results = evaluate(
        model, test_loader, device, mean, std,
        target_mode=target_mode, bin_centers=bin_centers,
    )
    metrics = print_metrics(results["preds"], results["targets"], results["gate_counts"])

    if args.uncertainty:
        print(f"Computing uncertainty  ({args.mc_samples} MC passes)…")
        unc = predict_with_uncertainty(
            model, test_loader, device, mean, std, n_samples=args.mc_samples,
            target_mode=target_mode, bin_centers=bin_centers,
        )
        avg_std = float(unc["std"].mean())
        print(f"  Mean prediction σ     : {avg_std:.4f}")
        print(f"  Mean 95% CI width     : {float((unc['ci_high'] - unc['ci_low']).mean()):.4f}")
        metrics["mc_mean_sigma"] = avg_std

        unc_plot = str(output_dir / "complexity_uncertainty.png")
        create_uncertainty_plot(results["targets"], unc, unc_plot)

    eval_plot = str(output_dir / "complexity_evaluation.png")
    create_diagnostic_plots(
        results["preds"],
        results["targets"],
        metrics["r2"],
        metrics["mae"],
        eval_plot,
    )

    metrics_path = output_dir / "complexity_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", metrics_path)

if __name__ == "__main__":
    main()

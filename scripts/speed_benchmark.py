#!/usr/bin/env python3
"""
speed_benchmark.py
==================
Benchmark model inference speed vs Yosys synthesis speed.

Selects representative circuits across size categories and times:
  - Model pipeline: parse -> metrics -> to_pytorch_geometric -> inference
  - Yosys pipeline: full synthesis with ABC

Outputs:
  - Per-circuit timing table
  - Average speedup factor
  - Bar chart PNG: model vs yosys time by circuit size category

Usage:
    python scripts/speed_benchmark.py --circuits data/raw --model best_complexity_model.pt
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_yosys_features import run_yosys
from netlist_parser import GateLevelNetlistParser
from circuit_complexity_model import ImprovedGIN_GAT

SIZE_CATEGORIES = [
    ("Tiny (1-10)",        1,      10),
    ("Small (11-50)",      11,     50),
    ("Medium (51-200)",    51,     200),
    ("Large (201-1K)",     201,    1000),
    ("1K-5K",              1001,   5000),
    ("5K-20K",             5001,   20000),
    ("20K-100K",           20001,  100000),
    ("100K+",              100001, float("inf")),
]

def get_gate_count(fpath):
    try:
        parser = GateLevelNetlistParser()
        parser.parse_verilog_netlist(str(fpath))
        m = parser.compute_structural_complexity()
        return m.get("gate_count", 0)
    except Exception:
        return 0

def time_model_pipeline(fpath, model, mean, std, feat_stats, n_runs=5, gate_count=0):
    """Time the full model pipeline (parse + metrics + graph build + inference)."""
    if gate_count > 50000:
        n_runs = 1
    elif gate_count > 10000:
        n_runs = 2
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        parser = GateLevelNetlistParser()
        parser.parse_verilog_netlist(str(fpath))
        metrics = parser.compute_structural_complexity()
        data = parser.to_pytorch_geometric(target_metric="complexity_score")
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        if feat_stats and hasattr(data, "global_feats") and data.global_feats is not None:
            fm = torch.tensor(feat_stats["feat_mean"], dtype=torch.float)
            fs = torch.tensor(feat_stats["feat_std"], dtype=torch.float)
            data.global_feats = (data.global_feats - fm) / fs
        with torch.no_grad():
            model.eval()
            data = data.to("cpu")
            out = model(data)
            _ = float(out.cpu().numpy().flatten()[0]) * std + mean
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    return np.median(times)

def time_yosys(fpath, liberty=None, n_runs=3, gate_count=0):
    """Time Yosys synthesis."""
    if gate_count > 50000:
        n_runs = 1
    elif gate_count > 10000:
        n_runs = 2
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = run_yosys(fpath, liberty_file=liberty)
        elapsed = (time.perf_counter() - t0) * 1000
        if result is not None:
            times.append(elapsed)
    return np.median(times) if times else None

def main():
    p = argparse.ArgumentParser(description="Speed benchmark: model vs Yosys")
    p.add_argument("--circuits", default="data/raw", help="Directory of .v files")
    p.add_argument("--model", default="models/best_complexity_model.pt")
    p.add_argument("--liberty", default="data/raw/library/GSCLib_3.0.lib",
                   help="Liberty file for gate-level netlists")
    p.add_argument("--output-dir", default="plots")
    p.add_argument("--max-per-category", type=int, default=5,
                   help="Max circuits per size category")
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
    print(f"Scanning {len(files)} files...")

    file_sizes = []
    for f in files:
        gc = get_gate_count(f)
        if gc > 0:
            file_sizes.append((f, gc))

    selected = []
    for cat_name, lo, hi in SIZE_CATEGORIES:
        matching = [(f, gc) for f, gc in file_sizes if lo <= gc <= hi]
        matching.sort(key=lambda x: x[1])
        chosen = matching[:args.max_per_category]
        for f, gc in chosen:
            selected.append((f, gc, cat_name))

    if not selected:
        print("No valid circuits found.")
        return

    print(f"\nBenchmarking {len(selected)} circuits across {len(SIZE_CATEGORIES)} size categories")
    print("=" * 95)
    print(f"{'File':<35} {'Gates':>6} {'Category':<16} {'Model(ms)':>10} {'Yosys(ms)':>10} {'Speedup':>8}")
    print("-" * 95)

    if selected:
        f0 = selected[0][0]
        time_model_pipeline(f0, model, mean, std, feat_stats, n_runs=2)

    results = []
    for fpath, gc, cat in selected:
        model_ms = time_model_pipeline(fpath, model, mean, std, feat_stats, gate_count=gc)
        yosys_ms = time_yosys(fpath, liberty=liberty, gate_count=gc)

        if yosys_ms is not None and yosys_ms > 0:
            speedup = yosys_ms / model_ms
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup = None
            speedup_str = "N/A"

        results.append({
            "file": fpath.name, "gates": gc, "category": cat,
            "model_ms": model_ms, "yosys_ms": yosys_ms, "speedup": speedup,
        })

        y_str = f"{yosys_ms:>10.1f}" if yosys_ms else "     FAIL"
        print(f"{fpath.name:<35} {gc:>6} {cat:<16} {model_ms:>10.1f} {y_str} {speedup_str:>8}")

    valid = [r for r in results if r["speedup"] is not None]
    if valid:
        speedups = [r["speedup"] for r in valid]
        model_times = [r["model_ms"] for r in valid]
        yosys_times = [r["yosys_ms"] for r in valid]

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  Circuits benchmarked:  {len(valid)}")
        print(f"  Avg model time:        {np.mean(model_times):.1f} ms")
        print(f"  Avg Yosys time:        {np.mean(yosys_times):.1f} ms")
        print(f"  Mean speedup:          {np.mean(speedups):.1f}x faster")
        print(f"  Median speedup:        {np.median(speedups):.1f}x faster")
        print(f"  Min speedup:           {np.min(speedups):.1f}x")
        print(f"  Max speedup:           {np.max(speedups):.1f}x")

    if valid:
        cat_model = {}
        cat_yosys = {}
        for r in valid:
            cat = r["category"]
            cat_model.setdefault(cat, []).append(r["model_ms"])
            cat_yosys.setdefault(cat, []).append(r["yosys_ms"])

        categories = [c for c, _, _ in SIZE_CATEGORIES if c in cat_model]
        model_avgs = [np.mean(cat_model[c]) for c in categories]
        yosys_avgs = [np.mean(cat_yosys[c]) for c in categories]

        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width / 2, model_avgs, width, label="GNN Model", color="steelblue")
        bars2 = ax.bar(x + width / 2, yosys_avgs, width, label="Yosys Synthesis", color="coral")

        ax.set_ylabel("Time (ms)", fontsize=13)
        ax.set_xlabel("Circuit Size Category", fontsize=13)
        ax.set_title("Speed Comparison: GNN Model vs Yosys Synthesis", fontsize=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        for i, (m, y) in enumerate(zip(model_avgs, yosys_avgs)):
            if m > 0:
                sp = y / m
                ax.annotate(f"{sp:.0f}x", xy=(i, max(m, y)),
                            fontsize=11, fontweight="bold", ha="center",
                            xytext=(0, 8), textcoords="offset points", color="red")

        ax.set_yscale("log")
        plt.tight_layout()
        out = output_dir / "speed_benchmark.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved: {out}")

if __name__ == "__main__":
    main()

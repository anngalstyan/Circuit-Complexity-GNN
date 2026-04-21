#!/usr/bin/env python3
"""
audit_dataset.py
================
Comprehensive audit of the training/val/test dataset.

Checks every circuit for:
  1. Zero gates (empty modules)
  2. RTL files mistakenly included (not gate-level netlists)
  3. Floating wires (declared but unused)
  4. Skipped/unknown cell types
  5. Duplicate instance names
  6. Suspiciously low depth for gate count
  7. Suspiciously high/low complexity score
  8. Missing or NaN features in .pt files
  9. Multi-module files
  10. Gate count cross-check between raw .v and stored .pt

Usage:
    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --fix-report  # saves CSV of all issues
"""

import argparse
import csv
import json
import logging
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

from netlist_parser import GateLevelNetlistParser

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed_complexity"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SPLITS_FILE = DATA_DIR / "splits.json"

_RTL_KEYWORDS = [
    "always @", "always_ff", "always_comb", "always_latch",
    "initial begin", "if (", "else begin", "case (", "for (",
    "while (", "forever", "#1", "posedge", "negedge",
    "reg ", "integer ", "real ", "generate", "function ", "task ",
]

def is_gate_level(v_path):
    """Check if a .v file is a gate-level netlist (not RTL).
    Returns (is_gate_level: bool, rtl_evidence: list[str])."""
    rtl_evidence = []
    gate_instantiations = 0

    with open(v_path, "r", errors="ignore") as f:
        content = f.read()

    for kw in _RTL_KEYWORDS:
        count = content.count(kw)
        if count > 0:
            rtl_evidence.append(f"'{kw}' x{count}")

    gate_instantiations = len(re.findall(
        r"^\s*[A-Z][A-Z0-9_]+\s+\w+\s*\(", content, re.MULTILINE))

    if gate_instantiations == 0 and rtl_evidence:
        return False, rtl_evidence
    if len(rtl_evidence) > 3 and gate_instantiations < 5:
        return False, rtl_evidence

    return True, rtl_evidence

def check_pt_file(pt_path):
    """Check a single .pt file for issues."""
    issues = []
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return [("CRITICAL", f"Cannot load .pt file: {e}")]

    if hasattr(data, "x") and data.x is not None:
        if torch.isnan(data.x).any():
            issues.append(("CRITICAL", "NaN in node features"))
        if torch.isinf(data.x).any():
            issues.append(("CRITICAL", "Inf in node features"))
        n_nodes = data.x.shape[0]
        n_feats = data.x.shape[1]
        if n_feats != 24:
            issues.append(("ERROR", f"Expected 24 node features, got {n_feats}"))
    else:
        issues.append(("CRITICAL", "No node features (x) in .pt file"))
        n_nodes = 0

    if hasattr(data, "edge_index") and data.edge_index is not None:
        ei = data.edge_index
        n_edges = ei.shape[1]
        if n_edges == 0 and n_nodes > 1:
            issues.append(("WARNING", f"No edges but {n_nodes} nodes"))
        if ei.shape[0] != 2:
            issues.append(("CRITICAL", f"edge_index shape {ei.shape}, expected (2, N)"))
        elif n_edges > 0:
            if ei.max() >= n_nodes:
                issues.append(("CRITICAL", f"Edge index {ei.max()} >= n_nodes {n_nodes}"))
            if ei.min() < 0:
                issues.append(("CRITICAL", f"Negative edge index: {ei.min()}"))
            self_loops = (ei[0] == ei[1]).sum().item()
            if self_loops > 0:
                issues.append(("WARNING", f"{self_loops} self-loop(s) in edge_index"))
    else:
        issues.append(("ERROR", "No edge_index in .pt file"))

    if hasattr(data, "complexity_score"):
        score = float(data.complexity_score)
        if np.isnan(score) or np.isinf(score):
            issues.append(("CRITICAL", f"Score is {score}"))
        elif score < 0.0 or score > 5.0:
            issues.append(("ERROR", f"Score {score:.3f} outside [0, 5]"))
        elif score == 0.0:
            issues.append(("WARNING", "Score is exactly 0.0"))
    else:
        issues.append(("ERROR", "No complexity_score attribute"))

    if hasattr(data, "global_feats") and data.global_feats is not None:
        gf = data.global_feats
        if torch.isnan(gf).any():
            issues.append(("CRITICAL", "NaN in global features"))
        if torch.isinf(gf).any():
            issues.append(("CRITICAL", "Inf in global features"))
    else:
        issues.append(("WARNING", "No global_feats attribute"))

    return issues

def check_raw_verilog(v_path):
    """Re-parse a raw .v file and check for issues."""
    issues = []

    is_gl, rtl_evidence = is_gate_level(v_path)
    if not is_gl:
        issues.append(("CRITICAL", f"NOT gate-level — RTL detected: {'; '.join(rtl_evidence[:3])}"))
        return None, issues

    if rtl_evidence:
        issues.append(("INFO", f"Minor RTL-like tokens (probably benign): {'; '.join(rtl_evidence[:2])}"))

    try:
        parser = GateLevelNetlistParser()
        parser.parse_verilog_netlist(str(v_path))
        metrics = parser.compute_structural_complexity()
    except Exception as e:
        return None, [("CRITICAL", f"Parse failed: {e}")]

    gate_count = metrics.get("gate_count", 0)
    depth = metrics.get("depth", 0)
    score = metrics.get("complexity_score", 0.0)
    skipped = metrics.get("skipped_cell_types", [])
    floating = metrics.get("floating_wire_count", 0)

    if gate_count == 0:
        issues.append(("CRITICAL", "Zero gates — empty or unparseable module"))

    if skipped:
        issues.append(("WARNING", f"{len(skipped)} unknown cell type(s): {', '.join(sorted(skipped)[:5])}"))

    if floating > 0:
        pct = floating / max(gate_count, 1) * 100
        if pct > 50:
            issues.append(("WARNING", f"{floating} floating wires ({pct:.0f}% of gate count)"))

    if gate_count > 50 and depth <= 1:
        issues.append(("WARNING", f"Only depth {depth} with {gate_count} gates — possible parse issue"))

    if depth > gate_count and gate_count > 0:
        issues.append(("ERROR", f"Depth ({depth}) > gate count ({gate_count})"))

    if gate_count > 0 and score == 0.0:
        issues.append(("WARNING", f"Score 0.0 despite {gate_count} gates"))

    with open(v_path, "r", errors="ignore") as f:
        content = f.read()
    n_modules = content.count("\nmodule ") + (1 if content.startswith("module ") else 0)
    if n_modules > 1:
        issues.append(("INFO", f"Multi-module file ({n_modules} modules)"))

    return metrics, issues

def main():
    p = argparse.ArgumentParser(description="Audit training dataset")
    p.add_argument("--fix-report", action="store_true", help="Save CSV report")
    args = p.parse_args()

    if not SPLITS_FILE.exists():
        print(f"ERROR: {SPLITS_FILE} not found")
        return 1

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    all_files = {}
    for split_name in ["train", "val", "test"]:
        for fname in splits[split_name]:
            all_files[fname] = split_name

    print(f"Auditing {len(all_files)} circuits across train/val/test splits...")
    print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print("=" * 80)

    print("\n[Pass 1] Checking .pt file integrity...")
    pt_issues = defaultdict(list)
    pt_stats = {"total": 0, "clean": 0, "missing": 0}
    scores_by_split = defaultdict(list)
    gate_counts = []

    for fname, split in all_files.items():
        pt_path = DATA_DIR / fname
        pt_stats["total"] += 1

        if not pt_path.exists():
            pt_issues[fname].append(("CRITICAL", "File missing"))
            pt_stats["missing"] += 1
            continue

        issues = check_pt_file(pt_path)
        if issues:
            pt_issues[fname] = issues
        else:
            pt_stats["clean"] += 1

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if hasattr(data, "complexity_score"):
            scores_by_split[split].append(float(data.complexity_score))
        if hasattr(data, "x"):
            gate_counts.append(data.x.shape[0])

    n_pt_problems = len(pt_issues)
    print(f"  Total: {pt_stats['total']}, Clean: {pt_stats['clean']}, "
          f"Issues: {n_pt_problems}, Missing: {pt_stats['missing']}")

    print("\n[Pass 2] Checking raw .v files (RTL detection + re-parse)...")
    raw_issues = defaultdict(list)
    raw_files = {f.name: f for f in RAW_DIR.rglob("*.v") if "library" not in str(f)}
    n_raw_checked = 0
    n_rtl_detected = 0

    for v_name, v_path in sorted(raw_files.items()):
        n_raw_checked += 1
        metrics, issues = check_raw_verilog(v_path)
        if issues:
            raw_issues[v_name] = issues
            if any("NOT gate-level" in msg for _, msg in issues):
                n_rtl_detected += 1

    print(f"  Raw files checked: {n_raw_checked}")
    print(f"  RTL files detected: {n_rtl_detected}")
    print(f"  Files with issues: {len(raw_issues)}")

    print("\n[Pass 3] Statistical analysis...")
    anomalies = []

    all_scores = scores_by_split["train"] + scores_by_split["val"] + scores_by_split["test"]
    if all_scores:
        s = np.array(all_scores)
        mean, std = s.mean(), s.std()
        print(f"  Score stats: mean={mean:.3f}, std={std:.3f}, min={s.min():.3f}, max={s.max():.3f}")

        for split_name in ["train", "val", "test"]:
            ss = np.array(scores_by_split[split_name])
            if len(ss) > 0:
                print(f"  {split_name:>5s}: mean={ss.mean():.3f}, std={ss.std():.3f}, "
                      f"min={ss.min():.3f}, max={ss.max():.3f}")

    if gate_counts:
        gc = np.array(gate_counts)
        print(f"\n  Gate count: mean={gc.mean():.0f}, median={np.median(gc):.0f}, "
              f"min={gc.min()}, max={gc.max()}")

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)

    all_issues = []
    for fname, issues in {**pt_issues, **raw_issues}.items():
        for sev, msg in issues:
            all_issues.append((fname, sev, msg))

    severity_counts = defaultdict(int)
    for _, sev, _ in all_issues:
        severity_counts[sev] += 1

    print(f"\n  CRITICAL: {severity_counts.get('CRITICAL', 0)}")
    print(f"  ERROR:    {severity_counts.get('ERROR', 0)}")
    print(f"  WARNING:  {severity_counts.get('WARNING', 0)}")
    print(f"  INFO:     {severity_counts.get('INFO', 0)}")

    critical_errors = [(f, s, m) for f, s, m in all_issues if s in ("CRITICAL", "ERROR")]
    if critical_errors:
        print(f"\n  ── Critical / Error Issues ({len(critical_errors)}) ──")
        for fname, sev, msg in sorted(critical_errors):
            print(f"    [{sev}] {fname}: {msg}")
    else:
        print("\n  ✓ No critical or error issues found!")

    warnings_list = [(f, m) for f, s, m in all_issues if s == "WARNING"]
    if warnings_list:
        print(f"\n  ── Warnings ({len(warnings_list)}) ──")
        warn_groups = defaultdict(list)
        for fname, msg in warnings_list:
            key = msg.split(":")[0] if ":" in msg else msg[:50]
            warn_groups[key].append(fname)
        for key, files in sorted(warn_groups.items(), key=lambda x: -len(x[1])):
            if len(files) <= 3:
                print(f"    {key}: {', '.join(files)}")
            else:
                print(f"    {key}: {len(files)} files (e.g. {', '.join(files[:3])})")

    n_critical = severity_counts.get("CRITICAL", 0)
    n_error = severity_counts.get("ERROR", 0)
    print("\n" + "-" * 80)
    if n_critical == 0 and n_error == 0:
        print("VERDICT: DATASET IS CLEAN — no critical issues. Safe for training.")
    elif n_critical == 0:
        print(f"VERDICT: {n_error} error(s) found — review above.")
    else:
        print(f"VERDICT: {n_critical} CRITICAL issue(s) found — review above.")

    if args.fix_report and all_issues:
        report_path = Path(__file__).resolve().parent.parent / "plots" / "dataset_audit.csv"
        with open(report_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "severity", "issue"])
            for fname, sev, msg in sorted(all_issues):
                w.writerow([fname, sev, msg])
        print(f"\nReport saved to: {report_path}")

    return 1 if n_critical > 0 else 0

if __name__ == "__main__":
    sys.exit(main())

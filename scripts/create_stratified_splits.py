"""
create_stratified_splits.py
============================
Create reproducible, stratified train / val / test splits.

Key improvements over naive random splitting
---------------------------------------------
- Decile-bin stratification keeps complexity distributions similar across splits.
- Forced high-complexity allocation: circuits above *high_threshold* always get
  at least *min_high_per_split* representatives in every split.  This prevents
  the test set from being nearly empty of hard cases when high-complexity
  circuits are rare.
- A SHA-256 fingerprint of the data directory is stored alongside splits so
  stale splits can be detected automatically.

Usage
-----
    python create_stratified_splits.py --data ./processed_complexity
    python create_stratified_splits.py --data ./processed_complexity \\
        --val-ratio 0.15 --test-ratio 0.15 --high-threshold 4.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

def fingerprint_directory(data_dir: Path) -> str:
    """Return a short SHA-256 fingerprint of all .pt filenames in *data_dir*.

    Used to detect stale splits when circuits are added or removed.
    """
    names = sorted(f.name for f in data_dir.glob("*.pt"))
    digest = hashlib.sha256(" ".join(names).encode()).hexdigest()[:16]
    return digest

def _base_name(fname: str) -> str:
    """Extract the base circuit name, stripping augmentation suffixes.

    Examples
    --------
    >>> _base_name("b03_heavy.pt")
    'b03'
    >>> _base_name("pipeline_8w_5s_original.pt")
    'pipeline_8w_5s'
    >>> _base_name("leon3_avnet_3s1500_feedback.pt")
    'leon3_avnet_3s1500'
    """
    for suffix in ("_original", "_heavy", "_deep", "_seq", "_feedback", "_xor"):
        if fname.endswith(suffix + ".pt"):
            return fname[:-(len(suffix) + 3)]
    return fname.replace(".pt", "")

def create_stratified_splits(
    data_dir: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    n_bins: int = 10,
    high_threshold: float = 4.0,
    min_high_per_split: int = 2,
) -> Dict[str, List[str]]:
    """Create stratified, **group-aware** train / val / test splits.

    All augmented variants of the same base circuit are kept in the same split
    to prevent data leakage.

    Parameters
    ----------
    data_dir:
        Directory containing preprocessed ``.pt`` files and ``splits.json``.
    val_ratio, test_ratio:
        Fraction of data for validation and test (train gets the remainder).
    seed:
        Random seed for reproducibility.
    n_bins:
        Number of decile bins for stratification.
    high_threshold:
        Complexity score above which a circuit is considered *high complexity*.
    min_high_per_split:
        Minimum number of high-complexity circuits guaranteed in each split.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``  (lists of filenames).
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    pt_files = [f for f in data_dir.glob("*.pt") if f.name != "metadata.json"]
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    scored: List[Tuple[str, float]] = []
    for f in pt_files:
        try:
            data = torch.load(f, weights_only=False)
            score = 0.0
            if hasattr(data, "complexity_score"):
                score = float(data.complexity_score)
            elif hasattr(data, "y"):
                score = float(data.y.item() if data.y.numel() == 1 else data.y[0])
            scored.append((f.name, score))
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)

    if not scored:
        raise RuntimeError("Could not load any .pt files.")

    logger.info("Loaded %d circuits", len(scored))

    groups: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for fname, score in scored:
        groups[_base_name(fname)].append((fname, score))

    logger.info("Grouped into %d base circuits (%d total files)",
                len(groups), len(scored))

    group_scores: List[Tuple[str, float]] = []
    for base, members in groups.items():
        orig = [s for n, s in members if "_original.pt" in n]
        rep_score = orig[0] if orig else np.mean([s for _, s in members])
        group_scores.append((base, rep_score))

    high   = [(b, s) for b, s in group_scores if s >= high_threshold]
    normal = [(b, s) for b, s in group_scores if s <  high_threshold]

    logger.info("High-complexity groups (≥%.1f): %d   Normal groups: %d",
                high_threshold, len(high), len(normal))

    rng.shuffle(high)
    n_high = len(high)
    n_high_test = max(min_high_per_split, int(n_high * test_ratio))
    n_high_val  = max(min_high_per_split, int(n_high * val_ratio))
    forced_test  = high[:n_high_test]
    forced_val   = high[n_high_test: n_high_test + n_high_val]
    forced_train = high[n_high_test + n_high_val:]

    normal.sort(key=lambda x: x[1])
    bin_size = max(len(normal) // n_bins, 1)

    bins: Dict[int, List] = defaultdict(list)
    for i, item in enumerate(normal):
        bins[min(i // bin_size, n_bins - 1)].append(item)

    train_groups: List = []
    val_groups:   List = []
    test_groups:  List = []

    for items in bins.values():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        test_groups.extend(items[:n_test])
        val_groups.extend(items[n_test: n_test + n_val])
        train_groups.extend(items[n_test + n_val:])

    def _expand(group_list):
        files = []
        for base, _ in group_list:
            files.extend(groups[base])
        return files

    train = _expand(forced_train + train_groups)
    val   = _expand(forced_val   + val_groups)
    test  = _expand(forced_test  + test_groups)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    train_bases = {_base_name(n) for n, _ in train}
    val_bases   = {_base_name(n) for n, _ in val}
    test_bases  = {_base_name(n) for n, _ in test}
    leak = (train_bases & val_bases) | (train_bases & test_bases) | (val_bases & test_bases)
    if leak:
        raise RuntimeError(f"Group leakage detected for: {leak}")
    logger.info("✓ No group leakage — all variants of each circuit stay in one split.")

    def _stats(items: List[Tuple[str, float]], label: str) -> None:
        scores = [s for _, s in items]
        high_n = sum(1 for s in scores if s >= high_threshold)
        logger.info(
            "  %-8s  n=%4d  min=%5.3f  max=%5.3f  mean=%5.3f  "
            "std=%5.3f  high=%d",
            label, len(scores),
            min(scores), max(scores), np.mean(scores), np.std(scores), high_n,
        )

    print("\nSplit statistics:")
    _stats(train, "Train")
    _stats(val,   "Val")
    _stats(test,  "Test")

    for split_a, name_a in [(train, "train"), (val, "val"), (test, "test")]:
        for split_b, name_b in [(train, "train"), (val, "val"), (test, "test")]:
            if name_a >= name_b:
                continue
            diff = abs(np.mean([s for _, s in split_a]) -
                       np.mean([s for _, s in split_b]))
            if diff > 0.5:
                logger.warning(
                    "Mean complexity difference between %s and %s is %.3f (>0.5) "
                    "— consider adjusting n_bins or adding more high-complexity circuits.",
                    name_a, name_b, diff,
                )

    splits = {
        "train": [n for n, _ in train],
        "val":   [n for n, _ in val],
        "test":  [n for n, _ in test],
        "_meta": {
            "seed":           seed,
            "val_ratio":      val_ratio,
            "test_ratio":     test_ratio,
            "high_threshold": high_threshold,
            "n_bins":         n_bins,
            "data_fingerprint": fingerprint_directory(data_dir),
        },
    }

    splits_path = data_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info("Splits saved to %s", splits_path)
    return splits

def check_splits_fresh(data_dir: Path) -> bool:
    """Return True if existing splits match the current data directory contents."""
    splits_path = data_dir / "splits.json"
    if not splits_path.exists():
        return False
    with open(splits_path) as f:
        splits = json.load(f)
    meta = splits.get("_meta", {})
    stored_fp = meta.get("data_fingerprint", "")
    current_fp = fingerprint_directory(data_dir)
    if stored_fp != current_fp:
        logger.warning(
            "Data directory has changed since splits were created "
            "(stored fingerprint %s ≠ current %s). "
            "Re-run create_stratified_splits.py.",
            stored_fp, current_fp,
        )
        return False
    return True

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Create stratified train/val/test splits")
    p.add_argument("--data",            type=Path, default=Path("./processed_complexity"))
    p.add_argument("--val-ratio",       type=float, default=0.15)
    p.add_argument("--test-ratio",      type=float, default=0.15)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--n-bins",          type=int,   default=10)
    p.add_argument("--high-threshold",  type=float, default=4.0,
                   help="Complexity score above which a circuit is 'high complexity'.")
    p.add_argument("--min-high",        type=int,   default=2,
                   help="Minimum high-complexity circuits guaranteed per split.")
    p.add_argument("--check",           action="store_true",
                   help="Only check whether existing splits are still fresh.")
    args = p.parse_args()

    if not args.data.exists():
        logger.error("Directory not found: %s", args.data)
        return

    if args.check:
        ok = check_splits_fresh(args.data)
        print("Splits are FRESH." if ok else "Splits are STALE — please regenerate.")
        return

    splits = create_stratified_splits(
        args.data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        n_bins=args.n_bins,
        high_threshold=args.high_threshold,
        min_high_per_split=args.min_high,
    )

    print(f"\nFinal counts:")
    print(f"  Train : {len(splits['train'])}")
    print(f"  Val   : {len(splits['val'])}")
    print(f"  Test  : {len(splits['test'])}")
    print(f"\nNext step: python circuit_complexity_model.py")

if __name__ == "__main__":
    main()

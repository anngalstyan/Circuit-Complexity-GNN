"""
Preprocess Dataset - Parse Once, Use Forever
Parses all netlists and saves them as .pt files for fast loading
"""

import re
import random
import torch
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import sys

from netlist_parser import GateLevelNetlistParser

class DatasetPreprocessor:
    """Parse netlists once and cache them"""

    def __init__(self, netlist_dir, output_dir='processed_data', target_metric='complexity_score'):
        self.netlist_dir = Path(netlist_dir)
        self.output_dir = Path(output_dir)
        self.target_metric = target_metric
        self.parser = GateLevelNetlistParser()

        self.output_dir.mkdir(exist_ok=True)

    def analyze_and_filter(self, max_gates=None):
        """Analyze dataset and optionally filter large circuits"""
        print("\n" + "="*70)
        print("Step 1: Analyzing Dataset")
        print("="*70)

        v_files = sorted(list(self.netlist_dir.glob('*.[vV]')))
        print(f"\nFound {len(v_files)} Verilog files")

        circuit_info = []

        print("\nAnalyzing circuits...")
        for vfile in tqdm(v_files, desc="Parsing"):
            try:
                self.parser.parse_verilog_netlist(str(vfile))
                gate_count = len(self.parser.gates)
                metrics = self.parser.compute_complexity_metrics()

                circuit_info.append({
                    'filename': vfile.name,
                    'path': str(vfile),
                    'gate_count': gate_count,
                    'depth': metrics['depth'],
                    'composite': metrics['composite'],
                })

            except Exception as e:
                print(f"\n  Warning: Failed to parse {vfile.name}: {e}")
                continue

        gate_counts = [c['gate_count'] for c in circuit_info]

        print(f"\nDataset Statistics:")
        print(f"  Total circuits:  {len(circuit_info)}")
        if not gate_counts:
            print("  No circuits parsed successfully!")
            return circuit_info
        print(f"  Min gates:       {min(gate_counts):,}")
        print(f"  Max gates:       {max(gate_counts):,}")
        print(f"  Average gates:   {np.mean(gate_counts):,.0f}")
        print(f"  Median gates:    {np.median(gate_counts):,.0f}")

        print(f"\nSize Distribution:")
        tiny = sum(1 for g in gate_counts if g < 100)
        small = sum(1 for g in gate_counts if 100 <= g < 1000)
        medium = sum(1 for g in gate_counts if 1000 <= g < 10000)
        large = sum(1 for g in gate_counts if 10000 <= g < 50000)
        huge = sum(1 for g in gate_counts if g >= 50000)

        print(f"  Tiny   (<100):      {tiny:3d} circuits")
        print(f"  Small  (100-1K):    {small:3d} circuits")
        print(f"  Medium (1K-10K):    {medium:3d} circuits")
        print(f"  Large  (10K-50K):   {large:3d} circuits")
        print(f"  Huge   (50K+):      {huge:3d} circuits")

        if max_gates:
            filtered = [c for c in circuit_info if c['gate_count'] <= max_gates]
            removed = len(circuit_info) - len(filtered)

            print(f"\nFiltering circuits > {max_gates:,} gates:")
            print(f"  Keeping:  {len(filtered)} circuits")
            print(f"  Removing: {removed} circuits")

            if removed > 0:
                print(f"\n  Removed circuits:")
                for c in circuit_info:
                    if c['gate_count'] > max_gates:
                        print(f"    - {c['filename']:40s} ({c['gate_count']:,} gates)")

            circuit_info = filtered

        return circuit_info

    def preprocess_and_save(self, circuit_info):
        """Parse and save all circuits as .pt files"""
        print("\n" + "="*70)
        print("Step 2: Preprocessing and Saving")
        print("="*70)

        print(f"\nProcessing {len(circuit_info)} circuits...")
        print(f"Output directory: {self.output_dir.absolute()}\n")

        successful = []
        failed = []

        for info in tqdm(circuit_info, desc="Saving"):
            try:
                self.parser.parse_verilog_netlist(info['path'])

                data = self.parser.to_pytorch_geometric(target_metric=self.target_metric)

                output_file = self.output_dir / f"{Path(info['filename']).stem}.pt"
                torch.save(data, output_file)

                successful.append(info['filename'])

            except Exception as e:
                failed.append((info['filename'], str(e)))
                print(f"\n  Failed: {info['filename']} - {e}")

        print(f"\nSuccessfully processed: {len(successful)} circuits")
        if failed:
            print(f"Failed: {len(failed)} circuits")

        metadata = {
            'num_circuits': len(successful),
            'target_metric': self.target_metric,
            'circuit_info': circuit_info,
            'failed': failed,
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata saved to: {self.output_dir / 'metadata.json'}")

        return successful, failed

    @staticmethod
    def _base_name(filename: str) -> str:
        """Extract base circuit name, stripping variant suffixes.

        Examples:
            'ac97_ctrl_original.pt' -> 'ac97_ctrl'
            'sr_latch_2.pt'         -> 'sr_latch'
            'b14_original.pt'       -> 'b14'
        """
        base = filename.replace('.pt', '')
        for suffix in ['_original', '_feedback', '_reconvergent', '_xorheavy']:
            base = base.replace(suffix, '')
        base = re.sub(r'_\d+$', '', base)
        return base

    def create_splits(self, test_size=0.15, val_size=0.15, seed=42,
                      max_variants=None):
        """Create group-aware train/val/test splits.

        All variants of the same base circuit (e.g. ac97_ctrl_original,
        ac97_ctrl_feedback) go into the SAME split, preventing data leakage
        from near-duplicate circuits across splits.

        Parameters
        ----------
        max_variants : int or None
            If set, cap each family to at most this many variants.
            Variants are selected to maximise score diversity within
            the family.  Reduces redundancy from near-identical circuits.
        """
        print("\n" + "="*70)
        print("Step 3: Creating Group-Aware Train/Val/Test Splits")
        print("="*70)

        pt_files = sorted(list(self.output_dir.glob('*.pt')))
        if len(pt_files) == 0:
            print("No .pt files found! Run preprocessing first.")
            return

        print(f"\nFound {len(pt_files)} preprocessed circuits")

        circuit_sizes = {}
        for pt_file in pt_files:
            data = torch.load(pt_file, weights_only=False)
            circuit_sizes[pt_file.name] = data.gate_count

        groups = defaultdict(list)
        for pt_file in pt_files:
            base = self._base_name(pt_file.name)
            groups[base].append(pt_file.name)

        total_before = sum(len(v) for v in groups.values())
        if max_variants is not None:
            for base in groups:
                variants = groups[base]
                if len(variants) > max_variants:
                    scored = [(f, torch.load(self.output_dir / f,
                               weights_only=False).complexity_score)
                              for f in variants]
                    scored.sort(key=lambda x: x[1])
                    step = max(1, len(scored) / max_variants)
                    kept = [scored[int(i * step)] for i in range(max_variants)]
                    groups[base] = [f for f, _ in kept]
            total_after = sum(len(v) for v in groups.values())
            print(f"  Dedup: {total_before} -> {total_after} circuits "
                  f"(max {max_variants} per family)")

        group_names = list(groups.keys())
        print(f"  {len(group_names)} unique circuit families")

        def _median_size(base):
            sizes = [circuit_sizes[f] for f in groups[base]]
            return sorted(sizes)[len(sizes) // 2]

        group_names.sort(key=_median_size)

        random.seed(seed)
        n_groups = len(group_names)
        n_test_g = max(1, int(n_groups * test_size))
        n_val_g  = max(1, int(n_groups * val_size))
        chunk_size = max(1, int(1.0 / test_size))

        test_groups, val_groups, train_groups = [], [], []
        for i in range(0, n_groups, chunk_size):
            chunk = group_names[i:i + chunk_size]
            random.shuffle(chunk)
            if chunk and len(test_groups) < n_test_g:
                test_groups.append(chunk.pop(0))
            if chunk and len(val_groups) < n_val_g:
                val_groups.append(chunk.pop(0))
            train_groups.extend(chunk)

        train_files = [f for g in train_groups for f in groups[g]]
        val_files   = [f for g in val_groups   for f in groups[g]]
        test_files  = [f for g in test_groups  for f in groups[g]]

        n = len(pt_files)
        print(f"\nSplit sizes (group-aware):")
        print(f"  Train: {len(train_files):3d} circuits  ({len(train_groups):3d} families, {len(train_files)/n*100:.1f}%)")
        print(f"  Val:   {len(val_files):3d} circuits  ({len(val_groups):3d} families, {len(val_files)/n*100:.1f}%)")
        print(f"  Test:  {len(test_files):3d} circuits  ({len(test_groups):3d} families, {len(test_files)/n*100:.1f}%)")

        train_bases = {self._base_name(f) for f in train_files}
        val_bases   = {self._base_name(f) for f in val_files}
        test_bases  = {self._base_name(f) for f in test_files}
        if train_bases & test_bases or train_bases & val_bases or val_bases & test_bases:
            print(f"\n  WARNING: Family overlap detected across splits!")
        else:
            print(f"\n  No family overlap across splits")

        splits = {
            'train': sorted(train_files),
            'val':   sorted(val_files),
            'test':  sorted(test_files),
        }

        with open(self.output_dir / 'splits.json', 'w') as f:
            json.dump(splits, f, indent=2)

        print(f"\nSplits saved to: {self.output_dir / 'splits.json'}")

        return splits

def main():
    """Main preprocessing pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess circuit dataset')
    parser.add_argument('--dataset', default='../dataset', help='Dataset directory')
    parser.add_argument('--output', default='processed_data', help='Output directory')
    parser.add_argument('--target', default='complexity_score', choices=['complexity_score', 'composite', 'gate_count', 'depth'])
    parser.add_argument('--max-gates', type=int, help='Filter circuits larger than this')
    parser.add_argument('--max-variants', type=int, default=None,
                        help='Cap each circuit family to at most N variants (dedup)')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip analysis, just preprocess')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Dataset Preprocessing Pipeline")
    print("="*70)
    print(f"\nInput:  {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Target: {args.target}")
    if args.max_gates:
        print(f"Filter: Max {args.max_gates:,} gates")
    if args.max_variants:
        print(f"Dedup:  Max {args.max_variants} variants per family")

    preprocessor = DatasetPreprocessor(
        netlist_dir=args.dataset,
        output_dir=args.output,
        target_metric=args.target
    )

    circuit_info = preprocessor.analyze_and_filter(max_gates=args.max_gates)

    if len(circuit_info) == 0:
        print("\nNo circuits to process!")
        return

    successful, failed = preprocessor.preprocess_and_save(circuit_info)

    if len(successful) == 0:
        print("\nNo circuits successfully processed!")
        return

    splits = preprocessor.create_splits(max_variants=args.max_variants)

    print("\n" + "="*70)
    print("Preprocessing Complete!")
    print("="*70)
    print(f"\nPreprocessed data saved in: {Path(args.output).absolute()}")
    print(f"   - {len(successful)} circuit files (.pt)")
    print(f"   - metadata.json (dataset info)")
    print(f"   - splits.json (train/val/test indices)")

    print(f"\nNext steps:")
    print(f"   1. Train with preprocessed data:")
    print(f"      python circuit_complexity_model.py --data-dir {args.output}")

    if failed:
        print(f"\n  Note: {len(failed)} circuits failed to process")
        print(f"   Check metadata.json for details")

if __name__ == "__main__":
    main()

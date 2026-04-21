#!/usr/bin/env python3
"""
extract_yosys_features.py
=========================
Runs Yosys on every .v file in a directory and extracts:
  - gate_count       : total logic cells after synthesis
  - dff_count        : flip-flop count
  - depth            : critical path in gate levels (via ABC)
  - wire_count       : number of internal wires
  - cell_area        : estimated area (generic units, no liberty file needed)

Then computes the complexity score:
  raw   = depth * log10(gate_count) + 2.0 * feedback_ratio * log10(gate_count+1)
  score = sigmoid_normalized(raw) in [0, 5]

feedback_ratio is computed by your existing parser (Kosaraju SCC detection).

Output: data/yosys_features.json  — used as ground-truth targets for training.

Usage:
    python scripts/extract_yosys_features.py --dir data/all_circuits
    python scripts/extract_yosys_features.py --dir data/all_circuits --output data/yosys_features.json
"""

import re
import json
import math
import argparse
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

YOSYS_SCRIPT_RTL = """
read_verilog -sv {verilog_file}
hierarchy -auto-top
proc
flatten
opt
techmap
opt
abc -g gates
stat
tee -o {stat_file} stat
"""

YOSYS_SCRIPT_GATELVL = """
read_liberty -lib {liberty_file}
read_verilog {verilog_file}
hierarchy -auto-top
stat -liberty {liberty_file}
tee -o {stat_file} stat -liberty {liberty_file}
"""

def parse_stat_output(stat_text: str) -> dict:
    """
    Parse the output of Yosys 'stat' command.
    Returns dict with gate_count, dff_count, wire_count, cell_area.
    """
    result = {
        'gate_count': 0,
        'dff_count':  0,
        'wire_count': 0,
        'cell_area':  0.0,
    }

    for line in stat_text.splitlines():
        line = line.strip()

        m = re.match(r'Number of cells:\s+(\d+)', line)
        if m:
            result['gate_count'] = int(m.group(1))
        m = re.match(r'(\d+)\s+.*?cells\s*$', line)
        if m:
            result['gate_count'] = int(m.group(1))

        m = re.match(r'\$_?[Dd][Ff][Ff].*?\s+(\d+)', line)
        if m:
            result['dff_count'] += int(m.group(1))
        m = re.match(r'(\d+)\s+[\d.E+\-]+\s+DFF\w*', line)
        if m:
            result['dff_count'] += int(m.group(1))

        m = re.match(r'Number of wires:\s+(\d+)', line)
        if m:
            result['wire_count'] = int(m.group(1))
        m = re.match(r'(\d+)\s+.*?wires\s*$', line)
        if m:
            result['wire_count'] = int(m.group(1))

        m = re.match(r'Chip area.*?:\s+([\d.]+)', line)
        if m:
            result['cell_area'] = float(m.group(1))

    return result

def parse_abc_depth(yosys_stdout: str) -> int:
    """
    Extract logic depth from ABC output in Yosys stdout.
    ABC prints a line like: 'ABC: netlist: ... lev=12'
    or from 'print_stats': 'lev = 12'
    """
    m = re.search(r'\blev\s*=\s*(\d+)', yosys_stdout)
    if m:
        return int(m.group(1))

    m = re.search(r'Depth\s*=\s*(\d+)', yosys_stdout, re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r'(\d+)\s+levels?', yosys_stdout, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return 0

def run_yosys(verilog_path: Path, liberty_file: str = None) -> dict:
    """
    Run Yosys on a single file and return extracted features.
    Returns None if Yosys fails.

    If *liberty_file* is given the gate-level flow is used (no synthesis),
    otherwise the RTL synthesis flow (techmap + abc).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        stat_file   = Path(tmpdir) / 'stat.txt'
        script_file = Path(tmpdir) / 'run.ys'

        if liberty_file:
            script = YOSYS_SCRIPT_GATELVL.format(
                verilog_file=str(verilog_path),
                stat_file=str(stat_file),
                liberty_file=str(liberty_file),
            )
        else:
            script = YOSYS_SCRIPT_RTL.format(
                verilog_file=str(verilog_path),
                stat_file=str(stat_file),
            )
        script_file.write_text(script)

        try:
            result = subprocess.run(
                ['yosys', '-q', str(script_file)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {verilog_path.name}")
            return None
        except FileNotFoundError:
            print("ERROR: yosys not found. Install with: brew install yosys")
            return None

        stdout = result.stdout + result.stderr

        if result.returncode != 0:
            if 'ERROR' in stdout:
                err = next((l for l in stdout.splitlines() if 'ERROR' in l), '')
                print(f"  FAILED ({verilog_path.name}): {err[:80]}")
            return None

        stat_text = stat_file.read_text() if stat_file.exists() else stdout
        features  = parse_stat_output(stat_text if stat_text else stdout)

        features['depth'] = parse_abc_depth(stdout)

        return features

def compute_feedback_ratio_from_verilog(verilog_path: Path) -> float:
    """
    Use the existing GateLevelNetlistParser to get feedback_ratio
    (Kosaraju SCC detection). This doesn't depend on Yosys.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        from netlist_parser import GateLevelNetlistParser
        parser = GateLevelNetlistParser()
        parser.parse_verilog_netlist(str(verilog_path))
        struct = parser.compute_structural_complexity()
        return struct['feedback_ratio']
    except Exception:
        return 0.0

def compute_complexity_score(depth: int, gate_count: int, feedback_ratio: float) -> dict:
    """
    Academically grounded complexity formula:
      raw   = depth * log10(max(gate_count, 1))
            + 2.0 * feedback_ratio * log10(max(gate_count, 1) + 1)
      score = 5 / (1 + exp(1.5 - raw/15))   [sigmoid, range 0–5]
    """
    log_g  = math.log10(max(gate_count, 1))
    log_g1 = math.log10(max(gate_count, 1) + 1)
    raw    = depth * log_g + 2.0 * feedback_ratio * log_g1
    score  = 5.0 / (1.0 + math.exp(1.5 - raw / 15.0))
    return {
        'raw_score':        round(raw, 4),
        'complexity_score': round(score, 4),
    }

def main():
    parser = argparse.ArgumentParser(description='Extract Yosys features for all circuits')
    parser.add_argument('--dir',    default='data/all_circuits',     help='Directory of .v files')
    parser.add_argument('--output', default='data/yosys_features.json', help='Output JSON file')
    parser.add_argument('--resume', action='store_true', help='Skip files already in output JSON')
    args = parser.parse_args()

    dirpath = Path(args.dir)
    files   = sorted(dirpath.glob('*.v')) + sorted(dirpath.glob('*.sv'))
    print(f"\nFound {len(files)} files in {dirpath}")

    existing = {}
    if args.resume and Path(args.output).exists():
        existing = json.loads(Path(args.output).read_text())
        print(f"Resuming — {len(existing)} already extracted")

    results = dict(existing)
    failed  = []

    for i, vfile in enumerate(files, 1):
        name = vfile.name
        if args.resume and name in results:
            continue

        print(f"[{i:3d}/{len(files)}] {name:<50}", end='', flush=True)

        features = run_yosys(vfile)
        if features is None:
            failed.append(name)
            print(" FAILED")
            continue

        features['feedback_ratio'] = compute_feedback_ratio_from_verilog(vfile)

        scores = compute_complexity_score(
            features['depth'],
            features['gate_count'],
            features['feedback_ratio'],
        )
        features.update(scores)
        results[name] = features

        print(f" gates={features['gate_count']:>6,}  depth={features['depth']:>4}  "
              f"fb={features['feedback_ratio']:.2f}  score={features['complexity_score']:.3f}")

        Path(args.output).write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"Done. {len(results)} circuits extracted, {len(failed)} failed.")
    if failed:
        print(f"\nFailed files:")
        for f in failed:
            print(f"  {f}")
    print(f"\nOutput: {args.output}")
    print(f"\nNext step:")
    print(f"  python scripts/preprocess_from_yosys.py \\")
    print(f"      --features {args.output} \\")
    print(f"      --circuits {args.dir} \\")
    print(f"      --output   data/processed_complexity")

if __name__ == '__main__':
    main()

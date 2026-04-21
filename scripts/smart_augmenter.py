#!/usr/bin/env python3
"""
Smart Circuit Augmenter v2 — Proportional Structural Augmentation
==================================================================
Creates augmented circuits whose **complexity scores differ meaningfully**
from the originals.  The key insight from v1 was that adding 4–8 isolated
gates to an 11 000-gate circuit produces a score delta of < 0.001.

Fix: every augmentation now scales **proportionally** to the original
circuit size and **integrates** new structures into existing wires
(not just inputs).

Augmentation strategies
-----------------------
1. feedback_heavy   – proportional feedback loops on internal wires
2. depth_extend     – additional logic layers on output paths
3. xor_mesh         – XOR parity tree tapping internal signals
4. sequential_inject– insert DFF / latch elements (→ sequential)
5. combined         – layers multiple strategies for maximum delta

Usage
-----
    python scripts/smart_augmenter.py \\
        --input data/raw --output data/augmented
"""

import argparse
import random
import re
from pathlib import Path
from collections import defaultdict
import shutil

_GATES_2IN = ["AND2X1", "OR2X1", "NAND2X1", "NOR2X1", "XOR2X1", "XNOR2X1"]
_GATES_1IN = ["INVX1", "BUFX2"]

def _parse_netlist(filepath: Path) -> dict:
    """Lightweight regex parser — enough for augmentation."""
    content = filepath.read_text()

    module_match = re.search(r"module\s+(\w+)", content)
    module_name = module_match.group(1) if module_match else "circuit"

    inputs = re.findall(r"input\s+(?:\[\d+:\d+\])?\s*(\w+)", content)
    outputs = re.findall(r"output\s+(?:\[\d+:\d+\])?\s*(\w+)", content)

    wires: list[str] = []
    for w in re.findall(r"wire\s+(?:\[\d+:\d+\])?\s*([^;]+);", content):
        wires.extend(x.strip() for x in w.split(",") if x.strip())

    gates = re.findall(r"(\w+X\d+)\s+(\w+)\s*\(([^)]+)\)", content)

    return {
        "module_name": module_name,
        "inputs": inputs,
        "outputs": outputs,
        "wires": wires,
        "gates": gates,
        "raw_content": content,
    }

def _internal_wires(circuit: dict) -> list[str]:
    """Return wires that are neither primary inputs nor outputs."""
    io = set(circuit["inputs"]) | set(circuit["outputs"])
    return [w for w in circuit["wires"] if w not in io]

def _inject_block(content: str, block: str) -> str:
    """Insert *block* just before ``endmodule``."""
    pos = content.rfind("endmodule")
    return content[:pos] + block + "\n" + content[pos:]

def _rename_module(content: str, old_name: str, suffix: str) -> tuple[str, str]:
    new_name = f"{old_name}_{suffix}"
    return content.replace(f"module {old_name}", f"module {new_name}", 1), new_name

class ProportionalAugmenter:
    """
    All augmentation sizes are proportional to the original circuit.
    """

    def __init__(self, input_dir: str, output_dir: str, seed: int = 42):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)

    def augment_feedback(self, filepath: Path) -> Path | None:
        """
        Add feedback loops proportional to circuit size.
        Each loop taps a random **internal** wire and creates a
        NAND/XOR → INV cycle that feeds back through the original
        net's fanin cone.
        """
        c = _parse_netlist(filepath)
        n_gates = len(c["gates"])
        if n_gates < 10:
            return None

        n_loops = max(3, n_gates // 50)
        chain_len = max(2, min(n_gates // 100, 8))

        internals = _internal_wires(c) or c["inputs"][:5]
        self.rng.shuffle(internals)

        content, mod = _rename_module(c["raw_content"], c["module_name"], "fb")
        block = "\n// === PROPORTIONAL FEEDBACK INJECTION ===\n"

        all_fb_wires: list[str] = []
        for i in range(n_loops):
            tap = internals[i % len(internals)]
            wires_this: list[str] = []
            for j in range(chain_len):
                w = f"_fb{i}_{j}"
                wires_this.append(w)
                all_fb_wires.append(w)

            for j in range(chain_len):
                g = self.rng.choice(["NAND2X1", "NOR2X1", "XOR2X1"])
                inp_a = tap if j == 0 else wires_this[j - 1]
                inp_b = wires_this[-1] if j == 0 else tap
                block += f"{g} _Ufb{i}_{j} (.A({inp_a}), .B({inp_b}), .Y({wires_this[j]}));\n"

        block = f"wire {', '.join(all_fb_wires)};\n" + block

        content = _inject_block(content, block)
        dest = self.output_dir / f"{filepath.stem}_feedback.v"
        dest.write_text(content)
        return dest

    def augment_depth(self, filepath: Path) -> Path | None:
        """
        Extend logic depth by inserting additional gate layers
        between random internal wires and the outputs.
        """
        c = _parse_netlist(filepath)
        n_gates = len(c["gates"])
        if n_gates < 10:
            return None

        extra_layers = max(2, min(n_gates // 80, 10))

        internals = _internal_wires(c) or c["inputs"][:5]
        self.rng.shuffle(internals)
        taps = internals[: min(30, max(2, len(internals) // 5))]

        content, mod = _rename_module(c["raw_content"], c["module_name"], "deep")
        block = "\n// === DEPTH EXTENSION ===\n"

        all_wires: list[str] = []
        for t_idx, tap in enumerate(taps):
            prev = tap
            for layer in range(extra_layers):
                w = f"_dp{t_idx}_{layer}"
                all_wires.append(w)
                g = self.rng.choice(_GATES_2IN)
                other = taps[(t_idx + 1) % len(taps)]
                block += f"{g} _Udp{t_idx}_{layer} (.A({prev}), .B({other}), .Y({w}));\n"
                prev = w

        if all_wires:
            block = f"wire {', '.join(all_wires)};\n" + block

        content = _inject_block(content, block)
        dest = self.output_dir / f"{filepath.stem}_deep.v"
        dest.write_text(content)
        return dest

    def augment_xor_mesh(self, filepath: Path) -> Path | None:
        """
        Insert XOR parity tree tapping many internal signals.
        XOR density is a direct formula term, so this shifts scores.
        """
        c = _parse_netlist(filepath)
        n_gates = len(c["gates"])
        if n_gates < 10:
            return None

        internals = _internal_wires(c) or c["inputs"][:5]
        self.rng.shuffle(internals)

        n_xor = min(100, max(5, n_gates // 20))
        taps = [internals[i % len(internals)] for i in range(n_xor + 1)]

        content, mod = _rename_module(c["raw_content"], c["module_name"], "xor")
        block = "\n// === XOR MESH ===\n"

        xor_wires: list[str] = []
        prev = taps[0]
        for i in range(n_xor):
            w = f"_xm{i}"
            xor_wires.append(w)
            g = self.rng.choice(["XOR2X1", "XNOR2X1"])
            block += f"{g} _Uxm{i} (.A({prev}), .B({taps[i + 1]}), .Y({w}));\n"
            prev = w

        if xor_wires:
            block = f"wire {', '.join(xor_wires)};\n" + block

        content = _inject_block(content, block)
        dest = self.output_dir / f"{filepath.stem}_xor.v"
        dest.write_text(content)
        return dest

    def augment_sequential(self, filepath: Path) -> Path | None:
        """
        Insert DFF elements to make a combinational circuit sequential.
        The seq_ratio and circuit_type formula terms change significantly.
        """
        c = _parse_netlist(filepath)
        n_gates = len(c["gates"])
        if n_gates < 10:
            return None

        n_dff = max(2, n_gates // 30)

        internals = _internal_wires(c) or c["inputs"][:5]
        self.rng.shuffle(internals)

        content, mod = _rename_module(c["raw_content"], c["module_name"], "seq")
        block = "\n// === SEQUENTIAL INJECTION ===\n"

        dff_wires: list[str] = []
        has_clk = "clk" in c["inputs"] or "CLK" in c["inputs"]
        clk_name = "clk" if "clk" in c["inputs"] else "CLK" if "CLK" in c["inputs"] else "_aug_clk"

        if not has_clk:
            content = content.replace(
                f"module {mod}",
                f"module {mod}",
                1,
            )
            last_input = content.rfind("input ")
            end_of_line = content.find(";", last_input)
            content = content[:end_of_line + 1] + f"\ninput {clk_name};" + content[end_of_line + 1:]

        for i in range(n_dff):
            tap = internals[i % len(internals)]
            d_wire = f"_dff_d{i}"
            q_wire = f"_dff_q{i}"
            dff_wires.extend([d_wire, q_wire])
            block += f"BUFX2 _Udff_buf{i} (.A({tap}), .Y({d_wire}));\n"
            block += f"DFFX1 _Udff{i} (.D({d_wire}), .CK({clk_name}), .Q({q_wire}));\n"

        if dff_wires:
            block = f"wire {', '.join(dff_wires)};\n" + block

        content = _inject_block(content, block)
        dest = self.output_dir / f"{filepath.stem}_seq.v"
        dest.write_text(content)
        return dest

    def augment_combined(self, filepath: Path) -> Path | None:
        """
        Apply feedback + XOR + depth extension together for maximum
        score delta.  Writes a single combined circuit.
        """
        c = _parse_netlist(filepath)
        n_gates = len(c["gates"])
        if n_gates < 10:
            return None

        internals = _internal_wires(c) or c["inputs"][:5]
        self.rng.shuffle(internals)

        content, mod = _rename_module(c["raw_content"], c["module_name"], "heavy")
        combined_block = "\n// === COMBINED HEAVY AUGMENTATION ===\n"
        all_wires: list[str] = []

        n_loops = max(2, n_gates // 60)
        for i in range(n_loops):
            tap = internals[i % len(internals)]
            w0, w1 = f"_ch_fb{i}_0", f"_ch_fb{i}_1"
            all_wires.extend([w0, w1])
            g = self.rng.choice(["NAND2X1", "NOR2X1", "XOR2X1"])
            combined_block += f"{g} _Uchfb{i}a (.A({tap}), .B({w1}), .Y({w0}));\n"
            combined_block += f"INVX1 _Uchfb{i}b (.A({w0}), .Y({w1}));\n"

        n_xor = max(3, n_gates // 30)
        prev = internals[0]
        for i in range(n_xor):
            w = f"_ch_xm{i}"
            all_wires.append(w)
            g = self.rng.choice(["XOR2X1", "XNOR2X1"])
            other = internals[(i + 1) % len(internals)]
            combined_block += f"{g} _Uchxm{i} (.A({prev}), .B({other}), .Y({w}));\n"
            prev = w

        extra = max(2, n_gates // 100)
        prev = internals[len(internals) // 2 % len(internals)]
        for i in range(extra):
            w = f"_ch_dp{i}"
            all_wires.append(w)
            g = self.rng.choice(_GATES_2IN)
            other = internals[(i + 3) % len(internals)]
            combined_block += f"{g} _Uchdp{i} (.A({prev}), .B({other}), .Y({w}));\n"
            prev = w

        if all_wires:
            combined_block = f"wire {', '.join(all_wires)};\n" + combined_block

        content = _inject_block(content, combined_block)
        dest = self.output_dir / f"{filepath.stem}_heavy.v"
        dest.write_text(content)
        return dest

    def process_all(self):
        """Run all augmentation strategies on every circuit."""
        print(f"\n{'='*70}")
        print("Smart Circuit Augmentation v2 — Proportional")
        print(f"{'='*70}")
        print(f"Input:  {self.input_dir.resolve()}")
        print(f"Output: {self.output_dir.resolve()}")

        v_files = sorted(self.input_dir.glob("*.v")) + sorted(self.input_dir.glob("*.V"))
        print(f"\nFound {len(v_files)} original circuits\n")

        stats: dict[str, int] = defaultdict(int)

        for vf in v_files:
            print(f"  {vf.name} ({len(_parse_netlist(vf)['gates'])} gates) ... ", end="")

            shutil.copy(vf, self.output_dir / f"{vf.stem}_original.v")
            stats["original"] += 1

            strategies = [
                ("feedback",   self.augment_feedback),
                ("deep",       self.augment_depth),
                ("xor",        self.augment_xor_mesh),
                ("sequential", self.augment_sequential),
                ("combined",   self.augment_combined),
            ]

            for label, fn in strategies:
                try:
                    if fn(vf):
                        stats[label] += 1
                except Exception:
                    pass

            print("done")

        print(f"\n{'='*70}")
        print("Augmentation complete")
        print(f"{'='*70}")
        total = sum(stats.values())
        for k, v in stats.items():
            print(f"  {k:20s}: {v:4d}")
        print(f"  {'─' * 30}")
        print(f"  {'TOTAL':20s}: {total:4d}")
        print(f"\nOutput: {self.output_dir.resolve()}")

        return stats

def main():
    ap = argparse.ArgumentParser(description="Smart circuit augmentation v2")
    ap.add_argument("--input", required=True, help="Directory with original .v files")
    ap.add_argument("--output", required=True, help="Output directory for augmented circuits")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    aug = ProportionalAugmenter(args.input, args.output, seed=args.seed)
    aug.process_all()

if __name__ == "__main__":
    main()

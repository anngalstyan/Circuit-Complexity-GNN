#!/usr/bin/env python3
"""
Generate Missing Circuit Families
===================================
Creates gate-level Verilog netlists for circuit families that are
missing or underrepresented in the dataset.

Generated families
------------------
1. Ripple-carry adders             (combinational, arithmetic)
2. Carry-lookahead adders          (combinational, arithmetic, reconvergent)
3. Array multipliers               (combinational, arithmetic, deep)
4. CRC generators                  (sequential, feedback-heavy)
5. Finite state machines           (sequential, various sizes)
6. UART-style serial controllers   (sequential, moderate)
7. Pipeline stages                 (sequential, deep)
8. Hamming ECC encoder/decoder     (combinational, XOR-heavy)
9. FIR filter datapaths            (sequential + combinational)
10. Sorting / comparator networks  (combinational, wide)
11. LFSR / PRNG                    (sequential, feedback)
12. Priority encoder trees         (combinational, tree)
13. Barrel shifters                (combinational, mux-heavy)

All circuits use standard cells: AND2X1, OR2X1, NAND2X1, NOR2X1,
XOR2X1, XNOR2X1, INVX1, BUFX2, DFFX1, MUX2X1.

Usage
-----
    python scripts/generate_circuit_families.py --output data/raw
"""

import argparse
import random
from pathlib import Path

class VerilogBuilder:
    """Helper for constructing gate-level netlists."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.wires: list[str] = []
        self.gates: list[str] = []
        self._uid = 0

    def add_input(self, name: str):
        self.inputs.append(name)

    def add_output(self, name: str):
        self.outputs.append(name)

    def add_input_bus(self, name: str, width: int):
        for i in range(width):
            self.inputs.append(f"{name}_{i}")

    def add_output_bus(self, name: str, width: int):
        for i in range(width):
            self.outputs.append(f"{name}_{i}")

    def wire(self, prefix: str = "w") -> str:
        w = f"{prefix}_{self._uid}"
        self._uid += 1
        self.wires.append(w)
        return w

    def gate(self, cell: str, a: str, b: str | None = None, y: str | None = None) -> str:
        if y is None:
            y = self.wire()
        inst = f"U{self._uid}"
        self._uid += 1
        if b is None:
            self.gates.append(f"{cell} {inst} (.A({a}), .Y({y}));")
        else:
            self.gates.append(f"{cell} {inst} (.A({a}), .B({b}), .Y({y}));")
        return y

    def dff(self, d: str, ck: str, q: str | None = None) -> str:
        if q is None:
            q = self.wire("q")
        inst = f"U{self._uid}"
        self._uid += 1
        self.gates.append(f"DFFX1 {inst} (.D({d}), .CK({ck}), .Q({q}));")
        return q

    def inv(self, a: str, y: str | None = None) -> str:
        return self.gate("INVX1", a, y=y)

    def buf(self, a: str, y: str | None = None) -> str:
        return self.gate("BUFX2", a, y=y)

    def and2(self, a: str, b: str, y: str | None = None) -> str:
        return self.gate("AND2X1", a, b, y)

    def or2(self, a: str, b: str, y: str | None = None) -> str:
        return self.gate("OR2X1", a, b, y)

    def nand2(self, a: str, b: str, y: str | None = None) -> str:
        return self.gate("NAND2X1", a, b, y)

    def nor2(self, a: str, b: str, y: str | None = None) -> str:
        return self.gate("NOR2X1", a, b, y)

    def xor2(self, a: str, b: str, y: str | None = None) -> str:
        return self.gate("XOR2X1", a, b, y)

    def xnor2(self, a: str, b: str, y: str | None = None) -> str:
        return self.gate("XNOR2X1", a, b, y)

    def full_adder(self, a: str, b: str, cin: str) -> tuple[str, str]:
        """Returns (sum, cout)."""
        axb = self.xor2(a, b)
        s = self.xor2(axb, cin)
        ab = self.and2(a, b)
        axb_cin = self.and2(axb, cin)
        cout = self.or2(ab, axb_cin)
        return s, cout

    def mux2(self, d0: str, d1: str, sel: str) -> str:
        """y = sel ? d1 : d0"""
        ns = self.inv(sel)
        t0 = self.and2(d0, ns)
        t1 = self.and2(d1, sel)
        return self.or2(t0, t1)

    def build(self) -> str:
        lines = [f"// Auto-generated: {self.module_name}"]
        ports = ", ".join(self.inputs + self.outputs)
        lines.append(f"module {self.module_name} ({ports});")
        for p in self.inputs:
            lines.append(f"input {p};")
        for p in self.outputs:
            lines.append(f"output {p};")
        if self.wires:
            lines.append(f"wire {', '.join(self.wires)};")
        lines.append("")
        lines.extend(self.gates)
        lines.append("")
        lines.append("endmodule")
        return "\n".join(lines)

def gen_ripple_adder(n: int) -> str:
    b = VerilogBuilder(f"ripple_adder_{n}bit")
    b.add_input_bus("a", n)
    b.add_input_bus("b", n)
    b.add_input("cin")
    b.add_output_bus("s", n)
    b.add_output("cout")

    carry = "cin"
    for i in range(n):
        si, carry = b.full_adder(f"a_{i}", f"b_{i}", carry)
        b.buf(si, f"s_{i}")
    b.buf(carry, "cout")
    return b.build()

def gen_cla_adder(n: int) -> str:
    """4-bit CLA groups chained for n bits (n should be multiple of 4)."""
    n = max(4, (n // 4) * 4)
    b = VerilogBuilder(f"cla_adder_{n}bit")
    b.add_input_bus("a", n)
    b.add_input_bus("b", n)
    b.add_input("cin")
    b.add_output_bus("s", n)
    b.add_output("cout")

    carry = "cin"
    for grp in range(n // 4):
        gs, ps = [], []
        for i in range(4):
            idx = grp * 4 + i
            g = b.and2(f"a_{idx}", f"b_{idx}")
            p = b.xor2(f"a_{idx}", f"b_{idx}")
            gs.append(g)
            ps.append(p)

        carries = [carry]
        for i in range(4):
            c = gs[i]
            for j in range(i, -1, -1):
                if j == 0:
                    term = b.and2(ps[i] if i == j else c, carries[0])
                else:
                    prop = ps[j]
                    for k in range(j + 1, i + 1):
                        prop = b.and2(prop, ps[k])
                    term = b.and2(prop, gs[j - 1] if j > 0 else carries[0])
                c = b.or2(c, term)
            carries.append(c)

        for i in range(4):
            idx = grp * 4 + i
            s = b.xor2(ps[i], carries[i])
            b.buf(s, f"s_{idx}")
        carry = carries[4]

    b.buf(carry, "cout")
    return b.build()

def gen_array_multiplier(n: int) -> str:
    b = VerilogBuilder(f"array_mult_{n}bit")
    b.add_input_bus("a", n)
    b.add_input_bus("b", n)
    b.add_output_bus("p", 2 * n)

    pp: list[list[str]] = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(b.and2(f"a_{j}", f"b_{i}"))
        pp.append(row)

    result = [pp[0][j] for j in range(n)]
    b.buf(result[0], "p_0")

    for i in range(1, n):
        new_result = []
        carry = b.wire("zero")
        zero = b.and2(f"a_0", b.inv(f"a_0"))
        carry = zero
        for j in range(n):
            if j == 0:
                s, carry = b.full_adder(result[j + 1] if j + 1 < len(result) else carry,
                                         pp[i][j], carry)
                b.buf(result[0], f"p_{i}") if i > 0 and j == 0 else None
            else:
                a_val = result[j + 1] if j + 1 < len(result) else b.and2(f"a_0", b.inv(f"a_0"))
                s, carry = b.full_adder(a_val, pp[i][j], carry)
            new_result.append(s)
        new_result.append(carry)
        if i > 0:
            pass
        result = new_result

    for j in range(len(result)):
        idx = n + j - 1
        if idx < 2 * n:
            b.buf(result[j], f"p_{idx}")

    return b.build()

def gen_crc(poly_bits: int, name: str, poly_taps: list[int]) -> str:
    """Galois LFSR-based CRC with given polynomial taps."""
    b = VerilogBuilder(name)
    b.add_input("clk")
    b.add_input("rst")
    b.add_input("data_in")
    b.add_output_bus("crc", poly_bits)

    regs = []
    for i in range(poly_bits):
        regs.append(f"r_{i}")
    for r in regs:
        b.wires.append(r)

    fb = b.xor2(regs[-1], "data_in")

    nxt = []
    for i in range(poly_bits):
        if i == 0:
            n = fb
        elif i in poly_taps:
            n = b.xor2(regs[i - 1], fb)
        else:
            n = b.buf(regs[i - 1])
        nxt.append(n)

    for i in range(poly_bits):
        b.dff(nxt[i], "clk", regs[i])

    for i in range(poly_bits):
        b.buf(regs[i], f"crc_{i}")

    return b.build()

def gen_fsm(n_states: int, n_inputs: int, n_outputs: int, name: str,
            rng: random.Random) -> str:
    """Generate a random FSM with n_states states."""
    import math
    state_bits = max(1, math.ceil(math.log2(max(n_states, 2))))

    b = VerilogBuilder(name)
    b.add_input("clk")
    b.add_input("rst")
    b.add_input_bus("in", n_inputs)
    b.add_output_bus("out", n_outputs)

    state_wires = [f"st_{i}" for i in range(state_bits)]
    nxt_wires = [f"nst_{i}" for i in range(state_bits)]
    for w in state_wires + nxt_wires:
        b.wires.append(w)

    for bit in range(state_bits):
        signals = state_wires + [f"in_{i}" for i in range(n_inputs)]
        result = signals[0]
        for j in range(1, len(signals)):
            op = rng.choice([b.and2, b.or2, b.xor2, b.nand2, b.nor2])
            result = op(result, signals[j])
        if rng.random() > 0.5:
            result = b.inv(result)
        b.buf(result, nxt_wires[bit])

    for i in range(state_bits):
        b.dff(nxt_wires[i], "clk", state_wires[i])

    for o in range(n_outputs):
        result = state_wires[o % state_bits]
        for j in range(1, state_bits):
            op = rng.choice([b.and2, b.or2, b.xor2])
            result = op(result, state_wires[(o + j) % state_bits])
        if rng.random() > 0.5:
            result = b.inv(result)
        b.buf(result, f"out_{o}")

    return b.build()

def gen_uart_tx(data_bits: int = 8) -> str:
    b = VerilogBuilder(f"uart_tx_{data_bits}bit")
    b.add_input("clk")
    b.add_input("rst")
    b.add_input("tx_start")
    b.add_input_bus("tx_data", data_bits)
    b.add_output("tx_out")
    b.add_output("tx_busy")

    import math
    cnt_bits = max(1, math.ceil(math.log2(data_bits + 2)))

    sr = []
    for i in range(data_bits):
        sr.append(f"sr_{i}")
        b.wires.append(sr[-1])

    cnt = []
    for i in range(cnt_bits):
        cnt.append(f"cnt_{i}")
        b.wires.append(cnt[-1])

    busy_w = b.wire("busy")

    for i in range(data_bits):
        shifted = sr[i + 1] if i + 1 < data_bits else busy_w
        loaded = b.mux2(shifted, f"tx_data_{i}", "tx_start")
        b.dff(loaded, "clk", sr[i])

    carry = busy_w
    for i in range(cnt_bits):
        s = b.xor2(cnt[i], carry)
        carry = b.and2(cnt[i], carry)
        nxt = b.mux2(s, b.and2(f"tx_data_0", b.inv(f"tx_data_0")),
                      "tx_start")
        b.dff(nxt, "clk", cnt[i])

    any_cnt = cnt[0]
    for i in range(1, cnt_bits):
        any_cnt = b.or2(any_cnt, cnt[i])
    busy_logic = b.or2("tx_start", any_cnt)
    b.buf(busy_logic, busy_w)

    b.buf(sr[0], "tx_out")
    b.buf(busy_w, "tx_busy")

    return b.build()

def gen_pipeline(width: int, stages: int) -> str:
    b = VerilogBuilder(f"pipeline_{width}w_{stages}s")
    b.add_input("clk")
    b.add_input_bus("din", width)
    b.add_output_bus("dout", width)

    prev_layer = [f"din_{i}" for i in range(width)]

    for s in range(stages):
        curr_layer = []
        for i in range(width):
            j = (i + 1) % width
            mixed = b.xor2(prev_layer[i], prev_layer[j])
            if i % 3 == 0:
                mixed = b.and2(mixed, prev_layer[(i + 2) % width])
            elif i % 3 == 1:
                mixed = b.or2(mixed, prev_layer[(i + 2) % width])
            q = b.dff(mixed, "clk")
            curr_layer.append(q)
        prev_layer = curr_layer

    for i in range(width):
        b.buf(prev_layer[i], f"dout_{i}")

    return b.build()

def gen_hamming_encoder(data_bits: int) -> str:
    """SEC (single-error-correcting) Hamming encoder."""
    import math
    r = 0
    while (1 << r) < data_bits + r + 1:
        r += 1
    total = data_bits + r

    b = VerilogBuilder(f"hamming_enc_{data_bits}bit")
    b.add_input_bus("d", data_bits)
    b.add_output_bus("c", total)

    data_pos = []
    d_idx = 0
    for pos in range(1, total + 1):
        if pos & (pos - 1) != 0:
            data_pos.append((pos, d_idx))
            d_idx += 1

    parity_results = {}
    for p in range(r):
        parity_pos = 1 << p
        xor_inputs = []
        for pos, d_idx in data_pos:
            if pos & parity_pos:
                xor_inputs.append(f"d_{d_idx}")
        if len(xor_inputs) >= 2:
            result = xor_inputs[0]
            for x in xor_inputs[1:]:
                result = b.xor2(result, x)
        elif xor_inputs:
            result = b.buf(xor_inputs[0])
        else:
            result = b.and2("d_0", b.inv("d_0"))
        parity_results[parity_pos] = result

    d_idx = 0
    for pos in range(1, total + 1):
        out_idx = pos - 1
        if out_idx >= total:
            break
        if pos & (pos - 1) == 0:
            b.buf(parity_results[pos], f"c_{out_idx}")
        else:
            b.buf(f"d_{d_idx}", f"c_{out_idx}")
            d_idx += 1

    return b.build()

def gen_fir_filter(taps: int, width: int) -> str:
    """Simple FIR filter: y = sum(coeff[i] * x[n-i])
    Coefficients are hardwired as AND masks for simplicity."""
    b = VerilogBuilder(f"fir_{taps}tap_{width}bit")
    b.add_input("clk")
    b.add_input_bus("x", width)
    b.add_output_bus("y", width)

    delay = [[f"x_{j}" for j in range(width)]]
    for t in range(1, taps):
        tap_regs = []
        for j in range(width):
            q = b.dff(delay[t - 1][j], "clk")
            tap_regs.append(q)
        delay.append(tap_regs)

    accum = [delay[0][j] for j in range(width)]
    for t in range(1, taps):
        for j in range(width):
            accum[j] = b.xor2(accum[j], delay[t][j])
            if j > 0:
                accum[j] = b.or2(accum[j], b.and2(delay[t][j], accum[j - 1]))

    for j in range(width):
        b.buf(accum[j], f"y_{j}")

    return b.build()

def gen_comparator_network(n_elements: int, elem_width: int) -> str:
    """Compare-and-swap network (simplified bitonic sort structure)."""
    b = VerilogBuilder(f"sort_net_{n_elements}x{elem_width}bit")
    for i in range(n_elements):
        b.add_input_bus(f"in{i}", elem_width)
    for i in range(n_elements):
        b.add_output_bus(f"out{i}", elem_width)

    elems = [[f"in{i}_{j}" for j in range(elem_width)] for i in range(n_elements)]

    def compare_swap(idx_a, idx_b):
        nonlocal elems
        a_bits = elems[idx_a]
        b_bits = elems[idx_b]

        gt = b.and2(a_bits[-1], b.inv(b_bits[-1]))
        for k in range(elem_width - 2, -1, -1):
            eq_k = b.xnor2(a_bits[k + 1], b_bits[k + 1]) if k < elem_width - 2 else \
                   b.xnor2(a_bits[-1], b_bits[-1])
            gt_k = b.and2(a_bits[k], b.inv(b_bits[k]))
            gt = b.or2(gt, b.and2(eq_k, gt_k))

        new_a = [b.mux2(a_bits[j], b_bits[j], gt) for j in range(elem_width)]
        new_b = [b.mux2(b_bits[j], a_bits[j], gt) for j in range(elem_width)]
        elems[idx_a] = new_a
        elems[idx_b] = new_b

    step = 1
    while step < n_elements:
        for i in range(0, n_elements - step, step * 2):
            compare_swap(i, i + step)
        step *= 2
    for i in range(0, n_elements - 1):
        compare_swap(i, i + 1)

    for i in range(n_elements):
        for j in range(elem_width):
            b.buf(elems[i][j], f"out{i}_{j}")

    return b.build()

def gen_lfsr(n_bits: int, taps: list[int]) -> str:
    """Galois LFSR with given feedback taps."""
    b = VerilogBuilder(f"lfsr_{n_bits}bit")
    b.add_input("clk")
    b.add_input("rst")
    b.add_input("seed_load")
    b.add_input_bus("seed", n_bits)
    b.add_output_bus("q", n_bits)

    regs = [f"r_{i}" for i in range(n_bits)]
    for r in regs:
        b.wires.append(r)

    fb = regs[n_bits - 1]

    for i in range(n_bits):
        if i == 0:
            nxt = fb
        elif i in taps:
            nxt = b.xor2(regs[i - 1], fb)
        else:
            nxt = b.buf(regs[i - 1])

        loaded = b.mux2(nxt, f"seed_{i}", "seed_load")
        b.dff(loaded, "clk", regs[i])

    for i in range(n_bits):
        b.buf(regs[i], f"q_{i}")

    return b.build()

def gen_priority_encoder(n_inputs: int) -> str:
    import math
    out_bits = max(1, math.ceil(math.log2(max(n_inputs, 2))))

    b = VerilogBuilder(f"prienc_{n_inputs}input")
    b.add_input_bus("req", n_inputs)
    b.add_output_bus("grant", out_bits)
    b.add_output("valid")

    masked = []
    for i in range(n_inputs):
        sig = f"req_{i}"
        for j in range(i + 1, n_inputs):
            sig = b.and2(sig, b.inv(f"req_{j}"))
        masked.append(sig)

    for bit in range(out_bits):
        contributors = [masked[i] for i in range(n_inputs) if i & (1 << bit)]
        if contributors:
            result = contributors[0]
            for c in contributors[1:]:
                result = b.or2(result, c)
        else:
            result = b.and2("req_0", b.inv("req_0"))
        b.buf(result, f"grant_{bit}")

    v = f"req_0"
    for i in range(1, n_inputs):
        v = b.or2(v, f"req_{i}")
    b.buf(v, "valid")

    return b.build()

def gen_barrel_shifter(width: int) -> str:
    import math
    shift_bits = max(1, math.ceil(math.log2(width)))

    b = VerilogBuilder(f"barrel_shift_{width}bit")
    b.add_input_bus("d", width)
    b.add_input_bus("sh", shift_bits)
    b.add_output_bus("q", width)

    prev = [f"d_{i}" for i in range(width)]
    for s in range(shift_bits):
        curr = []
        shift_amt = 1 << s
        for i in range(width):
            src = prev[(i - shift_amt) % width]
            curr.append(b.mux2(prev[i], src, f"sh_{s}"))
        prev = curr

    for i in range(width):
        b.buf(prev[i], f"q_{i}")

    return b.build()

def generate_all(output_dir: Path, rng: random.Random):
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    def save(name: str, content: str):
        path = output_dir / f"{name}.v"
        path.write_text(content)
        generated.append(name)
        print(f"  {name}.v")

    print("\n--- Arithmetic circuits ---")
    for n in [4, 8, 16, 32]:
        save(f"ripple_adder_{n}bit", gen_ripple_adder(n))
    for n in [8, 16, 32]:
        save(f"cla_adder_{n}bit", gen_cla_adder(n))
    for n in [4, 8]:
        save(f"array_mult_{n}bit", gen_array_multiplier(n))

    print("\n--- CRC generators ---")
    save("crc8_ccitt", gen_crc(8, "crc8_ccitt", [0, 1, 2]))
    save("crc16_ibm", gen_crc(16, "crc16_ibm", [0, 2, 15]))
    save("crc32_ieee", gen_crc(32, "crc32_ieee", [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 16, 22, 23, 26]))

    print("\n--- Finite state machines ---")
    for states, ins, outs in [(4, 2, 2), (8, 3, 3), (16, 4, 4), (32, 4, 6),
                               (64, 5, 8), (128, 6, 8)]:
        save(f"fsm_{states}state", gen_fsm(states, ins, outs,
                                            f"fsm_{states}state", rng))

    print("\n--- UART controllers ---")
    save("uart_tx_8bit", gen_uart_tx(8))

    print("\n--- Pipeline stages ---")
    for w, s in [(8, 3), (8, 5), (16, 4), (16, 8), (32, 5)]:
        save(f"pipeline_{w}w_{s}s", gen_pipeline(w, s))

    print("\n--- Hamming ECC encoders ---")
    for d in [4, 8, 16, 32, 64]:
        save(f"hamming_enc_{d}bit", gen_hamming_encoder(d))

    print("\n--- FIR filter datapaths ---")
    for taps, w in [(4, 8), (8, 8), (8, 16), (16, 8), (16, 16)]:
        save(f"fir_{taps}tap_{w}bit", gen_fir_filter(taps, w))

    print("\n--- Sorting / comparator networks ---")
    for n_elem, ew in [(4, 4), (4, 8), (8, 4), (8, 8)]:
        save(f"sort_net_{n_elem}x{ew}bit", gen_comparator_network(n_elem, ew))

    print("\n--- LFSR / PRNG ---")
    save("lfsr_8bit", gen_lfsr(8, [1, 2, 3, 7]))
    save("lfsr_16bit", gen_lfsr(16, [1, 2, 4, 15]))
    save("lfsr_32bit", gen_lfsr(32, [1, 5, 6, 31]))

    print("\n--- Priority encoders ---")
    for n in [4, 8, 16, 32]:
        save(f"prienc_{n}input", gen_priority_encoder(n))

    print("\n--- Barrel shifters ---")
    for w in [8, 16, 32]:
        save(f"barrel_shift_{w}bit", gen_barrel_shifter(w))

    print(f"\n{'='*50}")
    print(f"Generated {len(generated)} circuits in {output_dir}")
    return generated

def main():
    ap = argparse.ArgumentParser(description="Generate missing circuit families")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    generate_all(Path(args.output), rng)

if __name__ == "__main__":
    main()

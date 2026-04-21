"""
Shared pytest fixtures for circuit complexity tests.

Each fixture writes a minimal Verilog netlist to a temporary file and
yields its path.  Temporary files are cleaned up automatically.
"""

import pytest
import tempfile
import os


# ---------------------------------------------------------------------------
# Verilog circuit fixtures
# ---------------------------------------------------------------------------

_INVERTER_CHAIN = """\
module inverter_chain (input A, output Z);
  wire b, c;
  INVX1 u1 (.A(A), .Y(b));
  INVX1 u2 (.A(b), .Y(c));
  INVX1 u3 (.A(c), .Y(Z));
endmodule
"""

_NAND_GATE = """\
module nand2 (input A, input B, output Z);
  NAND2X1 u1 (.A(A), .B(B), .Y(Z));
endmodule
"""

_SR_LATCH = """\
module sr_latch (input S, input R, output Q, output QB);
  NOR2X1 u1 (.A(S), .B(QB), .Y(Q));
  NOR2X1 u2 (.A(R), .B(Q), .Y(QB));
endmodule
"""

_SIMPLE_DFF = """\
module simple_dff (input D, input CLK, output Q);
  DFFPOSX1 u1 (.D(D), .CLK(CLK), .Q(Q));
endmodule
"""

_AND_OR_TREE = """\
module and_or_tree (input A, input B, input C, output Z);
  wire ab;
  AND2X1 u1 (.A(A), .B(B), .Y(ab));
  OR2X1  u2 (.A(ab), .B(C), .Y(Z));
endmodule
"""

_DEEP_CHAIN = """\
module deep_chain (input A, output Z);
  wire w1, w2, w3, w4, w5, w6, w7, w8, w9;
  INVX1 u1  (.A(A),  .Y(w1));
  BUFX2 u2  (.A(w1), .Y(w2));
  INVX1 u3  (.A(w2), .Y(w3));
  BUFX2 u4  (.A(w3), .Y(w4));
  INVX1 u5  (.A(w4), .Y(w5));
  BUFX2 u6  (.A(w5), .Y(w6));
  INVX1 u7  (.A(w6), .Y(w7));
  BUFX2 u8  (.A(w7), .Y(w8));
  INVX1 u9  (.A(w8), .Y(w9));
  BUFX2 u10 (.A(w9), .Y(Z));
endmodule
"""

# -- assign statement tests --------------------------------------------------

_WIRE_ASSIGN = """\
module wire_assign_test (input A, input B, output Z);
  wire ab;
  AND2X1 u1 (.A(A), .B(B), .Y(ab));
  assign Z = ab;
endmodule
"""

# Yosys LUT-style assign: 2-input lookup table
_LUT_ASSIGN_2IN = """\
module lut_2in_test (input A, input B, output Z);
  assign Z = 4'b1000 >> { B, A };
endmodule
"""

# Yosys LUT-style assign: 1-input (single-driver buf)
_LUT_ASSIGN_1IN = """\
module lut_1in_test (input A, output Z);
  assign Z = 1'b0 >> { A };
endmodule
"""

# -- skipped / unknown cell --------------------------------------------------

_UNKNOWN_CELL = """\
module unknown_cell_test (input A, output Z);
  wire mid;
  INVX1    u1 (.A(A),   .Y(mid));
  MYCELLX1 u2 (.IN(mid), .OUT(Z));
endmodule
"""

# -- new cell types ----------------------------------------------------------

_BUFX1_CIRCUIT = """\
module bufx1_test (input A, output Z);
  BUFX1 u1 (.A(A), .Y(Z));
endmodule
"""

_TBUFX1_CIRCUIT = """\
module tbufx1_test (input A, output Z);
  TBUFX1 u1 (.A(A), .Y(Z));
endmodule
"""

_SDFFSRX1_CIRCUIT = """\
module sdffsrx1_test (input D, input CLK, output Q);
  SDFFSRX1 u1 (.D(D), .CLK(CLK), .Q(Q));
endmodule
"""

# OAI33X1: OR(A0,A1,A2) AND OR(B0,B1,B2) then INVERT — 6 data inputs
_OAI33X1_CIRCUIT = """\
module oai33_test (A0, A1, A2, B0, B1, B2, Y);
  input A0, A1, A2, B0, B1, B2;
  output Y;
  OAI33X1 u1 (.A0(A0), .A1(A1), .A2(A2), .B0(B0), .B1(B1), .B2(B2), .Y(Y));
endmodule
"""

# -- circuit type classification ---------------------------------------------

# 1 DFF + 4 AND gates  →  seq_ratio = 1/5 = 0.20  →  'sequential'
_MIXED_SEQUENTIAL = """\
module mixed_seq (input D, input CLK, input A, input B, output Q, output Z);
  wire q1, w3, w4;
  DFFPOSX1 u1 (.D(D),  .CLK(CLK), .Q(q1));
  AND2X1   u2 (.A(A),  .B(B),     .Y(Z));
  AND2X1   u3 (.A(q1), .B(A),     .Y(w3));
  AND2X1   u4 (.A(B),  .B(D),     .Y(w4));
  AND2X1   u5 (.A(w3), .B(w4),    .Y(Q));
endmodule
"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _tmp_verilog(content: str):
    """Write *content* to a temporary .v file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".v")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Original fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def inverter_chain_v():
    path = _tmp_verilog(_INVERTER_CHAIN)
    yield path
    os.unlink(path)


@pytest.fixture
def nand_gate_v():
    path = _tmp_verilog(_NAND_GATE)
    yield path
    os.unlink(path)


@pytest.fixture
def sr_latch_v():
    path = _tmp_verilog(_SR_LATCH)
    yield path
    os.unlink(path)


@pytest.fixture
def simple_dff_v():
    path = _tmp_verilog(_SIMPLE_DFF)
    yield path
    os.unlink(path)


@pytest.fixture
def and_or_tree_v():
    path = _tmp_verilog(_AND_OR_TREE)
    yield path
    os.unlink(path)


@pytest.fixture
def deep_chain_v():
    path = _tmp_verilog(_DEEP_CHAIN)
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# New fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wire_assign_v():
    path = _tmp_verilog(_WIRE_ASSIGN)
    yield path
    os.unlink(path)


@pytest.fixture
def lut_assign_2in_v():
    path = _tmp_verilog(_LUT_ASSIGN_2IN)
    yield path
    os.unlink(path)


@pytest.fixture
def lut_assign_1in_v():
    path = _tmp_verilog(_LUT_ASSIGN_1IN)
    yield path
    os.unlink(path)


@pytest.fixture
def unknown_cell_v():
    path = _tmp_verilog(_UNKNOWN_CELL)
    yield path
    os.unlink(path)


@pytest.fixture
def bufx1_v():
    path = _tmp_verilog(_BUFX1_CIRCUIT)
    yield path
    os.unlink(path)


@pytest.fixture
def tbufx1_v():
    path = _tmp_verilog(_TBUFX1_CIRCUIT)
    yield path
    os.unlink(path)


@pytest.fixture
def sdffsrx1_v():
    path = _tmp_verilog(_SDFFSRX1_CIRCUIT)
    yield path
    os.unlink(path)


@pytest.fixture
def oai33x1_v():
    path = _tmp_verilog(_OAI33X1_CIRCUIT)
    yield path
    os.unlink(path)


@pytest.fixture
def mixed_sequential_v():
    path = _tmp_verilog(_MIXED_SEQUENTIAL)
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# Edge-case fixtures (EC1–EC7)
# ---------------------------------------------------------------------------

_EMPTY_MODULE = """\
module empty_mod (input A, output Z);
endmodule
"""

_DUPLICATE_INSTANCE = """\
module dup_test (input A, input B, output Z);
  wire w1, w2;
  INVX1 u1 (.A(A), .Y(w1));
  INVX1 u1 (.A(B), .Y(w2));
  AND2X1 u2 (.A(w1), .B(w2), .Y(Z));
endmodule
"""

_SELF_LOOP = """\
module self_loop_test (input A, output Z);
  wire w1;
  AND2X1 u1 (.A(A), .B(w1), .Y(w1));
  INVX1  u2 (.A(w1), .Y(Z));
endmodule
"""

_CYCLIC_ASSIGN = """\
module cyclic_assign_test (input A, output Z);
  wire x, y;
  assign x = y;
  assign y = x;
  INVX1 u1 (.A(A), .Y(Z));
endmodule
"""

_FLOATING_WIRE = """\
module floating_test (input A, output Z);
  wire used, unused_wire;
  INVX1 u1 (.A(A), .Y(used));
  INVX1 u2 (.A(used), .Y(Z));
endmodule
"""

_MULTI_MODULE = """\
module first_mod (input A, output Z);
  INVX1 u1 (.A(A), .Y(Z));
endmodule
module second_mod (input B, output W);
  BUFX1 u1 (.A(B), .Y(W));
endmodule
"""

_SEQUENTIAL_ONLY = """\
module seq_only (input D, input CLK, output Q1, output Q2);
  wire mid;
  DFFPOSX1 u1 (.D(D),   .CLK(CLK), .Q(mid));
  DFFPOSX1 u2 (.D(mid), .CLK(CLK), .Q(Q1));
  DFFPOSX1 u3 (.D(mid), .CLK(CLK), .Q(Q2));
endmodule
"""


@pytest.fixture
def empty_module_v():
    path = _tmp_verilog(_EMPTY_MODULE)
    yield path
    os.unlink(path)


@pytest.fixture
def duplicate_instance_v():
    path = _tmp_verilog(_DUPLICATE_INSTANCE)
    yield path
    os.unlink(path)


@pytest.fixture
def self_loop_v():
    path = _tmp_verilog(_SELF_LOOP)
    yield path
    os.unlink(path)


@pytest.fixture
def cyclic_assign_v():
    path = _tmp_verilog(_CYCLIC_ASSIGN)
    yield path
    os.unlink(path)


@pytest.fixture
def floating_wire_v():
    path = _tmp_verilog(_FLOATING_WIRE)
    yield path
    os.unlink(path)


@pytest.fixture
def multi_module_v():
    path = _tmp_verilog(_MULTI_MODULE)
    yield path
    os.unlink(path)


@pytest.fixture
def sequential_only_v():
    path = _tmp_verilog(_SEQUENTIAL_ONLY)
    yield path
    os.unlink(path)

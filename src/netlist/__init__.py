"""
netlist — Modular circuit analysis package.

Sub-modules:
  parser     – Verilog netlist parsing and graph construction
  metrics    – Structural complexity scoring (v4 formula)
  converter  – PyTorch Geometric graph conversion

The top-level netlist_parser.py re-exports GateLevelNetlistParser
and standalone_analysis for backward compatibility.
"""

from netlist.parser import STDCELL_MAP, parse_verilog_netlist
from netlist.metrics import (
    compute_structural_complexity,
    compute_complexity_metrics,
    compute_depth,
    compute_scc,
    compute_reconvergent_ratio,
    classify_circuit_type,
)
from netlist.converter import to_pytorch_geometric, standalone_analysis

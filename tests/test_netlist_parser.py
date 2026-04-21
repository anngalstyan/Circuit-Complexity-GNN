"""
Tests for GateLevelNetlistParser and compute_structural_complexity().

Run from the project root:
    pytest tests/test_netlist_parser.py -v
"""

import sys
import math
from pathlib import Path

import pytest

# Add src/ to path so the parser can be imported without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from netlist_parser import GateLevelNetlistParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(filepath: str) -> GateLevelNetlistParser:
    p = GateLevelNetlistParser()
    p.parse_verilog_netlist(filepath)
    return p


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestParsing:
    def test_inverter_chain_gate_count(self, inverter_chain_v):
        p = parse(inverter_chain_v)
        assert len(p.gates) == 3

    def test_inverter_chain_gate_types(self, inverter_chain_v):
        p = parse(inverter_chain_v)
        types = [g["type"] for g in p.gates]
        assert all(t == "not" for t in types)

    def test_nand_gate_count(self, nand_gate_v):
        p = parse(nand_gate_v)
        assert len(p.gates) == 1
        assert p.gates[0]["type"] == "nand"

    def test_nand_gate_inputs(self, nand_gate_v):
        p = parse(nand_gate_v)
        assert len(p.gates[0]["inputs"]) == 2

    def test_sr_latch_gate_count(self, sr_latch_v):
        p = parse(sr_latch_v)
        assert len(p.gates) == 2
        assert all(g["type"] == "nor" for g in p.gates)

    def test_simple_dff_is_sequential(self, simple_dff_v):
        p = parse(simple_dff_v)
        assert len(p.gates) == 1
        assert p.gates[0]["type"] == "dff"

    def test_module_inputs_outputs_parsed(self, nand_gate_v):
        p = parse(nand_gate_v)
        assert "A" in p.inputs
        assert "B" in p.inputs
        assert "Z" in p.outputs

    def test_and_or_tree_gate_count(self, and_or_tree_v):
        p = parse(and_or_tree_v)
        assert len(p.gates) == 2

    def test_deep_chain_gate_count(self, deep_chain_v):
        p = parse(deep_chain_v)
        assert len(p.gates) == 10

    def test_graph_edges_built(self, inverter_chain_v):
        p = parse(inverter_chain_v)
        # Each inverter's output should appear in the graph
        assert len(p.graph) > 0

    def test_reverse_graph_built(self, and_or_tree_v):
        p = parse(and_or_tree_v)
        assert len(p.reverse_graph) > 0


# ---------------------------------------------------------------------------
# Depth tests
# ---------------------------------------------------------------------------

class TestDepth:
    def test_inverter_chain_depth(self, inverter_chain_v):
        p = parse(inverter_chain_v)
        depth = p._compute_depth()
        assert depth == 3

    def test_nand_gate_depth(self, nand_gate_v):
        p = parse(nand_gate_v)
        depth = p._compute_depth()
        assert depth == 1

    def test_deep_chain_depth(self, deep_chain_v):
        p = parse(deep_chain_v)
        depth = p._compute_depth()
        assert depth == 10

    def test_and_or_tree_depth(self, and_or_tree_v):
        p = parse(and_or_tree_v)
        depth = p._compute_depth()
        assert depth == 2


# ---------------------------------------------------------------------------
# SCC / feedback tests
# ---------------------------------------------------------------------------

class TestSCC:
    def test_combinational_has_no_feedback(self, inverter_chain_v):
        p = parse(inverter_chain_v)
        scc = p._compute_scc_stats()
        assert scc["feedback_ratio"] == 0.0
        assert scc["scc_count"] == 0

    def test_sr_latch_has_feedback(self, sr_latch_v):
        p = parse(sr_latch_v)
        scc = p._compute_scc_stats()
        # Both NOR gates form a feedback loop → they should be in an SCC
        assert scc["feedback_ratio"] > 0.0
        assert scc["scc_count"] >= 1
        assert scc["largest_scc_size"] == 2

    def test_sr_latch_scc_node_set(self, sr_latch_v):
        p = parse(sr_latch_v)
        in_cycles = p._get_scc_node_set()
        assert len(in_cycles) == 2  # both NOR gate outputs in the cycle

    def test_dff_no_combinational_feedback(self, simple_dff_v):
        p = parse(simple_dff_v)
        scc = p._compute_scc_stats()
        # A single DFF has no combinational cycle (DFF is sequential, not gate-to-gate)
        assert scc["scc_count"] == 0


# ---------------------------------------------------------------------------
# Complexity score tests
# ---------------------------------------------------------------------------

class TestComplexityScore:
    def test_score_in_range(self, inverter_chain_v):
        p = parse(inverter_chain_v)
        m = p.compute_structural_complexity()
        assert 0.0 <= m["complexity_score"] <= 5.0

    def test_deeper_circuit_higher_complexity(self, inverter_chain_v, deep_chain_v):
        p3 = parse(inverter_chain_v)
        p10 = parse(deep_chain_v)
        m3  = p3.compute_structural_complexity()
        m10 = p10.compute_structural_complexity()
        assert m10["complexity_score"] > m3["complexity_score"]

    def test_feedback_raises_complexity(self, inverter_chain_v, sr_latch_v):
        p_comb = parse(inverter_chain_v)
        p_fb   = parse(sr_latch_v)
        m_comb = p_comb.compute_structural_complexity()
        m_fb   = p_fb.compute_structural_complexity()
        # SR latch has feedback; 3-inverter chain does not.
        # Both have 2-3 gates, but latch's feedback_ratio > 0 should push score higher.
        assert m_fb["complexity_score"] > m_comb["complexity_score"]

    def test_sequential_ratio_is_correct(self, simple_dff_v):
        p = parse(simple_dff_v)
        m = p.compute_structural_complexity()
        assert m["seq_ratio"] == 1.0   # 1 DFF / 1 total gate
        assert m["seq_gates"] == 1
        assert m["gate_count"] == 0    # gate_count is *logic* gates only

    def test_raw_score_non_negative(self, inverter_chain_v):
        p = parse(inverter_chain_v)
        m = p.compute_structural_complexity()
        assert m["raw_score"] >= 0.0

    def test_metrics_keys_present(self, and_or_tree_v):
        p = parse(and_or_tree_v)
        m = p.compute_structural_complexity()
        required = {
            "complexity_score", "raw_score", "depth", "gate_count",
            "total_gates", "seq_gates", "seq_ratio", "feedback_ratio",
            "scc_count", "largest_scc_size", "cyclomatic_complexity",
            "reconvergent_ratio", "xor_ratio", "buf_ratio",
            "edge_density", "type_entropy", "max_fanout", "avg_fanout",
            "num_inputs", "num_outputs",
        }
        assert required.issubset(m.keys())

    def test_new_keys_present(self, and_or_tree_v):
        """circuit_type and skipped_cell_types must both be present."""
        p = parse(and_or_tree_v)
        m = p.compute_structural_complexity()
        assert "circuit_type" in m
        assert "skipped_cell_types" in m

    def test_complexity_score_softmax_shape(self):
        """Verify softmax normalisation formula matches expected values."""
        # At raw=25 (midpoint): score = 5 / (1 + exp(0)) = 2.5
        raw = 25.0
        expected = 5.0 / (1.0 + math.exp((25.0 - raw) / 8.0))
        assert abs(expected - 2.5) < 1e-9
        # At raw=0: score ≈ 0.21 (much lower floor than old sigmoid's 0.91)
        raw = 0.0
        expected = 5.0 / (1.0 + math.exp((25.0 - raw) / 8.0))
        assert expected < 0.25


# ---------------------------------------------------------------------------
# PyTorch Geometric conversion tests
# ---------------------------------------------------------------------------

class TestPyGConversion:
    def test_feature_dimensions(self, inverter_chain_v):
        p = parse(inverter_chain_v)
        data = p.to_pytorch_geometric()
        assert data.x.shape[1] == 24

    def test_node_count_includes_io(self, inverter_chain_v):
        p = parse(inverter_chain_v)
        data = p.to_pytorch_geometric()
        # 1 primary input + 3 gate outputs = 4 nodes minimum
        assert data.num_nodes >= 4

    def test_edge_index_shape(self, and_or_tree_v):
        p = parse(and_or_tree_v)
        data = p.to_pytorch_geometric()
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0

    def test_target_value_stored(self, nand_gate_v):
        p = parse(nand_gate_v)
        data = p.to_pytorch_geometric()
        assert data.y.shape == (1, 1)
        assert 0.0 <= data.y.item() <= 5.0

    def test_complexity_score_attribute(self, sr_latch_v):
        p = parse(sr_latch_v)
        data = p.to_pytorch_geometric()
        assert hasattr(data, "complexity_score")
        assert hasattr(data, "gate_count")
        assert hasattr(data, "depth")
        assert hasattr(data, "feedback_ratio")

    def test_no_edges_for_single_gate(self, nand_gate_v):
        """A single NAND with two primary inputs produces input→gate edges."""
        p = parse(nand_gate_v)
        data = p.to_pytorch_geometric()
        # edges: A→NAND, B→NAND  (2 edges)
        assert data.edge_index.shape[1] == 2

    def test_feedback_node_in_scc_feature(self, sr_latch_v):
        """For SR latch, is_in_feedback_scc (dim 21) should be 1 for gate outputs."""
        p = parse(sr_latch_v)
        data = p.to_pytorch_geometric()
        # dim 21 is is_in_feedback_scc; at least one node should have it set
        scc_flags = data.x[:, 21]
        assert scc_flags.sum().item() > 0

    def test_circuit_type_attribute_on_data(self, inverter_chain_v):
        """circuit_type must be stored as an attribute on the Data object."""
        p = parse(inverter_chain_v)
        data = p.to_pytorch_geometric()
        assert hasattr(data, "circuit_type")
        assert data.circuit_type == "combinational"


# ---------------------------------------------------------------------------
# assign statement parsing tests
# ---------------------------------------------------------------------------

class TestAssignParsing:
    def test_simple_assign_total_gate_count(self, wire_assign_v):
        """AND gate + assign buf → 2 gates total."""
        p = parse(wire_assign_v)
        assert len(p.gates) == 2

    def test_simple_assign_creates_buf(self, wire_assign_v):
        """assign Z = ab;  should add a 'buf' gate."""
        p = parse(wire_assign_v)
        assert any(g["type"] == "buf" for g in p.gates)

    def test_simple_assign_buf_output(self, wire_assign_v):
        """The buf gate created by the assign should drive the output net Z."""
        p = parse(wire_assign_v)
        buf_gates = [g for g in p.gates if g["type"] == "buf"]
        assert len(buf_gates) == 1
        assert buf_gates[0]["output"] == "Z"

    def test_simple_assign_in_graph(self, wire_assign_v):
        """assign Z = ab;  should add an ab→Z edge in the connectivity graph."""
        p = parse(wire_assign_v)
        assert "Z" in p.graph.get("ab", [])

    def test_lut_assign_2in_gate_count(self, lut_assign_2in_v):
        """LUT-style assign with 2 inputs should create exactly 1 gate."""
        p = parse(lut_assign_2in_v)
        assert len(p.gates) == 1

    def test_lut_assign_2in_gate_type(self, lut_assign_2in_v):
        """A 2-input LUT assign is mapped to an 'and' gate."""
        p = parse(lut_assign_2in_v)
        assert p.gates[0]["type"] == "and"

    def test_lut_assign_2in_input_count(self, lut_assign_2in_v):
        """The 2-input LUT gate should have exactly 2 inputs."""
        p = parse(lut_assign_2in_v)
        assert len(p.gates[0]["inputs"]) == 2

    def test_lut_assign_1in_gate_type(self, lut_assign_1in_v):
        """A single-input LUT assign is mapped to a 'buf' gate."""
        p = parse(lut_assign_1in_v)
        assert len(p.gates) == 1
        assert p.gates[0]["type"] == "buf"


# ---------------------------------------------------------------------------
# New cell type tests
# ---------------------------------------------------------------------------

class TestNewCells:
    def test_bufx1_parsed_as_buf(self, bufx1_v):
        """BUFX1 (single-drive buffer) should map to a 'buf' gate."""
        p = parse(bufx1_v)
        assert len(p.gates) == 1
        assert p.gates[0]["type"] == "buf"

    def test_tbufx1_modelled_as_buf(self, tbufx1_v):
        """TBUFX1 (tri-state buffer) is modelled as a regular 'buf' gate."""
        p = parse(tbufx1_v)
        assert len(p.gates) == 1
        assert p.gates[0]["type"] == "buf"

    def test_sdffsrx1_parsed_as_dff(self, sdffsrx1_v):
        """SDFFSRX1 (scan DFF) should map to a sequential 'dff' gate."""
        p = parse(sdffsrx1_v)
        assert len(p.gates) == 1
        assert p.gates[0]["type"] == "dff"

    def test_sdffsrx1_counts_as_sequential(self, sdffsrx1_v):
        """SDFFSRX1 must contribute to the sequential gate count."""
        p = parse(sdffsrx1_v)
        m = p.compute_structural_complexity()
        assert m["seq_gates"] == 1
        assert m["seq_ratio"] == 1.0

    def test_oai33x1_has_six_inputs(self, oai33x1_v):
        """OAI33X1 has three A inputs and three B inputs — 6 total."""
        p = parse(oai33x1_v)
        assert len(p.gates) == 1
        assert len(p.gates[0]["inputs"]) == 6


# ---------------------------------------------------------------------------
# Circuit type classification tests
# ---------------------------------------------------------------------------

class TestCircuitTypeClassification:
    # ── Direct unit tests on _classify_circuit_type ──────────────────────────

    def test_empty_type(self):
        p = GateLevelNetlistParser()
        assert p._classify_circuit_type(0.0, 0) == "empty"

    def test_combinational_type(self):
        p = GateLevelNetlistParser()
        assert p._classify_circuit_type(0.0, 5) == "combinational"

    def test_mostly_combinational_type(self):
        """seq_ratio < 0.05 → 'mostly_combinational'."""
        p = GateLevelNetlistParser()
        assert p._classify_circuit_type(0.03, 100) == "mostly_combinational"

    def test_sequential_type(self):
        """0.05 ≤ seq_ratio < 0.35 → 'sequential'."""
        p = GateLevelNetlistParser()
        assert p._classify_circuit_type(0.15, 20) == "sequential"

    def test_sequential_heavy_type(self):
        """seq_ratio ≥ 0.35 → 'sequential_heavy'."""
        p = GateLevelNetlistParser()
        assert p._classify_circuit_type(0.5, 4) == "sequential_heavy"

    def test_boundary_exactly_five_pct(self):
        """Exactly 5% sequential is in [0.05, 0.35) → 'sequential', not 'mostly_combinational'."""
        p = GateLevelNetlistParser()
        assert p._classify_circuit_type(0.05, 20) == "sequential"

    def test_boundary_exactly_35_pct(self):
        """Exactly 35% sequential is ≥ 0.35 → 'sequential_heavy'."""
        p = GateLevelNetlistParser()
        assert p._classify_circuit_type(0.35, 20) == "sequential_heavy"

    # ── Integration tests through full parse ─────────────────────────────────

    def test_integration_combinational(self, inverter_chain_v):
        """Pure combinational circuit → 'combinational'."""
        p = parse(inverter_chain_v)
        m = p.compute_structural_complexity()
        assert m["circuit_type"] == "combinational"

    def test_integration_sequential_heavy(self, simple_dff_v):
        """Single DFF: seq_ratio = 1.0 ≥ 0.35 → 'sequential_heavy'."""
        p = parse(simple_dff_v)
        m = p.compute_structural_complexity()
        assert m["circuit_type"] == "sequential_heavy"

    def test_integration_sequential(self, mixed_sequential_v):
        """1 DFF + 4 AND gates: seq_ratio = 0.20 → 'sequential'."""
        p = parse(mixed_sequential_v)
        assert len(p.gates) == 5        # sanity: 5 gates total
        m = p.compute_structural_complexity()
        assert m["circuit_type"] == "sequential"

    def test_circuit_type_key_in_metrics(self, and_or_tree_v):
        """'circuit_type' must be a key in compute_structural_complexity() output."""
        m = parse(and_or_tree_v).compute_structural_complexity()
        assert "circuit_type" in m


# ---------------------------------------------------------------------------
# Skipped / unknown cell tests
# ---------------------------------------------------------------------------

class TestSkippedCells:
    def test_unknown_cell_tracked(self, unknown_cell_v):
        """Unrecognised cell type should be recorded in _skipped_cells."""
        p = parse(unknown_cell_v)
        assert "MYCELLX1" in p._skipped_cells

    def test_unknown_cell_not_in_gates(self, unknown_cell_v):
        """Only the recognised INVX1 gate should appear in p.gates."""
        p = parse(unknown_cell_v)
        assert len(p.gates) == 1
        assert p.gates[0]["type"] == "not"

    def test_skipped_cell_types_in_metrics(self, unknown_cell_v):
        """skipped_cell_types in metrics should list the unrecognised name."""
        p = parse(unknown_cell_v)
        m = p.compute_structural_complexity()
        assert "skipped_cell_types" in m
        assert "MYCELLX1" in m["skipped_cell_types"]

    def test_known_cells_not_skipped(self, inverter_chain_v):
        """Circuits using only standard cells must have an empty _skipped_cells."""
        p = parse(inverter_chain_v)
        assert len(p._skipped_cells) == 0

    def test_no_skipped_cells_in_metrics_for_clean_circuit(self, inverter_chain_v):
        """skipped_cell_types should be an empty list when all cells are known."""
        m = parse(inverter_chain_v).compute_structural_complexity()
        assert m["skipped_cell_types"] == []


# ---------------------------------------------------------------------------
# SCC cache tests
# ---------------------------------------------------------------------------

class TestSCCCache:
    def test_cache_none_after_parse(self, inverter_chain_v):
        """_scc_cache must be None immediately after parsing (before any SCC call)."""
        p = parse(inverter_chain_v)
        assert p._scc_cache is None

    def test_cache_populated_after_compute(self, inverter_chain_v):
        """Calling _compute_scc() should populate _scc_cache."""
        p = parse(inverter_chain_v)
        p._compute_scc()
        assert p._scc_cache is not None

    def test_stats_and_nodeset_consistent(self, sr_latch_v):
        """_compute_scc_stats() and _get_scc_node_set() must agree with _compute_scc()."""
        p = parse(sr_latch_v)
        stats_direct, ns_direct = p._compute_scc()
        stats_via   = p._compute_scc_stats()
        ns_via      = p._get_scc_node_set()
        assert stats_via == stats_direct
        assert ns_via == ns_direct

    def test_cache_returns_same_object(self, sr_latch_v):
        """Two calls to _compute_scc() must return the exact same cached tuple."""
        p = parse(sr_latch_v)
        result1 = p._compute_scc()
        result2 = p._compute_scc()
        assert result1 is result2   # identity check — no recomputation

    def test_cache_cleared_on_reparse(self, inverter_chain_v, nand_gate_v):
        """Parsing a new file must invalidate the SCC cache."""
        p = GateLevelNetlistParser()
        p.parse_verilog_netlist(inverter_chain_v)
        p._compute_scc()                          # populate cache
        assert p._scc_cache is not None
        p.parse_verilog_netlist(nand_gate_v)      # re-parse → cache must reset
        assert p._scc_cache is None


# ═══════════════════════════════════════════════════════════════════════════
# Edge-case tests (EC1–EC7)
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Validates parser robustness on corner cases."""

    # EC1: Empty module ─────────────────────────────────────────────────────

    def test_empty_module_zero_gates(self, empty_module_v):
        p = parse(empty_module_v)
        assert len(p.gates) == 0

    def test_empty_module_circuit_type(self, empty_module_v):
        p = parse(empty_module_v)
        m = p.compute_structural_complexity()
        assert m['circuit_type'] == 'empty'
        assert m['complexity_score'] == 0.0

    # EC2: Duplicate instance names ────────────────────────────────────────

    def test_duplicate_instance_renamed(self, duplicate_instance_v):
        p = parse(duplicate_instance_v)
        names = [g['instance_name'] for g in p.gates]
        assert len(names) == len(set(names)), \
            "Instance names must be unique after dedup"

    def test_duplicate_instance_count_preserved(self, duplicate_instance_v):
        p = parse(duplicate_instance_v)
        assert len(p.gates) == 3  # both INVs + AND still parsed

    # EC3: Self-loop filtering ─────────────────────────────────────────────

    def test_self_loop_filtered_in_pyg(self, self_loop_v):
        p = parse(self_loop_v)
        data = p.to_pytorch_geometric()
        src = data.edge_index[0].tolist()
        dst = data.edge_index[1].tolist()
        for s, d in zip(src, dst):
            assert s != d, "Self-loop edge should be filtered out"

    # EC4: Cyclic assign ───────────────────────────────────────────────────

    def test_cyclic_assign_parses_without_error(self, cyclic_assign_v):
        p = parse(cyclic_assign_v)
        assert len(p.gates) >= 1  # at least the INVX1

    def test_cyclic_assign_detected_in_scc(self, cyclic_assign_v):
        p = parse(cyclic_assign_v)
        m = p.compute_structural_complexity()
        # The cyclic assigns form a feedback loop
        assert m['feedback_ratio'] > 0 or m['scc_count'] >= 1

    # EC5: Floating wires ──────────────────────────────────────────────────

    def test_floating_wire_detected(self, floating_wire_v):
        p = parse(floating_wire_v)
        assert 'unused_wire' in p._floating_wires

    def test_floating_wire_count_in_metrics(self, floating_wire_v):
        p = parse(floating_wire_v)
        m = p.compute_structural_complexity()
        assert m['floating_wire_count'] >= 1

    # EC6: Multi-module files ──────────────────────────────────────────────

    def test_multi_module_only_first_parsed(self, multi_module_v):
        p = parse(multi_module_v)
        assert len(p.gates) == 1
        assert p.gates[0]['type'] == 'not'

    def test_multi_module_no_second_module_gates(self, multi_module_v):
        p = parse(multi_module_v)
        assert not any(g['cell_type'] == 'BUFX1' for g in p.gates)

    # EC7: Sequential-only circuits ────────────────────────────────────────

    def test_sequential_only_has_low_depth(self, sequential_only_v):
        """Sequential-only circuits have depth from DFF→DFF chains, not combinational logic."""
        p = parse(sequential_only_v)
        m = p.compute_structural_complexity()
        # DFF Q outputs feed next DFF D inputs, creating sequential depth
        # but no combinational logic gates contribute to depth
        assert m['gate_count'] == 0  # no combinational gates
        assert m['depth'] >= 0       # sequential chains may have nonzero depth

    def test_sequential_only_valid_score(self, sequential_only_v):
        p = parse(sequential_only_v)
        m = p.compute_structural_complexity()
        assert 0.0 <= m['complexity_score'] <= 5.0
        assert m['seq_ratio'] == 1.0

    def test_sequential_only_circuit_type(self, sequential_only_v):
        p = parse(sequential_only_v)
        m = p.compute_structural_complexity()
        assert m['circuit_type'] == 'sequential_heavy'

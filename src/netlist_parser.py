
import logging
from collections import defaultdict

from netlist.parser import STDCELL_MAP, parse_verilog_netlist
from netlist.metrics import (
    compute_structural_complexity as _compute_structural_complexity,
    compute_complexity_metrics as _compute_complexity_metrics,
    compute_depth as _compute_depth,
    compute_scc as _compute_scc,
    compute_reconvergent_ratio as _compute_reconvergent_ratio,
    classify_circuit_type as _classify_circuit_type,
)
from netlist.converter import (
    to_pytorch_geometric as _to_pytorch_geometric,
    standalone_analysis,
)

logger = logging.getLogger(__name__)

class GateLevelNetlistParser:
    """Stateful wrapper around the ``netlist`` package.

    Preserves the original class-based API where you call
    ``parse_verilog_netlist(path)`` first, then ``compute_structural_complexity()``,
    ``to_pytorch_geometric()``, etc.  Internally every method delegates to
    the pure functions in ``netlist.parser``, ``netlist.metrics``, and
    ``netlist.converter``.
    """

    def __init__(self):
        self.stdcell_map = dict(STDCELL_MAP)
        self.gates = []
        self.inputs = []
        self.outputs = []
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)
        self.verilog_text = ""
        self._scc_cache = None
        self._skipped_cells: set = set()
        self._floating_wires: set = set()

    def parse_verilog_netlist(self, filepath):
        result = parse_verilog_netlist(filepath, stdcell_map=self.stdcell_map)
        self.gates = result['gates']
        self.inputs = result['inputs']
        self.outputs = result['outputs']
        self.graph = result['graph']
        self.reverse_graph = result['reverse_graph']
        self.verilog_text = result['verilog_text']
        self._skipped_cells = result['skipped_cells']
        self._floating_wires = result.get('floating_wires', set())
        self._scc_cache = None

    def _compute_depth(self):
        return _compute_depth(self.gates, self.inputs, self.graph)

    def _compute_scc(self):
        if self._scc_cache is not None:
            return self._scc_cache
        self._scc_cache = _compute_scc(self.gates, self.graph)
        return self._scc_cache

    def _compute_scc_stats(self):
        stats, _ = self._compute_scc()
        return stats

    def _get_scc_node_set(self):
        _, node_set = self._compute_scc()
        return node_set

    def _compute_reconvergent_ratio(self):
        return _compute_reconvergent_ratio(self.gates, self.reverse_graph)

    def _classify_circuit_type(self, seq_ratio, total_gates):
        return _classify_circuit_type(seq_ratio, total_gates)

    def compute_structural_complexity(self):
        return _compute_structural_complexity(
            self.gates, self.inputs, self.outputs,
            self.graph, self.reverse_graph, self._skipped_cells,
            floating_wires=self._floating_wires,
        )

    def compute_complexity_metrics(self):
        return _compute_complexity_metrics(self.gates, self.inputs, self.graph)

    def to_pytorch_geometric(self, target_metric='complexity_score'):
        return _to_pytorch_geometric(
            self.gates, self.inputs, self.outputs,
            self.graph, self.reverse_graph, self._skipped_cells,
            target_metric=target_metric,
        )

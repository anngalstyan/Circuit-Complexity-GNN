"""
netlist.converter — PyTorch Geometric graph conversion.

Responsibilities:
  - Convert parsed netlist data into a PyG ``Data`` object
  - Node feature engineering (24-dim: 16 one-hot + 8 structural)
  - Global feature vector (10-dim raw structural metrics)
  - ``standalone_analysis()`` subprocess-friendly entry point

This module contains NO parsing logic and NO scoring formula logic.
It imports those from ``netlist.parser`` and ``netlist.metrics``.
"""

import os
import math
import logging
from collections import defaultdict

import numpy as np
import torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from torch_geometric.data import Data

from netlist.parser import parse_verilog_netlist
from netlist.metrics import (
    compute_structural_complexity,
    compute_scc,
)

logger = logging.getLogger(__name__)

def to_pytorch_geometric(gates, inputs, outputs, graph, reverse_graph,
                         skipped_cells=None, target_metric='complexity_score',
                         precomputed_struct=None, precomputed_scc_nodes=None):
    """
    Convert parsed netlist to a PyG Data object.

    Node features are 24-dimensional:
      dims  0-15: one-hot gate type (16 classes)
      dims 16-23: per-node local + minimal global context

        16: fanin  / 5.0          local fan-in
        17: fanout / 5.0          local fan-out
        18: is_primary_input      boundary flag (0/1)
        19: is_primary_output     boundary flag (0/1)
        20: is_sequential         DFF or latch node (0/1)
        21: is_in_feedback_scc    participates in a cyclic SCC (0/1)
        22: log10(total_gates+1)  circuit scale
        23: seq_ratio             fraction of sequential gates

    Parameters
    ----------
    gates, inputs, outputs, graph, reverse_graph : from parse_verilog_netlist()
    skipped_cells : set, optional
    target_metric : str
        'complexity_score' | 'composite' | 'gate_count' | 'depth'
    precomputed_struct : dict, optional
        Pre-computed structural metrics (avoids redundant recomputation).
    precomputed_scc_nodes : set, optional
        Pre-computed SCC node set (avoids redundant Kosaraju pass).

    Returns
    -------
    torch_geometric.data.Data
    """
    if skipped_cells is None:
        skipped_cells = set()

    if precomputed_struct is not None:
        struct = precomputed_struct
    else:
        struct = compute_structural_complexity(
            gates, inputs, outputs, graph, reverse_graph, skipped_cells
        )

    if target_metric == 'complexity_score':
        y_val = struct['complexity_score']
    elif target_metric == 'composite':
        y_val = struct['depth'] * math.log10(max(struct['gate_count'], 1))
    elif target_metric == 'gate_count':
        y_val = math.log10(struct['gate_count'] + 1)
    elif target_metric == 'depth':
        y_val = struct['depth'] / 100.0
    else:
        y_val = struct['complexity_score']

    if precomputed_scc_nodes is not None:
        scc_nodes = precomputed_scc_nodes
    else:
        _, scc_nodes = compute_scc(gates, graph)
    primary_inputs  = set(inputs)
    primary_outputs = set(outputs)
    seq_types       = {'dff', 'latch'}

    fanin_count  = defaultdict(int)
    fanout_count = defaultdict(int)
    for g in gates:
        fanout_count[g['output']] += 0
        for inp in g['inputs']:
            fanin_count[g['output']] += 1
            fanout_count[inp]        += 1

    type_to_idx = {
        'input': 0, 'output': 1, 'and': 2, 'or': 3, 'not': 4, 'nand': 5,
        'nor': 6, 'xor': 7, 'xnor': 8, 'buf': 9, 'dff': 10, 'aoi21': 11,
        'aoi22': 12, 'oai21': 13, 'oai22': 14, 'latch': 15
    }

    log_total = math.log10(max(struct['total_gates'], 1) + 1)
    seq_ratio = struct['seq_ratio']

    node_id  = {}
    feat_rows = []

    def _add_node(name, type_idx, is_seq):
        if name in node_id:
            return
        node_id[name] = len(node_id)
        onehot = [0.0] * 16
        onehot[type_idx] = 1.0
        fanin  = min(fanin_count.get(name, 0)  / 5.0, 5.0)
        fanout = min(fanout_count.get(name, 0) / 5.0, 5.0)
        feat_rows.append(onehot + [
            fanin, fanout,
            1.0 if name in primary_inputs  else 0.0,
            1.0 if name in primary_outputs else 0.0,
            1.0 if is_seq                  else 0.0,
            1.0 if name in scc_nodes       else 0.0,
            log_total, seq_ratio,
        ])

    for n in inputs:
        _add_node(n, 0, False)
    for g in gates:
        _add_node(g['output'], type_to_idx.get(g['type'], 2),
                  g['type'] in seq_types)
    for n in outputs:
        _add_node(n, 1, False)

    src_list = []
    dst_list = []
    self_loop_count = 0
    for g in gates:
        dst = node_id[g['output']]
        for inp in g['inputs']:
            if inp in node_id:
                src = node_id[inp]
                if src == dst:
                    self_loop_count += 1
                    continue
                src_list.append(src)
                dst_list.append(dst)
    if self_loop_count:
        logger.warning("Filtered %d self-loop edge(s) during graph construction",
                       self_loop_count)

    x = torch.from_numpy(np.array(feat_rows, dtype=np.float32))
    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    y = torch.tensor([[y_val]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes        = len(node_id)
    data.gate_count       = struct['gate_count']
    data.depth            = struct['depth']
    data.feedback_ratio   = struct['feedback_ratio']
    data.complexity_score = struct['complexity_score']
    data.circuit_type     = struct['circuit_type']

    _N = max(struct['total_gates'], 1)

    data.global_feats = torch.tensor([
        math.log10(_N + 1) / 5.0,
        struct['depth'] / 50.0,
        struct['feedback_ratio'],
        struct['seq_ratio'],
        struct['largest_scc_size'] / _N,
        struct['xor_ratio'],
        struct['reconvergent_ratio'],
        struct['type_entropy'] / 4.0,
        struct['avg_fanout'] / 5.0,
        struct['edge_density'] / 3.0,
    ], dtype=torch.float)

    return data

def standalone_analysis(filepath):
    """Run full parse → metrics → PyG conversion in one call.

    Designed for ``ProcessPoolExecutor`` so the heavy pure-Python work
    runs in a separate process, avoiding GIL contention with the GUI.
    """
    import time as _time
    t0 = _time.time()

    parsed = parse_verilog_netlist(filepath)
    t_parse = _time.time()

    metrics = compute_structural_complexity(
        parsed['gates'], parsed['inputs'], parsed['outputs'],
        parsed['graph'], parsed['reverse_graph'], parsed['skipped_cells'],
    )
    t_metrics = _time.time()

    _, scc_nodes = compute_scc(parsed['gates'], parsed['graph'])
    data = to_pytorch_geometric(
        parsed['gates'], parsed['inputs'], parsed['outputs'],
        parsed['graph'], parsed['reverse_graph'], parsed['skipped_cells'],
        target_metric='complexity_score',
        precomputed_struct=metrics,
        precomputed_scc_nodes=scc_nodes,
    )
    t_graph = _time.time()

    return {
        'metrics': metrics,
        'data': data,
        'fast_path': False,  # always build full PyG graph
        'gates_slice': parsed['gates'][:500],
        'inputs': list(parsed['inputs']),
        'outputs': list(parsed['outputs']),
        'parse_ms': (t_parse - t0) * 1000,
        'metrics_ms': (t_metrics - t_parse) * 1000,
        'graph_ms': (t_graph - t_metrics) * 1000,
    }

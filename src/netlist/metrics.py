"""
netlist.metrics — Structural complexity scoring (v4 formula).

Responsibilities:
  - BFS depth computation
  - Kosaraju SCC decomposition (feedback detection)
  - Reconvergent fanout ratio
  - McCabe cyclomatic complexity
  - Full v4 structural complexity formula
  - Circuit type classification

All functions are stateless and operate on the parsed result dict
returned by ``netlist.parser.parse_verilog_netlist()``.
"""

import math
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

def compute_depth(gates, inputs, graph):
    """Longest topological path from primary inputs / sequential outputs.

    Parameters
    ----------
    gates : list[dict]
        Gate dicts with 'type', 'inputs', 'output' keys.
    inputs : list[str]
        Primary input net names.
    graph : dict[str, list[str]]
        Forward adjacency  (net → list of driven nets).

    Returns
    -------
    int
        Maximum combinational depth.
    """
    indeg = defaultdict(int)
    for g in gates:
        indeg[g['output']] += len(g['inputs'])

    seq_outputs = [g['output'] for g in gates if g['type'] in ('dff', 'latch')]
    sources = list(inputs) + seq_outputs
    q = deque(n for n in sources if graph[n])
    level = {n: 0 for n in q}
    max_d = 0
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            level[nxt] = max(level.get(nxt, 0), level[cur] + 1)
            max_d = max(max_d, level[nxt])
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    return max_d

def compute_scc(gates, graph):
    """Kosaraju two-pass SCC decomposition.

    Returns
    -------
    tuple[dict, set]
        (stats_dict, node_set) where *stats_dict* contains:
          feedback_ratio, scc_count, largest_scc_size,
          total_nodes_in_sccs, cyclomatic_complexity
        and *node_set* is the set of gate-output names sitting inside
        a non-trivial SCC.
    """
    empty_stats = {
        'feedback_ratio':       0.0,
        'scc_count':            0,
        'largest_scc_size':     0,
        'total_nodes_in_sccs':  0,
        'cyclomatic_complexity': 0,
    }
    if not gates:
        return (empty_stats, set())

    gate_outputs = {g['output'] for g in gates}
    nodes = list(gate_outputs)
    n = len(nodes)
    if n == 0:
        return (empty_stats, set())

    idx = {node: i for i, node in enumerate(nodes)}

    fwd = [[] for _ in range(n)]
    bwd = [[] for _ in range(n)]
    for g in gates:
        u = g['output']
        for v in graph.get(u, []):
            if v in gate_outputs:
                fwd[idx[u]].append(idx[v])
                bwd[idx[v]].append(idx[u])

    visited = [False] * n
    order = []

    def dfs1(v):
        stack = [(v, iter(fwd[v]))]
        visited[v] = True
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if not visited[child]:
                    visited[child] = True
                    stack.append((child, iter(fwd[child])))
            except StopIteration:
                order.append(node)
                stack.pop()

    for i in range(n):
        if not visited[i]:
            dfs1(i)

    visited2 = [False] * n
    sccs = []

    def dfs2(start, scc):
        stack = [start]
        visited2[start] = True
        scc.append(start)
        while stack:
            node = stack.pop()
            for child in bwd[node]:
                if not visited2[child]:
                    visited2[child] = True
                    scc.append(child)
                    stack.append(child)

    for v in reversed(order):
        if not visited2[v]:
            scc = []
            dfs2(v, scc)
            sccs.append(scc)

    non_trivial = [s for s in sccs if len(s) > 1]
    total_in    = sum(len(s) for s in non_trivial)
    largest     = max((len(s) for s in non_trivial), default=0)
    E           = sum(len(row) for row in fwd)
    M           = E - n + 2

    stats = {
        'feedback_ratio':       total_in / n,
        'scc_count':            len(non_trivial),
        'largest_scc_size':     largest,
        'total_nodes_in_sccs':  total_in,
        'cyclomatic_complexity': max(M, 0),
    }
    node_set = {nodes[i] for s in non_trivial for i in s}

    return (stats, node_set)

def compute_reconvergent_ratio(gates, reverse_graph):
    """Fraction of gate nodes with multiple distinct fanin paths from inputs."""
    if not gates:
        return 0.0
    reconvergent = 0
    rev = reverse_graph
    for g in gates:
        preds = rev.get(g['output'], [])
        if len(preds) < 2:
            continue
        pred_ancestors = [set(rev.get(p, [])) for p in preds]
        found = False
        for i in range(len(pred_ancestors)):
            for j in range(i + 1, len(pred_ancestors)):
                if pred_ancestors[i] & pred_ancestors[j]:
                    found = True
                    break
            if found:
                break
        if found:
            reconvergent += 1
    return reconvergent / len(gates)

def classify_circuit_type(seq_ratio, total_gates, has_cycles=False):
    """Classify circuit based on sequential content and structure.

    Returns one of:
      'empty'                – no gates parsed
      'combinational'        – no sequential cells and no cycles
      'mostly_combinational' – tiny sequential content (<5 %) and acyclic
      'sequential'           – 5–34 % sequential gates (typical FSM / datapath)
      'sequential_heavy'     – ≥ 35 % sequential gates (register-file, pipeline)
    """
    if total_gates == 0:
        return 'empty'
    if has_cycles:
        return 'sequential' if seq_ratio < 0.35 else 'sequential_heavy'
    if seq_ratio == 0.0:
        return 'combinational'
    if seq_ratio < 0.05:
        return 'mostly_combinational'
    if seq_ratio < 0.35:
        return 'sequential'
    return 'sequential_heavy'

def compute_structural_complexity(gates, inputs, outputs, graph,
                                  reverse_graph, skipped_cells=None,
                                  floating_wires=None):
    """
    Balanced structural complexity score (v4).

    Formula:
        chain_penalty = max(0, 1 - depth/N)
        raw = depth * log10(N) * chain_penalty   — depth × size
            + 2.0  * log10(N)                    — pure size
            + 2.0  * log10(depth+1) * chain_pen  — standalone depth
            + 3.0  * feedback_ratio              — flat feedback
            + 5.0  * feedback_ratio * log10(N)   — feedback × size
            + 3.0  * largest_scc_size / N         — SCC coverage
            + 2.5  * seq_ratio * log10(N+1)      — sequential state
            + 1.5  * log10(cyclomatic_M + 1)     — independent cycles
            + 2.0  * xor_ratio * log10(N)        — XOR complexity
            + 1.5  * reconvergent_ratio * log10(N) — reconvergence

    Softmax-normalized to [0, 5] display scale (wider dynamic range
    than the previous sigmoid — less compression at extremes).

    Parameters
    ----------
    gates, inputs, outputs : from ``parse_verilog_netlist()``
    graph, reverse_graph   : from ``parse_verilog_netlist()``
    skipped_cells          : set of unrecognised cell type names

    Returns
    -------
    dict
        Full metrics dictionary (22+ keys).
    """
    if skipped_cells is None:
        skipped_cells = set()

    total_gates = len(gates)
    type_counts = defaultdict(int)
    seq_gates = 0
    for g in gates:
        t = g['type']
        type_counts[t] += 1
        if t in ('dff', 'latch'):
            seq_gates += 1
    logic_gates = total_gates - seq_gates
    seq_ratio   = seq_gates / max(total_gates, 1)

    depth = compute_depth(gates, inputs, graph)

    if total_gates > 10_000:
        gate_outputs = {g['output'] for g in gates}
        indeg = defaultdict(int)
        for g in gates:
            for inp in g['inputs']:
                if inp in gate_outputs:
                    indeg[g['output']] += 1
        q = deque(go for go in gate_outputs if indeg[go] == 0)
        visited_count = 0
        while q:
            node = q.popleft()
            visited_count += 1
            for nxt in graph.get(node, []):
                if nxt in gate_outputs:
                    indeg[nxt] -= 1
                    if indeg[nxt] == 0:
                        q.append(nxt)
        is_acyclic = (visited_count == len(gate_outputs))
        if is_acyclic:
            scc_stats = {
                'feedback_ratio': 0.0, 'scc_count': 0,
                'largest_scc_size': 0, 'total_nodes_in_sccs': 0,
                'cyclomatic_complexity': 0,
            }
        else:
            scc_stats, _ = compute_scc(gates, graph)
    else:
        scc_stats, _ = compute_scc(gates, graph)

    if total_gates > 50_000:
        reconvergent_ratio = 0.0
    else:
        reconvergent_ratio = compute_reconvergent_ratio(gates, reverse_graph)

    xor_count = type_counts.get('xor', 0) + type_counts.get('xnor', 0)
    buf_count = type_counts.get('buf', 0) + type_counts.get('not', 0)
    xor_ratio = xor_count / max(total_gates, 1)
    buf_ratio = buf_count / max(total_gates, 1)

    all_nodes = set(graph.keys())
    num_edges = 0
    max_fanout = 0
    fanout_sum = 0
    fanout_cnt = 0
    for k, v in graph.items():
        n = len(v)
        num_edges += n
        if n:
            if n > max_fanout:
                max_fanout = n
            fanout_sum += n
            fanout_cnt += 1
            all_nodes.update(v)
    avg_fanout = fanout_sum / fanout_cnt if fanout_cnt else 0
    num_nodes  = len(all_nodes)
    edge_density = num_edges / max(num_nodes, 1)

    type_entropy = 0.0
    if total_gates > 0:
        type_entropy = -sum(
            (c / total_gates) * math.log2(c / total_gates)
            for c in type_counts.values() if c > 0
        )

    N      = max(total_gates, 1)
    log_N  = math.log10(N)
    log_N1 = math.log10(N + 1)
    log_d  = math.log10(depth + 1)

    chain_ratio   = depth / max(N, 1)
    chain_penalty = max(0.0, 1.0 - chain_ratio)

    fb      = scc_stats['feedback_ratio']
    scc_cov = scc_stats['largest_scc_size'] / N

    raw = (
        depth * log_N * chain_penalty
        + 2.0 * log_N
        + 2.0 * log_d * chain_penalty
        + 3.0 * fb
        + 5.0 * fb * log_N
        + 3.0 * scc_cov
        + 2.5 * seq_ratio * log_N1
        + 1.5 * math.log10(scc_stats['cyclomatic_complexity'] + 1)
        + 2.0 * xor_ratio * log_N
        + 1.5 * reconvergent_ratio * log_N
    )

    if total_gates == 0:
        complexity_score = 0.0
    else:
        _T   = 8.0
        _mid = 25.0
        complexity_score = 5.0 / (1.0 + math.exp((_mid - raw) / _T))

    return {
        'complexity_score':       round(complexity_score, 4),
        'raw_score':              round(raw, 4),
        'depth':                  depth,
        'gate_count':             logic_gates,
        'total_gates':            total_gates,
        'seq_gates':              seq_gates,
        'seq_ratio':              round(seq_ratio, 4),
        'feedback_ratio':         round(scc_stats['feedback_ratio'], 4),
        'scc_count':              scc_stats['scc_count'],
        'largest_scc_size':       scc_stats['largest_scc_size'],
        'total_nodes_in_sccs':    scc_stats['total_nodes_in_sccs'],
        'cyclomatic_complexity':  scc_stats['cyclomatic_complexity'],
        'reconvergent_ratio':     round(reconvergent_ratio, 4),
        'xor_ratio':              round(xor_ratio, 4),
        'buf_ratio':              round(buf_ratio, 4),
        'edge_density':           round(edge_density, 4),
        'type_entropy':           round(type_entropy, 4),
        'max_fanout':             max_fanout,
        'avg_fanout':             round(avg_fanout, 4),
        'num_inputs':             len(inputs),
        'num_outputs':            len(outputs),
        'num_nodes':              num_nodes,
        'num_edges':              num_edges,
        'circuit_type':           classify_circuit_type(
            seq_ratio, total_gates, has_cycles=(scc_stats['scc_count'] > 0)
        ),
        'skipped_cell_types':     sorted(skipped_cells),
        'floating_wire_count':    len(floating_wires) if floating_wires else 0,
    }

def compute_complexity_metrics(gates, inputs, graph):
    """Return gate_count, depth, and composite = depth * log10(gates)."""
    logic_gates = len([g for g in gates if g['type'] not in ('dff', 'latch')])
    depth = compute_depth(gates, inputs, graph)
    safe_gates = max(logic_gates, 1)
    composite = depth * math.log10(safe_gates)
    return {
        'gate_count': logic_gates,
        'depth': depth,
        'composite': composite,
    }

"""
netlist.parser — Verilog gate-level netlist parsing.

Responsibilities:
  - Standard cell library mapping (89 cell types)
  - Verilog netlist file parsing (module/input/output/assign/gate instantiation)
  - Forward and reverse graph construction

This module contains NO scoring logic and NO ML conversion logic.
"""

import re
import logging
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)

_RE_ASSIGN_LUT   = re.compile(r'assign\s+(\S+)\s*=\s*.*?\{([^}]+)\}\s*;')
_RE_ASSIGN_SIMPLE = re.compile(r'assign\s+(\S+)\s*=\s*(\S+)\s*;')
_RE_PIN           = re.compile(r'\.(\w+)\s*\(([^)]+)\)')
_RE_DECL_RANGE    = re.compile(r'^\[(\d+)\s*:\s*(\d+)\]\s*(.*)$')
_RE_DECL_STRIP    = re.compile(r'\b(?:wire|reg|logic|signed|unsigned)\b')

STDCELL_MAP = {
    'INVX1':    ('not',   ['A'],                   'Y'),
    'INVX2':    ('not',   ['A'],                   'Y'),
    'INVX4':    ('not',   ['A'],                   'Y'),
    'INVX8':    ('not',   ['A'],                   'Y'),
    'INVXL':    ('not',   ['A'],                   'Y'),
    'BUFX1':    ('buf',   ['A'],                   'Y'),
    'BUFX2':    ('buf',   ['A'],                   'Y'),
    'BUFX3':    ('buf',   ['A'],                   'Y'),
    'BUFX4':    ('buf',   ['A'],                   'Y'),
    'CLKBUFX1': ('buf',  ['A'],                   'Y'),
    'CLKBUFX2': ('buf',  ['A'],                   'Y'),
    'CLKBUFX3': ('buf',  ['A'],                   'Y'),
    'NAND2X1':  ('nand',  ['A', 'B'],              'Y'),
    'NAND2X2':  ('nand',  ['A', 'B'],              'Y'),
    'NOR2X1':   ('nor',   ['A', 'B'],              'Y'),
    'NOR2X2':   ('nor',   ['A', 'B'],              'Y'),
    'AND2X1':   ('and',   ['A', 'B'],              'Y'),
    'AND2X2':   ('and',   ['A', 'B'],              'Y'),
    'OR2X1':    ('or',    ['A', 'B'],              'Y'),
    'OR2X2':    ('or',    ['A', 'B'],              'Y'),
    'XOR2X1':   ('xor',   ['A', 'B'],              'Y'),
    'XNOR2X1':  ('xnor',  ['A', 'B'],              'Y'),
    'NAND3X1':  ('nand',  ['A', 'B', 'C'],         'Y'),
    'NOR3X1':   ('nor',   ['A', 'B', 'C'],         'Y'),
    'AND3X1':   ('and',   ['A', 'B', 'C'],         'Y'),
    'OR3X1':    ('or',    ['A', 'B', 'C'],         'Y'),
    'NAND4X1':  ('nand',  ['A', 'B', 'C', 'D'],   'Y'),
    'NOR4X1':   ('nor',   ['A', 'B', 'C', 'D'],   'Y'),
    'AND4X1':   ('and',   ['A', 'B', 'C', 'D'],   'Y'),
    'OR4X1':    ('or',    ['A', 'B', 'C', 'D'],   'Y'),
    'AOI21X1':  ('aoi21', ['A0', 'A1', 'B0'],       'Y'),
    'AOI22X1':  ('aoi22', ['A0', 'A1', 'B0', 'B1'], 'Y'),
    'OAI21X1':  ('oai21', ['A0', 'A1', 'B0'],       'Y'),
    'OAI22X1':  ('oai22', ['A0', 'A1', 'B0', 'B1'], 'Y'),
    'OAI33X1':  ('oai21', ['A0', 'A1', 'A2', 'B0', 'B1', 'B2'], 'Y'),
    'MX2X1':    ('xor',   ['A', 'B'],              'Y'),
    'DFFPOSX1': ('dff',   ['D', 'CLK'],            'Q'),
    'DFFPOSx1': ('dff',   ['D', 'CLK'],            'Q'),
    'DFFSRX1':  ('dff',   ['D', 'CLK'],            'Q'),
    'DFFX1':    ('dff',   ['D', 'CLK'],            'Q'),
    'SDFFX1':   ('dff',   ['D', 'CLK'],            'Q'),
    'SDFFSRX1': ('dff',   ['D', 'CLK'],            'Q'),
    'DFFNEGX1': ('dff',   ['D', 'CLK'],            'Q'),
    'DFFNEGX2': ('dff',   ['D', 'CLK'],            'Q'),
    'LATCHX1':  ('latch', ['D'],                   'Q'),
    'TBUFX1':   ('buf',   ['A'],                   'Y'),
    'TBUFX2':   ('buf',   ['A'],                   'Y'),
    'TBUFX4':   ('buf',   ['A'],                   'Y'),
    'TINVX1':   ('not',   ['A'],                   'Y'),
    'TLATX1':   ('latch', ['D'],                   'Q'),
    'ADDHX1':   ('xor',   ['A', 'B'],              'SO'),
    'ADDFX1':   ('xor',   ['A', 'B'],              'SO'),
}

def _append_unique(items, value):
    if value and value not in items:
        items.append(value)

def _expand_decl_nets(decl_text):
    """Expand Verilog declarations into net names.

    Examples:
      "[3:0] a"        -> ["a", "a[0]", "a[1]", "a[2]", "a[3]"]
      "[7:0] a, b"     -> ["a", "a[0]..a[7]", "b", "b[0]..b[7]"]
      "clk, rst_n"     -> ["clk", "rst_n"]
    """
    decl = decl_text.strip().rstrip(';')
    decl = _RE_DECL_STRIP.sub('', decl).strip()

    range_m = _RE_DECL_RANGE.match(decl)
    if range_m:
        msb = int(range_m.group(1))
        lsb = int(range_m.group(2))
        names_part = range_m.group(3).strip()
    else:
        msb = lsb = None
        names_part = decl

    out = []
    for raw_name in names_part.split(','):
        name = raw_name.strip()
        if not name:
            continue
        if '=' in name:
            name = name.split('=')[0].strip()
        out.append(name)
        if msb is not None:
            lo, hi = (lsb, msb) if lsb <= msb else (msb, lsb)
            for idx in range(lo, hi + 1):
                out.append(f"{name}[{idx}]")
    return out

def parse_verilog_netlist(filepath, stdcell_map=None):
    """Parse a gate-level Verilog netlist file.

    Returns a dict with keys:
      gates, inputs, outputs, graph, reverse_graph,
      verilog_text, skipped_cells

    This is a pure parsing function with no side effects.
    """
    if stdcell_map is None:
        stdcell_map = STDCELL_MAP

    path = Path(filepath)
    raw_text = path.read_text()
    lines = raw_text.splitlines()

    gates = []
    inputs = []
    outputs = []
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    skipped_cells = set()
    declared_wires = set()

    in_module = False
    module_parsed = False
    _stdcell_lookup = stdcell_map
    _pending_line = None
    _pending_lineno = 0
    for lineno, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line[0] == '/':
            continue
        c0 = line[0]
        if c0 == 'w' and (line.startswith('wire ') or line.startswith('wire\t')):
            decl = line[4:].lstrip()
            for name in _expand_decl_nets(decl):
                declared_wires.add(name)
            continue
        if c0 == 'e' and line.startswith('endmodule'):
            in_module = False
            if gates:
                module_parsed = True
            else:
                logger.warning(
                    "%s: empty module ending at line %d skipped, "
                    "looking for next module", path.name, lineno,
                )
                inputs.clear()
                outputs.clear()
                declared_wires.clear()
            continue
        if c0 == 'e' and line.startswith('end'):
            continue

        if line.startswith('module'):
            if module_parsed:
                logger.warning(
                    "%s: additional module at line %d ignored "
                    "(only first non-empty module parsed)", path.name, lineno,
                )
                break
            in_module = True
            if '(' in line and ')' in line:
                port_section = line[line.index('(')+1:line.rindex(')')].replace(';', '')
                for token in port_section.split(','):
                    token = token.strip()
                    if token.startswith('input '):
                        decl = token[len('input '):].strip()
                        for name in _expand_decl_nets(decl):
                            _append_unique(inputs, name)
                    elif token.startswith('output '):
                        decl = token[len('output '):].strip()
                        for name in _expand_decl_nets(decl):
                            _append_unique(outputs, name)
            continue
        if not in_module:
            continue

        if line.startswith('input '):
            decl = line[len('input '):]
            for name in _expand_decl_nets(decl):
                _append_unique(inputs, name)
            continue
        if line.startswith('output '):
            decl = line[len('output '):]
            for name in _expand_decl_nets(decl):
                _append_unique(outputs, name)
            continue

        if line.startswith('assign '):
            lut_m = _RE_ASSIGN_LUT.match(line)
            if lut_m:
                out = lut_m.group(1).strip()
                inp_list = [x.strip() for x in lut_m.group(2).split(',') if x.strip()]
                if out and inp_list:
                    gtype = 'buf' if len(inp_list) == 1 else 'and'
                    gates.append({'type': gtype, 'inputs': inp_list,
                                  'output': out, 'cell_type': 'assign',
                                  'instance_name': f'assign_{out}',
                                  'verilog_line': lineno})
                    for inp in inp_list:
                        graph[inp].append(out)
                        reverse_graph[out].append(inp)
            else:
                m = _RE_ASSIGN_SIMPLE.match(line)
                if m:
                    out = m.group(1).strip()
                    inp = m.group(2).strip()
                    if out and inp and '?' not in inp:
                        gates.append({'type': 'buf', 'inputs': [inp],
                                      'output': out, 'cell_type': 'assign',
                                      'instance_name': f'assign_{out}',
                                      'verilog_line': lineno})
                        graph[inp].append(out)
                        reverse_graph[out].append(inp)
            continue

        # Accumulate multi-line cell instantiations
        if _pending_line is not None:
            _pending_line += ' ' + line
            if ';' not in line:
                continue
            line = _pending_line
            lineno = _pending_lineno
            _pending_line = None
        elif '(' in line and ';' not in line:
            # Check if this looks like a cell instantiation (has a space before the paren)
            cell_part = line.split('(', 1)[0].strip()
            if ' ' in cell_part:
                _pending_line = line
                _pending_lineno = lineno
                continue

        if '(' in line and ';' in line:
            cell_part = line.split('(', 1)[0].strip()
            if ' ' not in cell_part:
                continue
            parts = cell_part.rsplit(None, 2)
            if len(parts) < 2:
                continue
            cell_type = parts[-2]
            if not cell_type[0].isalpha():
                continue
            if cell_type not in _stdcell_lookup:
                skipped_cells.add(cell_type)
                continue

            conn = dict(_RE_PIN.findall(line))

            gate_type, in_pins, out_pin = _stdcell_lookup[cell_type]
            gate_inputs = [conn.get(p) for p in in_pins if conn.get(p)]
            output = conn.get(out_pin)
            if not output or len(gate_inputs) == 0:
                continue

            instance_name = parts[-1]
            gates.append({'type': gate_type, 'inputs': gate_inputs, 'output': output,
                          'cell_type': cell_type, 'instance_name': instance_name,
                          'verilog_line': lineno})
            for i in gate_inputs:
                graph[i].append(output)
                reverse_graph[output].append(i)

    if skipped_cells:
        logger.warning(
            "%s: skipped %d unrecognised cell type(s): %s",
            path.name, len(skipped_cells), sorted(skipped_cells),
        )

    if len(gates) == 0:
        logger.warning("%s: module contains zero gates", path.name)

    seen_names: dict = {}
    for g in gates:
        name = g['instance_name']
        if name in seen_names:
            seen_names[name] += 1
            new_name = f"{name}_dup_{seen_names[name]}"
            logger.warning(
                "%s: duplicate instance name '%s' renamed to '%s'",
                path.name, name, new_name,
            )
            g['instance_name'] = new_name
        else:
            seen_names[name] = 0

    assign_edges: dict = {}
    for g in gates:
        if g['cell_type'] == 'assign' and len(g['inputs']) == 1:
            assign_edges[g['output']] = g['inputs'][0]
    for start in assign_edges:
        visited: set = set()
        cur = start
        while cur in assign_edges and cur not in visited:
            visited.add(cur)
            cur = assign_edges[cur]
        if cur in visited:
            logger.warning(
                "%s: cyclic assign chain detected involving wire '%s'",
                path.name, cur,
            )
            break

    used_wires = set(inputs) | set(outputs)
    for g in gates:
        used_wires.add(g['output'])
        used_wires.update(g['inputs'])
    floating_wires = declared_wires - used_wires
    if floating_wires:
        logger.warning(
            "%s: %d floating wire(s): %s",
            path.name, len(floating_wires), sorted(floating_wires),
        )

    logger.info("Parsed %s gates from %s", f"{len(gates):,}", path.name)

    return {
        'gates': gates,
        'inputs': inputs,
        'outputs': outputs,
        'graph': graph,
        'reverse_graph': reverse_graph,
        'verilog_text': '',
        'skipped_cells': skipped_cells,
        'floating_wires': floating_wires,
    }

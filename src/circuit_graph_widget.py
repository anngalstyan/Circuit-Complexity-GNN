"""
Circuit Graph Widget
====================
Split-pane PyQt5 widget: Verilog code viewer (left) + interactive circuit
schematic (right) with bidirectional click-to-highlight linkage.

Gate shapes are drawn with QPainterPath to resemble schematic symbols.
"""

import math
import re
from pathlib import Path

from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPathItem,
    QGraphicsTextItem, QGraphicsLineItem,
    QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QFrame, QApplication, QPushButton,
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimeLine
from PyQt5.QtGui import (
    QPainter, QPainterPath, QBrush, QPen, QColor, QFont, QPolygonF,
    QTextCursor, QTextCharFormat, QSyntaxHighlighter, QTextDocument,
)

def _C():
    try:
        from circuit_complexity_gui import C
        return C
    except ImportError:
        class _Fallback:
            PANEL = "#FFFFFF"; BORDER = "#E4E4E7"; BG = "#F7F7F8"
            T1 = "#111111"; T2 = "#555555"; T3 = "#999999"
            BLUE = "#2563EB"; BLUE_LIGHT = "#EFF6FF"
            DIVIDER = "#F0F0F2"; RED = "#DC2626"
            SIDEBAR = "#FAFAFA"
        return _Fallback

MONO = "'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace"

GATE_COLORS = {
    'input': "#3B82F6", 'output': "#8B5CF6",
    'and': "#16A34A", 'or': "#16A34A", 'nand': "#15803D", 'nor': "#15803D",
    'not': "#F59E0B", 'buf': "#F59E0B",
    'xor': "#0891B2", 'xnor': "#0891B2",
    'dff': "#DC2626", 'latch': "#DC2626",
    'aoi21': "#059669", 'aoi22': "#059669",
    'oai21': "#059669", 'oai22': "#059669",
}

class VerilogHighlighter(QSyntaxHighlighter):
    """Basic Verilog syntax highlighting for the code pane."""

    def __init__(self, document):
        super().__init__(document)
        self._rules = []

        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("#2563EB"))
        kw_fmt.setFontWeight(QFont.Bold)
        keywords = [
            r'\bmodule\b', r'\bendmodule\b', r'\binput\b', r'\boutput\b',
            r'\bwire\b', r'\breg\b', r'\bassign\b', r'\binout\b',
            r'\bparameter\b', r'\bsupply0\b', r'\bsupply1\b',
        ]
        for kw in keywords:
            self._rules.append((re.compile(kw), kw_fmt))

        cell_fmt = QTextCharFormat()
        cell_fmt.setForeground(QColor("#16A34A"))
        cell_fmt.setFontWeight(QFont.Bold)
        self._rules.append((re.compile(r'\b[A-Z][A-Z0-9_]{2,}\b'), cell_fmt))

        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor("#D97706"))
        self._rules.append((re.compile(r"\b\d+'[bBhHdDoO][0-9a-fA-F_]+\b"), num_fmt))
        self._rules.append((re.compile(r'\b\d+\b'), num_fmt))

        str_fmt = QTextCharFormat()
        str_fmt.setForeground(QColor("#7C3AED"))
        self._rules.append((re.compile(r'"[^"]*"'), str_fmt))

        self._comment_fmt = QTextCharFormat()
        self._comment_fmt.setForeground(QColor("#999999"))
        self._comment_fmt.setFontItalic(True)
        self._rules.append((re.compile(r'//[^\n]*'), self._comment_fmt))

    def highlightBlock(self, text):
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)

class GateShapeItem(QGraphicsPathItem):
    """Custom QPainterPath item that draws a distinct shape per gate type."""

    W = 50
    H = 30

    def __init__(self, gate_dict, x, y, gate_id):
        super().__init__()
        self.gate_data = gate_dict
        self.gate_id = gate_id
        self._highlighted = False

        gtype = gate_dict.get('type', 'and')
        color = QColor(GATE_COLORS.get(gtype, "#888888"))

        self._base_color = color
        self.setPen(QPen(color.darker(130), 1.5))
        self.setBrush(QBrush(color))

        self._build_shape(gtype)
        self.setPos(x, y)
        self.setFlag(self.ItemIsSelectable, True)
        self.setCursor(Qt.PointingHandCursor)

        inst = gate_dict.get('instance_name', '')
        ctype = gate_dict.get('cell_type', gtype)
        self.setToolTip(f"{inst}\nCell: {ctype}\nType: {gtype}")

        if inst and not inst.startswith('assign_'):
            txt = QGraphicsTextItem(inst[:10], self)
            txt.setDefaultTextColor(QColor(_C().T3))
            txt.setFont(QFont("SF Mono", 7))
            txt.setPos(-5, self.H / 2 + 2)

    def _build_shape(self, gtype):
        path = QPainterPath()
        w, h = self.W, self.H

        if gtype in ('and', 'or', 'nand', 'nor'):
            self._draw_logic_gate(path, gtype, w, h)
        elif gtype == 'not':
            self._draw_triangle(path, w, h, bubble=True)
            self.W, self.H = 35, 24
        elif gtype == 'buf':
            self._draw_triangle(path, w, h, bubble=False)
            self.W, self.H = 35, 24
        elif gtype in ('xor', 'xnor'):
            self._draw_xor(path, w, h, gtype == 'xnor')
        elif gtype in ('dff', 'latch'):
            self._draw_sequential(path, gtype, w, h)
        elif gtype == 'input':
            self._draw_port(path, 12)
            self.W, self.H = 24, 24
        elif gtype == 'output':
            self._draw_port(path, 12)
            self.W, self.H = 24, 24
        else:
            self._draw_logic_gate(path, gtype, w, h)

        self.setPath(path)

    def _draw_logic_gate(self, path, gtype, w, h):
        """Rounded rectangle with type label inside."""
        path.addRoundedRect(-w/2, -h/2, w, h, 5, 5)
        if gtype in ('nand', 'nor'):
            path.addEllipse(w/2, -3, 6, 6)

        txt = QGraphicsTextItem(gtype.upper(), self)
        txt.setDefaultTextColor(QColor("white"))
        txt.setFont(QFont("SF Pro Text", 8, QFont.Bold))
        br = txt.boundingRect()
        txt.setPos(-br.width()/2, -br.height()/2)

    def _draw_triangle(self, path, w, h, bubble):
        """Right-pointing triangle for NOT/BUF."""
        tw, th = 30, 24
        path.moveTo(-tw/2, -th/2)
        path.lineTo(tw/2, 0)
        path.lineTo(-tw/2, th/2)
        path.closeSubpath()
        if bubble:
            path.addEllipse(tw/2, -3, 6, 6)

    def _draw_xor(self, path, w, h, is_xnor):
        """Rounded rectangle with extra curved line on input side."""
        path.addRoundedRect(-w/2, -h/2, w, h, 5, 5)
        path.moveTo(-w/2 - 4, -h/2)
        path.quadTo(-w/2 - 8, 0, -w/2 - 4, h/2)
        if is_xnor:
            path.addEllipse(w/2, -3, 6, 6)

        txt = QGraphicsTextItem("XOR" if not is_xnor else "XNOR", self)
        txt.setDefaultTextColor(QColor("white"))
        txt.setFont(QFont("SF Pro Text", 7, QFont.Bold))
        br = txt.boundingRect()
        txt.setPos(-br.width()/2, -br.height()/2)

    def _draw_sequential(self, path, gtype, w, h):
        """Rectangle with clock mark for DFF/latch."""
        path.addRoundedRect(-w/2, -h/2, w, h, 3, 3)
        cx, cy = -w/2, 0
        path.moveTo(cx, cy - 5)
        path.lineTo(cx + 7, cy)
        path.lineTo(cx, cy + 5)

        lbl = "DFF" if gtype == 'dff' else "LAT"
        txt = QGraphicsTextItem(lbl, self)
        txt.setDefaultTextColor(QColor("white"))
        txt.setFont(QFont("SF Pro Text", 8, QFont.Bold))
        br = txt.boundingRect()
        txt.setPos(-br.width()/2, -br.height()/2)

    def _draw_port(self, path, r):
        """Circle for input/output port."""
        path.addEllipse(-r, -r, r*2, r*2)

    def set_highlighted(self, on):
        """Toggle highlight glow."""
        self._highlighted = on
        if on:
            self.setPen(QPen(QColor("#FACC15"), 3))
        else:
            self.setPen(QPen(self._base_color.darker(130), 1.5))
        self.update()

    @property
    def center_x(self):
        return self.pos().x()

    @property
    def center_y(self):
        return self.pos().y()

    @property
    def right_x(self):
        return self.pos().x() + self.W / 2

    @property
    def left_x(self):
        return self.pos().x() - self.W / 2

class CircuitSchematicView(QGraphicsView):
    """Interactive circuit schematic with BFS layered layout and gate shapes."""

    gate_clicked = pyqtSignal(dict)
    zoom_changed = pyqtSignal(float)
    MAX_RENDER_NODES = 500

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self._gate_items = {}
        self._highlighted_id = None

        self._zoom_level = 1.0
        self._zoom_target = 1.0
        self._zoom_timeline = None
        self._zoom_min = 0.05
        self._zoom_max = 10.0

        self.apply_theme()
        self._show_placeholder()

    def apply_theme(self):
        C = _C()
        self.setStyleSheet(
            f"border: 1px solid {C.BORDER}; border-radius: 8px; background: {C.PANEL};"
        )

    def _show_placeholder(self):
        self._scene.clear()
        self._gate_items.clear()
        C = _C()
        txt = self._scene.addText("Analyse a circuit to view its schematic",
                                  QFont("SF Pro Text", 12))
        txt.setDefaultTextColor(QColor(C.T3))

    def set_circuit(self, gates, inputs, outputs):
        """Render circuit. gates = list of gate dicts, inputs/outputs = lists of names."""
        self._scene.clear()
        self._gate_items.clear()
        total = len(gates) + len(inputs) + len(outputs)

        if total == 0:
            self._show_placeholder()
            return
        if total > self.MAX_RENDER_NODES:
            self._show_summary(gates, inputs, outputs)
            return

        self._render_schematic(gates, inputs, outputs)

    def _show_summary(self, gates, inputs, outputs):
        """For large circuits, show gate type distribution bar chart."""
        C = _C()
        y = 0
        total = len(gates) + len(inputs) + len(outputs)
        hdr = self._scene.addText(
            f"Circuit too large for full schematic ({total} nodes)",
            QFont("SF Pro Text", 13, QFont.Bold))
        hdr.setDefaultTextColor(QColor(C.T1))
        hdr.setPos(0, y); y += 40

        type_counts = {}
        for g in gates:
            t = g.get('type', '?')
            type_counts[t] = type_counts.get(t, 0) + 1
        type_counts['input'] = len(inputs)
        type_counts['output'] = len(outputs)

        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
        max_count = max(type_counts.values(), default=1)

        for gtype, count in sorted_types[:14]:
            color = QColor(GATE_COLORS.get(gtype, "#888888"))
            bar_w = int(count / max_count * 300)
            self._scene.addRect(130, y, bar_w, 18, QPen(Qt.NoPen), QBrush(color))

            lbl = self._scene.addText(gtype, QFont("SF Mono", 10))
            lbl.setDefaultTextColor(QColor(C.T2)); lbl.setPos(0, y - 2)

            cnt = self._scene.addText(str(count), QFont("SF Mono", 9))
            cnt.setDefaultTextColor(QColor(C.T3)); cnt.setPos(135 + bar_w, y - 1)
            y += 26

        y += 10
        stats = self._scene.addText(
            f"Inputs: {len(inputs)}  |  Outputs: {len(outputs)}  |  "
            f"Gates: {len(gates)}  |  Render limit: {self.MAX_RENDER_NODES}",
            QFont("SF Pro Text", 10))
        stats.setDefaultTextColor(QColor(C.T3)); stats.setPos(0, y)

    def _render_schematic(self, gates, inputs, outputs):
        """Full schematic with gate shapes and layered layout."""
        C = _C()

        node_layer = {}
        node_type = {}
        node_gate = {}
        gate_by_output = {}
        seq_types = {'dff', 'latch'}

        for g in gates:
            gate_by_output[g['output']] = g

        output_set = set(outputs)

        dff_gates = [g for g in gates if g['type'] in seq_types]
        dff_outputs = {g['output'] for g in dff_gates}

        all_gate_inputs = set()
        for g in gates:
            all_gate_inputs.update(g['inputs'])

        implicit_dff_inputs = [n for n in inputs if n not in all_gate_inputs]
        implicit_dff_edges = []
        for n in implicit_dff_inputs:
            for g in dff_gates:
                implicit_dff_edges.append((n, g['output'] + '__dff'))

        for n in inputs:
            node_layer[n] = 0
            node_type[n] = 'input'
        for n in dff_outputs:
            node_layer[n] = 0
            node_type[n] = gate_by_output[n]['type']
            node_gate[n] = gate_by_output[n]

        known_sources = set(inputs) | {g['output'] for g in gates}

        phantom_inputs = set()
        for g in gates:
            for inp in g['inputs']:
                if inp not in known_sources:
                    phantom_inputs.add(inp)
        for ph in phantom_inputs:
            node_layer[ph] = 0
            node_type[ph] = 'input'

        changed = True
        iteration = 0
        while changed and iteration < 200:
            changed = False
            iteration += 1
            for g in gates:
                if g['type'] in seq_types:
                    continue
                max_in_layer = -1
                all_resolved = True
                for inp in g['inputs']:
                    if inp in node_layer:
                        max_in_layer = max(max_in_layer, node_layer[inp])
                    else:
                        all_resolved = False
                if all_resolved and max_in_layer >= 0:
                    new_layer = max_in_layer + 1
                    if g['output'] not in node_layer or node_layer[g['output']] != new_layer:
                        node_layer[g['output']] = new_layer
                        node_type[g['output']] = g['type']
                        node_gate[g['output']] = g
                        changed = True

        changed = True
        iteration = 0
        while changed and iteration < 200:
            changed = False
            iteration += 1
            for g in gates:
                if g['type'] in seq_types:
                    continue
                if g['output'] in node_layer:
                    continue
                max_in_layer = -1
                for inp in g['inputs']:
                    if inp in node_layer:
                        max_in_layer = max(max_in_layer, node_layer[inp])
                if max_in_layer >= 0:
                    new_layer = max_in_layer + 1
                    node_layer[g['output']] = new_layer
                    node_type[g['output']] = g['type']
                    node_gate[g['output']] = g
                    changed = True

        for g in dff_gates:
            inp_layers = [node_layer[i] for i in g['inputs'] if i in node_layer]
            g_layer = (max(inp_layers) + 1) if inp_layers else 1
            dff_node = g['output'] + '__dff'
            node_layer[dff_node] = g_layer
            node_type[dff_node] = g['type']
            node_gate[dff_node] = g

        for g in gates:
            if g['type'] in seq_types:
                continue
            if g['output'] not in node_layer:
                node_layer[g['output']] = 1
                node_type[g['output']] = g['type']
                node_gate[g['output']] = g

        max_layer = max(node_layer.values(), default=0)
        out_port_layer = max_layer + 1
        output_port_edges = []
        for n in outputs:
            if n in node_layer:
                port_key = n + '__out'
                node_layer[port_key] = out_port_layer
                node_type[port_key] = 'output'
                output_port_edges.append((n, port_key))
            else:
                node_layer[n] = out_port_layer
                node_type[n] = 'output'

        layers = {}
        for node, layer in node_layer.items():
            layers.setdefault(layer, []).append(node)

        forward_adj = {}
        backward_adj = {}
        for g in gates:
            out = g['output']
            out_key = (out + '__dff') if g['type'] in seq_types else out
            for inp in g['inputs']:
                if inp in node_layer and out_key in node_layer:
                    forward_adj.setdefault(inp, []).append(out_key)
                    backward_adj.setdefault(out_key, []).append(inp)
            if g['type'] in seq_types and out in node_layer:
                for g2 in gates:
                    if out in g2['inputs']:
                        g2_key = (g2['output'] + '__dff') if g2['type'] in seq_types else g2['output']
                        if g2_key in node_layer:
                            forward_adj.setdefault(out, []).append(g2_key)
                            backward_adj.setdefault(g2_key, []).append(out)
        for src, port in output_port_edges:
            forward_adj.setdefault(src, []).append(port)
            backward_adj.setdefault(port, []).append(src)
        for src, dff_key in implicit_dff_edges:
            if src in node_layer and dff_key in node_layer:
                forward_adj.setdefault(src, []).append(dff_key)
                backward_adj.setdefault(dff_key, []).append(src)

        sorted_layers = sorted(layers.keys())
        layer_order = {li: list(layers[li]) for li in sorted_layers}

        def _pos_map(layer_idx):
            return {n: i for i, n in enumerate(layer_order[layer_idx])}

        for _sweep in range(6):
            for li in sorted_layers[1:]:
                if (li - 1) not in layer_order:
                    continue
                pp = _pos_map(li - 1)
                def _key_fwd(node, _pp=pp):
                    preds = backward_adj.get(node, [])
                    vals = [_pp[p] for p in preds if p in _pp]
                    return sum(vals) / len(vals) if vals else 0
                layer_order[li] = sorted(layer_order[li], key=_key_fwd)
            for li in reversed(sorted_layers[:-1]):
                if (li + 1) not in layer_order:
                    continue
                np_ = _pos_map(li + 1)
                def _key_bwd(node, _np=np_):
                    succs = forward_adj.get(node, [])
                    vals = [_np[s] for s in succs if s in _np]
                    return sum(vals) / len(vals) if vals else 0
                layer_order[li] = sorted(layer_order[li], key=_key_bwd)

        x_spacing = 180
        y_spacing = 75
        positions = {}
        for layer_idx in sorted_layers:
            nodes = layer_order[layer_idx]
            x = layer_idx * x_spacing
            for i, node in enumerate(nodes):
                y = i * y_spacing - (len(nodes) - 1) * y_spacing / 2
                positions[node] = (x, y)

        edge_pen = QPen(QColor("#B0B8C4"), 1.2)
        edge_pen.setCosmetic(True)
        arrow_brush = QBrush(QColor("#8892A0"))

        for g in gates:
            out_key = (g['output'] + '__dff') if g['type'] in seq_types else g['output']
            if out_key not in positions:
                continue
            dx, dy = positions[out_key]
            for inp in g['inputs']:
                if inp not in positions:
                    continue
                sx, sy = positions[inp]

                x0, y0 = sx + 25, sy
                x3, y3 = dx - 25, dy

                if x0 == x3 and y0 == y3:
                    continue

                gap = x3 - x0
                if gap < 0:
                    ctrl_off = max(abs(gap) * 0.6, 60)
                else:
                    ctrl_off = max(gap * 0.4, 30)

                wire = QPainterPath()
                wire.moveTo(x0, y0)
                wire.cubicTo(x0 + ctrl_off, y0,
                             x3 - ctrl_off, y3,
                             x3, y3)

                path_item = QGraphicsPathItem()
                path_item.setPath(wire)
                path_item.setPen(edge_pen)
                path_item.setBrush(QBrush(Qt.NoBrush))
                path_item.setZValue(-1)
                self._scene.addItem(path_item)

                ax, ay = x3, y3
                a_sz = 5
                p1 = QPointF(ax - a_sz, ay - a_sz * 0.4)
                p2 = QPointF(ax - a_sz, ay + a_sz * 0.4)
                arrow = self._scene.addPolygon(
                    QPolygonF([QPointF(ax, ay), p1, p2]),
                    QPen(Qt.NoPen), arrow_brush)
                arrow.setZValue(-1)

        feedback_pen = QPen(QColor("#E879A0"), 1.4, Qt.DashLine)
        feedback_pen.setCosmetic(True)
        for g in dff_gates:
            dff_key = g['output'] + '__dff'
            out_sig = g['output']
            if dff_key in positions and out_sig in positions:
                sx, sy = positions[dff_key]
                dx, dy = positions[out_sig]
                x0, y0 = sx + 25, sy
                x3, y3 = dx - 25, dy
                ctrl_off = max(abs(x3 - x0) * 0.6, 80)
                wire = QPainterPath()
                wire.moveTo(x0, y0)
                wire.cubicTo(x0 + ctrl_off, y0,
                             x3 - ctrl_off, y3,
                             x3, y3)
                path_item = QGraphicsPathItem()
                path_item.setPath(wire)
                path_item.setPen(feedback_pen)
                path_item.setBrush(QBrush(Qt.NoBrush))
                path_item.setZValue(-1)
                self._scene.addItem(path_item)

        for src, port in output_port_edges:
            if src in positions and port in positions:
                sx, sy = positions[src]
                dx, dy = positions[port]
                x0, y0 = sx + 25, sy
                x3, y3 = dx - 25, dy
                gap = x3 - x0
                ctrl_off = max(gap * 0.4, 30) if gap > 0 else 30
                wire = QPainterPath()
                wire.moveTo(x0, y0)
                wire.cubicTo(x0 + ctrl_off, y0,
                             x3 - ctrl_off, y3,
                             x3, y3)
                path_item = QGraphicsPathItem()
                path_item.setPath(wire)
                path_item.setPen(edge_pen)
                path_item.setBrush(QBrush(Qt.NoBrush))
                path_item.setZValue(-1)
                self._scene.addItem(path_item)
                ax, ay = x3, y3
                a_sz = 5
                p1 = QPointF(ax - a_sz, ay - a_sz * 0.4)
                p2 = QPointF(ax - a_sz, ay + a_sz * 0.4)
                arrow = self._scene.addPolygon(
                    QPolygonF([QPointF(ax, ay), p1, p2]),
                    QPen(Qt.NoPen), arrow_brush)
                arrow.setZValue(-1)

        ctrl_pen = QPen(QColor("#7EB8DA"), 1.2, Qt.DashDotLine)
        ctrl_pen.setCosmetic(True)
        for src, dff_key in implicit_dff_edges:
            if src in positions and dff_key in positions:
                sx, sy = positions[src]
                dx, dy = positions[dff_key]
                x0, y0 = sx + 25, sy
                x3, y3 = dx - 25, dy
                gap = x3 - x0
                ctrl_off = max(gap * 0.4, 30) if gap > 0 else max(abs(gap) * 0.6, 60)
                wire = QPainterPath()
                wire.moveTo(x0, y0)
                wire.cubicTo(x0 + ctrl_off, y0,
                             x3 - ctrl_off, y3,
                             x3, y3)
                path_item = QGraphicsPathItem()
                path_item.setPath(wire)
                path_item.setPen(ctrl_pen)
                path_item.setBrush(QBrush(Qt.NoBrush))
                path_item.setZValue(-1)
                self._scene.addItem(path_item)
                ax, ay = x3, y3
                a_sz = 5
                p1 = QPointF(ax - a_sz, ay - a_sz * 0.4)
                p2 = QPointF(ax - a_sz, ay + a_sz * 0.4)
                arrow = self._scene.addPolygon(
                    QPolygonF([QPointF(ax, ay), p1, p2]),
                    QPen(Qt.NoPen), QBrush(QColor("#7EB8DA")))
                arrow.setZValue(-1)

        for node, (x, y) in positions.items():
            gtype = node_type.get(node, 'and')
            real_node = node
            if node.endswith('__dff'):
                real_node = node[:-5]
            elif node.endswith('__out'):
                real_node = node[:-5]
            g_dict = node_gate.get(node, {
                'type': gtype, 'inputs': [], 'output': real_node,
                'cell_type': gtype, 'instance_name': real_node[:12],
                'verilog_line': 0,
            })

            gate_id = g_dict.get('instance_name', real_node)
            item = GateShapeItem(g_dict, x, y, gate_id)
            item.setZValue(1)
            self._scene.addItem(item)
            self._gate_items[gate_id] = item

            if real_node not in self._gate_items:
                self._gate_items[real_node] = item

        self._scene.selectionChanged.connect(self._on_selection_changed)

        self._scene.setSceneRect(
            self._scene.itemsBoundingRect().adjusted(-50, -50, 50, 50))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self.scale(1.5, 1.5)
        self._zoom_level = self.transform().m11()
        self._zoom_target = self._zoom_level

    def _on_selection_changed(self):
        for item in self._scene.selectedItems():
            if isinstance(item, GateShapeItem):
                self.gate_clicked.emit(item.gate_data)
                return

    def highlight_gate(self, gate_id):
        """Highlight a specific gate (called from code pane click)."""
        if self._highlighted_id and self._highlighted_id in self._gate_items:
            self._gate_items[self._highlighted_id].set_highlighted(False)

        if gate_id in self._gate_items:
            item = self._gate_items[gate_id]
            item.set_highlighted(True)
            self._highlighted_id = gate_id
            self.centerOn(item)

    def reset(self):
        self._show_placeholder()
        self._zoom_level = 1.0
        self._zoom_target = 1.0

    def wheelEvent(self, event):
        """Animated zoom on scroll."""
        if event.angleDelta().y() > 0:
            self._zoom_target *= 1.25
        else:
            self._zoom_target /= 1.25
        self._zoom_target = max(self._zoom_min,
                                min(self._zoom_max, self._zoom_target))
        self._animate_zoom()
        event.accept()

    def _animate_zoom(self):
        """Animate from current zoom level to target over 150 ms."""
        if self._zoom_timeline is not None:
            self._zoom_timeline.stop()

        start_zoom = self._zoom_level
        target_zoom = self._zoom_target

        timeline = QTimeLine(150, self)
        timeline.setUpdateInterval(16)
        timeline.setFrameRange(0, 100)

        def on_frame(frame):
            t = frame / 100.0
            t = 1.0 - (1.0 - t) ** 2
            new_zoom = start_zoom + (target_zoom - start_zoom) * t
            if self._zoom_level != 0:
                factor = new_zoom / self._zoom_level
                self.scale(factor, factor)
                self._zoom_level = new_zoom
            self.zoom_changed.emit(self._zoom_level)

        def on_finished():
            self._zoom_level = target_zoom
            self._zoom_timeline = None
            self.zoom_changed.emit(self._zoom_level)

        timeline.frameChanged.connect(on_frame)
        timeline.finished.connect(on_finished)
        timeline.start()
        self._zoom_timeline = timeline

    def fit_to_window(self):
        """Reset zoom and fit entire circuit in view."""
        if self._zoom_timeline is not None:
            self._zoom_timeline.stop()
            self._zoom_timeline = None
        self.resetTransform()
        self._zoom_level = 1.0
        self._zoom_target = 1.0
        if self._scene.items():
            self._scene.setSceneRect(
                self._scene.itemsBoundingRect().adjusted(-50, -50, 50, 50))
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
            self._zoom_level = self.transform().m11()
            self._zoom_target = self._zoom_level
        self.zoom_changed.emit(self._zoom_level)

    def zoom_step(self, factor):
        """Programmatic zoom step (used by toolbar buttons)."""
        self._zoom_target = self._zoom_level * factor
        self._zoom_target = max(self._zoom_min,
                                min(self._zoom_max, self._zoom_target))
        self._animate_zoom()

class VerilogCodePane(QTextEdit):
    """Read-only Verilog source viewer with syntax highlighting."""

    line_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.NoWrap)
        self._highlighter = None
        self._hl_format = QTextCharFormat()
        self._hl_format.setBackground(QColor("#FEF9C3"))
        self._programmatic = False
        self.apply_theme()

    def apply_theme(self):
        C = _C()
        self.setStyleSheet(f"""
            QTextEdit {{
                background: {C.PANEL};
                border: 1px solid {C.BORDER};
                border-radius: 8px;
                padding: 8px;
                font-size: 11px;
                font-family: {MONO};
                color: {C.T1};
            }}
        """)
        if C.PANEL == "#181825":
            self._hl_format.setBackground(QColor("#45475A"))
        else:
            self._hl_format.setBackground(QColor("#FEF9C3"))

    def set_source(self, text):
        self.setPlainText(text)
        if self._highlighter is None:
            self._highlighter = VerilogHighlighter(self.document())

    def highlight_line(self, lineno):
        """Highlight and scroll to a line."""
        if lineno < 1:
            return
        self._programmatic = True
        block = self.document().findBlockByLineNumber(lineno - 1)
        if not block.isValid():
            self._programmatic = False
            return

        cursor = QTextCursor(block)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)

        sel = QTextEdit.ExtraSelection()
        sel.format = self._hl_format
        sel.cursor = cursor
        self.setExtraSelections([sel])

        self.setTextCursor(QTextCursor(block))
        self.ensureCursorVisible()
        self._programmatic = False

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self._programmatic:
            return
        cursor = self.cursorForPosition(event.pos())
        lineno = cursor.blockNumber() + 1
        self.line_clicked.emit(lineno)

    def reset(self):
        self.clear()
        self.setExtraSelections([])

    def clear_highlight(self):
        self.setExtraSelections([])

class CircuitGraphWidget(QWidget):
    """Split-pane: Verilog code (left) + circuit schematic (right)."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toolbar = QFrame()
        self._toolbar.setFixedHeight(34)
        tb = QHBoxLayout(self._toolbar)
        tb.setContentsMargins(6, 3, 6, 3)
        tb.setSpacing(4)

        self._fit_btn = QPushButton("⟲  Fit to Window")
        self._fit_btn.setFixedHeight(26)
        self._fit_btn.setCursor(Qt.PointingHandCursor)
        self._fit_btn.setToolTip("Reset zoom and fit entire circuit in view")
        self._fit_btn.clicked.connect(self._on_fit)
        tb.addWidget(self._fit_btn)

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setFixedSize(26, 26)
        self._zoom_in_btn.setCursor(Qt.PointingHandCursor)
        self._zoom_in_btn.setToolTip("Zoom in")
        self._zoom_in_btn.clicked.connect(self._on_zoom_in)
        tb.addWidget(self._zoom_in_btn)

        self._zoom_out_btn = QPushButton("\u2212")
        self._zoom_out_btn.setFixedSize(26, 26)
        self._zoom_out_btn.setCursor(Qt.PointingHandCursor)
        self._zoom_out_btn.setToolTip("Zoom out")
        self._zoom_out_btn.clicked.connect(self._on_zoom_out)
        tb.addWidget(self._zoom_out_btn)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(60)
        tb.addWidget(self._zoom_label)

        tb.addStretch()
        layout.addWidget(self._toolbar)

        self._splitter = QSplitter(Qt.Horizontal)
        self._code_pane = VerilogCodePane()
        self._schematic = CircuitSchematicView()

        self._splitter.addWidget(self._code_pane)
        self._splitter.addWidget(self._schematic)
        self._splitter.setSizes([300, 500])
        self._splitter.setStyleSheet("QSplitter::handle { background: transparent; width: 3px; }")
        layout.addWidget(self._splitter, 1)

        self._gate_line_map = {}
        self._line_gate_map = {}
        self._programmatic = False

        self._schematic.gate_clicked.connect(self._on_gate_clicked)
        self._code_pane.line_clicked.connect(self._on_line_clicked)
        self._schematic.zoom_changed.connect(self._update_zoom_label)

        self.apply_theme()

    def set_result(self, r):
        """Set from analysis result dict."""
        gates = r.get("graph_gates", [])
        inputs = r.get("graph_inputs", [])
        outputs = r.get("graph_outputs", [])

        self._gate_line_map.clear()
        self._line_gate_map.clear()
        for i, g in enumerate(gates):
            gate_id = g.get('instance_name', f'gate_{i}')
            vline = g.get('verilog_line', 0)
            if vline:
                self._gate_line_map[gate_id] = vline
                self._line_gate_map[vline] = gate_id

        filepath = r.get("filepath")
        if filepath and Path(filepath).exists():
            try:
                self._code_pane.set_source(Path(filepath).read_text())
            except Exception:
                self._code_pane.set_source("(Could not read source file)")
        else:
            self._code_pane.set_source("(Source file not available)")

        self._schematic.set_circuit(gates, inputs, outputs)

    def _on_gate_clicked(self, gate_dict):
        """Gate clicked in schematic -> highlight in code."""
        if self._programmatic:
            return
        vline = gate_dict.get('verilog_line', 0)
        if vline:
            self._programmatic = True
            self._code_pane.highlight_line(vline)
            self._programmatic = False

    def _on_line_clicked(self, lineno):
        """Line clicked in code -> highlight gate in schematic."""
        if self._programmatic:
            return
        gate_id = self._line_gate_map.get(lineno)
        if gate_id:
            self._programmatic = True
            self._schematic.highlight_gate(gate_id)
            self._programmatic = False

    def _on_fit(self):
        self._schematic.fit_to_window()

    def _on_zoom_in(self):
        self._schematic.zoom_step(1.25)

    def _on_zoom_out(self):
        self._schematic.zoom_step(1 / 1.25)

    def _update_zoom_label(self, level):
        pct = max(1, int(level * 100))
        self._zoom_label.setText(f"{pct}%")

    def reset(self):
        self._code_pane.reset()
        self._schematic.reset()
        self._gate_line_map.clear()
        self._line_gate_map.clear()
        self._zoom_label.setText("100%")

    def apply_theme(self):
        C = _C()
        self._code_pane.apply_theme()
        self._schematic.apply_theme()

        self._toolbar.setStyleSheet(
            f"QFrame {{ background: {C.PANEL}; "
            f"border-bottom: 1px solid {C.BORDER}; }}"
        )
        btn_style = (
            f"QPushButton {{ background: {C.PANEL}; color: {C.T1}; "
            f"border: 1px solid {C.BORDER}; border-radius: 4px; "
            f"font-size: 12px; font-weight: 600; padding: 2px 8px; }}"
            f"QPushButton:hover {{ background: {C.DIVIDER}; }}"
        )
        self._fit_btn.setStyleSheet(btn_style)
        self._zoom_in_btn.setStyleSheet(btn_style)
        self._zoom_out_btn.setStyleSheet(btn_style)
        self._zoom_label.setStyleSheet(
            f"color: {C.T3}; font-size: 10px; border: none;"
        )

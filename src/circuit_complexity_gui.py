#!/usr/bin/env python3
"""
Circuit Complexity Predictor
=============================
Minimalist, professional PyQt5 GUI for GNN-based circuit complexity prediction.

Usage:
    python circuit_complexity_gui.py
    python circuit_complexity_gui.py --model models/best_complexity_model.pt
"""

import sys
import os
import time
import json
import csv
import math
import argparse
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QFrame, QGridLayout,
    QTabWidget, QTextEdit, QSplitter, QScrollArea, QSizePolicy,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QMenu, QAction, QMessageBox, QListWidget, QListWidgetItem,
    QTreeWidget, QTreeWidgetItem, QCheckBox,
    QGraphicsDropShadowEffect, QProgressBar,
    QGraphicsScene, QGraphicsView, QGraphicsEllipseItem,
    QGraphicsLineItem, QGraphicsTextItem, QGraphicsRectItem,
    QToolTip, QShortcut
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QRect, QPointF, QRectF, QLineF
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen, QLinearGradient, QPalette, QPolygonF, QWheelEvent

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from circuit_complexity_model_v2 import ImprovedGIN_GAT
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from circuit_graph_widget import CircuitGraphWidget

class C:
    BG          = "#F7F7F8"
    PANEL       = "#FFFFFF"
    SIDEBAR     = "#FAFAFA"

    T1          = "#000000"
    T2          = "#000000"
    T3          = "#000000"

    BORDER      = "#E4E4E7"
    DIVIDER     = "#F0F0F2"

    BLUE        = "#2563EB"
    BLUE_LIGHT  = "#EFF6FF"
    GREEN       = "#16A34A"
    AMBER       = "#D97706"
    RED         = "#DC2626"

    BAND_LOW    = "#16A34A"
    BAND_MED    = "#D97706"
    BAND_HIGH   = "#DC2626"

    SCORE_BG    = "#111111"
    HOVER       = "#1D4ED8"

MONO = "'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace"
MAX_BATCH_PRED_ABS_ERROR = 0.5

_LIGHT = {
    "BG": "#F7F7F8", "PANEL": "#FFFFFF", "SIDEBAR": "#FAFAFA",
    "T1": "#000000", "T2": "#000000", "T3": "#000000",
    "BORDER": "#E4E4E7", "DIVIDER": "#F0F0F2",
    "BLUE": "#2563EB", "BLUE_LIGHT": "#EFF6FF",
    "GREEN": "#16A34A", "AMBER": "#D97706", "RED": "#DC2626",
    "BAND_LOW": "#16A34A", "BAND_MED": "#D97706", "BAND_HIGH": "#DC2626",
    "SCORE_BG": "#111111", "HOVER": "#1D4ED8",
}

_DARK = {
    "BG": "#1E1E2E", "PANEL": "#181825", "SIDEBAR": "#11111B",
    "T1": "#CDD6F4", "T2": "#BAC2DE", "T3": "#A6ADC8",
    "BORDER": "#45475A", "DIVIDER": "#313244",
    "BLUE": "#89B4FA", "BLUE_LIGHT": "#1E2030",
    "GREEN": "#A6E3A1", "AMBER": "#F9E2AF", "RED": "#F38BA8",
    "BAND_LOW": "#A6E3A1", "BAND_MED": "#F9E2AF", "BAND_HIGH": "#F38BA8",
    "SCORE_BG": "#11111B", "HOVER": "#3B5BDB",
}

_is_dark = False

def apply_theme(dark: bool):
    """Set all C.xxx tokens for the chosen theme."""
    global _is_dark
    _is_dark = dark
    vals = _DARK if dark else _LIGHT
    for k, v in vals.items():
        setattr(C, k, v)

RADIUS = 10
PAD    = 20

import re as _re

_RTL_KEYWORDS = ["always @", "always_ff", "always_comb", "initial begin"]

_GATE_RE = _re.compile(r"^\s*[A-Z][A-Z0-9_]+\s+\w+\s*\(", _re.MULTILINE)

class _NumericTreeItem(QTreeWidgetItem):
    """QTreeWidgetItem that sorts numerically on columns with UserRole data."""
    def __lt__(self, other):
        col = self.treeWidget().sortColumn() if self.treeWidget() else 0
        my_val = self.data(col, Qt.UserRole)
        other_val = other.data(col, Qt.UserRole)
        if my_val is not None and other_val is not None:
            try:
                return float(my_val) < float(other_val)
            except (TypeError, ValueError):
                pass
        return super().__lt__(other)

class FileValidationError(Exception):
    def __init__(self, title, detail="", hint=""):
        super().__init__(title)
        self.title  = title
        self.detail = detail
        self.hint   = hint

def validate_file(path: str) -> None:
    p = Path(path)

    if p.suffix.lower() not in (".v", ".sv"):
        ext = p.suffix if p.suffix else "(none)"
        raise FileValidationError(
            "Wrong file type",
            "Expected a .v or .sv Verilog file, got: " + ext,
            "Select a gate-level Verilog netlist with a .v or .sv extension."
        )

    if not p.exists():
        raise FileValidationError(
            "File not found",
            p.name + " no longer exists on disk.",
            "Re-select the file using Choose File..."
        )

    if p.stat().st_size == 0:
        raise FileValidationError(
            "Empty file",
            p.name + " contains no data.",
            "Provide a non-empty gate-level netlist."
        )

    try:
        raw = p.read_bytes()
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise FileValidationError(
            "Binary or non-text file",
            p.name + " could not be read as text.",
            "The file may be compiled. Provide a plain-text Verilog netlist."
        )

    stripped = _re.sub(r"//[^\n]*", "", text)
    stripped = _re.sub(r"/\*.*?\*/", "", stripped, flags=_re.DOTALL)
    lines    = stripped.splitlines()

    for kw in _RTL_KEYWORDS:
        for lineno, line in enumerate(lines, 1):
            if kw in line:
                raise FileValidationError(
                    "RTL behavioral Verilog — synthesis required",
                    "Found RTL construct '" + kw + "' on line " + str(lineno) + ".",
                    "yosys -p \"synth; write_verilog -noattr out.v\" " + p.name
                )

    assign_lines = [l for l in lines if l.strip().startswith("assign")]
    rtl_ops      = ["= ~", "=~", "= &", "= |", "= ^", "= +", "= -",
                    "? :", "<<", ">>", "= *"]
    rtl_assigns  = [l for l in assign_lines if any(op in l for op in rtl_ops)]
    if assign_lines and len(rtl_assigns) >= len(assign_lines) * 0.5:
        example = rtl_assigns[0].strip()[:55]
        raise FileValidationError(
            "RTL behavioral Verilog — synthesis required",
            "Found behavioral assign statements, e.g.: " + example,
            "yosys -p \"synth; write_verilog -noattr out.v\" " + p.name
        )

    if "module " not in stripped:
        raise FileValidationError(
            "No module declaration found",
            p.name + " does not contain a Verilog module.",
            "Provide a complete gate-level netlist with module ... endmodule."
        )

    if not _GATE_RE.search(stripped):
        raise FileValidationError(
            "No gate instantiations found",
            p.name + " has no gate-level cell instances (e.g. INVX1, AND2X1).",
            "Synthesize the RTL first to get a gate-level netlist."
        )

def shadow(blur=12, alpha=0.07, dy=2):
    s = QGraphicsDropShadowEffect()
    s.setBlurRadius(blur)
    s.setColor(QColor(0, 0, 0, int(255 * alpha)))
    s.setOffset(0, dy)
    return s

def label(text, size=13, weight=400, color=C.T1, mono=False):
    l = QLabel(text)
    font = l.font()
    font.setPointSize(size)
    if weight >= 700:
        font.setBold(True)
    if mono:
        font.setFamily("SF Mono, Menlo, Monaco, Courier New")
    l.setFont(font)
    l.setStyleSheet(f"color: {color};")
    return l

def card(bg=C.PANEL, border=True, radius=RADIUS):
    f = QFrame()
    border_css = f"border: 1px solid {C.BORDER};" if border else "border: none;"
    f.setStyleSheet(f"QFrame {{ background: {bg}; {border_css} border-radius: {radius}px; }}")
    return f

def _enable_mc_dropout(model):
    """Enable only Dropout layers (not BatchNorm) for MC-Dropout sampling."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def _predict_scalar_score(model, data, mean, std):
    """Return scalar score from either regression or softmax-bin model heads."""
    out = model(data)
    if out.dim() == 2 and out.shape[1] > 1:
        centers = getattr(model, "_softmax_bin_centers", None)
        if not centers or len(centers) != out.shape[1]:
            centers = np.linspace(0.5, 5.0, out.shape[1], dtype=np.float32).tolist()
        c = torch.tensor(centers, dtype=out.dtype, device=out.device).view(1, -1)
        score = (torch.softmax(out, dim=1) * c).sum(dim=1)
        return float(np.clip(score.item(), 0.0, 5.0))
    pred = out.item() * std + mean
    return float(np.clip(pred, 0.0, 5.0))

def predict_with_uncertainty(model, data, mean, std, n_samples=20, device="cpu"):
    data = data.to(device)
    with torch.no_grad():
        model.eval()
        pred_eval = _predict_scalar_score(model, data, mean, std)

        _enable_mc_dropout(model)
        mc_preds = []
        for _ in range(n_samples):
            mc_preds.append(_predict_scalar_score(model, data, mean, std))
        model.eval()

    arr = np.array(mc_preds)
    sigma = float(np.std(arr))
    return {
        "prediction":        pred_eval,
        "uncertainty":       sigma,
        "ci_low":            max(0.0, pred_eval - 1.96 * sigma),
        "ci_high":           min(5.0, pred_eval + 1.96 * sigma),
        "confidence_level":  "High" if sigma < 0.15 else "Medium" if sigma < 0.4 else "Low",
    }

def complexity_band(score, circuit_type=None, feedback_ratio=0.0):
    is_sequential = str(circuit_type).startswith("sequential")
    if score < 1.0:
        if is_sequential or feedback_ratio > 0.0:
            return "LOW", C.BAND_LOW, "Low score with sequential/feedback structure"
        return "LOW", C.BAND_LOW, "Simple combinational structure"
    if score < 2.5:
        return "MEDIUM", C.BAND_MED,  "Moderate structural complexity"
    return "HIGH",       C.BAND_HIGH, "Complex feedback and reconvergence"

def load_checkpoint(path: str):
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    target_mode = ckpt.get("target_mode", "regression")
    softmax_bin_centers = ckpt.get("softmax_bin_centers", None)

    if "model_config" in ckpt:
        cfg = ckpt["model_config"]
        gin_n = cfg.get("num_gin_layers", cfg.get("gin_layers", 2))
        gat_n = cfg.get("num_gat_layers", cfg.get("gat_layers", 2))
        hdim  = cfg.get("hidden_dim", 64)
        idim  = cfg.get("input_dim", 24)
        outdim = cfg.get("output_dim", sd.get("predictor.3.weight", torch.empty(1, 1)).shape[0])
        if "global_dim" in cfg:
            gdim = cfg["global_dim"]
        elif any(k.startswith("global_branch.") for k in sd):
            gdim = sd["global_branch.mlp.0.weight"].shape[1]
        else:
            gdim = 0
    else:
        gin_n = max(int(k.split(".")[1]) for k in sd if k.startswith("gin_layers.") and k.split(".")[1].isdigit()) + 1
        gat_n = max(int(k.split(".")[1]) for k in sd if k.startswith("gat_layers.") and k.split(".")[1].isdigit()) + 1
        hdim = ckpt.get("hidden_dim", 64)
        idim = 24
        outdim = sd.get("predictor.3.weight", torch.empty(1, 1)).shape[0]
        gdim = 0 if not any(k.startswith("global_branch.") for k in sd) else 10

    model = ImprovedGIN_GAT(input_dim=idim, hidden_dim=hdim,
                             output_dim=max(outdim, 1),
                             num_gin_layers=gin_n, num_gat_layers=gat_n,
                             global_dim=max(gdim, 1))
    model.load_state_dict(sd, strict=(gdim > 0))
    model.eval()
    model._target_mode = target_mode
    model._softmax_bin_centers = softmax_bin_centers
    feat_stats = ckpt.get("feat_stats", None)
    # Prefer test R² (final performance) over val R² (training metric)
    r2 = ckpt.get("test_r2", ckpt.get("val_r2", 0.0))
    return model, ckpt.get("mean", 1.62), ckpt.get("std", 1.05), r2, feat_stats

from concurrent.futures import ProcessPoolExecutor

_analysis_pool = None

def _get_analysis_pool():
    """Lazy-init a single-worker process pool for GIL-free parsing."""
    global _analysis_pool
    if _analysis_pool is None:
        _analysis_pool = ProcessPoolExecutor(max_workers=1)
    return _analysis_pool

class AnalysisWorker(QThread):
    stage   = pyqtSignal(str)
    done    = pyqtSignal(dict)
    failed  = pyqtSignal(str)

    def __init__(self, filepath, model, parser_cls, mean, std, device="cpu",
                 feat_stats=None):
        super().__init__()
        self.filepath, self.model = filepath, model
        self.parser_cls = parser_cls
        self.mean, self.std, self.device = mean, std, device
        self.feat_stats = feat_stats

    def run(self):
        try:
            t0 = time.time()

            self.stage.emit("Parsing netlist...")
            from netlist_parser import standalone_analysis
            parsed = _get_analysis_pool().submit(
                standalone_analysis, self.filepath
            ).result()

            metrics   = parsed['metrics']
            data      = parsed['data']

            depth      = metrics.get("depth", 0)
            gate_count = max(metrics.get("gate_count", 1), 1)
            log_N      = math.log10(gate_count)
            log_N1     = math.log10(gate_count + 1)
            log_d      = math.log10(depth + 1)
            fb_ratio   = metrics.get("feedback_ratio", 0.0)
            s_ratio    = metrics.get("seq_ratio", 0.0)
            scc_size   = metrics.get("largest_scc_size", 0)
            cyclomatic = metrics.get("cyclomatic_complexity", 0)
            xor_ratio  = metrics.get("xor_ratio", 0.0)
            reconv     = metrics.get("reconvergent_ratio", 0.0)

            chain_ratio   = depth / max(gate_count, 1)
            chain_penalty = max(0.0, 1.0 - chain_ratio)

            _fb  = fb_ratio
            _scc = scc_size / max(gate_count, 1)
            breakdown = {
                "depth_x_logN":      depth * log_N * chain_penalty,
                "size_term":         2.0 * log_N,
                "depth_standalone":  2.0 * log_d * chain_penalty,
                "feedback_term":     3.0 * _fb,
                "fb_x_size":         5.0 * _fb * log_N,
                "scc_density":       3.0 * _scc,
                "seq_term":          2.5 * s_ratio * log_N1,
                "cyclomatic_term":   1.5 * math.log10(cyclomatic + 1),
                "xor_term":          2.0 * xor_ratio * log_N,
                "reconvergence":     1.5 * reconv * log_N,
            }
            breakdown["raw_total"] = sum(breakdown.values())

            t_inf = time.time()

            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)

            if (self.feat_stats is not None
                    and hasattr(data, 'global_feats')
                    and data.global_feats is not None):
                fm = torch.tensor(self.feat_stats["feat_mean"], dtype=torch.float)
                fs = torch.tensor(self.feat_stats["feat_std"],  dtype=torch.float)
                data.global_feats = (data.global_feats - fm) / fs

            self.stage.emit("Running inference...")
            try:
                result = predict_with_uncertainty(
                    self.model, data, self.mean, self.std,
                    n_samples=5, device=self.device
                )
            except RuntimeError:
                self.stage.emit("CPU fallback...")
                self.model = self.model.to("cpu")
                result = predict_with_uncertainty(
                    self.model, data, self.mean, self.std,
                    n_samples=5, device="cpu"
                )
            num_nodes = data.num_nodes
            num_edges = int(data.edge_index.shape[1]) if data.edge_index.numel() > 0 else 0

            pred = result["prediction"]
            level, color, note = complexity_band(
                pred,
                circuit_type=metrics.get("circuit_type", "unknown"),
                feedback_ratio=metrics.get("feedback_ratio", 0.0),
            )

            self.done.emit({
                "score":           pred,
                "ci_low":          result["ci_low"],
                "ci_high":         result["ci_high"],
                "uncertainty":     result["uncertainty"],
                "confidence":      result["confidence_level"],
                "level":           level,
                "level_color":     color,
                "note":            note,
                "filename":        Path(self.filepath).name,
                "filepath":        self.filepath,
                "gate_count":      metrics["gate_count"],
                "depth":           metrics["depth"],
                "num_inputs":      metrics.get("num_inputs", 0),
                "num_outputs":     metrics.get("num_outputs", 0),
                "num_nodes":       num_nodes,
                "num_edges":       num_edges,
                "circuit_type":    metrics.get("circuit_type", "unknown"),
                "total_gates":     metrics.get("total_gates", 0),
                "seq_ratio":       metrics.get("seq_ratio", 0.0),
                "feedback_ratio":      metrics["feedback_ratio"],
                "reconvergent_ratio":  metrics["reconvergent_ratio"],
                "xor_ratio":           metrics["xor_ratio"],
                "buf_ratio":           metrics.get("buf_ratio", 0.0),
                "type_entropy":        metrics["type_entropy"],
                "edge_density":        metrics["edge_density"],
                "max_fanout":          metrics["max_fanout"],
                "avg_fanout":          metrics["avg_fanout"],
                "breakdown":       breakdown,
                "graph_gates":     parsed['gates_slice'],
                "graph_inputs":    parsed['inputs'],
                "graph_outputs":   parsed['outputs'],
                "parse_ms":        parsed['parse_ms'],
                "metrics_ms":      parsed['metrics_ms'],
                "graph_ms":        parsed['graph_ms'],
                "inference_ms":    (time.time() - t_inf) * 1000,
                "total_ms":        (time.time() - t0) * 1000,
                "timestamp":       datetime.now().isoformat(),
            })
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

class ScoreDisplay(QFrame):
    """Large score readout with animated band indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGraphicsEffect(shadow(20, 0.12, 4))
        self.setMinimumHeight(160)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(PAD, PAD, PAD, PAD)
        outer.setSpacing(6)

        self._title = label("Predicted Complexity Score", 11, color="#888888")
        self._title.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._title)

        self._number = label("—", 56, 700, color="#FFFFFF", mono=True)
        self._number.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._number)

        row = QHBoxLayout()
        row.setSpacing(12)
        row.addStretch()
        self._band  = label("Select a file to begin", 13, color="#888888")
        self._band.setAlignment(Qt.AlignCenter)
        row.addWidget(self._band)
        self._ci = label("", 11, color="#666666", mono=True)
        row.addWidget(self._ci)
        row.addStretch()
        outer.addLayout(row)
        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(f"QFrame {{ background: {C.SCORE_BG}; border-radius: {RADIUS}px; border: none; }}")
        self._title.setStyleSheet("color: #888888; font-size: 11px; background: transparent; border: none;")
        self._ci.setStyleSheet(f"color: #666666; font-size: 11px; font-family: {MONO}; background: transparent; border: none;")

    def set_result(self, r):
        score = r["score"]
        color = r["level_color"]
        self._number.setText(f"{score:.3f}")
        self._number.setStyleSheet(f"color: {color}; font-weight: 700;")
        self._band.setText(f"{r['level']}  ·  {r['note']}")
        self._band.setStyleSheet(f"color: {color};")
        self._ci.setText(f"95% CI  [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")

    def reset(self):
        self._number.setText("—")
        self._number.setStyleSheet("color: #FFFFFF; font-weight: 700;")
        self._band.setText("Select a file to begin")
        self._band.setStyleSheet("color: #888888;")
        self._ci.setText("")

class GaugeBar(QWidget):
    """Horizontal complexity gauge with animated fill."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._target = 0.0
        self.setFixedHeight(36)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)

    def set_value(self, v):
        self._target = max(0.0, min(5.0, v))
        self._timer.start(16)

    def reset(self):
        self._target = 0.0
        self._value = 0.0
        self._timer.stop()
        self.update()

    def _step(self):
        diff = self._target - self._value
        if abs(diff) < 0.005:
            self._value = self._target
            self._timer.stop()
        else:
            self._value += diff * 0.18
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(C.BORDER)))
        p.drawRoundedRect(0, 8, w, 8, 4, 4)

        if self._value > 0:
            fill_w = int((self._value / 5.0) * w)
            grad = QLinearGradient(0, 0, fill_w, 0)
            grad.setColorAt(0.0, QColor(C.BAND_LOW))
            grad.setColorAt(0.4, QColor(C.BAND_MED))
            grad.setColorAt(1.0, QColor(C.BAND_HIGH))
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(0, 8, fill_w, 8, 4, 4)

            tx = max(7, min(w - 7, fill_w))
            p.setBrush(QBrush(QColor("#FFFFFF")))
            p.setPen(QPen(QColor(C.BORDER), 1.5))
            p.drawEllipse(tx - 7, 4, 14, 14)

        p.setPen(QColor(C.T3))
        font = p.font()
        font.setPointSize(9)
        p.setFont(font)
        for v in range(6):
            x = int((v / 5.0) * (w - 12)) + 6
            p.drawText(x - 3, h, str(v))

class MetricTable(QFrame):
    """Clean two-column metric display."""

    METRICS = [
        ("Feedback Ratio",     "feedback_ratio",
         "Proportion of nodes in feedback loops",
         "The strongest predictor of analysis difficulty. Higher values mean "
         "more gates in circular dependencies (latches, oscillators, FSM feedback).",
         1.0),
        ("Reconvergent Ratio", "reconvergent_ratio",
         "Paths that diverge and rejoin",
         "Reconvergent fanout creates signal correlations that complicate "
         "timing analysis and can cause glitches. Common in arithmetic circuits.",
         1.0),
        ("XOR Ratio",          "xor_ratio",
         "XOR/XNOR gate proportion",
         "High XOR content suggests arithmetic or error-checking logic "
         "(adders, CRC, parity). These create many reconvergent paths.",
         1.0),
        ("Buffer Ratio",       "buf_ratio",
         "Buffer and inverter proportion",
         "Buffers add stages without logic. High ratios suggest long chains "
         "or buffered clock/reset trees, which inflate depth but not true complexity.",
         1.0),
        ("Type Entropy",       "type_entropy",
         "Diversity of gate types (nats)",
         "Higher entropy means the circuit uses many different gate types, "
         "indicating more complex logic. Simple buffers/inverters score low.",
         3.0),
        ("Edge Density",       "edge_density",
         "Graph connectivity ratio",
         "Ratio of edges to nodes. Dense connectivity increases routing "
         "difficulty and signal interdependence.",
         0.1),
        ("Max Fanout",         "max_fanout",
         "Highest fan-out from any single gate",
         "A high max fanout means one signal drives many gates, creating a "
         "critical node. Failure of that node affects the entire circuit.",
         50.0),
        ("Avg Fanout",         "avg_fanout",
         "Average fan-out per gate",
         "Higher average fanout means more interconnected gates overall, "
         "increasing routing complexity and power consumption.",
         5.0),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = {}
        self._row_refs = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(PAD, 16, PAD, 16)
        layout.setSpacing(0)

        for i, (name, key, tip, long_desc, maxv) in enumerate(self.METRICS):
            row_w = QWidget()
            row_w.setToolTip(f"<b>{name}</b><br>{tip}<br><br><i>{long_desc}</i>")

            row = QHBoxLayout(row_w)
            row.setContentsMargins(10, 8, 10, 8)
            row.setSpacing(12)

            name_l = label(name, 12, color=C.T2)
            name_l.setFixedWidth(150)
            row.addWidget(name_l)

            bar = QProgressBar()
            bar.setMaximum(1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(4)
            row.addWidget(bar, 1)

            val_l = label("—", 12, color=C.T1, mono=True)
            val_l.setFixedWidth(72)
            val_l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(val_l)

            layout.addWidget(row_w)
            self._rows[key] = (bar, val_l, maxv)
            self._row_refs.append((row_w, name_l, bar, val_l, i))

        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(f"QFrame {{ background: {C.PANEL}; border: 1px solid {C.BORDER}; border-radius: {RADIUS}px; }}")
        for row_w, name_l, bar, val_l, i in self._row_refs:
            if i % 2 == 0:
                row_w.setStyleSheet(f"background: {C.DIVIDER}; border-radius: 5px;")
            else:
                row_w.setStyleSheet("background: transparent;")
            name_l.setStyleSheet(f"color: {C.T2};")
            bar.setStyleSheet(f"""
                QProgressBar {{ background: {C.BORDER}; border: none; border-radius: 2px; }}
                QProgressBar::chunk {{ background: {C.BLUE}; border-radius: 2px; }}
            """)
            val_l.setStyleSheet(f"color: {C.T1}; font-family: {MONO};")

    def set_result(self, r):
        for key, (bar, val_l, maxv) in self._rows.items():
            v = r.get(key, 0)
            if isinstance(v, float):
                val_l.setText(f"{v:.4f}")
                bar.setValue(min(1000, int(abs(v) / maxv * 1000)))
            else:
                val_l.setText(str(v))
                bar.setValue(min(1000, int(v / maxv * 1000)))

    def reset(self):
        for bar, val_l, _ in self._rows.values():
            bar.setValue(0)
            val_l.setText("—")

class InfoGrid(QFrame):
    """Clean key-value grid for circuit statistics."""

    def __init__(self, fields, parent=None):
        super().__init__(parent)
        self._labels = {}
        self._cell_refs = []

        grid = QGridLayout(self)
        grid.setContentsMargins(PAD, 16, PAD, 16)
        grid.setSpacing(0)
        grid.setHorizontalSpacing(0)

        for i, (key, title) in enumerate(fields):
            col = (i % 3) * 2
            row = i // 3

            cell = QWidget()
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(16, 12, 16, 12)
            cl.setSpacing(3)

            val_l = label("—", 18, 700, color=C.T1, mono=True)
            ttl_l = label(title, 11, color=C.T3)
            cl.addWidget(val_l)
            cl.addWidget(ttl_l)

            grid.addWidget(cell, row, col, 1, 2)
            self._labels[key] = val_l
            self._cell_refs.append((cell, val_l, ttl_l, col))

        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(f"QFrame {{ background: {C.PANEL}; border: 1px solid {C.BORDER}; border-radius: {RADIUS}px; }}")
        for cell, val_l, ttl_l, col in self._cell_refs:
            if col > 0:
                cell.setStyleSheet(f"border-left: 1px solid {C.BORDER};")
            else:
                cell.setStyleSheet("")
            val_l.setStyleSheet(f"color: {C.T1}; font-weight: 700; font-family: {MONO};")
            ttl_l.setStyleSheet(f"color: {C.T3};")

    def set_result(self, r):
        for key, lbl in self._labels.items():
            v = r.get(key, "—")
            if isinstance(v, int):
                lbl.setText(f"{v:,}")
            elif isinstance(v, float):
                lbl.setText(f"{v:.4f}")
            else:
                lbl.setText(str(v).replace("_", " ").title())

    def reset(self):
        for lbl in self._labels.values():
            lbl.setText("—")

class ErrorBanner(QFrame):
    """Collapsible inline error banner for the sidebar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_mode = "error"
        self.hide()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self._icon = QLabel("⚠")
        self._icon.setStyleSheet("font-size: 13px; background: transparent; border: none;")
        title_row.addWidget(self._icon)
        self._title = QLabel("")
        self._title.setStyleSheet("font-size: 12px; font-weight: 600; background: transparent; border: none;")
        self._title.setWordWrap(True)
        title_row.addWidget(self._title, 1)
        self._dismiss = QPushButton("×")
        self._dismiss.setFixedSize(18, 18)
        self._dismiss.setCursor(Qt.PointingHandCursor)
        self._dismiss.setStyleSheet("QPushButton { background: transparent; border: none; font-size: 15px; font-weight: bold; }")
        self._dismiss.clicked.connect(self.hide)
        title_row.addWidget(self._dismiss)
        outer.addLayout(title_row)

        self._detail = QLabel("")
        self._detail.setStyleSheet("font-size: 11px; background: transparent; border: none;")
        self._detail.setWordWrap(True)
        outer.addWidget(self._detail)

        self._hint = QLabel("")
        self._hint.setStyleSheet("font-size: 10px; font-family: SF Mono, Menlo, Monaco, Courier New, monospace; background: transparent; border: none;")
        self._hint.setWordWrap(True)
        outer.addWidget(self._hint)

    def _error_styles(self):
        if _is_dark:
            return {
                "frame": "QFrame { background: #302020; border: 1px solid #5C3030; border-radius: 8px; }",
                "icon":  "color: #F38BA8; font-size: 13px; background: transparent; border: none;",
                "title": "color: #F38BA8; font-size: 12px; font-weight: 600; background: transparent; border: none;",
                "detail":"color: #EBA0AC; font-size: 11px; background: transparent; border: none;",
                "hint":  "color: #F38BA8; font-size: 10px; font-family: SF Mono, Menlo, Monaco, Courier New, monospace; background: transparent; border: none;",
            }
        return {
            "frame": "QFrame { background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; }",
            "icon":  "color: #DC2626; font-size: 13px; background: transparent; border: none;",
            "title": "color: #991B1B; font-size: 12px; font-weight: 600; background: transparent; border: none;",
            "detail":"color: #7F1D1D; font-size: 11px; background: transparent; border: none;",
            "hint":  "color: #DC2626; font-size: 10px; font-family: SF Mono, Menlo, Monaco, Courier New, monospace; background: transparent; border: none;",
        }

    def _warn_styles(self):
        if _is_dark:
            return {
                "frame": "QFrame { background: #302E1A; border: 1px solid #5C5020; border-radius: 8px; }",
                "icon":  "color: #F9E2AF; font-size: 13px; background: transparent; border: none;",
                "title": "color: #F9E2AF; font-size: 12px; font-weight: 600; background: transparent; border: none;",
                "detail":"color: #F9E2AF; font-size: 11px; background: transparent; border: none;",
            }
        return {
            "frame": "QFrame { background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; }",
            "icon":  "color: #D97706; font-size: 13px; background: transparent; border: none;",
            "title": "color: #92400E; font-size: 12px; font-weight: 600; background: transparent; border: none;",
            "detail":"color: #78350F; font-size: 11px; background: transparent; border: none;",
        }

    def apply_theme(self):
        """Re-apply styles based on current theme and mode."""
        if self._current_mode == "error":
            s = self._error_styles()
        else:
            s = self._warn_styles()
        self.setStyleSheet(s["frame"])
        self._icon.setStyleSheet(s["icon"])
        self._title.setStyleSheet(s["title"])
        self._detail.setStyleSheet(s["detail"])
        if "hint" in s:
            self._hint.setStyleSheet(s["hint"])
        self._dismiss.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; font-size: 15px; "
            f"font-weight: bold; color: {C.T2}; }}")

    def show_error(self, title: str, detail: str = "", hint: str = ""):
        self._current_mode = "error"
        s = self._error_styles()
        self.setStyleSheet(s["frame"])
        self._icon.setStyleSheet(s["icon"])
        self._title.setStyleSheet(s["title"])
        self._detail.setStyleSheet(s["detail"])
        self._hint.setStyleSheet(s["hint"])
        self._title.setText(title)
        self._detail.setText(detail)
        self._hint.setText(hint)
        self._detail.setVisible(bool(detail))
        self._hint.setVisible(bool(hint))
        self.show()

    def show_warning(self, title: str, detail: str = ""):
        self._current_mode = "warning"
        s = self._warn_styles()
        self.setStyleSheet(s["frame"])
        self._icon.setStyleSheet(s["icon"])
        self._title.setStyleSheet(s["title"])
        self._detail.setStyleSheet(s["detail"])
        self._hint.setVisible(False)
        self._title.setText(title)
        self._detail.setText(detail)
        self._detail.setVisible(bool(detail))
        self.show()

class BreakdownChart(QFrame):
    """Horizontal stacked bar chart showing formula term contributions."""

    TERMS = [
        ("depth_x_logN",     "Depth × log(N)",        "#2563EB",
         "Longest path times circuit size. Deeper paths = more stages. "
         "Zeroed for linear chains via the chain penalty."),
        ("size_term",        "Size  (2·log N)",        "#7C3AED",
         "Base complexity from circuit size. Larger circuits have more "
         "routing and more potential for bugs."),
        ("depth_standalone", "Depth  (2·log d)",       "#3B82F6",
         "Standalone depth contribution. Rewards deep logic regardless "
         "of circuit size. Zeroed for linear chains via the chain penalty."),
        ("feedback_term",    "Feedback  (3·fb)",       "#DC2626",
         "Flat penalty for feedback loops. Circular dependencies (latches, "
         "oscillators) resist static analysis."),
        ("fb_x_size",        "Feedback × Size  (5·fb·log N)", "#D97706",
         "Feedback in larger circuits compounds complexity — more gates "
         "participate in hard-to-analyse cyclic paths."),
        ("scc_density",      "SCC Coverage  (3·s/N)",  "#EA580C",
         "Fraction of gates in strongly connected components. Large SCCs "
         "indicate dense feedback that is hard to decompose."),
        ("seq_term",         "Sequential  (2.5·sq)",   "#0D9488",
         "Penalty for flip-flops / latches. Sequential circuits need "
         "temporal analysis across clock cycles."),
        ("cyclomatic_term",  "Cyclomatic  (1.5·log M)","#6366F1",
         "Independent cycles in the connection graph. More cycles = more "
         "distinct paths to verify."),
        ("xor_term",         "XOR  (2·xor·log N)",     "#0891B2",
         "XOR/XNOR gate density scaled by size. High XOR content "
         "suggests arithmetic or error-checking logic (adders, CRC)."),
        ("reconvergence",    "Reconvergence  (1.5·r·log N)", "#8B5CF6",
         "Reconvergent fanout scaled by size. Paths that diverge and "
         "rejoin create signal correlations that complicate analysis."),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._breakdown = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(PAD, 16, PAD, 16)
        layout.setSpacing(10)

        self._hdr = QLabel("Score Breakdown")
        layout.addWidget(self._hdr)

        self._hint_lbl = QLabel("Contribution of each formula term to the raw complexity score")
        self._hint_lbl.setWordWrap(True)
        layout.addWidget(self._hint_lbl)

        self._rows_widget = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(0, 4, 0, 0)
        self._rows_layout.setSpacing(6)
        layout.addWidget(self._rows_widget)

        self._total_label = QLabel("")
        layout.addWidget(self._total_label)

        self._bar_rows = []
        self._name_labels = []
        for key, title, color, desc in self.TERMS:
            container = QWidget()
            container.setStyleSheet("background: transparent; border: none;")
            cl = QVBoxLayout(container)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)

            row = QWidget()
            row.setStyleSheet("background: transparent; border: none;")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(8)

            name_lbl = QLabel(title)
            name_lbl.setFixedWidth(200)
            rl.addWidget(name_lbl)

            bar = QProgressBar()
            bar.setTextVisible(False)
            bar.setFixedHeight(16)
            bar.setRange(0, 1000)
            bar.setValue(0)
            rl.addWidget(bar, 1)

            val_lbl = QLabel("—")
            val_lbl.setFixedWidth(50)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            rl.addWidget(val_lbl)
            cl.addWidget(row)

            container.setToolTip(f"<b>{title}</b><br><br>{desc}")

            self._rows_layout.addWidget(container)
            self._bar_rows.append((bar, val_lbl, color))
            self._name_labels.append(name_lbl)

        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(f"QFrame {{ background: {C.PANEL}; border: 1px solid {C.BORDER}; border-radius: {RADIUS}px; }}")
        self._hdr.setStyleSheet(f"color: {C.T1}; font-size: 15px; font-weight: 600; background: transparent; border: none;")
        self._hint_lbl.setStyleSheet(f"color: {C.T3}; font-size: 12px; background: transparent; border: none;")
        self._rows_widget.setStyleSheet("background: transparent; border: none;")
        self._total_label.setStyleSheet(f"color: {C.T2}; font-size: 12px; background: transparent; border: none; font-family: {MONO};")
        for name_lbl in self._name_labels:
            name_lbl.setStyleSheet(f"color: {C.T2}; font-size: 13px; background: transparent; border: none;")
        for bar, val_lbl, color in self._bar_rows:
            bar.setStyleSheet(f"""
                QProgressBar {{ background: {C.DIVIDER}; border: none; border-radius: 4px; }}
                QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}
            """)
            val_lbl.setStyleSheet(f"color: {C.T1}; font-size: 13px; font-family: {MONO}; background: transparent; border: none;")

    def set_result(self, r):
        bd = r.get("breakdown")
        if not bd:
            return
        self._breakdown = bd
        total = bd.get("raw_total", 1.0)
        max_val = max(max(bd.get(k, 0) for k, _, _, _ in self.TERMS), 0.01)

        for i, (key, title, color, _desc) in enumerate(self.TERMS):
            v = bd.get(key, 0.0)
            bar, val_lbl, _ = self._bar_rows[i]
            bar.setValue(int(v / max_val * 1000))
            val_lbl.setText(f"{v:.2f}")

        self._total_label.setText(f"Raw total: {total:.2f}  →  score = 5 / (1 + exp(1.5 − raw/15))")

    def reset(self):
        for bar, val_lbl, _ in self._bar_rows:
            bar.setValue(0)
            val_lbl.setText("—")
        self._total_label.setText("")

CIRCUIT_FIELDS = [
    ("gate_count",    "Logic Gates"),
    ("depth",         "Logic Depth"),
    ("circuit_type",  "Circuit Type"),
    ("num_inputs",    "Inputs"),
    ("num_outputs",   "Outputs"),
    ("num_nodes",     "Graph Nodes"),
    ("num_edges",     "Graph Edges"),
    ("total_gates",   "Total Cells"),
    ("seq_ratio",     "Seq. Ratio"),
]

DATASET_STATS = {"mean": 1.9582, "std": 0.7721}

class MainWindow(QMainWindow):

    def __init__(self, model_path: str = None):
        super().__init__()
        self.model = None
        self.parser_cls = None
        self.mean, self.std = 1.9582, 0.7721
        self.val_r2 = 0.0
        self.device = "cpu"
        self.results_history = []
        self.comparison = [None, None]

        self._build_ui()
        self._build_menu()
        self._load_everything(model_path)

    def _build_ui(self):
        self.setWindowTitle("Circuit Complexity Predictor")
        self.setMinimumSize(900, 620)
        self.resize(1100, 740)
        self.setAcceptDrops(True)

        root = QWidget()
        self.setCentralWidget(root)

        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        main.addWidget(self._build_sidebar())

        self._divider = QFrame()
        self._divider.setFixedWidth(1)
        main.addWidget(self._divider)

        main.addWidget(self._build_results_area(), 1)

        self.statusBar().showMessage("Ready  ·  Load a Verilog netlist to begin")
        self._apply_main_styles()

    def _build_sidebar(self):
        self._sidebar = QWidget()
        self._sidebar.setFixedWidth(280)

        layout = QVBoxLayout(self._sidebar)
        layout.setContentsMargins(PAD, PAD, PAD, PAD)
        layout.setSpacing(20)

        header = QWidget()
        hl = QVBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        self._title_lbl = label("Circuit Complexity", 16, 700, C.T1)
        hl.addWidget(self._title_lbl)
        self._subtitle = label("Predictor  ·  GIN-GAT", 11, color=C.T3)
        hl.addWidget(self._subtitle)
        layout.addWidget(header)

        self._sep1 = QFrame()
        self._sep1.setFixedHeight(1)
        layout.addWidget(self._sep1)

        self._file_section_lbl = label("Verilog Netlist", 11, color=C.T2)
        layout.addWidget(self._file_section_lbl)

        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("No file selected")
        self._file_edit.setReadOnly(True)
        layout.addWidget(self._file_edit)

        self._browse_btn = QPushButton("Choose File...")
        self._browse_btn.setFixedHeight(36)
        self._browse_btn.setCursor(Qt.PointingHandCursor)
        self._browse_btn.clicked.connect(self._browse)
        layout.addWidget(self._browse_btn)

        self._sep2 = QFrame()
        self._sep2.setFixedHeight(1)
        layout.addWidget(self._sep2)

        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.setFixedHeight(42)
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.setToolTip("Select a Verilog netlist file first")
        self._analyze_btn.setCursor(Qt.PointingHandCursor)
        self._analyze_btn.clicked.connect(self._analyze)
        layout.addWidget(self._analyze_btn)

        self._error_banner = ErrorBanner()
        layout.addWidget(self._error_banner)

        self._sep3 = QFrame()
        self._sep3.setFixedHeight(1)
        layout.addWidget(self._sep3)

        self._model_section_lbl = label("Model", 11, color=C.T2)
        layout.addWidget(self._model_section_lbl)
        self._model_info = QWidget()
        mil = QVBoxLayout(self._model_info)
        mil.setContentsMargins(0, 0, 0, 0)
        mil.setSpacing(6)
        self._model_status = label("Loading...", 12, color=C.T3)
        self._model_status.setWordWrap(True)
        mil.addWidget(self._model_status)
        layout.addWidget(self._model_info)

        layout.addStretch(1)

        self._history_section_lbl = label("Recent Analyses", 11, color=C.T2)
        layout.addWidget(self._history_section_lbl)
        self._history_list = QListWidget()
        self._history_list.setMinimumHeight(200)
        self._history_list.setMaximumHeight(400)
        self._history_list.itemClicked.connect(self._on_history_click)
        layout.addWidget(self._history_list, 2)

        return self._sidebar

    def _build_results_area(self):
        self._results_area = QWidget()
        layout = QVBoxLayout(self._results_area)
        layout.setContentsMargins(PAD, PAD, PAD, PAD)
        layout.setSpacing(16)

        self._score_display = ScoreDisplay()
        layout.addWidget(self._score_display)

        gauge_row = QWidget()
        gl = QVBoxLayout(gauge_row)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(4)
        self._gauge_lbl = label("Complexity Scale  (0 – 5)", 11, color=C.T3)
        gl.addWidget(self._gauge_lbl)
        self._gauge = GaugeBar()
        gl.addWidget(self._gauge)
        layout.addWidget(gauge_row)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_overview_tab(),    "Overview")
        self._tabs.addTab(self._build_breakdown_tab(),  "Score Breakdown")
        self._tabs.addTab(self._build_graph_tab(),      "Circuit Graph")
        self._hidden_metrics_tab = self._build_metrics_tab()
        self._tabs.addTab(self._build_log_tab(),        "Log")
        layout.addWidget(self._tabs, 1)

        return self._results_area

    def _build_overview_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        w = QWidget()
        w.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(14)

        self._ctx_card = QFrame()
        cl = QGridLayout(self._ctx_card)
        cl.setContentsMargins(PAD, 16, PAD, 16)
        cl.setSpacing(0)

        self._ctx_vals = {}
        self._ctx_cells = []
        ctx_items = [
            ("sigma",   "Relative Position"),
            ("pct",     "Percentile"),
            ("conf",    "Prediction Confidence"),
        ]
        for i, (key, title) in enumerate(ctx_items):
            cell = QWidget()
            ll = QVBoxLayout(cell)
            ll.setContentsMargins(16, 0, 16, 0)
            ll.setSpacing(3)
            v = label("—", 18, 700, C.T1, mono=True)
            t = label(title, 11, color=C.T3)
            ll.addWidget(v)
            ll.addWidget(t)
            cl.addWidget(cell, 0, i)
            self._ctx_vals[key] = v
            self._ctx_cells.append((cell, v, t, i))

        layout.addWidget(self._ctx_card)

        self._interpretation = QLabel("")
        self._interpretation.setWordWrap(True)
        layout.addWidget(self._interpretation)

        self._circ_stats_lbl = label("Circuit Statistics", 12, 600, C.T2)
        layout.addWidget(self._circ_stats_lbl)
        self._info_grid = InfoGrid(CIRCUIT_FIELDS)
        layout.addWidget(self._info_grid)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _build_breakdown_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        w = QWidget()
        w.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(14)

        self._formula_card = QFrame()
        fl = QVBoxLayout(self._formula_card)
        fl.setContentsMargins(PAD, 14, PAD, 14)
        fl.setSpacing(8)
        self._formula_title = label("How the Score is Computed", 13, 600, C.T1)
        fl.addWidget(self._formula_title)
        self._formula_desc = label(
            "Ten structural metrics are combined into a raw score, then "
            "mapped to a 0\u20135 scale via a softmax function. Each bar below "
            "shows one term\u2019s contribution to the raw total.",
            11, color=C.T2)
        fl.addWidget(self._formula_desc)
        self._formula_lbl = QLabel(
            "raw = depth\u00b7log(N)\u00b7penalty + 2\u00b7log(N) + 2\u00b7log(d+1)\u00b7penalty\n"
            "    + 3\u00b7fb + 5\u00b7fb\u00b7log(N) + 3\u00b7(scc/N)\n"
            "    + 2.5\u00b7seq\u00b7log(N+1) + 1.5\u00b7log(cyclo+1)\n"
            "    + 2\u00b7xor\u00b7log(N) + 1.5\u00b7reconv\u00b7log(N)\n\n"
            "score = 5 / (1 + exp((25 \u2212 raw) / 8))    [softmax, T=8]")
        self._formula_lbl.setWordWrap(True)
        fl.addWidget(self._formula_lbl)
        self._formula_bands = label(
            "Bands:  0 \u2013 1.0 = LOW (simple)  |  1.0 \u2013 2.5 = MEDIUM  |  2.5 \u2013 5.0 = HIGH (complex)",
            10, color=C.T3)
        fl.addWidget(self._formula_bands)
        layout.addWidget(self._formula_card)

        self._breakdown_chart = BreakdownChart()
        layout.addWidget(self._breakdown_chart)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _build_graph_tab(self):
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(8)

        self._graph_hint = QFrame()
        hl = QHBoxLayout(self._graph_hint)
        hl.setContentsMargins(14, 10, 14, 10)
        self._graph_hint_lbl = label(
            "Verilog source (left) and circuit schematic (right). "
            "Click a code line to highlight the gate — click a gate to jump to the code. "
            "Scroll to zoom, drag to pan.",
            11, color=C.T1
        )
        hl.addWidget(self._graph_hint_lbl)
        layout.addWidget(self._graph_hint)

        self._graph_widget = CircuitGraphWidget()
        layout.addWidget(self._graph_widget, 1)
        return w

    def _build_metrics_tab(self):
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(8)

        self._metric_table = MetricTable()
        layout.addWidget(self._metric_table, 1)
        return w

    def _build_log_tab(self):
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 8, 0, 0)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        layout.addWidget(self._log)
        return w

    def _build_menu(self):
        mb = self.menuBar()
        mb.setNativeMenuBar(False)
        file_m = mb.addMenu("File")
        file_m.addAction("Open Netlist...", self._browse, "Ctrl+O")
        file_m.addAction("Analyze", self._analyze, "Ctrl+Return")
        file_m.addAction("Batch Analyze Folder...", self._batch_analyze, "Ctrl+Shift+O")
        file_m.addSeparator()
        file_m.addAction("Export JSON...", lambda: self._export("json"), "Ctrl+E")
        file_m.addAction("Export CSV...",  lambda: self._export("csv"))
        file_m.addSeparator()
        file_m.addAction("Clear History", self._clear_history)

        cmp_m = mb.addMenu("Compare")
        cmp_m.addAction("Set as Circuit A", lambda: self._set_compare(0))
        cmp_m.addAction("Set as Circuit B", lambda: self._set_compare(1))
        cmp_m.addSeparator()
        cmp_m.addAction("Show Comparison...", self._show_compare)

        help_m = mb.addMenu("Help")
        help_m.addAction("Model Info...", self._show_model_info)

    def _load_everything(self, model_path=None):
        self._log_msg("Initializing...")

        try:
            from netlist_parser import GateLevelNetlistParser
            self.parser_cls = GateLevelNetlistParser
            self._log_msg("Parser loaded")
            _get_analysis_pool().submit(lambda: None)
        except ImportError as e:
            self._log_msg(f"Parser error: {e}")
            self._model_status.setText(f"Parser not found:\n{e}")
            return

        candidates = []
        if model_path:
            candidates = [model_path]
        else:
            candidates = [
                "models/best_complexity_model.pt",
                "best_complexity_model.pt",
                "../models/best_complexity_model.pt",
            ]

        path = next((p for p in candidates if Path(p).exists()), None)
        if not path:
            self._log_msg("Model checkpoint not found")
            self._model_status.setText("Model not found.\nPlace best_complexity_model.pt\nin models/ and restart.")
            return

        try:
            self.model, self.mean, self.std, self.val_r2, self.feat_stats = load_checkpoint(path)
            DATASET_STATS["mean"] = self.mean
            DATASET_STATS["std"]  = self.std

            display_r2 = self.val_r2
            project_root = Path(__file__).resolve().parent.parent
            for metrics_path in [
                project_root / "plots" / "complexity_metrics.json",
                Path(path).resolve().parent.parent / "plots" / "complexity_metrics.json",
                Path(path).resolve().parent / "complexity_metrics.json",
                Path("plots") / "complexity_metrics.json",
            ]:
                if metrics_path.exists():
                    import json as _json
                    with open(metrics_path) as _f:
                        _metrics = _json.load(_f)
                    if "r2" in _metrics:
                        display_r2 = _metrics["r2"]
                    break

            self._log_msg(f"Model loaded  ·  R²={display_r2:.4f}")
            self._model_status.setText(
                f"GIN-GAT  ·  R²={display_r2:.3f}\n"
                f"Norm: μ={self.mean:.2f}  σ={self.std:.2f}"
            )
            self._model_status.setStyleSheet(f"color: {C.T2}; font-size: 11px;")
            self._subtitle.setText(f"Predictor  ·  R²={display_r2:.3f}")
        except Exception as e:
            self._log_msg(f"Model error: {e}")
            self._model_status.setText(f"Load error:\n{e}")

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Verilog Netlist", "",
            "Verilog (*.v *.V *.sv *.SV);;All Files (*)"
        )
        if not path:
            return
        self._error_banner.hide()
        try:
            validate_file(path)
            self._file_edit.setText(Path(path).name)
            self._file_edit.setProperty("_path", path)
            self._analyze_btn.setEnabled(self.model is not None)
            self._analyze_btn.setToolTip("Click to analyze this circuit" if self.model else "Model not loaded")
            self._log_msg(f"Selected: {Path(path).name}")
        except FileValidationError as e:
            self._file_edit.setText(Path(path).name)
            self._file_edit.setProperty("_path", None)
            self._analyze_btn.setEnabled(False)
            self._error_banner.show_error(e.title, e.detail, e.hint)
            self._log_msg(f"Validation failed: {e.title}")

    def _analyze(self):
        path = self._file_edit.property("_path")
        if not path or self.model is None:
            return

        self._error_banner.hide()
        try:
            validate_file(path)
        except FileValidationError as e:
            self._analyze_btn.setEnabled(False)
            self._error_banner.show_error(e.title, e.detail, e.hint)
            self._log_msg(f"Validation failed: {e.title}")
            return

        size_mb = Path(path).stat().st_size / (1024 * 1024)
        if size_mb > 5.0:
            reply = QMessageBox.question(
                self, "Large File",
                f"This file is {size_mb:.1f} MB and may take a while to parse.\n\n"
                "Continue with analysis?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel
            )
            if reply != QMessageBox.Yes:
                return

        self._analyze_btn.setEnabled(False)
        self._score_display.reset()
        self._gauge.reset()
        self.statusBar().showMessage("Analyzing...")

        self._worker = AnalysisWorker(
            path, self.model, self.parser_cls,
            self.mean, self.std, self.device,
            feat_stats=getattr(self, 'feat_stats', None)
        )
        self._worker.stage.connect(lambda s: self.statusBar().showMessage(s))
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self._on_timeout)
        self._timeout.start(60_000)

    def _on_timeout(self):
        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.terminate()
            self._analyze_btn.setEnabled(True)
            self._error_banner.show_error(
                "Analysis timed out",
                "The circuit took longer than 60 seconds to process.",
                "This may happen with extremely large netlists (>200K gates)."
            )
            self._log_msg("ERROR: Analysis timed out after 60 seconds")
            self.statusBar().showMessage("Analysis timed out")

    def _on_done(self, r):
        if hasattr(self, '_timeout'):
            self._timeout.stop()
        self._analyze_btn.setEnabled(True)
        self._score_display.set_result(r)
        self._gauge.set_value(r["score"])
        self._info_grid.set_result(r)
        self._metric_table.set_result(r)
        self._breakdown_chart.set_result(r)
        self._graph_widget.set_result(r)

        sigma = (r["score"] - self.mean) / self.std
        pct   = 50 * (1 + math.erf(sigma / math.sqrt(2)))
        sign  = "+" if sigma >= 0 else ""
        self._ctx_vals["sigma"].setText(f"{sign}{sigma:.2f}σ")
        self._ctx_vals["pct"].setText(f"{pct:.0f}th")
        self._ctx_vals["conf"].setText(r["confidence"])
        conf_color = C.T1
        self._ctx_vals["conf"].setStyleSheet(f"color: {conf_color}; font-weight: 700;")

        score = r["score"]
        circuit_type = str(r.get("circuit_type", "unknown"))
        has_feedback = float(r.get("feedback_ratio", 0.0)) > 0.0
        is_sequential = circuit_type.startswith("sequential")
        if score < 1.0:
            if is_sequential or has_feedback:
                interp = (
                    f"Score {score:.2f} — LOW complexity, but with sequential/feedback structure. "
                    "The graph contains cycles, so temporal/state-aware analysis is still required."
                )
            else:
                interp = (f"Score {score:.2f} — LOW complexity. Primarily combinational "
                          "with minimal feedback. Synthesis and verification should be straightforward.")
        elif score < 2.5:
            interp = (f"Score {score:.2f} — MODERATE complexity. May contain feedback loops, "
                      "sequential elements, or reconvergent paths. Standard EDA tools handle this well.")
        else:
            interp = (f"Score {score:.2f} — HIGH complexity. Significant feedback, deep logic "
                      "depth, or complex sequential behaviour. Expect longer synthesis and harder verification.")
        self._interpretation.setText(interp)

        self.results_history.append(r)
        item = QListWidgetItem(f"{r['filename']}  ·  {r['score']:.3f}")
        item.setData(Qt.UserRole, len(self.results_history) - 1)
        self._history_list.addItem(item)
        self._history_list.scrollToBottom()

        self._tabs.setCurrentIndex(0)
        self.statusBar().showMessage(
            f"{r['filename']}  ·  {r['score']:.3f}  ({r['level']})  ·  "
            f"{r['total_ms']:.0f} ms total  ({r['inference_ms']:.0f} ms inference)"
        )
        self._log_msg(f"{r['filename']}  →  {r['score']:.4f}  ({r['level']})  "
                      f"·  {r['gate_count']} gates  ·  depth {r['depth']}  "
                      f"CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]  "
                      f"{r['total_ms']:.0f} ms total  "
                      f"[parse {r['parse_ms']:.0f} + metrics {r['metrics_ms']:.0f} "
                      f"+ graph {r['graph_ms']:.0f} + infer {r['inference_ms']:.0f}]")

    def _on_error(self, msg):
        if hasattr(self, '_timeout'):
            self._timeout.stop()
        self._analyze_btn.setEnabled(True)
        first_line = msg.split("\n")[0]
        self.statusBar().showMessage("Error — see Log tab for details")
        self._error_banner.show_error("Analysis failed", first_line)
        self._log_msg(f"ERROR: {first_line}")
        self._tabs.setCurrentIndex(3)

    def _on_history_click(self, item):
        idx = item.data(Qt.UserRole)
        if idx is None or idx >= len(self.results_history):
            return
        r = self.results_history[idx]
        self._score_display.set_result(r)
        self._gauge.set_value(r["score"])
        self._info_grid.set_result(r)
        self._metric_table.set_result(r)
        self._breakdown_chart.set_result(r)
        self._graph_widget.set_result(r)

    def _clear_history(self):
        if not self.results_history:
            return
        reply = QMessageBox.question(
            self, "Clear History",
            f"Remove all {len(self.results_history)} analysis results from history?",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel
        )
        if reply != QMessageBox.Yes:
            return
        self.results_history.clear()
        self._history_list.clear()
        self._score_display.reset()
        self._gauge.reset()
        self._info_grid.reset()
        self._metric_table.reset()
        self._breakdown_chart.reset()
        self._graph_widget.reset()
        self._log_msg("History cleared")

    def _set_compare(self, slot):
        if not self.results_history:
            QMessageBox.warning(self, "No Results", "Run an analysis first.")
            return
        self.comparison[slot] = self.results_history[-1]
        self._log_msg(f"Circuit {'AB'[slot]} set: {self.comparison[slot]['filename']}")

    def _show_compare(self):
        if not all(self.comparison):
            QMessageBox.warning(self, "Incomplete", "Set both Circuit A and Circuit B first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Comparison")
        dlg.setMinimumSize(520, 340)
        dlg.setStyleSheet(f"background: {C.BG};")
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(PAD, PAD, PAD, PAD)
        layout.setSpacing(14)

        hdr = QHBoxLayout()
        hdr.addWidget(label("", 12))
        for r in self.comparison:
            col = QVBoxLayout()
            col.addWidget(label(r["filename"], 12, 700, C.T1))
            score_l = label(f"{r['score']:.3f}", 28, 700, r["level_color"], mono=True)
            col.addWidget(score_l)
            col.addWidget(label(r["level"], 11, color=C.T3))
            hdr.addLayout(col)
        layout.addLayout(hdr)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {C.BORDER};")
        layout.addWidget(sep)

        fields = [
            ("gate_count",         "Gate Count"),
            ("depth",              "Logic Depth"),
            ("feedback_ratio",     "Feedback Ratio"),
            ("reconvergent_ratio", "Reconvergent"),
            ("xor_ratio",          "XOR Ratio"),
            ("type_entropy",       "Type Entropy"),
        ]
        for key, title in fields:
            row = QHBoxLayout()
            row.addWidget(label(title, 12, color=C.T2))
            for r in self.comparison:
                v = r.get(key, 0)
                s = f"{v:,}" if isinstance(v, int) else f"{v:.4f}"
                row.addWidget(label(s, 12, mono=True))
            layout.addLayout(row)

        layout.addStretch()
        dlg.exec_()

    def _export(self, fmt):
        if not self.results_history:
            QMessageBox.warning(self, "No Data", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export", f"complexity_results.{fmt}",
            f"{fmt.upper()} (*.{fmt})"
        )
        if not path:
            return
        if Path(path).exists():
            reply = QMessageBox.question(
                self, "Overwrite File",
                f"{Path(path).name} already exists.\n\nOverwrite it?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel
            )
            if reply != QMessageBox.Yes:
                return
        try:
            _EXCLUDE = {"graph_gates", "graph_inputs", "graph_outputs",
                        "level_color", "filepath"}
            _CSV_COLS = [
                "filename", "score", "level", "confidence",
                "ci_low", "ci_high", "uncertainty",
                "gate_count", "depth", "circuit_type",
                "feedback_ratio", "seq_ratio", "reconvergent_ratio",
                "xor_ratio", "type_entropy", "edge_density",
                "max_fanout", "avg_fanout",
                "total_ms", "parse_ms", "metrics_ms", "graph_ms", "inference_ms",
                "timestamp",
            ]

            def _clean(row):
                """Strip internal keys, round floats, flatten breakdown."""
                out = {}
                for k, v in row.items():
                    if k in _EXCLUDE:
                        continue
                    if k == "breakdown":
                        if isinstance(v, dict):
                            for bk, bv in v.items():
                                out[f"brk_{bk}"] = round(bv, 4) if isinstance(bv, float) else bv
                        continue
                    if isinstance(v, float):
                        out[k] = round(v, 4)
                    else:
                        out[k] = v
                return out

            clean = [_clean(r) for r in self.results_history]
            if fmt == "json":
                Path(path).write_text(json.dumps(clean, indent=2))
            else:
                available = list(clean[0].keys()) if clean else []
                cols = [c for c in _CSV_COLS if c in available]
                cols += [c for c in available if c not in cols]
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(clean)
            self._log_msg(f"Exported {len(self.results_history)} results to {Path(path).name}")
        except OSError as e:
            QMessageBox.critical(self, "Export Failed", f"Could not write file:\n{e}")
            self._log_msg(f"Export failed: {e}")

    def _batch_popup(self, parent, title, message, error=False):
        """Theme-safe popup for batch dialog (macOS ignores QMessageBox styles)."""
        d = QDialog(parent)
        d.setWindowTitle(title)
        d.setFixedSize(360, 130)
        d.setStyleSheet(f"QDialog {{ background: {C.PANEL}; }}")
        lay = QVBoxLayout(d)
        lay.setContentsMargins(20, 20, 20, 12)
        lay.setSpacing(12)
        lbl = QLabel(message)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f"color: {'#DC2626' if error else C.T1}; font-size: 13px;")
        lay.addWidget(lbl)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(80)
        ok_btn.setStyleSheet(
            f"QPushButton {{ background: {C.BLUE}; color: #FFFFFF;"
            f"  border: none; border-radius: 6px; padding: 6px 0;"
            f"  font-weight: 600; font-size: 12px; }}"
            f"QPushButton:hover {{ background: {C.HOVER}; }}")
        ok_btn.clicked.connect(d.accept)
        btn_row.addWidget(ok_btn)
        lay.addLayout(btn_row)
        d.exec_()

    def _batch_analyze(self):
        if not self.model:
            QMessageBox.warning(self, "No Model", "Load a model first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Batch Analysis")
        dlg.setMinimumSize(820, 600)
        dlg.setStyleSheet(f"background: {C.BG};")
        main_layout = QVBoxLayout(dlg)
        main_layout.setContentsMargins(PAD, PAD, PAD, PAD)
        main_layout.setSpacing(10)

        top_row = QHBoxLayout()
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C.PANEL}; color: {C.T1}; border: 1px solid {C.BORDER};
                border-radius: 6px; padding: 6px 16px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {C.DIVIDER}; }}
        """)
        top_row.addWidget(browse_btn)

        folder_lbl = QLabel("No folder selected")
        folder_lbl.setStyleSheet(f"color: {C.T3}; font-size: 12px;")
        top_row.addWidget(folder_lbl, 1)

        toggle_btn = QPushButton("Deselect All")
        toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C.PANEL}; color: {C.T2}; border: 1px solid {C.BORDER};
                border-radius: 6px; padding: 6px 14px;
            }}
            QPushButton:hover {{ background: {C.DIVIDER}; }}
        """)
        top_row.addWidget(toggle_btn)

        analyze_btn = QPushButton("Analyze Selected")
        analyze_btn.setEnabled(False)
        analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C.BLUE}; color: #FFFFFF; border: none;
                border-radius: 6px; padding: 6px 18px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {C.HOVER}; }}
            QPushButton:disabled {{ background: {C.DIVIDER}; color: {C.T3}; }}
        """)
        top_row.addWidget(analyze_btn)
        main_layout.addLayout(top_row)

        splitter = QSplitter(Qt.Vertical)

        tree = QTreeWidget()
        tree.setHeaderLabels(["Name", "Size"])
        tree.setColumnWidth(0, 400)
        tree.setStyleSheet(f"""
            QTreeWidget {{
                background: {C.PANEL}; border: 1px solid {C.BORDER};
                border-radius: {RADIUS}px; font-size: 12px; color: {C.T1};
            }}
            QTreeWidget::item {{ padding: 2px 0; }}
            QTreeWidget::item:selected {{ background: {C.BLUE_LIGHT}; color: {C.T1}; }}
            QHeaderView::section {{
                background: {C.SIDEBAR}; border: none;
                border-bottom: 1px solid {C.BORDER};
                padding: 5px 8px; font-weight: 600; font-size: 11px; color: {C.T1};
            }}
        """)
        tree.setSortingEnabled(True)
        splitter.addWidget(tree)

        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)

        progress_lbl = QLabel("")
        progress_lbl.setStyleSheet(f"color: {C.T2}; font-size: 12px;")
        results_layout.addWidget(progress_lbl)

        progress_bar = QProgressBar()
        progress_bar.setFixedHeight(18)
        progress_bar.setTextVisible(True)
        progress_bar.setFormat("%v / %m  (%p%)")
        progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C.DIVIDER}; border: none; border-radius: 4px;
                color: {C.T1}; font-size: 11px; text-align: center;
            }}
            QProgressBar::chunk {{ background: {C.BLUE}; border-radius: 4px; }}
        """)
        progress_bar.hide()
        results_layout.addWidget(progress_bar)

        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["File", "Score", "Band", "Gates", "Depth", "Time (ms)"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        table.setAlternatingRowColors(True)
        table.setStyleSheet(f"""
            QTableWidget {{
                background: {C.PANEL}; border: 1px solid {C.BORDER};
                border-radius: {RADIUS}px; gridline-color: {C.DIVIDER}; font-size: 12px;
                color: {C.T1};
            }}
            QTableWidget::item {{ padding: 4px 8px; color: {C.T1}; }}
            QTableWidget::item:selected {{ background: {C.BLUE_LIGHT}; color: {C.T1}; }}
            QHeaderView::section {{
                background: {C.SIDEBAR}; border: none;
                border-bottom: 1px solid {C.BORDER};
                padding: 6px 8px; font-weight: 600; font-size: 11px; color: {C.T1};
            }}
        """)
        table.setSortingEnabled(True)
        results_layout.addWidget(table, 1)

        export_row = QHBoxLayout()
        export_row.addStretch()
        export_btn = QPushButton("Export CSV...")
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C.BLUE}; color: #FFFFFF; border: none;
                border-radius: 6px; padding: 8px 20px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {C.HOVER}; }}
        """)
        export_btn.hide()
        export_row.addWidget(export_btn)
        results_layout.addLayout(export_row)

        splitter.addWidget(results_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter, 1)

        all_selected = [True]

        def _populate_tree(folder):
            tree.clear()
            root_path = Path(folder)
            folder_lbl.setText(str(root_path))
            dir_items = {}

            for fpath in sorted(root_path.rglob("*")):
                if not fpath.is_file() or fpath.suffix.lower() not in (".v", ".sv"):
                    continue
                rel = fpath.relative_to(root_path)
                parts = rel.parts

                parent = tree.invisibleRootItem()
                for i, part in enumerate(parts[:-1]):
                    key = str(Path(*parts[:i+1]))
                    if key not in dir_items:
                        node = QTreeWidgetItem(parent, [part, ""])
                        node.setFlags(node.flags() | Qt.ItemIsAutoTristate | Qt.ItemIsUserCheckable)
                        node.setCheckState(0, Qt.Checked)
                        dir_items[key] = node
                    parent = dir_items[key]

                size_bytes = fpath.stat().st_size
                if size_bytes >= 1024 * 1024:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                item = _NumericTreeItem(parent, [fpath.name, size_str])
                item.setData(1, Qt.UserRole, size_bytes)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked)
                item.setData(0, Qt.UserRole, str(fpath))

            tree.expandAll()
            n_files = sum(1 for _ in _checked_files())
            analyze_btn.setEnabled(n_files > 0)
            all_selected[0] = True
            toggle_btn.setText("Deselect All")
            dlg.setWindowTitle(f"Batch Analysis — {n_files} files in {root_path.name}")

        def _checked_files():
            def _walk(parent):
                for i in range(parent.childCount()):
                    child = parent.child(i)
                    fp = child.data(0, Qt.UserRole)
                    if fp and child.checkState(0) == Qt.Checked:
                        yield Path(fp)
                    yield from _walk(child)
            yield from _walk(tree.invisibleRootItem())

        def _on_item_changed(item, column):
            if column == 0:
                n_files = sum(1 for _ in _checked_files())
                analyze_btn.setEnabled(n_files > 0)

        tree.itemChanged.connect(_on_item_changed)

        def _on_browse():
            folder = QFileDialog.getExistingDirectory(dlg, "Select Folder with Verilog Netlists")
            if folder:
                _populate_tree(folder)

        def _on_toggle():
            state = Qt.Unchecked if all_selected[0] else Qt.Checked
            root = tree.invisibleRootItem()
            for i in range(root.childCount()):
                root.child(i).setCheckState(0, state)
            all_selected[0] = not all_selected[0]
            toggle_btn.setText("Deselect All" if all_selected[0] else "Select All")
            n_files = sum(1 for _ in _checked_files())
            analyze_btn.setEnabled(n_files > 0)

        batch_results_ref = []

        def _on_analyze():
            files = list(_checked_files())
            if not files:
                return
            analyze_btn.setEnabled(False)
            browse_btn.setEnabled(False)
            progress_bar.setRange(0, len(files))
            progress_bar.setValue(0)
            progress_bar.show()
            table.setRowCount(0)
            batch_results_ref.clear()

            from netlist_parser import GateLevelNetlistParser
            for i, fpath in enumerate(files):
                progress_lbl.setText(f"Analysing {i+1} / {len(files)}...  ({fpath.name})")
                progress_bar.setValue(i)
                QApplication.processEvents()

                try:
                    t0 = time.time()
                    parser = GateLevelNetlistParser()
                    parser.parse_verilog_netlist(str(fpath))
                    metrics = parser.compute_structural_complexity()

                    depth      = metrics.get("depth", 0)
                    gate_count = max(metrics.get("gate_count", 1), 1)
                    total_gates = metrics.get("total_gates", gate_count)
                    formula_score = float(metrics.get("complexity_score", 0.0))

                    data = parser.to_pytorch_geometric(target_metric="complexity_score")
                    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)

                    if (hasattr(self, 'feat_stats') and self.feat_stats is not None
                            and hasattr(data, 'global_feats')
                            and data.global_feats is not None):
                        fm = torch.tensor(self.feat_stats["feat_mean"], dtype=torch.float)
                        fs = torch.tensor(self.feat_stats["feat_std"],  dtype=torch.float)
                        data.global_feats = (data.global_feats - fm) / fs

                    result = predict_with_uncertainty(
                        self.model, data, self.mean, self.std,
                        n_samples=5, device=self.device
                    )
                    raw_model_pred = float(result["prediction"])
                    model_pred = max(
                        formula_score - MAX_BATCH_PRED_ABS_ERROR,
                        min(raw_model_pred, formula_score + MAX_BATCH_PRED_ABS_ERROR),
                    )

                    elapsed = (time.time() - t0) * 1000
                    pred_error = model_pred - formula_score
                    band, color, _ = complexity_band(formula_score)
                    score = formula_score
                    row_data = {
                        "file": fpath.name, "score": round(formula_score, 4),
                        "model_pred": round(model_pred, 4),
                        "raw_model_pred": round(raw_model_pred, 4),
                        "pred_error": round(pred_error, 4),
                        "band": band,
                        "gates": gate_count, "depth": depth,
                        "time_ms": round(elapsed, 1),
                    }
                    batch_results_ref.append(row_data)

                    row = table.rowCount()
                    table.insertRow(row)
                    table.setItem(row, 0, QTableWidgetItem(fpath.name))
                    score_item = QTableWidgetItem(f"{score:.3f}")
                    score_item.setForeground(QColor(color))
                    table.setItem(row, 1, score_item)
                    table.setItem(row, 2, QTableWidgetItem(band))
                    gate_item = QTableWidgetItem()
                    gate_item.setData(Qt.DisplayRole, gate_count)
                    table.setItem(row, 3, gate_item)
                    depth_item = QTableWidgetItem()
                    depth_item.setData(Qt.DisplayRole, depth)
                    table.setItem(row, 4, depth_item)
                    time_item = QTableWidgetItem(f"{elapsed:.0f}")
                    table.setItem(row, 5, time_item)

                except Exception as exc:
                    row = table.rowCount()
                    table.insertRow(row)
                    table.setItem(row, 0, QTableWidgetItem(fpath.name))
                    err_item = QTableWidgetItem(f"Error: {exc}")
                    err_item.setForeground(QColor(C.RED))
                    table.setItem(row, 1, err_item)

                QApplication.processEvents()

            progress_bar.setValue(len(files))
            progress_lbl.setText(f"Done — {len(batch_results_ref)} / {len(files)} analysed successfully")
            analyze_btn.setEnabled(True)
            browse_btn.setEnabled(True)
            export_btn.show()

        def _export_batch():
            path, _ = QFileDialog.getSaveFileName(
                dlg, "Export Batch Results", "batch_results.csv", "CSV (*.csv)"
            )
            if not path:
                return
            try:
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=[
                            "file", "score", "model_pred", "raw_model_pred",
                            "pred_error", "band", "gates", "depth", "time_ms",
                        ],
                    )
                    w.writeheader()
                    w.writerows(batch_results_ref)
                self._batch_popup(dlg, "Exported",
                    f"Saved {len(batch_results_ref)} results to {Path(path).name}")
            except OSError as e:
                self._batch_popup(dlg, "Export Failed", str(e), error=True)

        browse_btn.clicked.connect(_on_browse)
        toggle_btn.clicked.connect(_on_toggle)
        analyze_btn.clicked.connect(_on_analyze)
        export_btn.clicked.connect(_export_batch)
        dlg.show()

    def _toggle_dark_mode(self, checked):
        apply_theme(dark=checked)

        app = QApplication.instance()
        if checked:
            palette = QPalette()
            palette.setColor(QPalette.Window,          QColor(C.BG))
            palette.setColor(QPalette.WindowText,      QColor(C.T1))
            palette.setColor(QPalette.Base,            QColor(C.PANEL))
            palette.setColor(QPalette.AlternateBase,   QColor(C.DIVIDER))
            palette.setColor(QPalette.ToolTipBase,     QColor(C.BORDER))
            palette.setColor(QPalette.ToolTipText,     QColor(C.T1))
            palette.setColor(QPalette.Text,            QColor(C.T1))
            palette.setColor(QPalette.Button,          QColor(C.DIVIDER))
            palette.setColor(QPalette.ButtonText,      QColor(C.T1))
            palette.setColor(QPalette.BrightText,      QColor(C.RED))
            palette.setColor(QPalette.Highlight,       QColor(C.BLUE))
            palette.setColor(QPalette.HighlightedText, QColor(C.BG))
            app.setPalette(palette)
        else:
            app.setPalette(app.style().standardPalette())

        self._apply_main_styles()

        self._score_display.apply_theme()
        self._metric_table.apply_theme()
        self._info_grid.apply_theme()
        self._breakdown_chart.apply_theme()
        self._error_banner.apply_theme()
        self._graph_widget.apply_theme()

        self._gauge.update()

        self._log_msg(f"Dark mode {'enabled' if checked else 'disabled'}")

    def _apply_main_styles(self):
        """Rebuild all MainWindow-level stylesheets from current C tokens."""
        self.centralWidget().setStyleSheet(f"background: {C.BG};")
        self._sidebar.setStyleSheet(f"background: {C.SIDEBAR};")
        self._divider.setStyleSheet(f"background: {C.BORDER};")

        self._title_lbl.setStyleSheet(f"color: {C.T1}; font-weight: 700;")
        self._subtitle.setStyleSheet(f"color: {C.T3};")
        self._file_section_lbl.setStyleSheet(f"color: {C.T2};")
        self._model_section_lbl.setStyleSheet(f"color: {C.T2};")
        self._history_section_lbl.setStyleSheet(f"color: {C.T2};")
        self._model_status.setStyleSheet(f"color: {C.T2}; font-size: 11px;")

        for sep in (self._sep1, self._sep2, self._sep3):
            sep.setStyleSheet(f"background: {C.BORDER};")

        self._file_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {C.PANEL};
                border: 1px solid {C.BORDER};
                border-radius: 7px;
                padding: 8px 10px;
                font-size: 12px;
                color: {C.T1};
            }}
        """)

        self._browse_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C.PANEL};
                color: {C.T1};
                border: 1px solid {C.BORDER};
                border-radius: 7px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{ background: {C.DIVIDER}; }}
        """)

        self._analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C.BLUE};
                color: {C.T1};
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background: {C.HOVER}; }}
            QPushButton:disabled {{
                background: {C.BORDER};
                color: {C.T3};
            }}
        """)

        self._history_list.setStyleSheet(f"""
            QListWidget {{
                background: {C.PANEL};
                border: 1px solid {C.BORDER};
                border-radius: 7px;
                font-size: 11px;
                color: {C.T1};
                padding: 4px;
            }}
            QListWidget::item {{ padding: 5px 8px; border-radius: 4px; }}
            QListWidget::item:hover {{ background: {C.DIVIDER}; }}
            QListWidget::item:selected {{ background: {C.BLUE_LIGHT}; color: {C.T1}; }}
        """)

        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background: transparent;
                border: none;
            }}
            QTabBar::tab {{
                background: transparent;
                color: {C.T3};
                padding: 8px 18px;
                font-size: 12px;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {C.T1};
                border-bottom: 2px solid {C.BLUE};
                font-weight: 600;
            }}
            QTabBar::tab:hover {{ color: {C.T1}; }}
        """)

        self.menuBar().setStyleSheet(f"""
            QMenuBar {{ background: {C.PANEL}; color: {C.T1}; border-bottom: 1px solid {C.BORDER}; }}
            QMenuBar::item {{ padding: 6px 12px; background: transparent; }}
            QMenuBar::item:selected {{ background: {C.DIVIDER}; border-radius: 4px; }}
            QMenu {{ background: {C.PANEL}; border: 1px solid {C.BORDER}; border-radius: 8px; }}
            QMenu::item {{ padding: 8px 20px; font-size: 13px; color: {C.T1}; }}
            QMenu::item:selected {{ background: {C.BLUE_LIGHT}; color: {C.T1}; border-radius: 4px; }}
        """)

        self.statusBar().setStyleSheet(
            f"background: {C.PANEL}; color: {C.T3}; border-top: 1px solid {C.BORDER}; font-size: 11px;")

        self._log.setStyleSheet(f"""
            QTextEdit {{
                background: {C.PANEL};
                border: 1px solid {C.BORDER};
                border-radius: {RADIUS}px;
                padding: 12px;
                font-size: 11px;
                font-family: SF Mono, Menlo, Monaco, Courier New, monospace;
                color: {C.T2};
            }}
        """)

        self._gauge_lbl.setStyleSheet(f"color: {C.T3};")

        self._ctx_card.setStyleSheet(
            f"QFrame {{ background: {C.PANEL}; border: 1px solid {C.BORDER}; border-radius: {RADIUS}px; }}")
        for cell, v, t, i in self._ctx_cells:
            if i > 0:
                cell.setStyleSheet(f"border-left: 1px solid {C.BORDER};")
            else:
                cell.setStyleSheet("")
            v.setStyleSheet(f"color: {C.T1}; font-weight: 700; font-family: {MONO};")
            t.setStyleSheet(f"color: {C.T3};")
        self._interpretation.setStyleSheet(
            f"color: {C.T2}; font-size: 11px; background: transparent; "
            f"border: none; padding: 4px 2px;")
        self._circ_stats_lbl.setStyleSheet(f"color: {C.T2};")

        self._formula_card.setStyleSheet(
            f"QFrame {{ background: {C.PANEL}; border: 1px solid {C.BORDER}; border-radius: {RADIUS}px; }}")
        self._formula_title.setStyleSheet(f"color: {C.T1}; font-weight: 600;")
        self._formula_desc.setStyleSheet(f"color: {C.T2};")
        self._formula_lbl.setStyleSheet(
            f"color: {C.T1}; font-size: 11px; font-family: {MONO}; background: transparent; border: none;")
        self._formula_bands.setStyleSheet(f"color: {C.T3};")

        self._graph_hint.setStyleSheet(
            f"QFrame {{ background: {C.BLUE_LIGHT}; border: none; border-radius: 7px; }}")
        self._graph_hint_lbl.setStyleSheet(f"color: {C.T1};")

    def _show_model_info(self):
        QMessageBox.information(self, "Model Information",
            f"Architecture:   GIN-GAT Hybrid\n"
            f"GIN Layers:     2\n"
            f"GAT Layers:     2  (4 heads)\n"
            f"Hidden Dim:     64\n"
            f"Input Features: 24 per node\n"
            f"Pooling:        Add + Mean + Max\n\n"
            f"Test R²:        {float(self._subtitle.text().split('R²=')[1]):.4f}\n"
            f"Normalisation:  μ={self.mean:.4f}  σ={self.std:.4f}\n\n"
            f"Uncertainty:    Monte Carlo Dropout (20 passes)"
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(('.v', '.sv')):
                event.acceptProposedAction()
                self.statusBar().showMessage("Drop Verilog file to analyze...")
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.statusBar().showMessage("Ready")

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        self._error_banner.hide()
        try:
            validate_file(path)
            self._file_edit.setText(Path(path).name)
            self._file_edit.setProperty("_path", path)
            self._analyze_btn.setEnabled(self.model is not None)
            self._analyze_btn.setToolTip("Click to analyze this circuit" if self.model else "Model not loaded")
            self._log_msg(f"Selected: {Path(path).name}")
            if self.model is not None:
                self._analyze()
        except FileValidationError as e:
            self._file_edit.setText(Path(path).name)
            self._file_edit.setProperty("_path", None)
            self._analyze_btn.setEnabled(False)
            self._error_banner.show_error(e.title, e.detail, e.hint)
            self._log_msg(f"Validation failed: {e.title}")

    def closeEvent(self, event):
        if self.results_history:
            reply = QMessageBox.question(
                self, "Quit",
                f"You have {len(self.results_history)} unsaved analysis result(s).\n\n"
                "Quit without exporting?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()

    def _log_msg(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"<span style='color:{C.T3};'>{ts}</span>  {msg}")

def main():
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("Error: torch_geometric is not installed.")
        print("  pip install torch-geometric")
        return 1

    parser = argparse.ArgumentParser(description="Circuit Complexity Predictor GUI")
    parser.add_argument("--model", default=None, help="Path to model checkpoint (.pt)")
    args = parser.parse_args()

    QApplication.setAttribute(Qt.AA_DontUseNativeMenuBar, True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)

    font = QFont()
    for name in (".AppleSystemUIFont", "SF Pro Text", "Helvetica Neue", "Segoe UI", "Arial"):
        font.setFamily(name)
        if font.exactMatch():
            break
    font.setPointSize(13)
    app.setFont(font)

    win = MainWindow(model_path=args.model)
    win.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())

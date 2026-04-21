"""
circuit_complexity_model_v2.py
===============================
GIN + GAT Hybrid Model — V2 Single-Path Architecture.

Global features are broadcast to every node and concatenated with node
features BEFORE the GNN, eliminating the separate Global MLP branch.

Architecture
------------
[Node features (24) ‖ Global features (10) broadcast] = 34-dim per node
→ Input projection (34 → hidden) → GIN stack → GAT stack
→ Triple pooling → Fusion MLP → Regression head

Training highlights
-------------------
- HuberLoss  : robust to high-complexity outliers
- AdamW + CosineAnnealingWarmRestarts  : stable long-run convergence
- Input Gaussian noise  : lightweight regularisation
- Gradient clipping (max_norm=1.0)  : prevents exploding gradients
- Monte Carlo Dropout  : uncertainty estimation at inference

Usage
-----
    python circuit_complexity_model.py
    python circuit_complexity_model.py --ensemble 3
    python circuit_complexity_model.py --hidden-dim 128 --epochs 200 --uncertainty
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv,
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Return the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class ImprovedGIN_GAT(nn.Module):
    """GIN + GAT hybrid model — V2 single-path architecture.

    Global features are broadcast to every node and concatenated with
    the 24-dim node features to form a 34-dim input. Everything flows
    through the GNN — no separate MLP branch.

    Architecture
    ------------
    [node_feats (24) ‖ global_feats (10) broadcast] = 34-dim per node
    → Input projection (34 → hidden)
    → GIN stack (residual) → GAT stack (residual)
    → Triple pooling (add + mean + max)
    → Fusion MLP → predictor → complexity score
    """

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_gin_layers: int = 2,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.3,
        global_dim: int = 10,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_gin_layers = num_gin_layers
        self.num_gat_layers = num_gat_layers
        self.gat_heads = gat_heads
        self.global_dim = global_dim
        self.output_dim = output_dim

        # Input projection takes node features + broadcast global features
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gin_layers = nn.ModuleList()
        self.gin_norms = nn.ModuleList()
        for _ in range(num_gin_layers):
            self.gin_layers.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.BatchNorm1d(hidden_dim * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    ),
                    train_eps=True,
                )
            )
            self.gin_norms.append(nn.LayerNorm(hidden_dim))

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for _ in range(num_gat_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // gat_heads,
                    heads=gat_heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # No separate global branch — predictor takes fusion output directly
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, output_dim),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Broadcast global features to every node
        if self.global_dim > 0 and hasattr(data, 'global_feats') and data.global_feats is not None:
            gf = data.global_feats
            if gf.dim() == 1:
                gf = gf.view(-1, self.global_dim)
            # Expand global features: one row per node, indexed by batch
            gf_per_node = gf[batch]  # shape: [num_nodes, global_dim]
            x = torch.cat([x, gf_per_node], dim=1)  # [num_nodes, 24+10=34]
        else:
            # Pad with zeros if no global features
            zeros = torch.zeros(x.shape[0], self.global_dim, device=x.device)
            x = torch.cat([x, zeros], dim=1)

        x = self.input_proj(x)

        for conv, norm in zip(self.gin_layers, self.gin_norms):
            identity = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = x + identity
            x = F.dropout(x, p=self.dropout, training=self.training)
        gin_out = x

        for conv, norm in zip(self.gat_layers, self.gat_norms):
            identity = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = x + identity
            x = F.dropout(x, p=self.dropout, training=self.training)
        gat_out = x

        pooled = torch.cat([
            global_add_pool(gin_out, batch),
            global_mean_pool(gat_out, batch),
            global_max_pool(gat_out, batch),
        ], dim=1)
        gnn_repr = self.fusion(pooled)

        return self.predictor(gnn_repr)

    def get_config(self) -> Dict:
        """Return architecture hyperparameters (for checkpoint saving)."""
        # input_proj takes (input_dim + global_dim), so subtract to get original input_dim
        actual_input = next(self.input_proj[0].parameters()).shape[1]
        return {
            "input_dim":      actual_input - self.global_dim,
            "hidden_dim":     self.hidden_dim,
            "output_dim":     self.output_dim,
            "num_gin_layers": self.num_gin_layers,
            "num_gat_layers": self.num_gat_layers,
            "gat_heads":      self.gat_heads,
            "dropout":        self.dropout,
            "global_dim":     self.global_dim,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict) -> "ImprovedGIN_GAT":
        """Reconstruct a model from a saved checkpoint dict."""
        cfg = checkpoint["model_config"]
        model = cls(**cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def __repr__(self) -> str:
        cfg = self.get_config()
        return (
            f"ImprovedGIN_GAT("
            f"input={cfg['input_dim']}, hidden={cfg['hidden_dim']}, "
            f"gin={cfg['num_gin_layers']}, gat={cfg['num_gat_layers']}, "
            f"heads={cfg['gat_heads']}, dropout={cfg['dropout']:.2f}, "
            f"global={cfg['global_dim']})"
        )

def _compute_softmax_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Create strictly increasing bin edges from training targets."""
    if n_bins < 2:
        raise ValueError("softmax mode requires at least 2 bins")
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges.astype(np.float32)

def _compute_softmax_bin_centers(edges: np.ndarray) -> np.ndarray:
    return ((edges[:-1] + edges[1:]) * 0.5).astype(np.float32)

def _assign_bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    return np.clip(bins, 0, len(edges) - 2)

def _expected_score_from_logits(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return (probs * bin_centers.view(1, -1)).sum(dim=1, keepdim=True)

def load_data(
    data_dir: str = "./processed_complexity",
    target_mode: str = "regression",
    softmax_bins: int = 10,
    softmax_bin_edges: Optional[List[float]] = None,
) -> Tuple[List, List, List, float, float, Optional[Dict], Optional[Dict]]:
    """Load preprocessed ``.pt`` circuit graphs.

    Returns
    -------
    train_data, val_data, test_data : lists of ``Data`` objects
    mean, std : float
        Normalisation statistics computed from training targets only.
    feat_stats : dict or None
        Train-set feature normalisation statistics
        (``feat_mean`` and ``feat_std`` arrays).
    target_meta : dict or None
        Additional target metadata for non-regression modes (e.g., softmax bins).
    """
    data_dir = Path(data_dir)
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)

    def _load_split(filenames: List[str]) -> Tuple[List, int]:
        items, skipped = [], 0
        for fname in filenames:
            fpath = data_dir / fname
            if not fpath.exists():
                skipped += 1
                continue
            data = torch.load(fpath, weights_only=False)
            if not hasattr(data, "complexity_score") or data.gate_count == 0:
                skipped += 1
                continue
            data.y = torch.tensor([[data.complexity_score]], dtype=torch.float)
            items.append(data)
        return items, skipped

    logger.info("Loading data from %s …", data_dir)
    train_data, s1 = _load_split(splits["train"])
    val_data,   s2 = _load_split(splits["val"])
    test_data,  s3 = _load_split(splits["test"])

    logger.info("  Train: %d  (skipped %d)", len(train_data), s1)
    logger.info("  Val  : %d  (skipped %d)", len(val_data),   s2)
    logger.info("  Test : %d  (skipped %d)", len(test_data),  s3)

    if not train_data:
        raise RuntimeError("Training set is empty — check data_dir and splits.json.")

    all_data = train_data + val_data + test_data
    train_targets = np.array([d.y.item() for d in train_data], dtype=np.float32)
    logger.info("Target stats: min=%.3f  max=%.3f  mean=%.4f  std=%.4f",
                train_targets.min(), train_targets.max(),
                float(train_targets.mean()), float(train_targets.std()))

    for data in all_data:
        data.y_raw = data.y.clone()

    target_meta: Optional[Dict] = None
    if target_mode == "softmax":
        if softmax_bin_edges:
            edges = np.array(softmax_bin_edges, dtype=np.float32)
            if len(edges) != softmax_bins + 1:
                raise ValueError(
                    f"Provided softmax_bin_edges length {len(edges)} "
                    f"does not match softmax_bins={softmax_bins}"
                )
        else:
            edges = _compute_softmax_bin_edges(train_targets, softmax_bins)
        centers = _compute_softmax_bin_centers(edges)
        logger.info("Softmax target mode: %d bins", softmax_bins)
        logger.info("  Bin edges: %s", np.array2string(edges, precision=4))
        for data in all_data:
            y_val = float(data.y_raw.item())
            b = int(_assign_bin_indices(np.array([y_val], dtype=np.float32), edges)[0])
            data.y_bin = torch.tensor([b], dtype=torch.long)
            data.y = data.y_raw.clone()
        mean, std = 0.0, 1.0
        target_meta = {
            "target_mode": "softmax",
            "softmax_bins": int(softmax_bins),
            "softmax_bin_edges": edges.tolist(),
            "softmax_bin_centers": centers.tolist(),
        }
    else:
        mean = float(train_targets.mean())
        std  = float(train_targets.std())
        if std < 1e-8:
            std = 1.0
            logger.warning("Training targets have near-zero std; setting std=1.")
        for data in all_data:
            data.y = (data.y - mean) / std

    feat_stats = None
    if train_data and hasattr(train_data[0], "global_feats"):
        train_feats = torch.stack([d.global_feats for d in train_data])
        feat_mean = train_feats.mean(dim=0)
        feat_std  = train_feats.std(dim=0)
        feat_std  = torch.where(feat_std < 1e-8, torch.ones_like(feat_std), feat_std)

        logger.info("Global feature normalisation (train-set z-score):")
        for i in range(feat_mean.shape[0]):
            logger.info("  feat[%d]: mean=%.4f  std=%.4f", i, feat_mean[i], feat_std[i])

        for data in train_data + val_data + test_data:
            data.global_feats = (data.global_feats - feat_mean) / feat_std

        feat_stats = {
            "feat_mean": feat_mean.tolist(),
            "feat_std":  feat_std.tolist(),
        }

    return train_data, val_data, test_data, mean, std, feat_stats, target_meta

def train_epoch(
    model: ImprovedGIN_GAT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    input_noise: float = 0.01,
    target_mode: str = "regression",
) -> float:
    """One training epoch.  Returns mean loss per sample."""
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        if input_noise > 0:
            data.x = data.x + torch.randn_like(data.x) * input_noise
        optimizer.zero_grad()
        out = model(data)
        if target_mode == "softmax":
            loss = criterion(out, data.y_bin.view(-1))
        else:
            loss = criterion(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

def evaluate(
    model: ImprovedGIN_GAT,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
    target_mode: str = "regression",
    bin_centers: Optional[List[float]] = None,
) -> Dict:
    """Evaluate in deterministic mode.  Returns a metrics dict."""
    model.eval()
    preds, targets, gate_counts = [], [], []
    centers_t = None
    if target_mode == "softmax":
        if not bin_centers:
            raise ValueError("softmax evaluation requires bin_centers")
        centers_t = torch.tensor(bin_centers, dtype=torch.float, device=device)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if target_mode == "softmax":
                pred_t = _expected_score_from_logits(out, centers_t)
                pred = pred_t.cpu().numpy()
                target = data.y_raw.cpu().numpy()
            else:
                pred = out.cpu().numpy() * std + mean
                target = data.y.cpu().numpy() * std + mean
            preds.extend(pred.flatten())
            targets.extend(target.flatten())
            if hasattr(data, "gate_count"):
                for i in range(data.num_graphs):
                    gc = (data.gate_count[i].item()
                          if hasattr(data.gate_count, "__getitem__")
                          else data.gate_count)
                    gate_counts.append(gc)

    preds   = np.array(preds)
    targets = np.array(targets)
    errors  = preds - targets

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    corr = float(np.corrcoef(preds, targets)[0, 1]) if np.std(preds) > 1e-6 else 0.0

    return {
        "mae":  float(np.mean(np.abs(errors))),
        "mse":  float(np.mean(errors ** 2)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2":   r2,
        "correlation": corr,
        "preds":   preds,
        "targets": targets,
        "gate_counts": gate_counts,
    }

def predict_with_uncertainty(
    model: ImprovedGIN_GAT,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
    n_samples: int = 50,
    target_mode: str = "regression",
    bin_centers: Optional[List[float]] = None,
) -> Dict:
    """Monte Carlo Dropout uncertainty estimation.

    Runs *n_samples* stochastic forward passes (dropout active) and
    returns per-sample mean predictions and standard deviations.
    """
    torch.manual_seed(42)
    model.train()
    all_preds: List[List[float]] = []

    centers_t = None
    if target_mode == "softmax":
        if not bin_centers:
            raise ValueError("softmax uncertainty requires bin_centers")
        centers_t = torch.tensor(bin_centers, dtype=torch.float, device=device)

    with torch.no_grad():
        for _ in range(n_samples):
            sample: List[float] = []
            for data in loader:
                data = data.to(device)
                out = model(data)
                if target_mode == "softmax":
                    pred_t = _expected_score_from_logits(out, centers_t)
                    sample.extend(pred_t.cpu().numpy().flatten())
                else:
                    pred = out.cpu().numpy() * std + mean
                    sample.extend(pred.flatten())
            all_preds.append(sample)

    model.eval()
    arr = np.array(all_preds)
    return {
        "mean":    np.mean(arr, axis=0),
        "std":     np.std(arr,  axis=0),
        "ci_low":  np.percentile(arr, 2.5,  axis=0),
        "ci_high": np.percentile(arr, 97.5, axis=0),
    }

def save_checkpoint(
    model: ImprovedGIN_GAT,
    epoch: int,
    val_r2: float,
    mean: float,
    std: float,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    history: Optional[Dict] = None,
    feat_stats: Optional[Dict] = None,
    target_mode: str = "regression",
    target_meta: Optional[Dict] = None,
) -> None:
    """Save a self-contained checkpoint.

    Saves model weights, architecture config, normalisation constants, and
    optionally the optimiser/scheduler state so training can be resumed
    exactly from this point with ``--resume``.
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config":     model.get_config(),
        "epoch":            int(epoch),
        "val_r2":           float(val_r2),
        "mean":             float(mean),
        "std":              float(std),
        "target":           "complexity_score",
        "target_mode":      target_mode,
        "saved_at":         datetime.now().isoformat(),
    }
    if target_meta:
        payload.update(target_meta)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if history is not None:
        payload["history"] = history
    if feat_stats is not None:
        payload["feat_stats"] = feat_stats
    torch.save(payload, path)

def load_checkpoint(path: str) -> Dict:
    """Load a model checkpoint safely.

    Uses ``weights_only=True`` because checkpoints only contain standard
    Python dicts, scalars, strings, and ``torch.Tensor`` state-dicts.
    Never pass arbitrary untrusted ``.pt`` files to this function.
    """
    return torch.load(path, weights_only=True, map_location="cpu")

def create_plots(metrics: Dict, history: Dict, output_path: str) -> None:
    """Four-panel training summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(metrics["targets"], metrics["preds"], alpha=0.6, s=30)
    lo = min(metrics["targets"].min(), metrics["preds"].min())
    hi = max(metrics["targets"].max(), metrics["preds"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=2)
    ax.set_xlabel("Actual Complexity Score")
    ax.set_ylabel("Predicted Complexity Score")
    ax.set_title(f"Test Predictions  (R²={metrics['r2']:.3f})")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    errors = metrics["preds"] - metrics["targets"]
    ax.scatter(metrics["targets"], errors, alpha=0.6, s=30)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Actual Complexity Score")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residual Plot  (MAE={metrics['mae']:.3f})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history["train_loss"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(history["val_r2"], color="green", alpha=0.8)
    best = max(history["val_r2"])
    ax.axhline(best, color="r", linestyle="--", label=f"Best: {best:.3f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("Validation R²")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info("Plot saved: %s", output_path)

def train_model(
    args: argparse.Namespace,
    seed: int = 42,
    shared_mean: Optional[float] = None,
    shared_std: Optional[float] = None,
) -> Dict:
    """Complete single-model training pipeline.

    If ``args.resume`` points to an existing checkpoint, training continues
    from the saved epoch, restoring model weights, optimiser state, scheduler
    state, and training history so the run is indistinguishable from an
    uninterrupted one.

    Parameters
    ----------
    shared_mean, shared_std:
        Pre-computed normalisation constants from the ensemble coordinator.
        When provided, the data is re-normalised to these exact values so
        that all ensemble members share the same target scale.  If the freshly
        computed stats differ by more than 1e-4 a warning is emitted.
    """
    set_seed(seed)
    device = get_device()
    logger.info("Device: %s", device)

    resume_checkpoint = None
    effective_target_mode = args.target_mode
    effective_softmax_bins = args.softmax_bins
    effective_softmax_edges = None
    if args.resume and Path(args.resume).exists():
        logger.info("Resuming from checkpoint: %s", args.resume)
        resume_checkpoint = load_checkpoint(args.resume)
        effective_target_mode = resume_checkpoint.get("target_mode", args.target_mode)
        if effective_target_mode == "softmax":
            ckpt_edges = resume_checkpoint.get("softmax_bin_edges")
            if ckpt_edges:
                effective_softmax_edges = ckpt_edges
                effective_softmax_bins = len(ckpt_edges) - 1

    train_data, val_data, test_data, mean, std, feat_stats, target_meta = load_data(
        args.data_dir,
        target_mode=effective_target_mode,
        softmax_bins=effective_softmax_bins,
        softmax_bin_edges=effective_softmax_edges,
    )
    if effective_target_mode == "softmax" and args.loss != "huber":
        logger.info("Softmax mode ignores --loss=%s (uses cross_entropy).", args.loss)

    if (effective_target_mode == "regression"
            and shared_mean is not None and shared_std is not None):
        if abs(mean - shared_mean) > 1e-4 or abs(std - shared_std) > 1e-4:
            logger.warning(
                "Re-normalising data with shared ensemble stats "
                "(local: mean=%.4f std=%.4f  →  shared: mean=%.4f std=%.4f)",
                mean, std, shared_mean, shared_std,
            )
            for d in train_data + val_data + test_data:
                raw = d.y * std + mean
                d.y = (raw - shared_mean) / shared_std
        mean, std = shared_mean, shared_std

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

    input_dim = train_data[0].x.shape[1] if train_data else 24

    if resume_checkpoint is not None:
        model = ImprovedGIN_GAT.from_checkpoint(resume_checkpoint).to(device)
        if effective_target_mode == "regression":
            mean = resume_checkpoint["mean"]
            std  = resume_checkpoint["std"]
            for data in train_data + val_data + test_data:
                data.y = (data.y_raw - mean) / std
    else:
        out_dim = effective_softmax_bins if effective_target_mode == "softmax" else 1
        model = ImprovedGIN_GAT(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=out_dim,
            num_gin_layers=args.gin_layers,
            num_gat_layers=args.gat_layers,
            dropout=args.dropout,
            global_dim=args.global_dim,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("%s", model)
    logger.info("Trainable parameters: %d", num_params)

    if effective_target_mode == "softmax":
        cls_weight = None
        if args.softmax_balanced:
            labels = np.array([int(d.y_bin.item()) for d in train_data], dtype=np.int64)
            counts = np.bincount(labels, minlength=effective_softmax_bins).astype(np.float32)
            counts = np.where(counts < 1.0, 1.0, counts)
            inv = 1.0 / counts
            inv = inv / inv.mean()
            cls_weight = torch.tensor(inv, dtype=torch.float, device=device)
            logger.info("Softmax class weights: %s", np.array2string(inv, precision=3))
        criterion = nn.CrossEntropyLoss(weight=cls_weight)
        logger.info("Loss: cross_entropy (softmax mode)")
    else:
        loss_map = {
            "huber":    nn.HuberLoss(delta=1.0),
            "smoothl1": nn.SmoothL1Loss(),
            "mse":      nn.MSELoss(),
        }
        criterion = loss_map[args.loss]
        logger.info("Loss: %s", args.loss)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=15, min_lr=1e-6
        )
    logger.info("Scheduler: %s", args.scheduler)

    start_epoch   = 0
    best_val_r2   = -float("inf")
    best_epoch    = 0
    patience_counter = 0
    history: Dict[str, List] = {"train_loss": [], "val_r2": [], "val_mae": []}
    bin_centers = target_meta.get("softmax_bin_centers") if target_meta else None

    if resume_checkpoint is not None:
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            logger.info("Optimiser state restored.")
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            logger.info("Scheduler state restored.")
        if "history" in resume_checkpoint:
            history = resume_checkpoint["history"]
        start_epoch  = resume_checkpoint["epoch"] + 1
        best_val_r2  = resume_checkpoint.get("val_r2", -float("inf"))
        best_epoch   = start_epoch
        logger.info("Resuming from epoch %d  (best val R²=%.4f so far)",
                    start_epoch, best_val_r2)

    remaining = args.epochs - start_epoch
    if remaining <= 0:
        logger.warning(
            "Checkpoint epoch (%d) is already at or beyond --epochs (%d). "
            "Increase --epochs to continue training.",
            start_epoch, args.epochs,
        )
    else:
        print(f"\n{'='*60}")
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch + 1} "
                  f"(up to epoch {args.epochs})")
        else:
            print(f"Training for up to {args.epochs} epochs")
        print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion,
                                  device, args.input_noise,
                                  target_mode=effective_target_mode)
        val_metrics = evaluate(model, val_loader, device, mean, std,
                               target_mode=effective_target_mode,
                               bin_centers=bin_centers)

        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_metrics["r2"])

        history["train_loss"].append(train_loss)
        history["val_r2"].append(val_metrics["r2"])
        history["val_mae"].append(val_metrics["mae"])

        is_best = val_metrics["r2"] > best_val_r2
        if is_best:
            best_val_r2 = val_metrics["r2"]
            best_epoch  = epoch + 1
            patience_counter = 0
            save_checkpoint(
                model, epoch, val_metrics["r2"], mean, std, args.output,
                optimizer=optimizer, scheduler=scheduler, history=history,
                feat_stats=feat_stats,
                target_mode=effective_target_mode,
                target_meta=target_meta,
            )
            print(f"Epoch {epoch+1:3d} | loss {train_loss:.4f} | "
                  f"val R²={val_metrics['r2']:.4f}  MAE={val_metrics['mae']:.4f}  ★ BEST")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch+1:3d} | loss {train_loss:.4f} | "
                      f"val R²={val_metrics['r2']:.4f}  MAE={val_metrics['mae']:.4f} "
                      f" lr={lr:.1e}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {args.patience} epochs).")
            break

    print(f"\n{'='*60}")
    print(f"Test results  (best model from epoch {best_epoch})")
    print(f"{'='*60}")

    checkpoint = load_checkpoint(args.output)
    model = ImprovedGIN_GAT.from_checkpoint(checkpoint).to(device)
    effective_target_mode = checkpoint.get("target_mode", effective_target_mode)
    if effective_target_mode == "softmax":
        bin_centers = checkpoint.get("softmax_bin_centers", bin_centers)

    test_metrics = evaluate(model, test_loader, device, mean, std,
                            target_mode=effective_target_mode,
                            bin_centers=bin_centers)
    print(f"  R²           : {test_metrics['r2']:.4f}")
    print(f"  MAE          : {test_metrics['mae']:.4f}")
    print(f"  RMSE         : {test_metrics['rmse']:.4f}")
    print(f"  Pearson r    : {test_metrics['correlation']:.4f}")

    if args.uncertainty:
        print("\nComputing Monte Carlo Dropout uncertainty (50 passes)…")
        unc = predict_with_uncertainty(
            model, test_loader, device, mean, std, n_samples=50,
            target_mode=effective_target_mode, bin_centers=bin_centers,
        )
        print(f"  Mean σ : {unc['std'].mean():.4f}")

    # Save test R² back into the checkpoint so the GUI can display it
    checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
    checkpoint["test_r2"] = float(test_metrics["r2"])
    checkpoint["test_mae"] = float(test_metrics["mae"])
    checkpoint["test_rmse"] = float(test_metrics["rmse"])
    checkpoint["test_pearson"] = float(test_metrics["correlation"])
    torch.save(checkpoint, args.output)
    logger.info("Test metrics saved into checkpoint.")

    plot_path = args.output.replace(".pt", "_results.png")
    create_plots(test_metrics, history, plot_path)

    return {
        "test_r2":      test_metrics["r2"],
        "test_mae":     test_metrics["mae"],
        "best_val_r2":  best_val_r2,
        "best_epoch":   best_epoch,
        "model_path":   args.output,
    }

def evaluate_ensemble(
    model_paths: List[str],
    data_dir: str,
) -> Dict:
    """Evaluate an ensemble by averaging denormalised predictions.

    Unlike the per-model test scores stored during training (which measure
    each model individually), this function combines predictions from all
    members before computing metrics.  Each model's output is denormalised
    with its own stored mean/std so that members with slightly different
    normalisations are handled correctly.

    Returns
    -------
    dict with keys: r2, mae, rmse, preds, targets, n_models
    """
    device = get_device()
    data_dir_path = Path(data_dir)

    with open(data_dir_path / "splits.json") as f:
        splits = json.load(f)

    raw_test: List = []
    for fname in splits["test"]:
        fpath = data_dir_path / fname
        if not fpath.exists():
            continue
        d = torch.load(fpath, weights_only=False)
        if hasattr(d, "complexity_score") and d.gate_count > 0:
            raw_test.append(d)

    if not raw_test:
        raise RuntimeError("Test set is empty — check data_dir and splits.json.")

    raw_targets = np.array([d.complexity_score for d in raw_test])
    for d in raw_test:
        d.y_raw = torch.tensor([[d.complexity_score]], dtype=torch.float)

    all_preds: List[np.ndarray] = []
    for path in model_paths:
        if not Path(path).exists():
            logger.warning("Ensemble member not found, skipping: %s", path)
            continue
        ckpt = load_checkpoint(path)
        model = ImprovedGIN_GAT.from_checkpoint(ckpt).to(device)
        m, s = ckpt["mean"], ckpt["std"]
        tmode = ckpt.get("target_mode", "regression")
        centers = ckpt.get("softmax_bin_centers")

        if tmode == "softmax":
            edges = ckpt.get("softmax_bin_edges")
            if edges:
                edge_arr = np.array(edges, dtype=np.float32)
                for d in raw_test:
                    b = int(_assign_bin_indices(np.array([d.complexity_score], dtype=np.float32), edge_arr)[0])
                    d.y_bin = torch.tensor([b], dtype=torch.long)
            for d in raw_test:
                d.y = d.y_raw.clone()
        else:
            for d in raw_test:
                d.y = torch.tensor([[(d.complexity_score - m) / s]], dtype=torch.float)

        loader = DataLoader(raw_test, batch_size=64, shuffle=False)
        metrics = evaluate(
            model, loader, device, m, s,
            target_mode=tmode, bin_centers=centers,
        )
        all_preds.append(metrics["preds"])

    if not all_preds:
        raise RuntimeError("No ensemble members could be loaded.")

    ensemble_preds = np.mean(all_preds, axis=0)
    errors = ensemble_preds - raw_targets
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((raw_targets - raw_targets.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "r2":       r2,
        "mae":      float(np.mean(np.abs(errors))),
        "rmse":     float(np.sqrt(np.mean(errors ** 2))),
        "preds":    ensemble_preds,
        "targets":  raw_targets,
        "n_models": len(all_preds),
    }

def train_ensemble(args: argparse.Namespace) -> None:
    """Train an ensemble of *args.ensemble* models with different seeds.

    Normalisation (mean / std) is computed once from the training set and
    shared across all members so that every model predicts on the same scale.
    After training, ensemble inference (averaged predictions) is evaluated to
    give the true ensemble R² — which is typically better than any individual
    member's solo score.

    Note: for best results, ensemble members should use the same
    architecture hyperparameters (--hidden-dim, --dropout) that produced
    the best single model.
    """
    print(f"\n{'='*70}")
    print(f"Training ensemble of {args.ensemble} models")
    print(f"{'='*70}")

    shared_mean, shared_std = None, None
    if args.target_mode == "regression":
        _, _, _, shared_mean, shared_std, _, _ = load_data(
            args.data_dir, target_mode="regression"
        )
        logger.info(
            "Shared normalisation: mean=%.4f  std=%.4f", shared_mean, shared_std
        )
    else:
        logger.info("Softmax mode: skipping shared target normalisation.")

    results = []
    base_output = args.output
    for i in range(args.ensemble):
        print(f"\n--- Model {i+1}/{args.ensemble} ---")
        args.output = base_output.replace(".pt", f"_ensemble_{i+1}.pt")
        results.append(
            train_model(args, seed=42 + i * 100,
                        shared_mean=shared_mean, shared_std=shared_std)
        )

    r2s  = [r["test_r2"]  for r in results]
    maes = [r["test_mae"] for r in results]

    print(f"\n{'='*60}  Ensemble Summary")
    print(f"  {'Model':<12} {'R²':>8} {'MAE':>8}  (individual)")
    for i, r in enumerate(results):
        print(f"  Model {i+1:<6} {r['test_r2']:>8.4f} {r['test_mae']:>8.4f}")
    print(f"  {'Mean':<12} {np.mean(r2s):>8.4f} {np.mean(maes):>8.4f}")
    print(f"  {'Std':<12} {np.std(r2s):>8.4f} {np.std(maes):>8.4f}")

    model_paths = [r["model_path"] for r in results]
    ensemble_r2, ensemble_mae = float("nan"), float("nan")
    try:
        ens = evaluate_ensemble(model_paths, args.data_dir)
        ensemble_r2  = ens["r2"]
        ensemble_mae = ens["mae"]
        print(f"\n  Ensemble R²  (averaged predictions): {ensemble_r2:.4f}")
        print(f"  Ensemble MAE                        : {ensemble_mae:.4f}")
    except Exception as exc:
        logger.warning("Ensemble evaluation failed: %s", exc)

    summary = {
        "models":           [r["model_path"] for r in results],
        "test_r2_mean":     float(np.mean(r2s)),
        "test_r2_std":      float(np.std(r2s)),
        "test_mae_mean":    float(np.mean(maes)),
        "test_mae_std":     float(np.std(maes)),
        "ensemble_r2":      ensemble_r2,
        "ensemble_mae":     ensemble_mae,
    }
    with open("ensemble_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved ensemble_summary.json")

def _run_ablation(args: argparse.Namespace) -> None:
    """Run ablation study: GNN-only, Global-only, and Combined.

    Trains three model variants and prints a comparison table.
    Each variant uses the same data and training hyperparameters.
    """
    import copy

    configs = [
        {
            "name": "GNN-only (no global branch)",
            "suffix": "_gnn_only",
            "global_dim": 0,
            "gin_layers": args.gin_layers,
            "gat_layers": args.gat_layers,
        },
        {
            "name": "Global-only (no GNN layers)",
            "suffix": "_global_only",
            "global_dim": args.global_dim,
            "gin_layers": 0,
            "gat_layers": 0,
        },
        {
            "name": "Combined (GNN + Global)",
            "suffix": "_combined",
            "global_dim": args.global_dim,
            "gin_layers": args.gin_layers,
            "gat_layers": args.gat_layers,
        },
    ]

    results = []
    base_output = args.output

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"ABLATION: {cfg['name']}")
        print(f"{'='*70}")

        a = copy.deepcopy(args)
        a.global_dim  = cfg["global_dim"]
        a.gin_layers  = cfg["gin_layers"]
        a.gat_layers  = cfg["gat_layers"]
        a.output      = base_output.replace(".pt", f"{cfg['suffix']}.pt")
        a.resume      = None

        r = train_model(a)
        r["variant"] = cfg["name"]
        results.append(r)

    print(f"\n{'='*70}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*70}")
    print(f"{'Variant':<35s} {'Test R2':>8s} {'Test MAE':>9s} {'Best ValR2':>10s} {'Epoch':>6s}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['variant']:<35s} {r['test_r2']:>8.4f} {r['test_mae']:>9.4f} "
              f"{r['best_val_r2']:>10.4f} {r['best_epoch']:>6d}")
    print(f"{'='*70}")

    summary = {
        "ablation_results": [
            {"variant": r["variant"], "test_r2": r["test_r2"],
             "test_mae": r["test_mae"], "best_val_r2": r["best_val_r2"],
             "best_epoch": r["best_epoch"], "model_path": r["model_path"]}
            for r in results
        ]
    }
    out_path = base_output.replace(".pt", "_ablation.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAblation summary saved to: {out_path}")

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Train GIN-GAT Circuit Complexity Predictor")

    p.add_argument("--data-dir",  default="data/processed_complexity")
    p.add_argument("--output",    default="best_complexity_model_v2.pt")

    p.add_argument("--hidden-dim",  type=int,   default=32)
    p.add_argument("--gin-layers",  type=int,   default=2)
    p.add_argument("--gat-layers",  type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.15)
    p.add_argument("--global-dim",  type=int,   default=10)

    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--weight-decay",type=float, default=0.01)
    p.add_argument("--patience",    type=int,   default=40)
    p.add_argument("--input-noise", type=float, default=0.005)

    p.add_argument("--loss",      choices=["mse", "smoothl1", "huber"], default="huber")
    p.add_argument("--scheduler", choices=["plateau", "cosine"],        default="plateau")
    p.add_argument("--target-mode", choices=["regression", "softmax"],
                   default="regression",
                   help="regression: scalar score target; softmax: K-bin classification target.")
    p.add_argument("--softmax-bins", type=int, default=10,
                   help="Number of quantile bins for --target-mode softmax.")
    p.add_argument("--softmax-balanced", action="store_true",
                   help="Use inverse-frequency class weights in softmax mode.")

    p.add_argument("--resume", default=None, metavar="CHECKPOINT",
                   help="Path to a checkpoint to resume training from.  "
                        "Restores model weights, optimiser, scheduler, and history.  "
                        "Set --epochs to the new total budget (must be > checkpoint epoch).")

    p.add_argument("--ensemble",    type=int,   default=0,
                   help="Train N models with different seeds (0 = single model).")
    p.add_argument("--uncertainty", action="store_true",
                   help="Estimate prediction uncertainty via Monte Carlo Dropout.")
    p.add_argument("--ablation", action="store_true",
                   help="Run ablation study: GNN-only, Global-only, and Combined.")

    args = p.parse_args()

    print(f"\n{'='*70}")
    print("GIN+GAT Circuit Complexity Predictor")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    if args.ablation:
        _run_ablation(args)
    elif args.ensemble > 0:
        train_ensemble(args)
    else:
        train_model(args)

    print(f"\n{'='*70}")
    print("Done.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

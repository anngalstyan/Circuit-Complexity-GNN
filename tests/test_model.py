"""
Tests for ImprovedGIN_GAT model: architecture, forward pass,
checkpoint round-trip, and evaluate() metrics.

Run from the project root:
    pytest tests/test_model.py -v
"""

import sys
import math
import tempfile
import os
from pathlib import Path

import pytest
import numpy as np
import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from circuit_complexity_model import (
    ImprovedGIN_GAT,
    evaluate,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n_nodes: int = 6, n_edges: int = 5, seed: int = 0) -> Data:
    """Create a random Data object with 24-dim node features."""
    torch.manual_seed(seed)
    x = torch.rand(n_nodes, 24)
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    y = torch.tensor([[2.5]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.gate_count = n_nodes
    data.global_feats = torch.tensor([
        math.log10(n_nodes + 1) / 5.0,  # log(gates)
        0.1,                             # depth/50
        0.0, 0.0, 0.0,                  # feedback, seq_ratio, scc_cov
        0.0, 0.0,                        # xor_ratio, reconvergent
        0.5, 0.2, 0.3,                  # type_entropy, avg_fanout, edge_density
    ])
    return data


def _make_loader(n_graphs: int = 4):
    from torch_geometric.loader import DataLoader
    graphs = [_make_graph(n_nodes=5 + i, seed=i) for i in range(n_graphs)]
    return DataLoader(graphs, batch_size=2, shuffle=False), graphs


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestModelArchitecture:
    def test_default_construction(self):
        model = ImprovedGIN_GAT()
        assert model.hidden_dim == 64
        assert model.num_gin_layers == 2
        assert model.num_gat_layers == 2
        assert model.global_dim == 10

    def test_custom_construction(self):
        model = ImprovedGIN_GAT(input_dim=24, hidden_dim=32, num_gin_layers=1,
                                num_gat_layers=1, gat_heads=2, dropout=0.1)
        assert model.hidden_dim == 32
        assert model.num_gin_layers == 1
        assert model.num_gat_layers == 1

    def test_custom_global_dim(self):
        model = ImprovedGIN_GAT(global_dim=6)
        assert model.global_dim == 6
        cfg = model.get_config()
        assert cfg["global_dim"] == 6

    def test_get_config_roundtrip(self):
        model = ImprovedGIN_GAT(hidden_dim=32, dropout=0.1)
        cfg = model.get_config()
        assert cfg["hidden_dim"] == 32
        assert cfg["dropout"] == 0.1
        assert "input_dim" in cfg
        assert "global_dim" in cfg

    def test_parameter_count_positive(self):
        model = ImprovedGIN_GAT()
        n = sum(p.numel() for p in model.parameters())
        assert n > 0

    def test_repr_contains_dims(self):
        model = ImprovedGIN_GAT(hidden_dim=48)
        r = repr(model)
        assert "48" in r


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape_single_graph(self):
        model = ImprovedGIN_GAT()
        model.eval()
        data = _make_graph()
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (1, 1)

    def test_output_is_finite(self):
        model = ImprovedGIN_GAT()
        model.eval()
        data = _make_graph()
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        with torch.no_grad():
            out = model(data)
        assert torch.isfinite(out).all()

    def test_batched_output_shape(self):
        model = ImprovedGIN_GAT()
        model.eval()
        loader, _ = _make_loader(n_graphs=4)
        for batch in loader:
            with torch.no_grad():
                out = model(batch)
            assert out.shape[0] == batch.num_graphs
            assert out.shape[1] == 1

    def test_training_mode_dropout_varies(self):
        """With dropout active, two forward passes on the same input should differ."""
        torch.manual_seed(0)
        model = ImprovedGIN_GAT(dropout=0.5)
        model.train()
        data = _make_graph(n_nodes=20, n_edges=30)
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        out1 = model(data)
        out2 = model(data)
        # With p=0.5 dropout on 20+ nodes, outputs will almost certainly differ
        assert not torch.allclose(out1, out2)

    def test_eval_mode_deterministic(self):
        """In eval mode two passes should produce identical outputs."""
        torch.manual_seed(0)
        model = ImprovedGIN_GAT()
        model.eval()
        data = _make_graph(n_nodes=10, n_edges=15)
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        with torch.no_grad():
            out1 = model(data)
            out2 = model(data)
        assert torch.allclose(out1, out2)

    def test_backward_pass_gradients(self):
        model = ImprovedGIN_GAT()
        model.train()
        data = _make_graph()
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        out = model(data)
        loss = out.mean()
        loss.backward()
        # At least one parameter should have a non-None gradient
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_no_global_feats_fallback(self):
        """Model should handle missing global_feats gracefully."""
        model = ImprovedGIN_GAT()
        model.eval()
        data = _make_graph()
        del data.global_feats  # remove the attribute
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# Checkpoint round-trip tests
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_and_load_checkpoint(self):
        model = ImprovedGIN_GAT(hidden_dim=32)
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            save_checkpoint(model, epoch=5, val_r2=0.85,
                            mean=1.5, std=0.9, path=path)
            ckpt = load_checkpoint(path)
            assert ckpt["epoch"] == 5
            assert abs(ckpt["val_r2"] - 0.85) < 1e-6
            assert abs(ckpt["mean"] - 1.5) < 1e-6
            assert abs(ckpt["std"] - 0.9) < 1e-6
        finally:
            os.unlink(path)

    def test_from_checkpoint_restores_weights(self):
        model = ImprovedGIN_GAT(hidden_dim=32)
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            save_checkpoint(model, epoch=1, val_r2=0.5,
                            mean=1.0, std=1.0, path=path)
            ckpt = load_checkpoint(path)
            restored = ImprovedGIN_GAT.from_checkpoint(ckpt)
            # Weights should be identical
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), restored.named_parameters()
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)

    def test_checkpoint_preserves_config(self):
        model = ImprovedGIN_GAT(hidden_dim=48, dropout=0.2,
                                num_gin_layers=3, num_gat_layers=1)
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            save_checkpoint(model, epoch=10, val_r2=0.9,
                            mean=2.0, std=0.8, path=path)
            ckpt = load_checkpoint(path)
            cfg = ckpt["model_config"]
            assert cfg["hidden_dim"] == 48
            assert cfg["dropout"] == 0.2
            assert cfg["num_gin_layers"] == 3
            assert cfg["num_gat_layers"] == 1
        finally:
            os.unlink(path)

    def test_load_checkpoint_uses_weights_only(self):
        """load_checkpoint must use weights_only=True (no arbitrary code exec)."""
        import inspect
        src = inspect.getsource(load_checkpoint)
        assert "weights_only=True" in src

    def test_save_with_optimizer_state(self):
        model = ImprovedGIN_GAT(hidden_dim=32)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            save_checkpoint(model, epoch=2, val_r2=0.7,
                            mean=1.0, std=1.0, path=path, optimizer=opt)
            ckpt = load_checkpoint(path)
            assert "optimizer_state_dict" in ckpt
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# evaluate() function tests
# ---------------------------------------------------------------------------

class TestEvaluate:
    def _run_evaluate(self, mean: float = 2.0, std: float = 1.0):
        from torch_geometric.loader import DataLoader
        torch.manual_seed(42)
        model = ImprovedGIN_GAT(hidden_dim=32)
        model.eval()
        # Create graphs with known normalised targets
        graphs = []
        for i in range(8):
            g = _make_graph(n_nodes=6, seed=i)
            # Normalised target: raw - mean / std
            raw = 2.0 + i * 0.1
            g.y = torch.tensor([[(raw - mean) / std]], dtype=torch.float)
            g.gate_count = 6
            graphs.append(g)
        loader = DataLoader(graphs, batch_size=4, shuffle=False)
        return evaluate(model, loader, torch.device("cpu"), mean, std)

    def test_evaluate_returns_required_keys(self):
        metrics = self._run_evaluate()
        for key in ("mae", "mse", "rmse", "r2", "correlation", "preds", "targets"):
            assert key in metrics

    def test_evaluate_preds_finite(self):
        metrics = self._run_evaluate()
        assert np.all(np.isfinite(metrics["preds"]))

    def test_evaluate_targets_denormalized(self):
        mean, std = 2.0, 1.0
        metrics = self._run_evaluate(mean=mean, std=std)
        # Targets should be back in original scale (~2.0–2.7)
        assert metrics["targets"].min() >= 1.5
        assert metrics["targets"].max() <= 3.5

    def test_perfect_model_r2_near_one(self):
        """A model that memorises targets should score R² ≈ 1."""
        from torch_geometric.loader import DataLoader

        mean, std = 2.0, 1.0
        graphs = [_make_graph(seed=i) for i in range(6)]
        raw_targets = [2.0 + i * 0.3 for i in range(6)]
        for g, t in zip(graphs, raw_targets):
            g.y = torch.tensor([[(t - mean) / std]], dtype=torch.float)

        # Build a model that simply returns the stored target (mock)
        class PerfectModel(torch.nn.Module):
            def forward(self, data):
                return data.y

        loader = DataLoader(graphs, batch_size=6, shuffle=False)
        metrics = evaluate(PerfectModel(), loader, torch.device("cpu"), mean, std)
        assert metrics["r2"] > 0.99

    def test_mae_is_non_negative(self):
        metrics = self._run_evaluate()
        assert metrics["mae"] >= 0.0

    def test_rmse_geq_mae(self):
        metrics = self._run_evaluate()
        assert metrics["rmse"] >= metrics["mae"] - 1e-6

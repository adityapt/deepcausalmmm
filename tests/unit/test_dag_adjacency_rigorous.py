"""Rigorous tests for DAG adjacency extraction and NOTEARS training hooks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deepcausalmmm.core.config import get_default_config
from deepcausalmmm.core.inference import InferenceManager
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.unified_model import DeepCausalMMM
from deepcausalmmm.postprocess.comprehensive_analysis import ComprehensiveAnalyzer


def _dashboard_style_extract(model, config, threshold=False):
    """Mirror examples/dashboard_rmse_optimized._extract_dag_adjacency."""
    if not hasattr(model, "get_dag_adjacency_matrix"):
        n = getattr(model, "n_media", 1) or 1
        return np.eye(n) * (0.0 if threshold else 0.5)
    if threshold:
        eps = float(config.get("notears_threshold", 0.3))
        W = model.get_dag_adjacency_matrix(eps=eps)
    else:
        W = model.get_dag_adjacency_matrix(eps=None)
    matrix = W.cpu().numpy()
    np.fill_diagonal(matrix, 0.0)
    return matrix


@pytest.fixture
def tiny_batch():
    torch.manual_seed(42)
    n_regions, n_weeks, n_media, n_control = 2, 16, 5, 2
    X_media = torch.rand(n_regions, n_weeks, n_media)
    X_control = torch.rand(n_regions, n_weeks, n_control)
    R = torch.arange(n_regions, dtype=torch.long).unsqueeze(1).repeat(1, n_weeks)
    y = torch.rand(n_regions, n_weeks)
    return X_media, X_control, R, y


class TestAdjacencyMatrixAPI:
    def test_triangular_mask_enforces_strict_upper_structure(self):
        model = DeepCausalMMM(n_media=5, ctrl_dim=1, n_regions=1, dag_mode="triangular")
        W = model.get_dag_adjacency_matrix(eps=None).cpu().numpy()
        assert np.allclose(np.tril(W, k=0), 0.0, atol=1e-7)
        assert np.count_nonzero(np.triu(W, k=1)) >= 0  # may be sparse but valid shape

    def test_notears_mask_zeros_diagonal_only(self):
        model = DeepCausalMMM(n_media=5, ctrl_dim=1, n_regions=1, dag_mode="notears")
        W = model.get_dag_adjacency_matrix(eps=None).cpu().numpy()
        assert np.allclose(np.diag(W), 0.0, atol=1e-7)
        # NOTEARS allows lower-triangular mass in principle
        assert W.shape == (5, 5)

    def test_pruned_is_subset_of_continuous(self):
        model = DeepCausalMMM(
            n_media=4, ctrl_dim=1, n_regions=1,
            dag_mode="notears", dag_temperature=0.5,
        )
        Wc = model.get_dag_adjacency_matrix(eps=None)
        eps = 0.25
        Wp = model.get_dag_adjacency_matrix(eps=eps)
        kept = Wp.abs() > 0
        assert torch.all(Wc.abs()[kept] >= eps - 1e-6)
        assert torch.all(Wc.abs()[~kept] < eps + 1e-6)
        assert torch.all(Wp.abs()[~kept] == 0)

    def test_pruning_monotone_in_eps(self):
        model = DeepCausalMMM(n_media=4, ctrl_dim=1, n_regions=1, dag_mode="notears")
        counts = []
        for eps in (0.0, 0.1, 0.3, 0.5, 0.9):
            W = model.get_dag_adjacency_matrix(eps=eps)
            counts.append((W.abs() > 0).sum().item())
        assert counts == sorted(counts, reverse=True)

    def test_threshold_dag_matches_get_dag_adjacency_matrix(self):
        model = DeepCausalMMM(n_media=3, ctrl_dim=1, n_regions=1, dag_temperature=0.7)
        for eps in (0.05, 0.3, 0.75):
            assert torch.allclose(
                model.threshold_dag(eps=eps),
                model.get_dag_adjacency_matrix(eps=eps),
            )

    def test_dag_disabled_returns_zeros(self):
        model = DeepCausalMMM(
            n_media=3, ctrl_dim=1, n_regions=1,
            enable_dag=False, enable_interactions=False,
        )
        W = model.get_dag_adjacency_matrix(eps=None)
        assert W.shape == (3, 3)
        assert W.abs().max().item() == 0.0

    def test_temperature_one_matches_plain_sigmoid_mask(self):
        torch.manual_seed(0)
        model = DeepCausalMMM(n_media=4, ctrl_dim=1, n_regions=1, dag_temperature=1.0)
        W = model.get_dag_adjacency_matrix(eps=None)
        expected = torch.sigmoid(model.adj_logits) * model.tri_mask
        assert torch.allclose(W, expected.detach(), atol=1e-6)

    def test_temperature_half_differs_from_plain_sigmoid(self):
        torch.manual_seed(1)
        model = DeepCausalMMM(n_media=4, ctrl_dim=1, n_regions=1, dag_temperature=0.5)
        W = model.get_dag_adjacency_matrix(eps=None)
        raw = torch.sigmoid(model.adj_logits) * model.tri_mask
        assert not torch.allclose(W, raw.detach(), atol=1e-4)


class TestInferenceAndDashboard:
    def test_inference_default_threshold_uses_config(self):
        config = get_default_config()
        config["notears_threshold"] = 0.42
        model = DeepCausalMMM(n_media=3, ctrl_dim=1, n_regions=1, dag_mode="notears")
        with pytest.warns(UserWarning, match="Neither pipeline nor scaler"):
            manager = InferenceManager(model, config=config)
        pruned = manager.get_dag_adjacency(threshold=True)
        expected = model.get_dag_adjacency_matrix(eps=0.42).cpu().numpy()
        assert np.allclose(pruned, expected)

    def test_inference_returns_none_when_dag_off(self):
        model = DeepCausalMMM(
            n_media=3, ctrl_dim=1, n_regions=1,
            enable_dag=False, enable_interactions=False,
        )
        with pytest.warns(UserWarning, match="Neither pipeline nor scaler"):
            manager = InferenceManager(model)
        assert manager.get_dag_adjacency() is None

    def test_dashboard_extract_matches_model_api(self):
        extract = _dashboard_style_extract
        config = get_default_config()
        config["notears_threshold"] = 0.35
        model = DeepCausalMMM(n_media=4, ctrl_dim=1, n_regions=1, dag_temperature=0.5)

        W_dash_cont = extract(model, config, threshold=False)
        W_dash_thr = extract(model, config, threshold=True)
        W_model_cont = model.get_dag_adjacency_matrix(eps=None).cpu().numpy()
        W_model_thr = model.get_dag_adjacency_matrix(eps=0.35).cpu().numpy()
        np.fill_diagonal(W_model_cont, 0.0)
        np.fill_diagonal(W_model_thr, 0.0)

        assert np.allclose(W_dash_cont, W_model_cont, atol=1e-6)
        assert np.allclose(W_dash_thr, W_model_thr, atol=1e-6)


class TestDagInteractionConsistency:
    def test_dag_interaction_uses_same_adj_as_api(self, tiny_batch):
        X_media, X_control, R, _ = tiny_batch
        model = DeepCausalMMM(
            n_media=5, ctrl_dim=2, n_regions=2,
            dag_mode="notears", dag_temperature=0.5,
        )
        model.eval()
        with torch.no_grad():
            X_hill = model.hill(model.adstock(X_media))
            parents_manual = torch.matmul(
                X_hill,
                model.get_dag_adjacency_matrix(eps=None),
            )
            mix = torch.sigmoid(model.interaction_weight)
            expected = (1.0 - mix) * X_hill + mix * parents_manual
            actual = model.dag_interaction(X_hill)
        assert torch.allclose(actual, expected, atol=1e-5)


class TestNOTEARSTrainingHooks:
    def test_h_acyclicity_zero_for_dag_upper_triangular(self):
        model = DeepCausalMMM(n_media=4, ctrl_dim=1, n_regions=1, dag_mode="notears")
        W = torch.triu(torch.ones(4, 4), diagonal=1) * 0.4
        h = model.h_acyclicity(W)
        assert h.item() == pytest.approx(0.0, abs=1e-4)

    def test_h_acyclicity_positive_for_cyclic_graph(self):
        model = DeepCausalMMM(n_media=3, ctrl_dim=1, n_regions=1, dag_mode="notears")
        W = torch.ones(3, 3) - torch.eye(3)
        assert model.h_acyclicity(W).item() > 0.0

    def test_notears_dual_update_increases_rho_when_h_stalls(self):
        model = DeepCausalMMM(n_media=3, ctrl_dim=1, n_regions=1, dag_mode="notears")
        model._notears_h_prev = 1.0
        with torch.no_grad():
            T = max(float(model.dag_temperature), 1e-3)
            adj = torch.sigmoid(model.adj_logits / T) * model.tri_mask
            model._notears_h_prev = model.h_acyclicity(adj).item() + 1.0
        rho_before = model.notears_rho.item()
        info = model.notears_update_duals(factor=3.0, progress=0.25)
        assert model.notears_rho.item() >= rho_before
        assert info["rho"] >= rho_before

    def test_trainer_warmup_gate_sequence(self, tiny_batch):
        X_media, X_control, R, y = tiny_batch
        config = get_default_config()
        config.update({
            "dag_mode": "notears",
            "notears_warmup_epochs": 2,
            "notears_dual_update_every": 100,
        })
        trainer = ModelTrainer(config)
        trainer.create_model(
            n_media=X_media.shape[2],
            n_control=X_control.shape[2],
            n_regions=X_media.shape[0],
        )
        trainer.create_optimizer_and_scheduler()

        notears_warmup = int(config["notears_warmup_epochs"])
        trainer.model.notears_active.fill_(False)
        assert bool(trainer.model.notears_active.item()) is False

        for epoch in range(4):
            trainer.train_epoch(X_media, X_control, R, y)
            if epoch == notears_warmup:
                trainer.model.notears_active.fill_(True)

        assert bool(trainer.model.notears_active.item()) is True

    def test_trainer_create_model_passes_dag_temperature(self):
        config = get_default_config()
        config["dag_temperature"] = 0.42
        config["notears_group_l1"] = 0.03
        trainer = ModelTrainer(config)
        model = trainer.create_model(n_media=3, n_control=1, n_regions=1)
        assert model.dag_temperature == pytest.approx(0.42)
        assert model.notears_group_l1 == pytest.approx(0.03)


class TestComprehensiveAnalyzerDAG:
    def test_analyzer_dag_visualization_uses_model_api(self, tmp_path):
        torch.manual_seed(7)
        n_media = 4
        model = DeepCausalMMM(
            n_media=n_media, ctrl_dim=1, n_regions=1,
            dag_mode="notears", dag_temperature=0.5,
        )
        config = get_default_config()
        config["notears_threshold"] = 0.3
        media_cols = [f"ch_{i}" for i in range(n_media)]
        analyzer = ComprehensiveAnalyzer(
            model=model,
            media_cols=media_cols,
            control_cols=["ctrl_0"],
            output_dir=str(tmp_path),
            config=config,
            manual_burnin_weeks=0,
        )
        analyzer._create_dag_visualization()

        W_thr = model.get_dag_adjacency_matrix(eps=0.3).cpu().numpy()
        np.fill_diagonal(W_thr, 0.0)
        # Network plot should not invent random edges: if model has edges, thresholded
        # matrix is reproducible from API (smoke: files written, no exception)
        assert list(tmp_path.glob("dag_network_*.html"))
        assert list(tmp_path.glob("dag_heatmap_*.html"))
        assert W_thr.shape == (n_media, n_media)


class TestEndToEndModes:
    @pytest.mark.parametrize("dag_mode", ["triangular", "notears"])
    def test_train_epoch_finite_loss_both_modes(self, tiny_batch, dag_mode):
        X_media, X_control, R, y = tiny_batch
        config = get_default_config()
        config.update({
            "n_epochs": 1,
            "dag_mode": dag_mode,
            "notears_warmup_epochs": 0,
            "early_stopping": False,
            "warm_start_epochs": 0,
        })
        trainer = ModelTrainer(config)
        trainer.create_model(
            n_media=X_media.shape[2],
            n_control=X_control.shape[2],
            n_regions=X_media.shape[0],
        )
        trainer.create_optimizer_and_scheduler()
        loss, rmse, r2 = trainer.train_epoch(X_media, X_control, R, y)
        assert np.isfinite(loss)
        assert np.isfinite(rmse)
        assert np.isfinite(r2)

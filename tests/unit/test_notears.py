"""NOTEARS DAG structure-learning smoke tests."""

import torch
import pytest

from deepcausalmmm.core.unified_model import DeepCausalMMM
from deepcausalmmm.core.config import get_default_config


@pytest.fixture
def tiny_tensors():
    n_regions, n_weeks, n_media, n_control = 2, 12, 4, 2
    X_media = torch.rand(n_regions, n_weeks, n_media)
    X_control = torch.rand(n_regions, n_weeks, n_control)
    R = torch.arange(n_regions).long().unsqueeze(1).repeat(1, n_weeks)
    return X_media, X_control, R


def test_notears_config_keys_present():
    config = get_default_config()
    for key in (
        'dag_mode',
        'notears_lambda1',
        'notears_warmup_epochs',
        'notears_dual_factor',
        'notears_threshold',
        'dag_temperature',
        'notears_group_l1',
    ):
        assert key in config
    assert config['dag_mode'] == 'triangular'
    assert 'dag_top_n_edges' in config['visualization']


def test_triangular_and_notears_forward_backward(tiny_tensors):
    X_media, X_control, R = tiny_tensors
    for mode in ('triangular', 'notears'):
        model = DeepCausalMMM(
            n_media=4,
            ctrl_dim=2,
            n_regions=2,
            dag_mode=mode,
            notears_group_l1=0.01 if mode == 'notears' else 0.0,
        )
        y_pred, _, _, outputs = model(X_media, X_control, R)
        assert y_pred.shape == (2, 12)
        loss = y_pred.sum() + model.get_dag_loss()
        loss.backward()


def test_h_acyclicity_zero_on_empty_graph():
    model = DeepCausalMMM(n_media=3, ctrl_dim=1, n_regions=1, dag_mode='notears')
    W = torch.zeros(3, 3)
    h = model.h_acyclicity(W)
    assert h.item() == pytest.approx(0.0, abs=1e-5)


def test_notears_update_duals_returns_metrics():
    model = DeepCausalMMM(n_media=3, ctrl_dim=1, n_regions=1, dag_mode='notears')
    info = model.notears_update_duals(factor=3.0)
    assert 'h' in info and 'rho' in info and 'alpha' in info


def test_threshold_dag_respects_eps():
    model = DeepCausalMMM(n_media=3, ctrl_dim=1, n_regions=1, dag_mode='notears')
    W = model.threshold_dag(eps=0.99)
    assert W.shape == (3, 3)
    assert W.abs().max().item() <= 1.0 + 1e-6


def test_notears_warmup_gate_disables_dag_penalty():
    model = DeepCausalMMM(n_media=3, ctrl_dim=1, n_regions=1, dag_mode='notears')
    active_loss = model.get_dag_loss().item()
    model.notears_active.fill_(False)
    warmup_loss = model.get_dag_loss().item()
    # NOTEARS acyclicity term is off; only a tiny L1 on logits may remain.
    assert warmup_loss < active_loss * 0.01

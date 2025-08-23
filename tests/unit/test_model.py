"""Test DeepCausalMMM model functionality."""

import pytest
import torch
import numpy as np
from deepcausalmmm import DeepCausalMMM, get_default_config


@pytest.fixture
def sample_config():
    """Get a sample configuration for testing."""
    config = get_default_config()
    config['n_epochs'] = 10  # Reduce for testing
    return config


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    n_regions = 2
    n_timesteps = 20
    n_channels = 3
    n_controls = 2
    
    X_media = torch.randn(n_regions, n_timesteps, n_channels)
    X_control = torch.randn(n_regions, n_timesteps, n_controls)
    R = torch.arange(n_regions).long().unsqueeze(1).repeat(1, n_timesteps)  # Must be long for indexing
    y = torch.randn(n_regions, n_timesteps)
    
    return X_media, X_control, R, y


def test_model_initialization(sample_config):
    """Test that model initializes correctly."""
    model = DeepCausalMMM(
        n_media=3,
        ctrl_dim=2,
        n_regions=2
    )
    
    assert model.n_media == 3
    assert model.ctrl_dim == 2
    assert model.n_regions == 2


def test_model_forward_pass(sample_config, sample_data):
    """Test model forward pass produces expected outputs."""
    X_media, X_control, R, y = sample_data
    
    model = DeepCausalMMM(
        n_media=3,
        ctrl_dim=2,
        n_regions=2
    )
    
    # Forward pass
    y_pred, media_contrib, control_contrib, outputs = model(X_media, X_control, R)
    
    # Check output shapes
    assert y_pred.shape == y.shape
    assert media_contrib.shape == X_media.shape
    # Control contribution may have different shape due to model architecture
    assert control_contrib.shape[0] == X_control.shape[0]  # Same batch size
    assert control_contrib.shape[1] == X_control.shape[1]  # Same time steps
    
    # Check outputs are tensors
    assert isinstance(y_pred, torch.Tensor)
    assert isinstance(media_contrib, torch.Tensor)
    assert isinstance(control_contrib, torch.Tensor)


def test_model_training_mode(sample_config, sample_data):
    """Test model behavior in training vs eval mode."""
    X_media, X_control, R, y = sample_data
    
    model = DeepCausalMMM(
        n_media=3,
        ctrl_dim=2,
        n_regions=2
    )
    
    # Training mode
    model.train()
    y_pred_train, _, _, _ = model(X_media, X_control, R)
    
    # Eval mode
    model.eval()
    with torch.no_grad():
        y_pred_eval, _, _, _ = model(X_media, X_control, R)
    
    # Outputs should be tensors (exact values may differ due to dropout)
    assert isinstance(y_pred_train, torch.Tensor)
    assert isinstance(y_pred_eval, torch.Tensor)
    assert y_pred_train.shape == y_pred_eval.shape


def test_model_parameters_learnable(sample_config):
    """Test that model parameters are learnable."""
    model = DeepCausalMMM(
        n_media=3,
        ctrl_dim=2,
        n_regions=2
    )
    
    # Check that model has parameters
    params = list(model.parameters())
    assert len(params) > 0
    
    # Check that parameters require gradients
    for param in params:
        assert param.requires_grad

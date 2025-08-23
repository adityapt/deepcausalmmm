"""Integration tests for end-to-end functionality."""

import pytest
import numpy as np
import torch
from deepcausalmmm import DeepCausalMMM, get_default_config
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.data import UnifiedDataPipeline
from deepcausalmmm.core.scaling import SimpleGlobalScaler


@pytest.fixture
def synthetic_mmm_data():
    """Generate synthetic MMM data for testing."""
    np.random.seed(42)
    
    n_regions = 2
    n_weeks = 104  # 2 years of weekly data
    n_media_channels = 4
    n_control_vars = 3
    
    # Generate media data (impressions/spend)
    media_data = np.random.exponential(1000, (n_regions, n_weeks, n_media_channels))
    
    # Generate control variables (temperature, holidays, etc.)
    control_data = np.random.normal(0, 1, (n_regions, n_control_vars, n_weeks))
    
    # Generate target variable (sales/visits) with some relationship to media
    base_sales = 10000
    media_effect = np.sum(media_data * 0.001, axis=2)  # Simple linear effect
    noise = np.random.normal(0, 500, (n_regions, n_weeks))
    target = base_sales + media_effect + noise
    
    return media_data, control_data, target


def test_model_trainer_basic_training(synthetic_mmm_data):
    """Test basic model training functionality."""
    media_data, control_data, target = synthetic_mmm_data
    
    # Get a fast config for testing
    config = get_default_config()
    config['n_epochs'] = 50  # Reduce for testing
    config['learning_rate'] = 0.01
    config['hidden_dim'] = 32  # Smaller for testing
    
    # Create trainer
    trainer = ModelTrainer(config)
    
    # Create model
    model = trainer.create_model(
        n_media=media_data.shape[2],
        n_control=control_data.shape[1],
        n_regions=media_data.shape[0]
    )
    
    # Create optimizer and scheduler (required before training)
    trainer.create_optimizer_and_scheduler()
    
    # Convert to tensors
    X_media = torch.FloatTensor(media_data)
    X_control = torch.FloatTensor(control_data.transpose(0, 2, 1))  # Transpose to match expected shape
    R = torch.arange(media_data.shape[0]).long().unsqueeze(1).repeat(1, media_data.shape[1])
    y = torch.FloatTensor(target)
    
    # Train model (basic training without holdout)
    results = trainer.train(X_media, X_control, R, y, verbose=False)
    
    # Check that training completed
    assert 'train_losses' in results
    assert 'train_rmses' in results
    assert 'train_r2s' in results
    assert len(results['train_losses']) > 0
    
    # Check that loss decreased (basic sanity check)
    initial_loss = results['train_losses'][0]
    final_loss = results['train_losses'][-1]
    assert final_loss < initial_loss, "Training should reduce loss"


def test_unified_data_pipeline(synthetic_mmm_data):
    """Test UnifiedDataPipeline functionality."""
    media_data, control_data, target = synthetic_mmm_data
    
    # Create pipeline
    config = get_default_config()
    pipeline = UnifiedDataPipeline(config)
    
    # Test temporal split functionality  
    train_data, holdout_data = pipeline.temporal_split(media_data, control_data, target, 0.2)
    
    # Check that split data is reasonable
    assert isinstance(train_data, dict)
    assert isinstance(holdout_data, dict)
    assert 'X_media' in train_data
    assert 'X_control' in train_data
    assert 'y' in train_data
    
    # Check shapes are reasonable
    assert train_data['X_media'].shape[1] > 0  # Has some training weeks
    assert holdout_data['X_media'].shape[1] > 0  # Has some holdout weeks
    assert train_data['X_media'].shape[1] + holdout_data['X_media'].shape[1] == target.shape[1]  # Total weeks match
    
    # Test fit and transform
    train_tensors = pipeline.fit_and_transform_training(train_data)
    
    # Check that we get tensors back
    assert 'X_media' in train_tensors
    assert 'X_control' in train_tensors
    assert 'y' in train_tensors
    assert isinstance(train_tensors['X_media'], torch.Tensor)


def test_simple_global_scaler_integration(synthetic_mmm_data):
    """Test SimpleGlobalScaler with realistic data."""
    media_data, control_data, target = synthetic_mmm_data
    
    # Create and fit scaler
    scaler = SimpleGlobalScaler()
    X_media_scaled, X_control_scaled, y_scaled = scaler.fit_transform(
        media_data, 
        control_data.transpose(0, 2, 1), 
        target
    )
    
    # Check that scaling worked
    assert scaler.fitted
    
    # Check that scaled data has reasonable properties
    # Media should be share-of-voice (sum to 1 per timestep)
    if isinstance(X_media_scaled, torch.Tensor):
        media_sums = torch.sum(X_media_scaled, axis=2).numpy()
    else:
        media_sums = np.sum(X_media_scaled, axis=2)
    np.testing.assert_allclose(media_sums, 1.0, rtol=1e-5)
    
    # Target should be log-transformed (positive values)
    if isinstance(y_scaled, torch.Tensor):
        y_scaled_np = y_scaled.numpy()
    else:
        y_scaled_np = y_scaled
    assert np.all(y_scaled_np >= 0), "Log-transformed target should be non-negative"
    
    # Test inverse transform for target
    y_inv = scaler.inverse_transform_target(y_scaled)
    
    # Should recover original target data
    np.testing.assert_allclose(y_inv, target, rtol=1e-4)


def test_model_inference_basic(synthetic_mmm_data):
    """Test basic model inference functionality."""
    media_data, control_data, target = synthetic_mmm_data
    
    # Create a simple trained model
    config = get_default_config()
    config['n_epochs'] = 10  # Very fast training for testing
    config['hidden_dim'] = 16
    
    model = DeepCausalMMM(
        n_media=media_data.shape[2],
        ctrl_dim=control_data.shape[1],
        n_regions=media_data.shape[0]
    )
    
    # Convert to tensors
    X_media = torch.FloatTensor(media_data)
    X_control = torch.FloatTensor(control_data.transpose(0, 2, 1))
    R = torch.arange(media_data.shape[0]).long().unsqueeze(1).repeat(1, media_data.shape[1])
    
    # Test inference
    model.eval()
    with torch.no_grad():
        y_pred, media_contrib, control_contrib, outputs = model(X_media, X_control, R)
    
    # Check output shapes
    assert y_pred.shape == target.shape
    assert media_contrib.shape == X_media.shape
    # Control contribution may have different shape due to model architecture
    assert control_contrib.shape[0] == X_control.shape[0]  # Same batch size
    assert control_contrib.shape[1] == X_control.shape[1]  # Same time steps
    
    # Check that outputs are reasonable
    assert torch.all(torch.isfinite(y_pred)), "Predictions should be finite"
    assert torch.all(torch.isfinite(media_contrib)), "Media contributions should be finite"
    assert torch.all(torch.isfinite(control_contrib)), "Control contributions should be finite"

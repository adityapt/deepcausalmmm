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
        n_media_channels=media_data.shape[2],
        n_control_vars=control_data.shape[1],
        n_regions=media_data.shape[0]
    )
    
    # Convert to tensors
    X_media = torch.FloatTensor(media_data)
    X_control = torch.FloatTensor(control_data.transpose(0, 2, 1))  # Transpose to match expected shape
    R = torch.arange(media_data.shape[0]).float().unsqueeze(1).repeat(1, media_data.shape[1])
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
    pipeline = UnifiedDataPipeline()
    
    # Prepare data
    processed_data = pipeline.prepare_data(
        X_media=media_data,
        X_control=control_data.transpose(0, 2, 1),  # Transpose to match expected shape
        y=target,
        train_ratio=0.8
    )
    
    # Check that pipeline returns expected keys
    expected_keys = ['train_tensors', 'holdout_tensors', 'scaler']
    for key in expected_keys:
        assert key in processed_data
    
    # Check tensor shapes
    train_tensors = processed_data['train_tensors']
    assert 'X_media' in train_tensors
    assert 'X_control' in train_tensors
    assert 'y' in train_tensors
    
    # Check that data was split correctly
    original_weeks = media_data.shape[1]
    train_weeks = train_tensors['X_media'].shape[1]
    assert train_weeks < original_weeks, "Training data should be subset of original"


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
    media_sums = np.sum(X_media_scaled, axis=2)
    np.testing.assert_allclose(media_sums, 1.0, rtol=1e-5)
    
    # Target should be log-transformed (positive values)
    assert np.all(y_scaled >= 0), "Log-transformed target should be non-negative"
    
    # Test inverse transform
    X_media_inv, X_control_inv, y_inv = scaler.inverse_transform(
        X_media_scaled, X_control_scaled, y_scaled
    )
    
    # Should recover original data
    np.testing.assert_allclose(X_media_inv, media_data, rtol=1e-4)
    np.testing.assert_allclose(y_inv, target, rtol=1e-4)


def test_model_inference_basic(synthetic_mmm_data):
    """Test basic model inference functionality."""
    media_data, control_data, target = synthetic_mmm_data
    
    # Create a simple trained model
    config = get_default_config()
    config['n_epochs'] = 10  # Very fast training for testing
    config['hidden_dim'] = 16
    
    model = DeepCausalMMM(
        n_media_channels=media_data.shape[2],
        n_control_vars=control_data.shape[1],
        n_regions=media_data.shape[0],
        config=config
    )
    
    # Convert to tensors
    X_media = torch.FloatTensor(media_data)
    X_control = torch.FloatTensor(control_data.transpose(0, 2, 1))
    R = torch.arange(media_data.shape[0]).float().unsqueeze(1).repeat(1, media_data.shape[1])
    
    # Test inference
    model.eval()
    with torch.no_grad():
        y_pred, media_contrib, control_contrib, outputs = model(X_media, X_control, R)
    
    # Check output shapes
    assert y_pred.shape == target.shape
    assert media_contrib.shape == X_media.shape
    assert control_contrib.shape == X_control.shape
    
    # Check that outputs are reasonable
    assert torch.all(torch.isfinite(y_pred)), "Predictions should be finite"
    assert torch.all(torch.isfinite(media_contrib)), "Media contributions should be finite"
    assert torch.all(torch.isfinite(control_contrib)), "Control contributions should be finite"

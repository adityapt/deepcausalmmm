"""Test scaling functionality."""

import pytest
import numpy as np
import torch
from deepcausalmmm.core.scaling import SimpleGlobalScaler


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    n_regions = 2
    n_timesteps = 50
    n_channels = 3
    n_controls = 2
    
    # Generate realistic-looking data
    X_media = np.random.exponential(100, (n_regions, n_timesteps, n_channels))
    X_control = np.random.normal(0, 1, (n_regions, n_timesteps, n_controls))
    y = np.random.exponential(1000, (n_regions, n_timesteps))
    
    return X_media, X_control, y


def test_scaler_initialization():
    """Test that scaler initializes correctly."""
    scaler = SimpleGlobalScaler()
    
    assert not scaler.fitted
    assert scaler.params is None


def test_scaler_fit_transform(sample_data):
    """Test scaler fit and transform functionality."""
    X_media, X_control, y = sample_data
    scaler = SimpleGlobalScaler()
    
    # Fit the scaler
    scaler.fit(X_media, X_control, y)
    
    assert scaler.fitted
    assert scaler.params is not None
    
    # Transform the data
    X_media_scaled, X_control_scaled, y_scaled = scaler.transform(X_media, X_control, y)
    
    # Check output shapes
    assert X_media_scaled.shape == X_media.shape
    assert X_control_scaled.shape == X_control.shape
    assert y_scaled.shape == y.shape
    
    # Check that scaling actually changed the data
    assert not np.allclose(X_media_scaled, X_media)
    assert not np.allclose(X_control_scaled, X_control)
    assert not np.allclose(y_scaled, y)


def test_scaler_inverse_transform(sample_data):
    """Test scaler inverse transform functionality."""
    X_media, X_control, y = sample_data
    scaler = SimpleGlobalScaler()
    
    # Fit and transform
    scaler.fit(X_media, X_control, y)
    X_media_scaled, X_control_scaled, y_scaled = scaler.transform(X_media, X_control, y)
    
    # Test inverse transform for target (the main method available)
    y_inv = scaler.inverse_transform_target(y_scaled)
    
    # Check that inverse transform recovers original target data (within tolerance)
    np.testing.assert_allclose(y_inv, y, rtol=1e-4)


def test_scaler_fit_transform_shortcut(sample_data):
    """Test scaler fit_transform method."""
    X_media, X_control, y = sample_data
    scaler = SimpleGlobalScaler()
    
    # Use fit_transform
    X_media_scaled, X_control_scaled, y_scaled = scaler.fit_transform(X_media, X_control, y)
    
    assert scaler.fitted
    assert X_media_scaled.shape == X_media.shape
    assert X_control_scaled.shape == X_control.shape
    assert y_scaled.shape == y.shape


def test_scaler_error_before_fit():
    """Test that scaler raises error when used before fitting."""
    scaler = SimpleGlobalScaler()
    
    # Should raise error when trying to transform before fitting
    with pytest.raises(ValueError, match="Scaler must be fitted before transform"):
        scaler.transform(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]))

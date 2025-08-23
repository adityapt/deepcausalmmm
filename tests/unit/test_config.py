"""Test configuration functionality."""

import pytest
from deepcausalmmm.core.config import get_default_config, update_config


def test_get_default_config():
    """Test that default config is returned with expected keys."""
    config = get_default_config()
    
    # Check essential keys exist
    assert 'n_epochs' in config
    assert 'learning_rate' in config
    assert 'hidden_dim' in config
    assert 'batch_size' in config
    
    # Check reasonable default values
    assert config['n_epochs'] > 0
    assert 0 < config['learning_rate'] < 1
    assert config['hidden_dim'] > 0


def test_update_config():
    """Test config update functionality."""
    base_config = get_default_config()
    original_lr = base_config['learning_rate']
    
    # Update with new values
    updates = {'learning_rate': 0.001, 'n_epochs': 1000}
    updated_config = update_config(base_config, updates)
    
    # Check updates were applied
    assert updated_config['learning_rate'] == 0.001
    assert updated_config['n_epochs'] == 1000
    
    # Check other values preserved
    assert updated_config['hidden_dim'] == base_config['hidden_dim']
    
    # Check original config unchanged
    assert base_config['learning_rate'] == original_lr


def test_config_types():
    """Test that config values have correct types."""
    config = get_default_config()
    
    assert isinstance(config['n_epochs'], int)
    assert isinstance(config['learning_rate'], float)
    assert isinstance(config['hidden_dim'], int)
    assert isinstance(config['use_huber_loss'], bool)

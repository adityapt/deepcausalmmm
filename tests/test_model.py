"""
Tests for model components.
"""

import pytest
import torch
import numpy as np
from deepcausalmmm.core.model import CausalEncoder, GRUCausalMMM


class TestCausalEncoder:
    """Test CausalEncoder class."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        A_prior = torch.randn(5, 5)
        encoder = CausalEncoder(A_prior, d_in=5, d_hid=10)
        
        assert encoder.A.shape == (5, 5)
        assert encoder.edge is not None
        assert encoder.node is not None
    
    def test_forward_pass(self):
        """Test forward pass through encoder."""
        A_prior = torch.randn(3, 3)
        encoder = CausalEncoder(A_prior, d_in=3, d_hid=5)
        
        X = torch.randn(2, 3)  # batch_size=2, features=3
        output = encoder(X)
        
        assert output.shape == (2, 5)  # batch_size=2, hidden=5
        assert not torch.isnan(output).any()


class TestGRUCausalMMM:
    """Test GRUCausalMMM class."""
    
    def test_initialization(self):
        """Test model initialization."""
        A_prior = torch.randn(4, 4)
        model = GRUCausalMMM(
            A_prior, 
            n_media=4, 
            ctrl_dim=3, 
            hidden=32, 
            n_regions=2
        )
        
        assert model.n_media == 4
        assert model.ctrl_dim == 3
        assert model.hidden_size == 32
        assert model.n_regions == 2
    
    def test_adstock_transformation(self):
        """Test adstock transformation."""
        A_prior = torch.randn(3, 3)
        model = GRUCausalMMM(A_prior, n_media=3, ctrl_dim=2, hidden=16, n_regions=1)
        
        x = torch.randn(2, 5, 3)  # batch=2, time=5, media=3
        adstocked = model.adstock(x)
        
        assert adstocked.shape == x.shape
        assert not torch.isnan(adstocked).any()
    
    def test_hill_transformation(self):
        """Test Hill saturation transformation."""
        A_prior = torch.randn(3, 3)
        model = GRUCausalMMM(A_prior, n_media=3, ctrl_dim=2, hidden=16, n_regions=1)
        
        x = torch.randn(2, 5, 3)
        saturated = model.hill(x)
        
        assert saturated.shape == x.shape
        assert not torch.isnan(saturated).any()
        assert (saturated >= 0).all()  # Should be non-negative
    
    def test_forward_pass(self):
        """Test complete forward pass."""
        A_prior = torch.randn(3, 3)
        model = GRUCausalMMM(A_prior, n_media=3, ctrl_dim=2, hidden=16, n_regions=2)
        
        X_m = torch.randn(2, 5, 3)  # batch=2, time=5, media=3
        X_c = torch.randn(2, 5, 2)  # batch=2, time=5, ctrl=2
        R = torch.tensor([0, 1])    # 2 regions
        
        predictions, coefficients, contributions = model(X_m, X_c, R)
        
        assert predictions.shape == (2, 5)  # batch=2, time=5
        assert coefficients.shape == (2, 5, 3)  # batch=2, time=5, media=3
        assert contributions.shape == (2, 5, 3)  # batch=2, time=5, media=3
        
        assert not torch.isnan(predictions).any()
        assert not torch.isnan(coefficients).any()
        assert not torch.isnan(contributions).any()
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        A_prior = torch.randn(3, 3)
        model = GRUCausalMMM(A_prior, n_media=3, ctrl_dim=2, hidden=16, n_regions=2)
        
        X_m = torch.randn(2, 5, 3)
        X_c = torch.randn(2, 5, 2)
        R = torch.tensor([0, 1])
        
        importance = model.get_feature_importance(X_m, X_c, R)
        
        assert importance.shape == (3,)  # 3 media channels
        assert not torch.isnan(importance).any()
        assert (importance >= 0).all()  # Should be non-negative


if __name__ == "__main__":
    pytest.main([__file__]) 
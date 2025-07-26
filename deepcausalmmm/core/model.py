"""
Model definitions for DeepCausalMMM.

This module contains the core neural network architectures:
- CausalEncoder: Bayesian Network-based graph encoder
- GRUCausalMMM: Main model combining GRU with causal structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

from ..exceptions import ModelError


def gs(logits: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """
    Gumbel-Softmax sampling for discrete distributions.
    
    Args:
        logits: Input logits
        tau: Temperature parameter
        
    Returns:
        Sampled probabilities
    """
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    return torch.sigmoid((logits + g) / tau)


class CausalEncoder(nn.Module):
    """
    Bayesian Network-based causal encoder for media variables.
    
    This module uses a fixed DAG structure to encode causal relationships
    between media variables and produce belief vectors.
    """
    
    def __init__(self, A_prior: torch.Tensor, d_in: int = 10, d_hid: int = 10):
        """
        Initialize the causal encoder.
        
        Args:
            A_prior: Prior adjacency matrix for the DAG
            d_in: Input dimension (number of media variables)
            d_hid: Hidden dimension for the encoder
        """
        super().__init__()
        self.register_buffer("A", torch.tensor(A_prior, dtype=torch.float32))
        
        # Edge encoder: processes sender-receiver pairs
        self.edge = nn.Sequential(
            nn.Linear(d_in * 2, d_hid), 
            nn.ReLU(),
            nn.Linear(d_hid, d_hid), 
            nn.ReLU()
        )
        
        # Node encoder: combines node features with edge features
        self.node = nn.Sequential(
            nn.Linear(d_in + d_hid, d_hid), 
            nn.ReLU()
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the causal encoder.
        
        Args:
            X: Input tensor [batch_size, d_in]
            
        Returns:
            Encoded features [batch_size, d_hid]
        """
        # Compute sender and receiver features
        send = X @ self.A.t()  # [batch_size, d_in]
        recv = X @ self.A      # [batch_size, d_in]
        
        # Encode edge features
        He = self.edge(torch.cat([send, recv], dim=-1))  # [batch_size, d_in]
        
        # Encode node features with edge context
        Z = self.node(torch.cat([X, He], dim=-1))  # [batch_size, d_hid]
        
        return Z


class GRUCausalMMM(nn.Module):
    """
    Main model combining GRU with Bayesian Network causal structure.
    
    This model implements:
    - Adstock transformation for media carryover effects
    - Hill saturation for diminishing returns
    - GRU for time-varying coefficients
    - Bayesian Network for causal structure
    """
    
    def __init__(
        self, 
        A_prior: torch.Tensor, 
        n_media: int = 10, 
        ctrl_dim: int = 15, 
        hidden: int = 64, 
        n_regions: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize the GRU Causal MMM model.
        
        Args:
            A_prior: Prior adjacency matrix for media variables
            n_media: Number of media variables
            ctrl_dim: Dimension of control variables
            hidden: Hidden dimension for GRU
            n_regions: Number of regions
            dropout: Dropout rate
        """
        super().__init__()
        
        # Causal encoder for media variables
        self.enc = CausalEncoder(A_prior, d_in=n_media, d_hid=n_media)
        
        # GRU for time-varying coefficients
        self.gru = nn.GRU(
            input_size=n_media + ctrl_dim, 
            hidden_size=hidden, 
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Coefficient generator
        self.w_raw = nn.Linear(hidden, n_media)
        nn.init.xavier_uniform_(self.w_raw.weight, gain=0.10)
        nn.init.zeros_(self.w_raw.bias)
        
        # Adstock parameters (learnable decay rates)
        self.alpha = nn.Parameter(torch.full((n_media,), 0.5))
        
        # Hill saturation parameters
        torch.manual_seed(42)  # For reproducible initialization
        self.hill_a = nn.Parameter(torch.rand(n_media) * 0.8 + 0.6)  # Shape: 0.6-1.4
        self.hill_g = nn.Parameter(torch.rand(n_media) * 0.3 + 0.1)  # Half-saturation: 0.1-0.4
        
        # Control and region embeddings
        self.ctrl_mlp = nn.Linear(ctrl_dim, hidden)
        self.reg_emb = nn.Embedding(n_regions, hidden)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Learnable initial hidden state
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden))
        
        # Store dimensions
        self.hidden_size = hidden
        self.n_media = n_media
        self.ctrl_dim = ctrl_dim
        self.n_regions = n_regions
    
    def adstock(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adstock transformation to media variables.
        
        Args:
            x: Input tensor [batch_size, time_steps, n_media]
            
        Returns:
            Adstocked tensor [batch_size, time_steps, n_media]
        """
        B, T, C = x.shape
        alpha = torch.clamp(self.alpha, 0, 1).view(1, 1, -1)
        
        # Use list to collect results and avoid in-place operations
        out_list = [x[:, 0:1]]  # Start with first timestep
        
        for t in range(1, T):
            prev_adstock = out_list[-1]
            current = x[:, t:t+1] + alpha * prev_adstock
            out_list.append(current)
        
        # Concatenate all timesteps
        out = torch.cat(out_list, dim=1)
        return out
    
    def hill(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Hill saturation transformation.
        
        Args:
            x: Input tensor [batch_size, time_steps, n_media]
            
        Returns:
            Saturated tensor [batch_size, time_steps, n_media]
        """
        a = torch.clamp(self.hill_a, 0.1, 3.0).view(1, 1, -1)
        g = torch.clamp(self.hill_g, 0.1, 2.0).view(1, 1, -1)
        
        num = x.clamp(min=0).pow(a)
        return num / (num + g.pow(a) + 1e-8)
    
    def forward(
        self, 
        Xm: torch.Tensor, 
        Xc: torch.Tensor, 
        R: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            Xm: Media variables [batch_size, time_steps, n_media]
            Xc: Control variables [batch_size, time_steps, ctrl_dim]
            R: Region indices [batch_size]
            
        Returns:
            Tuple of (predictions, coefficients, contributions)
        """
        B, T, _ = Xm.shape
        
        # Causal node embeddings for each week
        Z_seq = torch.stack([self.enc(x) for x in Xm.unbind(1)], 1)  # [B, T, n_media]
        
        # Apply transformations: Raw → Adstock → Hill
        adstock_out = self.adstock(Xm)  # [B, T, n_media]
        
        # Channel-wise 0-1 scaling per region
        channel_max = adstock_out.amax(dim=1, keepdim=True).clamp_min(1e-6)
        adstock_norm = adstock_out / channel_max
        hill_out = self.hill(adstock_norm)  # [B, T, n_media]
        
        # Combine with causal embeddings
        media_in = hill_out + Z_seq  # [B, T, n_media]
        
        # Prepare GRU input
        gru_in = torch.cat([media_in, Xc], dim=-1)  # [B, T, n_media + ctrl_dim]
        
        # Initialize GRU with learnable warm-start
        h0 = self.h0.repeat(1, B, 1)  # [1, B, hidden]
        h_seq, _ = self.gru(gru_in, h0)  # [B, T, hidden]
        
        # Generate time-varying coefficients
        w_raw = self.w_raw(h_seq)  # [B, T, n_media]
        w_pos = F.softplus(w_raw)  # Always positive
        
        # Smooth first week to avoid spikes
        if T > 1:
            w_pos[:, 0, :] = 0.5 * w_pos[:, 1, :].detach() + 0.5 * w_pos[:, 0, :]
        
        # Calculate contributions
        media_contrib_scaled = hill_out * w_pos  # [B, T, n_media]
        media_term = media_contrib_scaled.sum(-1)  # [B, T]
        
        # Control and region terms
        ctrl_term = torch.relu(self.ctrl_mlp(Xc)).sum(-1) * 0.3
        reg_term = self.reg_emb(torch.zeros(B, dtype=torch.long)).sum(-1).unsqueeze(1).expand(-1, T) * 0.3
        
        # Final prediction
        y_scaled = media_term + ctrl_term + reg_term + self.bias
        
        return y_scaled, w_pos, media_contrib_scaled
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get model parameters for analysis.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'adstock_alpha': self.alpha.detach(),
            'hill_a': self.hill_a.detach(),
            'hill_g': self.hill_g.detach(),
            'bias': self.bias.detach(),
        }
    
    def get_feature_importance(self, Xm: torch.Tensor, Xc: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Calculate feature importance based on learned coefficients.
        
        Args:
            Xm: Media variables
            Xc: Control variables  
            R: Region indices
            
        Returns:
            Feature importance scores
        """
        with torch.no_grad():
            _, w_pos, _ = self.forward(Xm, Xc, R)
            # Average importance across time and regions
            importance = w_pos.mean(dim=(0, 1))  # [n_media]
            return importance 
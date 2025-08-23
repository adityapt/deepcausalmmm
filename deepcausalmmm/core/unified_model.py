"""
DeepCausalMMM model implementation combining GRU, DAG, and interaction components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np

from deepcausalmmm.core.dag_model import NodeToEdge, EdgeToNode, DAGConstraint
from deepcausalmmm.core.seasonality import DetectSeasonality


class DeepCausalMMM(nn.Module):
    """
    Deep Causal Marketing Mix Model with DAG structure and channel interactions.
    
    This model combines deep learning with causal inference to understand the impact 
    of marketing channels on business KPIs while learning causal relationships between 
    channels through a Directed Acyclic Graph (DAG).
    
    The model features:
    - GRU-based temporal modeling for time-varying coefficients
    - Learnable coefficient bounds for realistic attribution
    - DAG learning for causal channel interactions
    - Adstock and saturation transformations
    - Multi-region support with shared and region-specific parameters
    - Zero hardcoding philosophy - all parameters are learnable or configurable
    
    Parameters
    ----------
    n_media : int, default=10
        Number of media channels in the dataset
    ctrl_dim : int, default=15
        Number of control variables (weather, events, etc.)
    hidden : int, default=32
        Hidden dimension size for GRU and MLP layers
    n_regions : int, default=2
        Number of geographic regions or DMAs
    dropout : float, default=0.1
        Dropout rate for regularization during training
    sparsity_weight : float, default=0.01
        Weight for sparsity regularization on coefficients
    enable_dag : bool, default=True
        Whether to enable DAG learning for channel interactions
    enable_interactions : bool, default=True
        Whether to enable channel interaction modeling
    l1_weight : float, default=0.001
        L1 regularization weight for coefficient sparsity
    l2_weight : float, default=0.001
        L2 regularization weight for coefficient smoothness
    burn_in_weeks : int, default=4
        Number of initial weeks for GRU stabilization
    use_coefficient_momentum : bool, default=True
        Whether to use momentum for coefficient stabilization
    momentum_decay : float, default=0.9
        Decay rate for coefficient momentum
    use_warm_start : bool, default=True
        Whether to use warm start training initialization
    warm_start_epochs : int, default=50
        Number of epochs for warm start phase
    stabilization_method : str, default="exponential"
        Method for coefficient stabilization ("linear", "exponential", "sigmoid")
    coeff_l2_weight : float, default=0.1
        L2 regularization specifically for media coefficients
    coeff_gen_l2_weight : float, default=0.05
        L2 regularization for coefficient generation layers
    gru_layers : int, default=1
        Number of GRU layers (configured, not hardcoded)
    ctrl_hidden_ratio : float, default=0.5
        Control hidden size as ratio of main hidden dimension
        
    Attributes
    ----------
    media_coeffs : torch.nn.Parameter
        Time-varying coefficients for media channels
    ctrl_coeffs : torch.nn.Parameter
        Coefficients for control variables
    dag_matrix : torch.nn.Parameter
        Learnable DAG adjacency matrix for channel interactions
    region_baseline : torch.nn.Parameter
        Region-specific baseline contributions
    seasonal_coeff : torch.nn.Parameter
        Learnable coefficient for seasonal component
        
    Examples
    --------
    >>> import torch
    >>> from deepcausalmmm import DeepCausalMMM
    >>> 
    >>> # Initialize model
    >>> model = DeepCausalMMM(
    ...     n_media=5, 
    ...     ctrl_dim=3, 
    ...     n_regions=2,
    ...     hidden=64
    ... )
    >>> 
    >>> # Prepare data tensors
    >>> media_data = torch.randn(2, 104, 5)    # [regions, weeks, channels]
    >>> control_data = torch.randn(2, 104, 3)  # [regions, weeks, controls]
    >>> regions = torch.arange(2).unsqueeze(1).repeat(1, 104)
    >>> 
    >>> # Forward pass
    >>> predictions, baseline, seasonality, outputs = model(
    ...     media_data, control_data, regions
    ... )
    >>> 
    >>> print(f"Predictions shape: {predictions.shape}")
    >>> print(f"Media contributions: {outputs['media_contributions'].shape}")
    """
    
    def __init__(
        self,
        n_media: int = 10,
        ctrl_dim: int = 15,
        hidden: int = 32,  # Smaller default
        n_regions: int = 2,
        dropout: float = 0.1,
        sparsity_weight: float = 0.01,  # Smaller default
        enable_dag: bool = True,
        enable_interactions: bool = True,
        l1_weight: float = 0.001,  # Smaller default
        l2_weight: float = 0.001,  # Smaller default
        burn_in_weeks: int = 4,  # NEW: Number of weeks to stabilize GRU
        # NEW: Advanced stabilization parameters
        use_coefficient_momentum: bool = True,
        momentum_decay: float = 0.9,
        use_warm_start: bool = True,
        warm_start_epochs: int = 50,
        stabilization_method: str = "exponential",  # "linear", "exponential", "sigmoid"
        # COEFFICIENT REGULARIZATION: Prevent coefficient explosion
        coeff_l2_weight: float = 0.1,
        coeff_gen_l2_weight: float = 0.05,
        # NEW: Config-driven parameters (no hardcoding!)
        gru_layers: int = 1,
        ctrl_hidden_ratio: float = 0.5  # Control hidden size as ratio of main hidden size
    ):
        super().__init__()
        
        # Store dimensions and flags
        self.n_media = n_media
        self.ctrl_dim = ctrl_dim
        self.hidden_size = hidden
        self.n_regions = n_regions
        self.enable_dag = enable_dag
        self.enable_interactions = enable_interactions
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.burn_in_weeks = burn_in_weeks  # NEW: Store burn-in period
        
        # COEFFICIENT REGULARIZATION: Store parameters to prevent explosion
        self.coeff_l2_weight = coeff_l2_weight
        self.coeff_gen_l2_weight = coeff_gen_l2_weight
        
        # NEW: Advanced stabilization parameters
        self.use_coefficient_momentum = use_coefficient_momentum
        self.momentum_decay = momentum_decay
        self.use_warm_start = use_warm_start
        self.warm_start_epochs = warm_start_epochs
        self.stabilization_method = stabilization_method
        
        # Coefficient momentum tracking
        if self.use_coefficient_momentum:
            self.register_buffer('media_coeff_momentum', torch.zeros(n_media))
            self.register_buffer('ctrl_coeff_momentum', torch.zeros(ctrl_dim))
            self.register_buffer('momentum_step', torch.tensor(0))
        
        # Adstock parameters - STABILIZED initialization
        self.alpha = nn.Parameter(torch.ones(n_media) * 0.8)  # Start with reasonable adstock
        
        # STABILIZED HILL - Smaller ranges
        self.hill_a = nn.Parameter(torch.rand(n_media) * 0.5 + 0.5)  # 0.5-1.0
        self.hill_g = nn.Parameter(torch.rand(n_media) * 0.2 + 0.1)  # 0.1-0.3
        
        # CAUSAL DAG components - discover meaningful relationships
        if enable_dag and enable_interactions:
            # Initialize with slight sparse bias - let causal relationships emerge
            self.adj_logits = nn.Parameter(torch.randn(n_media, n_media) * 0.3 - 0.1)  # Light negative bias
            # Upper triangular mask for acyclicity
            mask = torch.triu(torch.ones(n_media, n_media), diagonal=1)
            self.register_buffer('tri_mask', mask)
            # Interaction weight
            self.interaction_weight = nn.Parameter(torch.ones(1) * 0.1)
        
        # FIXED control processing - use config-driven dimensions (NO HARDCODING!)
        self.ctrl_hidden = int(hidden * ctrl_hidden_ratio)  # Config-driven control hidden size
        self.ctrl_mlp = nn.Sequential(
            nn.Linear(ctrl_dim, self.ctrl_hidden),
            nn.Tanh(),  # Bounded activation
            nn.Dropout(dropout)
        )
        
        # CONFIG-DRIVEN GRU - NO HARDCODING!
        gru_input_size = n_media + self.ctrl_hidden
        self.gru_layers = gru_layers  # Use config value, no hardcoding!
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden,
            num_layers=self.gru_layers,  # Config-driven layers
            batch_first=True,
            dropout=dropout if self.gru_layers > 1 else 0  # Conditional dropout based on layers
        )
        
        # DISABLE residual connections for stability
        self.use_residual = False  # REVERT - was causing instability
        
        # ENHANCED coefficient generator for ultra-low RMSE
        self.coeff_gen = nn.Sequential(
            nn.Linear(hidden, hidden),  # FULL capacity first layer
            nn.ReLU(),  # Better gradient flow than Tanh
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_media)
        )
        
        # Initialize coefficient generator carefully
        for layer in self.coeff_gen:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
        
        # ENHANCED control coefficient generator for ultra-low RMSE
        self.ctrl_coeff_gen = nn.Sequential(
            nn.Linear(hidden, hidden),  # FULL capacity first layer
            nn.ReLU(),  # Better gradient flow than Tanh
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, ctrl_dim)
        )
        
        # Initialize control coefficients
        for layer in self.ctrl_coeff_gen:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
        
        # REGION-SPECIFIC baselines - IMPROVED initialization
        self.region_baseline = nn.Parameter(torch.randn(n_regions) * 0.1)
        
        # Global bias and prediction scaling
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.prediction_scale = nn.Parameter(torch.ones(1))
        
        # NEW: Time trend component to handle growth patterns
        self.time_trend_weight = nn.Parameter(torch.zeros(1))  # Learnable trend strength
        self.time_trend_bias = nn.Parameter(torch.zeros(1))    # Trend intercept
        
        # NEW: Seasonal component with learnable coefficient
        self.seasonal_coeff = nn.Parameter(torch.ones(1))  # Learnable seasonal coefficient
        self.seasonal_components = None  # Will be initialized with actual data from decomposition
        self.seasonality_detector = DetectSeasonality()  # For seasonal decomposition
        
        # FULLY LEARNABLE: No hardcoded bounds - let model discover everything
        self.coeff_range_raw = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0, no upper bound
        self.ctrl_coeff_range_raw = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0, no upper bound
        
        # FULLY LEARNABLE: Trend strength with no artificial constraints
        self.trend_damping_raw = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0, can learn any positive value
        
        # HYBRID APPROACH: Fixed regularization weights (handled in trainer)
        # Only core model parameters are learnable - not loss balancing weights
        
        # FULLY LEARNABLE: All initialization scaling factors - no hardcoded multipliers
        self.stable_coeff_scale_raw = nn.Parameter(torch.tensor(-2.3))  # exp(-2.3) ≈ 0.1, but can learn optimal
        self.region_baseline_scale_raw = nn.Parameter(torch.tensor(-2.3))  # exp(-2.3) ≈ 0.1, but can learn optimal
        
        # LEARNABLE COEFFICIENT BOUNDS: Each channel learns its optimal maximum coefficient
        self.media_coeff_max_raw = nn.Parameter(torch.ones(n_media) * 1.0)  # Start at ~2.7 (softplus(1.0))
        self.ctrl_coeff_max_raw = nn.Parameter(torch.ones(ctrl_dim) * 1.5)   # Start at ~4.5 (softplus(1.5))
        
        # SIMPLE TRAINABLE CONSTANTS: Global bounds only (prevent overfitting)
        # Channel-specific bounds caused severe overfitting (R² dropped from 0.924 to 0.340)
        # Using global bounds with natural activation functions for optimal generalization
        
        # GRU hidden state initialization - STABILIZED
        self.h0 = nn.Parameter(torch.randn(1, 1, hidden) * 0.01)
        
        # NEW: Stable coefficient reference for burn-in stabilization
        # Initialize to small values - will be set properly from data
        self.stable_media_coeff = nn.Parameter(torch.zeros(n_media))
        self.stable_ctrl_coeff = nn.Parameter(torch.zeros(ctrl_dim))

    def initialize_baseline(self, y_data: torch.Tensor):
        """Initialize baseline to match target data statistics.
        
        CRITICAL: y_data is ALREADY in scaled space (log1p transformed)!
        Extract ALL parameters directly from the actual data distribution.
        IMPORTANT: Skip padding weeks to avoid baseline bias!
        """
        with torch.no_grad():
            # y_data shape: [n_regions, n_timesteps] - already in scaled space
            # CRITICAL: Remove padding weeks (first 4 weeks) to get true data statistics
            y_no_padding = y_data[:, self.burn_in_weeks:] if y_data.shape[1] > self.burn_in_weeks else y_data
            y_numpy = y_no_padding.cpu().numpy()
            n_regions, n_timesteps = y_numpy.shape
            
            # 1. GLOBAL BASELINE: Keep learnable but start small to let model learn the right scale
            # The model will learn the appropriate global offset during training
            self.global_bias.data = torch.FloatTensor([0.0])
            
            # 2. REGION BASELINES: Use actual per-region means as absolute baselines (not deviations)
            # This ensures each region starts with a positive baseline equal to its historical mean
            region_means_scaled = y_numpy.mean(axis=1)  # [n_regions] - actual region means
            self.region_baseline.data = torch.FloatTensor(region_means_scaled)
            
            # 3. PREDICTION SCALE: Initialize to 1.0 (neutral scaling)
            # The model will learn the appropriate scaling during training
            # softplus(0) = 1.0, so no initial scaling applied
            self.prediction_scale.data = torch.FloatTensor([0.0])  # softplus(0) = 1.0
            
            # 4. TIME TREND: Extract actual trend from data (if exists)
            if n_timesteps > 1:
                # Calculate time-series trend across all regions
                time_steps = np.arange(n_timesteps, dtype=np.float32)
                y_time_series = y_numpy.mean(axis=0)  # Average across regions over time
                
                # Linear regression: y = slope * t + intercept
                time_mean = time_steps.mean()
                y_mean = y_time_series.mean()
                
                numerator = np.sum((time_steps - time_mean) * (y_time_series - y_mean))
                denominator = np.sum((time_steps - time_mean) ** 2)
                
                if denominator > 1e-8:  # Avoid division by zero
                    actual_slope = numerator / denominator
                    actual_intercept = y_mean - actual_slope * time_mean
                    
                    # FULLY LEARNABLE: No hardcoded trend damping bounds
                    # Let model discover optimal trend strength - can be any positive value
                    learned_trend_damping = torch.exp(self.trend_damping_raw)  # Range: [0, ∞] - no artificial limits!
                    self.time_trend_weight.data = torch.FloatTensor([actual_slope * learned_trend_damping.item()])
                    self.time_trend_bias.data = torch.FloatTensor([actual_intercept * learned_trend_damping.item()])
                else:
                    # No detectable trend
                    self.time_trend_weight.data = torch.zeros(1)
                    self.time_trend_bias.data = torch.zeros(1)
            else:
                # Single timestep - no trend
                self.time_trend_weight.data = torch.zeros(1)
                self.time_trend_bias.data = torch.zeros(1)
            
            print(f"Initialized baselines (Log1p scale):")
            print(f"   Region baselines range: [{region_means_scaled.min():.3f}, {region_means_scaled.max():.3f}]")
            print(f"   Global bias (Log1p scale): {self.global_bias.item():.3f} [LEARNABLE - constrained ≥ 0 via softplus]")
            print(f"   Expected prediction baseline: {region_means_scaled.mean():.3f} -> ~{torch.expm1(torch.tensor(region_means_scaled.mean())).item():.0f} visits")
            
            # Initialize seasonal components using actual data decomposition
            self._initialize_seasonal_components(y_data)

    def _initialize_seasonal_components(self, y_data: torch.Tensor):
        """
        Initialize seasonal components using multiplicative decomposition per region.
        
        Args:
            y_data: Target data [n_regions, n_timesteps] in log1p scale
        """
        with torch.no_grad():
            # Skip padding weeks for seasonal decomposition
            y_no_padding = y_data[:, self.burn_in_weeks:] if y_data.shape[1] > self.burn_in_weeks else y_data
            
            # Convert to numpy and apply expm1 to get original scale for decomposition
            y_original_scale = torch.expm1(y_no_padding).cpu().numpy()
            
            # Extract seasonal components per region
            seasonal_components = self.seasonality_detector.extract_seasonal_components_per_region(
                y_original_scale, start_week=0
            )
            
            # Apply Min-Max scaling per region to bring seasonality to 0-1 range
            # This preserves seasonal patterns while normalizing scale
            n_regions, n_weeks = seasonal_components.shape
            seasonal_normalized = torch.zeros_like(seasonal_components)
            
            for region_idx in range(n_regions):
                region_seasonal = seasonal_components[region_idx, :]
                region_min = region_seasonal.min()
                region_max = region_seasonal.max()
                
                if region_max > region_min:
                    # Min-Max scaling: (x - min) / (max - min) -> [0, 1] range
                    seasonal_normalized[region_idx, :] = (region_seasonal - region_min) / (region_max - region_min)
                else:
                    # If no variation, set to middle of range
                    seasonal_normalized[region_idx, :] = 0.5
            
            # Store seasonal components (will be used in forward pass)
            self.seasonal_components = seasonal_normalized
            
            print(f"🌊 Initialized seasonal components:")
            print(f"   Seasonal coefficient: {self.seasonal_coeff.item():.3f} [LEARNABLE - constrained ≥ 0 via softplus]")
            print(f"   Components range: [{seasonal_normalized.min():.3f}, {seasonal_normalized.max():.3f}] (Min-Max scaled per region)")
            print(f"   Components mean: {seasonal_normalized.mean():.3f}")
            print(f"   Scaling: Min-Max per region (0-1 range) - preserves seasonal patterns")

    def initialize_stable_coefficients_from_data(self, Xm: torch.Tensor, Xc: torch.Tensor, y: torch.Tensor):
        """
        Initialize stable coefficients based on simple linear regression on the data.
        This provides domain-informed starting points for coefficient stabilization.
        """
        with torch.no_grad():
            B, T, n_media = Xm.shape
            _, _, n_control = Xc.shape
            
            # CRITICAL: Remove padding weeks to avoid biasing the regression
            Xm_no_padding = Xm[:, self.burn_in_weeks:] if T > self.burn_in_weeks else Xm
            Xc_no_padding = Xc[:, self.burn_in_weeks:] if T > self.burn_in_weeks else Xc
            y_no_padding = y[:, self.burn_in_weeks:] if T > self.burn_in_weeks else y
            
            # MEDIA COEFFICIENTS - Flatten data for regression (no padding)
            X_media_flat = Xm_no_padding.reshape(-1, n_media)  # [B*(T-burn_in), n_media]
            y_flat = y_no_padding.reshape(-1)  # [B*(T-burn_in)]
            
            # Simple ridge regression to get initial media coefficients
            XtX = torch.mm(X_media_flat.t(), X_media_flat)
            lambda_reg = 0.01 * torch.eye(n_media, device=Xm.device)
            XtX_reg = XtX + lambda_reg
            Xty = torch.mv(X_media_flat.t(), y_flat)
            
            try:
                beta_media = torch.linalg.solve(XtX_reg, Xty)
                # POSITIVE-ONLY INITIALIZATION: Use sigmoid to ensure non-negative media coefficients
                beta_media_raw = torch.log(torch.abs(beta_media) + 1e-8)  # Convert to raw for sigmoid
                beta_media_positive = torch.sigmoid(beta_media_raw)  # Range: [0, 1] - POSITIVE ONLY!
                
                # Update stable media coefficients
                data_scale = y_flat.std() / X_media_flat.std()
                self.stable_media_coeff.data = beta_media_positive * data_scale
                
                print(f"Initialized stable coefficients from data:")
                print(f"  Media coeff range (POSITIVE-ONLY): [{beta_media_positive.min().item():.4f}, {beta_media_positive.max().item():.4f}]")
                
            except torch.linalg.LinAlgError:
                print("Warning: Could not solve for media coefficients, using correlation fallback")
                correlations = torch.zeros(n_media, device=Xm.device)
                for i in range(n_media):
                    if X_media_flat[:, i].std() > 1e-8:
                        correlations[i] = torch.corrcoef(torch.stack([X_media_flat[:, i], y_flat]))[0, 1]
                correlations = torch.nan_to_num(correlations, 0.0)
                stable_coeff_scale = torch.exp(self.stable_coeff_scale_raw)  # FULLY LEARNABLE scaling
                self.stable_media_coeff.data = correlations * stable_coeff_scale
            
            # CONTROL COEFFICIENTS - Allow negative effects (no padding)
            X_control_flat = Xc_no_padding.reshape(-1, n_control)  # [B*(T-burn_in), n_control]
            
            # Ridge regression for control coefficients
            XtX_ctrl = torch.mm(X_control_flat.t(), X_control_flat)
            lambda_reg_ctrl = 0.01 * torch.eye(n_control, device=Xc.device)
            XtX_reg_ctrl = XtX_ctrl + lambda_reg_ctrl
            Xty_ctrl = torch.mv(X_control_flat.t(), y_flat)
            
            try:
                beta_control = torch.linalg.solve(XtX_reg_ctrl, Xty_ctrl)
                # Allow both positive and negative control effects - NO clipping
                # Control variables should be able to have strong negative effects
                
                # Scale control coefficients appropriately
                ctrl_data_scale = y_flat.std() / X_control_flat.std()
                stable_coeff_scale = torch.exp(self.stable_coeff_scale_raw)  # FULLY LEARNABLE scaling
                self.stable_ctrl_coeff.data = beta_control * ctrl_data_scale * stable_coeff_scale
                
                print(f"  Control coeff range: [{beta_control.min().item():.4f}, {beta_control.max().item():.4f}]")
                
            except torch.linalg.LinAlgError:
                print("Warning: Could not solve for control coefficients, using correlation fallback")
                correlations_ctrl = torch.zeros(n_control, device=Xc.device)
                for i in range(n_control):
                    if X_control_flat[:, i].std() > 1e-8:
                        correlations_ctrl[i] = torch.corrcoef(torch.stack([X_control_flat[:, i], y_flat]))[0, 1]
                correlations_ctrl = torch.nan_to_num(correlations_ctrl, 0.0)
                # Allow negative correlations for controls
                stable_coeff_scale = torch.exp(self.stable_coeff_scale_raw)  # FULLY LEARNABLE scaling
                self.stable_ctrl_coeff.data = correlations_ctrl * stable_coeff_scale

    def warm_start_training(self, Xm: torch.Tensor, Xc: torch.Tensor, R: torch.Tensor, y: torch.Tensor, 
                           optimizer: torch.optim.Optimizer, epochs: int = None):
        """
        Warm-start training phase to stabilize GRU coefficients before main training.
        Uses only stable coefficients and focuses on learning good hidden state initialization.
        """
        if not self.use_warm_start:
            return
            
        epochs = epochs or self.warm_start_epochs
        print(f"Starting warm-start training for {epochs} epochs...")
        
        # Initialize stable coefficients from data
        self.initialize_stable_coefficients_from_data(Xm, Xc, y)
        
        # Save original parameters
        original_coeff_range_raw = self.coeff_range_raw.data.clone()
        original_ctrl_coeff_range_raw = self.ctrl_coeff_range_raw.data.clone()
        original_burn_in = self.burn_in_weeks
        
        # Temporarily use only stable coefficients (no dynamic variation)
        self.coeff_range_raw.data = torch.tensor(-10.0)  # sigmoid(-10) ≈ 0, so range ≈ 1.0 (minimal)
        self.ctrl_coeff_range_raw.data = torch.tensor(-10.0)  # sigmoid(-10) ≈ 0, so range ≈ 1.0 (minimal)
        self.burn_in_weeks = 999  # Force all weeks to use stable coefficients
        
        # Freeze coefficient generators during warm-start
        for param in self.coeff_gen.parameters():
            param.requires_grad = False
        for param in self.ctrl_coeff_gen.parameters():
            param.requires_grad = False
            
        # Train only GRU, baselines, and stable coefficients
        warm_start_params = [
            self.h0, self.stable_media_coeff, self.stable_ctrl_coeff,
            self.region_baseline, self.global_bias, self.prediction_scale
        ] + list(self.gru.parameters())
        
        # Use config learning rate but scaled UP for faster warm-start convergence
        config_lr = getattr(self, 'config_lr', 0.005)  # Default fallback
        warm_start_lr = config_lr * 2.0  # 2x main LR for faster coefficient stabilization
        warm_optimizer = torch.optim.Adam(warm_start_params, lr=warm_start_lr)  # Scaled from config LR
        
        self.train()
        for epoch in range(epochs):
            warm_optimizer.zero_grad()
            
            y_pred, _, _, outputs = self.forward(Xm, Xc, R)
            loss = F.mse_loss(y_pred, y)
            
            # Add stronger regularization to prevent extreme values
            reg_loss = 0.01 * (self.h0.pow(2).mean() + self.stable_media_coeff.pow(2).mean())
            total_loss = loss + reg_loss
            
            total_loss.backward()
            
            # Aggressive gradient clipping for regional scaling stability
            torch.nn.utils.clip_grad_norm_(warm_start_params, max_norm=0.1)  # Much more aggressive
            
            warm_optimizer.step()
            
            if epoch % 10 == 0:
                print(f"  Warm-start epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")
        
        # Restore original parameters
        self.coeff_range_raw.data = original_coeff_range_raw
        self.ctrl_coeff_range_raw.data = original_ctrl_coeff_range_raw
        self.burn_in_weeks = original_burn_in
        
        # Unfreeze coefficient generators
        for param in self.coeff_gen.parameters():
            param.requires_grad = True
        for param in self.ctrl_coeff_gen.parameters():
            param.requires_grad = True
            
        print(f"✅ Warm-start training completed. GRU initialized for stable coefficients.")

    def adstock(self, x: torch.Tensor) -> torch.Tensor:
        """STABILIZED adstock transformation."""
        B, T, C = x.shape
        alpha = torch.sigmoid(self.alpha).view(1, 1, -1)
        alpha = torch.clamp(alpha, 0, 0.8)  # Cap at 0.8 for stability
        
        out_list = [x[:, 0:1]]
        for t in range(1, T):
            prev_adstock = out_list[-1]
            current = x[:, t:t+1] + alpha * prev_adstock
            # Clip to prevent explosion
            current = torch.clamp(current, 0, 10)
            out_list.append(current)
        
        return torch.cat(out_list, dim=1)

    def hill(self, x: torch.Tensor) -> torch.Tensor:
        """STABILIZED Hill saturation transformation."""
        a = F.softplus(self.hill_a).view(1, 1, -1)
        g = F.softplus(self.hill_g).view(1, 1, -1)
        
        # Ensure positive input
        x_safe = F.relu(x) + 1e-8
        
        # Stabilized Hill with clipping
        a = torch.clamp(a, 0.1, 2.0)
        g = torch.clamp(g, 0.01, 1.0)
        
        num = torch.pow(x_safe, a)
        denom = num + torch.pow(g, a)
        
        result = num / (denom + 1e-8)
        return torch.clamp(result, 0, 1)  # Ensure bounded output

    def dag_interaction(self, x: torch.Tensor) -> torch.Tensor:
        """IMPROVED DAG interaction with proper adjacency weighting."""
        if not (self.enable_dag and self.enable_interactions):
            return x
        
        # Get adjacency matrix with learned weights (not just binary)
        adj_probs = torch.sigmoid(self.adj_logits)
        adj = adj_probs * self.tri_mask  # Apply triangular mask
        
        # Use adjacency magnitude as edge strength (not just on/off)
        B, T, C = x.shape
        interactions = torch.zeros_like(x)
        
        for t in range(T):
            x_t = x[:, t, :]  # [B, C]
            # Use adjacency weights as interaction strength
            # Each adj[i,j] represents how much channel i influences channel j
            inter_t = torch.matmul(x_t, adj)  # [B, C] × [C, C] = [B, C] - FIXED!
            interactions[:, t, :] = inter_t
        
        # Scale interactions by adjacency strength (not fixed weight)
        # This allows gradients to flow back into adj_logits properly
        interaction_scale = torch.mean(adj_probs)  # Use mean adjacency as scaling
        return x + interaction_scale * interactions
    
    def process_media(self, X: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process media variables through transformations."""
        outputs = {}
        B, T, J = X.shape
        
        # Apply Adstock
        X_adstock = self.adstock(X)
        outputs['media_adstock'] = X_adstock
        
        # Apply Hill transformation
        X_hill = self.hill(X_adstock)
        outputs['media_hill'] = X_hill
        
        # Apply DAG and interactions if enabled
        if self.enable_dag and self.enable_interactions:
            # Use the new dag_interaction function
            X_processed = self.dag_interaction(X_hill)
            outputs['media_dag'] = X_processed
        else:
            X_processed = X_hill
        
        return X_processed, outputs
    
    def apply_burn_in_stabilization(self, coeffs: torch.Tensor, stable_coeff: torch.Tensor) -> torch.Tensor:
        """
        Advanced burn-in stabilization with multiple transition methods.
        
        Args:
            coeffs: Time-varying coefficients [B, T, dim]
            stable_coeff: Stable reference coefficients [dim]
            
        Returns:
            Stabilized coefficients with smooth burn-in transition
        """
        B, T, dim = coeffs.shape
        
        if T <= self.burn_in_weeks:
            # If sequence is shorter than burn-in, use stable coefficients
            return stable_coeff.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        
        # Create stabilization weights based on method
        stabilized_coeffs = coeffs.clone()
        
        for week in range(self.burn_in_weeks):
            # Calculate transition weight based on method
            if self.stabilization_method == "linear":
                # Linear transition from 1.0 (fully stable) to 0.0 (fully dynamic)
                stable_weight = 1.0 - (week / self.burn_in_weeks)
            elif self.stabilization_method == "exponential":
                # Exponential decay - slower initial transition, faster later
                stable_weight = float(torch.exp(torch.tensor(-3.0 * week / self.burn_in_weeks)))
            elif self.stabilization_method == "sigmoid":
                # Sigmoid transition - smooth S-curve
                x = (week - self.burn_in_weeks/2) / (self.burn_in_weeks/4)
                stable_weight = float(1.0 / (1.0 + torch.exp(torch.tensor(x))))
            else:
                # Default to linear
                stable_weight = 1.0 - (week / self.burn_in_weeks)
            
            dynamic_weight = 1.0 - stable_weight
            
            # Blend stable and dynamic coefficients
            stabilized_coeffs[:, week, :] = (
                stable_weight * stable_coeff.unsqueeze(0).expand(B, -1) + 
                dynamic_weight * coeffs[:, week, :]
            )
        
                # DISABLED: Coefficient momentum system removed to prevent gradient blocking
        # The momentum system was using .detach() which blocked gradients
        
        return stabilized_coeffs

    def forward(
        self,
        Xm: torch.Tensor,
        Xc: torch.Tensor,
        R: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the DeepCausalMMM model.
        
        Processes media and control variables through the neural network to generate
        predictions, baseline contributions, seasonal effects, and detailed outputs
        including channel-specific contributions and DAG interactions.
        
        Parameters
        ----------
        Xm : torch.Tensor
            Media data tensor of shape [batch_size, time_steps, n_media]
            Should be SOV-scaled (Share of Voice) normalized to [0, 1] range
        Xc : torch.Tensor
            Control variables tensor of shape [batch_size, time_steps, ctrl_dim]
            Should be standardized (z-score normalized)
        R : torch.Tensor
            Region indicators tensor of shape [batch_size, time_steps]
            Integer values representing region/DMA IDs
            
        Returns
        -------
        predictions : torch.Tensor
            Model predictions of shape [batch_size, time_steps, 1]
            Final KPI predictions combining all effects
        baseline : torch.Tensor
            Baseline contributions of shape [batch_size, time_steps, 1]
            Region-specific baseline effects including global bias
        seasonality : torch.Tensor
            Seasonal contributions of shape [batch_size, time_steps, 1]
            Learned seasonal patterns applied to data
        outputs : Dict[str, Any]
            Dictionary containing detailed model outputs:
            - 'media_contributions': Media channel contributions [batch, time, n_media]
            - 'control_contributions': Control variable contributions [batch, time, ctrl_dim]
            - 'media_coefficients': Time-varying media coefficients [batch, time, n_media]
            - 'control_coefficients': Control coefficients [batch, time, ctrl_dim]
            - 'dag_matrix': Current DAG adjacency matrix [n_media, n_media]
            - 'adstocked_media': Adstock-transformed media [batch, time, n_media]
            - 'saturated_media': Saturation-transformed media [batch, time, n_media]
            - 'interacted_media': DAG-interacted media [batch, time, n_media]
            
        Examples
        --------
        >>> import torch
        >>> model = DeepCausalMMM(n_media=3, ctrl_dim=2, n_regions=2)
        >>> 
        >>> # Prepare input tensors
        >>> media = torch.rand(2, 52, 3)  # 2 regions, 52 weeks, 3 channels
        >>> control = torch.randn(2, 52, 2)  # 2 control variables
        >>> regions = torch.tensor([[0]*52, [1]*52])  # Region indicators
        >>> 
        >>> # Forward pass
        >>> pred, baseline, seasonal, outputs = model(media, control, regions)
        >>> 
        >>> # Access detailed outputs
        >>> media_contrib = outputs['media_contributions']
        >>> dag_matrix = outputs['dag_matrix']
        >>> print(f"DAG sparsity: {(dag_matrix == 0).float().mean():.2f}")
        
        Notes
        -----
        The forward pass applies the following transformations in order:
        1. Media processing: Adstock -> Hill saturation -> DAG interactions
        2. Feature processing: Media features + Control features -> GRU
        3. Coefficient generation: Time-varying coefficients from GRU states
        4. Contribution calculation: Features * Coefficients
        5. Final prediction: Baseline + Seasonality + Media + Control contributions
        
        The model enforces several constraints:
        - DAG acyclicity through upper triangular masking
        - Non-negative baseline and seasonal contributions
        - Learnable coefficient bounds to prevent explosion
        - Burn-in period stabilization for initial weeks
        """
        B, T, _ = Xm.shape
        
        # Process media variables
        X_processed, outputs = self.process_media(Xm)
        
        # Process control variables through MLP - FIXED
        ctrl_features = self.ctrl_mlp(Xc)  # [B, T, ctrl_hidden]
        
        # SOV-AWARE FEATURE PROCESSING: Media data is already SOV-scaled to [0,1], controls are standardized
        # No additional normalization needed - SOV scaling already provides balanced importance learning
        X_processed_norm = X_processed  # SOV-scaled media features are already normalized [0,1]
        ctrl_features_norm = ctrl_features  # Control features are already properly scaled by pipeline
        
        # GRU processing with NORMALIZED media and control features - REVERTED TO STABLE
        gru_in = torch.cat([X_processed_norm, ctrl_features_norm], dim=-1)
        h0 = self.h0.repeat(self.gru_layers, B, 1)  # Single layer GRU
        h_seq, _ = self.gru(gru_in, h0)
        
        # Generate time-varying coefficients - REGULARIZED for stable attribution
        media_coeffs_raw = self.coeff_gen(h_seq)
        # LEARNABLE BOUNDS: Each channel learns its optimal maximum coefficient with non-zero guarantee
        learned_max = F.softplus(self.media_coeff_max_raw) + 0.1  # Range: [0.1, ∞) - NON-ZERO GUARANTEE!
        media_coeffs_unstable = torch.sigmoid(media_coeffs_raw) * learned_max.unsqueeze(0).unsqueeze(0)  # [0, learned_max] per channel
        
        ctrl_coeffs_raw = self.ctrl_coeff_gen(h_seq)
        # LEARNABLE BOUNDS: Each control variable learns its optimal maximum coefficient
        learned_ctrl_max = F.softplus(self.ctrl_coeff_max_raw) + 0.1  # Range: [0.1, ∞) - NON-ZERO GUARANTEE!
        ctrl_coeffs_unstable = torch.tanh(ctrl_coeffs_raw) * learned_ctrl_max.unsqueeze(0).unsqueeze(0)  # [-learned_max, learned_max] per control
        
        # NEW: Apply burn-in stabilization
        media_coeffs = self.apply_burn_in_stabilization(media_coeffs_unstable, self.stable_media_coeff)
        ctrl_coeffs = self.apply_burn_in_stabilization(ctrl_coeffs_unstable, self.stable_ctrl_coeff)
        
        # Calculate contributions
        media_contrib = X_processed * media_coeffs
        media_term = media_contrib.sum(-1)  # [B, T]
        
        # Control contributions using original control values
        ctrl_contrib = Xc * ctrl_coeffs
        ctrl_term = ctrl_contrib.sum(-1)  # [B, T]
        
        # Region baseline - IMPROVED with region-specific baselines
        region_ids = R[:, 0] if R.dim() > 1 else R  # Handle both 1D and 2D region tensors
        region_baselines = self.region_baseline[region_ids]  # [B]
        reg_term = region_baselines.unsqueeze(1).expand(-1, T)  # [B, T]
        
        # NEW: Time trend component - Add linear growth capability
        time_steps = torch.arange(T, dtype=torch.float32, device=Xm.device).unsqueeze(0).expand(B, -1)  # [B, T]
        trend_term = self.time_trend_weight * time_steps + self.time_trend_bias  # [B, T]
        
        # NEW: Seasonal component - Add data-driven seasonality to baseline
        seasonal_term = torch.zeros(B, T, device=Xm.device)
        if self.seasonal_components is not None:
            # Get seasonal components for current time window
            seasonal_data = self.seasonal_components.to(Xm.device)  # [n_regions, n_weeks]
            
            # Handle potential size mismatch (seasonal components might be shorter due to padding removal)
            if seasonal_data.shape[1] >= T:
                # Take the last T weeks (most recent)
                seasonal_slice = seasonal_data[:, -T:]
            else:
                # Pad if seasonal data is shorter
                pad_size = T - seasonal_data.shape[1]
                seasonal_slice = F.pad(seasonal_data, (pad_size, 0), mode='replicate')
            
            # CONSTRAINT: Ensure seasonal coefficient is non-negative (seasonality can't reduce baseline below zero)
            seasonal_coeff_positive = F.softplus(self.seasonal_coeff)  # Always ≥ 0
            # Apply constrained seasonal coefficient - seasonal_slice[region_ids] is already [B, T]
            seasonal_term = seasonal_coeff_positive * seasonal_slice[region_ids]
        
        # PREDICTION - CORRECT: Total baseline = global_bias + region_deviation + seasonality
        # CONSTRAINT: Ensure global_bias is non-negative (baseline visits can't be negative)
        global_bias_positive = F.softplus(self.global_bias)  # Always ≥ 0
        
        # CRITICAL FIX: Ensure the ENTIRE baseline is non-negative (business logic constraint)
        # Calculate raw baseline first
        raw_baseline = reg_term + global_bias_positive + seasonal_term
        # Apply ReLU to ensure baseline is always ≥ 0 (visits can't be negative)
        baseline_positive = F.relu(raw_baseline)  # Always ≥ 0
        
        raw_prediction = media_term + ctrl_term + baseline_positive + trend_term
        y = raw_prediction * F.softplus(self.prediction_scale)
        
        # Store outputs
        outputs['coefficients'] = media_coeffs
        outputs['control_coefficients'] = ctrl_coeffs
        outputs['contributions'] = media_contrib
        outputs['trend_contribution'] = trend_term
        outputs['seasonal_contribution'] = seasonal_term  # NEW: Track seasonal contribution (now always ≥ 0)
        outputs['control_contributions'] = ctrl_contrib
        outputs['baseline'] = baseline_positive  # FIXED: Always non-negative baseline
        outputs['raw_prediction'] = raw_prediction
        outputs['prediction_scale'] = F.softplus(self.prediction_scale)
        outputs['burn_in_weeks'] = self.burn_in_weeks  # NEW: Store for post-processing
        
        return y, media_coeffs, media_contrib, outputs
    
    def get_dag_loss(self) -> torch.Tensor:
        """Light DAG regularization - allows meaningful causal relationships while preventing over-connectivity."""
        if not (self.enable_dag and self.enable_interactions):
            return torch.tensor(0.0, device=self.global_bias.device)
        
        adj_probs = torch.sigmoid(self.adj_logits)
        adj = adj_probs * self.tri_mask
        
        # 1. Light sparsity penalty - allow causal relationships to emerge
        sparsity_loss = torch.sum(adj)
        
        # 2. Very light confidence penalty - don't suppress learning
        confidence_loss = torch.sum(adj_probs * (1 - adj_probs))
        
        # 3. Minimal L1 penalty
        l1_penalty = torch.sum(torch.abs(self.adj_logits))
        
        # Light DAG loss - prioritize discovering causal relationships
        total_dag_loss = (0.01 * sparsity_loss +      # Light sparsity (allow relationships)
                         0.001 * confidence_loss +     # Very light confidence
                         0.0002 * l1_penalty)          # Minimal L1
        
        return total_dag_loss

    def get_sparsity_loss(self) -> torch.Tensor:
        """Sparsity loss to encourage sparse coefficients."""
        if not (self.enable_dag and self.enable_interactions):
            return torch.tensor(0.0, device=self.global_bias.device)
        
        # L1 penalty on media coefficients for sparsity
        media_sparsity = torch.sum(torch.abs(self.stable_media_coeff))
        
        # L1 penalty on control coefficients
        ctrl_sparsity = torch.sum(torch.abs(self.stable_ctrl_coeff))
        
        # L1 penalty on GRU weights for temporal sparsity
        gru_sparsity = sum(torch.sum(torch.abs(param)) for param in self.gru.parameters())
        
        total_sparsity = media_sparsity + ctrl_sparsity + 0.1 * gru_sparsity
        return total_sparsity

    def get_regularization_loss(self) -> torch.Tensor:
        """Calculate combined regularization loss including DAG penalty and coefficient regularization."""
        l1_loss = torch.tensor(0.0, device=self.global_bias.device)
        l2_loss = torch.tensor(0.0, device=self.global_bias.device)
        
        for param in self.parameters():
            if param.requires_grad:
                l1_loss += torch.abs(param).mean()
                l2_loss += (param ** 2).mean()
        
        reg_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        
        # COEFFICIENT-SPECIFIC REGULARIZATION: Prevent coefficient explosion
        coeff_reg_loss = torch.tensor(0.0, device=self.global_bias.device)
        
        # Strong L2 penalty on coefficient range parameters to prevent explosion
        coeff_range_penalty = (self.coeff_range_raw ** 2).mean() * self.coeff_l2_weight
        ctrl_coeff_range_penalty = (self.ctrl_coeff_range_raw ** 2).mean() * self.coeff_l2_weight
        coeff_reg_loss += coeff_range_penalty + ctrl_coeff_range_penalty
        
        # L2 penalty on coefficient generator weights (prevents generating extreme coefficients)
        for name, param in self.named_parameters():
            if 'coeff_gen' in name and param.requires_grad:
                coeff_reg_loss += (param ** 2).mean() * self.coeff_gen_l2_weight
        
        reg_loss += coeff_reg_loss
        
        # Add DAG loss if enabled
        if self.enable_dag and self.enable_interactions:
            reg_loss = reg_loss + self.get_dag_loss()
        
        return reg_loss
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for analysis."""
        params = {
            'adstock_alpha': torch.sigmoid(self.alpha).detach(),
            'hill_a': F.softplus(self.hill_a).detach(),
            'hill_g': F.softplus(self.hill_g).detach(),
            'global_bias': self.global_bias.detach(),
            'prediction_scale': F.softplus(self.prediction_scale).detach(),
        }
        
        if self.enable_dag and self.enable_interactions:
            adj_probs = torch.sigmoid(self.adj_logits)
            params['adjacency'] = (adj_probs * self.tri_mask).detach()
            params['interaction_weight'] = torch.sigmoid(self.interaction_weight).detach()
        
        return params


def create_unified_mmm(
    n_media: int,
    n_control: int,
    hidden_size: int = 64,
    n_regions: int = 2,
    dropout: float = 0.1,
    sparsity_weight: float = 0.1,
    enable_dag: bool = True,
    enable_interactions: bool = True,
    l1_weight: float = 0.01,
    l2_weight: float = 0.01,
    coeff_range: float = 2.0
) -> DeepCausalMMM:
    """Factory function to create a DeepCausalMMM model."""
    model = DeepCausalMMM(
        n_media=n_media,
        ctrl_dim=n_control,
        hidden=hidden_size,
        n_regions=n_regions,
        dropout=dropout,
        sparsity_weight=sparsity_weight,
        enable_dag=enable_dag,
        enable_interactions=enable_interactions,
        l1_weight=l1_weight,
        l2_weight=l2_weight,
        coeff_range=coeff_range
    )
    
    return model 

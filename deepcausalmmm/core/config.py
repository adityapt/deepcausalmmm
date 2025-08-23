"""Configuration settings for DeepCausalMMM model."""

from typing import Dict, Any

def get_default_config() -> Dict[str, Any]:
    """Get default configuration settings for the model.
    
    Returns:
        Dict containing all configuration parameters
    """
    return {
        # Random seed for reproducibility
        'random_seed': 42,
        
        # Model architecture parameters - STABLE PROVEN CONFIGURATION
        'hidden_dim': 320,  # PROVEN optimal balance
        'dropout': 0.08,    # PROVEN stable dropout
        'gru_layers': 1,     # REVERT to single layer for stability
        'ctrl_hidden_ratio': 0.5,  # Control hidden size as ratio of main hidden (NO HARDCODING!)
        'use_layer_norm': True,  # Add layer normalization for training stability
        'enable_dag': True,
        'enable_interactions': True,
        'use_residual_connections': False,  # DISABLE - was causing instability
        
        # Training parameters - REVERT TO STABLE SETTINGS
        'n_epochs': 6500,  # REVERT to proven stable epochs
        'learning_rate': 0.01,  # REVERT to proven stable LR
        'temporal_regularization': 0.04,  # REVERT to proven stable regularization
        'gru_sparsity_weight': 0.1,    # Weight for GRU parameter sparsity in total sparsity calculation
        'batch_size': None,  # None means use full batch
        'burn_in_weeks': 6,  # Standard burn-in
        
        # REVERT TO STABLE Learning Rate Scheduling
        'use_cosine_annealing': False,    # DISABLE - was causing instability
        'cosine_t_initial': 500,          # Original stable settings
        'cosine_t_mult': 1.2,             # Original stable settings
        'cosine_eta_min': 1e-6,           # Original stable settings
        'warmup_epochs': 0,               # DISABLE warmup for simplicity
        
        # REVERT TO WORKING Loss Function (Huber was the working version from today!)
        'use_huber_loss': True,      # Keep Huber Loss - was working well today
        'huber_delta': 0.25,         # PROVEN stable delta from working version
        'use_focal_loss': False,     # Keep focal loss disabled     # DISABLE - was causing instability
        'focal_alpha': 0.25,         # Focal loss alpha parameter
        'focal_gamma': 1.5,          # Focal loss gamma parameter
        'focal_loss_weight': 0.1,    # NEW: Configurable focal loss contribution weight
        
        # HYBRID APPROACH: Fixed regularization weights for stable training
        # These control loss balancing and should be stable, not learned
        'dag_weight': 0.008,          # Minimal DAG regularization for stability
        'sparsity_weight': 0.001,     # Minimal sparsity regularization for stability  
        'l1_weight': 1e-5,            # Ultra-light L1 regularization
        'l2_weight': 5e-5,            # Ultra-light L2 regularization
        
        # REVERTED COEFFICIENT REGULARIZATION to proven stable values
        'coeff_l2_weight': 0.03,      # REVERT to proven stable L2 penalty
        'coeff_gen_l2_weight': 0.015, # REVERT to proven stable L2 penalty
        
        # ADVANCED REGULARIZATION: Gradient clipping and weight decay scheduling
        'gradient_clip_norm': 2.0,    # Gradient clipping for stability
        'weight_decay_schedule': True, # Schedule weight decay during training
        'ema_decay': 0.999,           # Exponential Moving Average for model parameters
        'coeff_grad_clip': 1,       # GENTLER gradient clipping for coefficient parameters
        
        # NOTE: Core model parameters ARE learnable (coeff_range, trend_damping, etc.)
        # Only loss balancing weights are fixed for training stability
        
        # NOTE: ALL initialization scaling factors are now FULLY LEARNABLE parameters
        # The model will automatically discover optimal initialization scaling
        # No hardcoded multipliers - everything learned from data!
        # 'trend_damping_factor': LEARNABLE,     # Model learns optimal trend damping
        # 'stable_coeff_scale': LEARNABLE,       # Model learns optimal stable coefficient scaling
        # 'region_baseline_scale': LEARNABLE,    # Model learns optimal region baseline scaling
        # 'interaction_weight_init': LEARNABLE,  # Model learns optimal interaction weight scaling
        
            # Warm-start parameters - FAST CONVERGENCE
        'warm_start_epochs': 50,   # REDUCED for faster convergence with DMA scaling
        'momentum_decay': 0.975,
        
        # REVERT Early stopping to proven stable values
        'early_stopping': True,   # ENABLE for efficient training
        'patience': 1500,  # REVERT to proven stable patience
        'min_delta': 5e-6,  # REVERT to proven stable threshold
        'restore_best_weights': False,  # REVERT - keep simple
        
        # DAG learning parameters
        'min_temperature': 0.4,
        'max_grad_norm': 2.0,  # INCREASED for more aggressive learning
        
        # Optimizer settings
        'optimizer': {
            'type': 'adamw',
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 1e-5  # OPTIMAL weight decay for precision regularization
        },
        
        # Learning rate scheduler - ULTRA-AGGRESSIVE OPTIMIZATION
        'scheduler': {
            'type': 'cosine_annealing',  # Better convergence than plateau
            'T_max': 4000,              # Half of max epochs for cosine cycle
            'eta_min': 1e-6,            # Minimum LR
            'warmup_epochs': 100,       # LR warmup for stability
            'warmup_factor': 0.1        # Start at 10% of base LR
        },
        
        # Time series splitting parameters - REDUCE TEMPORAL GAP
        'holdout_ratio': 0.08,  # Reduced holdout to minimize temporal distribution shift
        'use_holdout': True,  # Whether to use holdout evaluation
        'min_train_weeks': 40,  # Reduced minimum weeks for training
        
        # Data processing constants
        'scaling_constants': {
            'iqr_to_std_factor': 1.349,  # IQR to std conversion factor
            'zero_threshold': 1e-8,      # Threshold for zero values
            'outlier_percentile': 0.97,  # BALANCED outlier smoothing - not too aggressive, not too loose
            'extreme_clip_threshold': 2.0,  # Threshold for extreme distribution shift
            'standard_clip_range': 5.0,     # BALANCED clipping range
            'aggressive_clip_range': 3.5,   # MODERATE clipping range
        },
        
        # Visualization constants
        'visualization': {
            'correlation_threshold': 0.65,   # Threshold for showing only strong correlations (top ~20%)
            'edge_width_multiplier': 8,      # Multiplier for edge width in DAG
            'node_opacity': 0.7,             # Node opacity in plots
            'line_opacity': 0.6,             # Line opacity in plots
            'fill_opacity': 0.1,             # Fill area opacity
            'marker_size': 8,                # Default marker size
            'subplot_vertical_spacing': 0.08,  # Vertical spacing between subplots
            'subplot_horizontal_spacing': 0.06,  # Horizontal spacing between subplots
        },
        
        # Training display constants
        'training_display': {
            'r2_display_min': -10.0,         # Minimum R² value for display
            'loss_approximation_factors': {   # For approximate loss decomposition
                'training_component': 0.7,
                'validation_component': 0.3,
            },
        },
        
        # Synthetic data generation parameters
        'synthetic_data': {
            'base_spend_range': (10000, 50000),    # Range for base media spend
            'seasonality_strength': 0.3,           # Strength of seasonal pattern
            'media_noise_level': 0.2,              # Noise level in media data
            'control_range': (-2, 2),              # Range for control variables
            'control_correlation': 0.7,            # Temporal correlation in controls
            'media_coeff_range': (0.1, 0.8),       # Range for media coefficients
            'control_coeff_range': (-0.5, 0.5),    # Range for control coefficients
            'base_level_range': (40000, 60000),    # Range for baseline levels
            'adstock_rate': 0.5,                   # Adstock transformation rate
            'saturation_param': 0.5,               # Saturation curve parameter
            'target_noise_level': 0.05,            # Noise level in target variable
        },
        
        # Output directory configuration
        'output_paths': {
            'dashboard_dir': 'dashboard_beautiful_comprehensive',  # Main dashboard directory
            'results_dir': 'results',                             # Results directory
            'plots_dir': 'plots',                                # Individual plots directory
            'data_dir': 'data',                                  # Data output directory
        },

    }

def update_config(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update base configuration with new values.
    
    Args:
        base_config: Base configuration dictionary
        updates: Dictionary containing updates to apply
        
    Returns:
        Updated configuration dictionary
    """
    config = base_config.copy()
    for key, value in updates.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key].update(value)
        else:
            config[key] = value
    return config 
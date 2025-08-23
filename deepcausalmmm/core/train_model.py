"""
Training functions for DeepCausalMMM models.
Updated to use UnifiedDataPipeline for consistent data processing.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error

from deepcausalmmm.core.unified_model import DeepCausalMMM, create_unified_mmm
from deepcausalmmm.core.config import get_default_config, update_config
from deepcausalmmm.core.scaling import SimpleGlobalScaler, GlobalScaler
from deepcausalmmm.core.data import UnifiedDataPipeline
from deepcausalmmm.core.trainer import ModelTrainer
# Device utilities available but not needed for this implementation


def calculate_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate R-squared score."""
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    return r2_score(y_true_np.flatten(), y_pred_np.flatten())


# DEPRECATED: Use ModelTrainer class instead
def train_model_with_config(
    model: DeepCausalMMM,
    X_media_padded: torch.Tensor,
    X_control_padded: torch.Tensor,
    R: Optional[torch.Tensor],
    y_padded: torch.Tensor,
    config: Dict[str, Any],
    verbose: bool = True,
    holdout_data: Optional[Dict[str, torch.Tensor]] = None,
    pipeline: Optional[Any] = None
) -> Tuple[List[float], List[float], List[float], float]:
    """
    Train model with config-driven parameters and warm-start.
    This matches the proven approach from dashboard_rmse_optimized.py
    
    Args:
        model: DeepCausalMMM model instance
        X_media_padded: Padded media data [n_regions, n_timesteps, n_channels]
        X_control_padded: Padded control data [n_regions, n_timesteps, n_controls]
        R: Region tensor (can be None)
        y_padded: Padded target data [n_regions, n_timesteps]
        config: Configuration dictionary
        verbose: Whether to print progress
        
    Returns:
        Tuple of (train_losses, train_rmses, train_r2s, best_rmse)
    """
    if verbose:
        print(f"\n🚀 Training Model with Config Parameters...")
        print(f"   🎯 Training Configuration from Config:")
        print(f"      📈 Epochs: {config['n_epochs']}")
        print(f"      🧠 Hidden units: {config['hidden_dim']}")
        print(f"      🔥 Warm-start: {config['warm_start_epochs']}")
        print(f"      📊 Learning rate: {config['learning_rate']}")
        print(f"      🎛️ Optimizer: {config['optimizer']}")
        print(f"      📅 Scheduler: {config['scheduler']}")
    
    # 1. Warm-start training for coefficient stabilization
    if verbose:
        print(f"   🔥 Config-driven warm-start training for {config['warm_start_epochs']} epochs...")
    
    # Create optimizer for warm-start (reduced learning rate)
    warm_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'] * 0.01,  # Reduced LR for warm-start
        weight_decay=config.get('optimizer', {}).get('weight_decay', 1e-7)
    )
    
    # Create region tensor if needed
    if R is None:
        R = torch.zeros(X_media_padded.shape[0], dtype=torch.long)
    
    # CRITICAL FIX: Initialize baseline with training data (was missing!)
    if hasattr(model, 'initialize_baseline'):
        if verbose:
            print(f"   📊 Initializing model baseline from training data...")
        model.initialize_baseline(y_padded)
    if hasattr(model, 'initialize_stable_coefficients_from_data'):
        if verbose:
            print(f"   🎯 Initializing stable coefficients from training data...")
        model.initialize_stable_coefficients_from_data(X_media_padded, X_control_padded, y_padded)
    
    model.warm_start_training(
        X_media_padded, X_control_padded, R, y_padded, 
        warm_optimizer, config.get('warm_start_epochs', 50)
    )
    
    if verbose:
        print(f"✅ Warm-start training completed. GRU initialized for stable coefficients.")
    
    # 2. Main training with full configuration
    if verbose:
        print(f"   🚀 Main training for {config['n_epochs']} epochs...")
    
    # Create main optimizer based on config - EXACT SAME as working dashboard
    if config.get('optimizer', 'adamw') == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create scheduler based on config - EXACT SAME as working dashboard
    if config.get('scheduler', 'reduce_on_plateau') == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=config.get('patience', 800), 
            factor=0.5, min_lr=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['n_epochs'])
    
    # Training tracking
    train_losses = []
    train_rmses = []
    train_r2s = []
    best_rmse = float('inf')
    best_holdout_loss = float('inf')  # Track best holdout loss for early stopping
    patience_counter = 0
    
    # Store last holdout metrics for progress bar
    last_holdout_rmse = None
    last_holdout_r2 = None
    last_holdout_loss = None
    
    # Training loop with progress bar
    pbar = tqdm(range(config['n_epochs']), desc="Config-Driven Training")
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        y_pred_full, media_contrib_full, control_contrib_full, outputs = model(
            X_media_padded, X_control_padded, R
        )
        
        # Calculate loss (MSE in log space)
        mse_loss = torch.nn.functional.mse_loss(y_pred_full, y_padded)
        
        # Add DAG and sparsity losses (CRITICAL FIX)
        dag_loss = model.get_dag_loss() if hasattr(model, 'get_dag_loss') else 0
        sparsity_loss = model.get_sparsity_loss() if hasattr(model, 'get_sparsity_loss') else 0
        
        # Add L1 and L2 regularization
        l1_reg = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        l2_reg = sum(torch.sum(param ** 2) for param in model.parameters())
        
        total_loss = (mse_loss + 
                     config.get('dag_weight', 0.1) * dag_loss + 
                     config.get('sparsity_weight', 0.1) * sparsity_loss +
                     config['l1_weight'] * l1_reg + 
                     config['l2_weight'] * l2_reg)
        
        # Backward pass with gradient clipping
        total_loss.backward()
        if config.get('max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Calculate RMSE in log space (consistent with loss)
            train_rmse = torch.sqrt(mse_loss).item()
            
            # Calculate R² (also in log space for consistency)
            y_eval = y_padded[:, config['burn_in_weeks']:].contiguous()
            pred_eval = y_pred_full[:, config['burn_in_weeks']:].contiguous()
            train_r2 = r2_score(y_eval.numpy().flatten(), pred_eval.numpy().flatten())
            
                    # Evaluate on holdout if available (every 10 epochs for better monitoring)
        holdout_rmse = None
        holdout_r2 = None
        holdout_loss = None
        if (holdout_data is not None and pipeline is not None and
            epoch >= 10):  # Every epoch, after epoch 10
                try:
                    # Debug info on first holdout evaluation
                    if epoch == 10:
                        print(f"\n   🔍 DEBUG: Holdout X_media shape: {holdout_data['X_media'].shape}")
                        print(f"   🔍 DEBUG: Holdout X_control shape: {holdout_data['X_control'].shape}")
                        print(f"   🔍 DEBUG: Burn-in weeks: {config['burn_in_weeks']}")
                        print(f"   🔍 DEBUG: Holdout weeks after burn-in: {holdout_data['X_media'].shape[1] - config['burn_in_weeks']}")
                    
                    # Evaluate on holdout data
                    holdout_pred_full, _, _, _ = model(
                        holdout_data['X_media'], holdout_data['X_control'], holdout_data['R']
                    )
                    
                    # Check if we have holdout data after removing padding (burn-in weeks)
                    total_holdout_weeks = holdout_pred_full.shape[1]
                    actual_holdout_weeks = total_holdout_weeks - config['burn_in_weeks']
                    if actual_holdout_weeks > 0:
                        # Calculate holdout metrics in original scale
                        # CRITICAL FIX: Data already has padding removed, so don't remove it again!
                        holdout_pred_orig = pipeline.inverse_transform_predictions(
                            holdout_pred_full[:, config['burn_in_weeks']:].detach(), remove_padding=False
                        )
                        holdout_true_orig = pipeline.inverse_transform_predictions(
                            holdout_data['y'][:, config['burn_in_weeks']:], remove_padding=False
                        )
                        
                        if holdout_pred_orig.numel() > 0 and holdout_true_orig.numel() > 0:
                            holdout_rmse = np.sqrt(mean_squared_error(
                                holdout_true_orig.numpy().flatten(),
                                holdout_pred_orig.numpy().flatten()
                            ))
                            holdout_r2 = r2_score(
                                holdout_true_orig.numpy().flatten(),
                                holdout_pred_orig.numpy().flatten()
                            )
                            
                            # Cap extremely large RMSE values for display
                            if holdout_rmse > 1e8:  # Cap at 100 million
                                holdout_rmse = 1e8
                            
                            # Calculate holdout loss in log space (consistent with training loss)
                            holdout_pred_log = holdout_pred_full[:, config['burn_in_weeks']:].detach()
                            holdout_true_log = holdout_data['y'][:, config['burn_in_weeks']:]
                            holdout_loss = torch.nn.functional.mse_loss(holdout_pred_log, holdout_true_log).item()
                            
                            # Store holdout metrics for progress bar
                            last_holdout_rmse = holdout_rmse
                            last_holdout_r2 = holdout_r2
                            last_holdout_loss = holdout_loss
                            
                            # Debug success on first holdout evaluation
                            if epoch == 10:
                                print(f"   ✅ DEBUG: Holdout evaluation successful - RMSE: {holdout_rmse:,.0f}, R²: {holdout_r2:.3f}")
                        else:
                            if epoch == 10:
                                print(f"   ⚠️ DEBUG: Empty holdout tensors after processing")
                    else:
                        if epoch == 10:
                            print(f"   ⚠️ DEBUG: No actual holdout weeks after removing padding ({actual_holdout_weeks} weeks)")
                            
                except Exception as e:
                    # Print debug info on errors
                    if epoch == 10:
                        print(f"\n   ⚠️ Holdout evaluation error (epoch {epoch}): {e}")
                    pass
        
        # Track metrics
        train_losses.append(total_loss.item())
        train_rmses.append(train_rmse)
        train_r2s.append(train_r2)
        
        # Update best RMSE (use training RMSE for consistency)
        if train_rmse < best_rmse:
            best_rmse = train_rmse
        
        # Balanced early stopping: consider both train RMSE and holdout loss
        improvement = False
        
        # Check training RMSE improvement
        if train_rmse < best_rmse:
            improvement = True
        
        # Check holdout loss improvement (if available)
        if last_holdout_loss is not None:
            if last_holdout_loss < best_holdout_loss:
                best_holdout_loss = last_holdout_loss
                improvement = True
        
        # Update patience counter based on ANY improvement
        if improvement:
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update progress bar with both train and holdout metrics (shortened names)
        progress_dict = {
            'TrL': f'{total_loss.item():.1f}',
            'TrR': f'{train_rmse:.4f}',
            'TrR²': f'{train_r2:.3f}',
            'Best': f'{best_rmse:.4f}'
        }
        
        # Use stored holdout metrics (from last evaluation) for progress bar
        if last_holdout_rmse is not None and last_holdout_loss is not None:
            # Cap R² display for readability (very negative values are not useful)
            r2_min = config.get('training_display', {}).get('r2_display_min', -10.0)
            r2_display = last_holdout_r2 if last_holdout_r2 > r2_min else r2_min
            progress_dict.update({
                'HoL': f'{last_holdout_loss:.1f}',
                'HoR': f'{last_holdout_rmse/1e6:.1f}M' if last_holdout_rmse > 0 else '0.0M',  # Show in millions for space
                'HoR²': f'{r2_display:.3f}'
            })
        
        pbar.set_postfix(progress_dict)
        
        # Scheduler step - EXACT SAME as working dashboard
        if config.get('scheduler', 'reduce_on_plateau') == 'reduce_on_plateau':
            scheduler.step(train_rmse)
        else:
            scheduler.step()
        
        # Early stopping
        if (config.get('early_stopping', False) and 
            patience_counter >= config.get('patience', 500)):
            if verbose:
                print(f"   ⏹️ Early stopping at epoch {epoch}")
                print(f"   🏆 Best RMSE: {best_rmse:.2f}")
            break
    
    pbar.close()
    
    if verbose:
        print(f"   ✅ Config-driven training completed!")
        print(f"   🏆 Final Best RMSE: {best_rmse:.2f}")
    
    return train_losses, train_rmses, train_r2s, best_rmse


# train_mmm_with_trainer function removed - users should use ModelTrainer class directly


def train_mmm(
    X_media: np.ndarray,
    X_control: np.ndarray,
    y: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    channel_names: Optional[List[str]] = None,
    control_names: Optional[List[str]] = None,
    verbose: bool = True,
    use_unified_pipeline: bool = False,
    train_ratio: Optional[float] = None
) -> Tuple[DeepCausalMMM, Dict[str, Any]]:
    """
    Train a DeepCausalMMM model with optional UnifiedDataPipeline.
    
    .. deprecated:: 1.0.0
        This function-based approach is deprecated. Please use the modern 
        class-based approach with ModelTrainer instead:
        
        ```python
        from deepcausalmmm.core.trainer import ModelTrainer
        trainer = ModelTrainer(config)
        model = trainer.create_model(n_media, n_control, n_regions)
        trainer.create_optimizer_and_scheduler()
        results = trainer.train(...)
        ```
    
    Args:
        X_media: Media variables [n_regions, n_timesteps, n_channels]
        X_control: Control variables [n_regions, n_timesteps, n_controls]
        y: Target variable [n_regions, n_timesteps]
        config: Configuration dictionary (uses default if None)
        channel_names: List of channel names
        control_names: List of control variable names
        verbose: Whether to print progress
        use_unified_pipeline: Whether to use UnifiedDataPipeline for train/holdout split
        train_ratio: Deprecated - use config['holdout_weeks'] instead
        
    Returns:
        Tuple of (trained_model, results_dict)
    """
    import warnings
    warnings.warn(
        "train_mmm() is deprecated and will be removed in v2.0.0. "
        "Please use the modern ModelTrainer class instead. "
        "See documentation for migration guide.",
        DeprecationWarning,
        stacklevel=2
    )
    if verbose:
        print("🎯 DEEPCAUSALMMM TRAINING")
        print("=" * 50)
        if use_unified_pipeline:
            print("🔧 Config-driven • 🎨 UnifiedDataPipeline • 🎯 RMSE Optimized")
        else:
            print("🔧 Config-driven • 🎨 SimpleGlobalScaler • 🎯 RMSE Optimized")
    
    # 1. Configuration setup
    if config is None:
        config = get_default_config()
        if verbose:
            print("📋 Using default configuration")
    else:
        if verbose:
            print("📋 Using provided configuration")
    
    if use_unified_pipeline:
        # Use UnifiedDataPipeline for consistent train/holdout processing
        return _train_with_unified_pipeline(
            X_media, X_control, y, config, channel_names, control_names, 
            verbose
        )
    else:
        # Use simple approach without holdout splitting
        return _train_simple(
            X_media, X_control, y, config, channel_names, control_names, verbose
        )


def _train_with_unified_pipeline(
    X_media: np.ndarray,
    X_control: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any],
    channel_names: Optional[List[str]],
    control_names: Optional[List[str]],
    verbose: bool
) -> Tuple[DeepCausalMMM, Dict[str, Any]]:
    """Train using UnifiedDataPipeline with train/holdout split."""
    
    # Get holdout ratio from config
    holdout_ratio = config.get('holdout_ratio', 0.27)
    
    if verbose:
        print(f"\n🎯 UNIFIED DATA PIPELINE TRAINING")
        print(f"🔧 Consistent train/holdout processing • Holdout ratio: {holdout_ratio:.1%}")
    
    # 1. Initialize unified data pipeline
    pipeline = UnifiedDataPipeline(config)
    
    # 2. Temporal split (using ratio-based time series approach)
    train_data, holdout_data = pipeline.temporal_split(
        X_media, X_control, y, holdout_ratio=holdout_ratio
    )
    
    # 3. Process training data (fit scaler + transform + pad)
    train_tensors = pipeline.fit_and_transform_training(train_data)
    
    # 4. Process holdout data (transform + pad using SAME scaler)
    holdout_tensors = pipeline.transform_holdout(holdout_data)
    
    # 5. Create model
    n_media = X_media.shape[2]
    n_control = X_control.shape[2]  # This is already 7 (original control vars)
    n_regions = X_media.shape[0]
    
    # Get actual dimensions from processed tensors (after seasonality addition)
    actual_n_media = train_tensors['X_media'].shape[2]
    actual_n_control = train_tensors['X_control'].shape[2]  # This will be 14 (7 original + 7 seasonality)
    
    model = DeepCausalMMM(
        n_media=actual_n_media,
        ctrl_dim=actual_n_control,
        n_regions=n_regions,
        hidden=config.get('hidden_dim', 64),
        dropout=config.get('dropout', 0.1),
        l1_weight=config.get('l1_weight', 0.001),
        l2_weight=config.get('l2_weight', 0.001),
        coeff_range=config.get('coeff_range', 1.0),
        burn_in_weeks=config.get('burn_in_weeks', 4),
        momentum_decay=config.get('momentum_decay', 0.9),
        warm_start_epochs=config.get('warm_start_epochs', 50),
        enable_dag=config.get('enable_dag', True),
        enable_interactions=config.get('enable_interactions', True)
    )
    
    # 6. Train model with holdout evaluation
    train_losses, train_rmses, train_r2s, best_rmse = train_model_with_config(
        model, train_tensors['X_media'], train_tensors['X_control'], 
        train_tensors['R'], train_tensors['y'], config, verbose,
        holdout_data=holdout_tensors, pipeline=pipeline
    )
    
    # 7. Final evaluation on both train and holdout
    if verbose:
        print("\n📊 Final Evaluation (Train + Holdout)...")
    
    model.eval()
    with torch.no_grad():
        # Training evaluation
        train_pred_full, train_media_contrib, train_control_contrib, _ = model(
            train_tensors['X_media'], train_tensors['X_control'], train_tensors['R']
        )
        
        # Holdout evaluation (if holdout data exists after removing padding)
        total_holdout_weeks = holdout_tensors['X_media'].shape[1]
        actual_holdout_weeks = total_holdout_weeks - config['burn_in_weeks']
        if actual_holdout_weeks > 0:
            holdout_pred_full, holdout_media_contrib, holdout_control_contrib, _ = model(
                holdout_tensors['X_media'], holdout_tensors['X_control'], holdout_tensors['R']
            )
            
            # Convert to original scale using pipeline
            # CRITICAL FIX: Data already has padding removed, so don't remove it again!
            holdout_pred_orig = pipeline.inverse_transform_predictions(
                holdout_pred_full[:, config['burn_in_weeks']:], remove_padding=False
            )
            holdout_true_orig = pipeline.inverse_transform_predictions(
                holdout_tensors['y'][:, config['burn_in_weeks']:], remove_padding=False
            )
            
            # Calculate holdout metrics
            holdout_rmse = np.sqrt(mean_squared_error(
                holdout_true_orig.numpy().flatten(),
                holdout_pred_orig.numpy().flatten()
            ))
            holdout_r2 = r2_score(
                holdout_true_orig.numpy().flatten(),
                holdout_pred_orig.numpy().flatten()
            )
            
            # Calculate holdout loss in log space (consistent with training)
            holdout_pred_log = holdout_pred_full[:, config['burn_in_weeks']:]
            holdout_true_log = holdout_tensors['y'][:, config['burn_in_weeks']:]
            holdout_loss = torch.nn.functional.mse_loss(holdout_pred_log, holdout_true_log).item()
            
            if verbose:
                print(f"   🧪 HOLDOUT RESULTS:")
                print(f"      Loss: {holdout_loss:.1f}")
                print(f"      RMSE: {holdout_rmse:,.0f}")
                print(f"      R²: {holdout_r2:.3f}")
        else:
            holdout_rmse = None
            holdout_r2 = None
            holdout_loss = None
            if verbose:
                print(f"   ⚠️ Holdout data too small for evaluation")
    
    # Training evaluation
    # CONSISTENCY FIX: Use same process as holdout - no double padding removal
    train_pred_orig = pipeline.inverse_transform_predictions(
        train_pred_full[:, config['burn_in_weeks']:], remove_padding=False
    )
    train_true_orig = pipeline.inverse_transform_predictions(
        train_tensors['y'][:, config['burn_in_weeks']:], remove_padding=False
    )
    
    final_train_rmse = np.sqrt(mean_squared_error(
        train_true_orig.numpy().flatten(),
        train_pred_orig.numpy().flatten()
    ))
    final_train_r2 = r2_score(
        train_true_orig.numpy().flatten(),
        train_pred_orig.numpy().flatten()
    )
    
    if verbose:
        print(f"   🚂 TRAINING RESULTS:")
        print(f"      RMSE: {final_train_rmse:,.0f}")
        print(f"      R²: {final_train_r2:.3f}")
        print(f"\n📊 SUMMARY:")
        print(f"   🚂 Train: RMSE {final_train_rmse:,.0f} | R² {final_train_r2:.3f}")
        if holdout_rmse is not None:
            print(f"   🧪 Holdout: RMSE {holdout_rmse:,.0f} | R² {holdout_r2:.3f}")
            generalization_gap = ((holdout_rmse - final_train_rmse) / final_train_rmse) * 100
            print(f"   📈 Generalization Gap: {generalization_gap:+.1f}%")
    
    # 8. Prepare results
        results = {
            'train_losses': train_losses,
        'train_rmses': train_rmses,
        'train_r2s': train_r2s,
        'best_rmse': best_rmse,
        'final_train_rmse': final_train_rmse,
        'final_train_r2': final_train_r2,
        'final_train_loss': train_losses[-1] if train_losses else 0.0,
        'final_holdout_rmse': holdout_rmse,
        'final_holdout_r2': holdout_r2,
        'final_holdout_loss': holdout_loss,
        'holdout_predictions_orig': holdout_pred_orig if holdout_rmse is not None else None,  # Add holdout predictions
        'pipeline': pipeline,
        'config': config,
        'predictions': train_pred_orig.numpy(),
        'media_contributions': pipeline.inverse_transform_contributions(
            train_media_contrib[:, config['burn_in_weeks']:], train_true_orig
        ).numpy(),
        'control_contributions': train_control_contrib[:, config['burn_in_weeks']:].numpy(),
        'channel_names': channel_names,
        'control_names': control_names,
            'model_params': {
            'n_regions': X_media.shape[0],
            'n_weeks': X_media.shape[1],
            'n_media_channels': X_media.shape[2],
            'n_control_channels': X_control.shape[2],
            'padding_weeks': config['burn_in_weeks'],
            'train_weeks': train_data['X_media'].shape[1],
            'holdout_weeks': holdout_data['X_media'].shape[1] if holdout_data['X_media'].shape[1] > 0 else 0
        }
    }
    
    if verbose:
        print(f"\n🎉 UNIFIED PIPELINE TRAINING COMPLETE!")
        print(f"🏆 Train RMSE: {final_train_rmse:,.0f} (R²: {final_train_r2:.3f})")
        if holdout_rmse is not None:
            print(f"🧪 Holdout RMSE: {holdout_rmse:,.0f} (R²: {holdout_r2:.3f})")
        print("=" * 50)
    
    return model, results


def _train_simple(
    X_media: np.ndarray,
    X_control: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any],
    channel_names: Optional[List[str]],
    control_names: Optional[List[str]],
    verbose: bool
) -> Tuple[DeepCausalMMM, Dict[str, Any]]:
    """Train using simple approach without holdout splitting."""
    
    # 2. Data preprocessing with SimpleGlobalScaler
    if verbose:
        print("\n🔧 Data Preprocessing with SimpleGlobalScaler...")
    
    scaler = SimpleGlobalScaler()
    if verbose:
        print("   📊 Created new SimpleGlobalScaler")
    
    # Scale data
    X_media_scaled, X_control_scaled, y_scaled = scaler.fit_transform(
        X_media, X_control, y
    )
    
    if verbose:
        print("   ✅ SimpleGlobalScaler applied successfully")
        print(f"   📊 Data shape: {X_media.shape[0]} regions × {X_media.shape[1]} weeks")
        print(f"   📺 Media channels: {X_media.shape[2]}")
        print(f"   🎛️ Control variables: {X_control.shape[2]}")
    
    # 3. Data padding for burn-in
    padding_weeks = config['burn_in_weeks']
    
    # Create padding tensors
    media_padding = torch.zeros(X_media_scaled.shape[0], padding_weeks, X_media_scaled.shape[2])
    control_padding = torch.zeros(X_control_scaled.shape[0], padding_weeks, X_control_scaled.shape[2])
    y_padding = torch.zeros(y_scaled.shape[0], padding_weeks)
    
    # Add padding
    X_media_padded = torch.cat([media_padding, X_media_scaled], dim=1)
    X_control_padded = torch.cat([control_padding, X_control_scaled], dim=1)
    y_padded = torch.cat([y_padding, y_scaled], dim=1)
    
    if verbose:
        print(f"   ✅ Added {padding_weeks} weeks padding for burn-in")
    
    # 4. Model creation
    if verbose:
        print("\n🧠 Creating Model from Configuration...")
    
    model = DeepCausalMMM(
        n_media=X_media.shape[2],
        ctrl_dim=X_control.shape[2],
        n_regions=X_media.shape[0],
        hidden=config.get('hidden_dim', 64),
        dropout=config.get('dropout', 0.1),
        l1_weight=config.get('l1_weight', 0.001),
        l2_weight=config.get('l2_weight', 0.001),
        coeff_range=config.get('coeff_range', 1.0),
        burn_in_weeks=config.get('burn_in_weeks', 4),
        momentum_decay=config.get('momentum_decay', 0.9),
        warm_start_epochs=config.get('warm_start_epochs', 50),
        enable_dag=config.get('enable_dag', True),
        enable_interactions=config.get('enable_interactions', True)
    )
    
    if verbose:
        print(f"   ✅ Model created with {config.get('hidden_dim', 64)} hidden units")
        print(f"   🔧 Config-driven parameters: dropout={config.get('dropout', 0.1)}, l1={config.get('l1_weight', 0.001)}, l2={config.get('l2_weight', 0.001)}")
    
    # Create region tensor for training and evaluation
    R = torch.zeros(X_media_padded.shape[0], dtype=torch.long)
    
    # 5. Training with config-driven approach
    train_losses, train_rmses, train_r2s, best_rmse = train_model_with_config(
        model, X_media_padded, X_control_padded, R, y_padded, config, verbose
    )
    
    # 6. Final evaluation
    if verbose:
        print("\n📊 Final Evaluation...")
    
    model.eval()
    with torch.no_grad():
        predictions_full, media_contrib_full, control_contrib_full, outputs = model(
            X_media_padded, X_control_padded, R
        )
        
        # Remove padding for evaluation
        predictions = predictions_full[:, padding_weeks:]
        media_contributions = media_contrib_full[:, padding_weeks:]
        control_contributions = control_contrib_full[:, padding_weeks:]
        
        # Convert to original scale using scaler
        predictions_orig = scaler.inverse_transform_target(predictions)
        y_orig = scaler.inverse_transform_target(y_scaled)
        
        # Calculate final metrics
        final_rmse = np.sqrt(mean_squared_error(y_orig.numpy().flatten(), predictions_orig.numpy().flatten()))
        final_r2 = r2_score(y_orig.numpy().flatten(), predictions_orig.numpy().flatten())
        relative_rmse = final_rmse / y_orig.mean().item() * 100
        
        if verbose:
            print(f"   🏆 FINAL RESULTS:")
            print(f"      RMSE: {final_rmse:,.0f}")
            print(f"      Relative RMSE: {relative_rmse:.1f}%")
            print(f"      R²: {final_r2:.3f}")
            print(f"      Training Best RMSE: {best_rmse:.4f} (log space)")
    
    # 7. Prepare results
    results = {
        'train_losses': train_losses,
        'train_rmses': train_rmses,
        'train_r2s': train_r2s,
        'best_rmse': best_rmse,
        'final_rmse': final_rmse,
        'final_r2': final_r2,
        'relative_rmse': relative_rmse,
        'scaler': scaler,
        'config': config,
        'predictions': predictions_orig.numpy(),
        'media_contributions': media_contributions.numpy(),
        'control_contributions': control_contributions.numpy(),
        'channel_names': channel_names,
        'control_names': control_names,
        'model_params': {
            'n_regions': X_media.shape[0],
            'n_weeks': X_media.shape[1],
            'n_media_channels': X_media.shape[2],
            'n_control_channels': X_control.shape[2],
            'padding_weeks': padding_weeks
        }
    }
    
    if verbose:
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"🏆 Final RMSE: {final_rmse:,.0f} ({relative_rmse:.1f}%)")
        print(f"📈 R²: {final_r2:.3f}")
        print("=" * 50)
        
        return model, results 


# Legacy function for backward compatibility
def train_unified_mmm(*args, **kwargs):
    """Legacy wrapper for train_mmm with unified pipeline. Use train_mmm instead."""
    kwargs['use_unified_pipeline'] = True
    return train_mmm(*args, **kwargs)